// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 model ownership and teardown boundary.
//!
//! A published bundle owns every model resource: the bounded SSD PLE reader,
//! mutable GPU state, finalized resident weights, and the attached canonical
//! load transaction/census (including external PLE descriptors).  No loader
//! local may outlive publication as a second owner.

use crate::artifact::{Qwen4PleRows, SidecarRows};
use crate::config::{Qwen4Config, Qwen4PleConfig};
use crate::ep::{Qwen4EpMesh, ReplicaFingerprint};
use crate::gpu_forward::{
    qwen4_forward_device_bytes, qwen4_prefill_chunk_requested, qwen4_prefill_chunk_rungs,
    qwen4_spec_logit_rows, Qwen4ForwardScratchMode, Qwen4GpuForward, Qwen4GpuForwardError,
    Qwen4OutputRows, QWEN4_FORWARD_HEADROOM_BYTES,
};
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_gpu::{MtpAppendScratch, MtpFillContext, MtpGpuStateSnapshot, MtpStep, Qwen4MtpGpu};
use crate::ple::PleHashMetadata;
use crate::state::decode_map_ahead_tokens;
use crate::state::Qwen4StateFormat;
use crate::state::{Qwen4State, Qwen4StateSnapshot, StateError};
use crate::weights::{
    ple_valid_rows_for_shard, Qwen4Manifest, Qwen4Placement, Qwen4TierRecipe, Qwen4Weights,
    WeightError, PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
};
use hipfire_runtime::external_rows::{RowEncoding, RowStore, RowStoreError};
use hipfire_runtime::model_source::{SourceFormat, SourceRangeDescriptor};
use hipfire_runtime::ple_hot_store::HotRowStore;
use hipfire_runtime::sampler::{
    apply_logit_policy_candidates_cpu, apply_logit_policy_cpu, PenaltyTable, SamplerConfig,
};
use hipfire_runtime::session_cache::{
    stride_boundaries, SessionCache, SessionDriver, SessionPieceKey, SessionRoute, SessionState,
    SnapshotParts, StateLayout,
};
use hipfire_runtime::spec_sampling::{SampleSpec, SparseDist};
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{WeightLoadTransaction, WeightStoreError};
use rdna_compute::sampling::PenaltyTableStage;
use rdna_compute::{Gpu, GpuTensor};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[path = "session_pair.rs"]
mod session_pair;
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
pub(crate) use session_pair::SlowlinkFailPoint;
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
pub use session_pair::SLOWLINK_PRIMITIVES;
pub(crate) use session_pair::{PairSessionStage, PromptTxn, SlowlinkOwner, SlowlinkScope};

#[path = "bundle/pp.rs"]
mod pp;
use hipfire_runtime::multi_gpu::executor::MeshRoute;
use pp::{Qwen4PpBandMeta, Qwen4PpMesh};
pub use pp::{Qwen4PpLaneExecutor, Qwen4PpWindow};

mod image_context;
use image_context::ImageContext;
pub use image_context::ImageSpan;

const PLE_RESET_TIMEOUT: Duration = Duration::from_secs(5);

fn pp_owner_required() -> BundleError {
    BundleError::Forward(
        "qwen4 topology.pp requires the admitted Gpus owner; use the mesh entry point".into(),
    )
}

/// Architecture-private owner for the fulfilled manifest census.
///
/// Assembly takes logical resident handles into [`Qwen4Weights`]. Arena
/// allocations, when selected, remain owned by this transaction while those
/// handles are borrowed views. Its immutable projections, aliases, external
/// PLE descriptors and arena roots drain after state and resident handles.
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
/// too (a missing head cannot continue; a retired request's head lags until
/// it catches up, at the latest in the terminal pending-seed flush), and
/// never while a verify row capture is armed. Plain data, so it runs without
/// a GPU.
fn live_position(
    route: SessionRoute,
    target_position: usize,
    head_position: Option<usize>,
    row_capture_armed: bool,
) -> Option<usize> {
    (!row_capture_armed && (route == SessionRoute::Ar || head_position == Some(target_position)))
        .then_some(target_position)
}

/// Environment flag that adds message-end snapshot boundaries (P1b), read once
/// when the bundle is assembled. Off by default.
const TURN_SNAPSHOTS_ENV: &str = "HIPFIRE_QWEN4_TURN_SNAPSHOTS";
/// Lineage-only split before the final message end, read once at assembly.
const TAIL_SNAPSHOT_ENV: &str = "HIPFIRE_QWEN4_TAIL_SNAPSHOT";
const TAIL_SNAPSHOT_MIN_GAP_ENV: &str = "HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_GAP";
const TAIL_SNAPSHOT_MIN_PROMPT_ENV: &str = "HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_PROMPT";
/// Below these the tail split's extra small tile costs more than the replay it
/// can save: a gap under 1024 rows replays in well under a second, and on a
/// short prompt the tile is a large share of the cold prefill.
const TAIL_SNAPSHOT_MIN_GAP_DEFAULT: usize = 1024;
const TAIL_SNAPSHOT_MIN_PROMPT_DEFAULT: usize = 16384;

/// The PLE rows owner of a bundle: present exactly when the profile declares
/// PLE (`config.ple`), so a profile without it holds no row store, hash
/// metadata, hot store or prewarm — and no placeholder for them.
pub(crate) struct Qwen4BundlePle {
    /// Bounded model-owned PLE reader/cache.  It must quiesce before source
    /// descriptors and the attached transaction are dropped.
    pub(crate) rows: RowStore,
    pub(crate) metadata: PleHashMetadata,
    /// Where the PLE rows come from: the model file itself or the sidecar
    /// whose descriptors the row store was built from (never
    /// [`Qwen4PleRows::Absent`]).
    source: Qwen4PleRows,
    /// PLE hot-row store and its worker thread (`attach_ple_hot_store`);
    /// `None` = off. Dropping it stops and joins the worker; `free_gpu` does
    /// so before the row store unloads.
    hot: Option<PleHotWorker>,
    /// Page-cache prewarm of a Q4LM PLE sidecar (`attach_ple_prewarm`); `None`
    /// = off.  Like `hot` it reads the sidecar, so dropping it stops and
    /// joins its thread, and `free_gpu` does so before the row store unloads.
    prewarm: Option<hipfire_runtime::ple_prewarm::PleRowsPrewarm>,
}

/// Published Qwen4 architecture owner.
pub struct Qwen4Bundle {
    pub config: Qwen4Config,
    pub weights: Qwen4Weights,
    pub state: Qwen4State,
    /// Resident image tower; absent when vision is disabled or no sidecar was admitted.
    pub vision: Option<(
        crate::vision::Qwen4VisionConfig,
        crate::vision::Qwen4VisionWeights,
    )>,
    pub(crate) image_context: Option<ImageContext>,
    /// PLE rows, hash metadata and their attachments; `None` exactly when
    /// `config.ple` is `None`.
    pub(crate) ple: Option<Qwen4BundlePle>,
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
    /// DFlash2 drafter's target layers and capture rows (`None`: no DFlash2
    /// drafter is armed; a forward then captures the MTP wide hidden).
    pub(crate) dflash: Option<crate::dflash_target::Qwen4DflashTarget>,
    /// Lab observation of every speculative forward's logits; unarmed it
    /// records nothing (`crate::lab_window`).
    #[cfg(feature = "lab")]
    lab_logits: crate::lab_window::LabLogitsTap,
    /// Staging of the GPU penalty prepass (`apply_penalty_table`), allocated
    /// by the first penalized row and kept until unload.
    penalty_stage: Option<PenaltyTableStage>,
    /// Prefill snapshot cache shared by every session plus this
    /// conversation's committed live state (`attach_session_cache`); `None` =
    /// off. Message-end snapshots of a live continuation live in their own
    /// `/turns` scope (`HIPFIRE_QWEN4_TURN_SNAPSHOTS`).
    session: Option<SessionDriver>,
    pub(crate) pair_session_stage: Option<PairSessionStage>,
    /// Slow-link (S1) layer-major prompt transaction owner: admitted profile,
    /// boundary stages, the open window, receipt and failure injection.
    pub(crate) slowlink: SlowlinkOwner,
    /// `HIPFIRE_QWEN4_TURN_SNAPSHOTS` (read once at assembly): also snapshot
    /// at message ends. Active only with `turn_end_token`.
    turn_snapshots: bool,
    tail_snapshot: bool,
    tail_snapshot_min_gap: usize,
    tail_snapshot_min_prompt: usize,
    /// The tokenizer's `<|im_end|>` id (`set_turn_end_token`).
    turn_end_token: Option<u32>,
    /// Request-state storage, part of every session snapshot's scope.
    state_format: Qwen4StateFormat,
    /// Exact AR lane store of the Flash-Next batch route (`stage_lanes`);
    /// `None` = not staged. Lane steps never touch `state` or `mtp`.
    pub(crate) lanes: Option<crate::lane::Qwen4LaneStore>,
    /// Expert-parallel mesh (`attach_ep_mesh`): this bundle is rank 0 and every
    /// daemon-facing call is replicated on the peer ranks. `None` = single
    /// device (the default; no behaviour change).
    pub(crate) ep: Option<Box<Qwen4EpMesh>>,
    /// Band state and generic execution plans. This bundle is the output band;
    /// callers borrow the sole hardware owner, `LoadedModel::pp_gpus`.
    pub(crate) pp: Option<Box<Qwen4PpMesh>>,
    /// This bundle's band identity under an attached PP mesh (index/count, layer
    /// range, complete placement, devices, mesh and load epoch, canonical
    /// chunk). Stamped on every band, root included, at attach so the session
    /// scope and boundary chunk never depend on the taken forward or the
    /// detached mesh; `None` outside PP.
    pub(crate) pp_band: Option<Qwen4PpBandMeta>,
    /// This rank's binding to its expert-parallel load's seal
    /// (`bind_ep_seal`); `None` on a single-device load.
    pub(crate) ep_seal: Option<crate::ep_seal::Qwen4EpRankSeal>,
    /// Accepted trunk overlay sidecar (`attach_trunk_overlay`), consumed by
    /// `attach_forward`; `None` = none pending.
    pub(crate) trunk_overlay: Option<crate::trunk_overlay::Qwen4TrunkOverlay>,
    pub(crate) npu_model_index: Option<Result<String, String>>,
    /// Actually reserved D1 CPU-owned expert window rows (logical window rows; 0 =
    /// none). Stored only by the loader after the real prewarm lands (see
    /// [`Self::note_cpu_d1_wide_storage_prewarmed`]), never inferred from scope: a
    /// wide window stays refused until it is nonzero. Storage only, never a
    /// qualification: the routing slice still gates on the per-call lab-wide scope
    /// and the caller's live thread-local capacity.
    pub(crate) cpu_d1_storage_rows: usize,
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
    // A Q8Ef snapshot carries the plain Q8 payload only: the live F16
    // residual plane is never snapshotted (it is zeroed on every restore), so
    // the size is the `storage()` format's, equal to Q8's.
    let gdn_storage = format.gdn.storage();
    let gdn_recurrent = gdn_storage.state_units(
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
    let gdn = part(gdn_recurrent.checked_mul(gdn_storage.dtype().size())?)?.checked_add(part(
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
        .checked_mul(config.hidden_size)?
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
            .checked_add(selected + part(4)? + part(feedback)?)?
            // The head's target boundary-hidden row.
            .checked_add(part(feedback)?)?;
    }
    u64::try_from(total).ok()
}

/// Device bytes one slow-link boundary stage costs (the target's fixed state:
/// GDN recurrent + conv, QSA selections, PLE convolution, HC feedback, each
/// part rounded to the cache's 256-byte packing): a snapshot of zero rows per
/// stream. The entry snapshot reuses the state's existing arena and adds none.
pub fn prompt_stage_bytes(config: &Qwen4Config, format: Qwen4StateFormat) -> Option<u64> {
    session_snapshot_bytes(config, format, false, 0)
}

/// Host-only session settings every PP band must share with the root: they shape
/// the boundary schedule (tail split, message ends), so they are part of each
/// band's scope and the root's values are authoritative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PpSessionPolicy {
    pub(crate) turn_snapshots: bool,
    pub(crate) tail_snapshot: bool,
    pub(crate) tail_snapshot_min_gap: usize,
    pub(crate) tail_snapshot_min_prompt: usize,
    pub(crate) turn_end_token: Option<u32>,
}

impl PpSessionPolicy {
    /// The schedule-relevant suffix of a PP band's scope. `turn_snapshots` is
    /// already carried by the base scope's `/turns`.
    fn scope_tag(&self) -> String {
        let token = self
            .turn_end_token
            .map_or_else(|| "-".to_string(), |token| token.to_string());
        format!(
            "/tail{}:{}:{}/te{token}",
            u8::from(self.tail_snapshot),
            self.tail_snapshot_min_gap,
            self.tail_snapshot_min_prompt
        )
    }
}

/// The session-cache scope of one bundle. Single (`band == None`) is exactly the
/// historical string for the F32 and Q8 GDN formats; Q8Ef appends the distinct
/// `/q8_ef-reset` spelling after the format (its snapshots carry Q8 payload
/// only and restore zeroes the live residual, so they never alias a plain Q8
/// or F32 scope). A PP band appends its band identity (index/count, layers,
/// complete placement, devices, mesh and load epoch) and the shared schedule
/// policy, so no two bands, placements, devices, meshes or loads alias.
pub(crate) fn session_scope(
    route: SessionRoute,
    chunk: usize,
    format: Qwen4StateFormat,
    turns: bool,
    band: Option<(&Qwen4PpBandMeta, PpSessionPolicy)>,
) -> String {
    // Message-end snapshots split prefill off the plain chunk schedule, so
    // they never share a domain with chunk-only ones.
    let turns = if turns { "/turns" } else { "" };
    let gdn_abi = match format.gdn {
        crate::GdnStateFormat::F32 | crate::GdnStateFormat::Q8 => "",
        crate::GdnStateFormat::Q8Ef => "/q8_ef-reset",
    };
    let mut scope = format!("qwen4/{route:?}/chunk{chunk}/{format:?}{gdn_abi}{turns}");
    if let Some((band, policy)) = band {
        scope.push_str(&band.scope_tag());
        scope.push_str(&policy.scope_tag());
    }
    scope
}

impl SessionState for Qwen4Bundle {
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String> {
        // An attached root is not a band: its aggregate scope is the mesh
        // record. Detached (inside mesh callbacks) it is the band's own.
        if self.pp.is_some() {
            return None;
        }
        let chunk = self.spec_chunk_rows()?;
        if route == SessionRoute::Mtp && self.mtp.is_none() {
            return None;
        }
        let mut scope = session_scope(
            route,
            chunk,
            self.state_format,
            self.turn_snapshots_active(),
            self.pp_band
                .as_ref()
                .map(|band| (band, self.pp_session_policy())),
        );
        // A slow-link load mixes in `tile_rows` tiles but checkpoints on its
        // own cadence (`checkpoint_stride_rows`): the pair is a distinct
        // numeric/cache identity, never aliasing an explicit chunk-1536
        // cadence or the incumbent chunk-8192 math.
        scope.push_str(&session_pair::slowlink_scope_tag(
            self.slowlink
                .profile
                .map(|profile| (profile.tile_rows, profile.checkpoint_stride_rows)),
        ));
        // Approximate CPU arithmetic is a different prefix-state domain. Bind
        // the actual armed layers and immutable hot-set/placement digest while
        // preserving the GPU-only and pipeline-band scope.
        if let Some(execution) = self.execution.as_ref() {
            let host = execution.host_exec_identity();
            if !host.is_empty() {
                scope.push('/');
                scope.push_str(&host);
            }
        }
        Some(scope)
    }

    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
        let Some(chunk) = self.session_cadence_rows() else {
            return Vec::new();
        };
        stride_boundaries(chunk, prompt, self.turn_end_token_active(), after, up_to)
    }

    fn lineage_boundary(&self, prompt: &[u32], after: usize, up_to: usize) -> Option<usize> {
        if !self.tail_snapshot {
            return None;
        }
        let chunk = self.session_cadence_rows()?;
        let token = self.turn_end_token?;
        let p = prompt[..up_to].iter().rposition(|&id| id == token)?;
        if p <= after
            || prompt.len() < self.tail_snapshot_min_prompt
            || p - after.max((p / chunk) * chunk) < self.tail_snapshot_min_gap
            || up_to / chunk > p / chunk
            || p % chunk == 0
            || self.turn_end_token_active() == Some(prompt[p - 1])
        {
            return None;
        }
        Some(p)
    }

    fn snapshot_parts(
        &mut self,
        route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String> {
        if self.pp.is_some() {
            return Err("session snapshots are not supported under qwen4 topology.pp yet".into());
        }
        self.quiesce_ple().map_err(|e| e.to_string())?;
        let mtp = match route {
            SessionRoute::Ar => None,
            SessionRoute::Mtp => Some(
                self.mtp
                    .as_ref()
                    .ok_or("Qwen4 MTP resources are not attached")?,
            ),
        };
        let staged = self
            .pair_session_stage
            .as_ref()
            .filter(|stage| stage.position == Some(position))
            .or_else(|| {
                self.slowlink
                    .stages
                    .iter()
                    .find(|stage| stage.position == Some(position))
            });
        if (staged.is_none() && self.state.position != position)
            || mtp.is_some_and(|mtp| mtp.position() != position)
        {
            return Err(format!("qwen4 session capture: owners not at {position}"));
        }
        let SnapshotParts {
            mut meta,
            mut layout,
        } = match staged {
            Some(stage) => stage
                .snapshot_parts(&self.state)
                .map_err(|e| e.to_string())?,
            None => {
                let (meta, layout) = self.state.session_parts().map_err(|e| e.to_string())?;
                SnapshotParts { meta, layout }
            }
        };
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
        if self.pp.is_some() {
            return Err("session restore is not supported under qwen4 topology.pp yet".into());
        }
        let (target_meta, mtp_meta) = meta
            .split_at_checked(self.state.session_meta_bytes())
            .ok_or("qwen4 session snapshot shape mismatch")?;
        if route == SessionRoute::Mtp {
            // Refuse an MTP snapshot of another shape or without a boundary
            // row for its position before any owner mutates.
            self.mtp
                .as_ref()
                .ok_or("Qwen4 MTP resources are not attached")?
                .check_session_restore(mtp_meta)
                .map_err(|e| e.to_string())?;
        }
        self.invalidate_ple_epoch().map_err(|e| e.to_string())?;
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

    /// An MTP snapshot is restorable only when its head metadata passes the
    /// same checks `restore_parts` runs first (shape, mark, and the boundary
    /// row for its own position), so the session cache plans around one an
    /// older build captured without the row: its prefix is prefilled again.
    /// Without attached head resources the restore itself reports that.
    fn snapshot_restorable(&self, route: SessionRoute, meta: &[u8]) -> bool {
        if route != SessionRoute::Mtp {
            return true;
        }
        let Some(mtp) = self.mtp.as_ref() else {
            return true;
        };
        meta.split_at_checked(self.state.session_meta_bytes())
            .is_some_and(|(_, mtp_meta)| mtp.check_session_restore(mtp_meta).is_ok())
    }

    fn finish_restore(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String> {
        if self.pp.is_some() {
            return Err("session restore is not supported under qwen4 topology.pp yet".into());
        }
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

    fn side_request_rows(&self, route: SessionRoute, through: usize) -> Option<Vec<usize>> {
        if through > self.state.max_seq_len || self.state.row_capture_armed {
            return None;
        }
        let mut rows = self.state.side_request_rows(through);
        if route == SessionRoute::Mtp {
            rows.extend(self.mtp.as_ref()?.side_request_rows(through));
        }
        Some(rows)
    }

    fn reset_for_side(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.begin_npu_trunk_request(gpu)
            .map_err(|e| e.to_string())?;
        if let Some(stage) = &mut self.pair_session_stage {
            stage.position = None;
        }
        self.slowlink.clear_active();
        self.invalidate_ple_epoch().map_err(|e| e.to_string())?;
        self.state.reset_for_side(gpu).map_err(|e| e.to_string())?;
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset_for_side(gpu).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        Qwen4Bundle::reset(self, gpu).map_err(|e| e.to_string())
    }
}

impl Qwen4Bundle {
    /// Assemble a complete Single bundle using metadata parsed from the
    /// artifact's canonical `qwen4_ple` object. `recipe` is the recipe the
    /// artifact was admitted under (`Qwen4HfqmArtifact::recipe()`).
    /// `state_format` is the request state's storage
    /// ([`crate::resolve_state_format`]); `backend` the QSA context storage
    /// load admission resolved, shared by target and MTP.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        config: Qwen4Config,
        recipe: &Qwen4TierRecipe,
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
            recipe,
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
    /// `metadata` is `Some` exactly when the profile declares PLE (the
    /// admission receipt's `ple`): the rows are then the embedded shard
    /// tensors; a profile without PLE passes `None` and gets no rows source.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_with_metadata(
        config: Qwen4Config,
        recipe: &Qwen4TierRecipe,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: impl Into<Option<PleHashMetadata>>,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, BundleError> {
        let metadata = metadata.into();
        let ple_source = if metadata.is_some() {
            Qwen4PleRows::Embedded
        } else {
            Qwen4PleRows::Absent
        };
        Self::assemble_with_ple_rows(
            config,
            recipe,
            transaction,
            placements,
            gpu,
            max_seq_len,
            metadata,
            ple_source,
            state_format,
            backend,
            None,
        )
    }

    /// Assemble with validated metadata and an explicit PLE rows source:
    /// [`Qwen4PleRows::Embedded`] reads the shard tensors fulfilled in
    /// `transaction` (the monolithic artifact, exactly the pre-sidecar path);
    /// [`Qwen4PleRows::Sidecar`] builds the row store from the sidecar's
    /// descriptors, and `transaction` then carries no PLE shards (the
    /// admitted manifest was trimmed).  The row geometry (shards, rows per
    /// shard, row width, heads) is the profile's `config.ple`.
    /// A profile without PLE takes `None` metadata and
    /// [`Qwen4PleRows::Absent`] and builds no PLE owner; any other pairing of
    /// the profile with the metadata and rows source, and a sidecar for a
    /// developer profile (sidecar rows, the hot store and the rank table are
    /// Flash-Next-only), is refused before weights are assembled or state is
    /// allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_with_ple_rows(
        config: Qwen4Config,
        recipe: &Qwen4TierRecipe,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: impl Into<Option<PleHashMetadata>>,
        ple_source: Qwen4PleRows,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
        expert_tiers: Option<crate::expert_tier_owners::ExpertTierLease>,
    ) -> Result<Self, BundleError> {
        let scope = crate::weights::Qwen4BandScope::single(&config);
        Self::assemble_band_with_ple_rows(
            config,
            recipe,
            transaction,
            placements,
            gpu,
            max_seq_len,
            metadata,
            ple_source,
            state_format,
            backend,
            expert_tiers,
            &scope,
        )
    }

    /// Assemble precisely one admitted pipeline band, including only its
    /// pinned PLE/input resources and band-local mutable state.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_band_with_ple_rows(
        config: Qwen4Config,
        recipe: &Qwen4TierRecipe,
        mut transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: impl Into<Option<PleHashMetadata>>,
        ple_source: Qwen4PleRows,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
        expert_tiers: Option<crate::expert_tier_owners::ExpertTierLease>,
        scope: &crate::weights::Qwen4BandScope,
    ) -> Result<Self, BundleError> {
        if let Err(error) = scope.validate(&config) {
            return Err(cleanup_transaction(
                BundleError::Forward(error),
                transaction.rollback(gpu),
            ));
        }
        let metadata = metadata.into();
        let ple_inputs = if scope.input {
            admit_ple_inputs(&config, metadata, ple_source)
        } else if metadata.is_none() && matches!(ple_source, Qwen4PleRows::Absent) {
            Ok(None)
        } else {
            Err(BundleError::Forward(
                "qwen4 topology.pp: only the input band owns PLE rows and metadata".into(),
            ))
        };
        let ple_inputs = match ple_inputs {
            Ok(inputs) => inputs,
            Err(error) => {
                return Err(cleanup_transaction(error, transaction.rollback(gpu)));
            }
        };
        let assembled = match expert_tiers {
            Some(lease) => Qwen4Weights::assemble_with_expert_tiers(
                &mut transaction,
                &config,
                recipe,
                placements,
                scope,
                lease,
            ),
            None => Qwen4Weights::assemble(&mut transaction, &config, recipe, placements, scope),
        };
        let weights = match assembled {
            Ok(weights) => weights,
            Err(error) => {
                return Err(cleanup_transaction(
                    BundleError::Weights(error),
                    transaction.rollback(gpu),
                ));
            }
        };
        let ple = match (ple_inputs, config.ple.as_ref()) {
            (Some(inputs), Some(ple_config)) => {
                // One prefill chunk prefetches `rows * head_count` n-gram rows
                // in a single row-store request, so staging holds the
                // requested chunk. The scratch mode is not known at assembly:
                // the largest-default mode (`FnSymGfx1201`, 16384 rows on
                // gfx1201) bounds every mode's chunk, so staging always
                // covers the chunk the forward later picks. An explicit
                // `HIPFIRE_PREFILL_CHUNK_ROWS` stays authoritative.
                let staging_rows = qwen4_prefill_chunk_requested(
                    gpu,
                    max_seq_len,
                    Qwen4ForwardScratchMode::FnSymGfx1201,
                ) * ple_config.head_count;
                match build_ple_row_store(
                    &inputs.source,
                    &transaction,
                    &weights.manifest,
                    ple_config,
                    &inputs.metadata,
                    staging_rows,
                    // `HIPFIRE_QWEN4_PLE_WIDE_READERS=0` keeps 16 concurrent
                    // reads for every PLE request, including a whole prefill
                    // chunk.
                    hipfire_config::developer_bool("HIPFIRE_QWEN4_PLE_WIDE_READERS", true),
                ) {
                    Ok(rows) => Some(Qwen4BundlePle {
                        rows,
                        metadata: inputs.metadata,
                        source: inputs.source,
                        hot: None,
                        prewarm: None,
                    }),
                    Err(error) => {
                        let weight_result = weights.free_gpu(gpu);
                        let cleanup = transaction.rollback(gpu);
                        return Err(cleanup_bundle_failure(error, weight_result, cleanup));
                    }
                }
            }
            _ => None,
        };
        let mut state = match Qwen4State::new_with_backend_scoped(
            gpu,
            &config,
            max_seq_len,
            state_format,
            backend,
            scope,
        ) {
            Ok(state) => state,
            Err(error) => {
                // `unload` consumes the reader and joins its worker even on a
                // quiesce error, so source descriptors cannot outlive failure.
                if let Some(ple) = ple {
                    let _ = ple.rows.unload();
                }
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
            vision: None,
            image_context: None,
            ple,
            weight_store: AttachedWeightStore::new(transaction),
            execution: None,
            mtp: None,
            spec_logits: None,
            spec_top1: None,
            spec_hidden: None,
            spec_host_top1: Vec::new(),
            dflash: None,
            #[cfg(feature = "lab")]
            lab_logits: crate::lab_window::LabLogitsTap::default(),
            penalty_stage: None,
            session: None,
            pair_session_stage: None,
            slowlink: SlowlinkOwner::default(),
            turn_snapshots: hipfire_config::developer_bool(TURN_SNAPSHOTS_ENV, false),
            tail_snapshot: hipfire_config::developer_bool(TAIL_SNAPSHOT_ENV, true),
            tail_snapshot_min_gap: hipfire_config::developer_var(TAIL_SNAPSHOT_MIN_GAP_ENV)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(TAIL_SNAPSHOT_MIN_GAP_DEFAULT),
            tail_snapshot_min_prompt: hipfire_config::developer_var(TAIL_SNAPSHOT_MIN_PROMPT_ENV)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(TAIL_SNAPSHOT_MIN_PROMPT_DEFAULT),
            turn_end_token: None,
            state_format,
            lanes: None,
            ep: None,
            pp: None,
            pp_band: None,
            ep_seal: None,
            trunk_overlay: None,
            npu_model_index: None,
            cpu_d1_storage_rows: 0,
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

    /// Return all numerically ordered PLE shard descriptors of the active
    /// rows source: the attached canonical transaction census for the
    /// embedded shards, or the sidecar's descriptors.
    ///
    /// This strict accessor is useful to admission and diagnostics callers;
    /// the bundle's own reader already owns the same sealed descriptors.
    pub fn ple_descriptors(&self) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
        let (Some(ple), Some(ple_config)) = (&self.ple, self.config.ple.as_ref()) else {
            return Err(no_ple_rows());
        };
        match &ple.source {
            Qwen4PleRows::Absent => Err(no_ple_rows()),
            Qwen4PleRows::Embedded => ple_descriptors(
                &self.weight_store.transaction,
                self.manifest(),
                ple_config,
                &ple.metadata,
            ),
            Qwen4PleRows::Sidecar {
                rows: SidecarRows::Descriptors(descriptors),
                ..
            } => sidecar_ple_descriptors(descriptors, &ple.metadata),
            Qwen4PleRows::Sidecar {
                path,
                rows: SidecarRows::Table(_),
                ..
            } => Err(BundleError::Weights(WeightError::DescriptorMismatch(
                format!(
                    "PLE rows {} are a rank-order table served through its store; \
                     it has no row-order shard descriptors",
                    path.display()
                ),
            ))),
        }
    }

    /// Where this bundle's PLE rows come from; `None` for a profile without
    /// PLE.
    pub fn ple_source(&self) -> Option<&Qwen4PleRows> {
        self.ple.as_ref().map(|ple| &ple.source)
    }

    /// The PLE row store; `None` for a profile without PLE.
    pub fn ple_rows(&self) -> Option<&RowStore> {
        self.ple.as_ref().map(|ple| &ple.rows)
    }

    pub fn ple_rows_mut(&mut self) -> Option<&mut RowStore> {
        self.ple.as_mut().map(|ple| &mut ple.rows)
    }

    /// Whole-request serving guard on the PLE row store's
    /// [`hipfire_runtime::external_rows::FetchActivity`]: the prewarm
    /// readers of the same table pause while it is held and resume once the
    /// store has been idle.  `None` for a profile without PLE.
    pub fn ple_serving_guard(&self) -> Option<hipfire_runtime::external_rows::ServingGuard> {
        self.ple.as_ref().map(|ple| ple.rows.activity().serving())
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

    /// The single S1 load plan shared by admission and its real allocation
    /// owners. `None` on an individually allocated incumbent load.
    pub fn slowlink_weight_arena_plan(
        &self,
    ) -> Option<&std::sync::Arc<rdna_compute::byte_arena::ModelWeightArenaPlan>> {
        self.weight_store.transaction.weight_arena_plan()
    }

    /// Actual live owning capacities, in the shared weight plan's root order.
    /// These are allocation capacities, not the planned requests.
    pub fn weight_arena_root_capacities(&self) -> Vec<usize> {
        self.weight_store.transaction.weight_arena_root_capacities()
    }

    /// Actual live shadow-root capacities, in plan order. An unallocated
    /// lazy root is `None`, never reported as an observed allocation.
    pub fn f16_shadow_arena_root_capacities(&self, gpu: &Gpu) -> Vec<Option<usize>> {
        match (
            self.slowlink_weight_arena_plan(),
            gpu.f16_shadow_arena_plan(),
        ) {
            (Some(weights), Some(shadows)) if std::sync::Arc::ptr_eq(weights, shadows) => {
                gpu.f16_shadow_arena_root_capacities()
            }
            _ => Vec::new(),
        }
    }

    /// Attach cheap base-HFQ identity captured before the source was moved.
    pub fn attach_npu_trunk_identity(&mut self, identity: Option<Result<String, String>>) {
        self.npu_model_index = identity;
    }

    /// Hand the bundle an accepted trunk overlay
    /// ([`crate::trunk_overlay::resolve_trunk_overlay`]).  [`Self::attach_forward`]
    /// applies it before `HIPFIRE_QWEN4_REQUANT`: a payload that fails its
    /// sha256 or the weights' pre-check logs `REFUSED` and the base weights
    /// run.  One overlay, before the forward attaches.
    pub fn attach_trunk_overlay(
        &mut self,
        overlay: crate::trunk_overlay::Qwen4TrunkOverlay,
    ) -> Result<(), BundleError> {
        if self.execution.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 trunk overlay must attach before the forward resources".to_string(),
            ));
        }
        if self.trunk_overlay.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 trunk overlay is already attached".to_string(),
            ));
        }
        self.trunk_overlay = Some(overlay);
        Ok(())
    }

    /// Apply the pending trunk overlay, if any.  A refusal (payload digest,
    /// weight pre-check) is logged and leaves the base weights; a failure
    /// after replacement started is an error.
    fn apply_pending_trunk_overlay(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<Option<rdna_compute::TrunkIu4Default>, BundleError> {
        let Some(overlay) = self.trunk_overlay.take() else {
            return Ok(None);
        };
        let payloads = match overlay.bound.read_verified() {
            Ok(payloads) => payloads,
            Err(reason) => {
                eprintln!(
                    "{}",
                    crate::trunk_overlay::overlay_refusal_line(overlay.path(), &reason)
                );
                return Ok(None);
            }
        };
        match self
            .weights
            .apply_trunk_overlay(gpu, &payloads)
            .map_err(BundleError::Forward)?
        {
            crate::weights::TrunkOverlayOutcome::Refused(reason) => {
                eprintln!(
                    "{}",
                    crate::trunk_overlay::overlay_refusal_line(overlay.path(), &reason)
                );
            }
            crate::weights::TrunkOverlayOutcome::Applied(applied) => {
                self.config.trunk_iu4_default = Some(overlay.default);
                eprintln!("{}", crate::trunk_overlay::applied_line(&overlay, &applied));
                return Ok(Some(overlay.default));
            }
        }
        Ok(None)
    }

    /// Whether every call of this load runs host arithmetic: the D1 CPU
    /// splice, or a D2 split requested by the explicit `memory.offload_exec=cpu`
    /// (the speculative-width caps key on this). A D2 split armed only by the
    /// measured per-load default is per call and does not count; see
    /// [`Self::split_layers`].
    pub fn has_cpu_deferred_experts(&self) -> bool {
        self.execution
            .as_ref()
            .is_some_and(Qwen4GpuForward::has_cpu_deferred_experts)
    }

    /// Verify rows the armed CPU routed-expert modes (D1 splice, any D2 tier
    /// split including the measured per-call default) run row-exact, from
    /// their own ownership (not [`Self::has_cpu_deferred_experts`], which
    /// excludes the per-call default); `None` when no layer is CPU-owned. The
    /// native MTP `auto` cap is one below the smaller of this and the GPU's
    /// row-exact rows.
    pub fn native_mtp_cpu_window_rows(&self) -> Option<usize> {
        self.execution
            .as_ref()
            .and_then(Qwen4GpuForward::native_mtp_cpu_window_rows)
    }
    /// Actually reserved D1 CPU-owned expert storage rows (logical window rows) of
    /// this load: `0` when no lab-wide 64-row window was admitted AND prewarmed
    /// (the legacy default, on-demand growth, byte-identical placement), else the
    /// stored admitted + prewarmed window saturating at
    /// [`hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS`]. Storage only —
    /// never a qualification, never a default: wide routing still reads
    /// [`Self::cpu_qualified_verify_rows`] with the per-call lab-wide scope, which
    /// additionally gates on the caller's live thread-local capacity.
    /// Stored (loader-proven), never inferred from scope: set to nonzero only by
    /// [`Self::note_cpu_d1_wide_storage_prewarmed`] after the real prewarm lands, so
    /// a wide window stays refused — fail-closed, never a metadata claim.
    pub fn cpu_d1_storage_capacity_rows(&self) -> usize {
        self.cpu_d1_storage_rows
    }

    /// Common actual CPU-owned expert capacity: the BINDING minimum of every actually
    /// owned store — the stored D1 window when a D1 splice is armed, and the minimum
    /// live D2 executor capacity across armed split layers. Unowned modes contribute
    /// nothing (they must not mask an owned one): D1 storage 0 on a D2-only load does
    /// NOT clip the D2 executor's 64, and absent executors do not clip a D1 64.
    /// `0` when no CPU expert mode owns storage. Storage only, never qualification:
    /// routing still gates on the per-call lab-wide scope + live capacities.
    /// Rate reads this as the physical cap (with the still-frozen qualified getter).
    pub fn cpu_storage_capacity_rows(&self) -> usize {
        let d1 = self
            .has_cpu_deferred_experts()
            .then(|| self.cpu_d1_storage_rows);
        match (d1, self.execution.as_ref()) {
            (Some(d1), Some(forward)) => match forward.d2_storage_capacity_rows() {
                Some(d2) => d1.min(d2),
                None => d1,
            },
            (Some(d1), None) => d1,
            (None, Some(forward)) => forward.d2_storage_capacity_rows().unwrap_or(0),
            (None, None) => 0,
        }
    }

    /// Qualified CPU verify width of this load, from actual storage plus scope:
    /// [`crate::program::QWEN4_CPU_SPLICE_MAX_ROWS`] (4) on the default route;
    /// the actual stored window only when `lab_armed` (the caller's per-call lab-wide
    /// presence gate, from the canonical override signal on the sealing device —
    /// never a stored constant) AND the admitted+prewarmed storage above covers it.
    /// The frozen GPU width is informational arm width only — it never narrows the
    /// CPU result below actual storage (wide8 presence with 64-row storage qualifies
    /// the FULL R1..64 CPU backend for both lab arms). No override, no constant-64
    /// claim: without the scope (or without admitted storage) this stays 4.
    /// Pass the ACTUAL per-call override presence: storage without presence must not
    /// qualify (prewarmed 64 + cleared override ⇒ product 4/K3, not a GPU fallback).
    pub fn cpu_qualified_verify_rows(&self, gpu: &Gpu, lab_armed: bool) -> usize {
        use hipfire_dispatch::families::moe::CpuWideVerifyScope;
        let scope = CpuWideVerifyScope::from_lab_presence(
            lab_armed,
            crate::gpu_forward::qwen4_row_exact_verify_rows(gpu),
        );
        scope.qualified_max_rows(self.cpu_storage_capacity_rows())
    }

    /// Planned D1 storage need in rows: 0 unless carrying an admitted native capture
    /// (the loader reserves the capture before any window grows it); the native-MTP
    /// capture rows are the window need (`max_k` drafts + 1, saturating the physical
    /// bound). `None` max_k (AR load) requests nothing. Pure, no allocation.
    pub fn cpu_d1_requested_window_rows(&self) -> usize {
        self.native_mtp_capture_rows()
    }

    /// Whether the load qualifies for a lab-wide D1 reservation before prewarm: the
    /// single non-EP/PP/lane Flash-Next scope with an armed D1 splice
    /// ([`Self::has_cpu_deferred_experts`]) whose allocated GPU scratch actually holds
    /// the capture rows. D2-only loads (armed split, no D1 splice) are excluded: they
    /// reserve their own executor storage and must never prewarm an unused D1 pinned
    /// stage (the TLS host-slot scratch is shared; the pinned stage is per-mode).
    /// Scope gate only (actual rows come from [`Self::cpu_d1_requested_window_rows`]);
    /// the stored capacity above stays 0 until the real prewarm lands.
    pub fn cpu_d1_owned_single_fn(&self) -> bool {
        if !self.config.profile.is_flash_next()
            || self.ep.is_some()
            || self.pp.is_some()
            || self.pp_band.is_some()
            || self.lanes.is_some()
        {
            return false;
        }
        if !self.has_cpu_deferred_experts() {
            return false;
        }
        let Some(forward) = self.execution.as_ref() else {
            return false;
        };
        let need = self.cpu_d1_requested_window_rows();
        need > 0
            && forward.scratch.max_chunk >= need
            && crate::gpu_forward::qwen4_spec_logit_rows(forward.scratch.max_chunk) >= need
    }

    /// Geometry `(hidden, intermediate, top_k)` the D1 splice actually runs: the
    /// model config (`None` for zero `top_k`, which admits no expert window).
    pub fn cpu_d1_geometry(&self) -> Option<(usize, usize, usize)> {
        (self.config.num_experts_per_tok != 0).then_some((
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            self.config.num_experts_per_tok,
        ))
    }

    /// Record the loader's real D1 prewarm: the actually reserved window `rows`
    /// (logical rows, 1..=STORAGE, derived from [`cpu_d1_required_storage_rows`] and
    /// reserved on the execution thread) after BOTH reservations land — the pinned
    /// transfer stage (`2 * (rows * dim + rows * top_k)` f32 values, exactly
    /// `moe_cpu_experts`' `(n_x + slots) * 8` stage bytes / 4) and the host-slot
    /// scratch (`rows * top_k` slots) — with the GPU full-slot views checked
    /// Refuses (returns `false`, storing nothing) when the scope/need is absent, when
    /// `rows` exceeds the physical storage bound, or when live thread-local capacity
    /// ([`hipfire_dispatch::families::moe::cpu_d1_prewarmed_scratch_rows`]) does not
    /// cover `rows` on the calling thread: no fake admitted metadata ever lands.
    /// Called once, before the first window; retained pointer invalidations precede
    /// the reservations (never after), so growth cannot strand live views.
    pub fn note_cpu_d1_wide_storage_prewarmed(&mut self, rows: usize, device_id: i32) -> bool {
        let Some((dim, mi, top_k)) = self.cpu_d1_geometry() else {
            return false;
        };
        if rows == 0
            || rows > hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS
            || !self.cpu_d1_owned_single_fn()
        {
            return false;
        }
        let need = crate::expert_residency::cpu_d1_required_storage_rows(
            true,
            true,
            self.cpu_d1_requested_window_rows(),
        );
        if need == 0 || rows > need {
            return false;
        }
        let live = hipfire_dispatch::families::moe::cpu_d1_prewarmed_scratch_rows(
            dim, mi, top_k, device_id,
        );
        if live < rows {
            return false;
        }
        self.cpu_d1_storage_rows = rows;
        true
    }

    /// Load-only D2 prewarm/reserve for the REAL requested window rows: resize every
    /// armed split layer's default-4 executor to `requested_rows` (preserving the
    /// load-immutable Return/Ready/Join selections), on the execution thread, BEFORE
    /// the first window — never with a window in flight (the old executor is freed
    /// before its replacement is built, so no old+new overlap peak exists to charge).
    /// Unarmed layers (no cold owner / no executor) and executors that already cover
    /// the request are a no-op. Returns whether any layer resized. A reservation
    /// failure leaves that layer without an executor and returns the error: the caller
    /// must fail the load (free GPU, load error), never run narrowed or silent.
    pub fn reserve_cpu_d2_wide_rows(
        &mut self,
        gpu: &mut Gpu,
        requested_rows: usize,
    ) -> Result<bool, BundleError> {
        let Some(forward) = self.execution.as_mut() else {
            return Ok(false);
        };
        let config = &self.config;
        let mut resized_any = false;
        for layer in forward.moe.iter_mut().flatten() {
            let resized = layer
                .reserve_d2_wide_rows(gpu, config, requested_rows)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
            resized_any |= resized;
        }
        Ok(resized_any)
    }

    /// Trunk layers armed for the opt-in CPU-cold/GPU-hot routed slot join.
    pub fn split_layers(&self) -> Vec<usize> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_layers)
            .unwrap_or_default()
    }

    /// Stable numerical identity of the frozen CPU split and expert hot set.
    pub fn split_digest(&self) -> Option<String> {
        self.execution
            .as_ref()
            .and_then(Qwen4GpuForward::split_digest)
    }

    /// Owner kind (`"tier"` or `"whole-layer"`) of every armed D2 split layer,
    /// as `(layer label, kind)` in layer order.
    pub fn split_owner_kinds(&self) -> Vec<(usize, &'static str)> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_owner_kinds)
            .unwrap_or_default()
    }

    /// D2 cold-result return mode (`"mapped"` or `"copy"`) of every armed
    /// D2 split layer, as `(layer label, mode)` in layer order.
    pub fn split_return_modes(&self) -> Vec<(usize, &'static str)> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_return_modes)
            .unwrap_or_default()
    }

    /// D2 input readiness mode (`"event"` or `"sync"`) of every armed D2 split
    /// layer, as `(layer label, mode)` in layer order.
    pub fn split_ready_modes(&self) -> Vec<(usize, &'static str)> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_ready_modes)
            .unwrap_or_default()
    }

    /// D2 join mode (`"cpwait"` or `"host"`) of every armed D2 split layer, as
    /// `(layer label, mode)` in layer order.
    pub fn split_join_modes(&self) -> Vec<(usize, &'static str)> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_join_modes)
            .unwrap_or_default()
    }

    /// Cumulative D2 readiness/join counters of every armed D2 split layer, as
    /// `(layer label, [ready_polls, ready_fallback_syncs, cpwait_joins,
    /// cpwait_aborts, watchdog_fires])` in layer order.
    pub fn split_readiness_counters(&self) -> Vec<(usize, [u64; 5])> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_readiness_counters)
            .unwrap_or_default()
    }

    /// Numerical identity of every armed CPU arithmetic mode (D1 and the D2
    /// owner kinds); empty when none is armed.
    pub fn host_exec_identity(&self) -> String {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::host_exec_identity)
            .unwrap_or_default()
    }

    /// Lab-only slot capture / serial-overlap controls. Off leaves production
    /// execution unchanged; endpoint overrides require explicit configuration.
    pub fn split_parity_configure(
        &mut self,
        config: hipfire_dispatch::cpu_split::SplitParityConfig,
    ) -> Result<(), BundleError> {
        self.execution
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 forward resources are not attached".into()))?
            .split_parity_configure(config)
            .map_err(|e| BundleError::Forward(e.to_string()))
    }

    /// Drain bounded lab parity records in execution order.
    pub fn split_take_parity_records(
        &mut self,
    ) -> Vec<hipfire_dispatch::cpu_split::SplitParityRecord> {
        self.execution
            .as_ref()
            .map(Qwen4GpuForward::split_take_parity_records)
            .unwrap_or_default()
    }

    /// Attach reusable ordinary-HIP execution resources after the manifest
    /// transaction and model state have assembled successfully.
    ///
    /// `mode` is the forward scratch plan the caller admitted and, for the
    /// loader, already charged to the automatic expert placement; the chunk
    /// rung search and the forward allocate exactly that plan.
    ///
    /// `max_chunk` caps the chunk: the requested rows
    /// ([`qwen4_prefill_chunk_requested`]) are taken `min` it, so it can only
    /// narrow them. The loader passes the chunk its expert placement was
    /// sized from (the same requested chunk, never widened here); other
    /// callers pass `max_seq`. The rung search below may still pick a smaller
    /// rung against live free memory.
    pub fn attach_forward(
        &mut self,
        gpu: &mut Gpu,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<(), BundleError> {
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
        // at assembly bounds it.  A profile without PLE has no row store.
        let ple_rows_cap = match (&self.ple, &self.config.ple) {
            (Some(ple), Some(ple_config)) => {
                ple.rows.max_rows_per_prefetch() / ple_config.head_count
            }
            _ => usize::MAX,
        };
        // An admitted slow-link profile fixes the forward's chunk at EXACTLY
        // its charged tile (never the incumbent helper's wider default, which
        // would retain scratch the profile did not charge); the caller's and
        // the PLE store's capacity must cover it, checked before any
        // allocation. Every other load keeps the incumbent request unchanged.
        let s1_tile = self.slowlink.profile.map(|profile| profile.tile_rows);
        let requested =
            session_pair::attach_chunk_request(s1_tile, max_chunk, ple_rows_cap, || {
                qwen4_prefill_chunk_requested(gpu, max_chunk, mode)
            })
            .map_err(BundleError::Forward)?;
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
        if let Some(qsa) = self.state.qsa.first().filter(|qsa| qsa.f16_arena.is_none()) {
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
        // The gfx1201 BF16 split selector (`HIPFIRE_QWEN4_QSA_BF16_SPLIT`)
        // keeps its score / filter scratch in the selector slot. Reserve the
        // route's constant upper bound now, before any capture or record, so
        // no retained decode tape ever sees that slot grow or move. A no-op
        // unless the route can run (not killed with `=0`, gfx1201, BF16 pooled keys).
        if let Some(qsa) = self.state.qsa.first() {
            let reserved = rdna_compute::tensor_ops::reserve_qsa_bf16_split_scratch(
                gpu,
                qsa.pooled_keys.dtype,
            )
            .map_err(BundleError::Hip)?;
            if reserved > 0 {
                eprintln!(
                    "  qwen4 QSA BF16 split selector: {} MiB score/filter scratch reserved",
                    reserved >> 20
                );
            }
        }
        // Admitted NPU numerics read stock MQ6, including z/a/b, before the
        // overlay swap. Effective off does no downloads, allocation or logging.
        #[cfg(target_os = "linux")]
        let mut npu_prepared = Qwen4GpuForward::prepare_npu_trunk(
            gpu,
            self,
            self.trunk_overlay.is_some(),
            requested,
            mode,
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?;
        let overlay_route = self.apply_pending_trunk_overlay(gpu)?;
        #[cfg(target_os = "linux")]
        if let Some(prepared) = &mut npu_prepared {
            prepared.set_overlay_route(overlay_route);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = overlay_route;
        self.weights
            .requant_from_env(gpu)
            .map_err(BundleError::Forward)?;
        // The largest rung whose chunk-sized resources fit the free device
        // memory beside the kernels' lazily sized workspaces; the smallest
        // rung is attempted regardless and fails at allocation if it must.
        let (free, _) = gpu.device_mem_info().map_err(BundleError::Hip)?;
        // A load that placed its experts for decode-share (`fixed` / `on`, a
        // donor tail in the lease) sized the placement for exactly this chunk and
        // this scratch plan: the chunk is final (no rung downshift) and a
        // compact -> Full re-plan cannot fit the donor tail.
        let sharing = crate::decode_share::DecodeShareMode::from_env().uses_donor_backing()
            && self
                .weights
                .expert_tiers()
                .is_some_and(|lease| lease.decode_share().is_some());
        let max_chunk = match s1_tile {
            // The charged tile, with no rung downshift: admission sized the
            // placement and the arena for exactly these rows.
            Some(tile) => tile,
            None if sharing => {
                let need = crate::gpu_forward::qwen4_decode_share_dedicated_bytes(
                    &self.config,
                    requested,
                    mode,
                )
                .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES))
                .ok_or_else(|| {
                    BundleError::Forward("decode-share dedicated forward bytes overflow".into())
                })?;
                if need > free as u64 {
                    return Err(BundleError::Forward(format!(
                        "decode-share: the {requested}-row chunk the placement was sized for needs {} MiB of dedicated forward resources plus headroom but only {} MiB are free; set HIPFIRE_QWEN4_DECODE_SHARE=off to load without sharing",
                        need >> 20,
                        free as u64 >> 20
                    )));
                }
                requested
            }
            None => qwen4_prefill_chunk_rungs(requested)
                .find(|&rows| {
                    qwen4_forward_device_bytes(&self.config, rows, mode)
                        .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES))
                        .is_some_and(|bytes| bytes <= free as u64)
                })
                .unwrap_or_else(|| requested.min(1536)),
        };
        eprintln!("  qwen4 prefill chunk: {max_chunk} rows (requested {requested})");
        let attempt = Qwen4GpuForward::new_inner(
            gpu,
            self,
            max_chunk,
            mode,
            #[cfg(target_os = "linux")]
            Some(npu_prepared),
        );
        let (mut forward, max_chunk) = match attempt {
            Ok(forward) => (forward, max_chunk),
            // The compact plan was chosen before the payload was proven; the
            // GPU symmetry check refused it and `new_inner` released all it
            // allocated. Re-plan Full against the memory now free. The NPU
            // trunk (gfx1151) never coexists with the gfx1201-only compact
            // plan, so nothing prepared is lost. S1 keeps its charged tile.
            Err(Qwen4GpuForwardError::CompactUnverified { verified, total }) if sharing => {
                return Err(BundleError::Forward(format!(
                    "decode-share: the compact scratch plan was refused by the GPU symmetry check ({verified}/{total} layers verified) and a Full plan cannot fit the donor tail; set HIPFIRE_QWEN4_DECODE_SHARE=off to load without sharing"
                )));
            }
            Err(Qwen4GpuForwardError::CompactUnverified { verified, total })
                if s1_tile.is_none() && mode == Qwen4ForwardScratchMode::FnSymGfx1201 =>
            {
                let (free, _) = gpu.device_mem_info().map_err(BundleError::Hip)?;
                let rows = full_fallback_rows(&self.config, requested, free as u64);
                eprintln!(
                    "  qwen4 forward scratch: compact plan refused by the GPU symmetry check ({verified}/{total} layers verified); re-planned Full at {rows} rows"
                );
                let forward = Qwen4GpuForward::new_inner(
                    gpu,
                    self,
                    rows,
                    Qwen4ForwardScratchMode::Full,
                    #[cfg(target_os = "linux")]
                    Some(None),
                )
                .map_err(|error| BundleError::Forward(error.to_string()))?;
                (forward, rows)
            }
            Err(error) => return Err(BundleError::Forward(error.to_string())),
        };
        if let Some(profile) = self.slowlink.profile {
            forward.set_slowlink_profile(profile);
        }
        // The selected S1 load binds EVERY admitted physical owner here, before
        // any product request can take the memory the placement charged: the
        // page ring, route region, HC arena, window scratch and boundary stages
        // (the native-MTP `pending` backup binds at `attach_mtp`). A mode or
        // native-kernel capability this load cannot run is a named decline (S1
        // stays unavailable and requests bypass with that reason, nothing was
        // allocated); an allocation failure is a named load failure, never a
        // mid-request fallback. Loads without a profile allocate nothing here.
        if self.slowlink.profile.is_some() {
            if let Err(reason) = crate::slowlink_prefill::bind_at_load(&mut forward, self, gpu) {
                let teardown = match forward.free_gpu(gpu) {
                    Ok(()) => "released to the device pool".to_string(),
                    Err(error) => format!("reported {error}"),
                };
                return Err(BundleError::Forward(format!(
                    "{reason}; forward teardown: {teardown}"
                )));
            }
        }
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
        self.ep_refuse("native MTP")?;
        Self::require_mtp_head(self.config.profile.id(), self.weights.mtp.is_some())?;
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
            // The forward is attached before the head: its EP binding (placement
            // and the rank's dummy expert) builds the head's EP decode table.
            self.execution
                .as_ref()
                .and_then(Qwen4GpuForward::ep_binding),
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?;
        self.mtp = Some(mtp);
        // The S1 entry `pending` backup is charged by the admission of a
        // native-MTP load: bind it with the head, never on a request.
        self.slowlink_bind_pending_save(gpu)?;
        Ok(())
    }

    /// Reserve the GDN capture ring the first native MTP window captures into,
    /// before any window or captured graph can grow it: the support and rows of
    /// [`crate::mtp_spec::native_mtp_capture`] for a `max_k`-draft window, ensured
    /// on the state ring (a no-op where the route captures no rows). The caller
    /// passes the resource (reserve) drafts; execution keeps its own `max_k`.
    /// Staged allocation failure leaves the bundle untouched.
    pub fn prepare_native_mtp_capture(
        &mut self,
        gpu: &mut Gpu,
        max_k: usize,
    ) -> Result<(), BundleError> {
        let Some((_, rows)) =
            crate::mtp_spec::native_mtp_capture(gpu, &self.config, self.state_format.gdn, max_k)
        else {
            return Ok(());
        };
        self.state
            .ensure_row_capture(gpu, rows, self.config.linear_conv_kernel_dim - 1)
            .map_err(BundleError::State)
    }

    /// Native MTP attaches only where the profile's weights carry an MTP head
    /// (`weights.mtp`); a head-free profile is refused with a typed error
    /// before any allocation.
    fn require_mtp_head(profile: &str, has_head: bool) -> Result<(), BundleError> {
        if has_head {
            Ok(())
        } else {
            Err(BundleError::NoMtpHead {
                profile: profile.to_string(),
            })
        }
    }

    pub(crate) fn ensure_spec_hidden(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<(), BundleError> {
        // The root-only spec hidden allocation is the output band's under PP
        // (the caller passes the output device); only EP has no such owner.
        if self.ep.is_some() {
            return Err(BundleError::Forward(
                "speculative decoding (spec hidden rows) is unsupported under qwen4 EP".to_string(),
            ));
        }
        if rows == 0 {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden capacity is zero".to_string(),
            ));
        }
        let rows = spec_hidden_capacity_rows(
            rows,
            self.spec_chunk_rows().unwrap_or(rows),
            self.npu_trunk_pipelined(),
        )
        .ok_or_else(|| BundleError::Forward("spec hidden capacity overflow".into()))?;
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
            .map(|forward| forward.prompt_chunk_rows())
            .or_else(|| self.pp_band.as_ref().map(|band| band.chunk()))
    }

    /// A whole-prompt entry begins (see [`Qwen4GpuForward::share_prompt_begin`]):
    /// drivers that issue several forward calls for one prompt bracket their
    /// loop so decode-share finishes the prompt once, after its last tile.
    /// No-op without decode-share or an attached forward.
    pub(crate) fn share_prompt_begin(&mut self) {
        if let Some(forward) = self.execution.as_mut() {
            forward.share_prompt_begin();
        }
    }

    /// The whole-prompt entry [`Self::share_prompt_begin`] opened ends (`ok`:
    /// it returned `Ok`); the outermost one finishes the prompt.
    pub(crate) fn share_prompt_end(&mut self, gpu: &mut Gpu, ok: bool) -> Result<(), BundleError> {
        match self.execution.as_mut() {
            Some(forward) => forward
                .share_prompt_end(gpu, ok)
                .map_err(|error| BundleError::Forward(error.to_string())),
            None => Ok(()),
        }
    }

    /// The scratch plan the attached forward actually allocated: differs from
    /// the plan `attach_forward` was given when a compact plan was refused by
    /// the GPU symmetry check and re-planned Full.
    pub fn forward_scratch_mode(&self) -> Option<Qwen4ForwardScratchMode> {
        self.execution.as_ref().map(|forward| forward.scratch.mode)
    }

    pub(crate) fn npu_trunk_pipelined(&self) -> bool {
        self.execution
            .as_ref()
            .is_some_and(|f| f.npu_trunk_pipelined())
    }

    pub(crate) fn npu_trunk_planned(&self) -> bool {
        self.execution
            .as_ref()
            .is_some_and(|f| f.npu_trunk_planned())
    }

    /// Whether the request in flight tripped the NPU trunk fault latch; clears the per-request flag.
    pub(crate) fn take_npu_trunk_fault(&mut self) -> bool {
        self.execution
            .as_mut()
            .is_some_and(|f| f.take_npu_trunk_fault())
    }

    pub(crate) fn begin_npu_trunk_request(&mut self, gpu: &Gpu) -> Result<(), BundleError> {
        if let Some(forward) = &mut self.execution {
            forward
                .begin_npu_trunk_request(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
        }
        Ok(())
    }

    /// Advance the NPU trunk executor before a driver asks for the next pair
    /// (`prefill_schedule_rows`, `npu_trunk_pair_shape`).
    pub(crate) fn poll_npu_trunk(&mut self, gpu: &Gpu) -> Result<(), BundleError> {
        if let Some(forward) = &mut self.execution {
            forward
                .poll_npu_trunk(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
        }
        Ok(())
    }

    /// The first still-pending snapshot boundary strictly after `position`.
    pub(crate) fn session_boundary_after(&self, position: usize) -> Option<usize> {
        self.session.as_ref()?.next_boundary_after(position)
    }

    /// The turn's armed lineage split row strictly after `position`. A prefill
    /// pair group ends there and the NPU request is planned through it: the
    /// pair executor carries one interior snapshot row, so a split inside a
    /// group that already holds one would drop the whole group to GPU tiles.
    pub(crate) fn session_lineage_after(&self, position: usize) -> Option<usize> {
        self.session
            .as_ref()?
            .turn_lineage()
            .filter(|&p| p > position)
    }

    /// Banks and snapshot cut of the planned pair starting at `position`, the
    /// geometry both the pair executor and the MTP head appends follow.
    pub(crate) fn npu_trunk_pair_shape(
        &self,
        position: usize,
        remaining: usize,
        chunk: usize,
    ) -> Option<crate::gpu_forward::PairShape> {
        let banks = self
            .execution
            .as_ref()?
            .npu_trunk_pair(position, remaining, chunk)?;
        Some(crate::gpu_forward::PairShape::new(
            banks,
            position,
            self.session_next_boundary(),
        ))
    }

    pub(crate) fn prefill_schedule_rows(&self, position: usize, end: usize, chunk: usize) -> usize {
        let end = self
            .session_lineage_after(position)
            .map_or(end, |p| p.min(end));
        let next = self.session_next_boundary();
        let boundaries = crate::gpu_forward::SessionBoundaries {
            next,
            after: next.and_then(|next| self.session_boundary_after(next)),
        };
        let planned = self
            .execution
            .as_ref()
            .is_some_and(|f| f.npu_trunk_pair(position, end - position, chunk).is_some());
        crate::gpu_forward::qwen4_prefill_segment_rows(position, end, chunk, boundaries, planned)
    }

    fn spec_prefill_capacity(&self, chunk: usize) -> usize {
        if self.npu_trunk_pipelined() {
            chunk * 2
        } else {
            chunk
        }
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
        // Host-only; every rank must plan session boundaries identically.
        let set = |bundle: &mut Self| {
            bundle.turn_end_token = token;
            Ok(())
        };
        if let Err(error) = self.ep_fanout("set_turn_end_token", set, &|bundle, _, _| {
            bundle.turn_end_token = token;
            Ok(())
        }) {
            eprintln!("  qwen4 EP: set_turn_end_token: {error}");
        }
    }

    /// Force message-end snapshots on or off regardless of the environment
    /// (hardware tests; production reads `HIPFIRE_QWEN4_TURN_SNAPSHOTS` once
    /// at assembly). Changing it changes the snapshot scope, so call it before
    /// the first prefill that fills the cache.
    pub fn set_turn_snapshots(&mut self, on: bool) {
        // Message-end snapshots stay refused under topology.pp (the aggregate
        // record pins cold chunk pieces only).
        if let Some(mesh) = self.pp.as_ref().filter(|_| on) {
            eprintln!("  {}", pp::turn_snapshots_refusal(mesh.stages()));
            return;
        }
        let set = |bundle: &mut Self| {
            bundle.turn_snapshots = on;
            Ok(())
        };
        if let Err(error) = self.ep_fanout("set_turn_snapshots", set, &|bundle, _, _| {
            bundle.turn_snapshots = on;
            Ok(())
        }) {
            eprintln!("  qwen4 EP: set_turn_snapshots: {error}");
        }
    }

    /// Override the tail-split thresholds read at assembly
    /// (`HIPFIRE_QWEN4_TAIL_SNAPSHOT_MIN_GAP` / `_MIN_PROMPT`); hardware tests
    /// use it to exercise the split on short prompts.
    pub fn set_tail_snapshot_limits(&mut self, min_gap: usize, min_prompt: usize) {
        let set = |bundle: &mut Self| {
            bundle.tail_snapshot_min_gap = min_gap;
            bundle.tail_snapshot_min_prompt = min_prompt;
            Ok(())
        };
        if let Err(error) = self.ep_fanout("set_tail_snapshot_limits", set, &|bundle, _, _| {
            bundle.tail_snapshot_min_gap = min_gap;
            bundle.tail_snapshot_min_prompt = min_prompt;
            Ok(())
        }) {
            eprintln!("  qwen4 EP: set_tail_snapshot_limits: {error}");
        }
    }

    /// Forward over `tokens`, returning every row's argmax pick (downloaded).
    pub(crate) fn spec_forward_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<Vec<u32>, BundleError> {
        self.spec_forward_rows_with_output(
            gpu,
            tokens,
            capture_hidden,
            Qwen4OutputRows::All,
            false,
            true,
        )
    }

    /// [`Self::spec_forward_rows`] for callers that consume the verify's
    /// resident logits on the device and read no argmax pick: identical
    /// forward, logits, hidden capture and state, minus the pick readback.
    pub(crate) fn spec_forward_rows_resident(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<(), BundleError> {
        self.spec_forward_rows_with_output(
            gpu,
            tokens,
            capture_hidden,
            Qwen4OutputRows::All,
            false,
            false,
        )
        .map(|_| ())
    }

    /// Final-row argmax of one forward over `tokens`: prompt fills and
    /// advances, which read no other row's logits.
    pub(crate) fn spec_prefill_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<u32, BundleError> {
        self.spec_forward_rows_with_output(
            gpu,
            tokens,
            capture_hidden,
            Qwen4OutputRows::Final,
            true,
            true,
        )?
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
        self.spec_prefill_rows_without_pick(gpu, tokens, true)
    }

    pub(crate) fn spec_prefill_rows_without_pick(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<(), BundleError> {
        self.ep_refuse("speculative decoding (spec prefill)")?;
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
            .prompt_chunk_rows();
        let capacity = self.spec_prefill_capacity(max_chunk);
        if tokens.len() > capacity {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block length {} exceeds capacity {capacity}",
                tokens.len()
            )));
        }
        // An armed DFlash2 drafter captures its target layers instead of the
        // MTP head's wide hidden.
        let layer_capture = if capture_hidden {
            self.dflash_layer_capture(tokens.len())?
        } else {
            None
        };
        let hidden = if capture_hidden && layer_capture.is_none() {
            let width = self
                .config
                .hc_count
                .checked_mul(self.config.hidden_size)
                .ok_or_else(|| BundleError::Forward("spec hidden width overflow".into()))?;
            let len = tokens
                .len()
                .checked_mul(width)
                .ok_or_else(|| BundleError::Forward("spec hidden row overflow".into()))?;
            Some(
                self.spec_hidden
                    .as_ref()
                    .ok_or_else(|| {
                        BundleError::Forward("Qwen4 spec hidden is not allocated".into())
                    })?
                    .sub_offset(0, len),
            )
        } else {
            None
        };
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
        forward.layer_capture = layer_capture;
        let result = forward
            .forward_chunk_silent(self, gpu, tokens, hidden.as_ref())
            .map_err(|error| BundleError::Forward(error.to_string()));
        forward.layer_capture = None;
        forward.prefill_scope = prior_scope;
        self.execution = Some(forward);
        result
    }

    /// One spec forward. `download_picks` reads the per-row argmax back; when
    /// `false` the argmax is still computed on the device (head and logits are
    /// unchanged) and the returned `Vec` is empty.
    fn spec_forward_rows_with_output(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
        output_rows: Qwen4OutputRows,
        prefill: bool,
        download_picks: bool,
    ) -> Result<Vec<u32>, BundleError> {
        self.ep_refuse("speculative decoding (spec verify forward)")?;
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 spec forward cannot process an empty block".to_string(),
            ));
        }
        if !prefill && self.has_cpu_deferred_experts() {
            let qualified = self
                .cpu_qualified_verify_rows(
                    gpu,
                    crate::gpu_forward::Qwen4GpuForward::cpu_lab_override_present(gpu),
                )
                .max(crate::program::QWEN4_CPU_SPLICE_MAX_ROWS);
            if tokens.len() > qualified {
                return Err(BundleError::Forward(format!(
                    "Qwen4 spec verify of {} rows exceeds the {}-row qualified CPU routed-expert window (legacy {})",
                    tokens.len(),
                    qualified,
                    crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
                )));
            }
        }
        let forward = self.execution.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        // A prompt block tiles at the prompt chunk; a verify window is bounded
        // by the bank it runs on (the resident bank on a decode-share load).
        let max_chunk = if prefill {
            forward.prompt_chunk_rows()
        } else {
            forward.scratch.max_chunk
        };
        let capacity = if prefill {
            self.spec_prefill_capacity(max_chunk)
        } else {
            max_chunk
        };
        if tokens.len() > capacity {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block length {} exceeds capacity {capacity}",
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
        let layer_capture = if capture_hidden {
            self.dflash_layer_capture(tokens.len())?
        } else {
            None
        };
        let hidden = if capture_hidden && layer_capture.is_none() {
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
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, prefill);
        forward.layer_capture = layer_capture;
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
        forward.layer_capture = None;
        forward.prefill_scope = prior_scope;
        self.execution = Some(forward);
        result?;
        #[cfg(feature = "lab")]
        self.lab_note_forward(gpu, tokens, &logits, output_count, prefill)?;
        if !download_picks {
            return Ok(Vec::new());
        }
        let bytes_len = output_count * std::mem::size_of::<i32>();
        if self.spec_host_top1.len() < bytes_len {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..bytes_len], &top1.buf)
            .map_err(BundleError::Hip)?;
        gpu.take_device_wait_error().map_err(BundleError::Hip)?;
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
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        self.copy_hidden_row(gpu, source, row, destination)
    }

    /// Copy row `row` (`hc_count * hidden_size` F32) of `source` into
    /// `destination`. `source` holds captured wide-hidden rows (the spec
    /// hidden, or a lane's rows of it).
    pub(crate) fn copy_hidden_row(
        &self,
        gpu: &mut Gpu,
        source: &GpuTensor,
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
        gpu.memcpy_dtod_at_ordered_async(
            &destination.buf,
            0,
            &source.buf,
            0,
            destination.byte_size(),
        )
        .map_err(BundleError::Hip)
    }

    /// Effective rows for an eager verify: admitted arithmetic (with the AR
    /// reference bound at entry), CPU-expert restriction, physical capacity,
    /// and the actually allocated forward/logit/argmax scratch. Before forward
    /// resources are attached this retains the historic conservative getter.
    /// The n-gram copy lane caps a window's take at this minus one.
    pub(crate) fn row_exact_verify_rows(&self, gpu: &Gpu) -> usize {
        let Some(forward) = self.execution.as_ref() else {
            return rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;
        };
        let logits_rows = self.spec_logits.as_ref().map_or(0, |tensor| {
            tensor
                .numel()
                .checked_div(self.config.vocab_size)
                .unwrap_or(0)
        });
        let top1_rows = self
            .spec_top1
            .as_ref()
            .map_or(0, |tensor| tensor.byte_size() / std::mem::size_of::<i32>());
        forward
            .guarded_wide_verify_rows(self, gpu)
            .min(logits_rows)
            .min(top1_rows)
    }

    /// The actual AR arithmetic reference latched by the most recent
    /// multi-row verify attempt, not the replay backend requested by flags.
    /// Singleton forwards do not bind a verify reference.
    #[cfg(feature = "lab")]
    pub(crate) fn last_verify_reference(
        &self,
    ) -> Option<rdna_compute::prefill_invariant::Qwen4VerifyReference> {
        self.execution
            .as_ref()
            .and_then(|forward| forward.last_verify_reference)
    }

    /// Checked device allocation plan for this loaded owner's capture ring,
    /// both halves and all retained inputs/output/table, not a measured peak.
    #[cfg(feature = "lab")]
    pub(crate) fn row_capture_planned_bytes(&self, rows: usize) -> Option<usize> {
        Qwen4State::row_capture_bytes_scoped(
            &self.config,
            self.state_format.gdn,
            rows,
            self.state.gdn.len(),
        )
    }

    /// Checked byte census of the admitted CPU-owned expert window (D1 + D2): the real host
    /// transfer-stage bytes, host-slot count, full-slot GPU view extents and the SUM of
    /// actual D2 executor instance bytes at the stored common capacity: the SourceB
    /// [`storage_bytes`]/[`transfer_bytes_for`] authorities plus the mapped cold-return
    /// and `masked_indices` device bytes per armed layer, with the thread-local host-slot
    /// scratch pool for the stored rows — every formula checked
    /// (the existing [`expert_residency`] + [`gpu_forward`] + SourceB authorities),
    /// `None` on overflow (the caller refuses before any allocation). No measured peak,
    /// no second planner: only the D1 stage row-count term is D1-specific (0 on a
    /// D2-only load reports the D2 shares and skips the pinned stage); input/output
    /// stages are the one D1 stage formula, GPU views are the allocated
    /// forward/provider extents, capture is the planned ring,
    /// [`Self::cpu_d1_requested_window_rows`] is the plan need used pre-reserve.
    /// `0`-storage loads report zeros (legacy default: nothing reserved).
    pub fn cpu_storage_bytes(&self) -> Option<CpuStorageBytes> {
        let rows = self.cpu_storage_capacity_rows();
        if rows == 0 {
            return Some(CpuStorageBytes::default());
        }
        let Some((dim, mi, top_k)) = self.cpu_d1_geometry() else {
            return None;
        };
        // D1 owns the pinned transfer stage only when a D1 splice is armed; a D2-only
        // load reports the shared scratch + D2 bytes with a 0 stage (nothing pinned).
        let d1_stage_values = if self.has_cpu_deferred_experts() {
            crate::expert_residency::cpu_d1_stage_values(dim, top_k, rows)?
        } else {
            0
        };
        let slots = crate::expert_residency::cpu_d1_host_slots(rows, top_k)?;
        let stage_bytes = d1_stage_values.checked_mul(4)?;
        let forward = self.execution.as_ref()?;
        let need = |have: usize, need: usize| (have >= need).then_some(need);
        let x_rot = need(forward.scratch.ffn_output.numel(), rows.checked_mul(dim)?)?;
        let topk = need(forward.scratch.moe_topk_indices.numel(), slots)?;
        let _ = need(forward.scratch.moe_topk_weights.numel(), topk)?;
        let expanded = need(
            forward.scratch.moe_down_expanded.numel(),
            slots.checked_mul(dim)?,
        )?;
        let spec = forward.forward_device_bytes(&self.config)?;
        let capture = self.row_capture_planned_bytes_lab(rows)?;
        let (d2_input, d2_mask, d2_output, d2_transfer, d2_mapped, d2_device) =
            self.cpu_d2_storage_bytes().unwrap_or((0, 0, 0, 0, 0, 0));
        // The thread-local host-slot scratch the settled reservation actually grew:
        // the live pool capacity translated back through SourceB's reserve formulas is
        // thread-local and mutable state, so the settled census charges the SAME checked
        // requested-size pool the preflight plan computed, never a live read.
        let host_scratch_bytes = {
            let count_f32 = |n: usize| n.checked_mul(4);
            let count_u32 = |n: usize| n.checked_mul(4);
            let slots2mi = slots.checked_mul(mi)?;
            count_f32(slots2mi.checked_mul(2)?)?
                .checked_add(count_f32(slots2mi)?)?
                .checked_add(count_f32(slots.checked_mul(dim)?)?)?
                .checked_add(count_f32((2usize).checked_mul(mi)?)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots.checked_add(1)?)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots.checked_add(1)?)?.checked_mul(2)?)?
                .checked_add(
                    std::mem::size_of::<usize>()
                        .checked_mul(slots.checked_add(1)?)?
                        .checked_mul(2)?,
                )?
        };
        Some(CpuStorageBytes {
            rows,
            slots,
            stage_values: d1_stage_values,
            stage_bytes,
            x_rot_elements: x_rot,
            topk_elements: topk,
            down_expanded_elements: expanded,
            forward_device_bytes: spec,
            capture_bytes: capture,
            d2_input_bytes: d2_input,
            d2_mask_bytes: d2_mask,
            d2_output_bytes: d2_output,
            d2_transfer_bytes: d2_transfer,
            d2_mapped_bytes: d2_mapped,
            d2_device_bytes: d2_device,
            host_scratch_bytes,
        })
    }

    /// Actual D2 executor storage bytes summed over live instances: per armed layer the
    /// SourceB [`storage_bytes`] triple for its allocated capacity rows plus the
    /// [`transfer_bytes_for`] up+down for this load's cold slots (all-cold upper bound
    /// `slots`), the mapped cold-return bytes (0 on the copy transport) and the
    /// `masked_indices` device bytes — each checked (overflow ⇒ `None` ⇒ caller refuses
    /// before allocation). `None` when no layer owns an executor (D2 contributes nothing)
    /// or on overflow; `(0,0,0,0,0,0)` never returned for an armed layer (capacity rows
    /// are always ≥ 1).
    pub(crate) fn cpu_d2_storage_bytes(
        &self,
    ) -> Option<(usize, usize, usize, usize, usize, usize)> {
        use hipfire_dispatch::cpu_split::CpuSplitExecutor;
        let forward = self.execution.as_ref()?;
        let mut total = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut armed = false;
        for layer in forward.moe.iter().flatten() {
            let Some(executor) = layer.cpu_split_storage_executor() else {
                continue;
            };
            armed = true;
            let cap = executor.capacity_rows();
            let (input, mask, output) =
                CpuSplitExecutor::storage_bytes(cap, executor.dim(), executor.top_k());
            if input == usize::MAX || mask == usize::MAX || output == usize::MAX {
                return None;
            }
            let Some((_, _, top_k)) = self.cpu_d1_geometry() else {
                return None;
            };
            let slots = cap.checked_mul(top_k)?;
            let mapped =
                executor.return_mode() == hipfire_dispatch::cpu_split::D2ReturnMode::Mapped;
            let (up, down) =
                CpuSplitExecutor::transfer_bytes_for(executor.dim(), top_k, mapped, cap, slots);
            if up == usize::MAX || down == usize::MAX {
                return None;
            }
            let mapped_bytes = if mapped {
                executor.dim().checked_mul(slots)?.checked_mul(4)?
            } else {
                0
            };
            total.0 = total.0.checked_add(input)?;
            total.1 = total.1.checked_add(mask)?;
            total.2 = total.2.checked_add(output)?;
            total.3 = total.3.checked_add(up)?.checked_add(down)?;
            total.4 = total.4.checked_add(mapped_bytes)?;
            total.5 = total.5.checked_add(mask)?;
        }
        armed.then_some(total)
    }

    /// Checked REQUESTED storage plan for the loader's load-time CPU preflight: the
    /// sum of actual armed-D2 per-layer requested bytes at `requested_rows` (replacement
    /// executors) + the D1 owned stage for `d1_rows` (0 = D2-only, skip the pinned stage)
    /// + the called-thread host-slot scratch real reservation bytes for `requested_rows`
    /// + reconciled-not-charged GPU views (full-slot views must fit the allocated
    /// forward scratch) and the planned capture ring. `None` on any overflow, on missing
    /// geometry/executors, or when the settled GPU scratch cannot hold the request.
    /// `0`-request loads report zeros (legacy default: nothing reserved). Called BEFORE
    /// any allocation; the sibling [`Self::cpu_storage_bytes`] census reads the ACTUAL
    /// settled instances so the loader reconciles plan with post-reserve fact. No
    /// allocation, no mutation, no Vec: read through the existing authorities. The caller
    /// owns the host/device charge of these bytes and the sequential-replacement
    /// discipline (old freed before new is built), so no old+new peak is claimed here.
    pub fn cpu_requested_storage_bytes(
        &self,
        requested_rows: usize,
        d1_rows: usize,
    ) -> Option<CpuStorageBytes> {
        if requested_rows > hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS {
            return None;
        }
        if requested_rows == 0 {
            return Some(CpuStorageBytes::default());
        }
        if d1_rows > requested_rows {
            return None;
        }
        let Some((dim, mi, top_k)) = self.cpu_d1_geometry() else {
            return None;
        };
        let forward = self.execution.as_ref()?;
        // D1 owned stage: real f32 values + bytes for the D1-owned rows only (0 when
        // D2-only; the TLS host-slot scratch below is still needed by D2). `slots` is
        // the full-request host-slot count, so the plan's view terms match the full-slot
        // extents the settled census reports.
        let d1_stage_values = if d1_rows > 0 {
            crate::expert_residency::cpu_d1_stage_values(dim, top_k, d1_rows)?
        } else {
            0
        };
        let d1_stage_bytes = d1_stage_values.checked_mul(4)?;
        let slots = requested_rows.checked_mul(top_k)?;
        // The thread-local host-slot scratch the loader reserves for the request: the
        // actual `MOE_SCRATCH` vector capacities from SourceB's reserve formulas (gu,
        // hidden, dn, gather, order/group_id/slot_group/slot_si, group_start/group_off),
        // checked so any overflow refuses before allocation. D1 and D2 share the one
        // TLS pool (`reserve_moe_cpu_host_slot_scratch(dim, mi, max_slots)`); the plan
        // charges the request's full pool once, never per mode and never per layer.
        let host_scratch_bytes = {
            let count_f32 = |n: usize| n.checked_mul(4);
            let count_u32 = |n: usize| n.checked_mul(4);
            let slots2mi = slots.checked_mul(mi)?;
            count_f32(slots2mi.checked_mul(2)?)?
                .checked_add(count_f32(slots2mi)?)?
                .checked_add(count_f32(slots.checked_mul(dim)?)?)?
                .checked_add(count_f32((2usize).checked_mul(mi)?)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots.checked_add(1)?)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots)?)?
                .checked_add(count_u32(slots.checked_add(1)?)?.checked_mul(2)?)?
                .checked_add(
                    std::mem::size_of::<usize>()
                        .checked_mul(slots.checked_add(1)?)?
                        .checked_mul(2)?,
                )?
        };
        // Requested D2 bytes: the same SourceB authorities the settled census reads,
        // evaluated at the REQUESTED rows (the replacement executors' capacity) with the
        // CURRENT per-layer mapped/copy selection. Old+new overlap is not charged: the
        // replacement discipline frees the old executor before building the new one.
        use hipfire_dispatch::cpu_split::CpuSplitExecutor;
        let mut d2_input = 0usize;
        let mut d2_mask = 0usize;
        let mut d2_output = 0usize;
        let mut d2_transfer = 0usize;
        let mut d2_mapped = 0usize;
        let mut d2_device = 0usize;
        for layer in forward.moe.iter().flatten() {
            let Some(executor) = layer.cpu_split_storage_executor() else {
                continue;
            };
            let (input, mask, output) =
                CpuSplitExecutor::storage_bytes(requested_rows, executor.dim(), executor.top_k());
            if input == usize::MAX || mask == usize::MAX || output == usize::MAX {
                return None;
            }
            let mapped =
                executor.return_mode() == hipfire_dispatch::cpu_split::D2ReturnMode::Mapped;
            let (up, down) = CpuSplitExecutor::transfer_bytes_for(
                executor.dim(),
                top_k,
                mapped,
                requested_rows,
                slots,
            );
            if up == usize::MAX || down == usize::MAX {
                return None;
            }
            if mapped {
                d2_mapped =
                    d2_mapped.checked_add(executor.dim().checked_mul(slots)?.checked_mul(4)?)?;
            }
            d2_device = d2_device.checked_add(mask)?;
            d2_input = d2_input.checked_add(input)?;
            d2_mask = d2_mask.checked_add(mask)?;
            d2_output = d2_output.checked_add(output)?;
            d2_transfer = d2_transfer.checked_add(up)?.checked_add(down)?;
        }
        let need = |have: usize, need: usize| (have >= need).then_some(need);
        let x_rot_elements = need(
            forward.scratch.ffn_output.numel(),
            requested_rows.checked_mul(dim)?,
        )?;
        let topk_elements = need(forward.scratch.moe_topk_indices.numel(), slots)?;
        let _ = need(forward.scratch.moe_topk_weights.numel(), topk_elements)?;
        let down_expanded_elements = need(
            forward.scratch.moe_down_expanded.numel(),
            slots.checked_mul(dim)?,
        )?;
        let forward_device_bytes = forward.forward_device_bytes(&self.config)?;
        let capture_bytes = self.row_capture_planned_bytes_lab(requested_rows)?;
        Some(CpuStorageBytes {
            rows: requested_rows,
            slots,
            stage_values: d1_stage_values,
            stage_bytes: d1_stage_bytes,
            x_rot_elements,
            topk_elements,
            down_expanded_elements,
            forward_device_bytes,
            capture_bytes,
            d2_input_bytes: d2_input,
            d2_mask_bytes: d2_mask,
            d2_output_bytes: d2_output,
            d2_transfer_bytes: d2_transfer,
            d2_mapped_bytes: d2_mapped,
            d2_device_bytes: d2_device,
            host_scratch_bytes,
        })
    }

    /// [`Self::row_capture_planned_bytes`] without the lab gate: the loader's
    /// production prewarm path reads the same checked plan the lab oracle does.
    pub(crate) fn row_capture_planned_bytes_lab(&self, rows: usize) -> Option<usize> {
        Qwen4State::row_capture_bytes_scoped(
            &self.config,
            self.state_format.gdn,
            rows,
            self.state.gdn.len(),
        )
    }

    /// Keep the first `keep` rows of the armed `tokens.len()`-row verify the
    /// active snapshot ticket brackets, replaying only the captured recurrence
    /// (not projections); the ticket stays active for commit or restore.
    pub(crate) fn rollback_verify_rows_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        keep: usize,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let ple_normed = {
            let tensor = self
                .execution
                .as_ref()
                .ok_or_else(|| {
                    BundleError::Forward("Qwen4 forward resources are not attached".to_string())
                })?
                .scratch
                .ple_normed_for_rollback()
                .map_err(|error| BundleError::Forward(error.to_string()))?;
            tensor.sub_offset(0, tensor.numel())
        };
        self.rollback_verify_rows_retain_with(gpu, snapshot, keep, tokens, &ple_normed)
    }

    /// [`Self::rollback_verify_rows_retain`] with the verify's PLE
    /// convolution input rows (`ple_normed`, from the block's row 0) given.
    pub(crate) fn rollback_verify_rows_retain_with(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        keep: usize,
        tokens: &[u32],
        ple_normed: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.ep_refuse("speculative decoding (verify rollback)")?;
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

    /// The forward scratch's PLE convolution input rows from row `row`
    /// through `row + rows` (`hc_count * hidden_size` F32 per row); a view
    /// running to the end of the scratch tensor, valid until the next forward.
    pub(crate) fn lane_ple_normed_rows(
        &self,
        row: usize,
        rows: usize,
    ) -> Result<GpuTensor, BundleError> {
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("PLE row width overflow".to_string()))?;
        let tensor = self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .ple_normed_for_rollback()
            .map_err(|error| BundleError::Forward(error.to_string()))?;
        let offset = row
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("PLE row offset overflow".to_string()))?;
        let end = row
            .checked_add(rows)
            .and_then(|end| end.checked_mul(width))
            .ok_or_else(|| BundleError::Forward("PLE row extent overflow".to_string()))?;
        if end > tensor.numel() {
            return Err(BundleError::Forward(
                "Qwen4 PLE rows are outside the forward scratch".to_string(),
            ));
        }
        Ok(tensor.sub_offset(offset, tensor.numel() - offset))
    }

    /// Start reading the PLE rows `tokens` (the next tokens after the
    /// committed history, in order) will need, so a forward over them later
    /// finds them cached. Best effort: a failure only loses the head start.
    pub(crate) fn warm_ple_rows(&self, tokens: &[u32]) {
        if let Some(ple) = &self.ple {
            let ids = self.state.ple_history.row_ids(&ple.metadata, tokens);
            let _ = ple.rows.warm(ids);
        }
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

    /// [`Self::spec_capture_token`] without the argmax pick readback.
    pub(crate) fn spec_capture_token_resident(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<(), BundleError> {
        self.spec_forward_rows_resident(gpu, std::slice::from_ref(&token), true)
    }

    /// One ordinary target-only forward of `token`: a single final row, no
    /// wide-hidden capture, so the replay graph route stays eligible. Returns
    /// the greedy argmax (read into the first 4 bytes of `spec_host_top1`, no
    /// allocation); the row's logits stay in row 0 of `spec_logits` for a
    /// sampled follow-up. Clears any stale PLE lookahead first.
    pub(crate) fn spec_ar_token(&mut self, gpu: &mut Gpu, token: u32) -> Result<u32, BundleError> {
        self.ep_refuse("speculative decoding (spec AR token)")?;
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
        #[cfg(feature = "lab")]
        self.lab_note_forward(gpu, std::slice::from_ref(&token), &logits, 1, false)?;
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..top1_bytes], &top1.buf)
            .map_err(BundleError::Hip)?;
        gpu.take_device_wait_error().map_err(BundleError::Hip)?;
        let bytes = &self.spec_host_top1[..top1_bytes];
        Ok(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Copy the wide HC stream row the last [`Self::spec_ar_token`] (or any
    /// ordinary one-token forward) left in the forward scratch into
    /// `destination` (F32, exactly `hc_count * hidden_size` elements). Valid
    /// only until the next forward. The AR floor's calibration token and
    /// every retired AR token take it (one copy or BF16 widen per token).
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

    /// [`Self::copy_ar_hidden_to`] into row `row` of the spec hidden capture,
    /// the row a later batched or per-row head append reads.
    pub(crate) fn copy_ar_hidden_to_spec_row(
        &self,
        gpu: &mut Gpu,
        row: usize,
    ) -> Result<(), BundleError> {
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
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
        self.copy_ar_hidden_to(gpu, &source.sub_offset(offset, width))
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
            self.execution
                .as_ref()
                .and_then(Qwen4GpuForward::mtp_ep_gather),
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?
        .ok_or_else(|| {
            BundleError::Forward("MTP prediction requested but no token produced".into())
        })
    }

    /// [`Self::mtp_forward_token`] with the draft read deferred: the draft's
    /// device work lands in `top1_out` (Raw, at least `Gpu::TOPK8_OUT_BYTES`)
    /// and [`Self::mtp_draft_finish`] parses it once downloaded.
    pub(crate) fn mtp_forward_token_deferred(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
        top1_out: &GpuTensor,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, position)?
            .forward_token_deferred(
                gpu,
                &self.weights,
                &self.config,
                token,
                backbone_hidden,
                position,
                fresh_qsa_selection,
                top1_out,
                self.execution
                    .as_ref()
                    .and_then(Qwen4GpuForward::mtp_ep_gather),
            )
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Parse a downloaded deferred draft (`top1_out`'s first
    /// `Gpu::TOPK8_OUT_BYTES` bytes): the draft token, with the head's
    /// margin and candidates updated as `draft` does.
    pub(crate) fn mtp_draft_finish(&mut self, top: &[u8]) -> Result<u32, BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .draft
            .draft_finish(top)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Arm (or disarm and clear) the lab taps: the speculative-forward logits
    /// tap (`keep_logits` also keeps the rows) and the state's residual/ring
    /// event tap. Arm before a restore to observe its pre-window events.
    #[cfg(feature = "lab")]
    pub fn lab_observe(&mut self, armed: bool, keep_logits: bool) {
        self.lab_logits.set_armed(armed, keep_logits);
        self.state.lab_observe_state(armed);
    }

    /// The lab taps are armed.
    #[cfg(feature = "lab")]
    pub fn lab_observed(&self) -> bool {
        self.lab_logits.armed()
    }

    /// Drain the speculative forwards' logits observations recorded since
    /// the last take.
    #[cfg(feature = "lab")]
    pub fn lab_take_forward_logits(&mut self) -> Vec<crate::lab_window::LabForwardLogits> {
        self.lab_logits.take()
    }

    /// Record one speculative forward's output rows while the lab tap is
    /// armed: a blocking read of the resident `logits` (`rows` rows). Does
    /// nothing, and reads nothing back, unarmed.
    #[cfg(feature = "lab")]
    fn lab_note_forward(
        &mut self,
        gpu: &Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        rows: usize,
        prefill: bool,
    ) -> Result<(), BundleError> {
        if !self.lab_logits.armed() {
            return Ok(());
        }
        let mut host = Vec::new();
        gpu.download_f32_into(logits, &mut host)
            .map_err(BundleError::Hip)?;
        self.lab_logits
            .record(tokens, prefill, rows, self.config.vocab_size, &host);
        Ok(())
    }

    /// Every byte family of the resident MTP head state
    /// ([`crate::mtp_gpu::lab_head_items`]) followed by its draft request
    /// state (`full_steps` u32 LE, `margin` f32 LE).
    #[cfg(feature = "lab")]
    pub fn lab_resident_head_items(&self, gpu: &Gpu) -> Result<Vec<(String, Vec<u8>)>, String> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP resources are not attached".to_string())?;
        let mut items = crate::mtp_gpu::lab_head_items(gpu, &mtp.state)?;
        let request = mtp.draft.request_state();
        let mut bytes = request.full_steps.to_le_bytes().to_vec();
        bytes.extend_from_slice(&request.margin.to_le_bytes());
        items.push(("draft_request_state".to_string(), bytes));
        Ok(items)
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
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))
    }

    pub(crate) fn set_mtp_draft_request_state(
        &mut self,
        state: hipfire_dispatch::pipeline::DraftHeadRequestState,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .map(|mtp| mtp.draft.set_request_state(state))
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_apply_penalty_table(gpu, logits, table);
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

    /// Resident (no download) contiguous view of rows
    /// `first_row..first_row + rows` of the last speculative forward's logits,
    /// `rows * vocab` f32 words with row stride `vocab`. A non-owning view:
    /// valid only until the next speculative forward, never freed.
    pub(crate) fn spec_logits_rows(
        &self,
        first_row: usize,
        rows: usize,
    ) -> Result<GpuTensor, BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self.spec_logits.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec logits are not attached".to_string())
        })?;
        first_row
            .checked_add(rows)
            .and_then(|end| end.checked_mul(vocab))
            .filter(|&end| rows > 0 && end <= logits.numel())
            .map(|_| logits.sub_offset(first_row * vocab, rows * vocab))
            .ok_or_else(|| {
                BundleError::Forward(format!(
                    "Qwen4 spec logit rows {first_row}..{} are outside capacity",
                    first_row.saturating_add(rows)
                ))
            })
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
                self.execution
                    .as_ref()
                    .and_then(Qwen4GpuForward::mtp_ep_gather),
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

    /// Save the target boundary hidden into the head's snapshot state: call
    /// right before the session capture at a prompt boundary, after the chunk's
    /// append left `pending` holding the target hidden of the boundary's last
    /// row (position `end - 1`). Reads `pending` only. A capture at a position
    /// this was not called for stores no usable row (see
    /// [`Self::session_restore_boundary_hidden`]).
    pub(crate) fn session_save_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .save_boundary_hidden(gpu, pending)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// After `session_begin`: when the head was just restored from a snapshot
    /// (restores refuse snapshots without a boundary row for their position),
    /// copy that row into `pending` and return `true`. `false` means no
    /// restore happened since (a live continuation, an AR route, or a repeat
    /// call): `pending` is untouched and already the live hidden.
    pub(crate) fn session_restore_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<bool, BundleError> {
        match self.mtp.as_mut() {
            Some(mtp) => mtp
                .restore_boundary_hidden(gpu, pending)
                .map_err(|error| BundleError::Forward(error.to_string())),
            None => Ok(false),
        }
    }

    /// Batched prompt fill: [`MtpStep::Append`] for `tokens` at
    /// `position..position + tokens.len()`, row `i` paired with captured spec
    /// hidden row `hidden_row0 + i`. The caller supplies the rows: the token
    /// at position `p` pairs with the target hidden of position `p - 1`
    /// (prompt position 0 pairs with its own hidden), the same rows as
    /// calling [`Self::mtp_append_token`] per row after `copy_spec_hidden_row_to`.
    ///
    /// `trunk_ids_row` is `Some(r)` when `tokens` are rows `r..` of the spec
    /// forward that just ran, with nothing run since: the head then embeds
    /// from that forward's device token ids when they hold exactly `tokens`.
    /// Otherwise it embeds from `prompt_ids`, device ids that already hold
    /// exactly `tokens` as i32 (the caller's once-per-chunk upload, see
    /// `MtpAppendScratch::upload_prompt_ids`), and uploads its own copy when
    /// that is `None` too.
    ///
    /// `context` names the caller and is passed through unchanged: only
    /// [`MtpFillContext::PromptFill`] can take the DET projections.
    pub(crate) fn mtp_append_rows(
        &mut self,
        gpu: &mut Gpu,
        context: MtpFillContext,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden_row0: usize,
        trunk_ids_row: Option<usize>,
        prompt_ids: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), BundleError> {
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
        self.mtp_append_rows_hidden(
            gpu,
            context,
            scratch,
            tokens,
            &hidden,
            trunk_ids_row,
            prompt_ids,
            position,
        )
    }

    /// [`Self::mtp_append_rows`] over an explicit F32 `hidden` tensor instead
    /// of captured spec rows: row `i` of `tokens` pairs with `hidden` row `i`
    /// (`hc_count * hidden_size` elements each; at least `tokens.len()` rows).
    /// This is how a prompt chunk's first row takes the carried `pending`
    /// hidden (position `p - 1`, from the previous chunk) through the same
    /// DET kernels as the rest of the chunk. `trunk_ids_row`, `prompt_ids` and
    /// `context` behave as in [`Self::mtp_append_rows`].
    pub(crate) fn mtp_append_rows_hidden(
        &mut self,
        gpu: &mut Gpu,
        context: MtpFillContext,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden: &GpuTensor,
        trunk_ids_row: Option<usize>,
        prompt_ids: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 MTP batched append has no rows".to_string(),
            ));
        }
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
                context,
                scratch,
                tokens,
                hidden,
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
            self.execution
                .as_ref()
                .and_then(Qwen4GpuForward::mtp_ep_gather),
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_forward_token(gpu, token, logits, top1);
        }
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_forward_token_or_argmax(gpu, token, logits);
        }
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_forward_chunk(gpu, tokens, logits, top1, false);
        }
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
        let result = forward
            .forward_chunk(self, gpu, tokens, logits, top1, None, Qwen4OutputRows::All)
            .map_err(|error| BundleError::Forward(error.to_string()));
        forward.prefill_scope = prior_scope;
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_forward_chunk(gpu, tokens, logits, top1, true);
        }
        // The one selector verdict of this request, before any mutation: a
        // qualified request runs layer-major, every other one the incumbent
        // chunk-major forward below.
        let entry = self.state.position;
        let output = crate::slowlink_prefill::PromptOutput::Final { logits, top1 };
        if self.slowlink_dispatch(gpu, tokens, output, SlowlinkScope::direct())? {
            return Ok(());
        }
        let result = self.forward_chunk_final_incumbent(gpu, tokens, logits, top1);
        if result.is_ok() {
            self.slowlink_record_incumbent(entry, &[(0, tokens.len())]);
        }
        result
    }

    /// The incumbent chunk-major body of [`Self::forward_chunk_final`]: no
    /// selector, no receipt (the prompt loop calls it once per chunk).
    fn forward_chunk_final_incumbent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
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
        forward.prefill_scope = prior_scope;
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        self.ep_refuse("silent chunk forward outside a replicated call")?;
        let entry = self.state.position;
        let output = crate::slowlink_prefill::PromptOutput::Silent;
        if self.slowlink_dispatch(gpu, tokens, output, SlowlinkScope::direct())? {
            return Ok(());
        }
        let result = self.forward_chunk_silent_incumbent(gpu, tokens);
        if result.is_ok() {
            self.slowlink_record_incumbent(entry, &[(0, tokens.len())]);
        }
        result
    }

    /// The incumbent chunk-major body of [`Self::forward_chunk_silent`].
    fn forward_chunk_silent_incumbent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
        let result = forward
            .forward_chunk_silent(self, gpu, tokens, None)
            .map_err(|error| BundleError::Forward(error.to_string()));
        forward.prefill_scope = prior_scope;
        self.execution = Some(forward);
        result
    }

    /// Name the tokens the next prefill forward is followed by, so it warms
    /// their PLE rows while its own chunk runs. Best effort; bytes unchanged.
    pub(crate) fn set_ple_lookahead(&mut self, tokens: &[u32]) {
        if self.ple.is_none() {
            return;
        }
        if let Some(forward) = self.execution.as_mut() {
            forward.set_ple_lookahead(tokens);
        }
    }

    fn invalidate_ple_epoch(&self) -> Result<(), BundleError> {
        let Some(ple) = &self.ple else {
            return Ok(());
        };
        ple.rows
            .reset_epoch(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)
    }

    /// Quiesce request-local PLE work without advancing the current epoch.
    /// Snapshot and commit preserve that epoch; reset and restore call
    /// `invalidate_ple_epoch` instead so work from the discarded state cannot
    /// publish after the state transition.
    fn quiesce_ple(&self) -> Result<(), BundleError> {
        let Some(ple) = &self.ple else {
            return Ok(());
        };
        ple.rows
            .quiesce(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)?;
        ple.rows.resume().map_err(BundleError::PleRows)
    }

    /// Reset every owner for a cold prefill.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_reset(gpu);
        }
        self.clear_image_context(gpu)?;
        // A new request starts with a clear router fault word. Only here, never
        // mid-pipeline: a clear would drop an earlier row's pending report.
        gpu.clear_router_fault();
        self.begin_npu_trunk_request(gpu)?;
        if let Some(stage) = &mut self.pair_session_stage {
            stage.position = None;
        }
        self.slowlink.clear_active();
        // `None` while the cache itself resets (it drives `self` then).
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
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
            .filter(|qsa| {
                rdna_compute::tensor_ops::qsa_gather_workspace_needed(
                    gpu,
                    qsa.format,
                    qsa.f16_arena.is_some(),
                )
            })
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

    /// Per-request decode map-ahead: map the target QSA context (and the
    /// attached MTP head's, when the request drafts) through
    /// `min(position + remaining_new_tokens + max_draft_rows, admitted
    /// capacity)` once, after prefill and before the first decode, so
    /// per-token growth checks never remap and no growth reaches a decode
    /// capture or record. Uses the forward's own growth path, with session
    /// snapshots yielding first. `max_draft_rows` is the most drafts any
    /// scheduled verify window carries (its rows are the seed plus those
    /// drafts, so it ends that many positions past the seed): the drafter's
    /// `proposal_capacity`, which is the native `max_k` or, with the n-gram
    /// pool armed, the wider `n_max` (a window `k > max_k` is a copy
    /// takeover verify; `mtp_step` refuses `k > proposal_capacity`). It must
    /// not be the configured MTP `k`. Mapping only grows capacity: no
    /// position, length or watermark commits, and the forward still maps its
    /// actual end on demand. `max_draft_rows == 0` (AR) leaves the MTP head
    /// unmapped. A failure is logged and swallowed: growth then happens on
    /// demand as before. Expert-parallel bundles and a zero
    /// `remaining_new_tokens` map nothing.
    pub fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        if self.ep.is_some() || self.pp.is_some() {
            return;
        }
        let Some(target) = decode_map_ahead_tokens(
            position,
            Some(remaining_new_tokens),
            max_draft_rows,
            self.state.max_seq_len,
        ) else {
            return;
        };
        self.release_session_for_growth(gpu, target);
        let mapped = self
            .state
            .ensure_mapped_capacity(gpu, target)
            .map_err(|error| error.to_string())
            .and_then(|()| match self.mtp.as_mut() {
                Some(mtp) if max_draft_rows != 0 => mtp
                    .ensure_mapped_capacity(gpu, target.min(mtp.context_capacity()))
                    .map_err(|error| error.to_string()),
                _ => Ok(()),
            });
        match mapped {
            Ok(()) => eprintln!(
                "  qwen4 QSA context mapped ahead for {target} tokens (decode budget {remaining_new_tokens})"
            ),
            Err(error) => eprintln!(
                "  qwen4 QSA map-ahead through {target} tokens failed ({error}); growing on demand"
            ),
        }
    }

    /// Attach the PLE hot-row store and its worker thread: the row store reads
    /// the rows its pack holds from the pack.  A geometry mismatch is logged
    /// and the store is dropped (stopped and joined).  The hot store, the
    /// prewarm and the rank table are Flash-Next-only: a profile without PLE
    /// or a developer profile refuses (and drops the worker).
    pub fn attach_ple_hot_store(&mut self, hot: PleHotWorker) -> Result<(), BundleError> {
        let ple = self.ple_attachment_owner("hot-row store")?;
        if let Err(error) = ple.rows.attach_hot_store(hot.store().clone()) {
            eprintln!("  qwen4 ple hot store: not attached to the row store ({error})");
            return Ok(());
        }
        ple.hot = Some(hot);
        Ok(())
    }

    /// The attached PLE hot-row store, if any.
    pub fn ple_hot_store(&self) -> Option<&std::sync::Arc<HotRowStore>> {
        self.ple.as_ref()?.hot.as_ref().map(PleHotWorker::store)
    }

    /// Attach the page-cache prewarm of a Q4LM PLE sidecar (see
    /// `hipfire_runtime::ple_prewarm`).  The bundle owns it from here:
    /// `free_gpu` stops and joins it before the row store unloads.
    pub fn attach_ple_prewarm(
        &mut self,
        prewarm: hipfire_runtime::ple_prewarm::PleRowsPrewarm,
    ) -> Result<(), BundleError> {
        self.ple_attachment_owner("sidecar prewarm")?.prewarm = Some(prewarm);
        Ok(())
    }

    /// The attached PLE sidecar prewarm, if any.
    pub fn ple_prewarm(&self) -> Option<&hipfire_runtime::ple_prewarm::PleRowsPrewarm> {
        self.ple.as_ref()?.prewarm.as_ref()
    }

    /// Keep the prewarm worker of a rank-order PLE table
    /// (`HotRowStore::spawn_table_worker`, wrapped with
    /// [`PleHotWorker::new`]) with the bundle.  Unlike
    /// [`Self::attach_ple_hot_store`] nothing is attached to the row store:
    /// it was built over this very store at assembly.  `free_gpu` stops and
    /// joins the worker before the row store unloads.
    pub fn attach_ple_table_worker(&mut self, worker: PleHotWorker) -> Result<(), BundleError> {
        self.ple_attachment_owner("rank-table worker")?.hot = Some(worker);
        Ok(())
    }

    /// The PLE owner an attachment (hot store, prewarm, rank-table worker)
    /// hangs from, or the refusal for a bundle that cannot carry one.
    fn ple_attachment_owner(&mut self, what: &str) -> Result<&mut Qwen4BundlePle, BundleError> {
        if !self.config.profile.is_flash_next() {
            return Err(BundleError::Config(format!(
                "the PLE {what} is Flash-Next-only; profile {} reads its embedded PLE rows directly",
                self.config.profile.id()
            )));
        }
        self.ple.as_mut().ok_or_else(no_ple_rows)
    }

    /// Attach the session cache (`hipfire_runtime::session_cache`): prefill
    /// then restores and captures whole-chunk snapshots through it.
    pub fn attach_session_cache(&mut self, mut cache: SessionCache) {
        if let Some(policy) = session_pair::pressure_host_snapshot_policy(
            self.slowlink
                .profile
                .map(|profile| profile.device_snapshot_budget_bytes),
            || cache.usage().budget_bytes,
        ) {
            cache.enable_pressure_host_snapshots(policy);
        }
        self.session = Some(SessionDriver::new(cache));
    }

    /// The attached session cache, if any.
    pub fn session_cache(&self) -> Option<&SessionCache> {
        self.session.as_ref().map(SessionDriver::cache)
    }

    /// Drop every session snapshot, pending ones included, releasing their
    /// device buffers, and the live state; with a disk tier attached,
    /// published snapshots are demoted to disk instead of dropped. No-op
    /// without a cache.
    pub fn session_clear(&mut self, gpu: &mut Gpu) {
        // Under an attached PP mesh `pp_session_clear` releases the aggregate
        // pins before any band's cache frees its snapshots.
        if self.pp.is_some() {
            return;
        }
        if self.ep.is_some() {
            return self.ep_session_clear(gpu);
        }
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

    /// [`ArchModel::session_plan`](hipfire_runtime::arch_model::ArchModel::session_plan)
    /// restricted to reuse points `<= cap` ([`SessionDriver::plan_capped`]).
    /// With `cap = prompt.len()` it is the plain plan; the EP mesh lowers `cap`
    /// until every rank agrees on one reuse point.
    pub(crate) fn session_plan_capped(
        &self,
        prompt: &[u32],
        route: SessionRoute,
        cap: usize,
    ) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| session.plan_capped(self, prompt, route, cap))
    }

    /// Whether a planned reuse continues current or parked live state rather
    /// than a pool/resident boundary. Used by request-source tracing.
    pub fn session_plan_is_live(&self, prompt: &[u32], route: SessionRoute, reused: usize) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.plan_is_live(self, prompt, route, reused))
    }

    /// The host-visible request configuration every EP rank must share.
    pub(crate) fn ep_replication_fingerprint(&self) -> ReplicaFingerprint {
        ReplicaFingerprint {
            session_cache: self.session.is_some(),
            chunk_rows: self.spec_chunk_rows(),
            state_format: self.state_format,
            turn_end_token: self.turn_end_token,
            turn_snapshots: self.turn_snapshots,
            tail_snapshot: self.tail_snapshot,
            tail_snapshot_min_gap: self.tail_snapshot_min_gap,
            tail_snapshot_min_prompt: self.tail_snapshot_min_prompt,
            has_mtp: self.mtp.is_some(),
            has_lanes: self.lanes.is_some(),
        }
    }

    /// Bound all row writes of the next request, including speculative
    /// lookahead. Without this advisory bound, side-request parking is off.
    pub(crate) fn session_request_write_limit(&mut self, through: usize) {
        let set = |bundle: &mut Self| {
            if let Some(session) = bundle.session.as_mut() {
                session.set_request_write_limit(through);
            }
            Ok(())
        };
        if let Err(error) = self.ep_fanout("session_request_write_limit", set, &|bundle, _, _| {
            if let Some(session) = bundle.session.as_mut() {
                session.set_request_write_limit(through);
            }
            Ok(())
        }) {
            eprintln!("  qwen4 EP: session_request_write_limit: {error}");
        }
    }

    /// Start a prefill of `prompt` on `route` that skips `reused` tokens
    /// ([`SessionDriver::begin`]): a live continuation of this conversation's
    /// previous turn, a snapshot restore, or a cold start. Message-end
    /// boundaries of a live continuation are captured when active.
    pub fn session_begin(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<(), BundleError> {
        self.ep_refuse("session_begin outside a replicated call")?;
        self.begin_npu_trunk_request(gpu)?;
        // A lookahead left by an aborted request names another prompt's rows.
        self.set_ple_lookahead(&[]);
        let capture = self.turn_snapshots_active();
        let result = match self.with_session(|session, bundle| {
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
        };
        if result.is_ok() {
            let end = self
                .session_lineage_after(reused)
                .map_or(prompt.len(), |p| p.min(prompt.len()));
            if let Some(forward) = &mut self.execution {
                forward
                    .plan_npu_trunk_request(gpu, end - reused, reused)
                    .map_err(|error| BundleError::Forward(error.to_string()))?;
            }
        }
        result
    }

    /// Next absolute position the running prefill must stop at and report.
    pub(crate) fn session_next_boundary(&self) -> Option<usize> {
        self.session.as_ref()?.next_boundary()
    }

    pub(crate) fn spec_advance_session_boundary(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<(), BundleError> {
        if self.session.as_ref().and_then(SessionDriver::turn_route) == Some(SessionRoute::Ar) {
            self.publish_pair_session_boundary(gpu, position)?;
        }
        Ok(())
    }

    /// Report that every owner consumed exactly `prefix` (the boundary).
    pub(crate) fn session_at_boundary(
        &mut self,
        gpu: &mut Gpu,
        prefix: &[u32],
    ) -> Result<(), BundleError> {
        self.ep_refuse("session_at_boundary outside a replicated call")?;
        self.with_session(|session, bundle| session.at_boundary(gpu, bundle, prefix))
            .unwrap_or(Ok(()))
            .map_err(BundleError::Forward)
    }

    /// The host-only schedule settings of this bundle (the root's are the
    /// authoritative copy every PP band adopts).
    pub(crate) fn pp_session_policy(&self) -> PpSessionPolicy {
        PpSessionPolicy {
            turn_snapshots: self.turn_snapshots,
            tail_snapshot: self.tail_snapshot,
            tail_snapshot_min_gap: self.tail_snapshot_min_gap,
            tail_snapshot_min_prompt: self.tail_snapshot_min_prompt,
            turn_end_token: self.turn_end_token,
        }
    }

    /// Adopt the root's schedule settings so every band's scope and boundaries
    /// match; called at every plan and prefill begin.
    pub(crate) fn pp_adopt_session_policy(&mut self, policy: PpSessionPolicy) {
        self.turn_snapshots = policy.turn_snapshots;
        self.tail_snapshot = policy.tail_snapshot;
        self.tail_snapshot_min_gap = policy.tail_snapshot_min_gap;
        self.tail_snapshot_min_prompt = policy.tail_snapshot_min_prompt;
        self.turn_end_token = policy.turn_end_token;
    }

    /// Whether `piece` is a published pool snapshot of this band (pure,
    /// host-only: no pin, no GPU).
    pub(crate) fn pp_session_piece_resident(&self, piece: &SessionPieceKey) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.piece_resident(piece))
    }

    /// This band's pool key of `prefix` on `route` under its current scope.
    pub(crate) fn pp_session_piece_key(
        &self,
        prefix: &[u32],
        route: SessionRoute,
    ) -> Option<SessionPieceKey> {
        self.session.as_ref()?.piece_key(self, prefix, route)
    }

    /// Cold-start (`reused == 0`) or restore this band's snapshot of the
    /// aggregate record's prefix and arm its boundaries. Never continues a
    /// live state ([`SessionDriver::begin_snapshot`]). Runs on the band's own
    /// `gpu`, detached from the mesh.
    pub(crate) fn pp_session_begin(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
    ) -> Result<(), String> {
        if route != SessionRoute::Ar {
            return Err(
                "qwen4 topology.pp: MTP-route session snapshots are not supported".to_string(),
            );
        }
        // A lookahead left by an aborted request names another prompt's rows.
        self.set_ple_lookahead(&[]);
        self.with_session(|session, bundle| {
            session.begin_snapshot(gpu, bundle, prompt, route, reused)
        })
        .unwrap_or_else(|| Err("qwen4 topology.pp band has no session cache".to_string()))
    }

    /// Capture this band at `position` of `prompt` (enqueued on its own stream
    /// after its forward work, pending until the aggregate commit) and name the
    /// pool piece it will publish.
    pub(crate) fn pp_session_capture(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        position: usize,
    ) -> Result<Option<SessionPieceKey>, String> {
        let prefix = prompt
            .get(..position)
            .ok_or("qwen4 topology.pp session boundary exceeds the prompt")?;
        self.session_at_boundary(gpu, prefix)
            .map_err(|error| error.to_string())?;
        let session = self
            .session
            .as_ref()
            .ok_or("qwen4 topology.pp band has no session cache")?;
        let route = session
            .turn_route()
            .ok_or("qwen4 topology.pp session boundary has no prefill in flight")?;
        Ok(session.piece_key(&*self, prefix, route))
    }

    /// The boundaries this band's armed turn has yet to report and its lineage
    /// split row; `None` without a session cache.
    pub(crate) fn pp_session_schedule(&self) -> Option<(Vec<usize>, Option<usize>)> {
        let session = self.session.as_ref()?;
        Some((session.turn_boundaries(), session.turn_lineage()))
    }

    /// Replace this band's armed schedule with the root's canonical one.
    pub(crate) fn pp_session_set_schedule(
        &mut self,
        boundaries: Vec<usize>,
        lineage: Option<usize>,
    ) -> Result<(), String> {
        self.session
            .as_mut()
            .ok_or_else(|| "qwen4 topology.pp band has no session cache".to_string())?
            .set_boundaries(boundaries, lineage)
    }

    /// Drop this band's live state and every uncommitted capture. Only after
    /// the device was drained (mesh reset).
    pub(crate) fn pp_session_abort(&mut self, gpu: &mut Gpu) {
        if let Some(session) = self.session.as_mut() {
            session.abort(gpu);
        }
    }

    /// Commit this band's pending captures (host only), keeping the live state
    /// when `consumed` is given and extends the turn ([`SessionDriver::commit_live`]).
    pub(crate) fn pp_session_commit_local(&mut self, consumed: Option<&[u32]>) {
        let Some(mut session) = self.session.take() else {
            return;
        };
        match consumed {
            Some(consumed) => session.commit_live(&*self, consumed),
            None => session.commit(),
        }
        self.session = Some(session);
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
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        if self.ep.is_some() {
            return self.ep_prefill_final(gpu, prompt, reused, logits);
        }
        // An NPU fault latches the route off; the request then reruns from the re-planned session start with the
        // exact GPU-only schedule.
        crate::npu_trunk::containment::rerun_after_fault(
            self,
            prompt.len(),
            reused,
            |bundle, start| bundle.prefill_final_from(gpu, prompt, start, logits),
            |bundle| bundle.take_npu_trunk_fault(),
            |bundle| bundle.session_plan_capped(prompt, SessionRoute::Ar, prompt.len()),
        )
    }

    fn prefill_final_from(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.share_prompt_begin();
        let result = self.prefill_final_tiles(gpu, prompt, reused, logits);
        let finish = self.share_prompt_end(gpu, result.is_ok());
        result.and(finish)
    }

    /// The chunk loop of [`Self::prefill_final_from`].
    fn prefill_final_tiles(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.session_begin(gpu, prompt, reused, SessionRoute::Ar)?;
        // The one selector verdict of the WHOLE remaining prompt, before the
        // session-boundary split below: a layer-major request keeps its
        // boundaries as in-window cuts, so it sweeps the experts once per
        // window rather than once per boundary chunk.
        if reused < prompt.len() && self.state.position == reused {
            let output = crate::slowlink_prefill::PromptOutput::Final { logits, top1: None };
            if self.slowlink_dispatch(gpu, &prompt[reused..], output, SlowlinkScope::prompt())? {
                return Ok(());
            }
        } else {
            self.slowlink_note_bypass(
                "no rows, or the entry position differs from the session start",
            );
        }
        let mut pos = reused;
        let mut windows: Vec<(usize, usize)> = Vec::new();
        while pos < prompt.len() {
            self.poll_npu_trunk(gpu)?;
            let end = if self.npu_trunk_planned() {
                pos + self.prefill_schedule_rows(
                    pos,
                    prompt.len(),
                    self.spec_chunk_rows().unwrap_or(1),
                )
            } else {
                self.session_next_boundary().unwrap_or(prompt.len())
            };
            self.set_ple_lookahead(&prompt[end..]);
            if end < prompt.len() {
                self.forward_chunk_silent_incumbent(gpu, &prompt[pos..end])?;
            } else {
                self.forward_chunk_final_incumbent(gpu, &prompt[pos..end], logits, None)?;
            }
            windows.push((pos - reused, end - pos));
            if self.session_next_boundary() == Some(end) {
                self.session_at_boundary(gpu, &prompt[..end])?;
            }
            pos = end;
        }
        self.slowlink_record_incumbent(reused, &windows);
        Ok(())
    }

    /// Request-state storage formats of the resident (and every lane) state.
    pub fn state_format(&self) -> Qwen4StateFormat {
        self.state_format
    }

    /// The exact singleton AR launch reference (`Eager` = ordinary HIP launches,
    /// `Recorded` = retained-body route), from the attached forward's production
    /// predicate; `None` while the forward is not attached. For the exact-load
    /// chooser dynamic-route invalidation and the AR cost receipt n=1.
    pub fn ar_verify_reference(
        &self,
        gpu: &Gpu,
    ) -> Option<rdna_compute::prefill_invariant::Qwen4VerifyReference> {
        let forward = self.execution.as_ref()?;
        Some(match forward.ar_launch_reference(self, gpu) {
            crate::gpu_forward::Qwen4ArLaunch::Eager => {
                rdna_compute::prefill_invariant::Qwen4VerifyReference::Eager
            }
            crate::gpu_forward::Qwen4ArLaunch::Recorded => {
                rdna_compute::prefill_invariant::Qwen4VerifyReference::Recorded
            }
        })
    }

    /// Measured native MTP capture rows: the state ring's actual
    /// `row_capture_rows` (0 where the route captures no rows), not a
    /// qualified cap or planner value. For the immutable load descriptor.
    pub fn native_mtp_capture_rows(&self) -> usize {
        self.state.row_capture_rows()
    }

    /// Stage the exact lane store: `max_lanes` request owners, at most
    /// `row_budget` decode lanes per step. Lane states are allocated on first
    /// admission into a slot. With `spec` every lane is an MTP lane (native
    /// MTP windows of at most `spec.k` drafts); refused when the bundle has
    /// no MTP head or `k + 1` verify rows exceed `LANE_MAX_ROWS`.
    ///
    /// Under an attached EP mesh the store is staged on every rank (AR lanes
    /// only; each rank checks its own free device memory) and rolled back on
    /// every rank if any rank fails ([`Self::ep_stage_lanes`]).
    pub fn stage_lanes(
        &mut self,
        gpu: &mut Gpu,
        max_lanes: usize,
        row_budget: usize,
        spec: Option<crate::lane::LaneSpecConfig>,
    ) -> Result<(), BundleError> {
        self.pp_refuse("continuous-batch lanes")?;
        if self.ep.is_some() {
            return self.ep_stage_lanes(gpu, max_lanes, row_budget, spec);
        }
        if self.lanes.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 lanes are already staged".to_string(),
            ));
        }
        if self.execution.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 lanes need the attached forward".to_string(),
            ));
        }
        if self
            .execution
            .as_ref()
            .is_some_and(|forward| forward.decode_share_active())
        {
            return Err(BundleError::Forward(
                "Qwen4 lanes cannot be staged on a decode-share load: lane steps and lane prefill \
                 waves need a bank of their own (not supported yet)"
                    .to_string(),
            ));
        }
        if let Some(spec) = spec {
            let verify_rows = if self.has_cpu_deferred_experts() {
                self.cpu_qualified_verify_rows(
                    gpu,
                    crate::gpu_forward::Qwen4GpuForward::cpu_lab_override_present(gpu),
                )
                .max(crate::program::QWEN4_CPU_SPLICE_MAX_ROWS)
            } else {
                crate::gpu_forward_lanes::LANE_MAX_ROWS
            };
            if self.mtp.is_none() {
                return Err(BundleError::Forward(
                    "Qwen4 MTP lanes need the attached MTP head".to_string(),
                ));
            }
            if spec.k == 0 || spec.k >= verify_rows {
                return Err(BundleError::Forward(format!(
                    "Qwen4 MTP lanes need 1 <= k <= {} for this execution mode, got {}",
                    verify_rows - 1,
                    spec.k
                )));
            }
            // The lanes' prompt fill captures wide hidden rows into the
            // resident spec buffer, as the singleton prefill does.
            let chunk = self.spec_chunk_rows().unwrap_or(1).max(1);
            self.ensure_spec_hidden(gpu, (spec.k + 1).max(chunk))?;
        }
        let store = crate::lane::Qwen4LaneStore::new(gpu, self, max_lanes, row_budget, spec)
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
    pub fn lane_executor(
        &mut self,
    ) -> Result<crate::lane_exec::Qwen4LaneExecutor<'_>, BundleError> {
        if self.lanes.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 lanes are not staged".to_string(),
            ));
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
        if self.ep.is_some() {
            return self.ep_admit_lane(gpu, epoch, prompt);
        }
        self.with_lanes(|store, bundle| store.admit(gpu, bundle, epoch, prompt))
    }

    /// Free `epoch`'s lane (its device state stays allocated for reuse).
    pub fn release_lane(
        &mut self,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        if self.ep.is_some() {
            return self.ep_release_lane(epoch);
        }
        self.with_lanes(|store, _| store.release(epoch))
    }

    /// Mark `epoch`'s lane untrusted: it runs no further step.
    pub fn poison_lane(&mut self, epoch: &hipfire_runtime::slot_batch::RequestEpoch) {
        if self.ep.is_some() {
            self.ep_poison_lane(epoch);
            return;
        }
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
        if self.ep.is_some() {
            return self.ep_force_lane_seed(epoch, token);
        }
        self.with_lanes(|store, _| store.force_seed(epoch, token))
    }

    /// Developer / oracle entry (the serve path never calls it): replace the
    /// row-local stage share policy of EVERY lane store of this load, so a
    /// measurement can compare shared and per-lane stage execution on the
    /// topology itself (the default policy is resolved per stage from the
    /// `LANE_ROW_TWIN_PROOFS` rows by each store's actually bound weight
    /// routes: a stage is shared only if every owned instance route matches a
    /// proof row, so a stage with an unproved instance stays per lane: for
    /// example a MoE layer outside the EP packed static-or-balanced class
    /// (`LaneExpertRoute::EpPackedStaticOrBalanced`), or any route outside the
    /// proof rows).
    ///
    /// - single: this bundle's store;
    /// - EP: every rank's store, replicated through the soft lane fan-out
    ///   (ranks must agree; a refusal all ranks report identically leaves the
    ///   mesh usable);
    /// - PP: every band's store, root included (host only; `gpus` is the mesh
    ///   owner the other PP lane calls require and is mandatory under PP).
    ///
    /// Refused, with no store changed: lanes not staged on every store, a
    /// step in flight, or a policy that fails
    /// [`crate::gpu_forward_lanes::LaneStagePolicy::validate`]. Stateful
    /// stages stay per lane (the policy type cannot say otherwise).
    pub fn set_lane_stage_policy(
        &mut self,
        gpus: Option<&mut hipfire_runtime::multi_gpu::Gpus>,
        policy: crate::gpu_forward_lanes::LaneStagePolicy,
    ) -> Result<(), BundleError> {
        policy.validate().map_err(BundleError::Forward)?;
        if self.pp.is_some() {
            let gpus = gpus.ok_or_else(|| {
                BundleError::Forward(
                    "qwen4 topology.pp: set_lane_stage_policy needs the mesh's Gpus".to_string(),
                )
            })?;
            return self.pp_set_lane_stage_policy(gpus, policy);
        }
        if self.ep.is_some() {
            return self.ep_set_lane_stage_policy(policy);
        }
        self.set_lane_stage_policy_local(policy)
            .map_err(BundleError::Forward)
    }

    /// Check this bundle's own store can take a policy change.
    pub(crate) fn lane_stage_policy_ready(&self) -> Result<(), String> {
        let store = self
            .lanes
            .as_ref()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        if store.phase != crate::lane_exec::LanePhase::Idle {
            return Err("set_lane_stage_policy: a step is in flight".to_string());
        }
        Ok(())
    }

    /// This bundle's store only (no mesh fan-out).
    pub(crate) fn set_lane_stage_policy_local(
        &mut self,
        policy: crate::gpu_forward_lanes::LaneStagePolicy,
    ) -> Result<(), String> {
        policy.validate()?;
        self.lane_stage_policy_ready()?;
        self.lanes
            .as_mut()
            .expect("checked by lane_stage_policy_ready")
            .set_stage_policy(policy);
        Ok(())
    }

    /// The resident owners must be at a clean window boundary to move into a
    /// lane: no armed GDN ring and no outstanding trunk or head snapshot.
    fn promotion_resident_ready(&self) -> Result<(), String> {
        if self.state.row_capture_armed {
            return Err("promotion: the resident GDN ring is armed".to_string());
        }
        if self.state.snapshot_active() {
            return Err("promotion: the resident trunk has an active snapshot ticket".to_string());
        }
        if self
            .mtp
            .as_ref()
            .is_some_and(|mtp| mtp.state.snapshot_active())
        {
            return Err(
                "promotion: the resident MTP head has an active snapshot ticket".to_string(),
            );
        }
        Ok(())
    }

    /// Promotion step 1 (fallible, resident untouched): admit `epoch` into a
    /// free lane as Decoding with `prompt`, the `generated` picks and
    /// `pending_seed` (the next AR row's token), `prefilled = prompt.len()`,
    /// allocating the slot's states if it has none. On an MTP lane store the
    /// singleton drafter's `parts` seed the lane: its pending hidden is
    /// copied in and the lane policy starts from its agreement (a retired
    /// request stays `ArRetired`); an AR store takes `None`. Refused: lanes
    /// not staged, a step in flight, no free lane, `position` not the
    /// resident `state.position`, an armed ring or an active snapshot ticket
    /// on the resident owners, a spec store without `parts` (or an AR store
    /// with), and `state.position != head.position` unless retired. Pair
    /// with [`Self::commit_promotion`] or [`Self::abort_promotion`].
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_promotion(
        &mut self,
        gpu: &mut Gpu,
        epoch: hipfire_runtime::slot_batch::RequestEpoch,
        prompt: Vec<u32>,
        generated: Vec<u32>,
        pending_seed: u32,
        position: usize,
        parts: Option<crate::mtp_spec::Qwen4PromotionParts<'_>>,
    ) -> Result<usize, BundleError> {
        if self.ep.is_some() {
            return Err(crate::ep::ep_err(
                "promotion into lanes is not supported yet",
            ));
        }
        let spec = self
            .lanes
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 lanes are not staged".to_string()))?
            .spec();
        if position != self.state.position {
            return Err(BundleError::Forward(format!(
                "promotion: position {position} is not the resident state's {}",
                self.state.position
            )));
        }
        self.promotion_resident_ready()
            .map_err(BundleError::Forward)?;
        match (spec, parts.as_ref()) {
            (Some(_), None) => {
                return Err(BundleError::Forward(
                    "promotion: an MTP lane store needs the drafter's parts".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(BundleError::Forward(
                    "promotion: an AR lane store takes no drafter parts".to_string(),
                ));
            }
            _ => {}
        }
        if let (Some(spec), Some(parts)) = (spec, parts.as_ref()) {
            let head = self.mtp.as_ref().ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?;
            if !parts.retired && head.position() != self.state.position {
                return Err(BundleError::Forward(format!(
                    "promotion: the MTP head is at {}, the resident state at {}",
                    head.position(),
                    self.state.position
                )));
            }
            // The lane keeps the resident state's GDN ring: size it for the
            // lane's windows (bytes unchanged; a no-op once large enough).
            self.state
                .ensure_row_capture(gpu, spec.k + 1, self.config.linear_conv_kernel_dim - 1)
                .map_err(|e| BundleError::Forward(format!("promotion: resident ring: {e}")))?;
        }
        self.with_lanes(|store, bundle| {
            store.admit_promoted(gpu, bundle, epoch, prompt, generated, pending_seed, parts)
        })
    }

    /// Undo a prepared promotion: release `epoch`'s lane. The resident state
    /// was never touched by [`Self::prepare_promotion`].
    pub fn abort_promotion(
        &mut self,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        if self.ep.is_some() {
            return self.ep_abort_promotion(epoch);
        }
        self.with_lanes(|store, _| store.release(epoch))
    }

    /// Promotion step 3: swap the resident trunk state (and, on an MTP lane
    /// store, the resident head state and draft request state) into the
    /// prepared lane `epoch`, then reset the resident owners for the next
    /// singleton request ([`Self::reset`]: session live record and turn
    /// dropped). Every refusal comes before the swap and leaves both sides
    /// untouched; once the lane holds the moved states only the reset can
    /// fail, and its error is returned with the lane already live.
    pub fn commit_promotion(
        &mut self,
        gpu: &mut Gpu,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        if self.ep.is_some() {
            return self.ep_commit_promotion(gpu, epoch);
        }
        self.with_lanes(|store, bundle| {
            if store.phase != crate::lane_exec::LanePhase::Idle {
                return Err("promotion: a lane step is in flight".to_string());
            }
            let ring_rows = store.spec().map(|spec| spec.k + 1);
            bundle.promotion_resident_ready()?;
            let lane = store
                .lane_mut(epoch)
                .ok_or_else(|| format!("promotion: unknown or stale epoch {epoch:?}"))?;
            if lane.host.status != crate::lane::LaneStatus::Decoding {
                return Err(format!(
                    "promotion: lane {epoch:?} is {:?}, not Decoding",
                    lane.host.status
                ));
            }
            // An MTP lane needs the resident head to move in. An AR lane takes
            // the trunk state only: a singleton that ran native MTP keeps its
            // head resident, and the reset below clears it.
            if lane.spec.is_some() && bundle.mtp.is_none() {
                return Err("promotion: an MTP lane needs the resident MTP head".to_string());
            }
            if let Some(rows) = ring_rows {
                if bundle.state.row_capture_rows() < rows {
                    return Err(format!(
                        "promotion: the resident ring holds {} rows, the lane needs {rows}",
                        bundle.state.row_capture_rows()
                    ));
                }
            }
            bundle.state.clamp_f16_valid();
            lane.state.clamp_f16_valid();
            std::mem::swap(&mut bundle.state, &mut lane.state);
            if let (Some(spec), Some(mtp)) = (lane.spec.as_mut(), bundle.mtp.as_mut()) {
                std::mem::swap(&mut mtp.state, &mut spec.head);
                spec.draft = mtp.draft.request_state();
                mtp.draft.reset_request_state();
            }
            Ok(())
        })?;
        self.reset(gpu)
    }

    /// A fresh lane trunk state: the resident state's formats, QSA backend
    /// and context bound, allocated through the same constructor over this
    /// bundle's band (the whole model under the single scope; a PP band's
    /// layers and endpoint state otherwise).
    pub(crate) fn new_lane_state(&self, gpu: &mut Gpu) -> Result<Qwen4State, BundleError> {
        // Attached roots need the mesh owner; detached band/rank calls
        // allocate the state local to their weight scope.
        if self.pp.is_some() {
            return Err(pp_owner_required());
        }
        Qwen4State::new_with_backend_scoped(
            gpu,
            &self.config,
            self.state.max_seq_len,
            self.state_format,
            self.state.qsa_backend(),
            self.weights.band_scope(),
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
        let tile_end =
            segment_start + ((prefilled - segment_start) / max_chunk + 1).saturating_mul(max_chunk);
        tile_end
            .min(segment_end)
            .saturating_sub(prefilled)
            .min(remaining)
    }

    /// Run `f` with the lane's `state` (and, when given, its head state and
    /// draft request state) swapped into the resident slots, the session
    /// cache out and the forward in lane scope with no PLE lookahead, so
    /// the singleton chunk/verify/head programs run on the lane's own
    /// states and never record or replay the resident state's retained
    /// body. Everything is back in place, the lane's updated draft request
    /// state written to `head`, before this returns; `Err` is only returned
    /// before anything was swapped (no forward attached, or `head` given
    /// with no MTP head attached).
    pub(crate) fn with_lane<R>(
        &mut self,
        state: &mut Qwen4State,
        head: Option<(
            &mut crate::mtp_gpu::MtpGpuState,
            &mut hipfire_dispatch::pipeline::DraftHeadRequestState,
        )>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, BundleError> {
        if self.execution.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 forward resources are not attached".to_string(),
            ));
        }
        if head.is_some() && self.mtp.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 MTP resources are not attached".to_string(),
            ));
        }
        let mut head = head;
        let session = self.session.take();
        self.state.clamp_f16_valid();
        state.clamp_f16_valid();
        std::mem::swap(&mut self.state, state);
        let resident_draft = match (head.as_mut(), self.mtp.as_mut()) {
            (Some((head_state, draft)), Some(mtp)) => {
                std::mem::swap(&mut mtp.state, &mut **head_state);
                let resident = mtp.draft.request_state();
                mtp.draft.set_request_state(**draft);
                Some(resident)
            }
            _ => None,
        };
        let prior_scope = match self.execution.as_mut() {
            Some(forward) => {
                let prior = forward.lane_scope;
                forward.lane_scope = true;
                // No lookahead names this lane's next rows: never adopt
                // another request's PLE ticket by accident (ids would have
                // to match exactly, which keeps bytes identical anyway, but
                // the prefetch is wasted).
                forward.set_ple_lookahead(&[]);
                prior
            }
            None => false,
        };
        let result = f(self);
        if let Some(forward) = self.execution.as_mut() {
            forward.lane_scope = prior_scope;
        }
        if let (Some(resident), Some((head_state, draft)), Some(mtp)) =
            (resident_draft, head.as_mut(), self.mtp.as_mut())
        {
            **draft = mtp.draft.request_state();
            mtp.draft.set_request_state(resident);
            std::mem::swap(&mut mtp.state, &mut **head_state);
        }
        self.state.clamp_f16_valid();
        state.clamp_f16_valid();
        std::mem::swap(&mut self.state, state);
        self.session = session;
        Ok(result)
    }

    /// One prefill tile of a lane: the singleton chunk program
    /// (`forward_chunk_final` / `forward_chunk_silent`) on `state`, inside
    /// [`Self::with_lane`] (lanes keep no session cache; the forward never
    /// records or replays the resident state's retained body). The resident
    /// state is back in place on every return path.
    pub(crate) fn lane_prefill_tile(
        &mut self,
        gpu: &mut Gpu,
        state: &mut Qwen4State,
        tokens: &[u32],
        final_logits: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        self.pp_refuse("continuous-batch lane prefill")?;
        self.with_lane(state, None, |bundle| {
            let mut forward = bundle.execution.take().ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?;
            let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
            let result = match final_logits {
                Some(logits) => forward.forward_chunk(
                    bundle,
                    gpu,
                    tokens,
                    logits,
                    None,
                    None,
                    Qwen4OutputRows::Final,
                ),
                None => forward.forward_chunk_silent(bundle, gpu, tokens, None),
            }
            .map_err(|error| BundleError::Forward(error.to_string()));
            forward.prefill_scope = prior_scope;
            bundle.execution = Some(forward);
            result
        })?
    }

    /// A fresh prefill-wave lane state: the same constructor as a lane
    /// store's states (the resident state's formats, QSA backend and context
    /// bound).
    pub fn new_wave_state(&self, gpu: &mut Gpu) -> Result<Qwen4State, BundleError> {
        self.new_lane_state(gpu)
    }

    /// One prefill wave: every lane's rows forwarded together
    /// ([`crate::gpu_forward_lanes`] module docs). Each lane state's QSA
    /// arenas are mapped through its end position first; the wave then runs
    /// one forward in which row-local stages follow `policy` and stateful
    /// stages run per lane on the lane's own state, publishing each lane's
    /// host state as the singleton forward does. `logits` is F32,
    /// `lanes.len() * vocab` (`WaveHead::Final`) or `total_rows * vocab`
    /// (`WaveHead::All`); `top1` is Raw, one i32 per output row. The wave
    /// never touches the resident `state`, the MTP head, the session cache
    /// or the retained tape (eager HIP); `execution` is back in place on
    /// every return path.
    pub fn prefill_wave(
        &mut self,
        gpu: &mut Gpu,
        lanes: &mut [crate::gpu_forward_lanes::WaveLane<'_>],
        head: crate::gpu_forward_lanes::WaveHead,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        policy: &crate::gpu_forward_lanes::LaneStagePolicy,
    ) -> Result<(), BundleError> {
        self.ep_refuse("continuous-batch wave prefill")?;
        for lane in lanes.iter_mut() {
            let end = lane
                .state
                .position
                .checked_add(lane.tokens.len())
                .ok_or_else(|| {
                    BundleError::Forward("Qwen4 wave lane position overflows".to_string())
                })?;
            if end > lane.state.max_seq_len {
                return Err(BundleError::Forward(format!(
                    "Qwen4 wave lane end position {end} exceeds its context capacity {}",
                    lane.state.max_seq_len
                )));
            }
            lane.state
                .ensure_mapped_capacity(gpu, end)
                .map_err(|error| {
                    BundleError::Forward(format!(
                        "map Qwen4 wave lane QSA context through {end} tokens: {error}"
                    ))
                })?;
        }
        crate::gpu_forward_lanes::forward_wave(gpu, self, lanes, head, logits, top1, policy)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub fn snapshot(&mut self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, BundleError> {
        self.ep_refuse("state snapshots")?;
        self.quiesce_ple()?;
        self.state.snapshot(gpu).map_err(BundleError::State)
    }

    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.ep_refuse("state restore")?;
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
        self.ep_refuse("state restore")?;
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

    /// [`Self::commit_validated`] that keeps the arena entry active and
    /// rollback-able instead (`SessionState::retain_validated` of the local
    /// ticket): the target half of [`Self::retain_window`] for one band of
    /// an aggregate window.
    pub(crate) fn retain_target_validated(&mut self, snapshot: Qwen4StateSnapshot) {
        self.state.retain_validated(snapshot);
    }

    /// The head half of [`Self::retain_window`], for an aggregate window.
    pub(crate) fn mtp_retain_validated(&mut self, snapshot: MtpGpuStateSnapshot) {
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.retain_validated(snapshot);
        }
    }

    /// Keep a committed window's two tickets rollback-able ([`Self::restore_retain`]
    /// and [`Self::mtp_restore_retain`]) until the next snapshot, reset or
    /// [`Self::release_window`]: both validate first, so a stale pair retains
    /// neither.
    pub(crate) fn retain_window(
        &mut self,
        target: Qwen4StateSnapshot,
        head: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.validate_commit(target)?;
        self.mtp_validate_commit(head)?;
        self.state.retain_validated(target);
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.retain_validated(head);
        }
        Ok(())
    }

    /// Release a window retained by [`Self::retain_window`]. A ticket that no
    /// longer validates (a snapshot, reset or session restore already
    /// superseded it) is left alone.
    pub(crate) fn release_window(&mut self, target: Qwen4StateSnapshot, head: MtpGpuStateSnapshot) {
        if self.validate_commit(target).is_ok() {
            self.commit_validated(target);
        }
        if self.mtp_validate_commit(head).is_ok() {
            self.mtp_commit_validated(head);
        }
    }

    pub fn commit(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.ep_refuse("state commit")?;
        self.quiesce_ple()?;
        self.state.commit(snapshot, gpu).map_err(BundleError::State)
    }

    /// Teardown is deliberately ordered: stop/quiesce PLE reads and release
    /// leases/page-cache resources, free mutable GPU state, free finalized
    /// resident weights, then drain the attached transaction/census.
    pub fn free_gpu(mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        if self.pp.is_some() {
            std::mem::forget(self);
            return Err(pp_owner_required());
        }
        // Peers drain and free first: their gather events and P2P reads end
        // before rank 0's buffers go.
        if let Err(error) = self.clear_image_context(gpu) {
            // A failed fence means image and trunk readers may still be live.
            // Quarantine the entire owner instead of recycling any of its
            // buffers into a subsequent model's allocation pool.
            std::mem::forget(self);
            return Err(error);
        }
        if let Some((_, vision)) = self.vision.take() {
            vision.free_gpu(gpu);
        }
        let ep_result = self.ep_close(gpu);
        let Qwen4Bundle {
            weights,
            state,
            ple,
            weight_store,
            execution,
            mtp,
            spec_logits,
            spec_top1,
            spec_hidden,
            dflash,
            session,
            pair_session_stage,
            slowlink,
            penalty_stage,
            lanes,
            ..
        } = self;
        let ple_rows = ple.map(|ple| {
            let Qwen4BundlePle {
                rows, hot, prewarm, ..
            } = ple;
            // The hot-row store's worker reads the pack and the model file:
            // stop and join it before the row store unloads.
            drop(hot);
            // The sidecar prewarm reads the sidecar the row store also reads.
            drop(prewarm);
            rows
        });
        if let Some(mut session) = session {
            session.clear(gpu);
        }
        let mut pair_stage_error = None;
        if let Some(stage) = pair_session_stage {
            for tensor in stage.fixed {
                if let Err(error) = gpu.free_tensor(tensor) {
                    pair_stage_error.get_or_insert(error);
                }
            }
        }
        for tensor in slowlink.into_tensors() {
            if let Err(error) = gpu.free_tensor(tensor) {
                pair_stage_error.get_or_insert(error);
            }
        }
        let ple_result = ple_rows
            .map(|rows| rows.unload().map(|_| ()).map_err(BundleError::PleRows))
            .unwrap_or(Ok(()));
        let execution_result = execution
            .map(|forward| forward.free_gpu(gpu).map_err(BundleError::Hip))
            .unwrap_or(Ok(()));
        let mtp_result = mtp
            .map(|mtp| {
                mtp.free_gpu(gpu)
                    .map_err(|error| BundleError::Forward(error.to_string()))
            })
            .unwrap_or(Ok(()));
        let mut spec_error = pair_stage_error;
        if let Some(Err(error)) = dflash.map(|target| target.free_gpu(gpu)) {
            spec_error.get_or_insert(error);
        }
        for tensor in [spec_logits, spec_top1, spec_hidden].into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                spec_error.get_or_insert(error);
            }
        }
        let spec_result = spec_error
            .map_or(Ok(()), |error| Err(BundleError::Hip(error)))
            .and(
                penalty_stage
                    .map(|stage| {
                        gpu.free_penalty_table_stage(stage)
                            .map_err(BundleError::Hip)
                    })
                    .unwrap_or(Ok(())),
            );
        // Lane states are request owners like `state`: free them first.
        let lanes_result = lanes
            .map(|store| store.free_gpu(gpu).map_err(BundleError::Forward))
            .unwrap_or(Ok(()));
        let state_result = lanes_result.and(state.free_gpu(gpu).map_err(BundleError::State));
        let weight_result = weights.free_gpu(gpu).map_err(BundleError::Hip);
        let store_result = weight_store.drain(gpu).map_err(BundleError::Hip);
        let freed = first_bundle_error([
            ple_result,
            execution_result,
            mtp_result,
            spec_result,
            state_result,
            weight_result,
            store_result,
        ]);
        ep_result.and(freed)
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
        match self.pp.as_ref() {
            Some(mesh) => mesh.sessions_enabled(),
            None => self.session.is_some(),
        }
    }

    fn session_request_write_limit(&mut self, through: usize) {
        Qwen4Bundle::session_request_write_limit(self, through);
    }

    fn session_plan(&self, prompt: &[u32], route: SessionRoute) -> usize {
        // Under an attached PP mesh only the aggregate record plans
        // (`pp_session_plan`); a root-only plan would restore one band.
        if self.pp.is_some() {
            return 0;
        }
        if self.ep.is_some() {
            return self.ep_session_plan(prompt, route);
        }
        self.session
            .as_ref()
            .map_or(0, |session| session.plan(self, prompt, route))
    }

    fn session_commit(&mut self) {
        // The aggregate publication is `pp_session_commit`; committing the root
        // alone would publish a partial record.
        if self.pp.is_some() {
            return;
        }
        if self.ep.is_some() {
            return self.ep_session_commit(None);
        }
        if let Some(session) = self.session.as_mut() {
            session.commit();
        }
    }

    /// Publish as [`Self::session_commit`] and keep the live state when
    /// `consumed` extends this turn's prompt and the owners sit exactly at its
    /// end. Later turns prefill only the suffix; their pool snapshots are
    /// session-exact, not cold-prefill references.
    fn session_commit_live(&mut self, consumed: &[u32]) {
        if self.pp.is_some() {
            return;
        }
        if self.ep.is_some() {
            return self.ep_session_commit(Some(consumed));
        }
        if let Some(mut session) = self.session.take() {
            session.commit_live(self, consumed);
            self.session = Some(session);
        }
    }

    fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        Qwen4Bundle::map_ahead_for_decode(
            self,
            gpu,
            position,
            remaining_new_tokens,
            max_draft_rows,
        );
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        if let Err(error) = Qwen4Bundle::free_gpu(*self, gpu) {
            eprintln!("Qwen4 bundle teardown failed: {error}");
        }
    }
}

/// Name prefix of the PLE shard tensors of `ple`
/// (`model.language_model.layers.{layer}.ple...ngram_embedding.shard_`); the
/// Flash-Next layer is 1.
fn ple_shard_name_prefix(ple: &Qwen4PleConfig) -> String {
    format!(
        "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.shard_",
        ple.layer
    )
}

fn ple_shard_index(ple: &Qwen4PleConfig, name: &str) -> Result<usize, WeightError> {
    let prefix = ple_shard_name_prefix(ple);
    let suffix = name
        .strip_prefix(prefix.as_str())
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
    if index >= ple.shard_count || index.to_string() != suffix {
        return Err(WeightError::DescriptorMismatch(format!(
            "PLE shard index {index} is outside canonical range in '{name}'"
        )));
    }
    Ok(index)
}

fn ordered_ple_entries<'a>(
    manifest: &'a Qwen4Manifest,
    ple: &Qwen4PleConfig,
) -> Result<Vec<&'a WeightEntry>, WeightError> {
    let entries = manifest.external_entries().collect::<Vec<_>>();
    if entries.len() != ple.shard_count {
        return Err(WeightError::PleShardCount {
            expected: ple.shard_count,
            actual: entries.len(),
        });
    }
    let mut ordered = vec![None; ple.shard_count];
    for entry in entries {
        let index = ple_shard_index(ple, &entry.name)?;
        if entry.layer != Some(ple.layer)
            || entry.logical_shape.as_slice() != [ple.shard_rows, ple.row_width]
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

/// [`validate_ple_metadata`] for an explicit profile geometry: the stored
/// head tables must match the profile's head count and its physical row
/// layout (`shard_count * shard_rows` padded rows, per-shard valid rows).
fn validate_ple_geometry(
    ple: &Qwen4PleConfig,
    metadata: &PleHashMetadata,
) -> Result<(), BundleError> {
    let mismatch = |message: String| BundleError::Weights(WeightError::DescriptorMismatch(message));
    if metadata.head_count() != ple.head_count {
        return Err(mismatch(format!(
            "PLE metadata has {} head tables; the profile declares {}",
            metadata.head_count(),
            ple.head_count
        )));
    }
    let physical_rows = (ple.shard_rows as u64)
        .checked_mul(ple.shard_count as u64)
        .ok_or_else(|| mismatch("PLE physical row count overflow".to_string()))?;
    if metadata.padded_rows() != physical_rows {
        return Err(mismatch(format!(
            "PLE metadata padded rows {} do not match physical rows {physical_rows}",
            metadata.padded_rows()
        )));
    }
    let valid_rows = (0..ple.shard_count).try_fold(0u64, |sum, shard| {
        sum.checked_add(ple.shard_valid_rows(shard) as u64)
            .ok_or_else(|| mismatch("PLE valid row count overflow".to_string()))
    })?;
    if metadata.valid_rows() != valid_rows {
        return Err(mismatch(format!(
            "PLE metadata valid rows {} do not match canonical rows {valid_rows}",
            metadata.valid_rows()
        )));
    }
    Ok(())
}

/// The refusal of a PLE operation on a profile that declares no PLE.
fn no_ple_rows() -> BundleError {
    BundleError::Weights(WeightError::DescriptorMismatch(
        "qwen4: this profile declares no PLE; it has no PLE rows source".into(),
    ))
}

/// Whether the Qwen4 executor can run `config` at all: the weights binder's
/// profile admission (a developer profile only under the
/// `qwen4-dev-profiles` feature, and only a dense FFN) and the forward's
/// (the RoPE base the QSA kernels implement).  Both refusals otherwise
/// surface after the weights were fulfilled onto the device; a loader calls
/// this right after artifact admission, before any allocation.
pub fn admit_executable_profile(config: &Qwen4Config) -> Result<(), BundleError> {
    crate::weights::admit_executable_profile(config).map_err(BundleError::Weights)?;
    Qwen4GpuForward::admit_profile(config).map_err(|error| BundleError::Forward(error.to_string()))
}

/// What a PLE-declaring profile builds its [`Qwen4BundlePle`] from.
struct PleInputs {
    metadata: PleHashMetadata,
    source: Qwen4PleRows,
}

/// The profile, the PLE hash metadata and the rows source must agree before
/// anything is assembled: `config.ple` `None` <=> no metadata <=> no rows
/// source.  Sidecar rows (and with them the hot store, the prewarm and the
/// rank table) are Flash-Next-only; a developer profile reads its embedded
/// rows.  The metadata must match the profile's PLE geometry.
fn admit_ple_inputs(
    config: &Qwen4Config,
    metadata: Option<PleHashMetadata>,
    source: Qwen4PleRows,
) -> Result<Option<PleInputs>, BundleError> {
    let refuse = |message: String| BundleError::Weights(WeightError::DescriptorMismatch(message));
    let profile = config.profile.id();
    match (config.ple.as_ref(), metadata, source) {
        (None, None, Qwen4PleRows::Absent) => Ok(None),
        (None, _, _) => Err(refuse(format!(
            "qwen4: profile {profile} declares no PLE, so it takes no PLE hash metadata and no PLE rows source"
        ))),
        (Some(_), None, _) => Err(refuse(format!(
            "qwen4: profile {profile} declares PLE but no PLE hash metadata was provided"
        ))),
        (Some(_), Some(_), Qwen4PleRows::Absent) => Err(refuse(format!(
            "qwen4: profile {profile} declares PLE but no PLE rows source was provided"
        ))),
        (Some(ple), Some(metadata), source) => {
            if !config.profile.is_flash_next() && !matches!(source, Qwen4PleRows::Embedded) {
                return Err(refuse(format!(
                    "qwen4: PLE sidecar rows are Flash-Next-only; profile {profile} reads its embedded PLE rows"
                )));
            }
            validate_ple_geometry(ple, &metadata)?;
            Ok(Some(PleInputs { metadata, source }))
        }
    }
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

/// Inputs of the PLE hot-row store, resolved before the weight sweep from the
/// PLE rows source's shard descriptors (no transaction exists yet).
#[derive(Debug)]
pub struct PleHotPlan {
    /// Absolute file offset of each PLE shard's first row in
    /// [`Self::source_path`], in shard order.
    pub shard_offsets: Vec<u64>,
    pub rows_per_shard: usize,
    /// Row encoding of the PLE shards: Q8F16 or Q4Lm (`ple_hot_plan` returns
    /// `None` for any other).
    pub row_encoding: RowEncoding,
    /// Values per PLE row.
    pub row_width: usize,
    pub encoded_row_bytes: usize,
    pub padded_rows: u64,
    /// Lowercase hex of [`PleHashMetadata::fingerprint`], the sidecar's key.
    pub fingerprint_hex: String,
    /// The rows source file the shard offsets address: the model file
    /// (monolithic artifact) or the PLE rows sidecar.  The hot-row pack is
    /// cut from, bound to and verified against exactly this file.
    pub source_path: PathBuf,
    /// Length of [`Self::source_path`], the pack header's binding.
    pub source_file_bytes: u64,
}

/// The hot-row store's plan, or `None` when the PLE shards are neither Q8F16
/// nor Q4Lm (the store holds raw encoded rows).  `descriptors` are the 128
/// ordered shard ranges of the rows source (`Qwen4HfqmArtifact::
/// ple_row_descriptors`); `source_path` / `source_file_bytes` name that
/// source's file.
pub fn ple_hot_plan(
    metadata: &PleHashMetadata,
    descriptors: &[SourceRangeDescriptor],
    source_path: &Path,
    source_file_bytes: u64,
) -> Result<Option<PleHotPlan>, String> {
    hot_plan_from_layouts(
        metadata,
        &shard_layouts(descriptors),
        source_path,
        source_file_bytes,
    )
}

/// [`ple_hot_plan`] over the layout facts of the descriptors.
fn hot_plan_from_layouts(
    metadata: &PleHashMetadata,
    descriptors: &[ShardLayout],
    source_path: &Path,
    source_file_bytes: u64,
) -> Result<Option<PleHotPlan>, String> {
    validate_ple_metadata(metadata).map_err(|error| error.to_string())?;
    if descriptors.len() != PLE_SHARD_COUNT {
        return Err(format!(
            "PLE rows source has {} shard descriptors, expected {PLE_SHARD_COUNT}",
            descriptors.len()
        ));
    }
    let mut encoding = None;
    let mut shard_offsets = Vec::with_capacity(descriptors.len());
    for (shard, descriptor) in descriptors.iter().enumerate() {
        let shard_encoding = match RowEncoding::from_dtype(&descriptor.dtype) {
            Some(shard_encoding @ (RowEncoding::Q8F16 | RowEncoding::Q4Lm)) => shard_encoding,
            _ => return Ok(None),
        };
        if *encoding.get_or_insert(shard_encoding) != shard_encoding {
            return Err(format!(
                "PLE shard {shard} is {} but shard 0 is {}",
                shard_encoding.dtype_name(),
                encoding.unwrap_or(shard_encoding).dtype_name()
            ));
        }
        let row_bytes = shard_encoding.encoded_row_bytes(PLE_ROW_WIDTH);
        let expected = (PLE_SHARD_ROWS * row_bytes) as u64;
        if descriptor.logical_shape.as_slice() != [PLE_SHARD_ROWS, PLE_ROW_WIDTH]
            || descriptor.length != expected
        {
            return Err(format!(
                "PLE shard {shard} is {:?} / {} bytes, expected [{PLE_SHARD_ROWS}, {PLE_ROW_WIDTH}] / {expected}",
                descriptor.logical_shape, descriptor.length
            ));
        }
        shard_offsets.push(descriptor.offset);
    }
    let Some(row_encoding) = encoding else {
        return Ok(None);
    };
    let fingerprint_hex = metadata
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(Some(PleHotPlan {
        shard_offsets,
        rows_per_shard: PLE_SHARD_ROWS,
        row_encoding,
        row_width: PLE_ROW_WIDTH,
        encoded_row_bytes: row_encoding.encoded_row_bytes(PLE_ROW_WIDTH),
        padded_rows: metadata.padded_rows(),
        fingerprint_hex,
        source_path: source_path.to_path_buf(),
        source_file_bytes,
    }))
}

/// A PLE hot-row store and its worker thread (verification, then prewarm).
/// Dropping stops the worker and joins it (the worker holds its own `Arc`),
/// so every load error path and unload quiesces it.
pub struct PleHotWorker {
    store: std::sync::Arc<HotRowStore>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PleHotWorker {
    pub fn new(store: std::sync::Arc<HotRowStore>, handle: std::thread::JoinHandle<()>) -> Self {
        Self {
            store,
            handle: Some(handle),
        }
    }

    pub fn store(&self) -> &std::sync::Arc<HotRowStore> {
        &self.store
    }
}

impl Drop for PleHotWorker {
    fn drop(&mut self) {
        self.store.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The sidecar's 128 shard descriptors, checked to the same standard as
/// [`ple_descriptors`] checks the embedded ones: count, order, shape, row
/// stride (Q8F16's 170 or Q4LM's 90 bytes, one encoding for all shards),
/// extent, valid rows (through the metadata) and one sealed
/// [`SourceFormat::PleRows`] identity.
fn sidecar_ple_descriptors(
    descriptors: &[SourceRangeDescriptor],
    metadata: &PleHashMetadata,
) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
    let mismatch = |message: String| BundleError::Weights(WeightError::DescriptorMismatch(message));
    validate_ple_metadata(metadata)?;
    check_sidecar_layout(&shard_layouts(descriptors)).map_err(mismatch)?;
    let first = &descriptors[0];
    if first.source_identity().format != SourceFormat::PleRows {
        return Err(mismatch(
            "PLE sidecar descriptors are not sealed from a PLE rows file".to_string(),
        ));
    }
    for (shard, descriptor) in descriptors.iter().enumerate() {
        if descriptor.source_identity() != first.source_identity() {
            return Err(mismatch(format!(
                "PLE sidecar shard {shard} source identity differs"
            )));
        }
    }
    Ok(descriptors.to_vec())
}

/// The part of a shard descriptor the PLE row layout checks read.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardLayout {
    dtype: String,
    logical_shape: Vec<usize>,
    offset: u64,
    length: u64,
}

fn shard_layouts(descriptors: &[SourceRangeDescriptor]) -> Vec<ShardLayout> {
    descriptors
        .iter()
        .map(|descriptor| ShardLayout {
            dtype: descriptor.dtype.clone(),
            logical_shape: descriptor.logical_shape.clone(),
            offset: descriptor.offset,
            length: descriptor.length,
        })
        .collect()
}

/// A sidecar's shard layout: [`PLE_SHARD_COUNT`] shards of one Q8F16/Q4Lm
/// encoding, each `[PLE_SHARD_ROWS, PLE_ROW_WIDTH]`, laid back to back in
/// shard order (so position proves order: shard `i` sits at
/// `first.offset + i * shard_bytes`).
fn check_sidecar_layout(layouts: &[ShardLayout]) -> Result<RowEncoding, String> {
    if layouts.len() != PLE_SHARD_COUNT {
        return Err(format!(
            "PLE sidecar has {} shard descriptors, expected {PLE_SHARD_COUNT}",
            layouts.len()
        ));
    }
    let first = &layouts[0];
    let encoding = match RowEncoding::from_dtype(&first.dtype) {
        Some(encoding @ (RowEncoding::Q8F16 | RowEncoding::Q4Lm)) => encoding,
        _ => {
            return Err(format!(
                "PLE sidecar rows are {}, expected Q8_0 or Q4LM",
                first.dtype
            ))
        }
    };
    let shard_bytes = (PLE_SHARD_ROWS * encoding.encoded_row_bytes(PLE_ROW_WIDTH)) as u64;
    for (shard, layout) in layouts.iter().enumerate() {
        let expected_offset = first.offset.checked_add(shard as u64 * shard_bytes);
        if RowEncoding::from_dtype(&layout.dtype) != Some(encoding)
            || layout.logical_shape.as_slice() != [PLE_SHARD_ROWS, PLE_ROW_WIDTH]
            || layout.length != shard_bytes
            || Some(layout.offset) != expected_offset
        {
            return Err(format!(
                "PLE sidecar shard {shard} is {} {:?} at {} for {} bytes, expected {} \
                 [{PLE_SHARD_ROWS}, {PLE_ROW_WIDTH}] at {expected_offset:?} for {shard_bytes} bytes",
                layout.dtype,
                layout.logical_shape,
                layout.offset,
                layout.length,
                encoding.dtype_name(),
            ));
        }
    }
    Ok(encoding)
}

/// The bundle's PLE row store over its rows source: the embedded shards'
/// descriptors (from the fulfilled transaction), a row-order sidecar's
/// descriptors, or a rank-order table's store.
fn build_ple_row_store(
    source: &Qwen4PleRows,
    transaction: &WeightLoadTransaction,
    manifest: &Qwen4Manifest,
    ple: &Qwen4PleConfig,
    metadata: &PleHashMetadata,
    staging_rows: usize,
    wide_readers: bool,
) -> Result<RowStore, BundleError> {
    let descriptors = match source {
        Qwen4PleRows::Absent => return Err(no_ple_rows()),
        Qwen4PleRows::Embedded => ple_descriptors(transaction, manifest, ple, metadata)?,
        Qwen4PleRows::Sidecar {
            rows: SidecarRows::Descriptors(descriptors),
            ..
        } => sidecar_ple_descriptors(descriptors, metadata)?,
        Qwen4PleRows::Sidecar {
            rows: SidecarRows::Table(store),
            ..
        } => {
            validate_ple_metadata(metadata)?;
            return RowStore::from_table_store(
                "qwen4-ple-reader",
                store.clone(),
                metadata.valid_rows(),
                staging_rows,
                wide_readers,
            )
            .map_err(BundleError::PleRows);
        }
    };
    RowStore::with_staging_rows(
        "qwen4-ple-reader",
        descriptors,
        metadata.valid_rows(),
        staging_rows,
        wide_readers,
    )
    .map_err(BundleError::PleRows)
}

fn ple_descriptors(
    transaction: &WeightLoadTransaction,
    manifest: &Qwen4Manifest,
    ple: &Qwen4PleConfig,
    metadata: &PleHashMetadata,
) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
    validate_ple_geometry(ple, metadata)?;
    let entries = ordered_ple_entries(manifest, ple).map_err(BundleError::Weights)?;
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
        // Q8F16 one; the stock-FP8 recipe's qt=56 rows declare one byte per
        // value.
        let accepted = [
            RowEncoding::Bf16,
            RowEncoding::Q8F16,
            RowEncoding::Fp8PerTensor,
        ]
        .into_iter()
        .filter_map(|encoding| encoding.checked_encoded_row_bytes(ple.row_width))
        .collect::<Vec<_>>();
        if !accepted.contains(&row_bytes) || valid_rows != ple.shard_valid_rows(index) {
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

/// Checked byte census of the admitted CPU-owned expert window (D1 + D2): the real
/// host transfer-stage bytes, host-slot count, full-slot GPU view extents and the
/// SUM of actual D2 executor instance bytes at the stored common capacity, plus the
/// real thread-local host-slot scratch pool bytes and the mapped cold-return bytes.
/// Checked, `None` on overflow; `0`-storage loads report zeros (legacy default:
/// nothing reserved). Read both pre-reserve (the REQUESTED plan:
/// [`Self::cpu_requested_storage_bytes`]) and post-reserve (the settled instances:
/// [`Self::cpu_storage_bytes`]) so the loader reconciles plan with fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuStorageBytes {
    /// Logical window rows actually reserved (`0` = legacy default).
    pub rows: usize,
    /// Computed host slots (`rows * top_k`; 640 at the physical 64 x top_k 10).
    pub slots: usize,
    /// Pinned transfer-stage f32 values (`2 * (rows * dim + rows * top_k)`).
    pub stage_values: usize,
    /// Pinned transfer-stage bytes (`stage_values * 4`).
    pub stage_bytes: usize,
    /// Full-slot `x_rot` view elements covered (`rows * dim`).
    pub x_rot_elements: usize,
    /// Full-slot router `topk` view elements covered (`slots`).
    pub topk_elements: usize,
    /// Full-slot expanded-down view elements covered (`slots * dim`).
    pub down_expanded_elements: usize,
    /// Allocated forward device bytes at the attached chunk/scratch plan.
    pub forward_device_bytes: u64,
    /// Planned native-MTP capture-ring bytes at `rows`.
    pub capture_bytes: usize,
    /// SUM of live D2 executor `input` bytes over armed layers (0 = no D2 owner).
    pub d2_input_bytes: usize,
    /// SUM of live D2 executor `mask` bytes over armed layers (0 = no D2 owner).
    pub d2_mask_bytes: usize,
    /// SUM of live D2 executor `output` bytes over armed layers (0 = no D2 owner).
    pub d2_output_bytes: usize,
    /// SUM of live D2 executor transfer up+down bytes over armed layers (0 = none).
    pub d2_transfer_bytes: usize,
    /// SUM of requested/live D2 mapped cold-return bytes over mapped layers (0 = copy).
    /// Host-mapped memory, never a device allocation; the charge is the host/GTT side.
    pub d2_mapped_bytes: usize,
    /// SUM of requested/live D2 device bytes (the `masked_indices` malloc per layer).
    /// The pinned/mapped transfer buffers above are host allocations; this field is
    /// the only device-side D2 allocation the preflight charges to device memory.
    pub d2_device_bytes: usize,
    /// Called-thread `MOE_SCRATCH` host-slot scratch pool bytes for the window (0 =
    /// legacy default). D1 and D2 share the one TLS pool; charged once per request.
    pub host_scratch_bytes: usize,
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
    /// Native MTP was requested for a profile whose weights carry no MTP head.
    NoMtpHead {
        profile: String,
    },
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
            Self::NoMtpHead { profile } => write!(
                f,
                "Qwen4 bundle: profile {profile} has no MTP head; native MTP cannot attach"
            ),
        }
    }
}

impl std::error::Error for BundleError {}

/// The chunk rows of the Full-plan re-attach after the GPU symmetry check
/// refused the compact plan: the largest rung of
/// [`qwen4_prefill_chunk_rungs`]`(requested)` whose Full forward bytes plus the
/// headroom fit `free` device bytes. When none fits, `requested.min(1536)` is
/// attempted regardless and fails at allocation if it must, as in the first
/// attach.
fn full_fallback_rows(config: &Qwen4Config, requested: usize, free: u64) -> usize {
    qwen4_prefill_chunk_rungs(requested)
        .find(|&rows| {
            qwen4_forward_device_bytes(config, rows, Qwen4ForwardScratchMode::Full)
                .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES))
                .is_some_and(|bytes| bytes <= free)
        })
        .unwrap_or_else(|| requested.min(1536))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ple::PLE_HEAD_COUNT;

    #[test]
    fn full_fallback_rows_picks_the_largest_fitting_rung() {
        let config = crate::config::compact_test_config();
        let need = |rows: usize| {
            qwen4_forward_device_bytes(&config, rows, Qwen4ForwardScratchMode::Full)
                .unwrap()
                .checked_add(QWEN4_FORWARD_HEADROOM_BYTES)
                .unwrap()
        };
        // Plenty of memory: the requested rows, even above the default rungs.
        assert_eq!(full_fallback_rows(&config, 8192, u64::MAX), 8192);
        assert_eq!(full_fallback_rows(&config, 8192, need(8192)), 8192);
        // One byte short of a rung steps down to the next one.
        assert_eq!(full_fallback_rows(&config, 8192, need(8192) - 1), 4096);
        assert_eq!(full_fallback_rows(&config, 8192, need(4096)), 4096);
        assert_eq!(full_fallback_rows(&config, 8192, need(4096) - 1), 2048);
        assert_eq!(full_fallback_rows(&config, 8192, need(2048)), 2048);
        assert_eq!(full_fallback_rows(&config, 8192, need(1536)), 1536);
        // Nothing fits: the smallest rung is attempted regardless.
        assert_eq!(full_fallback_rows(&config, 8192, need(1536) - 1), 1536);
        assert_eq!(full_fallback_rows(&config, 8192, 0), 1536);
        // A request below the smallest rung is attempted as asked.
        assert_eq!(full_fallback_rows(&config, 1024, 0), 1024);
        assert_eq!(full_fallback_rows(&config, 1024, u64::MAX), 1024);
    }

    /// The 128 canonical shard layouts of `encoding`, laid back to back from
    /// `base` (a sidecar's data section).
    fn layouts(encoding: RowEncoding, base: u64) -> Vec<ShardLayout> {
        let shard_bytes = (PLE_SHARD_ROWS * encoding.encoded_row_bytes(PLE_ROW_WIDTH)) as u64;
        (0..PLE_SHARD_COUNT)
            .map(|shard| ShardLayout {
                dtype: encoding.dtype_name().to_string(),
                logical_shape: vec![PLE_SHARD_ROWS, PLE_ROW_WIDTH],
                offset: base + shard as u64 * shard_bytes,
                length: shard_bytes,
            })
            .collect()
    }

    #[test]
    fn hot_plan_covers_q8f16_and_q4lm_sources() {
        let metadata = PleHashMetadata::qwen4();
        for (encoding, row_bytes) in [(RowEncoding::Q8F16, 170), (RowEncoding::Q4Lm, 90)] {
            let path = Path::new("/models/flash.ple-q4");
            let plan = hot_plan_from_layouts(&metadata, &layouts(encoding, 4096), path, 12_345)
                .unwrap()
                .expect("a plan for this encoding");
            assert_eq!(plan.row_encoding, encoding);
            assert_eq!(plan.encoded_row_bytes, row_bytes);
            assert_eq!(plan.row_width, PLE_ROW_WIDTH);
            assert_eq!(plan.rows_per_shard, PLE_SHARD_ROWS);
            assert_eq!(plan.padded_rows, metadata.padded_rows());
            assert_eq!(plan.shard_offsets.len(), PLE_SHARD_COUNT);
            assert_eq!(plan.shard_offsets[0], 4096);
            assert_eq!(
                plan.shard_offsets[1] - plan.shard_offsets[0],
                (PLE_SHARD_ROWS * row_bytes) as u64
            );
            assert_eq!(plan.source_path, path);
            assert_eq!(plan.source_file_bytes, 12_345);
            assert_eq!(plan.fingerprint_hex.len(), 64);
        }
    }

    #[test]
    fn hot_plan_is_off_for_bf16_and_refuses_inconsistent_shards() {
        let metadata = PleHashMetadata::qwen4();
        let path = Path::new("/models/flash.mq4-xts");
        // BF16 shards (an old artifact): no plan, not an error.
        let mut bf16 = layouts(RowEncoding::Q8F16, 0);
        for layout in &mut bf16 {
            layout.dtype = "BF16".to_string();
            layout.length = (PLE_SHARD_ROWS * 320) as u64;
        }
        assert!(hot_plan_from_layouts(&metadata, &bf16, path, 1)
            .unwrap()
            .is_none());
        // A short list, a wrong extent and a mixed encoding are errors.
        let q8 = layouts(RowEncoding::Q8F16, 0);
        assert!(hot_plan_from_layouts(&metadata, &q8[..127], path, 1).is_err());
        let mut short = q8.clone();
        short[5].length -= 1;
        let error = hot_plan_from_layouts(&metadata, &short, path, 1).unwrap_err();
        assert!(error.contains("shard 5"), "{error}");
        let mut mixed = q8.clone();
        mixed[7].dtype = "Q4LM".to_string();
        mixed[7].length = (PLE_SHARD_ROWS * 90) as u64;
        let error = hot_plan_from_layouts(&metadata, &mixed, path, 1).unwrap_err();
        assert!(error.contains("shard 7"), "{error}");
        let mut reshaped = q8;
        reshaped[0].logical_shape = vec![PLE_SHARD_ROWS, 128];
        assert!(hot_plan_from_layouts(&metadata, &reshaped, path, 1).is_err());
    }

    #[test]
    fn sidecar_layout_accepts_both_encodings_and_proves_order_by_position() {
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            let mut ok = layouts(encoding, 8192);
            assert_eq!(check_sidecar_layout(&ok).unwrap(), encoding);
            // Swapping two shards (out of order) is refused.
            ok.swap(2, 3);
            let error = check_sidecar_layout(&ok).unwrap_err();
            assert!(error.contains("shard 2"), "{error}");
        }
        let q4 = layouts(RowEncoding::Q4Lm, 0);
        // The 90-byte stride is part of the contract: a 170-byte extent under a
        // Q4LM label is refused, as is a short shard list or a BF16 sidecar.
        let mut wrong_stride = q4.clone();
        wrong_stride[0].length = (PLE_SHARD_ROWS * 170) as u64;
        assert!(check_sidecar_layout(&wrong_stride).is_err());
        assert!(check_sidecar_layout(&q4[..100]).is_err());
        let mut bf16 = q4;
        for layout in &mut bf16 {
            layout.dtype = "BF16".to_string();
        }
        let error = check_sidecar_layout(&bf16).unwrap_err();
        assert!(error.contains("expected Q8_0 or Q4LM"), "{error}");
    }

    #[test]
    fn rejects_metadata_with_same_padding_but_different_valid_rows() {
        let canonical = PleHashMetadata::qwen4();
        let mut sizes: [u64; PLE_HEAD_COUNT] = canonical.head_vocab_sizes().try_into().unwrap();
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

    use crate::config::Qwen4FfnConfig;

    fn dense_config(with_ple: bool, with_mtp: bool) -> Qwen4Config {
        crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            with_ple,
            with_mtp,
        )
    }

    fn ple_metadata_of(config: &Qwen4Config) -> PleHashMetadata {
        PleHashMetadata::from_json_for(
            &config.developer_metadata_json(),
            config.ple.as_ref().expect("a PLE profile"),
        )
        .expect("developer PLE metadata")
    }

    #[test]
    fn ple_inputs_must_agree_with_the_profile() {
        let pled = dense_config(true, true);
        let bare = dense_config(false, false);
        let metadata = ple_metadata_of(&pled);
        let sidecar = || Qwen4PleRows::Sidecar {
            path: PathBuf::from("/models/dev.ple-q8"),
            encoding: RowEncoding::Q8F16,
            rows: SidecarRows::Descriptors(Vec::new()),
        };
        // PLE <=> metadata <=> rows source.
        assert!(matches!(
            admit_ple_inputs(&pled, Some(metadata.clone()), Qwen4PleRows::Embedded),
            Ok(Some(_))
        ));
        assert!(matches!(
            admit_ple_inputs(&bare, None, Qwen4PleRows::Absent),
            Ok(None)
        ));
        let refused = |result: Result<Option<PleInputs>, BundleError>| match result {
            Ok(_) => panic!("expected a refusal"),
            Err(error) => error.to_string(),
        };
        assert!(refused(admit_ple_inputs(
            &bare,
            Some(metadata.clone()),
            Qwen4PleRows::Absent
        ))
        .contains("declares no PLE"));
        assert!(
            refused(admit_ple_inputs(&bare, None, Qwen4PleRows::Embedded))
                .contains("declares no PLE")
        );
        assert!(
            refused(admit_ple_inputs(&pled, None, Qwen4PleRows::Embedded))
                .contains("no PLE hash metadata")
        );
        assert!(refused(admit_ple_inputs(
            &pled,
            Some(metadata.clone()),
            Qwen4PleRows::Absent
        ))
        .contains("no PLE rows source"));
        // Sidecar rows are Flash-Next-only.
        assert!(
            refused(admit_ple_inputs(&pled, Some(metadata), sidecar())).contains("Flash-Next-only")
        );
        // The metadata must be this profile's geometry: the pinned Flash-Next
        // tables have 16 heads, the tiny profile 2.
        assert!(refused(admit_ple_inputs(
            &pled,
            Some(PleHashMetadata::qwen4()),
            Qwen4PleRows::Embedded
        ))
        .contains("head tables"));
    }

    #[test]
    fn flash_next_geometry_checks_equal_the_pinned_constants() {
        let config = crate::config::compact_test_config();
        let ple = config.ple.as_ref().expect("Flash-Next declares PLE");
        assert_eq!(
            (
                ple.shard_count,
                ple.shard_rows,
                ple.row_width,
                ple.head_count
            ),
            (
                PLE_SHARD_COUNT,
                PLE_SHARD_ROWS,
                PLE_ROW_WIDTH,
                PLE_HEAD_COUNT
            )
        );
        let metadata = PleHashMetadata::qwen4();
        validate_ple_geometry(ple, &metadata)
            .expect("the pinned metadata fits the pinned geometry");
        validate_ple_metadata(&metadata).expect("the Flash-Next sidecar/hot check agrees");
        assert!(matches!(
            admit_ple_inputs(&config, Some(metadata.clone()), Qwen4PleRows::Embedded),
            Ok(Some(_))
        ));
        // A different padded row count fails both checks the same way.
        let mut other = ple.clone();
        other.shard_rows -= 1;
        assert!(validate_ple_geometry(&other, &metadata).is_err());
        // The shard tensors are the pinned layer-1 names.
        assert_eq!(
            ple_shard_name_prefix(ple),
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
        );
        let manifest = Qwen4Manifest::build(&config).expect("FN manifest");
        let entries = ordered_ple_entries(&manifest, ple).expect("128 ordered shards");
        assert_eq!(entries.len(), PLE_SHARD_COUNT);
        assert!(entries.iter().all(|entry| entry.layer == Some(1)
            && entry.logical_shape == [PLE_SHARD_ROWS, PLE_ROW_WIDTH]));
    }

    #[test]
    fn synthetic_shards_follow_the_profile_geometry() {
        let config = dense_config(true, true);
        let ple = config.ple.as_ref().expect("PLE profile");
        let prefix = ple_shard_name_prefix(ple);
        assert!(
            prefix.contains(&format!(".layers.{}.ple.", ple.layer)),
            "{prefix}"
        );
        let name = |index: usize| format!("{prefix}{index}.weight");
        assert_eq!(ple_shard_index(ple, &name(0)).unwrap(), 0);
        assert!(ple_shard_index(ple, &name(ple.shard_count)).is_err());
        assert!(ple_shard_index(ple, &format!("{prefix}00.weight")).is_err());
        let manifest = Qwen4Manifest::build(&config).expect("synthetic manifest");
        let entries = ordered_ple_entries(&manifest, ple).expect("ordered synthetic shards");
        assert_eq!(entries.len(), ple.shard_count);
        for entry in &entries {
            assert_eq!(entry.layer, Some(ple.layer));
            assert_eq!(entry.logical_shape, [ple.shard_rows, ple.row_width]);
        }
        validate_ple_geometry(ple, &ple_metadata_of(&config)).expect("synthetic metadata");
        // The staged prefetch width comes from the profile, not the pinned 16.
        assert_eq!(ple.head_count, 2);
    }

    #[test]
    fn session_snapshots_omit_the_absent_ple_parts() {
        use crate::state::Qwen4StateFormat;
        let with = dense_config(true, true);
        let without = dense_config(false, true);
        let format = Qwen4StateFormat::F32;
        let part = |bytes: usize| bytes.next_multiple_of(256);
        let ple_bytes = with.ple_conv_history_rows() * with.hidden_size * with.hc_count * 4;
        assert!(ple_bytes > 0 && without.ple_conv_history_rows() == 0);
        for p in [1usize, 3, 8] {
            let with_ple = session_snapshot_bytes(&with, format, false, p).expect("with PLE");
            let no_ple = session_snapshot_bytes(&without, format, false, p).expect("without PLE");
            assert_eq!(with_ple - no_ple, part(ple_bytes) as u64, "p={p}");
        }
    }

    #[test]
    fn side_request_parking_cost_is_fixed_parts_plus_bounded_rows() {
        use rdna_compute::tensor_ops::{GdnStateFormat, QsaKvFormat};
        let config = crate::config::compact_test_config();
        // The measured chain used Q8 GDN; F32 below refers to QSA rows.
        for (qsa, row_bytes) in [(QsaKvFormat::Q8, 21_632u64), (QsaKvFormat::F32, 61_568)] {
            let format = Qwen4StateFormat {
                qsa,
                gdn: GdnStateFormat::Q8,
                ..Qwen4StateFormat::F32
            };
            let fixed = session_snapshot_bytes(&config, format, true, 0).unwrap();
            // The 34,180,608 B of fixed parts before the head's snapshot also
            // kept the target boundary-hidden row (one 40,960 B part).
            assert_eq!(fixed, 34_180_608 + 40_960);
            let bounded = session_snapshot_bytes(&config, format, true, 8192).unwrap();
            assert_eq!(bounded, fixed + 8192 * row_bytes);
        }
        let full_f32 = session_snapshot_bytes(&config, Qwen4StateFormat::F32, true, 0).unwrap();
        assert_eq!(full_f32, 118_230_528 + 40_960);
    }

    fn precision_state_format(gdn: rdna_compute::tensor_ops::GdnStateFormat) -> Qwen4StateFormat {
        Qwen4StateFormat {
            gdn,
            ..Qwen4StateFormat::F32
        }
    }

    /// The session-cache scope (the cache's key domain for every snapshot) is
    /// qualified by the full state format, so a Q8-state snapshot is never
    /// planned, matched or restored into an F32-state bundle, or the reverse.
    #[test]
    fn gdn_precision_owner_accounting_session_scope_is_format_qualified() {
        use rdna_compute::tensor_ops::GdnStateFormat;
        let q8 = precision_state_format(GdnStateFormat::Q8);
        let f32 = precision_state_format(GdnStateFormat::F32);
        for route in [SessionRoute::Ar, SessionRoute::Mtp] {
            for chunk in [256usize, 4096] {
                for turns in [false, true] {
                    let scope = |format| session_scope(route, chunk, format, turns, None);
                    let (q8_scope, f32_scope) = (scope(q8), scope(f32));
                    assert_ne!(q8_scope, f32_scope, "{route:?} chunk {chunk} turns {turns}");
                    assert!(q8_scope.contains("gdn: Q8") && !q8_scope.contains("gdn: F32"));
                    assert!(f32_scope.contains("gdn: F32") && !f32_scope.contains("gdn: Q8"));
                    // The GDN format is the only difference.
                    assert_eq!(
                        q8_scope.replace("gdn: Q8", "gdn: F32"),
                        f32_scope,
                        "{route:?} chunk {chunk} turns {turns}"
                    );
                    // Same format, same scope.
                    assert_eq!(scope(q8), q8_scope);
                    assert_eq!(scope(f32), f32_scope);
                }
            }
        }
    }

    /// A snapshot's fixed parts have the live state's byte lengths. Q8 and F32
    /// recurrent parts differ in length, so the restore's per-part length check
    /// (`hipfire-runtime` `session_cache`: "snapshot layout mismatch") refuses
    /// one format's snapshot in the other format's state; no code converts it.
    #[test]
    fn gdn_precision_owner_accounting_session_entries_are_not_interchangeable() {
        use rdna_compute::tensor_ops::GdnStateFormat;
        let config = crate::config::compact_test_config();
        let q8 = precision_state_format(GdnStateFormat::Q8);
        let f32 = precision_state_format(GdnStateFormat::F32);
        let recurrent_delta = (config.n_linear_layers() * (3_145_728 - 811_008)) as u64;
        for native in [false, true] {
            for p in [0usize, 1, 1536, 4096] {
                let q8_bytes = session_snapshot_bytes(&config, q8, native, p).unwrap();
                let f32_bytes = session_snapshot_bytes(&config, f32, native, p).unwrap();
                assert_eq!(
                    f32_bytes - q8_bytes,
                    recurrent_delta,
                    "native {native} p {p}"
                );
            }
        }
    }

    #[test]
    fn a_head_free_profile_refuses_native_mtp_with_a_typed_error() {
        assert!(Qwen4Bundle::require_mtp_head("tiny-dense", true).is_ok());
        match Qwen4Bundle::require_mtp_head("tiny-dense", false) {
            Err(error @ BundleError::NoMtpHead { .. }) => {
                let text = error.to_string();
                assert!(
                    text.contains("tiny-dense") && text.contains("no MTP head"),
                    "{text}"
                );
            }
            other => panic!("expected NoMtpHead, got {other:?}"),
        }
    }

    #[test]
    fn executable_profile_admission_runs_before_any_allocation() {
        // Flash-Next and a dense developer profile (with the kernel RoPE base)
        // are executable, with and without PLE and MTP.
        admit_executable_profile(&crate::config::compact_test_config()).expect("Flash-Next");
        for (with_ple, with_mtp) in [(true, true), (false, false), (true, false), (false, true)] {
            let mut config = dense_config(with_ple, with_mtp);
            config.rope_theta = crate::program::QWEN4_QSA_KERNEL_ROPE_THETA;
            admit_executable_profile(&config)
                .unwrap_or_else(|error| panic!("dense ple={with_ple} mtp={with_mtp}: {error}"));
        }
        // A RoPE base the QSA kernels do not implement is the forward's refusal.
        let error = admit_executable_profile(&dense_config(true, true)).unwrap_err();
        assert!(matches!(error, BundleError::Forward(_)), "{error}");
        // A synthetic MoE has no sealed route: the weights refusal text.
        let moe = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Moe {
                num_experts: 8,
                top_k: 2,
                intermediate_size: 64,
                shared_intermediate_size: Some(64),
                norm_topk_prob: true,
            },
            256,
            4,
            true,
            true,
        );
        let error = admit_executable_profile(&moe).unwrap_err();
        assert!(matches!(error, BundleError::Weights(_)), "{error}");
        assert!(error.to_string().contains("no sealed MoE route"), "{error}");
    }
}

/// The correctness twin retains one-chunk capture; only actual NPU pipelining
/// needs simultaneous A/B hidden rows.
pub(crate) fn spec_hidden_capacity_rows(
    rows: usize,
    chunk: usize,
    pipelined: bool,
) -> Option<usize> {
    if pipelined {
        Some(rows.max(chunk.checked_mul(2)?))
    } else {
        Some(rows)
    }
}
