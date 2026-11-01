// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 / Qwen3.8-Flash-Next architecture foundations.
//!
//! Product registration remains fail-closed until the model's lifecycle
//! milestones are accepted. This crate owns strict configuration and
//! admission, typed weights and GPU state, bounded PLE storage, and native MTP
//! transactions. CPU reference equations are opt-in parity fixtures.
#![recursion_limit = "256"]

/// Producer capability: native Qwen4 MTP ships a measured autoregressive floor
/// (`mtp_spec::MtpFloor`: per-request AR-vs-draft window pricing that retires
/// drafting when it loses to AR, on by default). `speculation.mtp = auto` arms
/// the head only for a producer declaring this; explicit `on` always arms.
pub const MTP_MEASURED_AR_FLOOR: bool = true;

pub mod admission;
pub mod artifact;
pub mod bundle;
#[cfg(any(test, feature = "reference-parity"))]
pub mod cache_quality;
pub mod chunk_dump;
pub mod config;
/// Per-load default of the D2 CPU tier split (measured engine-rate selector).
pub mod d2_default;
/// Decode-share developer knobs, mode enum and load admission.
pub mod decode_share;
/// Donor-tail loan state machine of decode-share (`fixed` / `on`).
pub(crate) mod decode_share_loan;
/// Layout of the weight-shadow region borrowed inside the donor tail.
pub mod decode_share_shadows;
/// Target side of the shared DFlash2 drafter (layer capture, embed, head).
pub(crate) mod dflash_target;
pub mod ep;
/// EP2 reserve decoupling (opt-in `HIPFIRE_EP_RESERVE_DECOUPLE=1`): pure
/// admission, plan, packing and receipt types.
pub mod ep_reserve;
pub mod ep_seal;
pub mod ep_trunk_split;
pub mod expert_keep;
pub use expert_keep::{ExpertKeep, EXPERT_KEEP_METADATA_KEY, EXPERT_KEEP_SCHEMA};
/// Per-layer contiguous expert slab arenas (K4 loader plan + fulfillment payload).
pub mod expert_arena;
/// Lever1 expert hotset sidecar parser and `HIPFIRE_QWEN4_EXPERT_HOTSET` policy.
pub mod expert_hotset;
pub mod expert_residency;
/// Aggregate MQ4-XTS expert tensors to per-expert slab spans (B5-C1 adapter).
pub mod expert_slab_aggregate;
/// Validated CPU-only per-expert payload spans and optional scale validation.
pub mod expert_slab_table;
/// Version-1 expert-slab container schema (qt=131..=134 / `(44,53)` per-expert entries).
pub mod expert_slabs;
pub mod expert_tier_owners;
/// Pure per-expert device/host-mapped placement plan (B5-C1).
pub mod expert_tier_plan;
pub mod gpu_forward;
pub mod gpu_forward_lanes;
/// Measured host-to-device link class (narrow-link G2 and chunk defaults).
pub mod host_link;
pub mod kv_backend;
/// Read-only lab observation of the production sampled MTP accepting window.
#[cfg(feature = "lab")]
pub mod lab_window;
pub mod lane;
pub mod lane_exec;
pub(crate) mod lane_token_ring;
pub mod measured_identity;
pub mod measured_resources;
pub mod mesh_bindings;
pub mod mesh_tp;
pub mod mtp_gpu;
pub mod mtp_lanes;
pub mod mtp_spec;
pub mod npu_trunk;
pub mod ops;
pub mod ple;
pub(crate) mod ple_stage;
pub mod program;
pub(crate) mod projection;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_forward;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_mtp;
pub mod slowlink_admission;
/// Admission-first single-device load shared by the S1 validation tools ONLY
/// (gated `test` / `lab` / `reference-parity`): the product carrier never
/// routes through it, and it stays compiled out of product builds.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
pub mod slowlink_load;
mod slowlink_prefill;
pub mod slowlink_sp;
pub mod state;
#[cfg(any(test, feature = "reference-parity"))]
pub mod state_parity;
pub mod sym_verdict_cache;
pub mod trunk_overlay;
pub mod weights;
#[cfg(feature = "reference-parity")]
pub mod wide_verify_oracle;

pub use admission::{
    AdmissionError, ArtifactFormat, EffectiveMesh, InputModality, MeshShape, QuantFormat,
    QuantGeometry, QuantTensorGeometry, Qwen4Admission, Qwen4AdmissionError, Qwen4AdmissionRequest,
    Qwen4Artifact, Qwen4ArtifactFormat, Qwen4Capabilities, Qwen4Capability, Qwen4QuantGeometry,
    RequestKind,
};
#[cfg(feature = "qwen4-dev-profiles")]
pub use artifact::admit_developer_inventory_headers;
pub use artifact::{
    admit_hfqm_artifact, admit_hfqm_artifact_on, admit_inventory_headers, Qwen4ArtifactError,
    Qwen4HfqmArtifact, Qwen4InventoryReceipt, Qwen4PleRows, SidecarRows,
};
pub use config::{
    ple_head_vocab_sizes, LayerType, Qwen4Config, Qwen4EquationSource, Qwen4FfnConfig,
    Qwen4MtpConfig, Qwen4PleConfig, Qwen4Profile, Qwen4SyntheticProfile, RecurrentStateDType,
    SourceDType, ARCHITECTURE_NAME, ARCH_ID, MODEL_TYPE, QWEN4_DEFAULT_CONTEXT, QWEN4_MAX_CONTEXT,
    TEXT_MODEL_TYPE,
};
pub use gpu_forward::{
    ep_reserve_receipt_scope, qwen4_compact_scratch_possible, qwen4_forward_device_bytes,
    qwen4_forward_scratch_mode, Qwen4ForwardScratchMode, QWEN4_COMPACT_FALLBACK_ROWS,
    QWEN4_COMPACT_SCRATCH_ENV,
};
pub use gpu_forward_lanes::{WaveHead, WaveLane};
pub use kv_backend::{qwen4_vmm_refusal, qwen4_vmm_supported, Qwen4ContextCommit, Qwen4KvBackend};
pub use ple::{
    PleHashMetadata, PleHistory, PleMetadataError, PleRowId, PLE_HEAD_COUNT, PLE_HEAD_OFFSETS,
    PLE_HEAD_VOCAB_SIZES, PLE_MAX_HEAD_COUNT, PLE_MULTIPLIERS, PLE_MULTIPLIER_COUNT,
    PLE_PADDED_ROWS, PLE_PADDING_MULTIPLE, PLE_ROW_WIDTH, PLE_VALID_ROWS,
};
pub use rdna_compute::tensor_ops::{GdnStateFormat, QsaKvFormat};
pub use state::{resolve_gdn_format, resolve_qsa_format, resolve_state_format, Qwen4StateFormat};
pub use trunk_overlay::{resolve_trunk_overlay, Qwen4TrunkOverlay, TrunkOverlayPolicy};
pub use weights::{Qwen4ManifestIdentity, Qwen4TierRecipe};
