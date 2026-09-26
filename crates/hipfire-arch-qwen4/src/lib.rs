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

pub mod admission;
pub mod artifact;
pub mod bundle;
pub mod config;
pub mod gpu_forward;
pub mod mtp_gpu;
pub mod mtp_spec;
pub mod ops;
pub mod ple;
pub mod program;
pub(crate) mod projection;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_forward;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_mtp;
pub mod state;
#[cfg(any(test, feature = "reference-parity"))]
pub mod state_parity;
pub mod weights;

pub use admission::{
    AdmissionError, ArtifactFormat, EffectiveMesh, InputModality, MeshShape, QuantFormat,
    QuantGeometry, QuantTensorGeometry, Qwen4Admission, Qwen4AdmissionError, Qwen4AdmissionRequest,
    Qwen4Artifact, Qwen4ArtifactFormat, Qwen4Capabilities, Qwen4Capability, Qwen4QuantGeometry,
    RequestKind,
};
pub use artifact::{admit_hfqm_artifact, Qwen4ArtifactError, Qwen4HfqmArtifact};
pub use config::{
    LayerType, Qwen4Config, Qwen4MtpConfig, RecurrentStateDType, SourceDType, ARCHITECTURE_NAME,
    ARCH_ID, MODEL_TYPE, TEXT_MODEL_TYPE,
};
pub use ple::{
    PleHashMetadata, PleHistory, PleMetadataError, PleRowId, PLE_HEAD_COUNT, PLE_HEAD_OFFSETS,
    PLE_HEAD_VOCAB_SIZES, PLE_MULTIPLIERS, PLE_MULTIPLIER_COUNT, PLE_PADDED_ROWS,
    PLE_PADDING_MULTIPLE, PLE_ROW_WIDTH, PLE_VALID_ROWS,
};
