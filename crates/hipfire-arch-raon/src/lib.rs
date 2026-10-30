// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini-preview (arch_id 19, `raon1_mini`): 25-layer hybrid of
//! RDA (vector-decay gated delta) and NoPE MLA mixers, four-stream Raon mHC,
//! SiTU dense/shared FFN and sigmoid top-6 MoE. Generic math lives in
//! rdna-compute/hipfire-dispatch; this crate binds Raon parameters.
pub mod arch_model;
pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod prefill;
pub mod projection;
pub mod rda;
pub mod state;
pub mod weights;

/// Architecture id for Raon-1-mini.
pub const ARCH_ID: u32 = 19;

pub use bundle::{DecodeMode, PrefillMode, RaonBundle, PREFILL_SCRATCH_ROWS};
pub use carrier::load_raon_bundle;
pub use config::{FfnKind, GenerationDefaults, LayerKind, RaonConfig};
pub use forward::{
    forward_rows, forward_rows_tapped, forward_token, forward_token_tapped, readout_logits,
    LayerTap, RaonWork, TapSite,
};
pub use projection::{project, project_row, project_rows, round_bf16_rows, OutRound};
pub use weights::{
    check_inventory, load_raon_weights, tensor_inventory, ExpertWeights, GatedFfnWeights,
    MhcSiteWeights, MlaWeights, MoeFfnWeights, RaonFfn, RaonLayerWeights, RaonMixer, RaonWeights,
    RdaWeights, TensorClass, TensorSpec,
};
