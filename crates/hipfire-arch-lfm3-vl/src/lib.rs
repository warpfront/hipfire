// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-VL (vision-language) architecture for hipfire — Phase 2 of the LFM3
//! port (`docs/plans/2026-10-31-lfm3-port.md`).
//!
//! Wraps the LFM3 text decoder (`hipfire-arch-lfm3`, arch_id 24) with the
//! LFM3-VL vision tower + multi-modal projector. The vision tower is a hybrid
//! stack of 14 `Lfm3VlVisionLayer`s — gated depthwise 2D conv layers and
//! bidirectional GQA attention layers with axial 2D RoPE — exactly the HF
//! `Lfm3VlVisionModel` in transformers `modeling_lfm3_vl.py`.
//!
//! Layout mirrors `hipfire-arch-qwen35-vl`:
//!   - `lfm3_vl` — vision tower (config, weights, GPU forward) + projector.
//!   - `arch`    — `Architecture` trait impl for the vision bring-up triple.
//!
//! Image preprocessing (smart-resize + patch extraction) is byte-identical to
//! Qwen3.5-VL's (same patch_size=16 / spatial_merge_size=2 / temporal=2 /
//! `pixel/127.5 - 1` normalization / merge-grouped patchify), so the daemon
//! reuses `hipfire_arch_qwen35_vl::image` for the CPU preprocessing pipeline
//! and feeds the extracted patches to [`lfm3_vl::vision_forward`].
//!
//! Forward dispatch is static (not trait-routed), matching the project
//! convention: `daemon.rs` calls `lfm3_vl::vision_forward` and
//! `lfm3::forward::decode_step` / `decode_step_with_embed` directly.

#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod lfm3_vl;

#[cfg(feature = "deltanet")]
pub use arch::Lfm3Vl;
#[cfg(feature = "deltanet")]
pub use lfm3_vl::{
    load_vision_weights, vision_config_from_hfq, vision_forward, ProjectorConfig, VisionConfig,
    VisionWeights,
};
