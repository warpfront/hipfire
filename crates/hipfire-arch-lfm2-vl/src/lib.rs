// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2-VL vision stack (SigLIP2-NaFlex tower + `multi_modal_projector`)
//! for the arch-11 carrier. Spec:
//! `docs/specs/2026-08-27-lfm2-vl-vision-runtime.md`; parent artifact
//! recipe `docs/lfm2-vl-mq4v2-spec.md` §3.3–3.4.
//!
//! Layout mirrors `hipfire-arch-qwen35-vl`: config + weights + GPU forward
//! here; per-arch image preprocessing in [`image`]; the text decoder stays
//! owned by `hipfire-arch-lfm2moe` (this crate never depends on it).

pub mod config;
pub mod image;
pub mod vision;

pub use config::{vision_config_from_hfq, VisionConfig};
pub use image::{load_and_preprocess, load_and_preprocess_from_bytes, Prepared};
pub use vision::{load_vision_weights, vision_forward, VisionWeights};
