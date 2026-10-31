// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for the LFM3-VL vision tower.
//!
//! Mirrors `hipfire-arch-qwen35-vl::arch`: trait-routed bring-up triple,
//! direct static calls for forward dispatch.
//!
//! Scope
//! -----
//! LFM3-VL is "LFM3 text decoder + LFM3-VL vision tower + projector". The text
//! path is owned by `hipfire-arch-lfm3::Lfm3` (arch_id 13) and loaded through
//! THAT trait impl. This impl covers ONLY the vision-tower + projector bring-up
//! (config + weights). A VL model load in `daemon.rs` runs both:
//!
//!   let text_cfg = <Lfm3   as Architecture>::config_from_hfq(&hfq)?;
//!   let vis_cfg  = <Lfm3Vl as Architecture>::config_from_hfq(&hfq)?;
//!   let text_w   = <Lfm3   as Architecture>::load_weights(&mut hfq, &text_cfg, gpu)?;
//!   let vis_w    = <Lfm3Vl as Architecture>::load_weights(&mut hfq, &vis_cfg, gpu)?;
//!
//! `State = ()` — the vision tower is stateless one-shot (encode N patches →
//! N/4 visual tokens, splice, free). The text decoder's conv/KV state is owned
//! by the `Lfm3` arch impl.

use crate::lfm3_vl::{load_vision_weights, vision_config_from_hfq, VisionConfig, VisionWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for LFM3-VL (vision-language). Loads the LFM3-VL vision tower
/// + multi-modal projector; pair with `hipfire-arch-lfm3::Lfm3` for the text
/// decoder side.
pub struct Lfm3Vl;

impl Architecture for Lfm3Vl {
    type Weights = VisionWeights;
    type State = ();
    type Config = VisionConfig;

    fn arch_id() -> u32 {
        // LFM3-VL ships under arch_id 13 (same as the LFM3 text backbone —
        // the VL model is LFM3 text + vision tower). Vision is detected by
        // tensor presence (`model.vision_tower.patchifier.weight`), matching
        // the qwen35-vl precedent where arch_id stays 5 for the VL variant.
        13
    }

    fn name() -> &'static str {
        "lfm3-vl"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        vision_config_from_hfq(hfq)
            .ok_or_else(|| "lfm3-vl: vision_config not found in HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        load_vision_weights(hfq, cfg, gpu)
            .map_err(|e| format!("lfm3-vl: load_vision_weights failed: {e:?}"))
    }

    fn new_state(_gpu: &mut Gpu, _cfg: &Self::Config) -> Result<Self::State, String> {
        // Vision tower is stateless one-shot.
        Ok(())
    }
}
