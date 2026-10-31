// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait placeholder for the LFM3-Omni depth decoder. The depth
//! decoder is not a standalone Architecture (it rides on arch_id 13 alongside
//! the LFM3 text backbone, detected by `model.depth_decoder.*` tensor
//! presence). This module exists for crate symmetry with the sibling
//! hipfire-arch-lfm3-{vl,audio} crates; the real entry points are the free
//! functions in `depth`.

#![allow(dead_code)]

use crate::depth::{depth_config_from_hfq, load_depth_weights, DepthConfig, DepthWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

pub struct Lfm3Omni;

impl Architecture for Lfm3Omni {
    type Weights = DepthWeights;
    type State = ();
    type Config = DepthConfig;

    fn arch_id() -> u32 {
        13
    }
    fn name() -> &'static str {
        "lfm3-omni"
    }
    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        depth_config_from_hfq(hfq)
            .ok_or_else(|| "lfm3-omni: depth_config not found in HFQ metadata".to_string())
    }
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        load_depth_weights(hfq, cfg, gpu)
            .map_err(|e| format!("lfm3-omni: load_depth_weights failed: {e:?}"))
    }
    fn new_state(_gpu: &mut Gpu, _cfg: &Self::Config) -> Result<Self::State, String> {
        Ok(())
    }
}
