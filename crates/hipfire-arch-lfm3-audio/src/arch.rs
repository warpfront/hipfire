// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for the LFM3-Audio speech encoder.
//! Placeholder — full impl lands with the encoder forward.

use crate::lfm3_audio::{AudioConfig, AudioWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

pub struct Lfm3Audio;

impl Architecture for Lfm3Audio {
    type Weights = AudioWeights;
    type State = ();
    type Config = AudioConfig;

    fn arch_id() -> u32 {
        13
    }
    fn name() -> &'static str {
        "lfm3-audio"
    }
    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        crate::lfm3_audio::audio_config_from_hfq(hfq)
            .ok_or_else(|| "lfm3-audio: audio_config not found in HFQ metadata".to_string())
    }
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        crate::lfm3_audio::load_audio_weights(hfq, cfg, gpu)
            .map_err(|e| format!("lfm3-audio: load_audio_weights failed: {e:?}"))
    }
    fn new_state(_gpu: &mut Gpu, _cfg: &Self::Config) -> Result<Self::State, String> {
        Ok(())
    }
}
