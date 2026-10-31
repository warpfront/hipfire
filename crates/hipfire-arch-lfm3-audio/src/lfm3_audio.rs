// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Audio speech encoder + projector (Phase 3 — implementation in progress).

use hipfire_runtime::hfq::HfqFile;
use hip_bridge::HipResult;
use rdna_compute::Gpu;

/// Audio encoder + projector config (parsed from the HFQ `config` envelope).
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub hidden_size: usize,
}

/// Placeholder — full weights land with the encoder forward.
pub struct AudioWeights;

pub fn audio_config_from_hfq(_hfq: &HfqFile) -> Option<AudioConfig> {
    None
}

pub fn load_audio_weights(_hfq: &HfqFile, _cfg: &AudioConfig, _gpu: &mut Gpu) -> HipResult<AudioWeights> {
    Ok(AudioWeights)
}

pub fn audio_forward(
    _gpu: &mut Gpu,
    _weights: &AudioWeights,
    _cfg: &AudioConfig,
    _mel: &[f32],
    _num_frames: usize,
) -> HipResult<Vec<f32>> {
    Ok(Vec::new())
}
