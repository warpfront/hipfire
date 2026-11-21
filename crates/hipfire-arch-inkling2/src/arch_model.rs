// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` bundle + skeleton state for Inkling-2-Mini (arch_id = 24).
//! Host-side config/weight index plus the GPU bundle (brief 01): VRAM
//! weights, per-visit KV/conv state, and a correct free path.

use crate::config::Inkling2Config;
use crate::gpu::{Inkling2GpuState, Inkling2GpuWeights};
use crate::weights::Inkling2Weights;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

/// Skeleton decode state: cursor only. The per-visit GPU state lives in
/// [`Inkling2GpuState`]; `reset_session_state` already resets both shapes.
#[derive(Default)]
pub struct Inkling2State {
    pub seq_pos: usize,
}

/// Owned bundle: config + host-side weight index + GPU weights/state + eos.
/// The host `weights` index is retained (brief 04/06 own `weights.rs`, which
/// has no drain API yet — see the carrier note); the GPU side owns VRAM only.
pub struct Inkling2Bundle {
    pub config: Inkling2Config,
    pub weights: Inkling2Weights,
    pub state: Inkling2State,
    pub eos_tok: u32,
    pub gpu_weights: Option<Inkling2GpuWeights>,
    pub gpu_state: Option<Inkling2GpuState>,
}

impl ArchModel for Inkling2Bundle {
    fn dim(&self) -> usize {
        self.config.n_embd
    }

    fn n_layers(&self) -> usize {
        // Visit slots (48 on Mini), not physical layers (24).
        self.config.loop_cfg.n_slot()
    }

    fn vocab_size(&self) -> usize {
        self.config.unpadded_vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "inkling2"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.seq_pos = 0;
        if let Some(gs) = self.gpu_state.as_mut() {
            gs.reset_session_state(gpu)?;
        }
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(gw) = self.gpu_weights {
            gw.free_gpu(gpu);
        }
        if let Some(gs) = self.gpu_state {
            gs.free_gpu(gpu);
        }
        // Host-side weight index only — drop is the whole teardown.
        drop(self.weights);
    }
}
