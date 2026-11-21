// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` bundle + skeleton state for Inkling-2-Mini (arch_id = 24).
//! Host-side only: the GPU forward and per-visit KV/conv state attach later.

use crate::config::Inkling2Config;
use crate::weights::Inkling2Weights;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

/// Skeleton decode state: cursor only. Per-visit KV/conv state lands here with
/// the GPU port; `reset_session_state` already resets the full shape.
#[derive(Default)]
pub struct Inkling2State {
    pub seq_pos: usize,
}

/// Owned skeleton bundle: config + host-side weight index + stub state + eos.
pub struct Inkling2Bundle {
    pub config: Inkling2Config,
    pub weights: Inkling2Weights,
    pub state: Inkling2State,
    pub eos_tok: u32,
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

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.seq_pos = 0;
        Ok(())
    }

    fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {
        // Host-side weight index only — drop is the whole teardown.
        drop(self);
    }
}
