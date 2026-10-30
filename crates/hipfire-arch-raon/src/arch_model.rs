// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! `ArchModel` for the Raon bundle (metadata, reset, map-ahead, free). Owner: slice A.
//!
//! Infrastructure only: generation composes the forward pass by downcasting
//! to [`RaonBundle`], exactly like the other arch bundles.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::bundle::RaonBundle;

impl ArchModel for RaonBundle {
    fn dim(&self) -> usize {
        self.config.hidden
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab
    }

    fn arch_key(&self) -> &'static str {
        "raon1_mini"
    }

    /// Raon keeps its context in per-layer MLA latent rows and RDA recurrent
    /// state inside [`RaonState`], not a `KvCache`: `None` is legitimate.
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    /// Zero the RDA recurrent state and conv histories and rewind the latent
    /// caches, so the next turn starts from an empty context.
    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    /// Grow the context-sized state ahead of a decode that ends at
    /// `position + remaining_new_tokens + max_draft_rows`. Best effort: a
    /// failure is logged and left to on-demand growth.
    fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        let rows = position
            .saturating_add(remaining_new_tokens)
            .saturating_add(max_draft_rows);
        if let Err(e) = self.state.ensure_capacity(gpu, rows) {
            eprintln!("raon1_mini: map-ahead to {rows} rows failed (growing on demand): {e}");
        }
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        (*self).free_all(gpu);
    }
}
