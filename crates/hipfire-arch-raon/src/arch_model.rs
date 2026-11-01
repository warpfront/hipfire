// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! `ArchModel` for the Raon bundle (metadata, reset, map-ahead, free). Owner: slice A.
//!
//! Infrastructure only: generation composes the forward pass by downcasting
//! to [`RaonBundle`], exactly like the other arch bundles.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::session_cache::SessionRoute;
use rdna_compute::Gpu;

use crate::bundle::RaonBundle;
use crate::mesh::RaonMeshModel;

impl ArchModel for RaonBundle {
    fn dim(&self) -> usize {
        self.config.hidden
    }

    fn n_layers(&self) -> usize {
        self.weights.band.total
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
    /// caches, so the next turn starts from an empty context; the session
    /// driver forgets its live state, whose rows this overwrites.
    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)
    }

    fn session_cache_attached(&self) -> bool {
        self.session_cache().is_some()
    }

    /// Prompt tokens up to `cap` the next prefill can restore (cold-exact: a
    /// snapshot on the prefill chunk grid); 0 without a cache.
    fn session_plan_capped(&self, prompt: &[u32], route: SessionRoute, cap: usize) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| session.plan_capped(self, prompt, route, cap))
    }

    fn session_plan_is_live(&self, prompt: &[u32], route: SessionRoute, reused: usize) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.plan_is_live(self, prompt, route, reused))
    }

    fn session_commit(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.commit();
        }
    }

    /// Grow the context-sized state ahead of a decode that ends at
    /// `position + remaining_new_tokens + max_draft_rows`, session snapshots
    /// yielding first (this turn's prefill captures may have taken memory the
    /// decode needs). Best effort: a failure is logged and left to on-demand
    /// growth.
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
        self.release_session_for_growth(gpu, rows);
        if let Err(e) = self.state.ensure_capacity(gpu, rows) {
            eprintln!("raon1_mini: map-ahead to {rows} rows failed (growing on demand): {e}");
        }
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        (*self).free_all(gpu);
    }
}

/// The pipeline mesh: whole-model scalars from the band views, and a
/// lifecycle over the load's [`Gpus`] (one band per rank device). The
/// single-device hooks refuse or leak by design: a mesh is reset, grown and
/// freed only through the `*_mesh` hooks, which the loader and generation
/// layer call because [`owns_mesh_lifecycle`](ArchModel::owns_mesh_lifecycle)
/// is true.
impl ArchModel for RaonMeshModel {
    fn dim(&self) -> usize {
        self.dims().0
    }

    fn n_layers(&self) -> usize {
        self.dims().1
    }

    fn vocab_size(&self) -> usize {
        self.dims().2
    }

    fn arch_key(&self) -> &'static str {
        "raon1_mini"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        Err("raon1_mini: a pipeline mesh resets over its mesh devices, not the daemon GPU".into())
    }

    fn owns_mesh_lifecycle(&self) -> bool {
        true
    }

    /// Drain the mesh, then zero every band's recurrent state and rewind its
    /// latent cursor on its own device ([`RaonMeshModel::reset`]).
    fn reset_session_state_mesh(&mut self, _gpu: &mut Gpu, gpus: &mut Gpus) -> Result<(), String> {
        self.reset(gpus)
    }

    fn map_ahead_for_decode_mesh(
        &mut self,
        _gpu: &mut Gpu,
        gpus: &mut Gpus,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        let rows = position
            .saturating_add(remaining_new_tokens)
            .saturating_add(max_draft_rows);
        if let Err(e) = self.map_ahead(gpus, rows) {
            eprintln!("raon1_mini: mesh map-ahead to {rows} rows failed (growing on demand): {e}");
        }
    }

    /// Without the mesh devices a band cannot be freed on its own device and
    /// nothing proves the edges drained: quarantine (leak) instead.
    fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {
        eprintln!("raon1_mini: pipeline mesh freed without its devices; leaking the bands");
        std::mem::forget(self);
    }

    fn free_gpu_mesh(self: Box<Self>, _gpu: &mut Gpu, gpus: &mut Gpus) -> Result<(), String> {
        (*self).free(gpus)
    }
}
