// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The architecture-agnostic view of a loaded model.
//!
//! ## Why this exists
//!
//! `hipfire_loader::ModelState` is a closed enum with one variant per
//! architecture. Every crate that needed a scalar off a loaded model therefore
//! had to name all eleven — including the product entry point, which computed
//! *three integers* through a seven-arm architecture dispatch:
//!
//! ```text
//! let (dim, layers, vocab) = match m.state.as_ref() {
//!     Some(ModelState::Qwen35(b)) => (b.config.dim,         b.config.n_layers,          b.config.vocab_size),
//!     Some(ModelState::Qwen2(b))  => (b.config.hidden_size, b.config.num_hidden_layers, b.config.vocab_size),
//!     …
//! ```
//!
//! Seven arms, one tuple, and the only real difference is that some configs
//! spell it `dim`/`n_layers` and others `hidden_size`/`num_hidden_layers`. A
//! naming inconsistency was being paid for with architecture dispatch in the
//! daemon.
//!
//! Measured before this trait existed: the loader and daemon between them held
//! 93 `ModelState::` references, but touched only **seven distinct members** of
//! the bundles they unwrapped — `config` (26 hits, only ever for those three
//! scalars), `state`, `reset_session_state`, `kv_cache`/`kv`, `dn_state` (since
//! deleted as vestigial) and `weights` (free-on-unload). That is the whole
//! surface, and it is what this trait exposes.
//!
//! ## Why it lives in `hipfire-runtime`
//!
//! `hipfire-loader` depends on every `hipfire-arch-*` crate; the arch crates
//! must not depend on the loader. A trait that arch crates implement and the
//! loader consumes therefore cannot live in the loader — that is a cycle. It
//! also cannot live in `saddle-core`, which sits below the runtime and must not
//! know about `KvCache`. `hipfire-runtime` is the one layer both sides already
//! depend on, so it is where the contract belongs.
//!
//! ## What this is NOT
//!
//! Not a forward-pass abstraction. Generation stays in `hipfire-generate`,
//! which is the architecture composition root by design and legitimately names
//! arch crates. This trait exists so that *infrastructure* — load
//! acknowledgement, session reset, unload — stops branching on architecture.

use crate::llama::KvCache;
use crate::multi_gpu::Gpus;
use rdna_compute::Gpu;

/// A loaded model, viewed without knowing its architecture.
///
/// Implemented by each architecture's bundle type in its own crate. The loader
/// stores `Box<dyn ArchModel>` so that adding an architecture does not edit a
/// closed enum, and the daemon asks questions instead of matching variants.
pub trait ArchModel: Send + std::any::Any {
    /// Hidden size. Spelled `dim` by some configs and `hidden_size` by others;
    /// the implementor resolves that, not the caller.
    fn dim(&self) -> usize;

    /// Number of decoder layers (`n_layers` / `num_hidden_layers`).
    fn n_layers(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Short stable identifier, e.g. `"qwen35"`. Matches the key used by
    /// [`crate::reset_core`]'s inventory so the two cannot drift.
    fn arch_key(&self) -> &'static str;

    /// The model's KV cache, when it owns one directly.
    ///
    /// `None` is legitimate: some bundles keep the cache elsewhere, and callers
    /// must treat absence as "not applicable", never as an error.
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache>;

    /// Drop per-session state so the next turn starts clean — recurrent state,
    /// conv rings, cache offsets. Position and conversation history are the
    /// caller's concern, not the model's.
    ///
    /// Default is a no-op because a pure-attention model with no recurrent
    /// state has nothing to reset, and forcing every implementor to write an
    /// empty body would obscure the ones that genuinely do work here.
    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        Ok(())
    }

    /// Whether a [`crate::session_cache::SessionCache`] is attached, i.e. the
    /// model reports reusable prompt prefixes itself.
    fn session_cache_attached(&self) -> bool {
        false
    }

    /// Prompt tokens the next prefill on `route` can restore instead of
    /// computing; 0 when nothing is cached.
    fn session_plan(&self, prompt: &[u32], route: crate::session_cache::SessionRoute) -> usize {
        self.session_plan_capped(prompt, route, prompt.len())
    }

    /// [`Self::session_plan`] restricted to reuse points `<= cap`. The one
    /// method a model with a session cache implements; default no cache.
    fn session_plan_capped(
        &self,
        _prompt: &[u32],
        _route: crate::session_cache::SessionRoute,
        _cap: usize,
    ) -> usize {
        0
    }

    /// The deepest reuse point `resumable` accepts (e.g. one a speculator's
    /// draft context can restart from, [`crate::spec::Speculator::can_resume_at`]):
    /// an unaccepted hit falls back to the next lower reuse point, else 0.
    fn session_plan_resumable(
        &self,
        prompt: &[u32],
        route: crate::session_cache::SessionRoute,
        resumable: &dyn Fn(usize) -> bool,
    ) -> usize {
        let mut start = self.session_plan(prompt, route);
        while start > 0 && !resumable(start) {
            start = self.session_plan_capped(prompt, route, start - 1);
        }
        start
    }

    /// Publish the snapshots the last prefill captured. Call once the turn's
    /// output is committed to the client.
    fn session_commit(&mut self) {}

    /// Publish this turn's snapshots (as [`Self::session_commit`]) AND keep
    /// the live state for a continuation whose prompt extends `consumed`, the
    /// consumed host history (empty = none, e.g. after an unrepaired
    /// terminal). A live continuation is session-exact (its state descends
    /// from decode), not cold-exact. The default publishes only.
    fn session_commit_live(&mut self, _consumed: &[u32]) {
        self.session_commit()
    }

    /// Once per request, after prefill and before the first decode forward
    /// (and so before any decode capture/record): map the model's
    /// context-growing state ahead for a decode that starts at `position`,
    /// may emit `remaining_new_tokens` more tokens and, when speculating,
    /// writes up to `max_draft_rows` rows past a window's seed (0 for AR).
    /// Best effort: an implementation logs and swallows failure, leaving
    /// on-demand growth. Default is a no-op for models whose state does not
    /// grow with context.
    fn map_ahead_for_decode(
        &mut self,
        _gpu: &mut Gpu,
        _position: usize,
        _remaining_new_tokens: usize,
        _max_draft_rows: usize,
    ) {
    }

    /// One-shot advisory for the next session begin: this request will write
    /// state rows at most up to absolute position `through` (prompt, decode
    /// and any speculative rows). A model that parks side state may use it to
    /// bound that parking before prefill. It is not validation and never
    /// rejects a request; the limit applies to the next begin only. Default
    /// is a no-op for models with no session side state.
    fn session_request_write_limit(&mut self, _through: usize) {}

    /// Downcast hatch for the architecture composition root.
    ///
    /// `hipfire-generate` legitimately needs the concrete bundle to call a
    /// per-architecture forward pass — that is what a composition root does.
    /// This exists so it can keep doing that once `ModelState` is replaced by
    /// `Box<dyn ArchModel>`.
    ///
    /// Crucially it borrows only the receiver, so a caller can hold the
    /// downcast bundle and a disjoint `LoadedModel` field at the same time.
    /// A whole-struct accessor cannot: that distinction is why the accessor
    /// experiment converted 15 sites of 154 and this hatch is expected to do
    /// better.

    /// Return every GPU buffer this model owns.
    ///
    /// Consumes the box: unload is terminal, and taking `self` by value makes
    /// use-after-free a compile error rather than a runtime one.
    fn free_gpu(self: Box<Self>, gpu: &mut Gpu);

    /// Whether this model runs on a `topology.pp` mesh and so owns its
    /// lifecycle over the load's [`Gpus`] (`LoadedModel::pp_gpus`): the
    /// loader and the generation layer then call the `*_mesh` hooks below
    /// with the daemon's `gpu` and the mesh devices, instead of the
    /// single-device methods. Default `false`: every other model keeps its
    /// existing single-device and Qwen3.5/Qwen4 route-specific lifecycle.
    fn owns_mesh_lifecycle(&self) -> bool {
        false
    }

    /// [`Self::reset_session_state`] for a model that owns a mesh lifecycle:
    /// the reset runs on every rank's device and is drained. Default is the
    /// single-device reset on the daemon `gpu`.
    fn reset_session_state_mesh(&mut self, gpu: &mut Gpu, _gpus: &mut Gpus) -> Result<(), String> {
        self.reset_session_state(gpu)
    }

    /// [`Self::map_ahead_for_decode`] for a model that owns a mesh lifecycle:
    /// grow every rank's context-sized state. Default is the single-device
    /// map-ahead on the daemon `gpu`.
    fn map_ahead_for_decode_mesh(
        &mut self,
        gpu: &mut Gpu,
        _gpus: &mut Gpus,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        self.map_ahead_for_decode(gpu, position, remaining_new_tokens, max_draft_rows)
    }

    /// [`Self::free_gpu`] for a model that owns a mesh lifecycle. Drains the
    /// mesh before freeing anything an epoch could still read, then frees
    /// every rank on its own device. `Err` means the drain failed and the
    /// model's resources were leaked on purpose (quarantine): the caller must
    /// then leak the `Gpus` too and never free under possibly in-flight work.
    /// Default is the single-device free on the daemon `gpu`.
    fn free_gpu_mesh(self: Box<Self>, gpu: &mut Gpu, _gpus: &mut Gpus) -> Result<(), String> {
        self.free_gpu(gpu);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle with no recurrent state and no directly-owned cache still
    /// satisfies the contract without writing either method — that is the point
    /// of the defaults, and a regression here would force boilerplate into
    /// every pure-attention arch crate.
    struct Minimal {
        dim: usize,
    }

    impl ArchModel for Minimal {
        fn dim(&self) -> usize {
            self.dim
        }
        fn n_layers(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            32
        }
        fn arch_key(&self) -> &'static str {
            "minimal"
        }
        fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
            None
        }
        fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {}
    }

    #[test]
    fn defaults_cover_a_stateless_arch() {
        let m = Minimal { dim: 8 };
        assert_eq!(m.dim(), 8);
        assert_eq!(m.arch_key(), "minimal");
    }

    /// A model that says nothing about meshes owns no mesh lifecycle, so the
    /// loader and generation layer keep their single-device and Qwen3.5/Qwen4
    /// routes for it.
    #[test]
    fn a_model_owns_no_mesh_lifecycle_by_default() {
        let m = Minimal { dim: 8 };
        assert!(!m.owns_mesh_lifecycle());
        let boxed: Box<dyn ArchModel> = Box::new(m);
        assert!(!boxed.owns_mesh_lifecycle());
    }

    #[test]
    fn trait_is_object_safe() {
        // The loader stores these behind a box; if this stops compiling the
        // whole design is void, so pin it rather than discovering it later.
        let m: Box<dyn ArchModel> = Box::new(Minimal { dim: 4 });
        assert_eq!(m.n_layers(), 2);
        assert_eq!(m.vocab_size(), 32);
    }

    /// Reuse points at 4096, 8192 and 12288.
    struct Snapshots;

    impl ArchModel for Snapshots {
        fn dim(&self) -> usize {
            8
        }
        fn n_layers(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            32
        }
        fn arch_key(&self) -> &'static str {
            "snapshots"
        }
        fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
            None
        }
        fn session_plan_capped(
            &self,
            prompt: &[u32],
            _route: crate::session_cache::SessionRoute,
            cap: usize,
        ) -> usize {
            let top = cap.min(prompt.len() - 1);
            top / 4096 * 4096
        }
        fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {}
    }

    #[test]
    fn unresumable_hit_falls_back_to_the_next_lower_reuse_point() {
        let prompt = vec![0u32; 13_000];
        let route = crate::session_cache::SessionRoute::Ar;
        let m = Snapshots;
        assert_eq!(m.session_plan(&prompt, route), 12288);
        // Resumable only with >= 2048 rows left to prefill: 12288 and 8192 leave
        // too few / enough.
        let leaves_window = |s: usize| prompt.len() - s >= 2048;
        assert_eq!(
            m.session_plan_resumable(&prompt, route, &leaves_window),
            8192
        );
        assert_eq!(m.session_plan_resumable(&prompt, route, &|_| false), 0);
        assert_eq!(m.session_plan_resumable(&prompt, route, &|_| true), 12288);
    }
}
