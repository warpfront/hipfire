// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! [`ArchModel`] impl for a loaded FLUX pipe — the loader's arch-agnostic
//! view (see the trait docs in `hipfire-runtime/src/arch_model.rs` for why
//! this exists and where it is NOT supposed to be used).
//!
//! Diffusion components are loadable but **never chat-served**:
//! `vocab_size` is 0, `kv_cache_mut` is `None`, `arch_key` is the
//! `"flux_mmdit"` string the reset/unload inventory would key on if a
//! component ever rode those paths (it does not today).

use crate::pipeline::FluxPipeBundle;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

/// The daemon-visible view of a full FLUX pipe (transformer + text encoders +
/// VAE + tokenizers). Loaded through the `FluxDiffusionCarrier` from a
/// diffusers pipe directory or from per-component HFQ packs; the daemon's
/// `img_generate` runs against this bundle.
pub struct FluxPipeModel {
    pub bundle: FluxPipeBundle,
}

impl ArchModel for FluxPipeModel {
    fn dim(&self) -> usize {
        self.bundle.transformer_cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.bundle.transformer_cfg.num_layers + self.bundle.transformer_cfg.num_single_layers
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn arch_key(&self) -> &'static str {
        "flux_mmdit"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }
    /// Release the pipe's GPU residency — transformer AND VAE decoder
    /// weights, both uploaded by [`FluxPipeBundle::ensure_gpu`]. This used to
    /// be empty, which leaked every uploaded buffer on model unload.
    ///
    /// The trait method cannot report failure, but `free_gpu` is best-effort
    /// and has already emptied every slot by the time it returns an error, so
    /// the error is a diagnostic, not a leak the caller can act on — log it
    /// rather than discarding it silently.
    fn free_gpu(mut self: Box<Self>, gpu: &mut Gpu) {
        if let Err(e) = self.bundle.free_gpu(gpu) {
            eprintln!("flux: releasing GPU residency on unload: {e}");
        }
    }
}
