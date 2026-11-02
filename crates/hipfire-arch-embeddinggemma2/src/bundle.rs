// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The loaded-model bundle: what the loader boxes as `Box<dyn ArchModel>`
//! and what the embed request path downcasts back to.

use crate::audio_tower::AudioTower;
use crate::config::{EmbedConfig, InputKind};
use crate::vision::VisionTower;
use crate::weights::EmbedWeights;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

pub struct EmbedGemmaBundle {
    pub config: EmbedConfig,
    pub weights: EmbedWeights,
    pub tokenizer: Tokenizer,
    /// The Gemma-4 vision tower, loaded from the `.vl` sidecar when one
    /// resolved. `None` = this checkpoint was loaded text-only, and an
    /// image-carrying request is refused with [`crate::carrier::NO_VISION_TOWER`]
    /// (never silently ignored).
    pub vision: Option<VisionTower>,
    /// The Gemma-4 audio tower, from the SAME `.vl` sidecar. `None` = no
    /// audio tower is resident, and an audio-carrying request is refused with
    /// [`crate::carrier::NO_AUDIO_TOWER`].
    pub audio: Option<AudioTower>,
}

/// Matryoshka dimensions this release advertises (`embedding_dim` plus the
/// model card's truncation points).
pub const MRL_DIMS: &[usize] = &[768, 512, 256, 128];

impl EmbedGemmaBundle {
    /// Sentence-transformers modules 1_Pooling + 2_Normalize for one encode,
    /// applied to the raw projected token matrix: mean over the `n` real
    /// tokens → optional Matryoshka truncation → L2. `raw` is
    /// `[ids.len(), embedding_dim]`; returns a `[dims]` single vector.
    ///
    /// Shared by the sequential body and the slot engine so the two wire
    /// paths cannot drift.
    pub fn finalize_embedding(
        &self,
        raw: &[f32],
        n: usize,
        dims: usize,
    ) -> Result<Vec<f32>, String> {
        let dim = self.config.embedding_dim;
        if raw.len() != n * dim {
            return Err(format!(
                "embeddinggemma2: encoder returned {} floats for {n}×{dim}",
                raw.len()
            ));
        }
        if dims != 0 && dims > dim {
            return Err(format!(
                "embeddinggemma2: requested {dims} dims but this model embeds at {dim}"
            ));
        }
        let mut emb = crate::post::mean_pool_rows(raw, n, dim);
        let emb = crate::post::mrl_truncate(&mut emb, dims);
        let mut emb = emb.to_vec();
        crate::post::l2_normalize(&mut emb);
        Ok(emb)
    }

    pub fn kind_prompt(&self, kind: InputKind) -> &str {
        self.config.kind_prompt(kind)
    }
}

impl ArchModel for EmbedGemmaBundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "embeddinggemma2"
    }

    /// No runtime KV cache: the encoder is one-shot per request.
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let bundle = *self;
        if let Some(vision) = bundle.vision {
            vision.free_gpu(gpu);
        }
        if let Some(audio) = bundle.audio {
            audio.free_gpu(gpu);
        }
        bundle.weights.free_gpu(gpu);
    }
}
