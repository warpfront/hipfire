// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-embeddinggemma2: Google's **EmbeddingGemma 2** single-vector
//! embedding architecture (`arch_id` 18, `google/embeddinggemma-2`).
//!
//! What the model is: a 24-layer Gemma-4-family **bidirectional encoder**
//! (dim 512, gated gelu-tanh FFN 2048, sandwich RMSNorm, per-layer embeddings)
//! with a 768-d `embedding_projection` head. Every layer runs non-causal
//! attention — the "sliding" layers see a symmetric ±512 window, the 5:1
//! "full" layers see the whole sequence. Sliding layers carry 4 Q heads of
//! 256 with 2 KV heads; full layers widen to head_dim 512 with 1 KV head
//! (MQA), per the checkpoint's `per_layer_config`. RoPE is half-split over
//! the whole head dim with θ = 10 000 (sliding) / 1 000 000 (full), Q/K get
//! learned per-head RMSNorm, V a scale-less one, and attention runs with
//! scaling 1.0 (the QK norms replace the usual 1/√d). The per-layer
//! embedding (PLE) signal is *projection-only*: a single
//! `[layers·512, 512]` linear on the (embed-scaled) token embeddings,
//! reshaped per layer, RMSNorm'd, and mixed into every layer's residual
//! stream after the MLP by a gated bottleneck block. After the final norm
//! the trunk projects every token to 768 dims; sentence-transformers then
//! mean-pools the tokens and L2-normalizes (Matryoshka truncation to
//! 512/256/128 with re-normalization is supported per request).
//!
//! What this crate owns:
//! - [`config`]: parse `text_config` (model_type `embedding_gemma2_text`) +
//!   the sentence-transformers prompt sidecars.
//! - [`names`]/[`weights`]: stream `language_model.*` tensors (F32/BF16/F16
//!   dir or `.hfq`, compressed tiers decoded through the runtime registry)
//!   to the GPU as f16 GEMM weights + f32 norms, embedding table host f32.
//! - [`forward`]: the GPU encoder — embed gather × √512, PLE projection,
//!   per-layer bidirectional attention (eager WMMA-GEMM decomposition with
//!   an additive band mask on sliding layers when the sequence outgrows the
//!   window; Q-tiled kernel below the eager threshold), MLP, PLE block,
//!   `layer_scalar`, final norm and the 768-d projection.
//! - [`post`]: mean pooling, Matryoshka truncation, L2 normalize — host
//!   math, unit-testable without a GPU.
//! - [`bundle`]/[`carrier`]: the `ArchModel` bundle and the loader entry
//!   the carrier calls; [`slot_engine`] batches concurrent requests into
//!   one forward.
//!
//! Multimodal scope: the checkpoint also ships a Gemma-4 vision tower, an
//! audio tower and their projectors (`vision_tower.*`, `audio_tower.*`,
//! `embed_vision.*`, `embed_audio.*`). Both legs are implemented and stream
//! from the same arch-18 `.vl` sidecar: the vision tower ([`vision`]) with
//! its image processor ([`image`]), the audio tower ([`audio_tower`]) with
//! its mel frontend ([`audio`]). A request's media is spliced as projected
//! rows over the placeholder rows before the text encode
//! ([`carrier::assemble_input`]). A bundle loaded without a sidecar is
//! text-only and refuses images with [`carrier::NO_VISION_TOWER`] and audio
//! with [`carrier::NO_AUDIO_TOWER`] — never silently. Text that literally
//! contains a placeholder token like `<|image|>` with no media attached
//! embeds as that token's row — the same thing the HF tokenizer does, so
//! prompt-paste parity is preserved.
//!
//! Numerics: activations f32, GEMM weights f16 (WMMA f16×f16 f32-accumulate)
//! — the same tradeoff the qwen35-embed encoder makes, ~1e-3 relative vs an
//! f32 reference. The checkpoint ships BF16; the model card's "do not use
//! float16" warning concerns serving the whole network in f16, not the
//! weight-operand precision of an f32-accumulate GEMM.

pub mod audio;
pub mod audio_tower;
pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod image;
pub mod names;
pub mod post;
pub mod slot_engine;
pub mod video;
pub mod vision;
pub mod weights;

pub use audio::{decode_wav, process_audio, AudioPlan, ProcessedAudio};
pub use audio_tower::{load_audio_tower, AudioTower};
pub use bundle::EmbedGemmaBundle;
pub use carrier::{
    assemble_input, encode_assembled, encode_media, load_embed_gemma_bundle,
    load_embed_gemma_from_source, load_embed_gemma_with_vision, media_rows, preflight_slot,
    AssembledInput, MediaKind, MediaSpan, NO_AUDIO_TOWER, NO_VIDEO_TOWER, NO_VISION_TOWER,
};
pub use config::{
    parse_audio_config, parse_vision_config, AudioConfig, AudioFrontendConfig, EmbedConfig,
    InputKind, OverflowStrategy, VideoFrontendConfig, VisionConfig,
};
pub use forward::{encode_embeddings, encode_embeddings_batch, EncodeInput, MediaRows};
pub use image::ProcessedImage;
pub use post::{l2_normalize_rows, mean_pool_rows, mrl_truncate};
pub use video::{process_video, sample_indices};
pub use vision::{encode_images, VisionTower};

/// `arch_id` this crate claims — see `docs/architecture-ids.md`.
pub const ARCH_ID: u32 = 18;
