// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Audio (audio-input / ASR) architecture for hipfire — Phase 3 of the
//! LFM3 port (`docs/plans/2026-10-31-lfm3-port.md`).
//!
//! Wraps the LFM3 text decoder (`hipfire-arch-lfm3`, arch_id 13) with the
//! LFM3-Audio speech encoder + audio projector. The speech encoder is a
//! conv2d mel downsampler followed by a 17-layer LFM2 backbone (hidden 768,
//! conv K=3, 8Q/4KV heads, sliding-window 32 causal attention), exactly the
//! HF `Lfm3AudioEncoder` in transformers `modeling_lfm3_audio.py`.
//!
//! Layout mirrors `hipfire-arch-lfm3-vl`:
//!   - `mel`          — CPU log-mel-spectrogram front-end (Slaney mel bank,
//!                      STFT, log10, affine normalization). Pure-CPU, ungated.
//!   - `lfm3_audio`   — config + weights + conv2d downsampler + 17-layer
//!                      encoder forward + audio projector (GPU).
//!   - `arch`         — `Architecture` trait impl for the bring-up triple.
//!
//! The mel front-end + conv2d downsampler run on CPU (the mel is small and
//! must byte-match the HF `Lfm3AudioFeatureExtractor`); the 17-layer encoder
//! backbone + projector run on GPU, reusing the LFM3 conv/attn/RoPE/GQA
//! primitives. arch_id stays 13 (audio detected by `encoder.downsample.*` /
//! `encoder.lfm.*` tensor presence, the qwen35-vl / lfm3-vl precedent).
//!
//! Forward dispatch is static (not trait-routed), matching the project
//! convention: `daemon.rs` calls `lfm3_audio::audio_forward` and
//! `lfm3::forward::decode_step` / `decode_step_with_embed` directly.

/// CPU log-mel-spectrogram front-end (no GPU needed). Always compiled.
pub mod mel;
/// Minimal PCM WAV decoder (no GPU needed). Always compiled.
pub mod wav;

#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod lfm3_audio;

#[cfg(feature = "deltanet")]
pub use arch::Lfm3Audio;
#[cfg(feature = "deltanet")]
pub use lfm3_audio::{
    audio_config_from_hfq, audio_forward, audio_forward_streaming, load_audio_weights,
    AudioConfig, AudioEncoderStreamingCache, AudioWeights,
};
