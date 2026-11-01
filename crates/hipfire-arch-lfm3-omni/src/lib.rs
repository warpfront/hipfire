// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni (audio output / TTS + omni generation) architecture for hipfire —
//! Phase 4 of the LFM3 port (`docs/plans/2026-10-31-lfm3-port.md`).
//!
//! Adds the audio-OUTPUT side of LFM3-Omni on top of the Phase 1-3 text / vision
//! / audio-input work. Three components, phased:
//!
//!   * **Depth decoder** (`depth`) — a 3-layer **bidirectional** LFM3 (hidden
//!     1024, 32Q/8KV heads, head_dim 32) over the 12 codebook slots of one
//!     speech frame. Predicts the 12 codebooks from the LM hidden state via
//!     MaskGIT iterative scheduling (`[4,4,4]`). Reuses `vit_attention_gqa_f32`
//!     (the Phase-2 bidirectional attention kernel — no new attention kernel
//!     needed), `rope_partial_interleaved_f32_batched`, `rmsnorm_batched`,
//!     `sigmoid_head_gate_mul_f32`, `gemm_f16`, `silu_mul_f32`.
//!   * **Liquid Codec** (planned, 4b) — DAC-style neural codec: 12 codebooks →
//!     24 kHz waveform (FSQ decode + 8-layer LFM3 decoder + iSTFT overlap-add).
//!   * **Omni generate loop** (planned, 4c) — the interleaved text/audio state
//!     machine in the daemon.
//!
//! arch_id stays 13 (the depth decoder + codebook_embedding are detected by
//! tensor presence, the qwen35-vl / lfm3-vl / lfm3-audio precedent). Forward
//! dispatch is static (not trait-routed), matching the project convention.
//!
//! Ground truth: transformers `modeling_lfm3_omni.py` /
//! `configuration_lfm3_omni.py` / `modeling_liquid_codec.py` (read 2026-10-31,
//! `LiquidAI/LFM3-Omni-2B`).

#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod codec;
#[cfg(feature = "deltanet")]
pub mod depth;
#[cfg(feature = "deltanet")]
pub mod omni;

#[cfg(feature = "deltanet")]
pub use codec::{
    codec_decode, load_codec_weights, wav_bytes, write_wav, CodecConfig, CodecWeights,
};
#[cfg(feature = "deltanet")]
pub use depth::{
    depth_config_from_hfq, get_audio_out_embeds, load_depth_weights, sample_audio_frame,
    DepthConfig, DepthRng, DepthWeights,
};
#[cfg(feature = "deltanet")]
pub use omni::{omni_generate, omni_generate_audio, OmniOutput, AUDIO_OUT_TOKEN, AUDIO_START_TOKEN, MIXED_START_TOKEN, TEXT_END_TOKEN};
