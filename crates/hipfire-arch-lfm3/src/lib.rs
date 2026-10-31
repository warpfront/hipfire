// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 architecture (arch_id 13): Liquid AI's third-generation foundation
//! model text backbone. An evolution of LFM2 (`hipfire-arch-lfm2moe`,
//! arch_id 11) with three layer types instead of two:
//!
//!   * `conv`             — double-gated LIV short-convolution (K=7, vs LFM2's 3)
//!   * `full_attention`   — GQA + per-head QK-norm + partial-rotary RoPE
//!                          (partial_rotary_factor 0.5, theta 1e6) + sigmoid
//!                          output gate per head
//!   * `sliding_attention`— GQA + full-dim RoPE (theta 1e4) + sliding window
//!                          (256) + sigmoid output gate per head
//!
//! Plus **sandwich norm** (`use_sandwich_norm`): an RMSNorm on the mixer and
//! FFN outputs *before* they enter the residual stream (`post_operator_norm`,
//! `post_ffn_norm`). FFN is dense SwiGLU only (no MoE in the text backbone).
//!
//! Kernel coverage — all pre-existing except one small fused gate kernel:
//!   * conv          -> conv1d_gated_decode_f32 (K=7; KMAX=8, runtime K)
//!   * full attn     -> rmsnorm_batched (QK) + rope_partial_halfsplit (n_rot=64)
//!                      + Q8 GQA flash attention
//!   * sliding attn  -> same rope (n_rot=head_dim) + AttnFlashQ8_0Windowed
//!                      (window=256, via KvTierPlan{q8_windowed,window})
//!   * sigmoid gate  -> sigmoid_head_gate_mul_f32 (NEW: per-head broadcast)
//!   * sandwich norm -> rmsnorm_f32 (post_operator/post_ffn) + add_f32 (residual)
//!   * dense MLP     -> Q8 SwiGLU (w1 gate, w3 up, silu_mul, w2 down)
//!
//! Ground truth: transformers `configuration_lfm3.py` / `modeling_lfm3.py`
//! / `lfm3_conv.py` (read 2026-10-31). Validation target: `LiquidAI/LFM3-1.7B`.
//! Omni multimodal (vision/audio/codec) is phased later — see
//! `docs/plans/2026-10-31-lfm3-port.md`.

pub mod arch;
pub mod config;
pub mod forward;
pub mod lfm3;
pub mod spec_impl;

pub use arch::Lfm3;
pub use config::{config_from_source, Lfm3Config, MixerKind};
pub use forward::{decode_step, decode_step_capture};
pub use lfm3::{load_weights_from_source, Lfm3State, Lfm3Weights};
pub use spec_impl::Lfm3Bundle;

/// Architecture id for LFM3.
pub const ARCH_ID: u32 = 13;
