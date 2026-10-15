// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-8B-A1B architecture (arch_id 11): a HYBRID model interleaving
//! double-gated LIV short-convolution mixers (18 layers) with GQA+QK-norm
//! attention mixers (6 layers), feeding a DeepSeek-style sigmoid-bias top-4
//! MoE FFN (layers >= num_dense_layers) or a dense SwiGLU MLP (the first
//! `num_dense_layers` layers). Per-layer mixer choice comes from the
//! checkpoint's `layer_types`.
//!
//! Single-lane decode is a typed `hipfire_dispatch` step program per layer
//! (`program.rs`); `forward.rs` keeps only the lifecycle around it (input
//! staging, hipGraph and Redline retained replay, oracle capture). The
//! dispatch steps it binds:
//!   * mixer norms / projections -> `RmsnormAutomatic`, `Gemv`, `GemvResidual`
//!   * LIV conv    -> `GatedShortConv` (B*x gate, depthwise causal conv,
//!                    C gate, rolling conv state)
//!   * attention   -> `QkNorm`, `Rope`, `Attend` (Q8 KV)
//!   * dense MLP   -> `SiluMul` between the gate/up and down projections
//!   * MoE         -> sealed `Moe` call: sigmoid router with the expert
//!                    selection bias (`MoeSelectionBias`), runtime top-k
//! The continuous-batching decode (`forward_batch.rs`, dense only) is still
//! its own loop.
pub mod arch;
pub mod arch_model;
pub mod batch;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod forward_batch;
pub mod lfm2moe;
mod program;
pub mod spec_impl;

pub use arch::Lfm2Moe;
pub use batch::Lfm2DecodeBatchState;
pub use carrier::load_lfm2moe_bundle;
pub use config::{config_from_source, Lfm2MoeConfig, MixerKind};
pub use forward::{decode_step, decode_step_capture};
pub use forward_batch::{
    batch_weight_formats_supported, forward_decode_batch_lfm, forward_decode_batch_prepared_lfm,
    prepare_decode_batch_inputs_lfm,
};
pub use lfm2moe::{load_weights_from_source, Lfm2MoeState, Lfm2MoeWeights};
pub use spec_impl::Lfm2MoeBundle;
/// Architecture id for LFM2.5-MoE.
pub const ARCH_ID: u32 = 11;
