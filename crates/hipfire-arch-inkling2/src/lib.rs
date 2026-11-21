// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2-Mini: a looped 24-physical-layer MoE text architecture.
//!
//! Phase 0 only: loop geometry, fail-closed validation, and CPU oracle
//! operators. Nothing here is wired into the loader or GPU runtime yet.
//! The schedule and validation mirror llama.cpp's `inkling2-loop.cpp` at
//! merge commit `29b52d398b294509746ec0a3f28948a3107c0213`.

pub mod config;
pub mod forward;
pub mod oracle;
pub mod weights;

pub use config::{LoopConfig, ARCH_ID, MTP_SIDECAR_ARCH_ID};

/// The verified Mini config.json metadata shape, shared by tests across modules.
#[cfg(test)]
pub(crate) const TEST_METADATA: &str = r#"{"config":{"eos_token_id":200006,"text_config":{
    "num_hidden_layers":24,"hidden_size":2304,"vocab_size":201024,
    "unpadded_vocab_size":200058,"num_attention_heads":24,"num_key_value_heads":8,
    "head_dim":128,"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,
    "qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
    "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"sconv_kernel_size":4,
    "n_layers_per_attn_res_block":6,"final_logit_softcapping":30.0,
    "n_routed_experts":128,"num_experts_per_tok":8,"intermediate_size":1152,
    "route_scale":8.0,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"rms_norm_eps":1e-6,
    "local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
    "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}
}}}"#;
