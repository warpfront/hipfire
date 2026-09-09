// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! hipfire-arch-spark25: Spark-X2.5-4B (`model_type: spark2_5`, arch_id 16).
//!
//! Dense GQA hybrid 3:1 SWA (window 512) + full attention, dual RoPE
//! (sliding 256@1e4, full 64@5e6), headwise sigmoid attn gate, exact-GELU SwiGLU.
//!
//! Carrier ownership lives in `hipfire-loader` (maple pattern). This crate
//! exports `load_spark25_bundle` + `Spark25Bundle` + free-function forward.

pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod spark25;

pub use bundle::{
    load_spark25_bundle, load_spark25_from_hfq, resolve_eos, Spark25Bundle, SPARK25_EOS_FALLBACK,
};
pub use config::{Spark25Config, Spark25LayerType};
pub use forward::{
    decode_step, decode_step_body, decode_step_body_capture, decode_step_capture, prefill,
    prefill_chunked_cancellable, prefill_chunked_capture, prefill_from,
    prepare_retained_decode_inputs, run_retained_decode_body, SparkPrefillScratch,
    SPARK_PREFILL_CHUNK,
};
pub use spark25::{Spark25LayerWeights, Spark25State, Spark25Weights};
