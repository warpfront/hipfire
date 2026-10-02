// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-runtime: GGUF model loading and LLaMA inference on RDNA GPUs.
//!
//! This crate is arch-agnostic. Architecture implementations live in
//! sibling crates (`hipfire-arch-qwen35`, `hipfire-arch-qwen35-vl`,
//! future `hipfire-arch-llama`, etc.) and depend on this crate for
//! shared infrastructure: HFQ/GGUF file readers, the LLaMA-style
//! scratch / KV / sampler primitives, tokenizer, prompt framing, eos
//! filter, loop guard, eviction (TriAttn, CASK), spec-decode primitives
//! (DFlash, DDTree), demand paging (cpu_router, weight_pager), and the
//! [`arch::Architecture`] trait.

pub mod admission;
pub mod arch;
pub mod arch_mapping;
pub mod arch_model;
pub mod arch_spec;
pub mod augmentor;
pub mod bf16_loader;
pub mod cache_plan;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod chatml;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
pub mod device_mesh;
#[cfg(feature = "deltanet")]
pub mod dflash;
/// Adaptive DFlash verify-block controller — family-free policy object shared
/// by arch speculators (`hipfire-arch-qwen35::dflash_spec`); the DSpark
/// analogue is `dspark_block_controller`. Pure math, no GPU types.
pub mod dflash_adaptive_block;
pub mod dflash_generic;
pub mod dspark_block_controller;
pub mod dspark_core;
pub mod ep;
pub mod eval_common;
pub mod external_rows;
pub mod gguf;
pub mod hfq;
pub mod hfq_parallel;
pub mod imagedec;
pub mod kv_adaptive;
pub mod kv_backend;
pub mod kv_mode;
pub mod llama;
pub mod llama_spec;
pub mod lloyd_lut;
pub mod loader_api;
pub mod loop_guard;
pub mod model_load;
pub mod model_source;
pub mod multi_gpu;
pub mod paro;
pub mod prefix;
pub mod prefix_index;
pub mod reset_core;
pub mod safetensors_source;
pub mod sampler;
pub mod sealed_moe;
pub mod serve;
pub mod checkpoint_pool;
pub mod serve_contract;
pub mod serve_fairness;
pub mod serve_wait;
/// `SlotBatch` — one forward step's ragged work across N slots. Pure CPU
/// data structure; no GPU dependencies. Moved from `hipfire-arch-qwen35`
/// (the multi-slot scheduler/batch substrate is model-agnostic). See module
/// docs for the per-slot-absolute `positions[]` invariant.
pub mod slot_batch;
/// `Scheduler` — decides what goes into each step's `SlotBatch`. Pure CPU
/// logic; no GPU dependencies. Round-robin, chunked prefill mixed with
/// decode. Moved from `hipfire-arch-qwen35` with `slot_batch`.
pub mod scheduler;
pub mod sidecar;
pub mod spec;

pub mod ngram_mod;
pub mod spec_ngram;
pub mod swap;
pub mod tp_shard;
#[cfg(feature = "deltanet")]
pub mod triattn;
pub mod weight_manifest;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
pub mod weight_store;

pub mod emit_text;
pub mod eos_filter;
pub mod prompt_frame;
pub mod semantic;
pub mod stop_sequence;
pub mod session_table;
pub mod tokenizer;

pub mod calibration;
pub mod tool_call;
pub mod weight_backend;

pub use crate::arch::{maybe_screen_mmq, screen_weight_tensor, MmqScreenable};
pub use crate::serve_contract::{
    ArchPolicy, CacheDomain, CanonicalError, CheckpointId, CommitBoundary, DeviceTopology,
    DrafterDecision, KvLayout, LastTokenHandling, MissReason, PrefixLookup, PrefixLookupResult,
    PublishLease, ReleaseDisposition, ReservationError, ResumeBundle, ResumePlan, ResumePlanError,
    SharingNamespace, StepNeeds, StepReservation, StepTicket, TemplateIdentity, TokenizerIdentity,
};
