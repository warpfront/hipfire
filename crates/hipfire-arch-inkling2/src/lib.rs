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
pub mod oracle;
pub mod weights;

pub use config::{LoopConfig, ARCH_ID, MTP_SIDECAR_ARCH_ID};
