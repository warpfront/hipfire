// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed operations of a multi-head latent attention (MLA) decoder layer with
//! manifold-constrained hyper-connections, a strided KV compressor, a sparse
//! indexer and a shared-plus-routed expert FFN (DeepSeek-V4 layout).
//!
//! Architecture crates bind resident weights, state and scratch into these
//! operations; the kernels, their order and the route choice live here.

pub mod attention;
pub mod compressor;
pub mod expert_ffn;
pub mod hc;
pub mod project;
pub mod query_kv;
pub mod rows;

pub use project::{project_prepared, project_prepared_rows, E8SoaKernel, RowsProjection};
