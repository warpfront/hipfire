// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Zero-cost, composable WMMA GEMM fragments. Scheduling and register
//! allocation belong to the caller; these parts preserve per-output order.
pub mod chain;
pub mod epilogue;
pub mod prefetch;
pub mod sched;
pub mod tile;
pub use chain::{Chain, Iu4, Iu8, MmaKind};
pub use epilogue::{Epilogue, DENSE_SILU_TEMPS};
pub use prefetch::Prefetch;
pub use tile::{FragmentLayout, Lds, RegisterDirect, Tile};
