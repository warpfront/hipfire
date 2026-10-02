// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// hipfire-dispatch: unified kernel dispatch abstraction.
//
// One entry point per kernel family. Models never match on DType.
// The dispatch layer selects the correct kernel based on quant format,
// arch capabilities, and feature flags — all resolved at init time.

#[macro_use]
mod macros;

pub mod context;
pub mod cpu_exec;
pub mod families;
pub mod ops;
pub mod pipeline;
pub use cpu_exec::{
    cpu_exec_counters, cpu_exec_enabled, cpu_exec_redline_conflict, cpu_offload_active,
    cpu_quant_for, host_mapped_cpu_capable, log_capture_disabled_once,
    reject_cpu_exec_under_redline, run_host_mapped_gemv, run_host_mapped_gemv_residual,
};
pub mod resource;
pub mod tables;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod coverage_tests;
