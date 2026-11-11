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
pub mod cpu_split;
pub mod cpwait;
pub mod families;
pub mod offload_split;
pub mod ops;
pub mod pipeline;
pub use cpu_exec::{
    cpu_exec_counters, cpu_exec_enabled, cpu_exec_redline_conflict, cpu_offload_active,
    cpu_quant_for, host_mapped_cpu_capable, log_capture_disabled_once, moe_cpu_experts,
    moe_cpu_experts_enabled, moe_layer_trace, moe_step_trace, moe_trace_enabled, moe_window_trace,
    passback_enabled, reject_cpu_exec_under_redline, run_host_mapped_gemv,
    run_host_mapped_gemv_residual, split_counters, MoeLayerTiming, MoeStepTime, MoeWindowTiming,
    CPU_EXECUTED_DTYPES,
};
pub use offload_split::{
    passback_capable_format, probe_synthetic, probe_weight, row_bytes_for, scheduler_snapshot,
    seed, PassbackOptions, ShapeSnapshot, SplitCalibration,
};
pub mod resource;
pub mod sp_plan;
pub mod tables;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod coverage_tests;
