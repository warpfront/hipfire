// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU execution core for host-mapped (spilled) weights.
//!
//! Partial offload (`memory.gpu_layer_budget`) places whole layers in
//! host-mapped system RAM, but today every kernel still runs *on the GPU*: the
//! GPU dereferences a device alias of host memory and reads those weight bytes
//! across PCIe once per token. `memory.offload_exec=cpu` instead executes the
//! weight-reading matmuls on the CPU, which bounds a spilled layer's per-token
//! cost by the link rather than by device DRAM — the same thing llama.cpp's
//! CPU backend does for its `-ngl` spill.
//!
//! Only the ops that *read weight bytes* move: GEMV / GEMV-with-residual and
//! their fused forms. Attention, softmax, rmsnorm, RoPE, qk-norm, the KV write,
//! the flash-attention families and the DeltaNet recurrence stay on the GPU, as
//! does the KV cache's residency. The numerical contract is llama.cpp-level
//! coherence, not bit-identity (the GPU and CPU paths are independent
//! implementations with different accumulation orders); see
//! `docs/perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`.
//!
//! This crate is deliberately a leaf: it depends on `rayon` and nothing else,
//! so its tests run in the GPU-free CI gate (`scripts/no-gpu-ci.sh`) with no GPU
//! stack linked. That is also why [`quant`] *transcribes* the canonical decoder
//! in `hipfire-runtime` instead of calling it —
//! `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs` holds the two copies
//! together by asserting bit-for-bit equality over the real tensors of the
//! on-disk fixtures.

pub mod epilogue;
pub mod gemv;
pub mod quant;
pub mod simd;

#[cfg(test)]
pub(crate) mod testfix;
