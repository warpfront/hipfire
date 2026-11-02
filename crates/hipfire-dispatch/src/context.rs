// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::resource::ResourceManager;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::FeatureFlags;
use rdna_compute::Gpu;
use std::sync::Arc;

/// Semantic workload carried through dispatch.
///
/// Most kernel choices are shape-driven. Speculative target verification is a
/// deliberate exception: the gfx12 wide-query attention route has repeatedly
/// regressed that short-batch regime, even when an opt-in overrides its normal
/// shape envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchWorkload {
    Standard,
    SpeculativeVerify,
}

/// Hardware snapshot plus call-site semantics shared immutably across a
/// dispatch sequence.
pub struct DispatchCtx {
    pub arch: ArchCaps,
    pub flags: Arc<FeatureFlags>,
    pub resources: ResourceManager,
    pub workload: DispatchWorkload,
    /// Batched masked Q8/fp8 attends of this context skip the LDS crossover
    /// kernel and always take the tile + reduce. Set only for DFlash
    /// ChainVerify blocks: at B = 16 the LDS kernel is 3.8-23x slower
    /// (ctx 64-4096, fp8 KV, R9700), and the tile + reduce output does not
    /// depend on `max_ctx_len`, so the eager verify and its HIP graph
    /// (recorded with `physical_cap`) produce the same bytes. MTP verify
    /// and every other workload keep the crossover (default `false`).
    pub verify_tile_attend: bool,
    /// Token-row count of the current batch (0 = unknown / not a batch).
    /// Lets shape-driven route choices (e.g. MoE grouped vs indexed) see
    /// the batch width without a signature change at every call site.
    pub batch_size: usize,
    /// Per-call actual eager-verify scope for the CPU-owned expert window.
    /// Default `false`; the FN forward sets it ONLY for a real eager verify
    /// call (non-prefill scope, `>= 2` rows, not recording/capturing, oracle
    /// override present). A whole 64-row ordinary prefill (whose rows equal
    /// the canonical chunk) MUST NOT set it: `rows == route_rows` alone never
    /// implies verify. Not a user knob; unrelated to DFlash
    /// `verify_tile_attend`. Lanes, S1 tiles, tests and non-FN seals keep the
    /// default `false` unchanged.
    pub cpu_wide_verify: bool,
    device_id: i32,
}

impl DispatchCtx {
    /// Create a `DispatchCtx` from the GPU's current state. This is cheap
    /// enough to call per-layer (ArchCaps is a few dozen bools and the
    /// FeatureFlags snapshot is reference-counted), but callers in tight loops
    /// should prefer creating it once and reusing the reference.
    pub fn new(gpu: &Gpu) -> Self {
        let flags = gpu.flags.clone();
        let arch = ArchCaps::new(&gpu.arch, flags.clone());
        Self {
            arch,
            flags,
            resources: ResourceManager::new(gpu),
            workload: DispatchWorkload::Standard,
            verify_tile_attend: false,
            batch_size: 0,
            cpu_wide_verify: false,
            device_id: gpu.device_id,
        }
    }

    /// Device identity bound to this dispatch context.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Attach call-site semantics to this otherwise hardware-derived context.
    pub fn with_workload(mut self, workload: DispatchWorkload) -> Self {
        self.workload = workload;
        self
    }

    /// Route this context's batched masked Q8/fp8 attends through the tile +
    /// reduce, bypassing the LDS crossover (DFlash ChainVerify only).
    pub fn with_verify_tile_attend(mut self, verify_tile_attend: bool) -> Self {
        self.verify_tile_attend = verify_tile_attend;
        self
    }

    /// Mark this call as an actual eager-verify scope for the CPU-owned
    /// expert window (see [`Self::cpu_wide_verify`]). The FN forward computes
    /// this; no other caller sets it.
    pub fn with_cpu_wide_verify(mut self, cpu_wide_verify: bool) -> Self {
        self.cpu_wide_verify = cpu_wide_verify;
        self
    }

    /// Whether this call runs inside an actual eager-verify scope for the
    /// CPU-owned expert window.
    pub fn cpu_wide_verify(&self) -> bool {
        self.cpu_wide_verify
    }

    /// Attach the token-row count of the current batch. 0 (default) means
    /// unknown — route choices must not depend on it then.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Construct a `DispatchCtx` for the given arch string and device without
    /// a live GPU. Only for use in tests.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test_device(arch: &str, device_id: i32) -> Self {
        use rdna_compute::feature_flags::FeatureFlags;
        let flags = Arc::new(FeatureFlags::for_test(arch));
        let arch_caps = ArchCaps::new(arch, flags.clone());
        Self {
            arch: arch_caps,
            flags,
            resources: crate::resource::ResourceManager::for_test(),
            workload: DispatchWorkload::Standard,
            verify_tile_attend: false,
            batch_size: 0,
            cpu_wide_verify: false,
            device_id,
        }
    }

    /// Construct a test context bound to device zero.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test(arch: &str) -> Self {
        Self::for_test_device(arch, 0)
    }
}
