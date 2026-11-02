// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Daemon-only diagnostic control, captured once at the next request's begin.

use std::sync::atomic::{AtomicBool, Ordering};

static DIAGNOSTIC_GPU_ROUTE: AtomicBool = AtomicBool::new(false);

/// Select GPU-only (`true`) or the configured NPU route (`false`) for future requests.
/// This neither changes the configured mode nor clears or trips the fault latch.
pub fn set_diagnostic_gpu_route(gpu: bool) {
    DIAGNOSTIC_GPU_ROUTE.store(gpu, Ordering::Relaxed);
}

/// Read at request begin, never while selecting an in-flight request's pairs.
pub fn diagnostic_gpu_route() -> bool {
    DIAGNOSTIC_GPU_ROUTE.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_route_control_is_reversible() {
        set_diagnostic_gpu_route(true);
        let gpu = diagnostic_gpu_route();
        set_diagnostic_gpu_route(false);
        assert!(gpu);
        assert!(!diagnostic_gpu_route());
    }
}
