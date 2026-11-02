// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-eight-token tiles of the legacy aggregate indexed MQ4 AR recipes.
//!
//! Gate/up and expanded unweighted down retain their original, distinct
//! kernarg layouts and arithmetic. These are not slab or symmetric-IU4 kernels.

pub mod down;
pub mod gate_up;

/// Emit the two original indexed projections for one supported architecture.
/// Emission alone does not qualify a codec, shape, row width or product route.
pub fn build_rowtiles(arch: crate::Arch) -> Result<Vec<crate::Emitted>, String> {
    let mut kernels = Vec::with_capacity(2);
    kernels.push(gate_up::emit(arch)?);
    kernels.push(down::emit(arch)?);
    Ok(kernels)
}
