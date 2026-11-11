// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave-private LDS misuse as rustc errors: no barrier transition and no
//! shared load can name a `WaveRegion` (so no other wave can be given access
//! to it), and its slice is read only once `Owned`. The positive control is
//! the legal protocol under wave-uniform control.
#[test]
fn private_region_misuse_is_a_compile_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/private_cross_wave_publish.rs");
    t.compile_fail("tests/ui/private_shared_load.rs");
    t.compile_fail("tests/ui/private_load_before_publish.rs");
    t.pass("tests/ui/private_fixed.rs");
}
