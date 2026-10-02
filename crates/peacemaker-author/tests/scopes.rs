//! Scope escapes as rustc errors: the lowering seam without an `Auth`, a
//! forged or minted `Auth`, a barrier in the readers-only continuation of a
//! handoff or in a wave-scope loop. Escapes the types cannot see
//! (rebuilding a `Workgroup` from wave scope, anything after the kernel
//! exit) are refused at run time (`tests/control.rs`).
#[test]
fn scope_escapes_are_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/backend_seam_in_wave_scope.rs");
    t.compile_fail("tests/ui/forged_auth.rs");
    t.compile_fail("tests/ui/minted_auth.rs");
    t.compile_fail("tests/ui/barrier_after_handoff.rs");
    t.compile_fail("tests/ui/barrier_in_loop_until.rs");
    t.pass("tests/ui/handoff_fixed.rs");
}
