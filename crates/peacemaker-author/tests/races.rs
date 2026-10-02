//! The two shipped LDS races as rustc errors (design §9), each with a
//! positive control that compiles and runs on the reference backend.
#[test]
fn race_classes_are_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/halo_verify_attn_race.rs");
    t.compile_fail("tests/ui/fa2_mailbox_race.rs");
    t.compile_fail("tests/ui/barrier_in_wave_scope.rs");
    t.pass("tests/ui/halo_verify_attn_fixed.rs");
    t.pass("tests/ui/fa2_mailbox_fixed.rs");
}
