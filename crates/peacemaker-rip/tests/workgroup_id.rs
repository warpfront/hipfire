//! The `workgroup_id(b, axis, dst)` builtin: the architected TTMP reads on gfx12,
//! the system SGPRs after the user SGPRs on gfx11, and the refusals that keep a
//! kernel from reading a work-group id its descriptor does not enable.

use peacemaker_isa::Arch;
use peacemaker_rip::diag::Severity;

/// A kernel whose `emit` runs `body` inside a `Workgroup` scope. The plan owns the
/// kernarg pointer `s0:1`, the output scalars `s4:7`, and (when `own_ids`) the
/// system SGPRs `s2:3` that gfx11 launches the work-group ids in.
fn kernel(y: bool, own_ids: bool, body: &str) -> String {
    let mut lines = vec![
        "fn emit(spec: Spec) -> Result<Emitted, String> {".to_string(),
        "    let kspec = KernelSpec {".to_string(),
        "        kernel_id: \"t\".into(), variant: \"v\".into(), arch: spec.arch, symbol: \"t_kernel\".into(),".to_string(),
        format!("        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: {y},"),
        "        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,".to_string(),
        "    };".to_string(),
        "    let mut p = RegPlan::new(256, 104)?;".to_string(),
        "    p.s::<2>(\"karg\", 0, Live::Whole)?;".to_string(),
    ];
    if own_ids {
        lines.push("    p.s::<2>(\"ids\", 2, Live::Whole)?;".to_string());
    }
    lines.extend([
        "    p.s::<4>(\"out\", 4, Live::Whole)?;".to_string(),
        "    let mut b = Builder::new(kspec, p);".to_string(),
        "    let mut wg = Workgroup::new(&mut b, spec.arch)?;".to_string(),
        body.to_string(),
        "    let end = wg.exit(\".Lend\")?;".to_string(),
        "    wg.end(end)?;".to_string(),
        "    b.finish()".to_string(),
        "}".to_string(),
        String::new(),
    ]);
    lines.join("\n")
}

fn compile(source: &str, arch: Arch) -> Result<String, String> {
    peacemaker_rip::compile_named(source, "wg.rip", arch, "v").map(|emitted| emitted.s_text)
}

const X_AND_Y: &str = "    workgroup_id(wg.isa(), 0, 4)?;\n    workgroup_id(wg.isa(), 1, 5)?;";

#[test]
fn gfx12_reads_the_architected_ttmps() {
    let s = compile(&kernel(true, false, X_AND_Y), Arch::Gfx1201).unwrap();
    assert!(s.contains("s_mov_b32 s4, ttmp9"), "{s}");
    assert!(s.contains("s_and_b32 s5, ttmp7, 0xffff"), "{s}");
    assert!(!s.contains("ttmp7, 16"), "{s}");
}

#[test]
fn gfx12_x_needs_no_y_flag_and_no_system_sgpr_ownership() {
    let s = compile(
        &kernel(false, false, "    workgroup_id(wg.isa(), 0, 4)?;"),
        Arch::Gfx1201,
    )
    .unwrap();
    assert!(s.contains("s_mov_b32 s4, ttmp9"), "{s}");
}

#[test]
fn gfx11_reads_the_system_sgprs_after_the_user_sgprs() {
    for arch in [Arch::Gfx1151, Arch::Gfx1100] {
        let s = compile(
            &kernel(
                true,
                true,
                "    workgroup_id(wg.isa(), 0, 4)?;\n    workgroup_id(wg.isa(), 1, 5)?;",
            ),
            arch,
        )
        .unwrap();
        assert!(s.contains("s_mov_b32 s4, s2"), "{arch:?}: {s}");
        assert!(s.contains("s_mov_b32 s5, s3"), "{arch:?}: {s}");
        assert!(!s.contains("ttmp"), "{arch:?}: {s}");
    }
}

#[test]
fn an_invalid_axis_names_the_three_axes() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for axis in ["3", "255", "-1"] {
            let src = kernel(
                true,
                true,
                &format!("    workgroup_id(wg.isa(), {axis}, 4)?;"),
            );
            let e = compile(&src, arch).err().expect("must be refused");
            assert!(e.starts_with("wg.rip:"), "{e}");
            assert!(
                e.contains("workgroup_id: axis must be 0 (x), 1 (y) or 2 (z)")
                    && e.contains(&format!("got {axis}")),
                "{arch:?}: {e}"
            );
        }
    }
}

#[test]
fn y_without_the_spec_flag_is_refused_with_the_fix() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let src = kernel(false, true, "    workgroup_id(wg.isa(), 1, 5)?;");
        let e = compile(&src, arch).err().expect("must be refused");
        assert!(
            e.contains("workgroup_id: axis 1 (y) needs the Y work-group id enabled"),
            "{arch:?}: {e}"
        );
        assert!(
            e.contains("system_sgpr_workgroup_id_y: true"),
            "{arch:?}: {e}"
        );
    }
}

#[test]
fn z_is_refused_on_every_arch_even_with_the_y_flag() {
    for arch in [Arch::Gfx1151, Arch::Gfx1100, Arch::Gfx1201] {
        for y in [false, true] {
            let src = kernel(y, true, "    workgroup_id(wg.isa(), 2, 6)?;");
            let e = compile(&src, arch).err().expect("must be refused");
            assert!(e.starts_with("wg.rip:"), "{e}");
            assert!(
                e.ends_with(
                    "workgroup id z is not enabled: KernelSpec has no z flag; fold z into x or y"
                ),
                "{arch:?} y={y}: {e}"
            );
        }
    }
}

#[test]
fn an_unowned_gfx11_source_is_r0401() {
    let src = kernel(false, false, "    workgroup_id(wg.isa(), 0, 4)?;");
    let e = compile(&src, Arch::Gfx1151).err().expect("must be refused");
    assert!(
        e.contains("s2 at ") && e.ends_with(" has no unique live owner"),
        "{e}"
    );

    let checked = peacemaker_rip::check(&src, "wg.rip", Arch::Gfx1151, "v");
    assert!(checked.emitted.is_none());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{:?}", checked.diagnostics);
    assert_eq!(errors[0].code, "R0401", "{:?}", errors[0]);

    // gfx12 reads ttmp9: the same plan is fine there.
    assert!(compile(&src, Arch::Gfx1201).is_ok());
}

#[test]
fn a_valid_kernel_checks_clean_on_both_architectures() {
    for (arch, own_ids) in [(Arch::Gfx1151, true), (Arch::Gfx1201, false)] {
        let src = kernel(
            true,
            own_ids,
            "    workgroup_id(wg.isa(), 0, 4)?;\n    workgroup_id(wg.isa(), 1, 5)?;",
        );
        let checked = peacemaker_rip::check(&src, "wg.rip", arch, "v");
        assert!(
            checked.diagnostics.is_empty(),
            "{arch:?}: {:?}",
            checked.diagnostics
        );
        assert!(checked.emitted.is_some());
    }
}

#[test]
fn the_name_is_a_known_builtin() {
    assert!(peacemaker_rip::native::is_fn("workgroup_id"));
}
