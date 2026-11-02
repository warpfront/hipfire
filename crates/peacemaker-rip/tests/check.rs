//! `check`, `variants` and the located-error plumbing of the evaluator: native
//! `Err(String)` values are located at the call, entry-contract errors at
//! `fn emit`, and the four production kernels still emit the exact assembly
//! recorded before the diagnostics work, plus the RDNA code-end guard added by
//! 906c75705 (128-byte alignment and 96 `s_code_end` dwords). The refreshed
//! `s_text` hashes preserve all earlier instructions; ELF comparison with
//! `scripts/check-pm-code-end.py` proved only the guard and derived layout changed.

use peacemaker_isa::kernels::qsa_gather::Spec;
use peacemaker_isa::Arch;
use peacemaker_rip::diag::Severity;

const GATHER: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_gather.rip");
const SCORE: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score.rip");
const SCORE_BF16: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score_bf16.rip");
const TOPK: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_topk.rip");

const GATHER_GFX1151: &str = "67e4cc5b06fc3ac8f3aa9e2e49505eca020fdb3407fe0b37d03ee0c4b6609876";
const GATHER_GFX1201: &str = "d2e4a96e7c5a08b85d18d575cf02ba9962d0a0ad8fa912f11e2dda05160e3f52";
const SCORE_GFX1151: &str = "9129a70e4a6b6b2a8ee199ac18a52912611474284b59ac1583fad611f1064d10";
const SCORE_BF16_GFX1151: &str = "9891cf93054a2e1fc11dca29bcb735b73e2ce1d7cbbdd1c73b426231d5ad4257";
const SELECT_GFX1151: &str = "183d802686e532077ce0ae68b0f9904c60578352bc13981393ab7bfda3b08710";

/// A kernel script whose `emit` runs `body` inside a `Workgroup` scope.
/// `body` starts on line 13, so the position of anything in it is found with
/// [`pos`] rather than counted by hand.
fn kernel(body: &str) -> String {
    [
        "region T;",
        "fn emit(spec: Spec) -> Result<Emitted, String> {",
        "    let kspec = KernelSpec {",
        "        kernel_id: \"t\".into(), variant: \"v\".into(), arch: spec.arch, symbol: \"t_kernel\".into(),",
        "        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,",
        "        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,",
        "    };",
        "    let mut p = RegPlan::new(256, 104)?;",
        "    for i in 0..4u8 { p.v::<8>(\"vec\", 8 * i, Live::Whole)?; }",
        "    p.s::<8>(\"scalars\", 0, Live::Whole)?;",
        "    let mut b = Builder::new(kspec, p);",
        "    let mut wg = Workgroup::new(&mut b, spec.arch)?;",
        body,
        "    let end = wg.exit(\".Lend\")?;",
        "    wg.end(end)?;",
        "    b.finish()",
        "}",
        "",
    ]
    .join("\n")
}

/// 1-based (line, column) of the first `needle` in `src`, `skip` bytes in.
fn pos(src: &str, needle: &str, skip: usize) -> (usize, usize) {
    let at = src
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in the source"))
        + skip;
    let before = &src[..at];
    (
        before.matches('\n').count() + 1,
        before.rsplit('\n').next().unwrap().chars().count() + 1,
    )
}

const STORE: &str = concat!(
    "    let t = wg.lds::<T>(\"t\", 0, 16)?;\n",
    "    op(wg.isa(), \"v_mov_b32_e32 v1, 0\", &[v(1)], &[])?;\n",
    "    let (t, st) = wg.ds_store(t, Instruction::new(\"ds_store_b32 v0, v1\", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;\n",
);

#[test]
fn barrier_without_its_wait_is_located_at_the_barrier_call() {
    // The same handoff as `lds_handoff_needs_its_wait_before_the_barrier`, without the wait.
    let src = kernel(&format!(
        "{STORE}    let (t,) = wg.barrier((ready(t, st),))?;"
    ));
    let (line, col) = pos(&src, "wg.barrier", 3);
    assert_eq!((line, col), (16, 19), "fixture layout moved");

    let e = peacemaker_rip::compile_named(&src, "barrier.rip", Arch::Gfx1151, "v")
        .err()
        .expect("must be refused");
    eprintln!("ACTUAL barrier-without-wait error: {e}");
    let prefix = format!("barrier.rip:{line}:{col}: barrier would publish ");
    assert!(e.starts_with(&prefix), "{e}");
    assert!(e.contains("with an undrained LDS store"), "{e}");

    let checked = peacemaker_rip::check(&src, "barrier.rip", Arch::Gfx1151, "v");
    assert!(checked.emitted.is_none() && checked.resources.is_none());
    assert_eq!(checked.variant, "v");
    assert_eq!(checked.diagnostics.len(), 1, "{:?}", checked.diagnostics);
    let d = &checked.diagnostics[0];
    assert_eq!((d.severity, d.code), (Severity::Error, "R0301"), "{d:?}");
    assert_eq!(d.file, "barrier.rip");
    let span = d.span.expect("located");
    assert_eq!((span.line, span.col), (line, col), "{d:?}");
    assert!(
        d.message.starts_with("barrier would publish ") && !d.message.contains("barrier.rip:"),
        "{}",
        d.message
    );

    // The fixed program (wait first) is clean.
    let fixed = kernel(&format!(
        "{STORE}    let d = wg.wait(st)?;\n    let (t,) = wg.barrier((ready(t, d),))?;"
    ));
    let ok = peacemaker_rip::check(&fixed, "barrier.rip", Arch::Gfx1151, "v");
    assert!(ok.diagnostics.is_empty(), "{:?}", ok.diagnostics);
    assert!(ok.emitted.unwrap().s_text.contains("s_barrier"));
}

#[test]
fn an_op_on_an_undeclared_register_is_located_at_the_op_call() {
    // The plan declares v0..v31; v200 has no owner.
    let src = kernel("    op(wg.isa(), \"v_mov_b32_e32 v200, 0\", &[v(200)], &[])?;");
    let (line, col) = pos(&src, "op(wg.isa()", 0);
    assert_eq!((line, col), (13, 5), "fixture layout moved");

    let e = peacemaker_rip::compile_named(&src, "undeclared.rip", Arch::Gfx1151, "v")
        .err()
        .expect("must be refused");
    eprintln!("ACTUAL undeclared-register error: {e}");
    assert!(
        e.starts_with(&format!("undeclared.rip:{line}:{col}: v200 at ")),
        "{e}"
    );
    assert!(e.ends_with(" has no unique live owner"), "{e}");

    let checked = peacemaker_rip::check(&src, "undeclared.rip", Arch::Gfx1151, "v");
    assert!(checked.emitted.is_none() && checked.resources.is_none());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{:?}", checked.diagnostics);
    assert_eq!(errors[0].code, "R0401", "{:?}", errors[0]);
    let span = errors[0].span.expect("located");
    assert_eq!((span.line, span.col), (line, col));
}

#[test]
fn entry_contract_errors_are_located_at_the_name_of_fn_emit() {
    let compile = |src: &str| {
        peacemaker_rip::compile_named(src, "s.rip", Arch::Gfx1151, "v")
            .err()
            .expect("must fail")
    };
    // No `emit` at all: position 1:1.
    assert_eq!(
        compile("fn other() {}"),
        "s.rip:1:1: no `fn emit(spec: Spec) -> Result<Emitted, String>` entry point"
    );
    // Wrong arity: the name `emit` (column 4 of `fn emit`).
    assert_eq!(
        compile(
            "\n\nfn emit(a: u8, b: u8, c: u8) -> Result<Emitted, String> { Err(\"x\".into()) }"
        ),
        "s.rip:3:4: `emit` takes (spec) or (arch, variant), not 3 arguments"
    );
    // The script's own error stays bare text, positioned at the name.
    assert_eq!(
        compile("\nfn emit(spec: Spec) -> Result<Emitted, String> { Err(\"boom\".into()) }"),
        "s.rip:2:4: boom"
    );
    // The wrong Ok payload.
    let e = compile("fn emit(spec: Spec) -> Result<Emitted, String> { Ok(1) }");
    assert!(e.starts_with("s.rip:1:4: "), "{e}");
}

#[test]
fn the_scripts_own_err_is_r0502_at_the_name_of_emit() {
    let src = "\nfn emit(spec: Spec) -> Result<Emitted, String> { Err(\"boom\".into()) }";
    let checked = peacemaker_rip::check(src, "s.rip", Arch::Gfx1151, "v");
    assert!(checked.emitted.is_none() && checked.resources.is_none());
    assert_eq!(checked.diagnostics.len(), 1, "{:?}", checked.diagnostics);
    let d = &checked.diagnostics[0];
    assert_eq!((d.severity, d.code), (Severity::Error, "R0502"), "{d:?}");
    assert_eq!(d.message, "boom");
    assert!(d.help.is_some() && d.suggestion.is_none(), "{d:?}");
    let span = d.span.expect("located");
    assert_eq!((span.line, span.col), (2, 4));
    assert_eq!(span.end_line, 2);
    assert_eq!(span.end_col, 8, "the span covers the name `emit`: {d:?}");

    // A native error that reaches the result through `?` is not a script error.
    let src = kernel("    op(wg.isa(), \"v_mov_b32_e32 v200, 0\", &[v(200)], &[])?;");
    let checked = peacemaker_rip::check(&src, "n.rip", Arch::Gfx1151, "v");
    assert!(
        checked.diagnostics.iter().all(|d| d.code != "R0502"),
        "{:?}",
        checked.diagnostics
    );
}

#[test]
fn a_workgroup_scope_closed_by_a_builder_mention_is_explained() {
    // `let b = wg.isa();` mentions `b`: the scope opened on line 12 ends there (line 13), so the
    // next `wg.*` has no driver.
    let src = kernel("    let b = wg.isa();");
    let note = "the scope opened by `Workgroup::new(&mut b, ..)` at line 12 ended at line 13, the first statement mentioning `b`";

    let e = peacemaker_rip::compile_named(&src, "u.rip", Arch::Gfx1151, "v")
        .err()
        .expect("must be refused");
    assert!(e.starts_with("u.rip:14:"), "{e}");
    assert!(e.contains("no active Workgroup") && e.contains(note), "{e}");
    assert!(e.contains("`let b = wg.isa();`"), "{e}");

    let checked = peacemaker_rip::check(&src, "u.rip", Arch::Gfx1151, "v");
    assert_eq!(checked.diagnostics.len(), 1, "{:?}", checked.diagnostics);
    let d = &checked.diagnostics[0];
    assert_eq!((d.severity, d.code), (Severity::Error, "R0305"), "{d:?}");
    assert!(d.message.contains(note), "{}", d.message);
    assert_eq!(d.span.expect("located").line, 14);
}

#[test]
fn declared_registers_that_miss_what_the_instruction_touches_error_at_the_call() {
    // Initialize v1 so the missing declared use, rather than undefined input, is the error.
    let src = kernel(concat!(
        "    op(wg.isa(), \"v_mov_b32_e32 v1, 0\", &[v(1)], &[])?;\n",
        "    op(wg.isa(), \"v_mov_b32_e32 v0, v1\", &[v(0)], &[])?;",
    ));
    let (line, col) = pos(&src, "op(wg.isa(), \"v_mov_b32_e32 v0, v1\"", 0);
    assert_eq!((line, col), (14, 5), "fixture layout moved");
    let checked = peacemaker_rip::check(&src, "w.rip", Arch::Gfx1151, "v");
    assert!(
        checked.emitted.is_none() && checked.resources.is_none(),
        "{:?}",
        checked.diagnostics
    );
    assert_eq!(checked.diagnostics.len(), 1, "{:?}", checked.diagnostics);
    let d = &checked.diagnostics[0];
    assert_eq!((d.severity, d.code), (Severity::Error, "R0603"), "{d:?}");
    assert_eq!(d.file, "w.rip");
    let span = d.span.expect("located");
    assert_eq!((span.line, span.col), (line, col));
    let e = peacemaker_rip::compile_named(&src, "w.rip", Arch::Gfx1151, "v")
        .err()
        .expect("must be refused");
    assert!(e.starts_with(&format!("w.rip:{line}:{col}: ")), "{e}");
    assert!(e.contains(&d.message), "{e}");
}

#[test]
fn check_reports_resources_from_the_descriptor_and_the_kernel_body() {
    let src = kernel("    op(wg.isa(), \"v_mov_b32_e32 v0, 0\", &[v(0)], &[])?;");
    let checked = peacemaker_rip::check(&src, "ok.rip", Arch::Gfx1151, "v");
    assert!(checked.diagnostics.is_empty(), "{:?}", checked.diagnostics);
    let e = checked.emitted.as_ref().expect("emitted");
    let r = checked.resources.as_ref().expect("resources");
    assert_eq!(r.lds_bytes, 1024);
    assert!(
        e.s_text
            .contains(&format!(".amdhsa_next_free_vgpr {}", r.vgprs)),
        "{}",
        e.s_text
    );
    assert!(
        e.s_text
            .contains(&format!(".amdhsa_next_free_sgpr {}", r.sgprs)),
        "{}",
        e.s_text
    );
    // v_mov_b32 and s_endpgm at least.
    assert!(r.instructions >= 2, "{r:?}");
    // The checked program is the compiled one.
    let compiled = peacemaker_rip::compile_named(&src, "ok.rip", Arch::Gfx1151, "v").unwrap();
    assert_eq!(e.s_text, compiled.s_text);
}

fn errors_of(c: &peacemaker_rip::Checked) -> Vec<&peacemaker_rip::diag::Diagnostic> {
    c.diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

#[test]
fn production_kernel_resources_match_the_recorded_descriptors() {
    let convert = peacemaker_rip::check(GATHER, "qsa_gather.rip", Arch::Gfx1151, "convert");
    assert!(errors_of(&convert).is_empty(), "{:?}", convert.diagnostics);
    assert_eq!(
        convert.resources,
        Some(peacemaker_rip::Resources {
            vgprs: 16,
            sgprs: 73,
            lds_bytes: 0,
            instructions: 62
        })
    );
    let attend = peacemaker_rip::check(GATHER, "qsa_gather.rip", Arch::Gfx1151, "attend");
    assert!(errors_of(&attend).is_empty(), "{:?}", attend.diagnostics);
    assert_eq!(
        attend.resources,
        Some(peacemaker_rip::Resources {
            vgprs: 160,
            sgprs: 78,
            lds_bytes: 18432,
            instructions: 4476
        })
    );
}

#[test]
fn variants_are_the_declared_kinds() {
    assert_eq!(
        peacemaker_rip::variants(GATHER, "qsa_gather.rip").unwrap(),
        ["Convert", "Attend"]
    );
    // `const VARIANTS` beats the missing `enum Kind` on the single-kernel sources.
    assert_eq!(
        peacemaker_rip::variants(SCORE, "qsa_score.rip").unwrap(),
        ["score", "score32"]
    );
    assert_eq!(
        peacemaker_rip::variants(SCORE_BF16, "qsa_score_bf16.rip").unwrap(),
        ["score-bf16", "score-bf16-rows32"]
    );
    assert_eq!(
        peacemaker_rip::variants(TOPK, "qsa_topk.rip").unwrap(),
        ["select"]
    );
    // Neither const nor `enum Kind`: no variants (the caller names one).
    let bare =
        "fn emit(arch: Arch, variant: &str) -> Result<Emitted, String> { Err(\"x\".into()) }";
    assert!(peacemaker_rip::variants(bare, "bare.rip")
        .unwrap()
        .is_empty());
    // A syntax error is a located error.
    let e = peacemaker_rip::variants("fn {", "bad.rip")
        .err()
        .expect("syntax error");
    assert!(e.starts_with("bad.rip:1:"), "{e}");
}

#[test]
fn arches_are_the_declared_arches() {
    assert_eq!(
        peacemaker_rip::arches(GATHER, "qsa_gather.rip").unwrap(),
        Some(vec!["gfx1151".to_string(), "gfx1201".to_string()])
    );
    assert_eq!(
        peacemaker_rip::arches(SCORE, "qsa_score.rip").unwrap(),
        Some(vec!["gfx1151".to_string()])
    );
    assert_eq!(
        peacemaker_rip::arches(SCORE_BF16, "qsa_score_bf16.rip").unwrap(),
        Some(vec!["gfx1151".to_string()])
    );
    assert_eq!(
        peacemaker_rip::arches(TOPK, "qsa_topk.rip").unwrap(),
        Some(vec!["gfx1151".to_string()])
    );
    // No `const ARCHES`: no declared arches (the caller picks the default).
    let bare =
        "fn emit(arch: Arch, variant: &str) -> Result<Emitted, String> { Err(\"x\".into()) }";
    assert_eq!(peacemaker_rip::arches(bare, "bare.rip").unwrap(), None);
    // A declared-but-malformed const is an error naming the const.
    let bad = "pub const ARCHES: &[&str] = &[1u32];\nfn emit(arch: Arch, variant: &str) -> Result<Emitted, String> { Err(\"x\".into()) }";
    let e = peacemaker_rip::arches(bad, "bad.rip")
        .err()
        .expect("malformed");
    assert!(e.contains("`ARCHES`"), "{e}");
    assert!(e.starts_with("bad.rip:"), "{e}");
}

/// The module text and every kernel of the gather module hash to the recorded baseline.
#[test]
fn gather_module_assembly_is_byte_identical_to_the_baseline() {
    for (arch, hash) in [
        (Arch::Gfx1151, GATHER_GFX1151),
        (Arch::Gfx1201, GATHER_GFX1201),
    ] {
        let name = format!("qsa_gather_pm_{}", arch.name());
        assert_eq!(Spec::module(arch).to_string(), name);
        let (kernels, _text, proof) = peacemaker_rip::module_named(
            GATHER,
            "qsa_gather.rip",
            arch,
            &["convert", "attend"],
            &name,
        )
        .unwrap();
        assert_eq!(proof.module, name);
        assert_eq!(kernels.len(), 2);
        assert_eq!(proof.s_text_sha256, hash, "{} module assembly", arch.name());
    }
}

/// The three single-kernel select sources hash to the recorded baseline on gfx1151.
#[test]
fn select_kernels_assembly_is_byte_identical_to_the_baseline() {
    let arch = Arch::Gfx1151;
    for (src, name, variant, hash) in [
        (SCORE, "qsa_score.rip", "score", SCORE_GFX1151),
        (
            SCORE_BF16,
            "qsa_score_bf16.rip",
            "score-bf16",
            SCORE_BF16_GFX1151,
        ),
        (TOPK, "qsa_topk.rip", "select", SELECT_GFX1151),
    ] {
        let e = peacemaker_rip::compile_named(src, name, arch, variant)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(e.proof.s_text_sha256, hash, "{name} {variant} assembly");
        let checked = peacemaker_rip::check(src, name, arch, variant);
        assert!(
            errors_of(&checked).is_empty(),
            "{name}: {:?}",
            checked.diagnostics
        );
        assert_eq!(checked.emitted.expect("emitted").s_text, e.s_text);
    }
}
