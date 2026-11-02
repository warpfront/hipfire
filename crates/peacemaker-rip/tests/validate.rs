//! Front-end instruction validation: `peacemaker_rip::validate` against the
//! contract (unknown mnemonics and malformed operands are errors; a def/use
//! list that omits a register the text names is an R0603 error carrying
//! the corrected Rust lists), the native `op` / `mem` / `sop` calls, the
//! R0604 read-before-definition check, and `Instruction`s consumed by the
//! driver; then the four production QSA sources through `check`, with
//! R0603 and R0604 counts reported separately.

use peacemaker_isa::reg::{Kind, RegRef};
use peacemaker_isa::Arch;
use peacemaker_rip::diag::{Diagnostic, Severity};
use peacemaker_rip::validate;

fn v(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    }
}
fn s(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 1,
    }
}
fn vr(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len,
    }
}
fn sr(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base,
        len,
    }
}

fn has(text: &str, needle: &str) -> bool {
    text.to_lowercase().contains(&needle.to_lowercase())
}

type Set = Vec<(bool, u8, u8)>;

/// Registers as a sorted, de-duplicated set, so effects compare independent of operand order.
fn set(regs: &[RegRef]) -> Set {
    let mut out: Set = regs
        .iter()
        .map(|r| (r.kind == Kind::V, r.base, r.len))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn effects(arch: Arch, text: &str) -> (Set, Set) {
    let (defs, uses) =
        validate::effects(arch, text).unwrap_or_else(|e| panic!("effects({text}): {e}"));
    (set(&defs), set(&uses))
}

// ---------------------------------------------------------------- unit: validate::instruction

#[test]
fn gfx12_has_no_s_add_i32_and_the_error_names_the_gfx11_spelling_and_its_replacement() {
    let e = validate::instruction(Arch::Gfx1201, "s_add_i32 s1, s2, 1", &[s(1)], &[s(2)])
        .err()
        .expect("s_add_i32 is unknown on gfx1201");
    assert!(e.contains("unknown mnemonic `s_add_i32` on gfx1201"), "{e}");
    assert!(e.contains("known on gfx11, not gfx12"), "{e}");
    assert!(e.contains("did you mean `s_add_co_i32`?"), "{e}");
    // The same text is the gfx11 spelling, and the gfx12 spelling is valid on gfx12.
    validate::instruction(Arch::Gfx1151, "s_add_i32 s1, s2, 1", &[s(1)], &[s(2)]).unwrap();
    validate::instruction(Arch::Gfx1201, "s_add_co_i32 s1, s2, 1", &[s(1)], &[s(2)]).unwrap();
}

#[test]
fn a_mistyped_mnemonic_suggests_the_closest_real_one() {
    for (typo, real) in [
        ("s_mov_b3", "s_mov_b32"),
        ("s_cmp_eq_u3", "s_cmp_eq_u32"),
        ("ds_stor_b32", "ds_store_b32"),
    ] {
        let e = validate::instruction(Arch::Gfx1151, &format!("{typo} s0, 0"), &[s(0)], &[])
            .err()
            .unwrap_or_else(|| panic!("{typo} is not a mnemonic"));
        assert!(e.contains(&format!("unknown mnemonic `{typo}`")), "{e}");
        assert!(e.contains(&format!("did you mean `{real}`?")), "{e}");
    }
}

#[test]
fn malformed_operands_are_errors_with_a_reason() {
    for text in [
        "v_mov_b32_e32 v[2:, 0",
        "v_mov_b32_e32 v[5:2], 0",
        "v_mov_b32_e32 v0,, 0",
        "s_mov_b32",
    ] {
        let e = validate::instruction(Arch::Gfx1151, text, &[v(0)], &[])
            .err()
            .unwrap_or_else(|| panic!("`{text}` must be rejected"));
        let mnemonic = text.split_whitespace().next().unwrap();
        assert!(
            e.contains(&format!("malformed operands for `{mnemonic}` on gfx1151: ")),
            "`{text}`: {e}"
        );
        assert!(
            e.split_once(": ")
                .is_some_and(|(_, reason)| !reason.trim().is_empty()),
            "`{text}` rejected without a reason: {e}"
        );
        assert!(
            !has(&e, "unknown mnemonic"),
            "`{text}`: the mnemonic is fine, the operands are not: {e}"
        );
    }
}

#[test]
fn a_use_missing_from_the_list_is_an_error_with_the_corrected_lists() {
    // `s3` is read by the text but absent from the declared uses.
    let e = validate::instruction(Arch::Gfx1151, "s_add_i32 s1, s2, s3", &[s(1)], &[s(2)])
        .err()
        .expect("a missing use is an error");
    assert!(e.contains("op declares defs [s1] and uses [s2] but `s_add_i32 s1, s2, s3` also reads s3: write &[s(1)], &[s(2), s(3)]"), "{e}");
}

#[test]
fn a_def_missing_from_the_list_is_an_error_naming_it() {
    let e = validate::instruction(Arch::Gfx1151, "s_add_i32 s1, s2, 5", &[], &[s(2)])
        .err()
        .expect("a missing def is an error");
    assert!(e.contains("op declares defs [] and uses [s2] but `s_add_i32 s1, s2, 5` also writes s1: write &[s(1)], &[s(2)]"), "{e}");
    // Both lists short: one error, both corrections, declared extras preserved.
    let e = validate::instruction(Arch::Gfx1151, "s_add_i32 s1, s2, s3", &[s(7)], &[])
        .err()
        .expect("both lists short is an error");
    assert!(
        e.contains("also writes s1 and reads s2, s3: write &[s(7), s(1)], &[s(2), s(3)]"),
        "{e}"
    );
}

#[test]
fn a_correct_instruction_is_accepted() {
    validate::instruction(Arch::Gfx1151, "v_mov_b32_e32 v0, 0", &[v(0)], &[]).unwrap();
    validate::instruction(Arch::Gfx1151, "s_add_i32 s1, s1, 1", &[s(1)], &[s(1)]).unwrap();
    validate::instruction(
        Arch::Gfx1201,
        "s_and_b32 exec_lo, exec_lo, s5",
        &[],
        &[s(5)],
    )
    .unwrap();
    validate::instruction(
        Arch::Gfx1201,
        "v_wmma_f32_16x16x16_f16 v[8:15], v[16:19], v[24:27], v[8:15]",
        &[vr(8, 8)],
        &[vr(16, 4), vr(24, 4), vr(8, 8)],
    )
    .unwrap();
}

#[test]
fn gfx942_is_refused_by_the_rdna_only_validator_and_rdna_targets_are_unaffected() {
    const REFUSAL: &str =
        "the .rip front end lowers onto the RDNA builder; gfx942 kernels use peacemaker_isa::cdna";
    let e = validate::instruction(Arch::Gfx942, "s_mov_b32 s0, 0", &[s(0)], &[])
        .err()
        .expect("gfx942 has no .rip lowering");
    assert_eq!(e, REFUSAL);
    let e = validate::effects(Arch::Gfx942, "s_mov_b32 s0, 0")
        .err()
        .expect("gfx942 has no .rip lowering");
    assert_eq!(e, REFUSAL);
    // Even text that is not a mnemonic reports the architecture refusal, not an RDNA parse error.
    let e = validate::instruction(Arch::Gfx942, "not_a_mnemonic", &[], &[])
        .err()
        .expect("gfx942 has no .rip lowering");
    assert_eq!(e, REFUSAL);
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        validate::instruction(arch, "s_mov_b32 s0, 0", &[s(0)], &[]).unwrap();
        validate::effects(arch, "s_mov_b32 s0, 0").unwrap();
    }
}

#[test]
fn any_same_kind_intersecting_range_covers_an_operand() {
    // Partial overlap is covered by contract: a declared v[8:9] intersects the text's v[8:15],
    // a declared v[2:3] intersects the text's v3, a declared s[0:1] intersects s1.
    validate::instruction(
        Arch::Gfx1201,
        "v_wmma_f32_16x16x16_f16 v[8:15], v[16:19], v[24:27], v[8:15]",
        &[vr(8, 2)],
        &[vr(16, 1), vr(27, 1), vr(15, 1)],
    )
    .unwrap();
    validate::instruction(
        Arch::Gfx1151,
        "v_mul_f32_e32 v1, v3, v4",
        &[v(1)],
        &[vr(2, 2), v(4)],
    )
    .unwrap();
    validate::instruction(
        Arch::Gfx1151,
        "s_add_u32 s4, s1, s6",
        &[s(4)],
        &[sr(0, 2), s(6)],
    )
    .unwrap();
    // Kind matters: a VGPR v5 does not cover the SGPR s5 the text reads.
    let e = validate::instruction(Arch::Gfx1151, "s_mov_b32 s4, s5", &[s(4)], &[v(5)])
        .err()
        .expect("kind mismatch is an error");
    assert!(e.contains("op declares defs [s4] and uses [v5] but `s_mov_b32 s4, s5` also reads s5: write &[s(4)], &[v(5), s(5)]"), "{e}");
    // So does a disjoint range of the same kind.
    let e = validate::instruction(Arch::Gfx1151, "s_mov_b32 s4, s9", &[s(4)], &[sr(5, 3)])
        .err()
        .expect("disjoint range is an error");
    assert!(e.contains("op declares defs [s4] and uses [s[5:7]] but `s_mov_b32 s4, s9` also reads s9: write &[s(4)], &[sr(5, 3), s(9)]"), "{e}");
}

// ---------------------------------------------------------------- unit: validate::effects

#[test]
fn effects_split_v_and_s_operands_into_defs_and_uses() {
    let a = Arch::Gfx1151;
    assert_eq!(
        effects(a, "v_mul_f32_e32 v1, v2, v3"),
        (set(&[v(1)]), set(&[v(2), v(3)]))
    );
    assert_eq!(effects(a, "v_mov_b32_e32 v0, 0"), (set(&[v(0)]), set(&[])));
    assert_eq!(
        effects(a, "s_add_i32 s1, s1, s2"),
        (set(&[s(1)]), set(&[s(1), s(2)]))
    );
    // A compare defines no V/S register: both operands are read.
    assert_eq!(effects(a, "s_cmp_eq_u32 s6, 0"), (set(&[]), set(&[s(6)])));
    assert_eq!(
        effects(a, "s_cmp_lt_u32 s6, s7"),
        (set(&[]), set(&[s(6), s(7)]))
    );
    // Ranges keep their width.
    assert_eq!(
        effects(a, "s_load_b256 s[8:15], s[0:1], null"),
        (set(&[sr(8, 8)]), set(&[sr(0, 2)]))
    );
    assert_eq!(
        effects(a, "buffer_load_b128 v[4:7], v0, s[8:11], 0 offen"),
        (set(&[vr(4, 4)]), set(&[v(0), sr(8, 4)]))
    );
    // A store reads its address and data, defines nothing.
    assert_eq!(
        effects(a, "ds_store_b32 v2, v0 offset:4"),
        (set(&[]), set(&[v(2), v(0)]))
    );
    assert_eq!(
        effects(a, "ds_load_b32 v2, v0"),
        (set(&[v(2)]), set(&[v(0)]))
    );
}

#[test]
fn effects_ignore_special_registers() {
    // exec, vcc, m0, null and ttmp are not RegPlan registers: they appear in neither list.
    assert_eq!(
        effects(Arch::Gfx1201, "s_and_b32 exec_lo, exec_lo, s5"),
        (set(&[]), set(&[s(5)]))
    );
    assert_eq!(
        effects(Arch::Gfx1151, "s_and_b32 s1, ttmp7, 0xffff"),
        (set(&[s(1)]), set(&[]))
    );
    assert_eq!(
        effects(Arch::Gfx1151, "v_cndmask_b32_e64 v0, v1, v2, vcc_lo"),
        (set(&[v(0)]), set(&[v(1), v(2)]))
    );
    assert_eq!(
        effects(Arch::Gfx1151, "s_mov_b32 s3, m0"),
        (set(&[s(3)]), set(&[]))
    );
}

#[test]
fn effects_reject_what_instruction_rejects() {
    let e = validate::effects(Arch::Gfx1201, "s_add_i32 s1, s2, 1")
        .err()
        .expect("unknown on gfx1201");
    assert!(
        e.contains("unknown mnemonic `s_add_i32`") && e.contains("did you mean `s_add_co_i32`?"),
        "{e}"
    );
    let e = validate::effects(Arch::Gfx1151, "s_mov_b3 s0, 0")
        .err()
        .expect("typo");
    assert!(
        e.contains("unknown mnemonic") && e.contains("did you mean `s_mov_b32`?"),
        "{e}"
    );
    let e = validate::effects(Arch::Gfx1151, "v_mov_b32_e32 v[2:, 0")
        .err()
        .expect("malformed");
    assert!(e.contains("malformed operands"), "{e}");
}

// ---------------------------------------------------------------- scripts: native op / mem / sop / Instruction::new

/// A one-wave kernel around `body`, which runs inside the open `wg` scope
/// (SGPRs `s0..s7` and VGPRs `v0..v31` are planned).
fn kernel(body: &str) -> String {
    format!(
        r#"region T;
fn emit(spec: Spec) -> Result<Emitted, String> {{
    let kspec = KernelSpec {{
        kernel_id: "t".into(), variant: "v".into(), arch: spec.arch, symbol: "t_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    }};
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 {{ p.v::<8>("vec", 8 * i, Live::Whole)?; }}
    p.s::<8>("scalars", 0, Live::Whole)?;
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    {body}
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}}
"#
    )
}

/// `Diagnostic` flattened to everything it says, for substring checks.
fn said(d: &Diagnostic) -> String {
    format!(
        "{} {} {}",
        d.message,
        d.help.as_deref().unwrap_or(""),
        d.notes.join(" ")
    )
}
fn errors(ds: &[Diagnostic]) -> Vec<String> {
    ds.iter()
        .filter(|d| d.severity == Severity::Error)
        .map(said)
        .collect()
}
fn warnings(ds: &[Diagnostic]) -> Vec<String> {
    ds.iter()
        .filter(|d| d.severity != Severity::Error)
        .map(said)
        .collect()
}

/// Check `body` on `arch`; returns (errors, warnings, emitted?).
fn run(body: &str, arch: Arch) -> (Vec<String>, Vec<String>, bool) {
    let c = peacemaker_rip::check(&kernel(body), "validate.rip", arch, "v");
    (
        errors(&c.diagnostics),
        warnings(&c.diagnostics),
        c.emitted.is_some(),
    )
}

/// Like `run`, but keeps the diagnostics (code, severity, span) whole.
fn run_diags(body: &str, arch: Arch) -> (Vec<Diagnostic>, bool) {
    let c = peacemaker_rip::check(&kernel(body), "validate.rip", arch, "v");
    (c.diagnostics, c.emitted.is_some())
}

fn coded<'a>(ds: &'a [Diagnostic], code: &str) -> Vec<&'a Diagnostic> {
    ds.iter().filter(|d| d.code == code).collect()
}

/// 1-based (line, column) of the first `needle` in `src`.
fn pos(src: &str, needle: &str) -> (usize, usize) {
    let at = src
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in the source"));
    let before = &src[..at];
    (
        before.matches('\n').count() + 1,
        before.rsplit('\n').next().unwrap().chars().count() + 1,
    )
}

/// Reads `s2` with a complete def/use list: gfx11 defines it (workgroup X id), gfx1201 does not.
const READ_S2: &str = r#"op(wg.isa(), "s_mov_b32 s5, s2", &[s(5)], &[s(2)])?;"#;

const GOOD: &str = r#"op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;"#;

#[test]
fn a_valid_native_op_emits_without_diagnostics() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (errs, warns, emitted) = run(GOOD, arch);
        assert!(
            errs.is_empty() && warns.is_empty() && emitted,
            "{}: errors {errs:?} warnings {warns:?} emitted {emitted}",
            arch.name()
        );
    }
}

#[test]
fn native_sop_with_the_gfx11_spelling_is_an_error_on_gfx12_before_the_builder_sees_it() {
    let body = r#"sop(wg.isa(), "s_add_i32 s1, s1, 1", &[1], &[1])?;"#;
    let (errs, _, emitted) = run(body, Arch::Gfx1201);
    assert!(!emitted && errs.len() == 1, "{errs:?}");
    assert!(
        errs[0].contains("unknown mnemonic `s_add_i32` on gfx1201")
            && errs[0].contains("known on gfx11, not gfx12")
            && errs[0].contains("s_add_co_i32"),
        "{}",
        errs[0]
    );
    let (errs, warns, emitted) = run(body, Arch::Gfx1151);
    assert!(
        errs.is_empty() && warns.is_empty() && emitted,
        "{errs:?} {warns:?}"
    );
    // The arch-spelling helper is the supported way to write it for both.
    let body = r#"sop(wg.isa(), format!("{} s1, s1, 1", s_add_i32(spec.arch)), &[1], &[1])?;"#;
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (errs, warns, emitted) = run(body, arch);
        assert!(
            errs.is_empty() && warns.is_empty() && emitted,
            "{}: {errs:?} {warns:?}",
            arch.name()
        );
    }
}

#[test]
fn native_op_rejects_a_typo_with_a_suggestion_and_malformed_operands_with_a_reason() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (errs, _, emitted) = run(r#"op(wg.isa(), "s_mov_b3 s1, 0", &[s(1)], &[])?;"#, arch);
        assert!(!emitted && errs.len() == 1, "{errs:?}");
        assert!(
            errs[0].contains("unknown mnemonic `s_mov_b3`")
                && errs[0].contains("did you mean `s_mov_b32`?"),
            "{}",
            errs[0]
        );
        let (errs, _, emitted) = run(
            r#"op(wg.isa(), "v_mov_b32_e32 v[5:2], 0", &[v(0)], &[])?;"#,
            arch,
        );
        assert!(
            !emitted
                && errs.len() == 1
                && errs[0].contains("malformed operands for `v_mov_b32_e32`"),
            "{errs:?}"
        );
    }
}

#[test]
fn native_op_with_a_missing_use_is_an_r0603_error_at_the_call_and_does_not_emit() {
    // v1 and v2 are initialised first, so the only diagnostic is the def/use list one.
    let body = r#"op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(wg.isa(), "v_mov_b32_e32 v2, 0", &[v(2)], &[])?;
    op(wg.isa(), "v_mul_f32_e32 v0, v1, v2", &[v(0)], &[v(1)])?;"#;
    let (line, col) = pos(&kernel(body), r#"op(wg.isa(), "v_mul_f32_e32"#);
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (ds, emitted) = run_diags(body, arch);
        assert!(!emitted, "{}: {ds:?}", arch.name());
        assert_eq!(ds.len(), 1, "{}: {ds:?}", arch.name());
        let d = &ds[0];
        assert_eq!(
            (d.code, d.severity),
            ("R0603", Severity::Error),
            "{}: {d:?}",
            arch.name()
        );
        let span = d.span.expect("located");
        assert_eq!((span.line, span.col), (line, col), "{}: {d:?}", arch.name());
        assert!(
            said(d).contains("also reads v2: write &[v(0)], &[v(1), v(2)]"),
            "{}: {d:?}",
            arch.name()
        );
    }
}

#[test]
fn native_sop_with_a_missing_use_is_an_r0603_error() {
    // s3 is initialised first (s2 is the workgroup X id on gfx11).
    let body = r#"sop(wg.isa(), "s_mov_b32 s3, 0", &[3], &[])?;
    sop(wg.isa(), "s_and_b32 s1, s2, s3", &[1], &[2])?;"#;
    let (ds, emitted) = run_diags(body, Arch::Gfx1151);
    assert!(!emitted, "{ds:?}");
    assert_eq!(ds.len(), 1, "{ds:?}");
    assert_eq!(
        (ds[0].code, ds[0].severity),
        ("R0603", Severity::Error),
        "{:?}",
        ds[0]
    );
    assert!(
        said(&ds[0]).contains("also reads s3: write &[s(1)], &[s(2), s(3)]"),
        "{:?}",
        ds[0]
    );
}

#[test]
fn native_mem_is_validated_like_op() {
    // An unknown memory mnemonic never reaches the builder ...
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (errs, _, emitted) = run(
            r#"mem(wg.isa(), "buffer_lod_b32 v1, v0, s[4:7], 0 offen", &[v(1)], &[v(0), sr(4, 4)], MemoryClass::VmemLoad)?;"#,
            arch,
        );
        assert!(!emitted && errs.len() == 1, "{}: {errs:?}", arch.name());
        assert!(
            errs[0].contains("unknown mnemonic `buffer_lod_b32`")
                && errs[0].contains("did you mean `buffer_load_b32`?"),
            "{}",
            errs[0]
        );
    }
    // ... and a missing SRD use is an R0603 error: the kernel does not emit.
    for (arch, counter, soff) in [(Arch::Gfx1151, "Vm", "0"), (Arch::Gfx1201, "Load", "null")] {
        let body = format!(
            r#"mem(wg.isa(), "buffer_load_b32 v1, v0, s[4:7], {soff} offen", &[v(1)], &[v(0)], MemoryClass::VmemLoad)?;
    wg.isa().wait(Counter::{counter}, 0)?;"#
        );
        let (ds, emitted) = run_diags(&body, arch);
        assert!(!emitted, "{}: {ds:?}", arch.name());
        let r0603 = coded(&ds, "R0603");
        assert_eq!(r0603.len(), 1, "{}: {ds:?}", arch.name());
        assert_eq!(
            r0603[0].severity,
            Severity::Error,
            "{}: {:?}",
            arch.name(),
            r0603[0]
        );
        assert!(
            said(r0603[0]).contains("also reads s[4:7]: write &[v(1)], &[v(0), sr(4, 4)]"),
            "{}: {:?}",
            arch.name(),
            r0603[0]
        );
    }
}

/// An `Instruction` that is built but never consumed carries no arch and is never validated; the
/// driver validates it, against its own arch, when it consumes it.
#[test]
fn instruction_new_is_validated_where_the_driver_consumes_it() {
    // Unconsumed: no constructor arch, so no error even for a mnemonic that does not exist.
    let (errs, warns, emitted) = run(
        &format!(r#"let _unused = Instruction::new("not_a_mnemonic v0", vec![], vec![]); {GOOD}"#),
        Arch::Gfx1201,
    );
    assert!(
        errs.is_empty() && warns.is_empty() && emitted,
        "{errs:?} {warns:?}"
    );

    // scmp: a typo is refused by the driver's validation.
    let (errs, _, emitted) = run(
        r#"let _c = wg.scmp(Instruction::new("s_cmp_eq_u3 s1, 0", vec![], vec![s(1)]))?;"#,
        Arch::Gfx1151,
    );
    assert!(!emitted && errs.len() == 1, "{errs:?}");
    assert!(
        errs[0].contains("unknown mnemonic `s_cmp_eq_u3`")
            && errs[0].contains("did you mean `s_cmp_eq_u32`?"),
        "{}",
        errs[0]
    );

    // ds_store / ds_load: consumed through the LDS ownership model; same validation first.
    let store = |text: &str, uses: &str| {
        // v1 is the store's data register: define it so the only warning a case can draw is the
        // def/use list one under test (v0 is the thread id, defined at entry).
        format!(
            r#"op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("{text}", vec![], vec![{uses}]).memory(MemoryClass::DsStore))?;
    let d = wg.wait(st)?;
    let (t,) = wg.barrier((ready(t, d),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"#
        )
    };
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (errs, warns, emitted) = run(&store("ds_store_b32 v0, v1", "v(0), v(1)"), arch);
        assert!(
            errs.is_empty() && warns.is_empty() && emitted,
            "{}: {errs:?} {warns:?}",
            arch.name()
        );
        let (errs, _, emitted) = run(&store("ds_stor_b32 v0, v1", "v(0), v(1)"), arch);
        assert!(
            !emitted && errs.len() == 1 && errs[0].contains("unknown mnemonic `ds_stor_b32`"),
            "{}: {errs:?}",
            arch.name()
        );
        // A missing use: an R0603 error naming the register; nothing is emitted.
        let (ds, emitted) = run_diags(&store("ds_store_b32 v0, v1", "v(0)"), arch);
        assert!(!emitted && ds.len() == 1, "{}: {ds:?}", arch.name());
        assert_eq!(
            (ds[0].code, ds[0].severity),
            ("R0603", Severity::Error),
            "{}: {:?}",
            arch.name(),
            ds[0]
        );
        assert!(
            said(&ds[0]).contains("also reads v1: write &[], &[v(0), v(1)]"),
            "{}: {:?}",
            arch.name(),
            ds[0]
        );
    }
}

/// The definedness follow-up of `check`: a native `op` that reads `s2` is an R0604 error on gfx1201
/// (the workgroup X id arrives in `ttmp9` there), located at the `op(..)` call; nothing is emitted
/// and `compile_named` refuses the same source.
#[test]
fn reading_s2_on_gfx1201_is_an_r0604_error_at_the_op_call_and_names_ttmp9() {
    let src = kernel(READ_S2);
    let (line, col) = pos(&src, "op(wg.isa()");
    let c = peacemaker_rip::check(&src, "validate.rip", Arch::Gfx1201, "v");
    assert!(c.emitted.is_none(), "{:?}", c.diagnostics);
    assert_eq!(c.diagnostics.len(), 1, "{:?}", c.diagnostics);
    let d = &c.diagnostics[0];
    assert_eq!((d.code, d.severity), ("R0604", Severity::Error), "{d:?}");
    let span = d.span.expect("R0604 is located");
    assert_eq!((span.line, span.col), (line, col), "{d:?}");
    assert!(
        d.message.contains("register read before it is defined: s2"),
        "{}",
        d.message
    );
    assert!(d.message.contains("`s_mov_b32 s5, s2`"), "{}", d.message);
    assert!(
        d.message.contains("nothing in the kernel writes it")
            && d.message.contains("the launch does not initialize it"),
        "{}",
        d.message
    );
    assert!(d.message.contains("ttmp9"), "{}", d.message);
    let e = peacemaker_rip::compile_named(&src, "validate.rip", Arch::Gfx1201, "v")
        .err()
        .expect("compile_named refuses a read nothing defines");
    assert!(e.starts_with(&format!("validate.rip:{line}:")), "{e}");
    assert!(
        e.contains("register read before it is defined: s2") && e.contains("ttmp9"),
        "{e}"
    );

    // The register the entry state does provide on gfx1201 is not flagged: the workgroup X id in ttmp9.
    let (ds, emitted) = run_diags(
        r#"op(wg.isa(), "s_mov_b32 s5, ttmp9", &[s(5)], &[])?;"#,
        Arch::Gfx1201,
    );
    assert!(ds.is_empty() && emitted, "{ds:?}");
}

/// R0604 is path-free: it fires only for a register nothing in the kernel writes and the launch does
/// not initialize. A read that precedes a later write of the same register is not R0604 (ordering is
/// the path analysis's business, not this check's).
#[test]
fn a_read_before_a_later_write_of_the_register_is_not_r0604() {
    let body = r#"op(wg.isa(), "s_mov_b32 s5, s2", &[s(5)], &[s(2)])?;
    op(wg.isa(), "s_mov_b32 s2, 0", &[s(2)], &[])?;"#;
    let (ds, emitted) = run_diags(body, Arch::Gfx1201);
    assert!(coded(&ds, "R0604").is_empty(), "{ds:?}");
    assert!(emitted, "{ds:?}");
}

/// On gfx11 the skeleton's spec enables the workgroup X id, which lands in `s2`: reading it is defined.
#[test]
fn reading_the_enabled_workgroup_id_s2_on_gfx11_does_not_warn() {
    let src = kernel(READ_S2);
    let (ds, emitted) = run_diags(READ_S2, Arch::Gfx1151);
    assert!(coded(&ds, "R0604").is_empty(), "{ds:?}");
    assert!(
        ds.is_empty() && emitted,
        "a correct instruction on defined registers is clean: {ds:?}"
    );
    assert!(peacemaker_rip::compile_named(&src, "validate.rip", Arch::Gfx1151, "v").is_ok());
}

// ---------------------------------------------------------------- production sources through `check`

const GATHER: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_gather.rip");
const SCORE: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score.rip");
const SCORE_BF16: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score_bf16.rip");
const TOPK: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_topk.rip");

/// Every supported (source, arch, variant) of the four production QSA kernels: the gather module's
/// `convert` / `attend` on gfx1151 and gfx1201; `score`, `score-bf16` and `select` on gfx1151 only
/// (the other arches are rejected by those sources, see `qsa_select*.rs`).
const PRODUCTION: &[(&str, &str, &[Arch], &[&str])] = &[
    (
        "qsa_gather",
        GATHER,
        &[Arch::Gfx1151, Arch::Gfx1201],
        &["convert", "attend"],
    ),
    ("qsa_score", SCORE, &[Arch::Gfx1151], &["score"]),
    (
        "qsa_score_bf16",
        SCORE_BF16,
        &[Arch::Gfx1151],
        &["score-bf16"],
    ),
    ("qsa_topk", TOPK, &[Arch::Gfx1151], &["select"]),
];

/// No production kernel may fail validation: R0603 (def/use list drift) and R0604 (read before it is
/// defined) are errors, so a clean check also means none of either. Per kernel/arch/variant counts of
/// R0603, R0604 and every other diagnostic are printed separately (`cargo test -p peacemaker-rip
/// --test validate production -- --nocapture`), each with its text, so nothing is discarded.
#[test]
fn production_qsa_sources_check_without_errors_and_report_r0603_and_r0604_separately() {
    let mut compiles = 0usize;
    let mut totals = [0usize; 3];
    for &(kernel, source, arches, variants) in PRODUCTION {
        for &arch in arches {
            for &variant in variants {
                let case = format!("{kernel}/{}/{variant}", arch.name());
                let checked =
                    peacemaker_rip::check(source, &format!("{kernel}.rip"), arch, variant);
                let mut counts = [0usize; 3];
                for d in &checked.diagnostics {
                    let b = match d.code {
                        "R0603" => 0,
                        "R0604" => 1,
                        _ => 2,
                    };
                    counts[b] += 1;
                    println!(
                        "DIAGNOSTIC {case}: {:?} {}: {}",
                        d.severity,
                        d.code,
                        said(d)
                    );
                }
                println!(
                    "DIAGNOSTICS {case}: R0603 {} R0604 {} other {}",
                    counts[0], counts[1], counts[2]
                );
                let errs = errors(&checked.diagnostics);
                assert!(errs.is_empty(), "{case}: {errs:?}");
                assert!(
                    checked.emitted.is_some(),
                    "{case}: no program emitted although there are no errors"
                );
                assert_eq!(
                    (counts[0], counts[1]),
                    (0, 0),
                    "{case}: {:?}",
                    checked.diagnostics
                );
                for (t, c) in totals.iter_mut().zip(counts) {
                    *t += c;
                }
                compiles += 1;
            }
        }
    }
    println!(
        "DIAGNOSTICS total over {compiles} compiles: R0603 {} R0604 {} other {}",
        totals[0], totals[1], totals[2]
    );
}
