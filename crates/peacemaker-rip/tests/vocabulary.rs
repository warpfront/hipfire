//! Native vocabulary is exercised through source compilation, not native dispatch
//! internals. Untyped escape hatches must still respect the typed core's seal.
//!
//! Newly reachable families (bound since this file): `KernargLayout::pointer_access`
//! (`read_only`/`write_only`/`read_write`); `RegPlan::add_range` (runtime names,
//! kinds, widths and lifetimes, with the checked allocator's refusals intact);
//! `Builder::release_store_sources`, `push`, `control`, `ds_crosslane`, `lds_slot`,
//! `lds_relayout`, `ds_store`, `ds_load`, `loop_`, `barrier_signal`, `barrier_wait`
//! and `barrier_with_scope`; runtime `VopdOp { op, dst, src0, src1 }` construction
//! for `Builder::vopd` (including `v`/`s`/`lit`/`inline` `src0` forms and pair
//! validation); the runtime `Wmma::iu4`/`fp8` constructors; and the runtime
//! `Wmma::swmmac(arch, Swmmac::<kind>, d, a, b, index, SwmmacMods { .. })`
//! constructor (all eleven `Swmmac` kinds, with the accumulator read declared).
//!
//! Still unreachable: borrowed `Builder::program`
//! and `Builder::ledger` inspection; untyped `Builder::label` (labels live on typed
//! `Workgroup`/`Wave`, untyped loop heads are placed by `loop_`); the ledger-wide
//! `Builder::wait_all` (explicit `wait(Counter, count)` is bound instead); the
//! `Builder::barrier` shorthand (the explicit
//! `barrier_with_scope(..., lds_only)` binding covers it); the const-generic
//! `Builder::vopd_ff` / `vopd_ff_shared_src1` witnesses (VGPR parity and bank
//! witnesses have no native representation); `RegPlan::temp_v` (needs a Rust
//! `Scope` and a const-generic typed result); `RegPlan::verify_access` and
//! `RegPlan::verify_lifetimes` (emission and `finish` already perform these
//! checks); and integer `rotate`/`prime`/`retire_cur` (driver ring transitions
//! need private runtime slot mapping, so they are refused at construction —
//! use `Workgroup::barrier`).
//!
//! `control` deliberately bypasses the frontend `validate::instruction` check:
//! branch labels are symbolic (`.Ldone`) and cannot round-trip through the
//! machine-table parser, so the core `Builder::control` policy (a branch or
//! `s_endpgm`, one statement) is the check. `push`, `ds_store`, `ds_load` and
//! `ds_crosslane` keep the frontend check — missing def/use and malformed-operand
//! refusals below pin that — and only `control` bypasses it, by necessity.

use peacemaker_isa::{Arch, Emitted};

const PLAN: &str = r#"
    let mut p = RegPlan::new(32, 16)?;
    p.v::<8>("vectors", 0, Live::Whole)?;
    p.s::<8>("scalars", 0, Live::Whole)?;
"#;

fn source(kernargs: &str, plan: &str, body: &str, typed: bool) -> String {
    let (open, close) = if typed {
        (
            "let mut wg = Workgroup::new(&mut b, spec.arch)?;",
            "let end = wg.exit(\".Lend\")?; wg.end(end)?;",
        )
    } else {
        (
            "",
            "b.control(Instruction::new(\"s_endpgm\", vec![], vec![]))?;",
        )
    };
    format!(
        r#"fn emit(spec: Spec) -> Result<Emitted, String> {{
    let kspec = KernelSpec {{
        kernel_id: "vocabulary".into(), variant: "v".into(),
        arch: spec.arch, symbol: "vocabulary_kernel".into(),
        kernargs: {kernargs}, user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 64, wave32: true, cu_mode: false,
    }};
    {plan}
    let mut b = Builder::new(kspec, p);
    {open}
    {body}
    {close}
    b.finish()
}}
"#
    )
}

fn kernel(body: &str) -> String {
    source("KernargLayout::new(0)", PLAN, body, false)
}

fn compile(src: &str, arch: Arch) -> Emitted {
    peacemaker_rip::compile_named(src, "vocabulary.rip", arch, "v")
        .unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn refuses(src: &str, arch: Arch, expected: &str) {
    let error = peacemaker_rip::compile_named(src, "vocabulary.rip", arch, "v")
        .err()
        .expect("source must be refused");
    // Ignore only the located prefix; compare the complete core refusal, not
    // a substring that could accidentally match a different evaluator error.
    let mut fields = error.splitn(4, ':');
    assert_eq!(fields.next(), Some("vocabulary.rip"), "{error}");
    assert!(fields.next().unwrap().parse::<usize>().unwrap() > 0);
    assert!(fields.next().unwrap().parse::<usize>().unwrap() > 0);
    assert_eq!(
        fields.next(),
        Some(format!(" {expected}").as_str()),
        "{error}"
    );
}

#[test]
fn pointer_access_preserves_all_three_metadata_values() {
    let src = source(
        r#"KernargLayout::new(24)
            .pointer_access("input", 0, "read_only")
            .pointer_access("output", 8, "write_only")
            .pointer_access("inout", 16, "read_write")"#,
        PLAN,
        "",
        false,
    );
    let emitted = compile(&src, Arch::Gfx1201);
    for access in ["read_only", "write_only", "read_write"] {
        assert_eq!(
            emitted
                .s_text
                .matches(&format!(".actual_access: {access}\n"))
                .count(),
            1
        );
    }
    for name in ["input", "output", "inout"] {
        assert!(emitted.s_text.contains(&format!(".name: {name}\n")));
    }
    assert!(emitted.s_text.contains("\ts_endpgm\n"));
}

#[test]
fn pointer_access_keeps_kernarg_bounds_checks() {
    refuses(
        &source(
            r#"KernargLayout::new(4).pointer_access("input", 0, "read_only")"#,
            PLAN,
            "",
            false,
        ),
        Arch::Gfx1201,
        "overlapping/out-of-bounds kernarg input",
    );
}

#[test]
fn add_range_accepts_runtime_names_kinds_widths_and_lifetimes() {
    let plan = r#"
        let mut p = RegPlan::new(32, 16)?;
        let width = 4;
        p.add_range(format!("v{}", width), "v", 0, width, Live::Whole)?;
        p.add_range("scalars", "s", 0, 4, Live::Whole)?;
        p.add_range("early", "v", 4, 1, Live::Between(".Learly".into(), ".Lmiddle".into()))?;
        p.add_range("late", "v", 4, 1, Live::Between(".Lmiddle".into(), ".Llate".into()))?;
    "#;
    let body = r#"
        wg.label(".Learly")?;
        wg.isa().push(Instruction::new("v_mov_b32_e32 v4, 1", vec![v(4)], vec![]))?;
        wg.label(".Lmiddle")?;
        wg.isa().push(Instruction::new("v_mov_b32_e32 v4, 2", vec![v(4)], vec![]))?;
        wg.isa().push(Instruction::new("s_mov_b32 s0, 3", vec![s(0)], vec![]))?;
        wg.label(".Llate")?;
    "#;
    let emitted = compile(
        &source("KernargLayout::new(0)", plan, body, true),
        Arch::Gfx1201,
    );
    assert!(emitted.s_text.contains("v_mov_b32_e32 v4, 1"));
    assert!(emitted.s_text.contains("v_mov_b32_e32 v4, 2"));
    assert!(emitted.s_text.contains("s_mov_b32 s0, 3"));
    assert_eq!(emitted.proof.reg_plan.len(), 4);
    assert_eq!(emitted.shape.next_free_vgpr, 5);
    assert_eq!(emitted.shape.next_free_sgpr, 4);
}

#[test]
fn add_range_preserves_checked_allocator_refusals() {
    for (declarations, expected) in [
        (
            r#"p.add_range("bad", "v", 0, 3, Live::Whole)?;"#,
            "register width must be 1, 2, 4 or 8",
        ),
        (
            r#"p.add_range("bad", "s", 1, 2, Live::Whole)?;"#,
            "misaligned SGPR 1:2",
        ),
        (
            r#"p.add_range("bad", "v", 31, 2, Live::Whole)?;"#,
            "bad exceeds register budget",
        ),
        (
            r#"p.add_range("first", "v", 0, 2, Live::Whole)?; p.add_range("second", "v", 1, 1, Live::Whole)?;"#,
            "second overlaps first with a shared lifetime",
        ),
    ] {
        let plan = format!("let mut p = RegPlan::new(32, 16)?; {declarations}");
        refuses(
            &source("KernargLayout::new(0)", &plan, "", false),
            Arch::Gfx1201,
            expected,
        );
    }
}

#[test]
fn release_store_sources_emits_the_gfx12_guard_without_draining_stores_early() {
    let src = kernel(
        r#"
        b.push(Instruction::new("v_mov_b32_e32 v1, 1", vec![v(1)], vec![]))?;
        b.push(Instruction::new("s_mov_b32 s2, 0", vec![s(2)], vec![]))?;
        b.push(Instruction::new("s_mov_b32 s3, 0", vec![s(3)], vec![]))?;
        b.push(Instruction::new("buffer_store_b32 v1, v0, s[0:3], s0 offen", vec![], vec![v(1), v(0), sr(0, 4), s(0)]).memory(MemoryClass::VmemStore))?;
        b.release_store_sources()?;
        b.push(Instruction::new("v_mov_b32_e32 v1, 7", vec![v(1)], vec![]))?;
        b.wait(Counter::Store, 0)?;
    "#,
    );
    let emitted = compile(&src, Arch::Gfx1201);
    let text = &emitted.s_text;
    let release = text.find("s_wait_alu depctr_vm_vsrc(0)").unwrap();
    let overwrite = text.find("v_mov_b32_e32 v1, 7").unwrap();
    let drain = text.find("s_wait_storecnt 0x0").unwrap();
    assert!(release < overwrite && overwrite < drain, "{text}");
    refuses(
        &kernel("b.release_store_sources()?;"),
        Arch::Gfx1151,
        "depctr_vm_vsrc requires gfx12",
    );
}

#[test]
fn crosslane_has_no_slot_but_keeps_its_load_dependency() {
    for (arch, wait) in [
        (Arch::Gfx1151, "s_waitcnt lgkmcnt(0)"),
        (Arch::Gfx1201, "s_wait_dscnt 0x0"),
    ] {
        for instruction in [
            r#"Instruction::new("ds_swizzle_b32 v2, v1 offset:0x1f", vec![v(2)], vec![v(1)]).memory(MemoryClass::DsLoad)"#,
            r#"Instruction::new("ds_bpermute_b32 v2, v0, v1", vec![v(2)], vec![v(0), v(1)]).memory(MemoryClass::DsLoad)"#,
        ] {
            let body = format!(
                r#"
                wg.isa().push(Instruction::new("v_mov_b32_e32 v1, 1", vec![v(1)], vec![]))?;
                wg.isa().ds_crosslane({instruction})?;
                wg.isa().push(Instruction::new("v_mov_b32_e32 v3, v2", vec![v(3)], vec![v(2)]))?;
            "#
            );
            let emitted = compile(&source("KernargLayout::new(0)", PLAN, &body, true), arch);
            let text = &emitted.s_text;
            assert!(
                text.find(wait).unwrap() < text.find("v_mov_b32_e32 v3, v2").unwrap(),
                "{text}"
            );
            assert!(emitted.proof.lds_slots.is_empty());
        }
    }
    for instruction in [
        r#"Instruction::new("ds_swizzle_b32 v2, v1 offset:0x1f", vec![v(2)], vec![v(1)])"#,
        r#"Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad)"#,
    ] {
        refuses(
            &kernel(&format!("b.ds_crosslane({instruction})?;")),
            Arch::Gfx1201,
            "ds_crosslane requires a ds_swizzle_b32 or ds_bpermute_b32 DS-load-class instruction",
        );
    }
}

#[test]
fn push_and_control_do_not_bypass_instruction_policy() {
    let emitted = compile(
        &kernel(
            r#"
        b.push(Instruction::new("s_cmp_eq_u32 s0, 0", vec![], vec![s(0)]))?;
        b.push(Instruction::new("v_mov_b32_e32 v1, 9", vec![v(1)], vec![]))?;
        b.control(Instruction::new("s_cbranch_scc1 .Ldone", vec![], vec![]))?;
        b.control(Instruction::new("s_branch .Ldone", vec![], vec![]))?;
        b.loop_(".Ldone", |draft| { Ok(()) })?;
    "#,
        ),
        Arch::Gfx1201,
    );
    for instruction in [
        "v_mov_b32_e32 v1, 9",
        "s_cbranch_scc1 .Ldone",
        "s_branch .Ldone",
        ".Ldone:",
        "s_endpgm",
    ] {
        assert!(emitted.s_text.contains(instruction), "{}", emitted.s_text);
    }
    for (body, expected) in [
        (
            r#"b.push(Instruction::new("s_endpgm", vec![], vec![]))?;"#,
            "s_endpgm changes control flow: branches and the program end go through the typed core",
        ),
        (
            r#"b.push(Instruction::new("s_barrier", vec![], vec![]))?;"#,
            "s_barrier is a barrier: barriers go through the typed core",
        ),
        (
            r#"b.push(Instruction::new("global_inv scope:SCOPE_SE", vec![], vec![]))?;"#,
            "clauses and global visibility require their builder methods",
        ),
        (
            r#"b.control(Instruction::new("s_nop 0", vec![], vec![]))?;"#,
            "\"s_nop 0\" is not a branch or s_endpgm",
        ),
        (
            r#"b.push(Instruction::new("ds_load_b32 v1, v0", vec![v(1)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"#,
            "DS access must carry an LDS slot through ds_load/ds_store",
        ),
    ] {
        refuses(&kernel(body), Arch::Gfx1201, expected);
    }
}

#[test]
fn control_rejects_nonbranches_sealed_use_and_embedded_newlines() {
    // A VALU instruction is not a branch either: the exact policy text names
    // the offending statement.
    refuses(
        &kernel(r#"b.control(Instruction::new("v_mov_b32_e32 v1, 9", vec![v(1)], vec![]))?;"#),
        Arch::Gfx1201,
        "\"v_mov_b32_e32 v1, 9\" is not a branch or s_endpgm",
    );
    // A branch-looking text with an embedded newline is no longer one
    // assembler statement, so it is refused with the same policy error (the
    // `Debug` rendering escapes the newline).
    refuses(
        &kernel(r#"b.control(Instruction::new("s_branch .Ldone\ns_nop 0", vec![], vec![]))?;"#),
        Arch::Gfx1201,
        "\"s_branch .Ldone\\ns_nop 0\" is not a branch or s_endpgm",
    );
    // Under a Workgroup seal even a well-formed branch goes through the driver.
    refuses(
        &source(
            "KernargLayout::new(0)",
            PLAN,
            r#"wg.isa().control(Instruction::new("s_branch .Ldone", vec![], vec![]))?;"#,
            true,
        ),
        Arch::Gfx1201,
        "typed kernel: LDS, barriers, branches and loops go through peacemaker-author",
    );
}

#[test]
fn raw_push_and_ds_paths_check_defs_uses_and_operand_shape() {
    // The frontend `validate::instruction` check runs on the raw `push` /
    // `ds_store` / `ds_load` / `ds_crosslane` paths before the builder's own
    // policy checks, so a def/use list that omits a register the text names
    // is refused with the corrected Rust lists.
    for (body, expected) in [
        (
            r#"b.push(Instruction::new("v_add_f32_e32 v2, v0, v1", vec![v(2)], vec![v(0)]))?;"#.to_string(),
            "op declares defs [v2] and uses [v0] but `v_add_f32_e32 v2, v0, v1` also reads v1: write &[v(2)], &[v(0), v(1)]".to_string(),
        ),
        (
            format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_store(slot, Instruction::new(\"ds_store_b32 v0, v1\", vec![], vec![v(1)]))?;"),
            "op declares defs [] and uses [v1] but `ds_store_b32 v0, v1` also reads v0: write &[], &[v(1), v(0)]".to_string(),
        ),
        (
            format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_load(slot, Instruction::new(\"ds_load_b32 v2, v0\", vec![], vec![v(0)]))?;"),
            "op declares defs [] and uses [v0] but `ds_load_b32 v2, v0` also writes v2: write &[v(2)], &[v(0)]".to_string(),
        ),
        (
            r#"b.ds_crosslane(Instruction::new("ds_swizzle_b32 v2, v1 offset:0x1f", vec![v(2)], vec![]).memory(MemoryClass::DsLoad))?;"#.to_string(),
            "op declares defs [v2] and uses [] but `ds_swizzle_b32 v2, v1 offset:0x1f` also reads v1: write &[v(2)], &[v(1)]".to_string(),
        ),
    ] {
        refuses(&kernel(&body), Arch::Gfx1201, &expected);
    }
    // Malformed operands keep the table parser's reason. The reason comes from
    // the machine-table decoder, so these assert the located shape with
    // `contains` rather than an exact string.
    for (path, call) in [
        (
            "push",
            r#"b.push(Instruction::new("v_mov_b32_e32 v[2:, 0", vec![v(0)], vec![]))?;"#,
        ),
        (
            "ds_store",
            r#"b.ds_store(0, Instruction::new("ds_store_b32 v0, v1 offset0:999", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;"#,
        ),
    ] {
        let error =
            peacemaker_rip::compile_named(&kernel(call), "vocabulary.rip", Arch::Gfx1201, "v")
                .err()
                .unwrap_or_else(|| panic!("{path} must refuse malformed operands"));
        assert!(error.contains("malformed operands for `"), "{error}");
    }
}

const STORE: &str = r#"Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore)"#;
const LOAD: &str =
    r#"Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad)"#;

#[test]
fn untyped_lds_publish_read_retire_and_relayout_carry_slot_ids() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let body = format!(
            r#"
            let slot = b.lds_slot("tile", 0, 16)?;
            b.push(Instruction::new("v_mov_b32_e32 v1, 1", vec![v(1)], vec![]))?;
            b.ds_store(slot, {STORE})?;
            b.barrier_with_scope(&[ready(slot)], "lds_only")?;
            b.ds_load(slot, {LOAD})?;
            b.push(Instruction::new("v_mov_b32_e32 v3, v2", vec![v(3)], vec![v(2)]))?;
            b.barrier_with_scope(&[retire(slot)], "lds_only")?;
            b.lds_relayout()?;
            let scratch = b.lds_slot("scratch", 0, 32)?;
        "#
        );
        let emitted = compile(&kernel(&body), arch);
        assert!(emitted.s_text.contains("ds_store_b32 v0, v1"));
        assert!(emitted.s_text.contains("ds_load_b32 v2, v0"));
        assert_eq!(emitted.proof.lds_slots.len(), 2);
        assert_eq!(emitted.proof.lds_slots[0].name, "tile");
        assert_eq!(emitted.proof.lds_slots[1].name, "scratch");
        assert_eq!(emitted.proof.barriers.len(), 2);
        assert_eq!(emitted.proof.barriers[0].transitions, ["Ready(0)"]);
        assert_eq!(emitted.proof.barriers[1].transitions, ["Retire(0)"]);
    }
}

#[test]
fn untyped_lds_checks_layout_instruction_classes_and_slot_states() {
    for (body, expected) in [
        ("b.lds_slot(\"empty\", 0, 0)?;".to_string(), "invalid LDS slot"),
        ("b.lds_slot(\"first\", 0, 16)?; b.lds_slot(\"second\", 8, 16)?;".to_string(), "overlapping LDS slots"),
        (format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_load(slot, {LOAD})?;"), "tile is not published"),
        (format!("b.ds_store(0, {STORE})?;"), "unknown LDS slot"),
        (format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_store(slot, {LOAD})?;"), "ds_store requires an LDS store instruction"),
        (format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_load(slot, {STORE})?;"), "ds_load requires an LDS load instruction"),
        (format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_store(slot, {STORE})?; b.lds_relayout()?;"), "tile is Publishing at relayout"),
        ("let slot = b.lds_slot(\"tile\", 0, 16)?; b.barrier_with_scope(&[ready(slot)], \"lds_only\")?;".to_string(), "invalid LDS transition Ready(0)"),
    ] {
        refuses(&kernel(&body), Arch::Gfx1201, expected);
    }
}

#[test]
fn split_barriers_emit_both_halves_and_hold_transitioning_slots() {
    let body = format!(
        r#"
        let slot = b.lds_slot("tile", 0, 16)?;
        b.push(Instruction::new("v_mov_b32_e32 v1, 1", vec![v(1)], vec![]))?;
        b.ds_store(slot, {STORE})?;
        b.barrier_signal(&[ready(slot)])?;
        b.push(Instruction::new("s_nop 0", vec![], vec![]))?;
        b.barrier_wait()?;
        b.ds_load(slot, {LOAD})?;
        b.barrier_signal(&[retire(slot)])?;
        b.barrier_wait()?;
    "#
    );
    let emitted = compile(&kernel(&body), Arch::Gfx1201);
    let text = &emitted.s_text;
    assert_eq!(text.matches("s_barrier_signal -1").count(), 2);
    assert_eq!(text.matches("s_barrier_wait 0xffff").count(), 2);
    assert!(text.find("s_wait_dscnt 0x0").unwrap() < text.find("s_barrier_signal -1").unwrap());
    assert!(text.find("s_barrier_signal -1").unwrap() < text.find("s_nop 0").unwrap());
    assert!(text.find("s_nop 0").unwrap() < text.find("s_barrier_wait 0xffff").unwrap());
    refuses(
        &kernel("b.barrier_signal(&[])?;"),
        Arch::Gfx1151,
        "split barriers require gfx12",
    );
    refuses(
        &kernel("b.barrier_wait()?;"),
        Arch::Gfx1201,
        "barrier wait without signal",
    );
    refuses(
        &kernel("b.barrier_signal(&[])?;"),
        Arch::Gfx1201,
        "unmatched barrier signal",
    );
    refuses(
        &kernel("b.barrier_signal(&[])?; b.barrier_signal(&[])?;"),
        Arch::Gfx1201,
        "split barrier already in flight",
    );
    refuses(
        &kernel("b.barrier_signal(&[])?; b.lds_relayout()?;"),
        Arch::Gfx1201,
        "relayout with a split barrier in flight",
    );
    refuses(&kernel(&format!("let slot = b.lds_slot(\"tile\", 0, 16)?; b.ds_store(slot, {STORE})?; b.barrier_signal(&[ready(slot)])?; b.ds_load(slot, {LOAD})?;")), Arch::Gfx1201, "LDS slot in split-barrier transition");
}

#[test]
fn barrier_scope_controls_global_visibility_on_each_architecture() {
    for (arch, barrier, invalidate) in [
        (Arch::Gfx1151, "s_barrier", "buffer_gl0_inv"),
        (
            Arch::Gfx1201,
            "s_barrier_wait 0xffff",
            "global_inv scope:SCOPE_SE",
        ),
    ] {
        let local = compile(&kernel(r#"b.barrier_with_scope(&[], "lds_only")?;"#), arch);
        assert!(local.s_text.contains(barrier));
        assert!(!local.s_text.contains(invalidate));
        let global = compile(&kernel(r#"b.barrier_with_scope(&[], "workgroup")?;"#), arch);
        assert!(global.s_text.find(barrier).unwrap() < global.s_text.find(invalidate).unwrap());
    }
}

#[test]
fn integer_ready_and_retire_take_exactly_one_slot() {
    // The integer overload names a backend slot, so a second token is an
    // arity error rather than a silently ignored Drained token.
    for (call, expected) in [
        ("ready(0, 1)", "`ready` takes 1 arguments, got 2"),
        ("retire(0, 1)", "`retire` takes 1 arguments, got 2"),
    ] {
        refuses(
            &kernel(&format!("let t = {call};")),
            Arch::Gfx1201,
            expected,
        );
    }
}

#[test]
fn ring_transitions_refuse_integer_slots_at_construction() {
    // `rotate`/`prime`/`retire_cur` lower driver rings, whose slot mapping is
    // private to the runtime: an integer names a backend slot, not a ring, so
    // the constructor refuses before any barrier sees the value.
    for name in ["rotate", "prime", "retire_cur"] {
        refuses(
            &kernel(&format!("let t = {name}(0);")),
            Arch::Gfx1201,
            &format!("{name} is a driver-only ring transition; raw Builder barriers accept only ready(slot)/retire(slot) with integer LDS slot ids"),
        );
    }
}

#[test]
fn loop_closures_reach_a_fixpoint_and_preserve_ledger_balance() {
    let emitted = compile(
        &kernel(
            r#"
        b.push(Instruction::new("s_mov_b32 s4, 0", vec![s(4)], vec![]))?;
        b.loop_(".Lloop", |draft| {
            draft.push(Instruction::new("s_add_co_u32 s4, s4, 1", vec![s(4)], vec![s(4)]))?;
            draft.control(Instruction::new("s_cbranch_scc0 .Lloop", vec![], vec![]))?;
            Ok(())
        })?;
    "#,
        ),
        Arch::Gfx1201,
    );
    assert_eq!(emitted.s_text.matches(".Lloop:").count(), 1);
    assert!(emitted.s_text.contains("s_cbranch_scc0 .Lloop"));
    assert_eq!(emitted.s_text.matches("s_add_co_u32 s4, s4, 1").count(), 1);
    assert_eq!(emitted.proof.loop_fixpoints.len(), 1);
    assert!(emitted.proof.loop_fixpoints[0].iterations >= 2);
    refuses(
        &kernel(
            r#"
        b.loop_(".Lloop", |draft| {
            draft.push(Instruction::new("s_load_b32 s4, s[0:1], 0", vec![s(4)], vec![sr(0, 2)]).memory(MemoryClass::SmemLoad))?;
            Ok(())
        })?;
    "#,
        ),
        Arch::Gfx1201,
        "loop .Lloop: back-edge ledger must equal the entry ledger",
    );
    refuses(
        &kernel(r#"b.loop_(".Lloop", |draft| { Err("loop body refused".into()) })?;"#),
        Arch::Gfx1201,
        "loop body refused",
    );
}

#[test]
fn untyped_builder_bindings_cannot_escape_a_workgroup_seal() {
    for body in [
        r#"wg.isa().control(Instruction::new("s_endpgm", vec![], vec![]))?;"#,
        r#"wg.isa().lds_slot("tile", 0, 16)?;"#,
        r#"wg.isa().lds_relayout()?;"#,
        r#"wg.isa().ds_store(0, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;"#,
        r#"wg.isa().ds_load(0, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"#,
        r#"wg.isa().loop_(".Lloop", |draft| { Ok(()) })?;"#,
        r#"wg.isa().barrier_signal(&[])?;"#,
        r#"wg.isa().barrier_wait()?;"#,
        r#"wg.isa().barrier_with_scope(&[], "lds_only")?;"#,
    ] {
        refuses(
            &source("KernargLayout::new(0)", PLAN, body, true),
            Arch::Gfx1201,
            "typed kernel: LDS, barriers, branches and loops go through peacemaker-author",
        );
    }
}

#[test]
fn string_mappings_and_wmma_shapes_reject_unknown_values() {
    refuses(
        &source(
            r#"KernargLayout::new(8).pointer_access("input", 0, "ReadOnly")"#,
            PLAN,
            "",
            false,
        ),
        Arch::Gfx1201,
        "pointer_access: unknown access `ReadOnly` (expected read_only, write_only, read_write)",
    );
    refuses(
        &source(
            "KernargLayout::new(0)",
            r#"let mut p = RegPlan::new(32, 16)?; p.add_range("bad", "V", 0, 1, Live::Whole)?;"#,
            "",
            false,
        ),
        Arch::Gfx1201,
        "add_range: unknown register kind `V` (expected v, s)",
    );
    refuses(
        &kernel(r#"b.barrier_with_scope(&[], "device")?;"#),
        Arch::Gfx1201,
        "barrier_with_scope: unknown scope `device` (expected lds_only, workgroup)",
    );
    for (instruction, expected) in [
        (
            r#"Wmma::iu4(spec.arch, vr(0, 4), vr(2, 2), vr(4, 2), None)"#,
            "WMMA operand requires 8 VGPRs, found v[0:3]",
        ),
        (
            r#"Wmma::iu4(spec.arch, vr(0, 8), v(2), vr(4, 2), None)"#,
            "WMMA operand requires 2 VGPRs, found v2",
        ),
        (
            r#"Wmma::iu4(spec.arch, vr(0, 8), vr(2, 2), vr(4, 2), Some(vr(0, 4)))"#,
            "WMMA operand requires 8 VGPRs, found v[0:3]",
        ),
        (
            r#"Wmma::fp8(spec.arch, vr(0, 8), vr(2, 2), v(4), None)?"#,
            "WMMA operand requires 2 VGPRs, found v4",
        ),
    ] {
        refuses(
            &kernel(&format!("b.push({instruction})?;")),
            Arch::Gfx1201,
            expected,
        );
    }
}

#[test]
fn raw_barriers_reject_driver_region_and_ring_transitions() {
    let refusal = "raw Builder barriers require ready(slot)/retire(slot) with integer LDS slot ids; driver region/ring transitions must use Workgroup barriers";
    for barrier in [
        "barrier_signal(&[transition])",
        "barrier_with_scope(&[transition], \"lds_only\")",
    ] {
        let body = format!(
            r#"
            let region = wg.lds::<Tile>("tile", 0, 16)?;
            let (region, pending) = wg.ds_store(region, {STORE})?;
            let drained = wg.wait(pending)?;
            let transition = ready(region, drained);
            wg.isa().{barrier}?;
        "#
        );
        let src = format!(
            "region Tile;\n{}",
            source("KernargLayout::new(0)", PLAN, &body, true)
        );
        refuses(&src, Arch::Gfx1201, refusal);
    }
    refuses(
        &kernel(r#"b.barrier_with_scope(&[ready(99)], "lds_only")?;"#),
        Arch::Gfx1201,
        "invalid LDS transition Ready(99)",
    );
}

#[test]
fn wmma_runtime_constructors_emit_checked_instructions() {
    let plan = r#"
        let mut p = RegPlan::new(32, 16)?;
        for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
        p.s::<8>("scalars", 0, Live::Whole)?;
    "#;
    // Inputs and accumulators live in v8..v31: define them before use.
    let init: String = (8..32u8)
        .map(|i| format!("wg.isa().push(Instruction::new(\"v_mov_b32_e32 v{i}, {i}\", vec![v({i})], vec![]))?;"))
        .collect::<Vec<_>>()
        .join("\n");
    for (arch, mnemonic) in [
        (Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4"),
        (Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4"),
    ] {
        for accumulator in ["None", "Some(vr(16, 8))"] {
            let body = format!("{init}\n                wg.isa().push(Wmma::iu4(spec.arch, vr(0, 8), vr(8, 2), vr(10, 2), {accumulator}))?;");
            let emitted = compile(&source("KernargLayout::new(0)", plan, &body, true), arch);
            let acc = if accumulator == "None" {
                "0"
            } else {
                "v[16:23]"
            };
            assert!(emitted.s_text.contains(&format!(
                "{mnemonic} v[0:7], v[8:9], v[10:11], {acc} neg_lo:[1,1,0]"
            )));
        }
    }
    for accumulator in ["None", "Some(vr(16, 8))"] {
        let body = format!("{init}\n            wg.isa().push(Wmma::fp8(spec.arch, vr(0, 8), vr(8, 2), vr(10, 2), {accumulator})?)?;");
        let emitted = compile(
            &source("KernargLayout::new(0)", plan, &body, true),
            Arch::Gfx1201,
        );
        assert!(emitted
            .s_text
            .contains("v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11],"));
        refuses(
            &source("KernargLayout::new(0)", plan, &body, true),
            Arch::Gfx1151,
            "WMMA opcode requires gfx1201",
        );
    }
}

/// SWMMAC kernel: D at v0, A at v8, B at v16, index at v24; v0..=v24 are
/// defined first (SWMMAC reads D in place as the accumulator).
fn swmmac_source(call: &str) -> String {
    let plan = r#"
        let mut p = RegPlan::new(32, 16)?;
        for i in 0..4u8 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
        p.s::<8>("scalars", 0, Live::Whole)?;
    "#;
    let init: String = (0..25u8)
        .map(|i| format!("wg.isa().push(Instruction::new(\"v_mov_b32_e32 v{i}, {i}\", vec![v({i})], vec![]))?;"))
        .collect::<Vec<_>>()
        .join("\n");
    source(
        "KernargLayout::new(0)",
        plan,
        &format!("{init}\n                {call}"),
        true,
    )
}

const SWMMAC_NO_MODS: &str = "SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0, clamp: false }";

#[test]
fn swmmac_every_kind_emits_its_checked_instruction() {
    for (kind, mnemonic, d, a, b) in [
        ("F32F16", "v_swmmac_f32_16x16x32_f16", 8, 4, 8),
        ("F32Bf16", "v_swmmac_f32_16x16x32_bf16", 8, 4, 8),
        ("F16F16", "v_swmmac_f16_16x16x32_f16", 4, 4, 8),
        ("Bf16Bf16", "v_swmmac_bf16_16x16x32_bf16", 4, 4, 8),
        ("I32Iu8", "v_swmmac_i32_16x16x32_iu8", 8, 2, 4),
        ("I32Iu4", "v_swmmac_i32_16x16x32_iu4", 8, 1, 2),
        ("I32Iu4X64", "v_swmmac_i32_16x16x64_iu4", 8, 2, 4),
        ("F32Fp8Fp8", "v_swmmac_f32_16x16x32_fp8_fp8", 8, 2, 4),
        ("F32Fp8Bf8", "v_swmmac_f32_16x16x32_fp8_bf8", 8, 2, 4),
        ("F32Bf8Fp8", "v_swmmac_f32_16x16x32_bf8_fp8", 8, 2, 4),
        ("F32Bf8Bf8", "v_swmmac_f32_16x16x32_bf8_bf8", 8, 2, 4),
    ] {
        let range = |base: u8, n: u8| {
            if n == 1 {
                format!("v{base}")
            } else {
                format!("v[{base}:{}]", base + n - 1)
            }
        };
        let src = swmmac_source(&format!(
            "wg.isa().push(Wmma::swmmac(spec.arch, Swmmac::{kind}, vr(0, {d}), vr(8, {a}), vr(16, {b}), v(24), {SWMMAC_NO_MODS})?)?;"
        ));
        let emitted = compile(&src, Arch::Gfx1201);
        let expected = format!(
            "{mnemonic} {}, {}, {}, v24\n",
            range(0, d),
            range(8, a),
            range(16, b)
        );
        assert!(emitted.s_text.contains(&expected), "{kind}: {expected}");
        refuses(&src, Arch::Gfx1151, "SWMMAC requires gfx12");
        refuses(&src, Arch::Gfx1100, "SWMMAC requires gfx12");
    }
}

#[test]
fn swmmac_modifiers_render_in_llvm_mc_order() {
    let src = swmmac_source(
        "wg.isa().push(Wmma::swmmac(spec.arch, Swmmac::I32Iu8, vr(0, 8), vr(8, 2), vr(16, 4), v(24), SwmmacMods { index_key: 1, neg_lo: 3, neg_hi: 0, clamp: true })?)?;",
    );
    assert!(compile(&src, Arch::Gfx1201).s_text.contains(
        "v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[16:19], v24 index_key:1 neg_lo:[1,1,0] clamp\n"
    ));
    let src = swmmac_source(
        "wg.isa().push(Wmma::swmmac(spec.arch, Swmmac::F32F16, vr(0, 8), vr(8, 4), vr(16, 8), v(24), SwmmacMods { index_key: 1, neg_lo: 3, neg_hi: 3, clamp: false })?)?;",
    );
    assert!(compile(&src, Arch::Gfx1201).s_text.contains(
        "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[16:23], v24 index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]\n"
    ));
}

#[test]
fn swmmac_refusals_carry_the_rust_error_text() {
    let call = |kind: &str, d: &str, a: &str, b: &str, idx: &str, mods: &str| {
        swmmac_source(&format!(
            "wg.isa().push(Wmma::swmmac(spec.arch, Swmmac::{kind}, {d}, {a}, {b}, {idx}, {mods})?)?;"
        ))
    };
    refuses(
        &call(
            "F32F16",
            "vr(0, 8)",
            "vr(8, 3)",
            "vr(16, 8)",
            "v(24)",
            SWMMAC_NO_MODS,
        ),
        Arch::Gfx1201,
        "v_swmmac_f32_16x16x32_f16: A requires 4 VGPRs, found v[8:10]",
    );
    refuses(
        &call(
            "F16F16",
            "vr(0, 8)",
            "vr(8, 4)",
            "vr(16, 8)",
            "v(24)",
            SWMMAC_NO_MODS,
        ),
        Arch::Gfx1201,
        "v_swmmac_f16_16x16x32_f16: D requires 4 VGPRs, found v[0:7]",
    );
    refuses(
        &call(
            "F32F16",
            "vr(0, 8)",
            "vr(8, 4)",
            "vr(16, 8)",
            "vr(24, 2)",
            SWMMAC_NO_MODS,
        ),
        Arch::Gfx1201,
        "WMMA operand requires 1 VGPRs, found v[24:25]",
    );
    refuses(
        &call(
            "F32Fp8Fp8",
            "vr(0, 8)",
            "vr(8, 2)",
            "vr(16, 4)",
            "v(24)",
            "SwmmacMods { index_key: 0, neg_lo: 1, neg_hi: 0, clamp: false }",
        ),
        Arch::Gfx1201,
        "v_swmmac_f32_16x16x32_fp8_fp8 does not accept neg_lo",
    );
    refuses(
        &call(
            "I32Iu4X64",
            "vr(0, 8)",
            "vr(8, 2)",
            "vr(16, 4)",
            "v(24)",
            "SwmmacMods { index_key: 1, neg_lo: 0, neg_hi: 0, clamp: false }",
        ),
        Arch::Gfx1201,
        "v_swmmac_i32_16x16x64_iu4 has no index_key",
    );
    refuses(
        &call(
            "F32F16",
            "vr(0, 8)",
            "vr(8, 4)",
            "vr(16, 8)",
            "v(24)",
            "SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0, clamp: true }",
        ),
        Arch::Gfx1201,
        "v_swmmac_f32_16x16x32_f16 does not accept clamp",
    );
    refuses(
        &call(
            "F32F16",
            "vr(0, 8)",
            "vr(8, 4)",
            "vr(16, 8)",
            "v(24)",
            "SwmmacMods { index_key: 0, neg_lo: 4, neg_hi: 0, clamp: false }",
        ),
        Arch::Gfx1201,
        "v_swmmac_f32_16x16x32_f16: neg_lo/neg_hi bit 2 (C) is not encodable",
    );
    refuses(
        &call(
            "F32F16",
            "vr(0, 8)",
            "vr(8, 4)",
            "vr(16, 8)",
            "v(24)",
            "SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0 }",
        ),
        Arch::Gfx1201,
        "SwmmacMods is missing `clamp`",
    );
}

/// The original SWMMAC defect, caught by the front end's effect check: text
/// that reads the destination accumulator in place but declares only A, B and
/// the index is refused (the corrected `Wmma::swmmac` declares D in `uses`).
#[test]
fn swmmac_text_without_the_accumulator_read_is_refused() {
    let text = "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[16:23], v24";
    refuses(
        &swmmac_source(&format!(
            "wg.isa().push(Instruction::new(\"{text}\", vec![vr(0, 8)], vec![vr(8, 4), vr(16, 8), v(24)]))?;"
        )),
        Arch::Gfx1201,
        &format!(
            "op declares defs [v[0:7]] and uses [v[8:11], v[16:23], v24] but `{text}` also reads v[0:7]: write &[vr(0, 8)], &[vr(8, 4), vr(16, 8), v(24), vr(0, 8)]"
        ),
    );
    // Declaring D as an accumulator read makes the same text acceptable.
    let emitted = compile(
        &swmmac_source(&format!(
            "wg.isa().push(Instruction::new(\"{text}\", vec![vr(0, 8)], vec![vr(8, 4), vr(16, 8), v(24), vr(0, 8)]))?;"
        )),
        Arch::Gfx1201,
    );
    assert!(emitted.s_text.contains(text));
}

#[test]
fn vopd_runtime_packets_keep_pair_validation() {
    for (op, mnemonic) in [
        ("add", "v_dual_add_f32"),
        ("sub", "v_dual_sub_f32"),
        ("subrev", "v_dual_subrev_f32"),
        ("mul", "v_dual_mul_f32"),
        ("fmac", "v_dual_fmac_f32"),
    ] {
        // v1..v5 are read by the packet: define them before use.
        let vinit: String = (1..6u8)
            .map(|i| format!("wg.isa().push(Instruction::new(\"v_mov_b32_e32 v{i}, {i}\", vec![v({i})], vec![]))?;"))
            .collect::<Vec<_>>()
            .join("\n");
        for operand in ["v(2)", "s(0)", "(\"lit\", 0x3f800000)", "(\"inline\", -1)"] {
            let body = format!(
                r#"
                {vinit}
                wg.isa().vopd(
                    VopdOp {{ op: "{op}", dst: 0, src0: {operand}, src1: 4 }},
                    VopdOp {{ op: "mul", dst: 1, src0: v(3), src1: 5 }}
                )?;
            "#
            );
            for arch in [Arch::Gfx1151, Arch::Gfx1201] {
                let emitted = compile(&source("KernargLayout::new(0)", PLAN, &body, true), arch);
                assert!(emitted.s_text.contains(mnemonic));
                assert!(emitted.s_text.contains(" :: v_dual_mul_f32 v1, v3, v5"));
                assert_eq!(emitted.proof.vopd_pairs, 1);
            }
        }
    }
    for (x, y, arch, expected) in [
        (
            (0, "v(2)", 4),
            (2, "v(3)", 5),
            Arch::Gfx1201,
            "VOPD destination parity collision",
        ),
        (
            (0, "v(2)", 4),
            (1, "v(6)", 5),
            Arch::Gfx1201,
            "VOPD src0 bank collision",
        ),
        (
            (0, "v(2)", 4),
            (1, "v(3)", 0),
            Arch::Gfx1201,
            "VOPD src1 bank collision",
        ),
        (
            (0, "(\"lit\", 1)", 4),
            (1, "(\"lit\", 2)", 5),
            Arch::Gfx1201,
            "VOPD supports only one shared literal",
        ),
        (
            (0, "v(2)", 4),
            (1, "v(3)", 0),
            Arch::Gfx1151,
            "VOPD src1 bank collision",
        ),
        (
            (0, "v(1)", 4),
            (1, "v(2)", 5),
            Arch::Gfx1151,
            "gfx11 VOPD half reads the other half's destination",
        ),
    ] {
        let body = format!(
            r#"
            b.vopd(
                VopdOp {{ op: "add", dst: {}, src0: {}, src1: {} }},
                VopdOp {{ op: "mul", dst: {}, src0: {}, src1: {} }}
            )?;
        "#,
            x.0, x.1, x.2, y.0, y.1, y.2
        );
        refuses(&kernel(&body), arch, expected);
    }
}

/// Remaining Rust methods without a corresponding native method. This is an
/// inventory, not a promise that compile-time Rust capabilities are scriptable.
/// Workgroup/Wave (the `Nat::Wg` surface) and KernargLayout have no remaining
/// omissions; renamed wave/workgroup exit methods count as bound.
pub const STILL_UNREACHABLE: &[(&str, &str)] = &[
    ("Builder::program", "Borrowed Rust Program inspection has no script-native representation; position() exposes its length."),
    ("Builder::ledger", "Borrowed Rust Ledger inspection has no script-native representation."),
    ("Builder::label", "Labels are exposed on typed Workgroup/Wave; untyped loop heads are placed by loop_."),
    ("Builder::wait_all", "The untyped builder currently exposes explicit wait(Counter, count), not ledger-wide draining."),
    ("Builder::barrier", "The explicit barrier_with_scope(..., lds_only) binding covers the untyped shorthand."),
    ("Builder::vopd_ff", "Rust const-generic VGPR parity and bank witnesses have no native representation."),
    ("Builder::vopd_ff_shared_src1", "Rust const-generic VGPR parity witnesses have no native representation."),
    ("RegPlan::temp_v", "Requires a Rust Scope and const-generic typed register result; no native Scope value exists."),
    ("RegPlan::verify_access", "Requires Rust register-label order inspection; Builder emission already performs this check."),
    ("RegPlan::verify_lifetimes", "Requires Rust label-order inspection; Builder::finish already performs this check."),
    ("raw barriers: rotate", "Driver ring transitions require private runtime slot mapping; use Workgroup::barrier."),
    ("raw barriers: prime", "Driver ring transitions require private runtime slot mapping; use Workgroup::barrier."),
    ("raw barriers: retire_cur", "Driver ring transitions require private runtime slot mapping; use Workgroup::barrier."),
];
