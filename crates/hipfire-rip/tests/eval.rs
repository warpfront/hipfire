//! Evaluator behavior at the language boundaries a kernel script relies on:
//! integer typing, lexical capture, places, control flow, formatting,
//! error positions, and the builder / driver vocabulary.

use hipfire_isa::Arch;

fn run(src: &str) -> Result<String, String> {
    hipfire_rip::eval(src, "main")
}
fn ok(src: &str) -> String {
    run(src).unwrap_or_else(|e| panic!("{e}"))
}
fn err(src: &str) -> String {
    run(src).err().expect("the script must fail")
}

#[test]
fn integer_types_are_exact_and_literals_adopt_them() {
    assert_eq!(ok("fn main() -> u8 { let a: u8 = 200; a + 50 }"), "250");
    assert!(err("fn main() -> u8 { let a: u8 = 200; a + 56 }").contains("overflow"));
    assert!(err("fn main() -> u8 { let a: u8 = 0; a - 1 }").contains("overflow"));
    // Mixed typed operands are a type error, as in rustc.
    assert!(err("fn main() -> u32 { let a: u8 = 1; let b: u32 = 2; a + b }").contains("mismatched"));
    // `as` truncates, `u32::from` widens.
    assert_eq!(ok("fn main() -> u8 { 0x1ffu32 as u8 }"), "255");
    assert_eq!(ok("fn main() -> u32 { u32::from(7u8) + 1 }"), "8");
    // A declared parameter type coerces literals and rejects out-of-range ones.
    assert_eq!(ok("fn f(x: u8) -> u8 { x } fn main() -> u8 { f(255) }"), "255");
    assert!(err("fn f(x: u8) -> u8 { x } fn main() -> u8 { f(256) }").contains("out of range"));
}

#[test]
fn closures_capture_the_environment_they_were_written_in() {
    let src = "fn main() -> u32 { let x: u32 = 1; let f = |y: u32| x + y; let x: u32 = 100; f(2) + x }";
    assert_eq!(ok(src), "103");
    // The captured binding is shared, not copied: a later write is seen.
    assert_eq!(ok("fn main() -> u32 { let mut n: u32 = 1; let get = || n; n = 5; get() }"), "5");
}

#[test]
fn places_methods_and_enums_follow_rust() {
    let src = r#"
        #[derive(Clone, Copy, PartialEq)]
        enum Kind { A, B }
        impl Kind {
            const ALL: [Kind; 2] = [Kind::A, Kind::B];
            fn tag(self) -> &'static str { match self { Self::A => "a", Self::B => "b" } }
        }
        struct Acc { n: u32, items: Vec<u32> }
        impl Acc {
            fn add(&mut self, x: u32) { self.n += x; self.items.push(x); }
        }
        fn main() -> String {
            let mut acc = Acc { n: 0, items: vec![] };
            for (i, k) in Kind::ALL.into_iter().enumerate() {
                acc.add(i as u32 + 10);
                if k == Kind::B { acc.add(1); }
            }
            format!("{}{} {} {:?}", Kind::A.tag(), Kind::B.tag(), acc.n, acc.items)
        }
    "#;
    assert_eq!(ok(src), "\"ab 22 [10, 11, 1]\"");
}

#[test]
fn control_flow_loops_labels_and_patterns() {
    let src = r#"
        fn main() -> u32 {
            let mut total: u32 = 0;
            'outer: for i in 0..10u32 {
                for j in 0..10u32 {
                    if j > i { continue 'outer; }
                    if i == 6 { break 'outer; }
                    total += j;
                }
            }
            let found = loop { total += 1; if total % 7 == 0 { break total; } };
            let pair = (3u32, Some(4u32));
            let (a, Some(b)) = pair else { return 0 };
            if let Some(x) = pair.1 && x > 3 { found + a + b } else { 0 }
        }
    "#;
    // Rows 0..=5 sum j=0..=i (35); the loop then stops at the first multiple of 7 (42); 42 + 3 + 4.
    assert_eq!(ok(src), "49");
    assert_eq!(ok("fn f(n: u32) -> u32 { match n { 0 => 10, 1..=3 => 20, x if x > 100 => 30, _ => 40 } } fn main() -> u32 { f(0) + f(2) + f(500) + f(50) }"), "100");
}

#[test]
fn question_mark_and_early_return_propagate_script_errors() {
    let src = r#"
        fn g(x: u32) -> Result<u32, String> { if x > 1 { Err(format!("bad {x}")) } else { Ok(x) } }
        fn f(x: u32) -> Result<u32, String> { let y = g(x)?; Ok(y + 1) }
        fn main() -> String { format!("{:?} {:?}", f(1), f(2)) }
    "#;
    assert_eq!(ok(src), "\"Ok(2) Err(\\\"bad 2\\\")\"");
    // collect over Results short-circuits and keeps the first Err.
    let src = "fn main() -> Result<Vec<u32>, String> { [1u32, 2, 3].into_iter().map(|x| if x == 2 { Err(\"two\".to_string()) } else { Ok(x) }).collect::<Result<Vec<_>, _>>() }";
    assert_eq!(ok(src), "Err(\"two\")");
}

#[test]
fn format_strings_cover_positional_named_captured_and_radix() {
    let src = "fn main() -> String { let n = 255u32; let w = 7u32; format!(\"{0}-{0:#x}-{v:04}-{n:x}-{w:>3}-{{}}\", n, v = w) }";
    assert_eq!(ok(src), "\"255-0xff-0007-ff-  7-{}\"");
}

#[test]
fn faults_carry_their_source_position_and_unreached_code_is_not_examined() {
    let e = err("fn main() -> u32 {\n    let a: u8 = 1;\n    a.nope()\n}");
    assert!(e.starts_with("rip:3:"), "{e}");
    assert!(e.contains("no method `nope`"), "{e}");
    // Unsupported syntax in a function nobody calls is fine ...
    assert_eq!(ok("fn unused() { let _ = async { 1 }; } fn main() -> u32 { 3 }"), "3");
    // ... and an executed one is a located error, never a no-op.
    let e = err("fn main() -> u32 { weird!(1); 2 }");
    assert!(e.contains("macro `weird!` is not supported") && e.starts_with("rip:1:"), "{e}");
    // A syntax error is positioned too.
    let e = hipfire_rip::parse("fn main() {\n  let = ;\n}", "t.rip").err().expect("syntax error");
    assert!(e.starts_with("t.rip:2:"), "{e}");
}

fn spec_script(body: &str) -> String {
    format!(
        r#"
        region T;
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

#[test]
fn builder_vocabulary_emits_instructions_and_the_driver_scope_ends_at_the_next_builder_use() {
    let src = spec_script(r#"op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;"#);
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let e = hipfire_rip::compile(&src, arch, "v").unwrap_or_else(|e| panic!("{e}"));
        assert!(e.s_text.contains("v_mov_b32_e32 v0, 0") && e.s_text.contains("s_endpgm"), "{}", e.s_text);
        assert_eq!(hipfire_rip::symbol_of(&e).unwrap(), "t_kernel");
    }
    // Using `wg` after the statement that mentions `b` again is outside the scope.
    let src = r#"
        fn emit(spec: Spec) -> Result<Emitted, String> {
            let kspec = KernelSpec { kernel_id: "t".into(), variant: "v".into(), arch: spec.arch, symbol: "t".into(),
                kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
                workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false };
            let mut b = Builder::new(kspec, RegPlan::new(256, 104)?);
            let mut wg = Workgroup::new(&mut b, spec.arch)?;
            b.enable_delay_alu();
            let end = wg.exit(".Lend")?;
            wg.end(end)?;
            b.finish()
        }"#;
    let e = hipfire_rip::compile(src, Arch::Gfx1151, "v").err().unwrap();
    assert!(e.contains("no active Workgroup"), "{e}");
}

#[test]
fn lds_handoff_needs_its_wait_before_the_barrier() {
    let store = r#"
        let t = wg.lds::<T>("t", 0, 16)?;
        let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    "#;
    let good = spec_script(&format!("{store} let d = wg.wait(st)?; let (t,) = wg.barrier((ready(t, d),))?; wg.ds_load(&t, Instruction::new(\"ds_load_b32 v2, v0\", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"));
    let e = hipfire_rip::compile(&good, Arch::Gfx1151, "v").unwrap_or_else(|e| panic!("{e}"));
    assert!(e.s_text.contains("s_barrier"));
    // The same program without the wait: the driver names the transition it refuses.
    let bad = spec_script(&format!("{store} let (t,) = wg.barrier((ready(t, st),))?; wg.ds_load(&t, Instruction::new(\"ds_load_b32 v2, v0\", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"));
    let e = hipfire_rip::compile(&bad, Arch::Gfx1151, "v").err().unwrap();
    assert!(e.contains("undrained"), "{e}");
}

#[test]
fn chains_emit_wmma_with_exact_operands_and_reject_unsupported_shapes() {
    let ok_src = spec_script(r#"let acc = Chain::new(8, 8, "f16"); acc.step(wg.isa(), 16, 4, 24, 4)?;"#);
    let e = hipfire_rip::compile(&ok_src, Arch::Gfx1201, "v").unwrap_or_else(|e| panic!("{e}"));
    assert!(e.s_text.contains("v_wmma_f32_16x16x16_f16 v[8:15], v[16:19], v[24:27], v[8:15]"), "{}", e.s_text);
    let bad = spec_script(r#"let acc = Chain::new(8, 8, "iu4"); acc.step(wg.isa(), 16, 4, 24, 4)?;"#);
    assert!(hipfire_rip::compile(&bad, Arch::Gfx1201, "v").err().unwrap().contains("unsupported matrix kind"));
    let bad = spec_script(r#"let acc = Chain::new(8, 8, "f16"); acc.step(wg.isa(), 16, 3, 24, 4)?;"#);
    assert!(hipfire_rip::compile(&bad, Arch::Gfx1201, "v").err().unwrap().contains("operand widths"));
}

#[test]
fn register_plan_widths_other_than_1_2_4_8_are_refused_by_the_plan() {
    let src = r#"fn main() -> Result<(), String> { let mut p = RegPlan::new(256, 104)?; p.v::<32>("big", 0, Live::Whole)?; Ok(()) }"#;
    assert!(ok(src).contains("register width must be 1, 2, 4 or 8"));
}

#[test]
fn shared_helpers_spell_their_instructions_per_arch() {
    let src = r#"
        fn emit(spec: Spec) -> Result<Emitted, String> {
            let kspec = KernelSpec { kernel_id: "t".into(), variant: "v".into(), arch: spec.arch, symbol: "t_kernel".into(),
                kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
                workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false };
            let mut p = RegPlan::new(256, 104)?;
            for i in 0..4u8 { p.s::<8>("s", 8 * i, Live::Whole)?; }
            p.v::<8>("v", 0, Live::Whole)?;
            let mut b = Builder::new(kspec, p);
            let mut wg = Workgroup::new(&mut b, spec.arch)?;
            smem(wg.isa(), 8, 8, 0, 0)?;
            srd_tail(wg.isa(), 12, None)?;
            add64(wg.isa(), 16, 8, 24)?;
            sop(wg.isa(), format!("{} s17, s17, 1", s_add_i32(spec.arch)), &[17], &[17])?;
            let end = wg.exit(".Lend")?;
            wg.end(end)?;
            b.finish()
        }"#;
    let g11 = hipfire_rip::compile(src, Arch::Gfx1151, "v").unwrap_or_else(|e| panic!("{e}")).s_text;
    let g12 = hipfire_rip::compile(src, Arch::Gfx1201, "v").unwrap_or_else(|e| panic!("{e}")).s_text;
    for want in ["s_load_b256 s[8:15], s[0:1], null", "s_mov_b32 s15, 0x31004000", "s_add_u32 s16, s8, s24", "s_addc_u32 s17, s9, 0", "s_add_i32 s17, s17, 1"] {
        assert!(g11.contains(want), "gfx1151 lacks `{want}`");
    }
    for want in ["s_load_b256 s[8:15], s[0:1], 0x0", "s_add_co_u32 s16, s8, s24", "s_add_co_ci_u32 s17, s9, 0", "s_add_co_i32 s17, s17, 1"] {
        assert!(g12.contains(want), "gfx1201 lacks `{want}`");
    }
}
