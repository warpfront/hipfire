// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scripts over N-slot rings: `Ring::new(a, b[, c[, d]])`, `ring.depth()`,
//! `ring.starting_at(k)`, the store-free `advance(ring)` transition and the
//! two-armed `wg.wg_if_else(..)` whose arms may hold barriers.

use peacemaker_isa::Arch;

const PREAMBLE: &str = r#"
region KTile;

fn store() -> Instruction {
    Instruction::new("ds_store_b128 v1, v[2:5]", vec![], vec![v(1), vr(2, 4)]).memory(MemoryClass::DsStore)
}

fn load(off: usize) -> Instruction {
    Instruction::new(format!("ds_load_b128 v[8:11], v6 offset:{off}"), vec![vr(8, 4)], vec![v(6)]).memory(MemoryClass::DsLoad)
}

fn emit(spec: Spec) -> Result<Emitted, String> {
    let kspec = KernelSpec {
        kernel_id: "ring_depth".into(), variant: "v".into(),
        arch: spec.arch, symbol: "ring_depth".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 32768, wave32: true, cu_mode: false,
    };
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4 { p.v::<8>("vectors", 8 * i, Live::Whole)?; }
    p.s::<8>("scalars", 0, Live::Whole)?;
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    for r in [1, 2, 3, 4, 5, 6] { op(wg.isa(), format!("v_mov_b32_e32 v{r}, 0"), &[v(r)], &[])?; }
    let k0 = wg.lds::<KTile>("K0", 0, 4096)?;
    let k1 = wg.lds::<KTile>("K1", 4096, 4096)?;
    let k2 = wg.lds::<KTile>("K2", 8192, 4096)?;
    let k3 = wg.lds::<KTile>("K3", 12288, 4096)?;
    let k4 = wg.lds::<KTile>("K4", 16384, 4096)?;
"#;

const TAIL: &str = r#"
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}
"#;

fn script(body: &str) -> String {
    format!("{PREAMBLE}{body}{TAIL}")
}

fn compile(body: &str) -> Result<String, String> {
    peacemaker_rip::compile_named(&script(body), "ring_depth.rip", Arch::Gfx1201, "v")
        .map(|e| e.s_text)
}

/// Store into the ring's next buffer, drain and publish with `how`.
const FILL: &str = r#"
    let (ring, st) = wg.ds_store(ring, store())?;
    let d = wg.wait(st)?;
    let (ring,) = wg.barrier((HOW(ring, d),))?;
"#;

fn fill(how: &str) -> String {
    FILL.replace("HOW", how)
}

#[test]
fn ring_new_takes_three_regions_and_fills_over_the_vacant_cur() {
    let body = format!(
        r#"
    let ring = Ring::new(k0, k1, k2);
{}{}
    let off = ring.depth() * 1000 + ring.cur_index();
    wg.ds_load_cur(&ring, load(off))?;
    let (ring,) = wg.barrier((advance(ring),))?;
    let off = ring.depth() * 1000 + ring.cur_index();
    wg.ds_load_cur(&ring, load(off))?;
"#,
        fill("prime"),
        fill("rotate")
    );
    let asm = compile(&body).unwrap();
    // depth 3, current buffer 0 after the fill, 1 after the advance.
    assert!(
        asm.contains("ds_load_b128 v[8:11], v6 offset:3000"),
        "{asm}"
    );
    assert!(
        asm.contains("ds_load_b128 v[8:11], v6 offset:3001"),
        "{asm}"
    );
}

#[test]
fn ring_new_takes_four_regions() {
    let body = format!(
        r#"
    let ring = Ring::new(k0, k1, k2, k3);
{}{}{}
    let off = ring.depth() * 1000 + ring.cur_index();
    wg.ds_load_cur(&ring, load(off))?;
"#,
        fill("prime"),
        fill("rotate"),
        fill("rotate")
    );
    let asm = compile(&body).unwrap();
    // A depth-4 ring fills over three barriers; buffer 0 is current then.
    assert!(asm.contains("offset:4000"), "{asm}");
}

#[test]
fn ring_new_refuses_a_depth_outside_two_to_four() {
    let e = compile("    let ring = Ring::new(k0, k1, k2, k3, k4);").unwrap_err();
    assert!(e.contains("`Ring::new` takes 2 to 4 LDS regions"), "{e}");
    let e = compile("    let ring = Ring::new(k0);").unwrap_err();
    assert!(e.contains("`Ring::new` takes 2 to 4 LDS regions"), "{e}");
}

#[test]
fn starting_at_picks_the_first_store_buffer() {
    let body = format!(
        r#"
    let ring = Ring::new(k0, k1, k2).starting_at(2);
{}{}
    let off = ring.depth() * 1000 + ring.cur_index();
    wg.ds_load_cur(&ring, load(off))?;
"#,
        fill("prime"),
        fill("rotate")
    );
    // The first store went to buffer 2 and the second to buffer 0: current 2.
    let asm = compile(&body).unwrap();
    assert!(asm.contains("offset:3002"), "{asm}");
    // Once a barrier published something the start is fixed.
    let late = format!(
        r#"
    let ring = Ring::new(k0, k1, k2);
{}
    let ring = ring.starting_at(0);
"#,
        fill("prime")
    );
    let e = compile(&late).unwrap_err();
    assert!(e.contains("before anything is published"), "{e}");
    let e = compile("    let ring = Ring::new(k0, k1, k2).starting_at(3);").unwrap_err();
    assert!(e.contains("outside a depth-3 ring"), "{e}");
}

#[test]
fn advance_needs_a_published_ahead_buffer() {
    // Depth 2 never has one.
    let body = format!(
        r#"
    let ring = Ring::new(k0, k1);
{}
    wg.ds_load_cur(&ring, load(0))?;
    let (ring,) = wg.barrier((advance(ring),))?;
"#,
        fill("prime")
    );
    let e = compile(&body).unwrap_err();
    assert!(e.contains("advance of ring"), "{e}");
}

fn if_else(then_start: usize, else_start: usize) -> Result<String, String> {
    let arm = |start: usize| {
        format!(
            r#"|wg, ring| {{
        let ring = ring.starting_at({start});
{}
        Ok(ring)
    }}"#,
            fill("prime")
        )
    };
    let body = format!(
        r#"
    let ring = Ring::new(k0, k1, k2);
    let c = wg.scmp_wg_uniform(Instruction::new("s_cmp_lg_u32 s0, 0", vec![], vec![s(0)]))?;
    let ring = wg.wg_if_else(c, ".Lelse", ".Ljoin", ring, {}, {})?;
"#,
        arm(then_start),
        arm(else_start)
    );
    compile(&body)
}

#[test]
fn wg_if_else_arms_may_hold_barriers_when_they_end_equal() {
    let asm = if_else(1, 1).unwrap();
    let at = |needle: &str| {
        asm.find(needle)
            .unwrap_or_else(|| panic!("{needle}\n{asm}"))
    };
    assert!(at("s_cbranch_scc1 .Lelse") < at("s_branch .Ljoin"));
    assert!(at("s_branch .Ljoin") < at(".Lelse:"));
    assert!(at(".Lelse:") < at(".Ljoin:"));
    // One barrier per arm.
    assert_eq!(asm.matches("s_barrier_wait").count(), 2, "{asm}");
}

#[test]
fn wg_if_else_refuses_arms_that_end_in_different_ring_states() {
    let e = if_else(0, 1).unwrap_err();
    assert!(e.contains("the arms end in different LDS states"), "{e}");
}
