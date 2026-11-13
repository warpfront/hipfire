//! gfx942 uses the source driver and the checked CDNA backend, never the RDNA codec.
use peacemaker_isa::{
    reg::{Kind, RegRef},
    Arch,
};
use peacemaker_rip::{check, validate};

fn source(body: &str) -> String {
    format!(
        r#"
const ARCHES: &[&str] = &["gfx942"];
enum Kind {{ Default }}
region Scratch;
fn emit(spec: Spec) -> Result<Emitted, String> {{
    let arch = spec.arch;
    let k = KernelSpec {{ kernel_id: "cdna_test".into(), variant: "default".into(), arch,
        symbol: "cdna_test".into(), kernargs: KernargLayout::new(8).pointer("out", 0),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 256,
        group_segment_fixed_size: 4096, wave32: false, cu_mode: false }};
    let mut p = RegPlan::new(256, 100)?.with_agpr_budget(4)?;
    p.s::<2>("kernargs", 0, Live::Whole)?;
    p.s::<1>("wgid", 2, Live::Whole)?;
    p.s::<4>("args", 4, Live::Whole)?;
    p.v::<8>("scratch", 0, Live::Whole)?;
    p.a::<4>("acc", 0, Live::Whole)?;
    let mut b = Builder::new(k, p);
    let mut wg = Workgroup::new(&mut b, arch)?;
    let end = wg.exit(".Lend")?;
    {body}
    wg.end(end)?;
    b.finish()
}}
"#
    )
}

#[test]
fn wave64_checked_driver_compiles_and_emits_cdna_descriptor() {
    let result = check(
        &source(
            r#"
        op(wg.isa(), "v_mov_b32_e32 v1, v0", &[v(1)], &[v(0)])?;
        op(wg.isa(), "v_accvgpr_write_b32 a0, v1", &[a(0)], &[v(1)])?;
        op(wg.isa(), "v_accvgpr_read_b32 v2, a0", &[v(2)], &[a(0)])?;
    "#,
        ),
        "cdna.rip",
        Arch::Gfx942,
        "default",
    );
    assert!(result.emitted.is_some(), "{:?}", result.diagnostics);
    let asm = &result.emitted.unwrap().s_text;
    assert!(asm.contains("gfx942"));
    assert!(asm.contains(".amdhsa_accum_offset"));
    assert!(!asm.contains(".amdhsa_wavefront_size32 1"));
}

#[test]
fn cdna_mfma_effects_keep_agpr_and_vgpr_banks_distinct() {
    let (defs, uses) = validate::effects(
        Arch::Gfx942,
        "v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[0:3]",
    )
    .unwrap();
    assert!(defs.contains(&RegRef {
        kind: Kind::A,
        base: 0,
        len: 4
    }));
    assert!(uses.contains(&RegRef {
        kind: Kind::V,
        base: 0,
        len: 2
    }));
    assert!(uses.contains(&RegRef {
        kind: Kind::A,
        base: 0,
        len: 4
    }));
    let error = validate::instruction(
        Arch::Gfx942,
        "v_accvgpr_read_b32 v1, a0",
        &[RegRef {
            kind: Kind::V,
            base: 1,
            len: 1,
        }],
        &[],
    )
    .unwrap_err();
    assert!(error.contains("reads a0"), "{error}");
}

#[test]
fn cdna_driver_rejects_unpublished_lds_and_unowned_registers() {
    for (body, needle) in [
        (
            r#"let r = wg.lds::<Scratch>("r", 0, 256)?;
            wg.ds_load(r, Instruction::new("ds_read_b32 v1, v0", &[v(1)], &[v(0)]).memory(MemoryClass::DsLoad))?;"#,
            "Published",
        ),
        (
            r#"op(wg.isa(), "v_mov_b32_e32 v9, v0", &[v(9)], &[v(0)])?;"#,
            "owner",
        ),
    ] {
        let result = check(&source(body), "cdna.rip", Arch::Gfx942, "default");
        assert!(result.emitted.is_none());
        assert!(
            format!("{:?}", result.diagnostics)
                .to_lowercase()
                .contains(&needle.to_lowercase()),
            "{:?}",
            result.diagnostics
        );
    }
}

#[test]
fn cdna_publishes_and_retires_lds_through_full_barriers() {
    let result = check(
        &source(
            r#"
        op(wg.isa(), "s_mov_b32 m0, -1", &[], &[])?;
        let r = wg.lds::<Scratch>("r", 0, 256)?;
        let (r, pending) = wg.ds_store(r, Instruction::new("ds_write_b32 v0, v0", &[], &[v(0)]).memory(MemoryClass::DsStore))?;
        let d = wg.wait(pending)?;
        let (r,) = wg.barrier((ready(r, d),))?;
        wg.ds_load(r, Instruction::new("ds_read_b32 v1, v0", &[v(1)], &[v(0)]).memory(MemoryClass::DsLoad))?;
        let (r,) = wg.barrier((retire(r),))?;
    "#,
        ),
        "cdna.rip",
        Arch::Gfx942,
        "default",
    );
    assert!(result.emitted.is_some(), "{:?}", result.diagnostics);
    assert!(result.emitted.unwrap().s_text.contains("s_barrier"));
}

#[test]
fn cdna_wave64_definedness_checks_high_exec_mask_source() {
    let result = check(
        &source(
            r#"
        sop(wg.isa(), "s_mov_b32 s4, -1", &[4], &[])?;
        op(wg.isa(), "s_mov_b64 exec, s[4:5]", &[], &[sr(4, 2)])?;
    "#,
        ),
        "cdna.rip",
        Arch::Gfx942,
        "default",
    );
    assert!(result.emitted.is_none());
    assert!(
        format!("{:?}", result.diagnostics).contains("s5"),
        "{:?}",
        result.diagnostics
    );
}

#[test]
fn cdna_exec_if_sets_and_restores_both_half_waves() {
    let result = check(
        &source(
            r#"
        let c = wg.scmp(Instruction::new("s_cmp_eq_u32 s2, 0", &[], &[s(2)]))?;
        wg.exec_if(c, |w| {
            op(w.isa(), "v_mov_b32_e32 v1, v0", &[v(1)], &[v(0)])
        })?;
    "#,
        ),
        "cdna.rip",
        Arch::Gfx942,
        "default",
    );
    assert!(result.emitted.is_some(), "{:?}", result.diagnostics);
    let asm = &result.emitted.unwrap().s_text;
    assert!(asm.contains("s_cselect_b64 exec, -1, 0"), "{asm}");
    assert!(asm.contains("s_mov_b64 exec, -1"), "{asm}");
    assert!(!asm.contains("exec_lo"), "{asm}");
}

const MOE: &str = include_str!("../../../kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.rip");

#[test]
fn grouped_moe_has_all_three_entries_and_a_native_module() {
    let (emitted, asm, _) = peacemaker_rip::module(
        MOE,
        Arch::Gfx942,
        &["f32", "xbf16", "pad16"],
        "fp8_moe_cdna",
    )
    .unwrap();
    assert_eq!(emitted.len(), 3);
    let elf = peacemaker_isa::native::assemble(&asm, Arch::Gfx942).unwrap();
    let lifted = peacemaker_lift::cdna::lift_object(&elf).unwrap();
    assert_eq!(lifted.kernels.len(), 3);
    for k in &lifted.kernels {
        peacemaker_isa::cdna::lifted::certify(k).unwrap();
    }
}

#[test]
fn grouped_moe_runs_both_x_types_and_padding_on_the_cdna_emulator() {
    let mut weights = vec![0x38u8; 2048];
    weights.extend_from_slice(&[0x80, 0x3f]);
    weights.resize(2304, 0);
    let hex: String = weights.iter().map(|b| format!("{b:02x}")).collect();
    for (variant, dtype, x) in [
        ("f32", "f32", serde_json::json!(vec![1.0f32; 128])),
        ("xbf16", "u16", serde_json::json!(vec![0x3f80u16; 128])),
    ] {
        let expected: Vec<f32> = [vec![128.0; 16], vec![0.0; 16]].concat();
        let case = serde_json::json!({
            "name": "K128-live-and-padding", "block": [256,1,1], "grid": [1,1,1],
            "buffers": [
                {"name":"W","dtype":"bytes","values":hex},
                {"name":"td","dtype":"i32","values":[0,0,2,0]},
                {"name":"rows","dtype":"i32","values":[0,-1]},
                {"name":"X","dtype":dtype,"values":x},
                {"name":"Y","dtype":"f32","expect":expected}
            ],
            "kernargs":{"size":64,"fields":[
                {"offset":0,"ptr":"W"},{"offset":8,"u64":2304},{"offset":16,"ptr":"td"},
                {"offset":24,"ptr":"rows"},{"offset":32,"ptr":"X"},{"offset":40,"ptr":"Y"},
                {"offset":48,"u32":16},{"offset":52,"u32":128},{"offset":56,"u32":1},{"offset":60,"u32":1}
            ]}
        });
        let outcome = peacemaker_rip::run::execute(
            MOE,
            "moe.rip",
            Arch::Gfx942,
            variant,
            &case.to_string(),
            false,
        );
        assert_eq!(outcome.report["ok"], true, "{}", outcome.report);
    }
}

#[test]
fn grouped_moe_multiblock_scales_rows_and_expert_stride_are_exact() {
    // Dyadic weights/scales and integer X keep every partial sum exactly representable,
    // making this an independent byte oracle, not a tolerance check on MFMA order.
    const R: usize = 144;
    const K: usize = 256;
    let stride = (R * K + 8).next_multiple_of(256);
    let scale = |r: usize, kb: usize| [[0.5f32, 2.0], [1.0, 0.5]][r / 128][kb];
    let weight = |expert: usize, r: usize, k: usize| {
        let m = (r * 7 + k * 3 + k / 16 * 5 + k / 128 + expert) % 4;
        let negative = (r * 11 + k / 4 + k / 16 + expert) % 3 == 0;
        let value = [0.25f32, 0.5, 1.0, 2.0][m];
        (
            [0x28u8, 0x30, 0x38, 0x40][m] | if negative { 0x80 } else { 0 },
            if negative { -value } else { value },
        )
    };
    let mut weights = vec![0u8; 2 * stride];
    for expert in 0..2 {
        for r in 0..R {
            for k in 0..K {
                weights[expert * stride + r * K + k] = weight(expert, r, k).0;
            }
        }
        for rb in 0..2 {
            for kb in 0..2 {
                let bits = (scale(rb * 128, kb).to_bits() >> 16) as u16;
                let at = expert * stride + R * K + (rb * 2 + kb) * 2;
                weights[at..at + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    let x: Vec<f32> = (0..3 * K)
        .map(|i| ((i / K * 11 + i % K * 7 + i % K / 16 * 3) % 13) as f32 - 6.0)
        .collect();
    let expected: Vec<f32> = [None, Some(1usize), Some(2)]
        .into_iter()
        .flat_map(|xr| {
            let x = &x;
            (0..R).map(move |r| {
                xr.map_or(0.0, |xr| {
                    (0..K).fold(0.0, |sum, k| {
                        sum + weight(1, r, k).1 * scale(r, k / 128) * x[xr * K + k]
                    })
                })
            })
        })
        .collect();
    let hex: String = weights.iter().map(|b| format!("{b:02x}")).collect();
    for (variant, dtype, values) in [
        ("f32", "f32", serde_json::json!(x)),
        (
            "xbf16",
            "u16",
            serde_json::json!(x
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>()),
        ),
    ] {
        let case = serde_json::json!({
            "name":"K256-two-scale-rows-expert1", "block":[256,1,1], "grid":[3,1,1],
            "buffers":[
                {"name":"W","dtype":"bytes","values":hex},
                {"name":"td","dtype":"i32","values":[1,0,3,0]},
                {"name":"rows","dtype":"i32","values":[-1,10,23]},
                {"name":"X","dtype":dtype,"values":values},
                {"name":"Y","dtype":"f32","expect":expected}
            ],
            "kernargs":{"size":64,"fields":[
                {"offset":0,"ptr":"W"},{"offset":8,"u64":stride},{"offset":16,"ptr":"td"},
                {"offset":24,"ptr":"rows"},{"offset":32,"ptr":"X"},{"offset":40,"ptr":"Y"},
                {"offset":48,"u32":R},{"offset":52,"u32":K},
                {"offset":56,"u32":10},{"offset":60,"u32":2}
            ]}
        });
        let outcome = peacemaker_rip::run::execute(
            MOE,
            "moe.rip",
            Arch::Gfx942,
            variant,
            &case.to_string(),
            false,
        );
        assert_eq!(outcome.report["ok"], true, "{}", outcome.report);
    }
}

#[test]
fn pad16_runs_two_workgroups_and_leaves_tail_guards_untouched() {
    let ids: Vec<i32> = (0..259).map(|i| i % 13 - 1).collect();
    let expected: Vec<i32> = ids
        .iter()
        .enumerate()
        .flat_map(|(t, &id)| [id, 16 * t as i32, 16, 0])
        .collect();
    let case = serde_json::json!({
        "name":"descriptor-tail", "block":[256,1,1], "grid":[2,1,1],
        "buffers":[
            {"name":"ids","dtype":"i32","values":ids},
            {"name":"td","dtype":"i32","expect":expected}
        ],
        "kernargs":{"size":20,"fields":[
            {"offset":0,"ptr":"ids"},{"offset":8,"ptr":"td"},{"offset":16,"u32":259}
        ]}
    });
    let outcome = peacemaker_rip::run::execute(
        MOE,
        "moe.rip",
        Arch::Gfx942,
        "pad16",
        &case.to_string(),
        false,
    );
    assert_eq!(outcome.report["ok"], true, "{}", outcome.report);
}

#[test]
fn open_m0_lds_entry_obligation_is_not_suppressed() {
    // The CDNA footprint follows ISA §11.3.1: DS memory reads M0. Whether
    // gfx942 hardware ignores it (as LLVM assumes) is OPEN, not an exemption.
    let result = check(
        &source(
            r#"
        let r = wg.lds::<Scratch>("r", 0, 256)?;
        let (r, p) = wg.ds_store(r, Instruction::new("ds_write_b32 v0, v0", &[], &[v(0)]).memory(MemoryClass::DsStore))?;
        let d = wg.wait(p)?;
        let (r,) = wg.barrier((ready(r, d),))?;
        wg.ds_load(r, Instruction::new("ds_read_b32 v1, v0", &[v(1)], &[v(0)]).memory(MemoryClass::DsLoad))?;
        let (r,) = wg.barrier((retire(r),))?;
    "#,
        ),
        "cdna.rip",
        Arch::Gfx942,
        "default",
    );
    assert!(result.emitted.is_none());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code == "R0604" && d.message.contains("m0")),
        "{:?}",
        result.diagnostics
    );
}
