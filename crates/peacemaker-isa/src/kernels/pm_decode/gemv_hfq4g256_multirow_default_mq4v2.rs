//! gfx1201 multirow-r2 candidate: independent rows and four ordered streams.
//! G0 DOG region admitted by the real-H2 oracle; full-symbol admission is separate.

use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Live, V},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

const MODULE: &str = "gemv_hfq4g256_multirow_default_mq4v2";
const G0_SYMBOL: &str = "gemv_mq4g256v2_multirow_r2_dog_g0";

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    use crate::plan::Access::{ReadOnly, WriteOnly};
    use crate::reg::Kind;
    let mut regs = RegPlan::new(32, 24)?;
    for (name, base, width) in [
        ("tid", 0, 1),
        ("address", 1, 1),
        ("packed0", 2, 1),
        ("packed1", 3, 1),
        ("header0", 4, 1),
        ("header1", 5, 1),
        ("weight", 6, 1),
        ("dot0", 7, 1),
        ("dot1", 8, 1),
        ("shuffle_address", 9, 1),
        ("x_lo", 12, 4),
        ("x_hi", 16, 4),
        ("acc0", 20, 4),
        ("acc1", 24, 4),
        ("sum0", 28, 1),
        ("sum1", 29, 1),
        ("exchange0", 30, 1),
        ("exchange1", 31, 1),
    ] {
        regs.add_range(name, Kind::V, base, width, Live::Whole)?;
    }
    for (name, base, width) in [
        ("kernarg", 0, 2),
        ("row0", 3, 1),
        ("a_x", 4, 4),
        ("y_m_k", 8, 4),
        ("groups", 12, 1),
        ("stride", 13, 1),
        ("row1", 14, 1),
        ("group", 15, 1),
        ("row_ptr0", 16, 2),
        ("row_ptr1", 18, 2),
        ("temporary", 20, 1),
        ("x_offset", 21, 1),
        ("weight_offset", 22, 1),
        ("quads_end", 23, 1),
    ] {
        regs.add_range(name, Kind::S, base, width, Live::Whole)?;
    }
    let spec = KernelSpec {
        kernel_id: MODULE.into(),
        variant: "multirow-r2-exact".into(),
        arch: Arch::Gfx1201,
        symbol: "gemv_mq4g256v2_multirow_r2".into(),
        kernargs: KernargLayout::new(32)
            .pointer_access("A", 0, ReadOnly)
            .pointer_access("x", 8, ReadOnly)
            .pointer_access("y", 16, WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, regs);
    b.enable_delay_alu();
    b.push(
        Instruction::new(
            "s_load_b128 s[4:7], s[0:1], 0x0",
            vec![crate::S::<4>(4).reg()],
            vec![crate::S::<2>(0).reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(
        Instruction::new(
            "s_load_b128 s[8:11], s[0:1], 0x10",
            vec![crate::S::<4>(8).reg()],
            vec![crate::S::<2>(0).reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    // gfx12 receives the workgroup index in ttmp9, not an ordinary SGPR.
    b.push(Instruction::new(
        "s_lshl_b32 s3, ttmp9, 1",
        vec![crate::S::<1>(3).reg()],
        vec![],
    ))?;
    b.push(Instruction::new(
        "s_cmp_ge_i32 s3, s10",
        vec![],
        vec![crate::S::<1>(3).reg(), crate::S::<1>(10).reg()],
    ))?;
    b.control(Instruction::new(
        "s_cbranch_scc1 .Lmultirow_exit",
        vec![],
        vec![],
    ))?;
    b.push(Instruction::new(
        "s_lshr_b32 s12, s11, 8",
        vec![crate::S::<1>(12).reg()],
        vec![crate::S::<1>(11).reg()],
    ))?;
    b.push(Instruction::new(
        "s_mul_i32 s13, s12, 0x88",
        vec![crate::S::<1>(13).reg()],
        vec![crate::S::<1>(12).reg()],
    ))?;
    b.push(Instruction::new(
        "s_add_co_i32 s14, s3, 1",
        vec![crate::S::<1>(14).reg()],
        vec![crate::S::<1>(3).reg()],
    ))?;
    b.push(Instruction::new(
        "s_cmp_lt_i32 s14, s10",
        vec![],
        vec![crate::S::<1>(14).reg(), crate::S::<1>(10).reg()],
    ))?;
    b.push(Instruction::new(
        "s_cselect_b32 s14, s14, s3",
        vec![crate::S::<1>(14).reg()],
        vec![crate::S::<1>(14).reg(), crate::S::<1>(3).reg()],
    ))?;
    // The incumbent raw buffer uses the same 32-bit row offsets. The full
    // lm-head matrix is admitted only inside that existing offset range.
    for (row, pointer) in [(3, 16), (14, 18)] {
        b.push(Instruction::new(
            format!("s_mul_i32 s{pointer}, s{row}, s13"),
            vec![crate::S::<1>(pointer).reg()],
            vec![crate::S::<1>(row).reg(), crate::S::<1>(13).reg()],
        ))?;
        b.push(Instruction::new(
            format!("s_mov_b32 s{}, 0", pointer + 1),
            vec![crate::S::<1>(pointer + 1).reg()],
            vec![],
        ))?;
        b.push(Instruction::new(
            format!(
                "s_add_nc_u64 s[{pointer}:{}], s[4:5], s[{pointer}:{}]",
                pointer + 1,
                pointer + 1
            ),
            vec![crate::S::<2>(pointer).reg()],
            vec![crate::S::<2>(4).reg(), crate::S::<2>(pointer).reg()],
        ))?;
    }
    for acc in 20..28 {
        b.push(Instruction::new(
            format!("v_mov_b32_e32 v{acc}, 0"),
            vec![V::<1>(acc).reg()],
            vec![],
        ))?;
    }
    b.push(Instruction::new(
        "s_mov_b32 s15, 0",
        vec![crate::S::<1>(15).reg()],
        vec![],
    ))?;
    b.push(Instruction::new(
        "s_and_b32 s23, s12, -4",
        vec![crate::S::<1>(23).reg()],
        vec![crate::S::<1>(12).reg()],
    ))?;
    b.wait_all()?;
    b.push(Instruction::new(
        "s_cmp_eq_u32 s23, 0",
        vec![],
        vec![crate::S::<1>(23).reg()],
    ))?;
    b.control(Instruction::new(
        "s_cbranch_scc1 .Lmultirow_tail",
        vec![],
        vec![],
    ))?;
    b.loop_(".Lmultirow_quad", |b| {
        for stream in 0..4 {
            emit_group(b, stream)?;
        }
        b.push(Instruction::new(
            "s_cmp_lt_u32 s15, s23",
            vec![],
            vec![crate::S::<1>(15).reg(), crate::S::<1>(23).reg()],
        ))?;
        b.control(Instruction::new(
            "s_cbranch_scc1 .Lmultirow_quad",
            vec![],
            vec![],
        ))
    })?;
    b.label(".Lmultirow_tail")?;
    for stream in 0..3 {
        b.push(Instruction::new(
            "s_cmp_lt_u32 s15, s12",
            vec![],
            vec![crate::S::<1>(15).reg(), crate::S::<1>(12).reg()],
        ))?;
        b.control(Instruction::new(
            "s_cbranch_scc0 .Lmultirow_reduce",
            vec![],
            vec![],
        ))?;
        emit_group(&mut b, stream)?;
    }
    b.label(".Lmultirow_reduce")?;
    for (base, sum, exchange) in [(20, 28, 30), (24, 29, 31)] {
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{sum}, v{base}, v{}", base + 1),
            vec![V::<1>(sum).reg()],
            vec![V::<1>(base).reg(), V::<1>(base + 1).reg()],
        ))?;
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{exchange}, v{}, v{}", base + 2, base + 3),
            vec![V::<1>(exchange).reg()],
            vec![V::<1>(base + 2).reg(), V::<1>(base + 3).reg()],
        ))?;
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{sum}, v{sum}, v{exchange}"),
            vec![V::<1>(sum).reg()],
            vec![V::<1>(sum).reg(), V::<1>(exchange).reg()],
        ))?;
        b.ds_crosslane(
            Instruction::new(
                format!(
                    "ds_swizzle_b32 v{exchange}, v{sum} offset:swizzle(BITMASK_PERM,\"1pppp\")"
                ),
                vec![V::<1>(exchange).reg()],
                vec![V::<1>(sum).reg()],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{sum}, v{sum}, v{exchange}"),
            vec![V::<1>(sum).reg()],
            vec![V::<1>(sum).reg(), V::<1>(exchange).reg()],
        ))?;
    }
    for offset in [8, 4, 2, 1] {
        b.push(Instruction::new(
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
            vec![],
            vec![V::<1>(0).reg()],
        ))?;
        b.push(Instruction::new(
            "s_wait_alu depctr_va_vcc(0)",
            vec![],
            vec![],
        ))?;
        b.push(Instruction::new(
            format!("v_cndmask_b32_e64 v9, 0, {offset}, vcc_lo"),
            vec![V::<1>(9).reg()],
            vec![],
        ))?;
        b.push(Instruction::new(
            "v_add_lshl_u32 v9, v9, v0, 2",
            vec![V::<1>(9).reg()],
            vec![V::<1>(9).reg(), V::<1>(0).reg()],
        ))?;
        for (sum, exchange) in [(28, 30), (29, 31)] {
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{exchange}, v9, v{sum}"),
                    vec![V::<1>(exchange).reg()],
                    vec![V::<1>(9).reg(), V::<1>(sum).reg()],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            b.push(Instruction::new(
                format!("v_add_f32_e32 v{sum}, v{sum}, v{exchange}"),
                vec![V::<1>(sum).reg()],
                vec![V::<1>(sum).reg(), V::<1>(exchange).reg()],
            ))?;
        }
    }
    // Store row0, then store row1 only when the unclamped row exists.
    b.push(Instruction::new(
        "v_cmpx_eq_u32_e32 0, v0",
        vec![],
        vec![V::<1>(0).reg()],
    ))?;
    b.control(Instruction::new(
        "s_cbranch_execz .Lmultirow_exit",
        vec![],
        vec![],
    ))?;
    b.push(Instruction::new(
        "s_lshl_b32 s20, s3, 2",
        vec![crate::S::<1>(20).reg()],
        vec![crate::S::<1>(3).reg()],
    ))?;
    b.push(Instruction::new(
        "v_mov_b32_e32 v1, s20",
        vec![V::<1>(1).reg()],
        vec![crate::S::<1>(20).reg()],
    ))?;
    b.push(
        Instruction::new(
            "global_store_b32 v1, v28, s[8:9]",
            vec![],
            vec![V::<1>(1).reg(), V::<1>(28).reg(), crate::S::<2>(8).reg()],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.push(Instruction::new(
        "s_cmp_eq_u32 s14, s3",
        vec![],
        vec![crate::S::<1>(14).reg(), crate::S::<1>(3).reg()],
    ))?;
    b.control(Instruction::new(
        "s_cbranch_scc1 .Lmultirow_exit",
        vec![],
        vec![],
    ))?;
    b.push(
        Instruction::new(
            "global_store_b32 v1, v29, s[8:9] offset:4",
            vec![],
            vec![V::<1>(1).reg(), V::<1>(29).reg(), crate::S::<2>(8).reg()],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.wait_all()?;
    b.label(".Lmultirow_exit")?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    Ok(vec![b.finish()?])
}

/// One group, assigned to its unchanged g%4 stream, for both output rows.
fn emit_group(b: &mut Builder, stream: u8) -> Result<(), String> {
    b.push(Instruction::new(
        "s_mul_i32 s22, s15, 0x88",
        vec![crate::S::<1>(22).reg()],
        vec![crate::S::<1>(15).reg()],
    ))?;
    b.push(Instruction::new(
        "s_lshl_b32 s21, s15, 10",
        vec![crate::S::<1>(21).reg()],
        vec![crate::S::<1>(15).reg()],
    ))?;
    b.push(Instruction::new(
        "v_lshrrev_b32_e32 v1, 4, v0",
        vec![V::<1>(1).reg()],
        vec![V::<1>(0).reg()],
    ))?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 2, v1",
        vec![V::<1>(1).reg()],
        vec![V::<1>(1).reg()],
    ))?;
    b.push(Instruction::new(
        "v_add_nc_u32_e32 v1, s22, v1",
        vec![V::<1>(1).reg()],
        vec![crate::S::<1>(22).reg(), V::<1>(1).reg()],
    ))?;
    for (header, pointer) in [(4, 16), (5, 18)] {
        b.push(
            Instruction::new(
                format!(
                    "global_load_b32 v{header}, v1, s[{pointer}:{}]",
                    pointer + 1
                ),
                vec![V::<1>(header).reg()],
                vec![V::<1>(1).reg(), crate::S::<2>(pointer).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 2, v0",
        vec![V::<1>(1).reg()],
        vec![V::<1>(0).reg()],
    ))?;
    b.push(Instruction::new(
        "v_add_nc_u32_e32 v1, s22, v1",
        vec![V::<1>(1).reg()],
        vec![crate::S::<1>(22).reg(), V::<1>(1).reg()],
    ))?;
    for (packed, pointer) in [(2, 16), (3, 18)] {
        b.push(
            Instruction::new(
                format!(
                    "global_load_b32 v{packed}, v1, s[{pointer}:{}] offset:8",
                    pointer + 1
                ),
                vec![V::<1>(packed).reg()],
                vec![V::<1>(1).reg(), crate::S::<2>(pointer).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 5, v0",
        vec![V::<1>(1).reg()],
        vec![V::<1>(0).reg()],
    ))?;
    b.push(Instruction::new(
        "v_add_nc_u32_e32 v1, s21, v1",
        vec![V::<1>(1).reg()],
        vec![crate::S::<1>(21).reg(), V::<1>(1).reg()],
    ))?;
    for (x, suffix) in [(12, ""), (16, " offset:16")] {
        b.push(
            Instruction::new(
                format!("global_load_b128 v[{x}:{}], v1, s[6:7]{suffix}", x + 3),
                vec![V::<4>(x).reg()],
                vec![V::<1>(1).reg(), crate::S::<2>(6).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    emit_dog(b)?;
    // Preserve the selected operand order of each stream add.
    for (acc, dot, row) in [(20 + stream, 7, 0), (24 + stream, 8, 1)] {
        let (lhs, rhs) = if row == 0 { (dot, acc) } else { (acc, dot) };
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{acc}, v{lhs}, v{rhs}"),
            vec![V::<1>(acc).reg()],
            vec![V::<1>(lhs).reg(), V::<1>(rhs).reg()],
        ))?;
    }
    b.push(Instruction::new(
        "s_add_co_i32 s15, s15, 1",
        vec![crate::S::<1>(15).reg()],
        vec![crate::S::<1>(15).reg()],
    ))?;
    b.wait_all()
}

fn emit_dog(b: &mut Builder) -> Result<(), String> {
    for (packed, header, dot, order) in [
        (V::<1>(2), V::<1>(4), V::<1>(7), [0, 1, 2, 3, 4, 5, 6, 7]),
        (V::<1>(3), V::<1>(5), V::<1>(8), [1, 0, 2, 3, 4, 5, 6, 7]),
    ] {
        let weight = V::<1>(6);
        for (term, nibble) in order.into_iter().enumerate() {
            b.push(Instruction::new(
                format!("v_bfe_u32 v6, {}, {}, 4", packed.reg(), nibble * 4),
                vec![weight.reg()],
                vec![packed.reg()],
            ))?;
            b.push(Instruction::new(
                "v_cvt_f32_ubyte0_e32 v6, v6",
                vec![weight.reg()],
                vec![weight.reg()],
            ))?;
            b.push(Instruction::new(
                format!(
                    "v_fma_mix_f32 v6, {h}, v6, {h} op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                    h = header.reg()
                ),
                vec![weight.reg()],
                vec![header.reg(), weight.reg()],
            ))?;
            let x = V::<1>(12 + nibble);
            let mnemonic = if term == 0 {
                "v_mul_f32_e32"
            } else {
                "v_fmac_f32_e32"
            };
            let mut uses = vec![x.reg(), weight.reg()];
            if term != 0 {
                uses.push(dot.reg());
            }
            b.push(Instruction::new(
                format!("{mnemonic} {}, {}, v6", dot.reg(), x.reg()),
                vec![dot.reg()],
                uses,
            ))?;
        }
    }
    Ok(())
}

/// One group of each of two independent rows, before stream accumulation.
///
/// Test-only launch contract: one wave32; A contains exactly two 136-byte
/// groups, x contains 256 f32s, y contains 64 lane partials (row-major).
/// M=2 and K=256 occupy the frozen by-value slots but are not shape selectors.
/// This symbol is never returned by the production entry point.
///
/// The selected clang24 tail DAG starts row0 with x0*w0, then fmac x1*w1;
/// row1 starts x1*w1, then fmac x0*w0. The real-H2 oracle confirmed this
/// row mapping; swapping the first two terms changes output bits.
pub fn build_dog_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(20, 10)?;
    let tid = regs.v::<1>("tid", 0, Live::Whole)?;
    let addr = regs.v::<1>("address", 1, Live::Whole)?;
    let packed0 = regs.v::<1>("packed0", 2, Live::Whole)?;
    let packed1 = regs.v::<1>("packed1", 3, Live::Whole)?;
    let header0 = regs.v::<1>("header0", 4, Live::Whole)?;
    let header1 = regs.v::<1>("header1", 5, Live::Whole)?;
    regs.v::<1>("weight", 6, Live::Whole)?;
    let dot0 = regs.v::<1>("dot0", 7, Live::Whole)?;
    let dot1 = regs.v::<1>("dot1", 8, Live::Whole)?;
    let x_lo = regs.v::<4>("x_lo", 12, Live::Whole)?;
    let x_hi = regs.v::<4>("x_hi", 16, Live::Whole)?;
    let args = regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<1>("workgroup_x", 2, Live::Whole)?;
    let a_x = regs.s::<4>("a_x", 4, Live::Whole)?;
    let y = regs.s::<2>("y", 8, Live::Whole)?;
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: MODULE.into(),
            variant: "dog-g0-not-production".into(),
            arch: Arch::Gfx1201,
            symbol: G0_SYMBOL.into(),
            kernargs: KernargLayout::new(32)
                .pointer("A", 0)
                .pointer("x", 8)
                .pointer("y", 16)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value"),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    b.push(
        Instruction::new(
            "s_load_b128 s[4:7], s[0:1], 0x0",
            vec![a_x.reg()],
            vec![args.reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(
        Instruction::new(
            "s_load_b64 s[8:9], s[0:1], 0x10",
            vec![y.reg()],
            vec![args.reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    // Each subgroup of 16 lanes selects its own packed half scale/zero.
    b.push(Instruction::new(
        "v_lshrrev_b32_e32 v1, 4, v0",
        vec![addr.reg()],
        vec![tid.reg()],
    ))?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 2, v1",
        vec![addr.reg()],
        vec![addr.reg()],
    ))?;
    for (header, offset) in [(header0, 0), (header1, 136)] {
        b.push(
            Instruction::new(
                if offset == 0 {
                    format!("global_load_b32 {}, v1, s[4:5]", header.reg())
                } else {
                    format!(
                        "global_load_b32 {}, v1, s[4:5] offset:{offset}",
                        header.reg()
                    )
                },
                vec![header.reg()],
                vec![addr.reg(), crate::S::<2>(4).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 2, v0",
        vec![addr.reg()],
        vec![tid.reg()],
    ))?;
    for (packed, offset) in [(packed0, 8), (packed1, 144)] {
        b.push(
            Instruction::new(
                format!(
                    "global_load_b32 {}, v1, s[4:5] offset:{offset}",
                    packed.reg()
                ),
                vec![packed.reg()],
                vec![addr.reg(), crate::S::<2>(4).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 5, v0",
        vec![addr.reg()],
        vec![tid.reg()],
    ))?;
    for (x, offset) in [(x_lo, 0), (x_hi, 16)] {
        b.push(
            Instruction::new(
                if offset == 0 {
                    format!("global_load_b128 {}, v1, s[6:7]", x.reg())
                } else {
                    format!("global_load_b128 {}, v1, s[6:7] offset:{offset}", x.reg())
                },
                vec![x.reg()],
                vec![addr.reg(), crate::S::<2>(6).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    emit_dog(&mut b)?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v1, 2, v0",
        vec![addr.reg()],
        vec![tid.reg()],
    ))?;
    for (dot, offset) in [(dot0, 0), (dot1, 128)] {
        b.push(
            Instruction::new(
                if offset == 0 {
                    format!("global_store_b32 v1, {}, s[8:9]", dot.reg())
                } else {
                    format!("global_store_b32 v1, {}, s[8:9] offset:{offset}", dot.reg())
                },
                vec![],
                vec![addr.reg(), dot.reg(), y.reg()],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.wait_all()?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn dog_region_native_m7() {
        let emitted = build_dog_g0().expect("checked DOG region");
        let object = crate::native::assemble(&emitted.s_text, Arch::Gfx1201)
            .expect("native region code object");
        let out = std::env::temp_dir().join(format!(
            "peacemaker-isa-g0-multirow_default-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&out).unwrap();
        let prefix = out.join(format!("{MODULE}.g0"));
        std::fs::write(prefix.with_extension("g0.s"), &emitted.s_text).unwrap();
        std::fs::write(
            prefix.with_extension("g0.proof.json"),
            serde_json::to_vec_pretty(&emitted.proof).unwrap(),
        )
        .unwrap();
        let contract = serde_json::json!({
            "symbol": G0_SYMBOL,
            "counts": {
                "v_fma_mix_f32": 16,
                "v_mul_f32_e32": 2,
                "v_fmac_f32_e32": 14
            },
            "forbidden": ["scratch_load_b32", "scratch_store_b32"],
            "vgpr_max": 20,
            "sgpr_max": 10,
            "require_wave32": true,
            "require_zero_spills": true,
            "require_zero_private": true,
            "launch_dynamic_lds_bytes": 0,
            "group_segment_fixed_bytes": 0
        });
        std::fs::write(
            prefix.with_extension("g0.shape.json"),
            serde_json::to_vec_pretty(&contract).unwrap(),
        )
        .unwrap();
        let co = prefix.with_extension("g0.co");
        std::fs::write(&co, object).unwrap();
        let m7 = crate::pm_check::m7(&co, "gfx1201", G0_SYMBOL).expect("M7{}");
        println!("{m7}");
        assert_eq!(m7["obligations"], serde_json::json!({}));
        std::fs::write(
            prefix.with_extension("g0.m7.json"),
            serde_json::to_vec_pretty(&m7).unwrap(),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&out);
    }
}
