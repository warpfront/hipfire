//! Exact gfx1201 QKVZA MQ4v2 decode projection (one wave per output row).
//! Frozen HIP recipe SHA256: 24946fc118db455e19f96b0e02fd67bb6bfaf46ed4a917e23352bef6153e1ba5.
//! The four group streams and clang24's element-1-first dot fold are preserved.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{mem, op, smem, sr, srd_tail, v, vr};
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Ok(vec![build_projection()?])
}

fn build_projection() -> Result<Emitted, String> {
    use crate::kernels::common::sop;
    let mut regs = RegPlan::new(76, 32)?;
    for (base, len, name) in [
        (0, 1, "lane"),
        (1, 1, "packed_offset"),
        (2, 1, "zero"),
        (3, 1, "x_offset"),
        (4, 4, "streams"),
        (8, 8, "quad_x0_or_tail_x"),
        (16, 8, "quad_x1_or_tail_scratch"),
        (24, 8, "quad_x2"),
        (32, 8, "quad_x3"),
        (40, 8, "quad_headers_or_dots_and_dequant3"),
        (48, 4, "quad_packed_or_dequant3"),
        (52, 8, "dequant0"),
        (60, 8, "dequant1"),
        (68, 8, "dequant2"),
    ] {
        regs.add_range(name, crate::reg::Kind::V, base, len, Live::Whole)?;
    }
    for (base, len, name) in [
        (0, 2, "kernarg"),
        (4, 4, "weight_resource"),
        (8, 2, "x"),
        (10, 2, "output"),
        (12, 1, "row"),
        (13, 1, "groups"),
        (14, 1, "quads"),
        (15, 1, "tail"),
        (16, 1, "quad_index"),
        (17, 1, "weight_offset"),
        (18, 1, "total_rows"),
        (20, 4, "matrix_rows"),
        (24, 1, "k"),
        (26, 2, "row_address"),
        (28, 2, "row_product"),
    ] {
        regs.add_range(name, crate::reg::Kind::S, base, len, Live::Whole)?;
    }
    let mut args = KernargLayout::new(92);
    for (name, offset) in [("A_qkv", 0), ("A_z", 8), ("A_beta", 16), ("A_alpha", 24)] {
        args = args.pointer(name, offset);
    }
    args = args.pointer_access("x", 32, crate::plan::Access::ReadOnly);
    for (name, offset) in [("y_qkv", 40), ("y_z", 48), ("y_beta", 56), ("y_alpha", 64)] {
        args = args.pointer_access(name, offset, crate::plan::Access::WriteOnly);
    }
    for (name, offset) in [
        ("qkv_m", 72),
        ("z_m", 76),
        ("beta_m", 80),
        ("alpha_m", 84),
        ("K", 88),
    ] {
        args = args.hidden(name, offset, 4, "by_value");
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "pm_decode_fused_qkvza_hfq4g256_mq4v2".into(),
            variant: "four_stream".into(),
            arch: Arch::Gfx1201,
            symbol: "fused_qkvza_mq4g256v2".into(),
            kernargs: args,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    smem(&mut b, 20, 4, 0, 72)?;
    smem(&mut b, 24, 1, 0, 88)?;
    smem(&mut b, 8, 2, 0, 32)?;
    sop(&mut b, "s_mov_b32 s12, ttmp9", &[12], &[])?;
    sop(&mut b, "s_add_co_u32 s18, s20, s21", &[18], &[20, 21])?;
    sop(&mut b, "s_add_co_u32 s18, s18, s22", &[18], &[18, 22])?;
    sop(&mut b, "s_add_co_u32 s18, s18, s23", &[18], &[18, 23])?;
    sop(&mut b, "s_cmp_ge_u32 s12, s18", &[], &[12, 18])?;
    op(&mut b, "s_cbranch_scc1 .Lprojection_end", &[], &[])?;
    for (index, label) in [".Lroute_qkv", ".Lroute_z", ".Lroute_beta"]
        .iter()
        .enumerate()
    {
        let count = 20 + index as u8;
        sop(
            &mut b,
            format!("s_cmp_lt_u32 s12, s{count}"),
            &[],
            &[12, count],
        )?;
        op(&mut b, format!("s_cbranch_scc1 {label}"), &[], &[])?;
        sop(
            &mut b,
            format!("s_sub_co_u32 s12, s12, s{count}"),
            &[12],
            &[12, count],
        )?;
    }
    smem(&mut b, 4, 2, 0, 24)?;
    smem(&mut b, 10, 2, 0, 64)?;
    op(&mut b, "s_branch .Lrow_ready", &[], &[])?;
    for (label, input, output) in [
        (".Lroute_qkv", 0, 40),
        (".Lroute_z", 8, 48),
        (".Lroute_beta", 16, 56),
    ] {
        b.label(label)?;
        smem(&mut b, 4, 2, 0, input)?;
        smem(&mut b, 10, 2, 0, output)?;
        op(&mut b, "s_branch .Lrow_ready", &[], &[])?;
    }
    b.label(".Lrow_ready")?;
    b.wait_all()?;
    sop(&mut b, "s_lshr_b32 s13, s24, 8", &[13], &[24])?;
    sop(&mut b, "s_lshr_b32 s14, s13, 2", &[14], &[13])?;
    sop(&mut b, "s_and_b32 s15, s13, 3", &[15], &[13])?;
    sop(&mut b, "s_mul_i32 s26, s13, 0x88", &[26], &[13])?;
    sop(&mut b, "s_mov_b32 s27, 0", &[27], &[])?;
    sop(&mut b, "s_mov_b32 s28, s12", &[28], &[12])?;
    sop(&mut b, "s_mov_b32 s29, 0", &[29], &[])?;
    op(
        &mut b,
        "s_mul_u64 s[26:27], s[26:27], s[28:29]",
        &[sr(26, 2)],
        &[sr(26, 2), sr(28, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[26:27], s[4:5], s[26:27]",
        &[sr(26, 2)],
        &[sr(4, 2), sr(26, 2)],
    )?;
    op(
        &mut b,
        "s_mov_b64 s[4:5], s[26:27]",
        &[sr(4, 2)],
        &[sr(26, 2)],
    )?;
    sop(&mut b, "s_and_b32 s5, s5, 0xffff", &[5], &[5])?;
    srd_tail(&mut b, 4, None)?;
    sop(&mut b, "s_mov_b32 s16, 0", &[16], &[])?;
    sop(&mut b, "s_mov_b32 s17, 0", &[17], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_mov_b32_e32 v2, 0", &[v(2)], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[v(3)], &[v(0)])?;
    for acc in 4..8 {
        op(&mut b, format!("v_mov_b32_e32 v{acc}, 0"), &[v(acc)], &[])?;
    }
    sop(&mut b, "s_cmp_eq_u32 s14, 0", &[], &[14])?;
    op(&mut b, "s_cbranch_scc1 .Ltail_groups", &[], &[])?;
    b.wait_all()?;
    b.loop_(".Lquad_loop", |b| {
        projection_quad(b)?;
        sop(b, "s_add_co_u32 s17, s17, 0x220", &[17], &[17])?;
        crate::kernels::common::add64_imm(b, 8, 4096)?;
        sop(b, "s_add_co_u32 s16, s16, 1", &[16], &[16])?;
        sop(b, "s_cmp_lt_u32 s16, s14", &[], &[16, 14])?;
        op(b, "s_cbranch_scc1 .Lquad_loop", &[], &[])
    })?;
    b.label(".Ltail_groups")?;
    // Tail traffic is global in the incumbent, unlike quad weight-buffer loads.
    sop(&mut b, "s_mov_b32 s28, s17", &[28], &[17])?;
    sop(&mut b, "s_mov_b32 s29, 0", &[29], &[])?;
    op(
        &mut b,
        "s_add_nc_u64 s[26:27], s[26:27], s[28:29]",
        &[sr(26, 2)],
        &[sr(26, 2), sr(28, 2)],
    )?;
    for stream in 0..3 {
        sop(
            &mut b,
            format!("s_cmp_lt_u32 s15, {}", stream + 1),
            &[],
            &[15],
        )?;
        op(&mut b, "s_cbranch_scc1 .Lfold_streams", &[], &[])?;
        projection_group(&mut b, stream)?;
    }
    b.label(".Lfold_streams")?;
    projection_reduce(&mut b)?;
    op(&mut b, "v_cmpx_eq_u32_e32 0, v0", &[], &[v(0)])?;
    op(&mut b, "s_cbranch_execz .Lprojection_end", &[], &[])?;
    sop(&mut b, "s_mov_b32 s28, s12", &[28], &[12])?;
    sop(&mut b, "s_mov_b32 s29, 0", &[29], &[])?;
    op(
        &mut b,
        "s_lshl_b64 s[28:29], s[28:29], 2",
        &[sr(28, 2)],
        &[sr(28, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[28:29], s[10:11], s[28:29]",
        &[sr(28, 2)],
        &[sr(10, 2), sr(28, 2)],
    )?;
    mem(
        &mut b,
        "global_store_b32 v2, v4, s[28:29]",
        &[],
        &[v(2), v(4), sr(28, 2)],
        MemoryClass::VmemStore,
    )?;
    b.label(".Lprojection_end")?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

/// Sixteen loads issue before any consumer. Nine checked progressive waits
/// follow the incumbent producer DAG; headers and packed words become scratch
/// only after their final dequant use. All four element-1-first folds are intact.
fn projection_quad(b: &mut Builder) -> Result<(), String> {
    use crate::{kernels::common::s, ledger::Counter};
    b.clause(|b| {
        for stream in 0..4u8 {
            let header = 40 + stream * 2;
            let offset = u32::from(stream) * 136;
            let immediate = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
            mem(b, format!("buffer_load_b64 v[{header}:{}], v2, s[4:7], s17 offen{immediate} scope:SCOPE_DEV", header+1),
                &[vr(header,2)], &[v(2),sr(4,4),s(17)], MemoryClass::VmemLoad)?;
        }
        for stream in 0..4u8 {
            let packed = 48 + stream;
            mem(b, format!("buffer_load_b32 v{packed}, v1, s[4:7], s17 offen offset:{} scope:SCOPE_DEV", u32::from(stream)*136+8),
                &[v(packed)], &[v(1),sr(4,4),s(17)], MemoryClass::VmemLoad)?;
        }
        Ok(())
    })?;
    b.clause(|b| {
        for half in 0..2u8 {
            for stream in 0..4u8 {
                let dst = 8 + stream * 8 + half * 4;
                let offset = u32::from(stream) * 1024 + u32::from(half) * 16;
                let immediate = if offset == 0 {
                    String::new()
                } else {
                    format!(" offset:{offset}")
                };
                mem(
                    b,
                    format!(
                        "global_load_b128 v[{dst}:{}], v3, s[8:9]{immediate}",
                        dst + 3
                    ),
                    &[vr(dst, 4)],
                    &[v(3), sr(8, 2)],
                    MemoryClass::VmemLoad,
                )?;
            }
        }
        Ok(())
    })?;
    // Explicit producer waits preserve BUFFER/GLOBAL overlap instead of the
    // automatic ledger's conservative mixed-family drain.
    op(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    for pair in 0..2u8 {
        b.wait(Counter::Load, 14 - pair * 2)?;
        for stream in pair * 2..pair * 2 + 2 {
            let dst = 40 + stream;
            let header = 40 + stream * 2;
            op(
                b,
                format!(
                    "v_cndmask_b32_e64 v{dst}, v{}, v{header}, vcc_lo",
                    header + 1
                ),
                &[v(dst)],
                &[v(header + 1), v(header)],
            )?;
        }
    }
    b.wait(Counter::Load, 11)?;
    quad_extract(b, 0)?;
    b.wait(Counter::Load, 10)?;
    quad_extract(b, 1)?;
    quad_dequant_pair(b, 0)?;
    b.wait(Counter::Load, 9)?;
    quad_extract(b, 2)?;
    b.wait(Counter::Load, 6)?;
    quad_dot_term(b, 0, 1)?;
    quad_extract(b, 3)?;
    // Stagger adjacent streams by one term to satisfy the checked VOPD banks.
    quad_dual_term(b, 0, 0, 1)?;
    quad_delay(b, 1)?;
    quad_dual_term(b, 0, 2, 0)?;
    quad_delay(b, 1)?;
    quad_dual_term(b, 0, 3, 2)?;
    quad_dequant_pair(b, 2)?;
    b.wait(Counter::Load, 2)?;
    quad_dot_term(b, 2, 1)?;
    quad_dual_term(b, 0, 4, 3)?;
    quad_delay(b, 2)?;
    quad_dual_term(b, 2, 0, 1)?;
    for (element01, element23) in [(5, Some(2)), (6, Some(3)), (7, None)] {
        quad_delay(b, 2)?;
        quad_dual_term(b, 0, element01, element01 - 1)?;
        if let Some(element) = element23 {
            quad_delay(b, 2)?;
            quad_dual_term(b, 2, element, if element == 2 { 0 } else { 2 })?;
        }
    }
    quad_dot_term(b, 1, 7)?;
    b.wait(Counter::Load, 1)?;
    quad_delay(b, 3)?;
    quad_dual_term(b, 2, 4, 3)?;
    b.wait(Counter::Load, 0)?;
    for element in 5..8 {
        quad_delay(b, 1)?;
        quad_dual_term(b, 2, element, element - 1)?;
    }
    for stream in 0..2 {
        let acc = 4 + stream;
        let dot = 40 + stream;
        op(
            b,
            format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
            &[v(acc)],
            &[v(acc), v(dot)],
        )?;
    }
    quad_dot_term(b, 3, 7)?;
    for stream in 2..4 {
        let acc = 4 + stream;
        let dot = 40 + stream;
        op(
            b,
            format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
            &[v(acc)],
            &[v(acc), v(dot)],
        )?;
    }
    b.wait_all()
}

fn quad_level(stream: u8, element: u8) -> u8 {
    if stream == 3 {
        44 + element
    } else {
        52 + stream * 8 + element
    }
}

fn quad_extract(b: &mut Builder, stream: u8) -> Result<(), String> {
    let packed = 48 + stream;
    // Group 3's final nibble overwrites its packed word only after all reads.
    for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
        let dst = quad_level(stream, element);
        op(
            b,
            format!("v_bfe_u32 v{dst}, v{packed}, {}, 4", element * 4),
            &[v(dst)],
            &[v(packed)],
        )?;
    }
    Ok(())
}

fn quad_dequant_pair(b: &mut Builder, first: u8) -> Result<(), String> {
    for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
        for stream in first..first + 2 {
            let dst = quad_level(stream, element);
            op(
                b,
                format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
                &[v(dst)],
                &[v(dst)],
            )?;
        }
    }
    for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
        for stream in first..first + 2 {
            let dst = quad_level(stream, element);
            let header = 40 + stream;
            op(b, format!("v_fma_mix_f32 v{dst}, v{header}, v{dst}, v{header} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[v(dst)], &[v(header),v(dst)])?;
        }
    }
    Ok(())
}

fn quad_dot_term(b: &mut Builder, stream: u8, element: u8) -> Result<(), String> {
    let dot = 40 + stream;
    let weight = quad_level(stream, element);
    let x = 8 + stream * 8 + element;
    if element == 1 {
        op(
            b,
            format!("v_mul_f32_e32 v{dot}, v{weight}, v{x}"),
            &[v(dot)],
            &[v(weight), v(x)],
        )
    } else {
        op(
            b,
            format!("v_fmac_f32_e32 v{dot}, v{weight}, v{x}"),
            &[v(dot)],
            &[v(dot), v(weight), v(x)],
        )
    }
}

fn quad_dual_term(b: &mut Builder, first: u8, x_element: u8, y_element: u8) -> Result<(), String> {
    use crate::vopd::{Operand, VopdF32, VopdOp};
    let term = |stream: u8, element: u8| VopdOp {
        op: if element == 1 {
            VopdF32::Mul
        } else {
            VopdF32::Fmac
        },
        dst: 40 + stream,
        src0: Operand::V(quad_level(stream, element)),
        src1: 8 + stream * 8 + element,
    };
    b.vopd(term(first, x_element), term(first + 1, y_element))
}

fn quad_delay(b: &mut Builder, dependency: u8) -> Result<(), String> {
    // Proven producer hints yield between dependent dual packets; they do not
    // alter the per-stream RN/FMA sequence or add a new arithmetic operation.
    op(
        b,
        format!("s_delay_alu instid0(VALU_DEP_{dependency})"),
        &[],
        &[],
    )
}

fn projection_group(b: &mut Builder, stream: u8) -> Result<(), String> {
    let group = u32::from(stream) * 136;
    let group_offset = if group == 0 {
        String::new()
    } else {
        format!(" offset:{group}")
    };
    mem(
        b,
        format!("global_load_b64 v[16:17], v2, s[26:27]{group_offset}"),
        &[vr(16, 2)],
        &[v(2), sr(26, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!("global_load_b32 v18, v1, s[26:27] offset:{}", group + 8),
        &[v(18)],
        &[v(1), sr(26, 2)],
        MemoryClass::VmemLoad,
    )?;
    for (dst, offset) in [
        (8, u32::from(stream) * 1024),
        (12, u32::from(stream) * 1024 + 16),
    ] {
        let immediate = if offset == 0 {
            String::new()
        } else {
            format!(" offset:{offset}")
        };
        mem(
            b,
            format!(
                "global_load_b128 v[{dst}:{}], v3, s[8:9]{immediate}",
                dst + 3
            ),
            &[vr(dst, 4)],
            &[v(3), sr(8, 2)],
            MemoryClass::VmemLoad,
        )?;
    }
    op(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    op(
        b,
        "v_cndmask_b32_e64 v16, v17, v16, vcc_lo",
        &[v(16)],
        &[v(17), v(16)],
    )?;
    for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
        op(
            b,
            format!("v_bfe_u32 v19, v18, {}, 4", element * 4),
            &[v(19)],
            &[v(18)],
        )?;
        op(b, "v_cvt_f32_ubyte0_e32 v19, v19", &[v(19)], &[v(19)])?;
        op(
            b,
            "v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]",
            &[v(20)],
            &[v(16), v(19)],
        )?;
        let x = 8 + element;
        let mnemonic = if element == 1 {
            "v_mul_f32_e32"
        } else {
            "v_fmac_f32_e32"
        };
        let mut uses = vec![v(20), v(x)];
        if element != 1 {
            uses.push(v(21));
        }
        op(b, format!("{mnemonic} v21, v20, v{x}"), &[v(21)], &uses)?;
    }
    let acc = 4 + stream;
    op(
        b,
        format!("v_add_f32_e32 v{acc}, v{acc}, v21"),
        &[v(acc)],
        &[v(acc), v(21)],
    )?;
    b.wait_all()
}

fn projection_reduce(b: &mut Builder) -> Result<(), String> {
    op(b, "v_add_f32_e32 v4, v4, v5", &[v(4)], &[v(4), v(5)])?;
    op(b, "v_add_f32_e32 v22, v6, v7", &[v(22)], &[v(6), v(7)])?;
    op(b, "v_add_f32_e32 v4, v4, v22", &[v(4)], &[v(4), v(22)])?;
    b.ds_crosslane(
        Instruction::new(
            "ds_swizzle_b32 v21, v4 offset:swizzle(BITMASK_PERM,\"1pppp\")",
            vec![v(21)],
            vec![v(4)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(crate::ledger::Counter::Ds, 0)?;
    op(b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    for offset in [8, 4, 2, 1] {
        op(
            b,
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
            &[],
            &[v(0)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v22, 0, {offset}, vcc_lo"),
            &[v(22)],
            &[],
        )?;
        op(
            b,
            "v_add_lshl_u32 v22, v22, v0, 2",
            &[v(22)],
            &[v(22), v(0)],
        )?;
        b.ds_crosslane(
            Instruction::new(
                "ds_bpermute_b32 v21, v22, v4",
                vec![v(21)],
                vec![v(22), v(4)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        op(b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    }
    Ok(())
}

/// Four contiguous groups, one activation quad, and 32 reduced lane outputs.
/// The symbol and ABI deliberately differ from the production projection.
#[cfg(all(test, feature = "toolchain"))]
fn build_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(32, 16)?;
    regs.v::<1>("lane", 0, Live::Whole)?;
    regs.v::<1>("weight_offset", 1, Live::Whole)?;
    regs.v::<1>("header_offset", 2, Live::Whole)?;
    regs.v::<1>("activation_offset", 3, Live::Whole)?;
    regs.v::<4>("streams", 4, Live::Whole)?;
    regs.v::<4>("activation_lo", 8, Live::Whole)?;
    regs.v::<4>("activation_hi", 12, Live::Whole)?;
    regs.v::<2>("headers", 16, Live::Whole)?;
    regs.v::<1>("packed", 18, Live::Whole)?;
    regs.v::<1>("level", 19, Live::Whole)?;
    regs.v::<1>("weight", 20, Live::Whole)?;
    regs.v::<1>("dot", 21, Live::Whole)?;
    regs.v::<1>("pair", 22, Live::Whole)?;
    regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<4>("weights", 4, Live::Whole)?;
    regs.s::<2>("activation", 8, Live::Whole)?;
    regs.s::<2>("output", 10, Live::Whole)?;
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "pm_decode_qkvza_g0".into(),
            variant: "quad_region".into(),
            arch: Arch::Gfx1201,
            symbol: "pm_decode_qkvza_g0".into(),
            kernargs: KernargLayout::new(24)
                .pointer("weights", 0)
                .pointer("activation", 8)
                .pointer("lane_output", 16),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    smem(&mut b, 4, 2, 0, 0)?;
    smem(&mut b, 8, 2, 0, 8)?;
    smem(&mut b, 10, 2, 0, 16)?;
    srd_tail(&mut b, 4, None)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_mov_b32_e32 v2, 0", &[v(2)], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[v(3)], &[v(0)])?;
    for stream in 0..4u8 {
        let acc = 4 + stream;
        op(&mut b, format!("v_mov_b32_e32 v{acc}, 0"), &[v(acc)], &[])?;
        let group = u32::from(stream) * 136;
        mem(
            &mut b,
            format!(
                "buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:{group}"
            ),
            &[vr(16, 2)],
            &[v(2), sr(4, 4)],
            MemoryClass::VmemLoad,
        )?;
        mem(
            &mut b,
            format!(
                "buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:{}",
                group + 8
            ),
            &[v(18)],
            &[v(1), sr(4, 4)],
            MemoryClass::VmemLoad,
        )?;
        for (dst, offset) in [
            (8, u32::from(stream) * 1024),
            (12, u32::from(stream) * 1024 + 16),
        ] {
            mem(
                &mut b,
                format!(
                    "global_load_b128 v[{dst}:{}], v3, s[8:9] offset:{offset}",
                    dst + 3
                ),
                &[vr(dst, 4)],
                &[v(3), sr(8, 2)],
                MemoryClass::VmemLoad,
            )?;
        }
        op(&mut b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
        op(
            &mut b,
            "v_cndmask_b32_e64 v16, v17, v16, vcc_lo",
            &[v(16)],
            &[v(17), v(16)],
        )?;
        // LLVM24 starts with element 1, then fuses element 0, then 2..7.
        for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
            op(
                &mut b,
                format!("v_bfe_u32 v19, v18, {}, 4", element * 4),
                &[v(19)],
                &[v(18)],
            )?;
            op(&mut b, "v_cvt_f32_ubyte0_e32 v19, v19", &[v(19)], &[v(19)])?;
            op(
                &mut b,
                "v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                &[v(20)],
                &[v(16), v(19)],
            )?;
            let x = 8 + element;
            if element == 1 {
                op(
                    &mut b,
                    format!("v_mul_f32_e32 v21, v20, v{x}"),
                    &[v(21)],
                    &[v(20), v(x)],
                )?;
            } else {
                op(
                    &mut b,
                    format!("v_fmac_f32_e32 v21, v20, v{x}"),
                    &[v(21)],
                    &[v(21), v(20), v(x)],
                )?;
            }
        }
        op(
            &mut b,
            format!("v_add_f32_e32 v{acc}, v{acc}, v21"),
            &[v(acc)],
            &[v(acc), v(21)],
        )?;
    }
    op(&mut b, "v_add_f32_e32 v4, v4, v5", &[v(4)], &[v(4), v(5)])?;
    op(&mut b, "v_add_f32_e32 v22, v6, v7", &[v(22)], &[v(6), v(7)])?;
    op(&mut b, "v_add_f32_e32 v4, v4, v22", &[v(4)], &[v(4), v(22)])?;
    // clang24's shfl-down16 sets lane bit 4; out-of-range lanes read themselves.
    b.ds_crosslane(
        Instruction::new(
            "ds_swizzle_b32 v21, v4 offset:swizzle(BITMASK_PERM,\"1pppp\")",
            vec![v(21)],
            vec![v(4)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(crate::ledger::Counter::Ds, 0)?;
    op(&mut b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    for offset in [8, 4, 2, 1] {
        op(
            &mut b,
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
            &[],
            &[v(0)],
        )?;
        op(
            &mut b,
            format!("v_cndmask_b32_e64 v22, 0, {offset}, vcc_lo"),
            &[v(22)],
            &[],
        )?;
        op(
            &mut b,
            "v_add_lshl_u32 v22, v22, v0, 2",
            &[v(22)],
            &[v(22), v(0)],
        )?;
        b.ds_crosslane(
            Instruction::new(
                "ds_bpermute_b32 v21, v22, v4",
                vec![v(21)],
                vec![v(22), v(4)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        op(&mut b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    }
    mem(
        &mut b,
        "global_store_b32 v1, v4, s[10:11]",
        &[],
        &[v(1), v(4), sr(10, 2)],
        MemoryClass::VmemStore,
    )?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    #[test]
    fn production_contract() -> Result<(), String> {
        let emitted = super::build_gfx1201()?;
        assert_eq!(emitted.len(), 1);
        let text = &emitted[0].s_text;
        for field in [
            ".name: fused_qkvza_mq4g256v2",
            ".kernarg_segment_size: 92",
            ".kernarg_segment_align: 8",
            ".max_flat_workgroup_size: 32",
            ".group_segment_fixed_size: 0",
            ".private_segment_fixed_size: 0",
            ".wavefront_size: 32",
            ".amdhsa_float_round_mode_32 0",
            ".amdhsa_float_denorm_mode_32 3",
        ] {
            assert!(
                text.contains(field),
                "missing frozen contract field {field}"
            );
        }
        let quad = text
            .split(".Lquad_loop:")
            .nth(1)
            .expect("quad label")
            .split(".Ltail_groups:")
            .next()
            .expect("tail label");
        let waits: Vec<_> = quad
            .lines()
            .filter_map(|line| {
                line.trim().strip_prefix("s_wait_loadcnt ").map(|count| {
                    u8::from_str_radix(
                        count.strip_prefix("0x").unwrap_or(count),
                        if count.starts_with("0x") { 16 } else { 10 },
                    )
                    .expect("load count")
                })
            })
            .collect();
        assert_eq!(waits, [14, 12, 11, 10, 9, 6, 2, 1, 0]);
        assert_eq!(quad.matches("s_clause 0x7").count(), 2);
        assert_eq!(quad.matches(" :: ").count(), 14);
        assert_eq!(quad.matches("s_delay_alu").count(), 12);
        assert!(text.contains(".vgpr_count: 76"));
        crate::native::assemble(text, crate::Arch::Gfx1201)?;
        Ok(())
    }
    #[test]
    fn g0_quad_m7() -> Result<(), String> {
        let emitted = super::build_g0()?;
        let root =
            std::env::temp_dir().join(format!("peacemaker-isa-g0-qkvza-{}", std::process::id()));
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let base = root.join("fused_qkvza_hfq4g256_mq4v2.g0");
        std::fs::write(base.with_extension("g0.s"), &emitted.s_text).map_err(|e| e.to_string())?;
        let bytes = crate::native::assemble(&emitted.s_text, crate::Arch::Gfx1201)?;
        let object = base.with_extension("g0.co");
        std::fs::write(&object, bytes).map_err(|e| e.to_string())?;
        let evidence = crate::pm_check::m7(&object, "gfx1201", "pm_decode_qkvza_g0")?;
        println!("QKVZA G0 M7: {evidence}");
        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }
}
