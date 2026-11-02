//! Exact gfx1201 gate/up projection arithmetic, frozen to clang24 MQ4v2.
//! The 52-byte ABI, 32-thread launcher and zero LDS are inherited unchanged.
//! Independent group streams own g%4; each group's ordered eight-term DAG
//! is followed by the pairwise fold and shuffle-down 16,8,4,2,1. Gate and up
//! remain separate projections: this symbol performs no SiLU.
use crate::insn::MemoryClass;
use crate::kernels::common::{mem, op, s, smem, sr, v, vr};
use crate::plan::Access;
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

const SYMBOL: &str = "fused_gate_up_mq4g256v2";
#[cfg(all(test, feature = "toolchain"))]
const G0_SYMBOL: &str = "fused_gate_up_mq4g256v2_g0";

fn builder(symbol: &str) -> Result<Builder, String> {
    let spec = KernelSpec {
        kernel_id: "fused_gate_up_hfq4g256_mq4v2".into(),
        variant: "gfx1201".into(),
        arch: Arch::Gfx1201,
        symbol: symbol.into(),
        kernargs: KernargLayout::new(52)
            .pointer_access("A_gate", 0, Access::ReadOnly)
            .pointer_access("A_up", 8, Access::ReadOnly)
            .pointer_access("x", 16, Access::ReadOnly)
            .pointer_access("y_gate", 24, Access::WriteOnly)
            .pointer_access("y_up", 32, Access::WriteOnly)
            .hidden("gate_m", 40, 4, "by_value")
            .hidden("up_m", 44, 4, "by_value")
            .hidden("K", 48, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut regs = RegPlan::new(if symbol == SYMBOL { 64 } else { 32 }, 32)?;
    for (name, base) in [
        ("lane", 0),
        ("scratch", 4),
        ("x_lo", 8),
        ("x_hi", 12),
        ("levels_lo", 16),
        ("levels_hi", 20),
        ("accumulators", 24),
        ("addresses", 28),
    ] {
        regs.v::<4>(name, base, Live::Whole)?;
    }
    if symbol == SYMBOL {
        for (name, base) in [
            ("quad_x0", 32),
            ("quad_x1", 40),
            ("quad_x2", 48),
            ("quad_x3", 56),
        ] {
            regs.v::<8>(name, base, Live::Whole)?;
        }
    }
    for (name, base) in [
        ("abi", 0),
        ("pointers0", 4),
        ("pointers1", 8),
        ("resource", 12),
        ("shape", 16),
        ("loop", 20),
        ("masks", 24),
        ("temps", 28),
    ] {
        regs.s::<4>(name, base, Live::Whole)?;
    }
    Ok(Builder::new(spec, regs))
}

/// One group's eight-term DAG. clang24 starts with term 1, folds term 0
/// into it, then terms 2..7; the stream accumulator is added only last.
fn group_arithmetic(b: &mut Builder, accumulator: u8) -> Result<(), String> {
    for nibble in 0..8u8 {
        let dst = 16 + nibble;
        op(
            b,
            format!("v_bfe_u32 v{dst}, v4, {}, 4", nibble * 4),
            &[v(dst)],
            &[v(4)],
        )?;
        op(
            b,
            format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
            &[v(dst)],
            &[v(dst)],
        )?;
        op(
            b,
            format!("v_fma_mix_f32 v{dst}, v5, v{dst}, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
            &[v(dst)],
            &[v(5), v(dst)],
        )?;
    }
    op(b, "v_mul_f32_e32 v6, v9, v17", &[v(6)], &[v(9), v(17)])?;
    for nibble in [0u8, 2, 3, 4, 5, 6, 7] {
        op(
            b,
            format!("v_fmac_f32_e32 v6, v{}, v{}", 8 + nibble, 16 + nibble),
            &[v(6)],
            &[v(6), v(8 + nibble), v(16 + nibble)],
        )?;
    }
    op(
        b,
        format!("v_add_f32_e32 v{accumulator}, v{accumulator}, v6"),
        &[v(accumulator)],
        &[v(accumulator), v(6)],
    )
}

#[cfg(all(test, feature = "toolchain"))]
/// G0 only: one 136-byte group and 256 input floats, producing all 32
/// post-reduction lane values in y_gate. Lane zero is the one-group dot.
fn build_region() -> Result<Emitted, String> {
    let mut b = builder(G0_SYMBOL)?;
    smem(&mut b, 4, 2, 0, 0)?;
    smem(&mut b, 8, 4, 0, 16)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_lshlrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?;
    // Header is independent of the input lane: load both half-wave headers.
    op(&mut b, "v_mov_b32_e32 v3, 0", &[v(3)], &[])?;
    mem(
        &mut b,
        "global_load_b64 v[6:7], v3, s[4:5]",
        &[vr(6, 2)],
        &[v(3), sr(4, 2)],
        MemoryClass::VmemLoad,
    )?;
    op(&mut b, "v_cmp_lt_u32_e64 s24, v0, 16", &[s(24)], &[v(0)])?;
    op(
        &mut b,
        "v_cndmask_b32_e64 v5, v7, v6, s24",
        &[v(5)],
        &[v(7), v(6), s(24)],
    )?;
    mem(
        &mut b,
        "global_load_b32 v4, v1, s[4:5] offset:8",
        &[v(4)],
        &[v(1), sr(4, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "global_load_b128 v[8:11], v2, s[8:9]",
        &[vr(8, 4)],
        &[v(2), sr(8, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "global_load_b128 v[12:15], v2, s[8:9] offset:16",
        &[vr(12, 4)],
        &[v(2), sr(8, 2)],
        MemoryClass::VmemLoad,
    )?;
    op(&mut b, "v_mov_b32_e32 v24, 0", &[v(24)], &[])?;
    group_arithmetic(&mut b, 24)?;
    for register in 25..28 {
        op(
            &mut b,
            format!("v_mov_b32_e32 v{register}, 0"),
            &[v(register)],
            &[],
        )?;
    }
    op(
        &mut b,
        "v_add_f32_e32 v24, v24, v25",
        &[v(24)],
        &[v(24), v(25)],
    )?;
    op(
        &mut b,
        "v_add_f32_e32 v26, v26, v27",
        &[v(26)],
        &[v(26), v(27)],
    )?;
    op(
        &mut b,
        "v_add_f32_e32 v24, v24, v26",
        &[v(24)],
        &[v(24), v(26)],
    )?;
    reduce(&mut b)?;
    mem(
        &mut b,
        "global_store_b32 v1, v24, s[10:11]",
        &[],
        &[v(1), v(24), sr(10, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// The incumbent shuffle-down keeps the source lane itself when it would
/// cross the wave boundary; in particular the swizzle forces bit 4 to one.
fn reduce(b: &mut Builder) -> Result<(), String> {
    use crate::insn::Instruction;
    use crate::ledger::Counter;
    b.ds_crosslane(
        Instruction::new(
            "ds_swizzle_b32 v6, v24 offset:swizzle(BITMASK_PERM,\"1pppp\")",
            vec![v(6)],
            vec![v(24)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(Counter::Ds, 0)?;
    op(b, "v_add_f32_e32 v24, v24, v6", &[v(24)], &[v(24), v(6)])?;
    for offset in [8, 4, 2, 1] {
        op(
            b,
            format!("v_cmp_lt_u32_e64 s24, v0, {}", 32 - offset),
            &[s(24)],
            &[v(0)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v3, 0, {offset}, s24"),
            &[v(3)],
            &[s(24)],
        )?;
        op(b, "v_add_lshl_u32 v3, v3, v0, 2", &[v(3)], &[v(3), v(0)])?;
        b.ds_crosslane(
            Instruction::new("ds_bpermute_b32 v6, v3, v24", vec![v(6)], vec![v(3), v(24)])
                .memory(MemoryClass::DsLoad),
        )?;
        b.wait(Counter::Ds, 0)?;
        op(b, "v_add_f32_e32 v24, v24, v6", &[v(24)], &[v(24), v(6)])?;
    }
    Ok(())
}

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    let mut b = builder(SYMBOL)?;
    smem(&mut b, 4, 4, 0, 0)?;
    smem(&mut b, 8, 4, 0, 16)?;
    smem(&mut b, 16, 4, 0, 32)?;
    smem(&mut b, 28, 1, 0, 48)?;
    // gfx12's workgroup ID x is supplied in ttmp9, not in the user SGPRs.
    op(&mut b, "s_cmp_lt_i32 ttmp9, s18", &[], &[s(18)])?;
    op(&mut b, "s_cselect_b32 s20, 0, s18", &[s(20)], &[s(18)])?;
    op(&mut b, "s_cselect_b32 s4, s4, s6", &[s(4)], &[s(4), s(6)])?;
    op(&mut b, "s_cselect_b32 s5, s5, s7", &[s(5)], &[s(5), s(7)])?;
    op(
        &mut b,
        "s_cselect_b32 s10, s10, s16",
        &[s(10)],
        &[s(10), s(16)],
    )?;
    op(
        &mut b,
        "s_cselect_b32 s11, s11, s17",
        &[s(11)],
        &[s(11), s(17)],
    )?;
    op(&mut b, "s_sub_co_i32 s30, ttmp9, s20", &[s(30)], &[s(20)])?;
    op(&mut b, "s_ashr_i32 s31, s30, 31", &[s(31)], &[s(30)])?;
    // K is host-admitted as a positive multiple of 256.
    op(&mut b, "s_ashr_i32 s21, s28, 8", &[s(21)], &[s(28)])?;
    op(&mut b, "s_mul_i32 s22, s21, 0x88", &[s(22)], &[s(21)])?;
    op(&mut b, "s_ashr_i32 s23, s22, 31", &[s(23)], &[s(22)])?;
    op(
        &mut b,
        "s_mul_u64 s[22:23], s[22:23], s[30:31]",
        &[sr(22, 2)],
        &[sr(22, 2), sr(30, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[12:13], s[4:5], s[22:23]",
        &[sr(12, 2)],
        &[sr(4, 2), sr(22, 2)],
    )?;
    op(&mut b, "s_and_b32 s13, s13, 0xffff", &[s(13)], &[s(13)])?;
    op(&mut b, "s_mov_b32 s14, -1", &[s(14)], &[])?;
    op(&mut b, "s_mov_b32 s15, 0x31004000", &[s(15)], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_lshlrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?;
    for accumulator in 24..28 {
        op(
            &mut b,
            format!("v_mov_b32_e32 v{accumulator}, 0"),
            &[v(accumulator)],
            &[],
        )?;
    }
    for register in [20, 22, 23] {
        op(
            &mut b,
            format!("s_mov_b32 s{register}, 0"),
            &[s(register)],
            &[],
        )?;
    }
    op(&mut b, "s_lshr_b32 s28, s21, 2", &[s(28)], &[s(21)])?;
    op(&mut b, "s_cmp_eq_u32 s28, 0", &[], &[s(28)])?;
    op(&mut b, "s_cbranch_scc1 .Lgate_up_tail", &[], &[])?;
    // A whole quad's independent loads and dot chains overlap. Each group's
    // eight-term DAG and its stream accumulator remain separate.
    b.loop_(".Lgate_up_quads", |b| {
        quad(b)?;
        b.wait_all()?;
        op(b, "s_add_co_i32 s20, s20, 4", &[s(20)], &[s(20)])?;
        op(b, "s_add_co_i32 s22, s22, 0x220", &[s(22)], &[s(22)])?;
        op(b, "s_add_co_i32 s23, s23, 0x1000", &[s(23)], &[s(23)])?;
        op(b, "s_add_co_i32 s28, s28, -1", &[s(28)], &[s(28)])?;
        op(b, "s_cmp_eq_u32 s28, 0", &[], &[s(28)])?;
        op(b, "s_cbranch_scc0 .Lgate_up_quads", &[], &[])
    })?;
    b.label(".Lgate_up_tail")?;
    for accumulator in 24..27 {
        op(&mut b, "s_cmp_lt_u32 s20, s21", &[], &[s(20), s(21)])?;
        op(&mut b, "s_cbranch_scc0 .Lgate_up_fold", &[], &[])?;
        load_group(&mut b)?;
        group_arithmetic(&mut b, accumulator)?;
        b.wait_all()?;
        op(&mut b, "s_add_co_i32 s20, s20, 1", &[s(20)], &[s(20)])?;
        op(&mut b, "s_add_co_i32 s22, s22, 0x88", &[s(22)], &[s(22)])?;
        op(&mut b, "s_add_co_i32 s23, s23, 0x400", &[s(23)], &[s(23)])?;
    }
    b.label(".Lgate_up_fold")?;
    op(
        &mut b,
        "v_add_f32_e32 v24, v24, v25",
        &[v(24)],
        &[v(24), v(25)],
    )?;
    // Incumbent's second pair is acc3 + acc2 (the original operand order).
    op(
        &mut b,
        "v_add_f32_e32 v26, v27, v26",
        &[v(26)],
        &[v(27), v(26)],
    )?;
    op(
        &mut b,
        "v_add_f32_e32 v24, v24, v26",
        &[v(24)],
        &[v(24), v(26)],
    )?;
    reduce(&mut b)?;
    op(&mut b, "v_cmpx_eq_u32_e32 0, v0", &[], &[v(0)])?;
    op(&mut b, "s_cbranch_execz .Lgate_up_end", &[], &[])?;
    op(
        &mut b,
        "s_lshl_b64 s[30:31], s[30:31], 2",
        &[sr(30, 2)],
        &[sr(30, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[10:11], s[10:11], s[30:31]",
        &[sr(10, 2)],
        &[sr(10, 2), sr(30, 2)],
    )?;
    op(&mut b, "v_mov_b32_e32 v3, 0", &[v(3)], &[])?;
    mem(
        &mut b,
        "global_store_b32 v3, v24, s[10:11]",
        &[],
        &[v(3), v(24), sr(10, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    b.label(".Lgate_up_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    Ok(vec![b.finish()?])
}

/// Eighteen independent loads: six header transfers, four packed words,
/// eight x vectors. The eight newest loads are x, so Loadcnt(8) retires the
/// weight batch without draining x. Subsequent waits are ledger-derived.
/// Only four dequant temporaries are live; this keeps the full twin at 64 VGPRs.
fn quad(b: &mut Builder) -> Result<(), String> {
    use crate::ledger::Counter;
    op(b, "v_mov_b32_e32 v3, s22", &[v(3)], &[s(22)])?;
    op(b, "v_add_nc_u32_e32 v28, s22, v1", &[v(28)], &[s(22), v(1)])?;
    op(b, "v_add_nc_u32_e32 v29, s23, v2", &[v(29)], &[s(23), v(2)])?;
    for group in 0..4u8 {
        let header = 8 + group * 2;
        let offset = u32::from(group) * 136;
        if group < 2 {
            for half in 0..2u8 {
                let dst = header + half;
                let off = offset + u32::from(half) * 4;
                let suffix = if off == 0 {
                    String::new()
                } else {
                    format!(" offset:{off}")
                };
                mem(
                    b,
                    format!(
                        "buffer_load_b32 v{dst}, v3, s[12:15], null offen{suffix} scope:SCOPE_DEV"
                    ),
                    &[v(dst)],
                    &[v(3), sr(12, 4)],
                    MemoryClass::VmemLoad,
                )?;
            }
        } else {
            mem(b, format!("buffer_load_b64 v[{header}:{}], v3, s[12:15], null offen offset:{offset} scope:SCOPE_DEV", header + 1),
                &[vr(header, 2)], &[v(3), sr(12, 4)], MemoryClass::VmemLoad)?;
        }
        mem(
            b,
            format!(
                "buffer_load_b32 v{}, v28, s[12:15], null offen offset:{} scope:SCOPE_DEV",
                16 + group,
                offset + 8
            ),
            &[v(16 + group)],
            &[v(28), sr(12, 4)],
            MemoryClass::VmemLoad,
        )?;
    }
    for group in 0..4u8 {
        for half in 0..2u8 {
            let dst = 32 + group * 8 + half * 4;
            let off = u32::from(group) * 1024 + u32::from(half) * 16;
            let suffix = if off == 0 {
                String::new()
            } else {
                format!(" offset:{off}")
            };
            mem(
                b,
                format!("global_load_b128 v[{dst}:{}], v29, s[8:9]{suffix}", dst + 3),
                &[vr(dst, 4)],
                &[v(29), sr(8, 2)],
                MemoryClass::VmemLoad,
            )?;
        }
    }
    op(b, "v_cmp_lt_u32_e64 s24, v0, 16", &[s(24)], &[v(0)])?;
    op(
        b,
        "v_cndmask_b32_e64 v8, v9, v8, s24",
        &[v(8)],
        &[v(9), v(8), s(24)],
    )?;
    b.wait(Counter::Load, 8)?;
    for header in [10, 12, 14] {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{header}, v{}, v{header}, s24",
                header + 1
            ),
            &[v(header)],
            &[v(header + 1), v(header), s(24)],
        )?;
    }
    for nibble in [1u8, 0, 2, 3, 4, 5, 6, 7] {
        for group in 0..4u8 {
            let dst = 4 + group;
            op(
                b,
                format!("v_bfe_u32 v{dst}, v{}, {}, 4", 16 + group, nibble * 4),
                &[v(dst)],
                &[v(16 + group)],
            )?;
        }
        for group in 0..4u8 {
            let dst = 4 + group;
            op(
                b,
                format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
                &[v(dst)],
                &[v(dst)],
            )?;
        }
        for group in 0..4u8 {
            let dst = 4 + group;
            let header = 8 + group * 2;
            op(b, format!("v_fma_mix_f32 v{dst}, v{header}, v{dst}, v{header} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[v(dst)], &[v(header), v(dst)])?;
        }
        for group in 0..4u8 {
            let dst = 20 + group;
            let x = 32 + group * 8 + nibble;
            let level = 4 + group;
            if nibble == 1 {
                op(
                    b,
                    format!("v_mul_f32_e32 v{dst}, v{x}, v{level}"),
                    &[v(dst)],
                    &[v(x), v(level)],
                )?;
            } else {
                op(
                    b,
                    format!("v_fmac_f32_e32 v{dst}, v{x}, v{level}"),
                    &[v(dst)],
                    &[v(dst), v(x), v(level)],
                )?;
            }
        }
    }
    for group in 0..4u8 {
        let dst = 24 + group;
        op(
            b,
            format!("v_add_f32_e32 v{dst}, v{dst}, v{}", 20 + group),
            &[v(dst)],
            &[v(dst), v(20 + group)],
        )?;
    }
    Ok(())
}

/// The raw-buffer resource and device scope match the frozen incumbent.
fn load_group(b: &mut Builder) -> Result<(), String> {
    op(b, "v_mov_b32_e32 v3, s22", &[v(3)], &[s(22)])?;
    op(b, "v_add_nc_u32_e32 v28, s22, v1", &[v(28)], &[s(22), v(1)])?;
    op(b, "v_add_nc_u32_e32 v29, s23, v2", &[v(29)], &[s(23), v(2)])?;
    mem(
        b,
        "buffer_load_b64 v[6:7], v3, s[12:15], null offen scope:SCOPE_DEV",
        &[vr(6, 2)],
        &[v(3), sr(12, 4)],
        MemoryClass::VmemLoad,
    )?;
    op(b, "v_cmp_lt_u32_e64 s24, v0, 16", &[s(24)], &[v(0)])?;
    op(
        b,
        "v_cndmask_b32_e64 v5, v7, v6, s24",
        &[v(5)],
        &[v(7), v(6), s(24)],
    )?;
    mem(
        b,
        "buffer_load_b32 v4, v28, s[12:15], null offen offset:8 scope:SCOPE_DEV",
        &[v(4)],
        &[v(28), sr(12, 4)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        "global_load_b128 v[8:11], v29, s[8:9]",
        &[vr(8, 4)],
        &[v(29), sr(8, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        "global_load_b128 v[12:15], v29, s[8:9] offset:16",
        &[vr(12, 4)],
        &[v(29), sr(8, 2)],
        MemoryClass::VmemLoad,
    )
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn g0_region_m7() {
        let emitted = build_region().unwrap();
        let root =
            std::env::temp_dir().join(format!("peacemaker-isa-g0-gate_up-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let stem = root.join("fused_gate_up_hfq4g256_mq4v2.g0");
        std::fs::write(stem.with_extension("g0.s"), &emitted.s_text).unwrap();
        std::fs::write(
            stem.with_extension("g0.proof.json"),
            serde_json::to_vec_pretty(&emitted.proof).unwrap(),
        )
        .unwrap();
        let shape = serde_json::json!({
            "symbol": G0_SYMBOL,
            "counts": {"v_fma_mix_f32": 8, "v_fmac_f32_e32": 7,
                       "ds_swizzle_b32": 1, "ds_bpermute_b32": 4},
            "forbidden": ["scratch_*", "s_waitcnt", "v_wmma_*"],
            "vgpr_max": 32, "sgpr_max": 32,
            "require_wave32": true, "require_zero_spills": true,
            "require_zero_private": true, "launch_dynamic_lds_bytes": 0
        });
        std::fs::write(
            stem.with_extension("g0.shape.json"),
            serde_json::to_vec_pretty(&shape).unwrap(),
        )
        .unwrap();
        let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
        let path = stem.with_extension("g0.co");
        std::fs::write(&path, elf).unwrap();
        let m7 = crate::pm_check::m7(&path, "gfx1201", G0_SYMBOL).unwrap();
        println!("gate_up G0 M7: {m7}");
        assert_eq!(m7["lift"], "byte-exact");
        assert_eq!(m7["obligations"], serde_json::json!({}));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn production_projection_m7() {
        let emitted = build_gfx1201().unwrap();
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].shape.barriers, 0);
        assert!(emitted[0].proof.lds_slots.is_empty());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/pm-decode/gfx1201");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join(format!(
            "fused_gate_up_hfq4g256_mq4v2.test-{}.co",
            std::process::id()
        ));
        let elf = crate::native::assemble(&emitted[0].s_text, Arch::Gfx1201).unwrap();
        std::fs::write(&path, elf).unwrap();
        let m7 = crate::pm_check::m7(&path, "gfx1201", SYMBOL).unwrap();
        println!("gate_up production M7: {m7}");
        std::fs::remove_file(path).unwrap();
        assert_eq!(m7["lift"], "byte-exact");
        assert_eq!(m7["obligations"], serde_json::json!({}));
        assert_eq!(m7["ambiguous_delays"], 0);
    }
}
