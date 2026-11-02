// SPDX-License-Identifier: Apache-2.0
//! Exact gfx1201 scalar QKV projection twin. Four independent group streams
//! retain the clang24 nibble-dot DAG and tail assignment, followed by the
//! frozen pairwise combine and shfl-down16/8/4/2/1 wave reduction.
//! The separately named G0 export writes lane partials and is diagnostic only.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{mem, op, s, smem, sop, sr, v, vr};
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

const MODULE: &str = "fused_qkv_hfq4g256_mq4v2";
#[cfg(all(test, feature = "toolchain"))]
const G0_SYMBOL: &str = "fused_qkv_mq4g256v2_g0_group";

fn kernargs() -> KernargLayout {
    let mut args = KernargLayout::new(72);
    for (name, offset) in [
        ("A_q", 0),
        ("A_k", 8),
        ("A_v", 16),
        ("x", 24),
        ("y_q", 32),
        ("y_k", 40),
        ("y_v", 48),
    ] {
        let access = if offset < 32 {
            crate::plan::Access::ReadOnly
        } else {
            crate::plan::Access::WriteOnly
        };
        args = args.pointer_access(name, offset, access);
    }
    for (i, name) in ["q_m", "k_m", "v_m", "K"].iter().enumerate() {
        args = args.hidden(name, 56 + i as u32 * 4, 4, "by_value");
    }
    args
}

/// Exact clang24 lane dot: product 1, FMA 0, then FMAs 2..7.
/// Dequantization consumes the packed low-half scale/high-half zero point
/// directly, using the selected body's mixed-precision FMA operand modes.
fn group_dot(b: &mut Builder) -> Result<(), String> {
    for i in 0..8u8 {
        let dst = 16 + i;
        op(
            b,
            format!("v_bfe_u32 v{dst}, v6, {}, 4", i * 4),
            &[v(dst)],
            &[v(6)],
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
    op(b, "v_mul_f32_e32 v24, v17, v9", &[v(24)], &[v(17), v(9)])?;
    for i in [0u8, 2, 3, 4, 5, 6, 7] {
        op(
            b,
            format!("v_fmac_f32_e32 v24, v{}, v{}", 16 + i, 8 + i),
            &[v(24)],
            &[v(24), v(16 + i), v(8 + i)],
        )?;
    }
    Ok(())
}

fn project_group(b: &mut Builder, stream: u8) -> Result<(), String> {
    let offset = u32::from(stream) * 136;
    let suffix = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    op(b, "v_add_nc_u32_e32 v4, s40, v3", &[v(4)], &[s(40), v(3)])?;
    op(b, "v_add_nc_u32_e32 v7, s40, v1", &[v(7)], &[s(40), v(1)])?;
    mem(
        b,
        format!("global_load_b32 v5, v4, s[32:33]{suffix}"),
        &[v(5)],
        &[v(4), sr(32, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!("global_load_b32 v6, v7, s[32:33]{suffix}"),
        &[v(6)],
        &[v(7), sr(32, 2)],
        MemoryClass::VmemLoad,
    )?;
    let xoff = u32::from(stream) * 1024;
    let xsuffix = if xoff == 0 {
        String::new()
    } else {
        format!(" offset:{xoff}")
    };
    mem(
        b,
        format!("global_load_b128 v[8:11], v2, s[10:11]{xsuffix}"),
        &[vr(8, 4)],
        &[v(2), sr(10, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!(
            "global_load_b128 v[12:15], v2, s[10:11] offset:{}",
            xoff + 16
        ),
        &[vr(12, 4)],
        &[v(2), sr(10, 2)],
        MemoryClass::VmemLoad,
    )?;
    group_dot(b)?;
    let acc = 26 + stream;
    op(
        b,
        format!("v_add_f32_e32 v{acc}, v{acc}, v24"),
        &[v(acc)],
        &[v(acc), v(24)],
    )?;
    b.wait_all()
}

/// Batch the quad's memory operations and interleave four independent dot
/// chains. First-half x loads precede all second-half loads, so Builder can
/// prove progressive waits instead of draining the counter for each group.
/// Retain all dequantized weights; reuse dead headers and packed words to fit 80 VGPRs.
fn project_quad(b: &mut Builder) -> Result<(), String> {
    b.clause(|b| {
        for stream in 0..4u8 {
            let offset = u32::from(stream) * 136;
            let suffix = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
            let header = 8 + stream * 2;
            if stream == 0 || stream == 3 {
                mem(b, format!("buffer_load_b64 v[{header}:{}], v31, s[36:39], s40 offen{suffix} scope:SCOPE_DEV", header + 1),
                    &[vr(header, 2)], &[v(31), sr(36, 4), s(40)], MemoryClass::VmemLoad)?;
            } else {
                mem(b, format!("buffer_load_b32 v{header}, v31, s[36:39], s40 offen{suffix} scope:SCOPE_DEV"),
                    &[v(header)], &[v(31), sr(36, 4), s(40)], MemoryClass::VmemLoad)?;
                mem(b, format!("buffer_load_b32 v{}, v31, s[36:39], s40 offen offset:{} scope:SCOPE_DEV", header + 1, offset + 4),
                    &[v(header + 1)], &[v(31), sr(36, 4), s(40)], MemoryClass::VmemLoad)?;
            }
        }
        for stream in 0..4u8 {
            let offset = u32::from(stream) * 136;
            let suffix = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
            let packed = 20 + stream;
            mem(b, format!("buffer_load_b32 v{packed}, v1, s[36:39], s40 offen{suffix} scope:SCOPE_DEV"),
                &[v(packed)], &[v(1), sr(36, 4), s(40)], MemoryClass::VmemLoad)?;
        }
        Ok(())
    })?;
    b.clause(|b| {
        for half in [0u8, 4] {
            for stream in 0..4u8 {
                let base = 32 + stream * 8 + half;
                let offset = u32::from(stream) * 1024 + u32::from(half) * 4;
                let suffix = if offset == 0 {
                    String::new()
                } else {
                    format!(" offset:{offset}")
                };
                mem(
                    b,
                    format!(
                        "global_load_b128 v[{base}:{}], v2, s[10:11]{suffix}",
                        base + 3
                    ),
                    &[vr(base, 4)],
                    &[v(2), sr(10, 2)],
                    MemoryClass::VmemLoad,
                )?;
            }
        }
        Ok(())
    })?;
    // Addresses are already issued. Keep this SCC live through the vector-only
    // body so loop bookkeeping overlaps dequantization instead of its epilogue.
    sop(b, "s_add_co_i32 s40, s40, 0x220", &[40], &[40])?;
    sop(b, "s_sub_co_i32 s34, s34, 1", &[34], &[34])?;
    sop(b, "s_cmp_lg_u32 s34, 0", &[], &[34])?;
    op(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    for stream in 0..4u8 {
        if stream == 1 {
            b.wait(crate::ledger::Counter::Load, 12)?;
        }
        let header = 8 + stream * 2;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{header}, vcc_lo",
                4 + stream,
                header + 1
            ),
            &[v(4 + stream)],
            &[v(header), v(header + 1)],
        )?;
    }
    b.wait(crate::ledger::Counter::Load, 8)?;
    // Consume streams 2/3 before stream 1 reuses their packed registers.
    // Stream 1's term 5 replaces its own packed word, so extract it last.
    for stream in [2u8, 3, 0, 1] {
        for term in [1u8, 0, 2, 3, 4, 6, 7, 5] {
            let weight = quad_weight(stream, term);
            let packed = 20 + stream;
            let insn = match term {
                0 => format!("v_and_b32_e32 v{weight}, 15, v{packed}"),
                7 => format!("v_lshrrev_b32_e32 v{weight}, 28, v{packed}"),
                _ => format!("v_bfe_u32 v{weight}, v{packed}, {}, 4", term * 4),
            };
            op(b, insn, &[v(weight)], &[v(packed)])?;
        }
    }
    for term in 0..8u8 {
        for stream in 0..4u8 {
            let weight = quad_weight(stream, term);
            op(
                b,
                format!("v_cvt_f32_ubyte0_e32 v{weight}, v{weight}"),
                &[v(weight)],
                &[v(weight)],
            )?;
        }
    }
    for term in 0..8u8 {
        for stream in 0..4u8 {
            let weight = quad_weight(stream, term);
            let header = 4 + stream;
            op(b, format!("v_fma_mix_f32 v{weight}, v{header}, v{weight}, v{header} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[v(weight)], &[v(header), v(weight)])?;
        }
    }
    // Dequantization hides all eight activation loads; retire them together.
    b.wait(crate::ledger::Counter::Load, 0)?;
    // All headers are now dead: their four registers become the lane dots.
    for stream in [0u8, 2] {
        let weight = quad_weight(stream, 1);
        op(
            b,
            format!(
                "v_mul_f32_e32 v{}, v{weight}, v{}",
                4 + stream,
                33 + stream * 8
            ),
            &[v(4 + stream)],
            &[v(weight), v(33 + stream * 8)],
        )?;
    }
    // Even streams lead their odd partner by one term, preserving each
    // dot's 1 -> 0 -> 2..7 chain while satisfying both VOPD source banks.
    for (lead, lag) in [(0u8, 1u8), (2, 0), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)] {
        for stream in [0u8, 2] {
            use crate::vopd::{Operand, VopdF32, VopdOp};
            // The nearest consumed dot producer is two VALU packets back.
            op(b, "s_delay_alu instid0(VALU_DEP_2)", &[], &[])?;
            b.vopd(
                VopdOp {
                    op: VopdF32::Fmac,
                    dst: 4 + stream,
                    src0: Operand::V(quad_weight(stream, lead)),
                    src1: 32 + stream * 8 + lead,
                },
                VopdOp {
                    op: if lag == 1 {
                        VopdF32::Mul
                    } else {
                        VopdF32::Fmac
                    },
                    dst: 5 + stream,
                    src0: Operand::V(quad_weight(stream + 1, lag)),
                    src1: 40 + stream * 8 + lag,
                },
            )?;
        }
    }
    for stream in [1u8, 3] {
        let weight = quad_weight(stream, 7);
        op(
            b,
            format!(
                "v_fmac_f32_e32 v{}, v{weight}, v{}",
                4 + stream,
                39 + stream * 8
            ),
            &[v(4 + stream)],
            &[v(4 + stream), v(weight), v(39 + stream * 8)],
        )?;
    }
    for stream in [0u8, 2] {
        use crate::vopd::{Operand, VopdF32, VopdOp};
        op(b, "s_delay_alu instid0(VALU_DEP_2)", &[], &[])?;
        b.vopd(
            VopdOp {
                op: VopdF32::Add,
                dst: 26 + stream,
                src0: Operand::V(26 + stream),
                src1: 4 + stream,
            },
            VopdOp {
                op: VopdF32::Add,
                dst: 27 + stream,
                src0: Operand::V(27 + stream),
                src1: 5 + stream,
            },
        )?;
    }
    Ok(())
}

const fn quad_weight(stream: u8, term: u8) -> u8 {
    [8, 16, 64, 72][stream as usize] + term
}

/// This is the incumbent shfl_down semantics, including out-of-range lanes
/// reading themselves rather than wrapping to the beginning of the wave.
fn reduce_wave(b: &mut Builder) -> Result<(), String> {
    op(b, "v_add_f32_e32 v24, v26, v27", &[v(24)], &[v(26), v(27)])?;
    op(b, "v_add_f32_e32 v25, v28, v29", &[v(25)], &[v(28), v(29)])?;
    op(b, "v_add_f32_e32 v24, v24, v25", &[v(24)], &[v(24), v(25)])?;
    b.ds_crosslane(
        Instruction::new(
            "ds_swizzle_b32 v30, v24 offset:swizzle(BITMASK_PERM,\"1pppp\")",
            vec![v(30)],
            vec![v(24)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(crate::ledger::Counter::Ds, 0)?;
    op(b, "v_add_f32_e32 v24, v24, v30", &[v(24)], &[v(24), v(30)])?;
    for step in [8u8, 4, 2, 1] {
        op(
            b,
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - step),
            &[],
            &[v(0)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v31, 0, {step}, vcc_lo"),
            &[v(31)],
            &[],
        )?;
        op(
            b,
            "v_add_lshl_u32 v31, v31, v0, 2",
            &[v(31)],
            &[v(31), v(0)],
        )?;
        b.ds_crosslane(
            Instruction::new(
                "ds_bpermute_b32 v30, v31, v24",
                vec![v(30)],
                vec![v(31), v(24)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        op(b, "v_add_f32_e32 v24, v24, v30", &[v(24)], &[v(24), v(30)])?;
    }
    Ok(())
}

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(80, 48)?;
    regs.add_range("workitem", Kind::V, 0, 1, Live::Whole)?;
    for base in 1..8 {
        regs.add_range(
            &format!("address_or_packed_{base}"),
            Kind::V,
            base,
            1,
            Live::Whole,
        )?;
    }
    for base in [8, 16, 24] {
        regs.add_range(&format!("fragment_{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    for base in [32, 40, 48, 56, 64, 72] {
        regs.add_range(&format!("quad_x_{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    for (base, len) in [(0, 2), (4, 4), (8, 8), (16, 8), (24, 8), (32, 8), (40, 2)] {
        regs.add_range(&format!("scalar_{base}"), Kind::S, base, len, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: MODULE.into(),
            variant: "four-stream-quad-ilp".into(),
            arch: Arch::Gfx1201,
            symbol: "fused_qkv_mq4g256v2".into(),
            kernargs: kernargs(),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    smem(&mut b, 4, 4, 0, 0)?;
    smem(&mut b, 8, 4, 0, 16)?;
    smem(&mut b, 12, 4, 0, 32)?;
    smem(&mut b, 16, 4, 0, 48)?;
    smem(&mut b, 20, 2, 0, 64)?;
    // These lane-only operations overlap the outstanding kernarg loads.
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_add_nc_u32_e32 v1, 8, v1", &[v(1)], &[v(1)])?;
    op(&mut b, "v_lshlrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?;
    op(&mut b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    op(&mut b, "v_cndmask_b32_e64 v3, 4, 0, vcc_lo", &[v(3)], &[])?;
    op(&mut b, "v_mov_b32_e32 v31, 0", &[v(31)], &[])?;
    for acc in 26..30 {
        op(&mut b, format!("v_mov_b32_e32 v{acc}, 0"), &[v(acc)], &[])?;
    }
    b.wait_all()?;
    sop(&mut b, "s_add_co_i32 s22, s18, s19", &[22], &[18, 19])?;
    sop(&mut b, "s_add_co_i32 s23, s22, s20", &[23], &[22, 20])?;
    sop(&mut b, "s_cmp_ge_i32 ttmp9, s23", &[], &[23])?;
    op(&mut b, "s_cbranch_scc1 .Lqkv_done", &[], &[])?;
    sop(&mut b, "s_sub_co_i32 s30, ttmp9, s18", &[30], &[18])?;
    sop(&mut b, "s_cmp_lt_i32 ttmp9, s18", &[], &[18])?;
    sop(&mut b, "s_cselect_b32 s24, ttmp9, s30", &[24], &[30])?;
    for (dst, q, k) in [(26, 4, 6), (27, 5, 7), (28, 12, 14), (29, 13, 15)] {
        sop(
            &mut b,
            format!("s_cselect_b32 s{dst}, s{q}, s{k}"),
            &[dst],
            &[q, k],
        )?;
    }
    sop(&mut b, "s_sub_co_i32 s30, ttmp9, s22", &[30], &[22])?;
    sop(&mut b, "s_cmp_lt_i32 ttmp9, s22", &[], &[22])?;
    sop(&mut b, "s_cselect_b32 s24, s24, s30", &[24], &[24, 30])?;
    for (dst, src) in [(26, 8), (27, 9), (28, 16), (29, 17)] {
        sop(
            &mut b,
            format!("s_cselect_b32 s{dst}, s{dst}, s{src}"),
            &[dst],
            &[dst, src],
        )?;
    }
    sop(&mut b, "s_ashr_i32 s25, s24, 31", &[25], &[24])?;
    sop(&mut b, "s_ashr_i32 s31, s21, 8", &[31], &[21])?;
    sop(&mut b, "s_lshr_b32 s34, s31, 2", &[34], &[31])?;
    sop(&mut b, "s_and_b32 s35, s31, 3", &[35], &[31])?;
    sop(&mut b, "s_mul_i32 s32, s31, 0x88", &[32], &[31])?;
    sop(&mut b, "s_ashr_i32 s33, s32, 31", &[33], &[32])?;
    op(
        &mut b,
        "s_mul_u64 s[32:33], s[32:33], s[24:25]",
        &[sr(32, 2)],
        &[sr(32, 2), sr(24, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[32:33], s[26:27], s[32:33]",
        &[sr(32, 2)],
        &[sr(26, 2), sr(32, 2)],
    )?;
    op(
        &mut b,
        "s_mov_b64 s[36:37], s[32:33]",
        &[sr(36, 2)],
        &[sr(32, 2)],
    )?;
    sop(&mut b, "s_and_b32 s37, s37, 0xffff", &[37], &[37])?;
    sop(&mut b, "s_mov_b32 s38, -1", &[38], &[])?;
    sop(&mut b, "s_mov_b32 s39, 0x31004000", &[39], &[])?;
    sop(&mut b, "s_mov_b32 s40, 0", &[40], &[])?;
    sop(&mut b, "s_cmp_eq_u32 s34, 0", &[], &[34])?;
    op(&mut b, "s_cbranch_scc1 .Lqkv_tails", &[], &[])?;
    b.loop_(".Lqkv_quads", |b| {
        project_quad(b)?;
        op(b, "v_add_nc_u32_e32 v2, 0x1000, v2", &[v(2)], &[v(2)])?;
        op(b, "s_cbranch_scc1 .Lqkv_quads", &[], &[])
    })?;
    b.label(".Lqkv_tails")?;
    for stream in 0..3 {
        sop(
            &mut b,
            format!("s_cmp_lt_u32 s35, {}", stream + 1),
            &[],
            &[35],
        )?;
        op(&mut b, "s_cbranch_scc1 .Lqkv_reduce", &[], &[])?;
        project_group(&mut b, stream)?;
    }
    b.label(".Lqkv_reduce")?;
    reduce_wave(&mut b)?;
    op(&mut b, "v_cmpx_eq_u32_e32 0, v0", &[], &[v(0)])?;
    op(&mut b, "s_cbranch_execz .Lqkv_done", &[], &[])?;
    op(
        &mut b,
        "s_lshl_b64 s[24:25], s[24:25], 2",
        &[sr(24, 2)],
        &[sr(24, 2)],
    )?;
    op(
        &mut b,
        "s_add_nc_u64 s[32:33], s[28:29], s[24:25]",
        &[sr(32, 2)],
        &[sr(28, 2), sr(24, 2)],
    )?;
    op(&mut b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    mem(
        &mut b,
        "global_store_b32 v1, v24, s[32:33]",
        &[],
        &[v(1), v(24), sr(32, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    b.label(".Lqkv_done")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    Ok(vec![b.finish()?])
}

/// Preserved arithmetic oracle fixture; unavailable to production callers.
#[cfg(all(test, feature = "toolchain"))]
fn build_g0() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(32, 16)?;
    regs.add_range("workitem", Kind::V, 0, 1, Live::Whole)?;
    for base in [1, 2, 3, 4, 5, 6, 7, 24, 25] {
        regs.add_range(&format!("v{base}"), Kind::V, base, 1, Live::Whole)?;
    }
    for base in [8, 16] {
        regs.add_range(&format!("v{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)?;
    regs.add_range("wg_x", Kind::S, 2, 1, Live::Whole)?;
    for base in [4, 6, 8] {
        regs.add_range(&format!("s{base}"), Kind::S, base, 2, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: MODULE.into(),
            variant: "g0-single-group-lane-dot".into(),
            arch: Arch::Gfx1201,
            symbol: G0_SYMBOL.into(),
            kernargs: kernargs(),
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
    smem(&mut b, 6, 2, 0, 24)?;
    smem(&mut b, 8, 2, 0, 32)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_add_nc_u32_e32 v2, 8, v1", &[v(2)], &[v(1)])?;
    op(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[v(3)], &[v(0)])?;
    op(&mut b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    op(&mut b, "v_cndmask_b32_e64 v4, 4, 0, vcc_lo", &[v(4)], &[])?;
    mem(
        &mut b,
        "global_load_b32 v5, v4, s[4:5]",
        &[v(5)],
        &[v(4), sr(4, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "global_load_b32 v6, v2, s[4:5]",
        &[v(6)],
        &[v(2), sr(4, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "global_load_b128 v[8:11], v3, s[6:7]",
        &[vr(8, 4)],
        &[v(3), sr(6, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "global_load_b128 v[12:15], v3, s[6:7] offset:16",
        &[vr(12, 4)],
        &[v(3), sr(6, 2)],
        MemoryClass::VmemLoad,
    )?;
    group_dot(&mut b)?;
    // Keep G0 identical to the admitted zero-seeded stream update.
    op(&mut b, "v_mov_b32_e32 v25, 0", &[v(25)], &[])?;
    op(
        &mut b,
        "v_add_f32_e32 v24, v25, v24",
        &[v(24)],
        &[v(25), v(24)],
    )?;
    mem(
        &mut b,
        "global_store_b32 v1, v24, s[8:9]",
        &[],
        &[v(1), v(24), sr(8, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    Ok(vec![b.finish()?])
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn production_emit_and_m7() {
        let emitted = build_gfx1201().expect("checked full-symbol emission");
        assert_eq!(emitted.len(), 1);
        let e = &emitted[0];
        for directive in [
            ".kernarg_segment_size: 72",
            ".kernarg_segment_align: 8",
            ".max_flat_workgroup_size: 32",
            ".wavefront_size: 32",
            ".group_segment_fixed_size: 0",
            ".private_segment_fixed_size: 0",
            ".amdhsa_float_round_mode_32 0",
            ".amdhsa_float_round_mode_16_64 0",
            ".amdhsa_float_denorm_mode_32 3",
            ".amdhsa_float_denorm_mode_16_64 3",
        ] {
            assert!(
                e.s_text.contains(directive),
                "missing frozen ABI/FP directive {directive}"
            );
        }
        assert_eq!(e.s_text.matches(".actual_access: read_only").count(), 4);
        assert_eq!(e.s_text.matches(".actual_access: write_only").count(), 3);
        assert_eq!(e.shape.ds, 5);
        assert_eq!(e.proof.loop_fixpoints.len(), 1);
        assert!(e.proof.lds_slots.is_empty());
        let quad = e
            .s_text
            .split(".Lqkv_quads:")
            .nth(1)
            .unwrap()
            .split("v_add_nc_u32_e32 v2, 0x1000, v2")
            .next()
            .unwrap();
        let issued_before_wait = quad
            .lines()
            .take_while(|line| !line.contains("s_wait_loadcnt"))
            .filter(|line| line.contains("buffer_load_") || line.contains("global_load_"))
            .count();
        assert_eq!(
            issued_before_wait, 18,
            "quad loads must overlap before the first wait"
        );
        assert_eq!(quad.matches("s_wait_loadcnt").count(), 4);
        assert_eq!(quad.matches(" :: ").count(), 16);
        assert_eq!(quad.matches("s_delay_alu").count(), 16);
        assert_eq!(
            quad.lines()
                .find(|line| line.contains("s_wait_loadcnt"))
                .unwrap()
                .trim(),
            "s_wait_loadcnt 0x11",
            "the oldest header needs only the first of 18 loads"
        );
        let live_scc = quad.split("s_cmp_lg_u32 s34, 0").nth(1).unwrap();
        assert!(
            !live_scc.lines().any(|line| {
                let op = line.trim();
                op.starts_with("s_") && !op.starts_with("s_wait_") && !op.starts_with("s_delay_alu")
            }),
            "early loop SCC must survive the quad body"
        );
        for stream in 0..4u8 {
            let mnemonic = if stream % 2 == 0 {
                "v_mul_f32_e32"
            } else {
                "v_dual_mul_f32"
            };
            assert!(quad.contains(&format!(
                "{mnemonic} v{}, v{}, v{}",
                4 + stream,
                quad_weight(stream, 1),
                33 + stream * 8
            )));
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/pm-decode/gfx1201");
        std::fs::create_dir_all(&root).unwrap();
        // Concurrent crate tests must not overwrite the certified production object.
        let object = root.join(format!("{MODULE}.test.{}.co", std::process::id()));
        let elf = crate::native::assemble(&e.s_text, Arch::Gfx1201).unwrap();
        std::fs::write(&object, elf).unwrap();
        let m7 = crate::pm_check::m7(&object, "gfx1201", "fused_qkv_mq4g256v2");
        std::fs::remove_file(&object).unwrap();
        let m7 = m7.unwrap();
        assert_eq!(m7["obligations"], serde_json::json!({}));
        assert_eq!(m7["ambiguous_delays"], 0);
        println!("{m7}");
    }

    #[test]
    fn g0_emit_and_m7() {
        let emitted = build_g0().expect("checked G0 emission");
        let e = &emitted[0];
        let root = std::env::temp_dir().join(format!(
            "peacemaker-isa-g0-fused_qkv-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = |suffix: &str| root.join(format!("{MODULE}.g0.{suffix}"));
        std::fs::write(path("s"), &e.s_text).unwrap();
        std::fs::write(
            path("proof.json"),
            serde_json::to_vec_pretty(&e.proof).unwrap(),
        )
        .unwrap();
        let contract = crate::toolchain::IsaShapeContract {
            symbol: G0_SYMBOL.into(),
            vgpr_max: Some(32),
            sgpr_max: Some(16),
            require_wave32: true,
            require_zero_spills: true,
            require_zero_private: true,
            launch_dynamic_lds_bytes: Some(0),
            forbidden: vec!["scratch_*".into()],
            ..Default::default()
        };
        std::fs::write(
            path("shape.json"),
            serde_json::to_vec_pretty(&contract).unwrap(),
        )
        .unwrap();
        let elf = crate::native::assemble(&e.s_text, Arch::Gfx1201).expect("native code object");
        std::fs::write(path("co"), elf).unwrap();
        let m7 = crate::pm_check::m7(&path("co"), "gfx1201", G0_SYMBOL)
            .expect("G0 requires M7 obligations {}");
        println!("{m7}");
        std::fs::write(path("m7.json"), serde_json::to_vec_pretty(&m7).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }
}
