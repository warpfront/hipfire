// SPDX-License-Identifier: Apache-2.0
//! gfx1151 G80 BF16 scatter and exact signed-IU8 WMMA oracle.
//! Pack/epilogue HIP gaps are recorded in release-0.4.2/PM-GAPS.md.
use crate::insn::MemoryClass;
use crate::kernels::common::{bload, mem, op, s, smem, sr, srd_tail, v, vr};
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const SYMBOLS: [&str; 3] = [
    "npu_trunk_consume_bf16_pm",
    "npu_trunk_consume_f32_pm",
    "npu_trunk_i8_gemm_pm",
];
type R = Result<(), String>;
fn alu(b: &mut Builder, text: impl Into<String>, d: u8, u: &[u8]) -> R {
    op(
        b,
        text,
        &[v(d)],
        &u.iter().map(|&r| v(r)).collect::<Vec<_>>(),
    )
}
fn builder(symbol: &str, args: KernargLayout) -> Result<Builder, String> {
    let mut regs = RegPlan::new(40, 40)?;
    for r in (0..16).chain(32..40) {
        regs.v::<1>("work", r, Live::Whole)?;
    }
    for r in [16, 20] {
        regs.v::<4>("fragment", r, Live::Whole)?;
    }
    regs.v::<8>("accumulator", 24, Live::Whole)?;
    regs.s::<2>("kernargs", 0, Live::Whole)?;
    for r in [4, 8, 12] {
        regs.s::<4>("descriptor", r, Live::Whole)?;
    }
    for r in [2, 3].into_iter().chain(16..40) {
        regs.s::<1>("scalar", r, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "npu_trunk".into(),
            variant: symbol.into(),
            arch: Arch::Gfx1151,
            symbol: symbol.into(),
            kernargs: args,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    Ok(b)
}
fn descriptors(b: &mut Builder, pointers: usize) -> R {
    for j in 0..pointers {
        let r = 4 + 4 * j as u8;
        smem(b, r, 2, 0, 8 * j as u32)?;
        srd_tail(b, r, None)?;
    }
    Ok(())
}
// Division by 80/1280: index <=65536, f32 quotient within one; exact integer corrections.
fn divrem(b: &mut Builder, src: u8, q: u8, rem: u8, divisor: u32) -> R {
    alu(b, format!("v_cvt_f32_u32_e32 v{q}, v{src}"), q, &[src])?;
    alu(
        b,
        format!(
            "v_mul_f32_e32 v{q}, {:#x}, v{q}",
            (1.0f32 / divisor as f32).to_bits()
        ),
        q,
        &[q],
    )?;
    alu(b, format!("v_cvt_u32_f32_e32 v{q}, v{q}"), q, &[q])?;
    alu(
        b,
        format!("v_mul_lo_u32 v{rem}, {divisor:#x}, v{q}"),
        rem,
        &[q],
    )?;
    alu(
        b,
        format!("v_sub_nc_u32_e32 v{rem}, v{src}, v{rem}"),
        rem,
        &[src, rem],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s32, 0, v{rem}"),
        &[s(32)],
        &[v(rem)],
    )?;
    alu(b, format!("v_add_nc_u32_e32 v30, -1, v{q}"), 30, &[q])?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{q}, v{q}, v30, s32"),
        &[v(q)],
        &[v(q), v(30), s(32)],
    )?;
    alu(
        b,
        format!("v_add_nc_u32_e32 v30, {divisor:#x}, v{rem}"),
        30,
        &[rem],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{rem}, v{rem}, v30, s32"),
        &[v(rem)],
        &[v(rem), v(30), s(32)],
    )?;
    op(
        b,
        format!("v_cmp_le_u32_e64 s32, {divisor:#x}, v{rem}"),
        &[s(32)],
        &[v(rem)],
    )?;
    alu(b, format!("v_add_nc_u32_e32 v30, 1, v{q}"), 30, &[q])?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{q}, v{q}, v30, s32"),
        &[v(q)],
        &[v(q), v(30), s(32)],
    )?;
    alu(
        b,
        format!("v_subrev_nc_u32_e32 v30, {divisor:#x}, v{rem}"),
        30,
        &[rem],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{rem}, v{rem}, v30, s32"),
        &[v(rem)],
        &[v(rem), v(30), s(32)],
    )
}
fn consume(bf16: bool) -> Result<Emitted, String> {
    let mut b = builder(
        SYMBOLS[if bf16 { 0 } else { 1 }],
        KernargLayout::new(28)
            .pointer("C", 0)
            .pointer("Y", 8)
            .hidden("padded_rows", 16, 4, "by_value")
            .hidden("col0", 20, 4, "by_value")
            .hidden("cols", 24, 4, "by_value"),
    )?;
    descriptors(&mut b, 2)?;
    smem(&mut b, 12, 1, 0, 16)?;
    smem(&mut b, 13, 1, 0, 20)?;
    smem(&mut b, 14, 1, 0, 24)?;
    op(&mut b, "s_lshl_b32 s15, s2, 5", &[s(15)], &[s(2)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v1, s15, v0",
        &[v(1)],
        &[s(15), v(0)],
    )?;
    op(
        &mut b,
        "v_cmp_gt_u32_e64 s34, s14, v1",
        &[s(34)],
        &[s(14), v(1)],
    )?;
    op(&mut b, "s_mov_b32 exec_lo, s34", &[], &[s(34)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v2, s13, v1",
        &[v(2)],
        &[s(13), v(1)],
    )?;
    divrem(&mut b, 2, 3, 4, 1280)?; // panel / panel column
    divrem(&mut b, 4, 5, 6, 80)?; // N tile / N local
                                  // BF16 element offset: panel*P*1280 + col*P*160 + pair*P*80 + wave*20480 + parity*10240 + local.
    op(&mut b, "v_mul_lo_u32 v7, s12, v3", &[v(7)], &[s(12), v(3)])?;
    alu(&mut b, "v_mul_lo_u32 v7, 0x500, v7", 7, &[7])?;
    alu(&mut b, "v_lshrrev_b32_e32 v8, 1, v5", 8, &[5])?;
    op(&mut b, "v_mul_lo_u32 v8, s12, v8", &[v(8)], &[s(12), v(8)])?;
    alu(&mut b, "v_mul_lo_u32 v8, 0xa0, v8", 8, &[8])?;
    alu(&mut b, "v_add_nc_u32_e32 v7, v7, v8", 7, &[7, 8])?;
    op(&mut b, "s_bfe_u32 s15, s3, 0x10007", &[s(15)], &[s(3)])?;
    op(&mut b, "s_mul_i32 s15, s15, s12", &[s(15)], &[s(15), s(12)])?;
    op(&mut b, "s_mul_i32 s15, s15, 0x50", &[s(15)], &[s(15)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v7, s15, v7",
        &[v(7)],
        &[s(15), v(7)],
    )?;
    op(&mut b, "s_lshr_b32 s15, s3, 8", &[s(15)], &[s(3)])?;
    op(&mut b, "s_mul_i32 s15, s15, 0x5000", &[s(15)], &[s(15)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v7, s15, v7",
        &[v(7)],
        &[s(15), v(7)],
    )?;
    alu(&mut b, "v_and_b32_e32 v8, 1, v5", 8, &[5])?;
    alu(&mut b, "v_mul_lo_u32 v8, 0x2800, v8", 8, &[8])?;
    alu(&mut b, "v_add_nc_u32_e32 v7, v7, v8", 7, &[7, 8])?;
    op(&mut b, "s_bfe_u32 s15, s3, 0x40003", &[s(15)], &[s(3)])?;
    op(&mut b, "s_mul_i32 s15, s15, 0x280", &[s(15)], &[s(15)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v7, s15, v7",
        &[v(7)],
        &[s(15), v(7)],
    )?;
    alu(&mut b, "v_lshrrev_b32_e32 v8, 3, v6", 8, &[6])?;
    alu(&mut b, "v_lshlrev_b32_e32 v8, 6, v8", 8, &[8])?;
    alu(&mut b, "v_add_nc_u32_e32 v7, v7, v8", 7, &[7, 8])?;
    op(&mut b, "s_and_b32 s15, s3, 7", &[s(15)], &[s(3)])?;
    op(&mut b, "s_lshl_b32 s15, s15, 3", &[s(15)], &[s(15)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v7, s15, v7",
        &[v(7)],
        &[s(15), v(7)],
    )?;
    alu(&mut b, "v_and_b32_e32 v8, 7, v6", 8, &[6])?;
    alu(&mut b, "v_add_nc_u32_e32 v7, v7, v8", 7, &[7, 8])?;
    alu(&mut b, "v_lshlrev_b32_e32 v7, 1, v7", 7, &[7])?;
    mem(
        &mut b,
        "buffer_load_u16 v9, v7, s[4:7], 0 offen",
        &[v(9)],
        &[v(7), sr(4, 4)],
        MemoryClass::VmemLoad,
    )?;
    op(&mut b, "s_mul_i32 s15, s3, s14", &[s(15)], &[s(3), s(14)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v10, s15, v1",
        &[v(10)],
        &[s(15), v(1)],
    )?;
    alu(
        &mut b,
        format!("v_lshlrev_b32_e32 v10, {}, v10", if bf16 { 1 } else { 2 }),
        10,
        &[10],
    )?;
    if !bf16 {
        alu(&mut b, "v_lshlrev_b32_e32 v9, 16, v9", 9, &[9])?;
    }
    mem(
        &mut b,
        format!(
            "buffer_store_{} v9, v10, s[8:11], 0 offen",
            if bf16 { "b16" } else { "b32" }
        ),
        &[],
        &[v(9), v(10), sr(8, 4)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}
fn gemm() -> Result<Emitted, String> {
    let mut b = builder(
        SYMBOLS[2],
        KernargLayout::new(40)
            .pointer("A", 0)
            .pointer("W", 8)
            .pointer("I", 16)
            .hidden("k", 24, 4, "by_value")
            .hidden("rows", 28, 4, "by_value")
            .hidden("n", 32, 4, "by_value")
            .hidden("padded_rows", 36, 4, "by_value"),
    )?;
    descriptors(&mut b, 3)?;
    for (r, off) in [(16, 24), (17, 28), (18, 32), (19, 36)] {
        smem(&mut b, r, 1, 0, off)?;
    }
    alu(&mut b, "v_and_b32_e32 v1, 15, v0", 1, &[0])?;
    op(&mut b, "s_lshl_b32 s20, s3, 4", &[s(20)], &[s(3)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v1, s20, v1",
        &[v(1)],
        &[s(20), v(1)],
    )?;
    alu(&mut b, "v_and_b32_e32 v2, 15, v0", 2, &[0])?;
    op(&mut b, "s_lshl_b32 s20, s2, 4", &[s(20)], &[s(2)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v2, s20, v2",
        &[v(2)],
        &[s(20), v(2)],
    )?;
    // Clamp fragment rows; masked stores drop the extra tile lanes.
    op(&mut b, "s_add_i32 s20, s18, -1", &[s(20)], &[s(18)])?;
    op(&mut b, "v_min_u32_e32 v2, s20, v2", &[v(2)], &[s(20), v(2)])?;
    op(&mut b, "v_mul_lo_u32 v3, s16, v2", &[v(3)], &[s(16), v(2)])?;
    // A segment = padded_rows/256 * k/64 *4608; row's col and wave are fixed.
    op(&mut b, "s_lshr_b32 s20, s19, 8", &[s(20)], &[s(19)])?;
    op(&mut b, "s_lshr_b32 s21, s16, 6", &[s(21)], &[s(16)])?;
    op(&mut b, "s_mul_i32 s20, s20, s21", &[s(20)], &[s(20), s(21)])?;
    op(&mut b, "s_mul_i32 s20, s20, 0x1200", &[s(20)], &[s(20)])?;
    alu(&mut b, "v_bfe_u32 v4, v1, 7, 1", 4, &[1])?;
    alu(&mut b, "v_lshlrev_b32_e32 v4, 1, v4", 4, &[4])?;
    alu(&mut b, "v_bfe_u32 v5, v1, 3, 1", 5, &[1])?;
    alu(&mut b, "v_add_nc_u32_e32 v4, v4, v5", 4, &[4, 5])?;
    op(&mut b, "v_mul_lo_u32 v4, s20, v4", &[v(4)], &[s(20), v(4)])?;
    alu(&mut b, "v_lshrrev_b32_e32 v5, 8, v1", 5, &[1])?;
    op(&mut b, "v_mul_lo_u32 v5, s21, v5", &[v(5)], &[s(21), v(5)])?;
    alu(&mut b, "v_mul_lo_u32 v5, 0x1200, v5", 5, &[5])?;
    alu(&mut b, "v_add_nc_u32_e32 v4, v4, v5", 4, &[4, 5])?;
    alu(&mut b, "v_bfe_u32 v5, v1, 4, 3", 5, &[1])?;
    alu(&mut b, "v_lshlrev_b32_e32 v5, 9, v5", 5, &[5])?;
    alu(&mut b, "v_add_nc_u32_e32 v4, v4, v5", 4, &[4, 5])?;
    alu(&mut b, "v_and_b32_e32 v5, 7, v1", 5, &[1])?;
    alu(&mut b, "v_lshlrev_b32_e32 v5, 3, v5", 5, &[5])?;
    alu(&mut b, "v_add_nc_u32_e32 v4, v4, v5", 4, &[4, 5])?;
    for r in 24..32 {
        alu(&mut b, format!("v_mov_b32_e32 v{r}, 0"), r, &[])?;
    }
    op(&mut b, "s_mov_b32 s22, 0", &[s(22)], &[])?;
    b.loop_(".Lnpu_k", |b| {
        // packed A K16 fragments; four contiguous 4-byte groups.
        op(b, "s_lshr_b32 s23, s22, 6", &[s(23)], &[s(22)])?;
        op(b, "s_mul_i32 s23, s23, 0x1200", &[s(23)], &[s(23)])?;
        op(b, "s_and_b32 s24, s22, 63", &[s(24)], &[s(22)])?;
        op(b, "s_lshr_b32 s24, s24, 3", &[s(24)], &[s(24)])?;
        op(b, "s_lshl_b32 s24, s24, 6", &[s(24)], &[s(24)])?;
        op(b, "s_add_u32 s23, s23, s24", &[s(23)], &[s(23), s(24)])?;
        op(b, "v_add_nc_u32_e32 v6, s23, v4", &[v(6)], &[s(23), v(4)])?;
        op(b, "v_add_nc_u32_e32 v7, s22, v3", &[v(7)], &[s(22), v(3)])?;
        for (i, o) in [0, 4, 64, 68].into_iter().enumerate() {
            bload(b, 1, 16 + i as u8, 6, 4, o)?;
        }
        bload(b, 4, 20, 7, 8, 0)?;
        op(
            b,
            "v_wmma_i32_16x16x16_iu8 v[24:31], v[20:23], v[16:19], v[24:31] neg_lo:[1,1,0]",
            &[vr(24, 8)],
            &[vr(20, 4), vr(16, 4), vr(24, 8)],
        )?;
        op(b, "s_add_u32 s22, s22, 16", &[s(22)], &[s(22)])?;
        op(b, "s_cmp_lt_u32 s22, s16", &[], &[s(22), s(16)])?;
        op(b, "s_cbranch_scc1 .Lnpu_k", &[], &[])
    })?;
    op(
        &mut b,
        "v_cmp_gt_u32_e64 s34, s17, v1",
        &[s(34)],
        &[s(17), v(1)],
    )?;
    op(&mut b, "v_mul_lo_u32 v8, s18, v1", &[v(8)], &[s(18), v(1)])?;
    alu(&mut b, "v_lshrrev_b32_e32 v9, 4, v0", 9, &[0])?;
    op(&mut b, "s_lshl_b32 s20, s2, 4", &[s(20)], &[s(2)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v9, s20, v9",
        &[v(9)],
        &[s(20), v(9)],
    )?;
    for e in 0..8 {
        alu(
            &mut b,
            format!("v_add_nc_u32_e32 v10, {}, v9", 2 * e),
            10,
            &[9],
        )?;
        op(
            &mut b,
            "v_cmp_gt_u32_e64 s36, s18, v10",
            &[s(36)],
            &[s(18), v(10)],
        )?;
        op(&mut b, "s_and_b32 exec_lo, s34, s36", &[], &[s(34), s(36)])?;
        alu(&mut b, "v_add_nc_u32_e32 v11, v8, v10", 11, &[8, 10])?;
        alu(&mut b, "v_lshlrev_b32_e32 v11, 2, v11", 11, &[11])?;
        mem(
            &mut b,
            format!("buffer_store_b32 v{}, v11, s[12:15], 0 offen", 24 + e),
            &[],
            &[v(24 + e), v(11), sr(12, 4)],
            MemoryClass::VmemStore,
        )?;
        op(&mut b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    }
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}
pub fn emit_module() -> Result<(Vec<Emitted>, String, crate::kernels::iu4_gemm::ModuleProof), String>
{
    let es = vec![consume(true)?, consume(false)?, gemm()?];
    let (text, proof) = crate::kernels::iu4_gemm::module(&es, "npu_trunk_pm_gfx1151")?;
    Ok((es, text, proof))
}
