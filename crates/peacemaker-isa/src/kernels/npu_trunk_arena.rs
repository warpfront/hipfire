// SPDX-License-Identifier: Apache-2.0
//! Direct row-major BF16 readers for F32/BF16 caches and oracle projection copies.
use crate::insn::MemoryClass;
use crate::kernels::common::{mem, op, s, smem, sr, srd_tail, v};
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const SYMBOLS: [&str; 3] = [
    "npu_arena_copy_bf16_f32_pm",
    "npu_arena_copy_bf16_bf16_pm",
    "npu_arena_copy_f32_f32_pm",
];
fn copy(which: usize) -> Result<Emitted, String> {
    let mut regs = RegPlan::new(16, 24)?;
    regs.s::<2>("kernargs", 0, Live::Whole)?;
    for r in [4, 8] {
        regs.s::<4>("descriptor", r, Live::Whole)?;
    }
    for r in [2, 3].into_iter().chain(12..24) {
        regs.s::<1>("scalar", r, Live::Whole)?;
    }
    for r in 0..16 {
        regs.v::<1>("work", r, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "npu_trunk_arena".into(),
            variant: SYMBOLS[which].into(),
            arch: Arch::Gfx1151,
            symbol: SYMBOLS[which].into(),
            kernargs: KernargLayout::new(40)
                .pointer("src", 0)
                .pointer("dst", 8)
                .hidden("src_pitch", 16, 4, "by_value")
                .hidden("src_col0", 20, 4, "by_value")
                .hidden("dst_pitch", 24, 4, "by_value")
                .hidden("dst_col0", 28, 4, "by_value")
                .hidden("cols", 32, 4, "by_value")
                .hidden("position_start", 36, 4, "by_value"),
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
    for (r, off) in [(4, 0), (8, 8)] {
        smem(&mut b, r, 2, 0, off)?;
        srd_tail(&mut b, r, None)?;
    }
    for (r, off) in [(12, 16), (13, 20), (14, 24), (15, 28), (16, 32), (17, 36)] {
        smem(&mut b, r, 1, 0, off)?;
    }
    op(&mut b, "s_lshl_b32 s18, s2, 5", &[s(18)], &[s(2)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v1, s18, v0",
        &[v(1)],
        &[s(18), v(0)],
    )?;
    op(
        &mut b,
        "v_cmp_gt_u32_e64 s19, s16, v1",
        &[s(19)],
        &[s(16), v(1)],
    )?;
    op(&mut b, "s_mov_b32 exec_lo, s19", &[], &[s(19)])?;
    op(&mut b, "s_mul_i32 s18, s3, s12", &[s(18)], &[s(3), s(12)])?;
    op(&mut b, "s_add_u32 s18, s18, s13", &[s(18)], &[s(18), s(13)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v2, s18, v1",
        &[v(2)],
        &[s(18), v(1)],
    )?;
    op(
        &mut b,
        format!(
            "v_lshlrev_b32_e32 v2, {}, v2",
            if which == 2 { 2 } else { 1 }
        ),
        &[v(2)],
        &[v(2)],
    )?;
    mem(
        &mut b,
        format!(
            "buffer_load_{} v3, v2, s[4:7], 0 offen",
            if which == 2 { "b32" } else { "u16" }
        ),
        &[v(3)],
        &[v(2), sr(4, 4)],
        MemoryClass::VmemLoad,
    )?;
    op(&mut b, "s_add_u32 s18, s3, s17", &[s(18)], &[s(3), s(17)])?;
    op(&mut b, "s_mul_i32 s18, s18, s14", &[s(18)], &[s(18), s(14)])?;
    op(&mut b, "s_add_u32 s18, s18, s15", &[s(18)], &[s(18), s(15)])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v4, s18, v1",
        &[v(4)],
        &[s(18), v(1)],
    )?;
    op(
        &mut b,
        format!(
            "v_lshlrev_b32_e32 v4, {}, v4",
            if which == 1 { 1 } else { 2 }
        ),
        &[v(4)],
        &[v(4)],
    )?;
    if which == 0 {
        op(&mut b, "v_lshlrev_b32_e32 v3, 16, v3", &[v(3)], &[v(3)])?;
    }
    mem(
        &mut b,
        format!(
            "buffer_store_{} v3, v4, s[8:11], 0 offen",
            if which == 1 { "b16" } else { "b32" }
        ),
        &[],
        &[v(3), v(4), sr(8, 4)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}
pub fn emit_module() -> Result<(Vec<Emitted>, String, crate::kernels::iu4_gemm::ModuleProof), String>
{
    let es = vec![copy(0)?, copy(1)?, copy(2)?];
    let (text, proof) = crate::kernels::iu4_gemm::module(&es, "npu_trunk_arena_pm_gfx1151")?;
    Ok((es, text, proof))
}
