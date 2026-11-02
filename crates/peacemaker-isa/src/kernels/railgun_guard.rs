// SPDX-License-Identifier: Apache-2.0
//! railgun conditional-node kernels (#867), one gfx1151 wave32 module
//! `railgun_guard_pm_gfx1151`. A ControlWord is the first `u32` of its
//! allocation; nonzero means set.
//!
//! - `railgun_guard_add_pm(flag, x, y, n)`: when `*flag != 0`, `y[i] += x[i]`
//!   (wrapping `u32`) for `i < n`, 32 lanes per workgroup.
//! - `railgun_guard_step_pm(flag, y, limit)`: when `*flag != 0`,
//!   `*flag = y[0] < limit` (the bounded-WHILE continue decision).
//! - `railgun_flag_set_pm(flag, y, limit)`: unguarded `*flag = y[0] < limit`
//!   (the device-written IF condition).
//!
//! The guarded kernels open with the guard prologue railgun-cert certifies
//! (`railgun_cert::guard_flag`): load the flag pointer from kernarg +0, load
//! the flag through it, `s_cmp_eq_u32 flag, 0` and branch around a lone
//! `s_endpgm`. A clear flag ends every wave before any other memory access.
use crate::insn::MemoryClass;
use crate::kernels::common::{mem, op, s, smem, sop, sr, srd_tail, v};
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const MODULE: &str = "railgun_guard_pm_gfx1151";
pub const SYMBOLS: [&str; 3] = [
    "railgun_guard_add_pm",
    "railgun_guard_step_pm",
    "railgun_flag_set_pm",
];
/// Lanes (and work-items) per workgroup of `railgun_guard_add_pm`.
pub const ADD_BLOCK: u16 = 32;

/// The flag pointer at kernarg +0, the flag in `s6`; leaves when it is zero.
fn guard(b: &mut Builder, symbol: &str) -> Result<(), String> {
    smem(b, 4, 2, 0, 0)?;
    smem(b, 6, 1, 4, 0)?;
    sop(b, "s_cmp_eq_u32 s6, 0", &[], &[6])?;
    op(b, format!("s_cbranch_scc0 .L{symbol}_go"), &[], &[])?;
    op(b, "s_endpgm", &[], &[])?;
    b.label(&format!(".L{symbol}_go"))
}

fn builder(symbol: &str, kernargs: KernargLayout) -> Result<Builder, String> {
    let mut regs = RegPlan::new(8, 24)?;
    regs.s::<2>("kernargs", 0, Live::Whole)?;
    regs.s::<1>("workgroup_x", 2, Live::Whole)?;
    regs.s::<2>("flag_ptr", 4, Live::Whole)?;
    for r in [8, 12] {
        regs.s::<4>("descriptor", r, Live::Whole)?;
    }
    for r in [3, 6, 7].into_iter().chain(16..24) {
        regs.s::<1>("scalar", r, Live::Whole)?;
    }
    for r in 0..8 {
        regs.v::<1>("work", r, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "railgun_guard".into(),
            variant: symbol.into(),
            arch: Arch::Gfx1151,
            symbol: symbol.into(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: ADD_BLOCK,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    Ok(b)
}

fn add() -> Result<Emitted, String> {
    let mut b = builder(
        SYMBOLS[0],
        KernargLayout::new(32)
            .pointer("flag", 0)
            .pointer("x", 8)
            .pointer("y", 16)
            .hidden("n", 24, 4, "by_value"),
    )?;
    guard(&mut b, SYMBOLS[0])?;
    for (r, off) in [(8, 8), (12, 16)] {
        smem(&mut b, r, 2, 0, off)?;
        srd_tail(&mut b, r, None)?;
    }
    smem(&mut b, 16, 1, 0, 24)?;
    sop(&mut b, "s_lshl_b32 s17, s2, 5", &[17], &[2])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v1, s17, v0",
        &[v(1)],
        &[s(17), v(0)],
    )?;
    op(
        &mut b,
        "v_cmp_gt_u32_e64 s18, s16, v1",
        &[s(18)],
        &[s(16), v(1)],
    )?;
    op(&mut b, "s_mov_b32 exec_lo, s18", &[], &[s(18)])?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v1", &[v(1)], &[v(1)])?;
    mem(
        &mut b,
        "buffer_load_b32 v2, v1, s[8:11], 0 offen",
        &[v(2)],
        &[v(1), sr(8, 4)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        &mut b,
        "buffer_load_b32 v3, v1, s[12:15], 0 offen",
        &[v(3)],
        &[v(1), sr(12, 4)],
        MemoryClass::VmemLoad,
    )?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v2, v2, v3",
        &[v(2)],
        &[v(2), v(3)],
    )?;
    mem(
        &mut b,
        "buffer_store_b32 v2, v1, s[12:15], 0 offen",
        &[],
        &[v(2), v(1), sr(12, 4)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// `*flag = y[0] < limit` from lane 0; `guarded` adds the guard prologue.
fn decide(symbol: &str, guarded: bool) -> Result<Emitted, String> {
    let mut b = builder(
        symbol,
        KernargLayout::new(24)
            .pointer("flag", 0)
            .pointer("y", 8)
            .hidden("limit", 16, 4, "by_value"),
    )?;
    if guarded {
        guard(&mut b, symbol)?;
    } else {
        smem(&mut b, 4, 2, 0, 0)?;
    }
    smem(&mut b, 8, 2, 0, 8)?;
    srd_tail(&mut b, 8, None)?;
    smem(&mut b, 16, 1, 0, 16)?;
    sop(&mut b, "s_mov_b32 s12, s4", &[12], &[4])?;
    sop(&mut b, "s_mov_b32 s13, s5", &[13], &[5])?;
    srd_tail(&mut b, 12, None)?;
    op(&mut b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    op(&mut b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    mem(
        &mut b,
        "buffer_load_b32 v2, v1, s[8:11], 0 offen",
        &[v(2)],
        &[v(1), sr(8, 4)],
        MemoryClass::VmemLoad,
    )?;
    op(
        &mut b,
        "v_cmp_gt_u32_e64 s18, s16, v2",
        &[s(18)],
        &[s(16), v(2)],
    )?;
    op(&mut b, "v_cndmask_b32_e64 v3, 0, 1, s18", &[v(3)], &[s(18)])?;
    mem(
        &mut b,
        "buffer_store_b32 v3, v1, s[12:15], 0 offen",
        &[],
        &[v(3), v(1), sr(12, 4)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

pub fn emit_module() -> Result<(Vec<Emitted>, String, crate::kernels::iu4_gemm::ModuleProof), String>
{
    let es = vec![
        add()?,
        decide(SYMBOLS[1], true)?,
        decide(SYMBOLS[2], false)?,
    ];
    let (text, proof) = crate::kernels::iu4_gemm::module(&es, MODULE)?;
    Ok((es, text, proof))
}
