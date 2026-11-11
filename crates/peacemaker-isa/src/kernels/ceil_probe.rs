// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) `v_ceil_f32` device probe for gfx1100, gfx1151 and gfx1201:
//! `pm_ceil_probe_<arch>`. It exists to prove, on hardware, that the typed/text Builder path
//! emits the `v_ceil_f32_e32` table row bit-exactly against hipcc `ceilf` over every f32 bit
//! pattern (`crates/peacemaker-ir/isa/emu/numerics-probe/ceil/`).
//!
//! ABI (16 bytes): `out` `*u32` @0, `base` u32 @8, `n` u32 @12. Grid `[ceil(n / 256), 1, 1]`,
//! block 256 (one thread per word), no LDS, no barrier, no scratch. Thread `t = 256 wg + tid`
//! (for `t < n`) stores `out[t] = bits(ceil(f32_from_bits(base + t)))`; the input pattern is
//! computed with an integer add, so no float opcode touches it before the ceil.
use super::common::{op, s, smem, sop, sr, srd_tail, v};
use crate::{
    arch::WgIdSource,
    insn::MemoryClass,
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, Target, Workgroup};

type R = Result<(), String>;

/// Threads (words) per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 16;

pub fn symbol(arch: Arch) -> String {
    format!("pm_ceil_probe_{}", arch.name())
}

const KARG: u8 = 0;
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 1;
/// `s_load_b128` destination s[16:19]: out lo/hi, base, n (clear of the padded gfx1100 user SGPRs s2..15).
const ARG: u8 = 16;
const BASE: u8 = ARG + 2;
const N: u8 = ARG + 3;
const WGX: u8 = 20;
const BYTES: u8 = 21;
const SRD: u8 = 24;
const VALID: u8 = 28;
const S_END: u8 = 32;

const TID: u8 = 0;
const VT: u8 = 1;
const VX: u8 = 2;
const VY: u8 = 3;
const VOFF: u8 = 4;
const V_END: u8 = 8;

fn wg_x_in(arch: Arch) -> Result<u8, String> {
    match arch
        .workgroup_ids(arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS))?
        .x
    {
        WgIdSource::Sgpr(n) => Ok(n),
        other => Err(format!(
            "pm_ceil_probe: no workgroup id SGPR on {}: {other:?}",
            arch.name()
        )),
    }
}

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("out", 0, Access::WriteOnly)
        .hidden("base", 8, 4, "by_value")
        .hidden("n", 12, 4, "by_value")
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    if !arch.gfx12() {
        p.s::<1>("wg_x_in", wg_x_in(arch)?, Live::Whole)?;
    }
    for base in (16u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1100, gfx1151, gfx1201).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(format!(
            "pm_ceil_probe: built for gfx1100, gfx1151 and gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "pm_ceil_probe".into(),
        variant: "v_ceil_f32".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(arch)?);
    b.enable_delay_alu();
    match arch {
        Arch::Gfx1100 => kernel(&mut Workgroup::<Gfx1100, Builder>::new(&mut b)?, arch)?,
        Arch::Gfx1151 => kernel(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, arch)?,
        _ => kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, arch)?,
    }
    b.finish()
}

fn m(x: u8) -> RegRef {
    sr(x, 2)
}

fn kernel<T: Target>(wg: &mut Workgroup<'_, T, Builder>, arch: Arch) -> R {
    let end = wg.exit(".Lceil_probe_end")?;
    let b = wg.isa();
    smem(b, ARG, 4, KARG, 0)?;
    if arch.gfx12() {
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    } else {
        let wg_in = wg_x_in(arch)?;
        sop(b, format!("s_mov_b32 s{WGX}, s{wg_in}"), &[WGX], &[wg_in])?;
    }
    // Descriptor over `out`: n words.
    sop(b, format!("s_lshl_b32 s{BYTES}, s{N}, 2"), &[BYTES], &[N])?;
    sop(b, format!("s_mov_b32 s{SRD}, s{ARG}"), &[SRD], &[ARG])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD + 1, ARG + 1),
        &[SRD + 1],
        &[ARG + 1],
    )?;
    srd_tail(b, SRD, Some(BYTES))?;
    // t = 256 wg + tid; x = base + t (integer add: the f32 bit pattern); y = ceil(x).
    op(
        b,
        format!("v_lshl_add_u32 v{VT}, s{WGX}, 8, v{TID}"),
        &[v(VT)],
        &[s(WGX), v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VX}, s{BASE}, v{VT}"),
        &[v(VX)],
        &[s(BASE), v(VT)],
    )?;
    op(
        b,
        format!("v_ceil_f32_e32 v{VY}, v{VX}"),
        &[v(VY)],
        &[v(VX)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF}, 2, v{VT}"),
        &[v(VOFF)],
        &[v(VT)],
    )?;
    // Only lanes with t < n store.
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{VALID}, s{N}, v{VT}"),
        &[m(VALID)],
        &[s(N), v(VT)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{VALID}"),
        &[],
        &[m(VALID)],
    )?;
    let soffset = if arch.gfx12() { "null" } else { "0" };
    super::common::mem(
        b,
        format!(
            "buffer_store_b32 v{VY}, v{VOFF}, s[{SRD}:{}], {soffset} offen",
            SRD + 3
        ),
        &[],
        &[v(VOFF), v(VY), sr(SRD, 4)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    wg.end(end)
}
