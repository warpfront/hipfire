// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab receipt kernels of [`crate::libm`]: one thread per element, `y[i] = f(x[i])` (`e[i]` the `powf`
//! exponent), built only through the shared libm emitters, for gfx1100 / gfx1151 / gfx1201. The exhaustive
//! (`sinf`, `cosf`: every f32 bit pattern) and sampled (`powf`) comparisons against the hipcc objects run in
//! `examples/libm_exact.rs`; none of these kernels is on a product path.
//!
//! ABI (36 bytes): `y` f32 @0, `y2` f32 @8 (`sincosf`'s cosine), `x` f32 @16, `e` f32 @24 (`powf`'s exponent),
//! `n` i32 @32. Grid `[ceil(n / 256), 1, 1]`, block 256, one raw-buffer-bounded dword per thread (a thread past
//! `n` loads 0 and its store is dropped by the descriptor), no LDS, no barrier.
use super::common::{bload, mem, op, s, smem, sop, sr, srd_tail, v};
use crate::insn::{MemoryClass, Sop};
use crate::libm::{self, LargeArg, Pow, Trig};
use crate::{
    arch::WgIdSource,
    plan::Access,
    reg::{Kind, Live},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

type R = Result<(), String>;

/// Threads per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 36;

/// Which function the kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fun {
    Sinf,
    Cosf,
    /// `y = sinf(x)`, `y2 = cosf(x)` through the shared reduction.
    Sincosf,
    Powf,
}

impl Fun {
    pub fn name(self) -> &'static str {
        match self {
            Fun::Sinf => "sinf",
            Fun::Cosf => "cosf",
            Fun::Sincosf => "sincosf",
            Fun::Powf => "powf",
        }
    }
}

/// `libm_exact_<fun>[_masked]_pm_<arch>`.
pub fn symbol(arch: Arch, fun: Fun, masked: bool) -> String {
    format!(
        "libm_exact_{}{}_pm_{}",
        fun.name(),
        if masked { "_masked" } else { "" },
        arch.name()
    )
}

const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 1;

// SGPRs. The workgroup id lives below `S_ARGS` on every arch (gfx1100 pads the user SGPRs to 16).
const KARG: u8 = 0;
const S_ARGS: u8 = 40; // y (40), y2 (42), x (44), e (46)
const S_N: u8 = 48;
const S_WGX: u8 = 49;
const S_BYTES: u8 = 50;
const SRD_Y: u8 = 52;
const SRD_Y2: u8 = 56;
const SRD_X: u8 = 60;
const SRD_E: u8 = 64;
const S_TMP: u8 = 68; // libm scratch (even): 17 SGPRs at most
const S_END: u8 = 88;

// VGPRs.
const TID: u8 = 0;
const VT: u8 = 1;
const VOFF: u8 = 2;
const VX: u8 = 3;
const VE: u8 = 4;
const VY: u8 = 5;
const VY2: u8 = 6;
const V_TMP: u8 = 8; // libm scratch: 13 VGPRs at most
const V_END: u8 = 24;

fn wg_x_in(arch: Arch) -> Result<Option<u8>, String> {
    match arch
        .workgroup_ids(arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS))?
        .x
    {
        WgIdSource::Sgpr(n) => Ok(Some(n)),
        WgIdSource::Ttmp9 => Ok(None),
        other => Err(format!(
            "libm_exact: no workgroup id on {}: {other:?}",
            arch.name()
        )),
    }
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    p.add_range("kernarg_ptr", Kind::S, KARG, 2, Live::Whole)?;
    if let Some(n) = wg_x_in(arch)? {
        p.add_range("wg_x_in", Kind::S, n, 1, Live::Whole)?;
    }
    for base in (S_ARGS..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    Ok(p)
}

fn srd(b: &mut Builder, dst: u8, ptr: u8) -> R {
    sop(b, format!("s_mov_b32 s{dst}, s{ptr}"), &[dst], &[ptr])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", dst + 1, ptr + 1),
        &[dst + 1],
        &[ptr + 1],
    )?;
    srd_tail(b, dst, Some(S_BYTES))
}

fn bstore(b: &mut Builder, data: u8, desc: u8) -> R {
    let soffset = if b.spec.arch.gfx12() { "null" } else { "0" };
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{VOFF}, s[{desc}:{}], {soffset} offen",
            desc + 3
        ),
        &[],
        &[v(VOFF), v(data), sr(desc, 4)],
        MemoryClass::VmemStore,
    )
}

/// The kernel (`masked`: the `Masked` large-argument policy instead of the `execz` branch; ignored by `powf`).
pub fn emit(arch: Arch, fun: Fun, masked: bool) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(format!("libm_exact: not built for {}", arch.name()));
    }
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("y", 0, Access::WriteOnly)
        .pointer_access("y2", 8, Access::WriteOnly)
        .pointer_access("x", 16, Access::ReadOnly)
        .pointer_access("e", 24, Access::ReadOnly)
        .hidden("n", 32, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "libm_exact".into(),
        variant: format!("{}{}", fun.name(), if masked { "_masked" } else { "" }),
        arch,
        symbol: symbol(arch, fun, masked),
        kernargs,
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(arch)?);
    b.enable_delay_alu();
    kernel(&mut b, arch, fun, masked)?;
    b.finish()
}

fn kernel(b: &mut Builder, arch: Arch, fun: Fun, masked: bool) -> R {
    smem(b, S_ARGS, 8, KARG, 0x00)?;
    smem(b, S_N, 1, KARG, 0x20)?;
    match wg_x_in(arch)? {
        None => sop(b, format!("s_mov_b32 s{S_WGX}, ttmp9"), &[S_WGX], &[])?,
        Some(n) => sop(b, format!("s_mov_b32 s{S_WGX}, s{n}"), &[S_WGX], &[n])?,
    }
    sop(
        b,
        format!("s_lshl_b32 s{S_BYTES}, s{S_N}, 2"),
        &[S_BYTES],
        &[S_N],
    )?;
    srd(b, SRD_Y, S_ARGS)?;
    srd(b, SRD_Y2, S_ARGS + 2)?;
    srd(b, SRD_X, S_ARGS + 4)?;
    srd(b, SRD_E, S_ARGS + 6)?;
    op(
        b,
        format!("v_lshl_add_u32 v{VT}, s{S_WGX}, 8, v{TID}"),
        &[v(VT)],
        &[s(S_WGX), v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF}, 2, v{VT}"),
        &[v(VOFF)],
        &[v(VT)],
    )?;
    bload(b, 1, VX, VOFF, SRD_X, 0)?;
    let large = if masked {
        LargeArg::Masked
    } else {
        LargeArg::Branch { id: 0 }
    };
    match fun {
        Fun::Sinf | Fun::Cosf | Fun::Sincosf => {
            let t = Trig {
                x: VX,
                sin: matches!(fun, Fun::Sinf | Fun::Sincosf).then_some(VY),
                cos: match fun {
                    Fun::Cosf => Some(VY),
                    Fun::Sincosf => Some(VY2),
                    _ => None,
                },
                v: V_TMP,
                s: S_TMP,
            };
            libm::sincosf(b, &t, large)?;
            bstore(b, VY, SRD_Y)?;
            if fun == Fun::Sincosf {
                bstore(b, VY2, SRD_Y2)?;
            }
        }
        Fun::Powf => {
            bload(b, 1, VE, VOFF, SRD_E, 0)?;
            libm::powf(
                b,
                &Pow {
                    dst: VY,
                    x: VX,
                    y: VE,
                    v: V_TMP,
                    s: S_TMP,
                },
            )?;
            bstore(b, VY, SRD_Y)?;
        }
    }
    b.wait_all()?;
    b.control(Sop::End.encode(arch)?)
}

/// The kernel as one assembled code object (`.hsaco`-style ELF) for `arch`.
pub fn emit_elf(arch: Arch, fun: Fun, masked: bool) -> Result<Vec<u8>, String> {
    let e = emit(arch, fun, masked)?;
    crate::native::assemble(&e.s_text, arch)
}
