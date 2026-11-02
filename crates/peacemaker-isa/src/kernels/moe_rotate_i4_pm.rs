// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-local R128 `block_i4_128` (A4) producer with a **runtime row width**, emitted with the checked ISA
//! builder for gfx1201 only: the single symbol `moe_rotate_i4_pm_gfx1201` in the module
//! `moe_rotate_i4_pm_gfx1201` (bundle `kernels/moe_rotate_i4_pm_gfx1201.hxaco`). It is the PM twin of
//! `kernels/src/qwen4_moe_rotate128_i4.hip` (`qwen4_moe_rotate128_i4`, rotation width 128, C2 `{5,7}`
//! quantizer) for any `K` that is a multiple of 128, in particular the A3B `K = 512` down input.
//!
//! This is `kernels/mq_i4_a4.rs` (the `K = 640` specialization) with only the prologue changed: the rotation, the
//! quantizer, the store and the lane setup are that file's functions, called, not copied, so the arithmetic of every
//! record byte is the same instruction sequence (the `quantize` clamp stays `v_maxmin_num_f32`). The old family's
//! emitted bytes are untouched.
//!
//! ABI (56-byte kernarg segment): `h@0` (read-only, grouped BF16 rows `[P][K]` as `u16` bf16 bits), `sorted@8`
//! (read-only, `P` i32 flat-slot ids, `-1` = pad row), `xq@16` (write-only, the sidecar: `K/128 * L` records of 72
//! bytes), `signs1@24` / `signs2@32` (read-only, 128 f32 each), `K@40` (u32 columns), `P@44` (u32 rows), `L@48`
//! (u32 flat slots).
//!
//! Launch: grid `[P, K/128, 1]`, block `[32, 1, 1]`, one wave32 per (row, K128 group), no LDS, no barrier. The
//! workgroup id is `x = ttmp9` (row, may exceed 65535) and `y = ttmp7 & 0xffff` (group; `K/128 <= 65535` is the host's
//! check; `system_sgpr_workgroup_id_y` is set), so there is no division. `row >= P` or `group >= K/128` returns.
//!
//! Row-locality: a wave reads `sorted[row]`, the 256 bytes of `h` row `row` group `g` (`h + row*2K + g*256`) and the
//! sign tables, and writes exactly record `g*L + sorted[row]` (`xq + 72*(g*L + slot)`). A pad row (`slot < 0`)
//! returns before any load of `h` and writes nothing; slots no valid row names are never written. `slot < L` is the
//! host's contract (each valid slot appears once). Address domain (host-checked): `K % 128 == 0`, `K/128 <= 65535`,
//! `P`, `L` > 0, `slot < L`, `(K/128) * L <= 2^32` records (the largest record index `g*L + slot` is at most `u32::MAX`, as
//! it is a u32; the 72-byte scaling and the `h` pitch `2K` products are exact 64-bit `s_mul_i32` + `s_mul_hi_u32` pairs).
//!
//! Numerics: those of `mq_i4_a4` (FWHT128 with signs1 / signs2 and the `1/sqrt(128)` f32 scale, then the
//! `{5,7}` two-candidate quantizer = `fake_quantize_block_i4_128(candidates = 2)`), K-independent.
//! Resources: no LDS, no barrier, no scratch; 72 VGPRs, wave32.

use crate::kernels::common::{s_add_i32, smem, sop};
use crate::kernels::mq_i4_a4 as a4;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 56;
/// Workgroup size (threads): one wave32.
pub const WORKGROUP_SIZE: u32 = a4::WORKGROUP_SIZE;
/// Rotation width and A4 group width in columns.
pub const GROUP: u32 = a4::GROUP;
/// Bytes of one `block_i4_128` record.
pub const RECORD_BYTES: u32 = a4::RECORD_BYTES;
/// Sign-table seeds of the R128 profile (`signs1`, `signs2`).
pub const SIGN_SEEDS: (u32, u32) = a4::SIGN_SEEDS;
/// VGPRs reserved (whole 8-register granules).
pub const VGPRS: u16 = a4::VGPRS;

/// The only entry: `(arch)`; gfx1201 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
}
impl Spec {
    /// The exported symbol (equal to the module name).
    pub fn symbol(&self) -> String {
        Self::module(self.arch)
    }
    /// The label prefix.
    pub fn tag(&self) -> &'static str {
        "r128"
    }
    pub fn module(arch: Arch) -> String {
        format!("moe_rotate_i4_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("moe_rotate_i4_pm builder is exact-gfx1201 only".into());
        }
        Ok(())
    }
}

/// The module's entries in order (one).
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    vec![Spec { arch }]
}

// ----------------------------------------------------------------------- SGPRs
// The `mq_i4_a4` assignment, with `s_load_b128` s[16:19] = signs2, K, P; `L` is its own dword at s43 and the row
// byte pitch `2K` is s44 (both unused by `mq_i4_a4`).
const ARG_K: u8 = 18;
const ARG_P: u8 = 19;
const ARG_L: u8 = 43;
const ROWB: u8 = 44;

/// `mq_i4_a4`'s register plan plus the two SGPRs above.
fn plan() -> Result<crate::RegPlan, String> {
    let mut regs = a4::plan()?;
    regs.add_range("l_argument", Kind::S, ARG_L, 1, Live::Whole)?;
    regs.add_range("row_pitch", Kind::S, ROWB, 1, Live::Whole)?;
    Ok(regs)
}

/// The grouped-position twin (`moe_rotate_i4_gp_pm_gfx1201`): the same ABI, grid, rotation, quantizer and record bytes,
/// stored at record `g * P + row` (`xq + 72 * (g * P + row)`) instead of `g * L + sorted[row]`, so the down GEMM's tile
/// reads 16 adjacent records. A pad row still returns before any load and writes nothing; the plane holds `K/128 * P`
/// records (host-checked `(K/128) * P <= 2^32`). `L` is read by the ABI but unused.
pub fn emit_gp(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    emit_impl(spec, true)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    emit_impl(spec, false)
}

/// Symbol of the grouped-position twin.
pub fn symbol_gp(arch: Arch) -> String {
    format!("moe_rotate_i4_gp_pm_{}", arch.name())
}

fn emit_impl(spec: Spec, gp: bool) -> Result<Emitted, String> {
    let g = blocks::Gen::new(spec, gp)?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    b.enable_delay_alu();
    blocks::prologue(&mut b, &g)?;
    let e = blocks::rotate(&mut b)?;
    blocks::quantize(&mut b, e)?;
    blocks::store(&mut b)?;
    b.finish()
}

/// The emit's phases as public blocks: the sequence in [`emit_impl`] is the sequence the `.rip` twin
/// (`moe_rotate_i4_pm.rip`) writes, call for call, over the `rot::*` natives. `rotate`, `quantize` and `store`
/// are `mq_i4_a4`'s (the file this kernel shares its arithmetic with); only the prologue is this kernel's.
pub mod blocks {
    use super::*;

    /// One entry: the flat-slot producer (`gp = false`) or its grouped-position twin.
    pub struct Gen {
        spec: Spec,
        gp: bool,
    }

    impl Gen {
        pub fn new(spec: Spec, gp: bool) -> Result<Gen, String> {
            spec.validate()?;
            Ok(Gen { spec, gp })
        }
        pub fn gp(&self) -> bool {
            self.gp
        }
        pub fn plan(&self) -> Result<crate::RegPlan, String> {
            super::plan()
        }
        pub fn kernel_spec(&self) -> KernelSpec {
            let spec = self.spec;
            let kernargs = KernargLayout::new(KERNARG_BYTES)
                .pointer_access("h", 0, Access::ReadOnly)
                .pointer_access("sorted", 8, Access::ReadOnly)
                .pointer_access("xq", 16, Access::WriteOnly)
                .pointer_access("signs1", 24, Access::ReadOnly)
                .pointer_access("signs2", 32, Access::ReadOnly)
                .hidden("K", 40, 4, "by_value")
                .hidden("P", 44, 4, "by_value")
                .hidden("L", 48, 4, "by_value");
            KernelSpec {
                kernel_id: "moe_rotate_i4_pm".into(),
                variant: if self.gp {
                    format!("{}_gp", spec.tag())
                } else {
                    spec.tag().into()
                },
                arch: spec.arch,
                symbol: if self.gp {
                    symbol_gp(spec.arch)
                } else {
                    spec.symbol()
                },
                kernargs,
                user_sgpr_count: 2,
                system_sgpr_workgroup_id_y: true,
                workgroup_size: WORKGROUP_SIZE as u16,
                group_segment_fixed_size: 0,
                wave32: true,
                cu_mode: false,
            }
        }
    }

    /// Kernel arguments, the (row, group) of the workgroup and its exits, the row's slot, the h / xq pointers and
    /// the lane setup and loads.
    pub fn prologue(b: &mut Builder, g: &Gen) -> R {
        super::prologue(b, g.gp)
    }
    /// The FWHT128 rotation (`mq_i4_a4::rotate`); returns the four registers `quantize` reads.
    pub fn rotate(b: &mut Builder) -> Result<[u8; 4], String> {
        a4::rotate(b)
    }
    /// The `{5,7}` two-candidate quantizer.
    pub fn quantize(b: &mut Builder, e: [u8; 4]) -> R {
        a4::quantize(b, e)
    }
    /// The record store (and the exit).
    pub fn store(b: &mut Builder) -> R {
        a4::store(b)
    }
}

/// The module as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Kernel arguments, the (row, group) of the workgroup (exit past `P` rows or `K/128` groups), the row's slot (exit on
/// a pad row), the h / xq pointers, then `mq_i4_a4`'s lane setup and loads.
fn prologue(b: &mut Builder, gp: bool) -> R {
    let a = b.spec.arch;
    let exit = a4::lbl();
    smem(b, a4::ARG_H, 8, a4::KARG, 0)?;
    smem(b, a4::ARG_S2, 4, a4::KARG, 32)?;
    smem(b, ARG_L, 1, a4::KARG, 48)?;
    sop(b, format!("s_mov_b32 s{}, ttmp9", a4::ROW), &[a4::ROW], &[])?;
    sop(
        b,
        format!("s_and_b32 s{}, ttmp7, 0xffff", a4::GRP),
        &[a4::GRP],
        &[],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{}, s{ARG_P}", a4::ROW),
        &[],
        &[a4::ROW, ARG_P],
    )?;
    a4::branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(
        b,
        format!("s_lshr_b32 s{}, s{ARG_K}, 7", a4::T0),
        &[a4::T0],
        &[ARG_K],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{}, s{}", a4::GRP, a4::T0),
        &[],
        &[a4::GRP, a4::T0],
    )?;
    a4::branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // slot = sorted[row]; a pad row writes nothing.
    a4::mad64(b, a4::SORT_PTR, a4::ARG_SORTED, a4::ROW, 4)?;
    smem(b, a4::SLOT, 1, a4::SORT_PTR, 0)?;
    sop(
        b,
        format!("s_cmp_lt_i32 s{}, 0", a4::SLOT),
        &[],
        &[a4::SLOT],
    )?;
    a4::branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // h + row*2K + g*256.
    sop(
        b,
        format!("s_lshl_b32 s{ROWB}, s{ARG_K}, 1"),
        &[ROWB],
        &[ARG_K],
    )?;
    a4::mad64_reg(b, a4::H_PTR, a4::ARG_H, a4::ROW, ROWB)?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{}, 8", a4::T0, a4::GRP),
        &[a4::T0],
        &[a4::GRP],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, s{}",
            super::common::s_add_u32(a),
            a4::H_PTR,
            a4::H_PTR,
            a4::T0
        ),
        &[a4::H_PTR],
        &[a4::H_PTR, a4::T0],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, 0",
            super::common::s_addc_u32(a),
            a4::H_PTR + 1,
            a4::H_PTR + 1
        ),
        &[a4::H_PTR + 1],
        &[a4::H_PTR + 1],
    )?;
    // xq + 72 * (g*L + slot). The record index goes to ROWB (the row pitch is dead by now), not T0: `mad64` clobbers T0
    // with the low product before the high product reads `x`, so `x` must not be T0. With `x = ROWB` the 64-bit product
    // is exact for every u32 index.
    // The grouped-position twin indexes by (P, row): `g * P + row`.
    let (stride, index) = if gp {
        (ARG_P, a4::ROW)
    } else {
        (ARG_L, a4::SLOT)
    };
    sop(
        b,
        format!("s_mul_i32 s{ROWB}, s{}, s{stride}", a4::GRP),
        &[ROWB],
        &[a4::GRP, stride],
    )?;
    sop(
        b,
        format!("{} s{ROWB}, s{ROWB}, s{index}", s_add_i32(a)),
        &[ROWB],
        &[ROWB, index],
    )?;
    a4::mad64(b, a4::XQ_PTR, a4::ARG_XQ, ROWB, RECORD_BYTES)?;
    a4::lane_setup(b)
}

/// The shape contract of the emitted entry (the `mq_i4_a4` contract under this symbol). Committed as
/// `kernels/moe_rotate_i4_pm.gfx1201.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    a4::contract_json(&spec.symbol(), e)
}
