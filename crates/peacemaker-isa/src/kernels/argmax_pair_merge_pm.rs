// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 byte-identical twin of the hipcc entry
//! `argmax_f32_pair_merge` (`kernels/src/argmax_pair_merge.hip`, module
//! `argmax_pair_merge`): `argmax_pair_merge_pm_gfx1201`.
//!
//! ABI (eight arguments, 52 bytes; the exact hipcc kernarg order): local_logits `*f32` @0,
//! local_top1 `*i32` @8, peer_logits `*f32` @16, peer_top1 `*i32` @24, out_top1 `*i32` @32,
//! rows i32 @40, w0 i32 @44, w1 i32 @48. Grid `[ceil(rows / 256), 1, 1]`, block 256 (one
//! thread per row), no LDS, no barrier, no scratch. Contract: `rows`, `w0`, `w1` nonnegative i32; the
//! element index is a full 64-bit `row * w + i` (the hipcc `size_t` math), no width narrowing.
//!
//! Law (per row):
//! ```text
//!   v0 = local_logits[row*w0 + local_top1[row]]     v1 = peer_logits[row*w1 + peer_top1[row]]
//!   f0 = isfinite(v0)   f1 = isfinite(v1)           (NaN and +-inf are not finite)
//!   out_top1[row] = (f0 && (!f1 || v0 >= v1)) ? i0 : (f1 ? w0 + i1 : 0)
//! ```
//! `>=` keeps shard 0 on ties (+0 == -0 included); neither finite gives 0. `out_top1` may
//! alias `local_top1`: both index words are loaded before the row's store. The peer pointers
//! are peer-device (UVA) memory, so every pointer access is a 64-bit-address `global_*`
//! instruction (never a buffer descriptor).
use super::common::{op, s, smem, sop, sr, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1201, Target, Workgroup};

type R = Result<(), String>;

/// Threads (rows) per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 52;

pub fn symbol(arch: Arch) -> String {
    format!("argmax_pair_merge_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("argmax_pair_merge_pm_{}", arch.name())
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// Kernarg block s[8:15]: l0 8:9, i0 10:11, l1 12:13, i1 14:15; s[16:19]: out 16:17, rows 18, w0 19; w1 s20.
const L0: u8 = 8;
const I0: u8 = 10;
const L1: u8 = 12;
const I1: u8 = 14;
const OUT: u8 = 16;
const ROWS: u8 = 18;
const W0: u8 = 19;
const W1: u8 = 20;
const WGX: u8 = 21;
// Lane masks: one SGPR each (wave32 VOP3 convention).
const F0: u8 = 24;
const F1: u8 = 26;
const GE: u8 = 28;
const TAKE: u8 = 30;
const VALID: u8 = 32;
const NF1: u8 = 34;
const CY: u8 = 36;
const S_END: u8 = 40;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const ROW: u8 = 1;
const A0: u8 = 2; // v[2:3] &i0[row]
const IDX0: u8 = 4;
const A1: u8 = 6; // v[6:7] &i1[row]
const IDX1: u8 = 8;
const P0: u8 = 10; // v[10:11] &l0[..]
const P1: u8 = 12; // v[12:13] &l1[..]
const V0: u8 = 14;
const V1: u8 = 15;
const ALT: u8 = 16;
const RES: u8 = 17;
const AO: u8 = 18; // v[18:19] &out[row]
const V_END: u8 = 24;

pub fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("local_logits", 0, Access::ReadOnly)
        .pointer_access("local_top1", 8, Access::ReadOnly)
        .pointer_access("peer_logits", 16, Access::ReadOnly)
        .pointer_access("peer_top1", 24, Access::ReadOnly)
        .pointer_access("out_top1", 32, Access::WriteOnly)
        .hidden("rows", 40, 4, "by_value")
        .hidden("w0", 44, 4, "by_value")
        .hidden("w1", 48, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1201 only).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!(
            "argmax_pair_merge_pm: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "argmax_pair_merge_pm".into(),
        variant: "twin".into(),
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
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, arch)?;
    b.finish()
}

/// The single-symbol module `argmax_pair_merge_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn m(x: u8) -> RegRef {
    s(x)
}
fn p2(x: u8) -> RegRef {
    vr(x, 2)
}
/// `v[dst:dst+1] = base + 4 * (row * w + sext(idx))`, the element index a full 64-bit product
/// (hipcc casts row, w, idx to `size_t`): signed mad over nonnegative row / w, shift, 64-bit add.
fn addr(b: &mut Builder, dst: u8, idx: u8, base: u8, w: u8) -> R {
    let hi = dst + 1;
    op(
        b,
        format!("v_mov_b32_e32 v{dst}, v{idx}"),
        &[v(dst)],
        &[v(idx)],
    )?;
    op(
        b,
        format!("v_ashrrev_i32_e32 v{hi}, 31, v{idx}"),
        &[v(hi)],
        &[v(idx)],
    )?;
    op(
        b,
        format!("v_mad_co_i64_i32 v[{dst}:{hi}], null, v{ROW}, s{w}, v[{dst}:{hi}]"),
        &[p2(dst)],
        &[v(ROW), s(w), p2(dst)],
    )?;
    op(
        b,
        format!("v_lshlrev_b64_e64 v[{dst}:{hi}], 2, v[{dst}:{hi}]"),
        &[p2(dst)],
        &[p2(dst)],
    )?;
    op(
        b,
        format!("v_add_co_u32 v{dst}, s{CY}, s{base}, v{dst}"),
        &[v(dst), m(CY)],
        &[s(base), v(dst)],
    )?;
    op(
        b,
        format!(
            "v_add_co_ci_u32_e64 v{hi}, null, s{}, v{hi}, s{CY}",
            base + 1
        ),
        &[v(hi)],
        &[s(base + 1), v(hi), m(CY)],
    )
}
fn gload(b: &mut Builder, dst: u8, ptr: u8) -> R {
    b.push(
        Instruction::new(
            format!("global_load_b32 v{dst}, v[{ptr}:{}], off", ptr + 1),
            vec![v(dst)],
            vec![p2(ptr)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn gstore(b: &mut Builder, ptr: u8, data: u8) -> R {
    b.push(
        Instruction::new(
            format!("global_store_b32 v[{ptr}:{}], v{data}, off", ptr + 1),
            vec![],
            vec![p2(ptr), v(data)],
        )
        .memory(MemoryClass::VmemStore),
    )
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Workgroup<'_, T, Builder>, arch: Arch) -> R {
    let end = wg.exit(".Lpair_end")?;
    let b = wg.isa();
    smem(b, L0, 8, KARG, 0x00)?;
    smem(b, OUT, 4, KARG, 0x20)?;
    smem(b, W1, 1, KARG, 0x30)?;
    // Workgroup id x on gfx12 is `ttmp9`.
    let _ = arch; // gfx12 workgroup id x is `ttmp9` (the same read as ep_tile_mask)
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    // row = 256 wg + tid; valid lanes `row < rows` (signed, as the hipcc `row >= rows` return).
    op(
        b,
        format!("v_lshl_add_u32 v{ROW}, s{WGX}, 8, v{TID}"),
        &[v(ROW)],
        &[s(WGX), v(TID)],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{VALID}, s{ROWS}, v{ROW}"),
        &[m(VALID)],
        &[s(ROWS), v(ROW)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{VALID}"),
        &[],
        &[m(VALID)],
    )?;
    // i0 = local_top1[row], i1 = peer_top1[row] (both before the store: out may alias local_top1).
    op(
        b,
        format!(
            "v_mad_co_u64_u32 v[{A0}:{}], null, v{ROW}, 4, s[{I0}:{}]",
            A0 + 1,
            I0 + 1
        ),
        &[p2(A0)],
        &[v(ROW), sr(I0, 2)],
    )?;
    op(
        b,
        format!(
            "v_mad_co_u64_u32 v[{A1}:{}], null, v{ROW}, 4, s[{I1}:{}]",
            A1 + 1,
            I1 + 1
        ),
        &[p2(A1)],
        &[v(ROW), sr(I1, 2)],
    )?;
    gload(b, IDX0, A0)?;
    gload(b, IDX1, A1)?;
    // &l0[row*w0 + i0], &l1[row*w1 + i1]
    addr(b, P0, IDX0, L0, W0)?;
    addr(b, P1, IDX1, L1, W1)?;
    gload(b, V0, P0)?;
    gload(b, V1, P1)?;
    // f = isfinite (class mask: -normal|-denorm|-0|+0|+denorm|+normal), ge = v0 >= v1.
    op(
        b,
        format!("v_cmp_class_f32_e64 s{F0}, v{V0}, 0x1f8"),
        &[m(F0)],
        &[v(V0)],
    )?;
    op(
        b,
        format!("v_cmp_class_f32_e64 s{F1}, v{V1}, 0x1f8"),
        &[m(F1)],
        &[v(V1)],
    )?;
    op(
        b,
        format!("v_cmp_ge_f32_e64 s{GE}, v{V0}, v{V1}"),
        &[m(GE)],
        &[v(V0), v(V1)],
    )?;
    // take0 = f0 && (!f1 || ge)
    op(b, format!("s_not_b32 s{NF1}, s{F1}"), &[m(NF1)], &[m(F1)])?;
    op(
        b,
        format!("s_or_b32 s{TAKE}, s{NF1}, s{GE}"),
        &[m(TAKE)],
        &[m(NF1), m(GE)],
    )?;
    op(
        b,
        format!("s_and_b32 s{TAKE}, s{F0}, s{TAKE}"),
        &[m(TAKE)],
        &[m(F0), m(TAKE)],
    )?;
    // alt = f1 ? w0 + i1 : 0 ; res = take0 ? i0 : alt
    op(
        b,
        format!("v_add_nc_u32_e32 v{ALT}, s{W0}, v{IDX1}"),
        &[v(ALT)],
        &[s(W0), v(IDX1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{ALT}, 0, v{ALT}, s{F1}"),
        &[v(ALT)],
        &[v(ALT), m(F1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{RES}, v{ALT}, v{IDX0}, s{TAKE}"),
        &[v(RES)],
        &[v(ALT), v(IDX0), m(TAKE)],
    )?;
    op(
        b,
        format!(
            "v_mad_co_u64_u32 v[{AO}:{}], null, v{ROW}, 4, s[{OUT}:{}]",
            AO + 1,
            OUT + 1
        ),
        &[p2(AO)],
        &[v(ROW), sr(OUT, 2)],
    )?;
    gstore(b, AO, RES)?;
    b.wait_all()?;
    wg.end(end)
}

/// Shape contract of the emitted kernel (`peacemaker-isa emit --co --shape`).
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b128",
        "s_load_b256",
        "global_load_b32",
        "global_store_b32",
        "s_barrier_signal",
        "s_barrier_wait",
        "s_endpgm",
    ] {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
