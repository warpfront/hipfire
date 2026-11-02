// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1100 / gfx1201 tile-ownership mask of the Flash-Next
//! owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`):
//! `ep_tile_mask_pm_<arch>`. One integer compare/select over the padded tile
//! capacity, writing a COPY of the grouped-GEMM tile ids in which every tile of
//! an expert this rank does not own is the sentinel `-1` (the symmetric IU4
//! GEMMs return on that sentinel before any weight or activation load). The
//! source `expert_tile_ids` is never modified.
//!
//! ABI (six arguments, 36 bytes): tile_ids `*i32` @0, tile_ids_owned `*i32`
//! @8, owner_table `*i32` @16 (one owner rank per expert), tiles i32 @24,
//! n_experts i32 @28, rank i32 @32. Grid `[ceil(tiles / 256), 1, 1]`, block
//! 256 (one thread per tile), no LDS, no barrier, no scratch. Contract (the
//! caller's checks): `tiles >= 1`, `tile_ids` and `tile_ids_owned` hold `tiles`
//! words, `owner_table` holds `n_experts` words.
//!
//! Law (exact integers; the owner is a table input, never a constant):
//! ```text
//!   live(t) = 0 <= tile_ids[t] < n_experts                    (unsigned compare)
//!   take(t) = live(t) && owner_table[tile_ids[t]] == rank
//!   tile_ids_owned[t] = take(t) ? tile_ids[t] : -1            (t < tiles)
//! ```
//! A negative id (the grouping's padding tail) and an id outside the table map
//! to `-1`.
//!
//! Schedule: lane `t = 256 wg + tid` (clamped to the last tile for the loads,
//! so no load is out of range; the store is exec-masked to `t < tiles`) loads
//! its id, the owner of `live ? id : 0`, and selects with two lane masks.
use super::common::{bload, op, s, smem, sop, sr, srd_tail, v};
use crate::{
    arch::WgIdSource,
    insn::MemoryClass,
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1100, Gfx1201, Target, Workgroup};

type R = Result<(), String>;

/// Threads (tiles) per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 36;

pub fn symbol(arch: Arch) -> String {
    format!("ep_tile_mask_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("ep_tile_mask_pm_{}", arch.name())
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
/// Real user SGPRs (the kernarg pointer) and enabled system SGPRs (workgroup id x).
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 1;
/// gfx11 system SGPR right after the DECLARED user SGPRs: workgroup id x (gfx12 reads `ttmp9`). gfx1100 pads the
/// descriptor to user + system >= 16 (`Arch::declared_user_sgprs`, LLVM FeatureUserSGPRInit16Bug), so the id is `s15` there.
fn wg_x_in(arch: Arch) -> Result<u8, String> {
    match arch
        .workgroup_ids(arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS))?
        .x
    {
        WgIdSource::Sgpr(n) => Ok(n),
        other => Err(format!(
            "ep_tile_mask_pm: no workgroup id SGPR on {}: {other:?}",
            arch.name()
        )),
    }
}
/// The kernarg block `s_load_b256` destination: s[8:15] (ids 8:9, owned 10:11, table 12:13, tiles 14, n_experts 15),
/// except on gfx1100 where s14/s15 belong to the padded user SGPRs / the workgroup id: the block moves to `S_END`.
struct Args {
    ids: u8,
    owned: u8,
    table: u8,
    tiles: u8,
    nexp: u8,
}
fn args(arch: Arch) -> Args {
    let base = if arch.min_user_system_sgprs() > 0 {
        S_END
    } else {
        8
    };
    Args {
        ids: base,
        owned: base + 2,
        table: base + 4,
        tiles: base + 6,
        nexp: base + 7,
    }
}
const RANK: u8 = 16;
const WGX: u8 = 17;
const TLM1: u8 = 18;
const BYTES: u8 = 19;
// Raw buffer descriptors.
const SRD_IDS: u8 = 20;
const SRD_OWNED: u8 = 24;
const SRD_TABLE: u8 = 28;
// Lane-mask pairs (the high word is never written in wave32; M7 reads VOP3 masks as pairs).
const LIVE: u8 = 32;
const EQ: u8 = 34;
const TAKE: u8 = 36;
const VALID: u8 = 38;
const MASKS: [u8; 4] = [LIVE, EQ, TAKE, VALID];
const S_END: u8 = 40;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const VT: u8 = 1;
const VTC: u8 = 2;
const VOFF: u8 = 3;
const VID: u8 = 4;
const VTOFF: u8 = 5;
const VTOFFS: u8 = 6;
const VOWN: u8 = 7;
const VOUT: u8 = 8;
const V_END: u8 = 16;

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("tile_ids", 0, Access::ReadOnly)
        .pointer_access("tile_ids_owned", 8, Access::WriteOnly)
        .pointer_access("owner_table", 16, Access::ReadOnly)
        .hidden("tiles", 24, 4, "by_value")
        .hidden("n_experts", 28, 4, "by_value")
        .hidden("rank", 32, 4, "by_value")
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    // gfx1100: s[2:16) are the (padded) user SGPRs and the workgroup id, the named blocks start at s16 and the
    // relocated kernarg block follows `S_END`.
    let (first, end) = if arch.min_user_system_sgprs() > 0 {
        (16u8, S_END + 8)
    } else {
        (8u8, S_END)
    };
    if !arch.gfx12() {
        p.s::<1>("wg_x_in", wg_x_in(arch)?, Live::Whole)?;
    }
    for base in (first..end).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1100 and gfx1201; every other target is refused).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1201) {
        return Err(format!(
            "ep_tile_mask_pm: built for gfx1100 and gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "ep_tile_mask_pm".into(),
        variant: "owner".into(),
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
        _ => kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, arch)?,
    }
    b.finish()
}

/// The single-symbol module `ep_tile_mask_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn m(x: u8) -> RegRef {
    sr(x, 2)
}
/// `s{dst}:{dst+3}` raw buffer descriptor of the pointer pair at `s{ptr}` with `num_records` from `s{records}`.
fn srd(b: &mut Builder, dst: u8, ptr: u8, records: u8) -> R {
    sop(b, format!("s_mov_b32 s{dst}, s{ptr}"), &[dst], &[ptr])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", dst + 1, ptr + 1),
        &[dst + 1],
        &[ptr + 1],
    )?;
    srd_tail(b, dst, Some(records))
}
/// Raw buffer store of one dword at `v{voff}` (soffset zero: `null` on gfx12, `0` on gfx11).
fn bstore(b: &mut Builder, data: u8, voff: u8, desc: u8) -> R {
    let soffset = if b.spec.arch.gfx12() { "null" } else { "0" };
    super::common::mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{desc}:{}], {soffset} offen",
            desc + 3
        ),
        &[],
        &[v(voff), v(data), sr(desc, 4)],
        MemoryClass::VmemStore,
    )
}

fn s_sub_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Workgroup<'_, T, Builder>, arch: Arch) -> R {
    let end = wg.exit(".Lmask_end")?;
    let b = wg.isa();
    let Args {
        ids,
        owned,
        table,
        tiles,
        nexp,
    } = args(arch);
    smem(b, ids, 8, KARG, 0x00)?;
    smem(b, RANK, 1, KARG, 0x20)?;
    // Workgroup id x: gfx12 `ttmp9`, gfx11 the system SGPR after the user SGPRs.
    if arch.gfx12() {
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    } else {
        let wg_in = wg_x_in(arch)?;
        sop(b, format!("s_mov_b32 s{WGX}, s{wg_in}"), &[WGX], &[wg_in])?;
    }
    for mk in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", mk + 1), &[mk + 1], &[])?
    }
    // Last tile (the load clamp), extents, descriptors.
    sop(
        b,
        format!("{} s{TLM1}, s{tiles}, 1", s_sub_i32(arch)),
        &[TLM1],
        &[tiles],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{BYTES}, s{tiles}, 2"),
        &[BYTES],
        &[tiles],
    )?;
    srd(b, SRD_IDS, ids, BYTES)?;
    srd(b, SRD_OWNED, owned, BYTES)?;
    // The owner table holds n_experts words.
    sop(
        b,
        format!("s_lshl_b32 s{BYTES}, s{nexp}, 2"),
        &[BYTES],
        &[nexp],
    )?;
    srd(b, SRD_TABLE, table, BYTES)?;
    // t = 256 wg + tid; the clamped tile and its byte offset.
    op(
        b,
        format!("v_lshl_add_u32 v{VT}, s{WGX}, 8, v{TID}"),
        &[v(VT)],
        &[s(WGX), v(TID)],
    )?;
    op(
        b,
        format!("v_min_u32_e32 v{VTC}, s{TLM1}, v{VT}"),
        &[v(VTC)],
        &[s(TLM1), v(VT)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF}, 2, v{VTC}"),
        &[v(VOFF)],
        &[v(VTC)],
    )?;
    // id = tile_ids[t].
    bload(b, 1, VID, VOFF, SRD_IDS, 0)?;
    // live = 0 <= id < n_experts (unsigned compare covers negatives).
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{LIVE}, s{nexp}, v{VID}"),
        &[m(LIVE)],
        &[s(nexp), v(VID)],
    )?;
    // owner = owner_table[live ? id : 0].
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VTOFF}, 2, v{VID}"),
        &[v(VTOFF)],
        &[v(VID)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{VTOFFS}, 0, v{VTOFF}, s{LIVE}"),
        &[v(VTOFFS)],
        &[v(VTOFF), m(LIVE)],
    )?;
    bload(b, 1, VOWN, VTOFFS, SRD_TABLE, 0)?;
    // take = live && owner == rank.
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{EQ}, s{RANK}, v{VOWN}"),
        &[m(EQ)],
        &[s(RANK), v(VOWN)],
    )?;
    op(
        b,
        format!("s_and_b32 s{TAKE}, s{LIVE}, s{EQ}"),
        &[m(TAKE)],
        &[m(LIVE), m(EQ)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{VOUT}, -1, v{VID}, s{TAKE}"),
        &[v(VOUT)],
        &[v(VID), m(TAKE)],
    )?;
    // Only lanes of tiles (`t < tiles`) store.
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{VALID}, s{tiles}, v{VT}"),
        &[m(VALID)],
        &[s(tiles), v(VT)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{VALID}"),
        &[],
        &[m(VALID)],
    )?;
    bstore(b, VOUT, VOFF, SRD_OWNED)?;
    b.wait_all()?;
    wg.end(end)
}

/// Shape contract of the emitted kernel (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory and control mnemonics
/// the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b64",
        "s_load_b256",
        "buffer_load_b32",
        "buffer_store_b32",
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
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
