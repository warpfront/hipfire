// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 balanced-ownership assignment of the
//! Flash-Next expert-parallel exchange (default on; `HIPFIRE_EP_BALANCED_OWNERS=0` disables):
//! `qwen4_ep_owner_assign_pm_gfx1201`, the byte twin of the product kernel
//! contract of `Gpu::qwen4_ep_owner_assign` and of the host reference
//! `qwen4_ep_owner_map` (rdna-compute `moe.rs`). gfx1201 only: every other
//! target is refused (EP is a gfx1201 feature, there is no hipcc fallback).
//!
//! ABI (eleven arguments, 76 bytes): ids `*i32` @0, full_gate_up `*u64` @8,
//! full_down `*u64` @16, live_gate_up `*u64` @24, live_down `*u64` @32,
//! slot_owner `*i32` @40, dummy_gate_up u64 @48, dummy_down u64 @56, slots i32
//! @64, n_ranks i32 @68, self_rank i32 @72. Grid `[1,1,1]`, block 512 (one
//! thread per expert, `n_exp == 512`), 2112 bytes of static LDS, no scratch, no
//! atomics. Contract (the caller's checks): `slots` in `0..=512`, `n_ranks` in
//! `2..=4`, `self_rank < n_ranks`, the full / live tables hold 512 `u64`, `ids`
//! holds `slots` words.
//!
//! Law (exact integers, identical to the host reference):
//! ```text
//!   routed[e] = some slot s < slots has ids[s] == e           (ids outside 0..512 route nothing)
//!   idx(e)    = number of routed experts with id < e
//!   owner(e)  = idx(e) % n_ranks
//!   live_*[e] = routed[e] && owner(e) == self_rank ? full_*[e] : dummy_*     (all 512 entries)
//!   slot_owner[s] = 0 <= ids[s] < 512 ? owner(ids[s]) : -1                   (s < slots)
//! ```
//!
//! Schedule (every wave runs the same code; wave `w` owns experts `32 w ..
//! 32 w + 32`, lane `l` is expert `32 w + l`):
//! 1. Routed mask. The ids are taken 32 per trip (`ceil(slots / 32)` trips,
//!    workgroup uniform) through a raw buffer descriptor, each lane loading
//!    `ids[min(32 trip + l, slots - 1)]` (a clamped duplicate adds nothing to an
//!    idempotent OR); the 32 ids of the trip are broadcast with `v_readlane_b32`
//!    and each is compared with the lane's expert id, the compare masks OR-ed
//!    into one wave-uniform 32-bit routed mask.
//! 2. Prefix. Each wave publishes its mask word (`ds_store_b32`, wave `w` at
//!    byte `4 w` of the 64-byte `masks` region), one workgroup barrier, every wave
//!    reads the sixteen words (`ds_load_b128` x4, broadcast) and sums the
//!    `s_bcnt1_i32_b32` of the waves below its own into a scalar prefix.
//!    `idx(e) = prefix + v_mbcnt_lo_u32_b32(mask)` (the routed lanes below).
//! 3. Owner. `owner = idx % n_ranks` by restoring division: eight conditional
//!    subtracts of `n_ranks << k`, `k = 7..=0` (`idx + (-(n_ranks << k))`, then
//!    unsigned `v_min_u32` keeps the difference only when it did not wrap),
//!    exact for `idx < 2^9` and `n_ranks >= 2`; `idx(e) <= e <= 511`.
//! 4. Tables. `take = routed & (owner == self_rank)` is a lane mask; the two
//!    pointer entries each lane loaded early are selected against the dummy
//!    pointers with `v_cndmask_b32` and written with two `buffer_store_b32` per
//!    table.
//! 5. Slot owners. Each lane publishes `owner(e)` to the 2048-byte `owners`
//!    region (`ds_store_b32` at `64 + 4 e`), second barrier, then lane `s <
//!    slots` reads `owners[ids[s]]` for an in-range id (`-1` otherwise) and
//!    stores it (exec masked to `s < slots`).
//!
//! Static resources: wave32, workgroup 512, two uniform barriers.
use super::common::{bload, lit, mem, op, s, s_add_i32, smem, sop, sr, srd_tail, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{ready, Free, Gfx1201, LdsRegion, Published, Scc, WgUniform, Workgroup};

type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
type R = Result<(), String>;

/// Experts (one thread each).
pub const EXPERTS: u32 = 512;
/// The `masks` LDS region: sixteen wave mask words.
const MASKS_BASE: u32 = 0;
const MASKS_BYTES: u32 = 64;
/// The `owners` LDS region: one owner word per expert.
const OWNERS_BASE: u32 = 64;
const OWNERS_BYTES: u32 = 4 * EXPERTS;
/// Static LDS.
pub const STATIC_LDS: u32 = MASKS_BYTES + OWNERS_BYTES;

pub enum Masks {}
pub enum Owners {}

pub fn symbol(arch: Arch) -> String {
    format!("qwen4_ep_owner_assign_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("qwen4_ep_owner_assign_pm_{}", arch.name())
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg blob: ids 8:9, full_gate_up 10:11, full_down 12:13, live_gate_up 14:15,
// live_down 16:17, slot_owner 18:19, dummy_gate_up 20:21, dummy_down 22:23,
// slots 24, n_ranks 25, self_rank 26.
const IDS: u8 = 8;
const FGU: u8 = 10;
const FDN: u8 = 12;
const LGU: u8 = 14;
const LDN: u8 = 16;
const SOW: u8 = 18;
const DGU: u8 = 20;
const DDN: u8 = 22;
const SLOTS: u8 = 24;
const RANKS: u8 = 25;
const SELF: u8 = 26;
// Raw buffer descriptors.
const SRD_IDS: u8 = 28;
const SRD_FGU: u8 = 32;
const SRD_FDN: u8 = 36;
const SRD_LGU: u8 = 40;
const SRD_LDN: u8 = 44;
const SRD_SOW: u8 = 48;
const WAVE: u8 = 52;
const SLM1: u8 = 53;
const IDSB: u8 = 54;
const TBLB: u8 = 55;
const NEGR: u8 = 56;
const T0: u8 = 57;
const CHUNK: u8 = 58;
const PRE: u8 = 59;
const CNT: u8 = 60;
const TST: u8 = 61;
/// Lane-mask pairs (the high word is never written in wave32; M7 reads VOP3 masks as pairs).
const ACC: u8 = 64;
const TAKE: u8 = 66;
const MEQ: u8 = 68;
const MVAL: u8 = 70;
const MSLOT: u8 = 72;
const MX0: u8 = 74;
const MX1: u8 = 76;
const MASKS: [u8; 7] = [ACC, TAKE, MEQ, MVAL, MSLOT, MX0, MX1];
const X0: u8 = 78;
const X1: u8 = 79;
/// The sixteen wave masks, broadcast to SGPRs.
const SM: u8 = 80;
const S_END: u8 = 96;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const TID4: u8 = 1;
const VLW: u8 = 2;
const VOW: u8 = 3;
const VZ: u8 = 4;
const VT: u8 = 5;
const VID: u8 = 6;
const VOFF8: u8 = 7;
/// full_gate_up[e] 8:9, full_down[e] 10:11, the selected live entries 12:13 / 14:15.
const FG: u8 = 8;
const FD: u8 = 10;
const LG: u8 = 12;
const LD: u8 = 14;
/// The sixteen wave mask words read back from LDS.
const MK: u8 = 16;
const IDX: u8 = 32;
const OWN: u8 = 33;
const TMP: u8 = 34;
const VPRE: u8 = 35;
const OID: u8 = 36;
const OADDR: u8 = 37;
const OLD: u8 = 38;
const RES: u8 = 39;
const VLANE: u8 = 40;
const V_END: u8 = 48;

fn kernargs() -> KernargLayout {
    KernargLayout::new(76)
        .pointer_access("ids", 0, Access::ReadOnly)
        .pointer_access("full_gate_up", 8, Access::ReadOnly)
        .pointer_access("full_down", 16, Access::ReadOnly)
        .pointer_access("live_gate_up", 24, Access::WriteOnly)
        .pointer_access("live_down", 32, Access::WriteOnly)
        .pointer_access("slot_owner", 40, Access::WriteOnly)
        .hidden("dummy_gate_up", 48, 8, "by_value")
        .hidden("dummy_down", 56, 8, "by_value")
        .hidden("slots", 64, 4, "by_value")
        .hidden("n_ranks", 68, 4, "by_value")
        .hidden("self_rank", 72, 4, "by_value")
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

/// The builder-emitted kernel (gfx1201 only; every other target is refused).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!(
            "qwen4_ep_owner_assign_pm: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "qwen4_ep_owner_assign_pm".into(),
        variant: "balanced".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: EXPERTS as u16,
        group_segment_fixed_size: STATIC_LDS,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The single-symbol module `qwen4_ep_owner_assign_pm_<arch>`.
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
fn cmp_wg(wg: &mut Wg, text: String, uses: &[RegRef]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
/// `s{dst} = s{src}` of both dwords of a 64-bit pair, then the descriptor tail
/// (`num_records` from `records`).
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
/// Raw buffer store of one dword at `v{voff} + offset` (soffset zero).
fn bstore(b: &mut Builder, data: u8, voff: u8, desc: u8, offset: u32) -> R {
    let imm = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{desc}:{}], null offen{imm}",
            desc + 3
        ),
        &[],
        &[v(voff), v(data), sr(desc, 4)],
        MemoryClass::VmemStore,
    )
}
/// Raw buffer load of one `u64` at `v{voff}`.
fn bload64(b: &mut Builder, dst: u8, voff: u8, desc: u8) -> R {
    mem(
        b,
        format!(
            "buffer_load_b64 v[{dst}:{}], v{voff}, s[{desc}:{}], null offen",
            dst + 1,
            desc + 3
        ),
        &[vr(dst, 2)],
        &[v(voff), sr(desc, 4)],
        MemoryClass::VmemLoad,
    )
}

// ---- kernel -----------------------------------------------------------------
fn kernel(wg: &mut Wg) -> R {
    let masks = wg.lds::<Masks>("masks", MASKS_BASE, MASKS_BYTES)?;
    let owners = wg.lds::<Owners>("owners", OWNERS_BASE, OWNERS_BYTES)?;
    let end = wg.exit(".Lown_end")?;
    prologue(wg)?;
    routed_mask(wg)?;
    // Early pointer loads: in flight across the mask exchange.
    let b = wg.isa();
    bload64(b, FG, VOFF8, SRD_FGU)?;
    bload64(b, FD, VOFF8, SRD_FDN)?;
    let mp = publish_masks(wg, masks)?;
    owner_of_expert(wg, &mp)?;
    tables(wg)?;
    let op_ = publish_owners(wg, owners)?;
    slot_owners(wg, &op_)?;
    wg.end(end)
}

fn prologue(wg: &mut Wg) -> R {
    let b = wg.isa();
    smem(b, IDS, 8, KARG, 0x00)?;
    smem(b, LDN, 8, KARG, 0x20)?;
    smem(b, SLOTS, 2, KARG, 0x40)?;
    smem(b, SELF, 1, KARG, 0x48)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{TID4}, 2, v{TID}"),
        &[v(TID4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF8}, 3, v{TID}"),
        &[v(VOFF8)],
        &[v(TID)],
    )?;
    op(b, format!("v_mov_b32_e32 v{VZ}, 0"), &[v(VZ)], &[])?;
    op(
        b,
        format!("v_and_b32_e32 v{VLANE}, 31, v{TID}"),
        &[v(VLANE)],
        &[v(TID)],
    )?;
    // Wave index (wave-uniform), its mask-word LDS address, this lane's owner-word LDS address.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
        &[v(VT)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{VT}"),
        &[s(WAVE)],
        &[v(VT)],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 2"), &[T0], &[WAVE])?;
    op(
        b,
        format!("v_mov_b32_e32 v{VLW}, s{T0}"),
        &[v(VLW)],
        &[s(T0)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VOW}, {}, v{TID4}", lit(OWNERS_BASE)),
        &[v(VOW)],
        &[v(TID4)],
    )?;
    for mk in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", mk + 1), &[mk + 1], &[])?
    }
    sop(b, format!("s_mov_b32 s{ACC}, 0"), &[ACC], &[])?;
    // Clamp bound, descriptor extents, descriptors.
    sop(
        b,
        format!("s_sub_co_i32 s{SLM1}, s{SLOTS}, 1"),
        &[SLM1],
        &[SLOTS],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{IDSB}, s{SLOTS}, 2"),
        &[IDSB],
        &[SLOTS],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{TBLB}, {}", lit(8 * EXPERTS)),
        &[TBLB],
        &[],
    )?;
    srd(b, SRD_IDS, IDS, IDSB)?;
    srd(b, SRD_FGU, FGU, TBLB)?;
    srd(b, SRD_FDN, FDN, TBLB)?;
    srd(b, SRD_LGU, LGU, TBLB)?;
    srd(b, SRD_LDN, LDN, TBLB)?;
    srd(b, SRD_SOW, SOW, IDSB)?;
    sop(
        b,
        format!("s_sub_co_i32 s{NEGR}, 0, s{RANKS}"),
        &[NEGR],
        &[RANKS],
    )
}

/// `ACC` = the routed lane mask of this wave's 32 experts.
fn routed_mask(wg: &mut Wg) -> R {
    sop(wg.isa(), format!("s_mov_b32 s{CHUNK}, 0"), &[CHUNK], &[])?;
    let none = cmp_wg(wg, format!("s_cmp_eq_u32 s{SLOTS}, 0"), &[s(SLOTS)])?;
    wg.wg_skip_if(none, ".Lown_routed_done", (), |wg, ()| {
        wg.loop_carried(".Lown_routed", (), |wg, ()| {
            let b = wg.isa();
            // Slot `chunk + lane`, clamped to the last slot (a duplicate id adds nothing).
            op(
                b,
                format!("v_add_nc_u32_e32 v{VT}, s{CHUNK}, v{VLANE}"),
                &[v(VT)],
                &[s(CHUNK), v(VLANE)],
            )?;
            op(
                b,
                format!("v_min_u32_e32 v{VT}, s{SLM1}, v{VT}"),
                &[v(VT)],
                &[s(SLM1), v(VT)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{VT}, 2, v{VT}"),
                &[v(VT)],
                &[v(VT)],
            )?;
            bload(b, 1, VID, VT, SRD_IDS, 0)?;
            for k in 0..32u8 {
                let (x, mk) = if k % 2 == 0 { (X0, MX0) } else { (X1, MX1) };
                op(
                    b,
                    format!("v_readlane_b32 s{x}, v{VID}, {k}"),
                    &[s(x)],
                    &[v(VID)],
                )?;
                op(
                    b,
                    format!("v_cmp_eq_u32_e64 s{mk}, s{x}, v{TID}"),
                    &[m(mk)],
                    &[s(x), v(TID)],
                )?;
                op(
                    b,
                    format!("s_or_b32 s{ACC}, s{ACC}, s{mk}"),
                    &[m(ACC)],
                    &[m(ACC), m(mk)],
                )?;
            }
            add_i(wg.isa(), CHUNK, CHUNK, "32")?;
            let more = cmp_wg(
                wg,
                format!("s_cmp_lt_u32 s{CHUNK}, s{SLOTS}"),
                &[s(CHUNK), s(SLOTS)],
            )?;
            Ok(((), more))
        })
    })
}

fn add_i(b: &mut Builder, d: u8, x: u8, imm: &str) -> R {
    sop(
        b,
        format!("{} s{d}, s{x}, {imm}", s_add_i32(b.spec.arch)),
        &[d],
        &[x],
    )
}

/// Publish this wave's routed mask word; barrier.
fn publish_masks(
    wg: &mut Wg,
    masks: LdsRegion<Masks, Free>,
) -> Result<LdsRegion<Masks, Published>, String> {
    op(
        wg.isa(),
        format!("v_mov_b32_e32 v{VT}, s{ACC}"),
        &[v(VT)],
        &[m(ACC)],
    )?;
    let st = wg.begin_write(masks);
    let (mw, pend) = wg.ds_store(st, ds_st(format!("ds_store_b32 v{VLW}, v{VT}"), VLW, v(VT)))?;
    let d = wg.wait(pend)?;
    let (mp,) = wg.barrier((ready(mw, d),))?;
    Ok(mp)
}

/// `idx(e)` from the sixteen wave masks, then `OWN = idx % n_ranks`.
fn owner_of_expert(wg: &mut Wg, mp: &LdsRegion<Masks, Published>) -> R {
    for k in 0..4u32 {
        let dst = MK + 4 * k as u8;
        let off = if k == 0 {
            String::new()
        } else {
            format!(" offset:{}", 16 * k)
        };
        wg.ds_load(
            mp,
            ds_ld(
                format!("ds_load_b128 v[{dst}:{}], v{VZ}{off}", dst + 3),
                vr(dst, 4),
                VZ,
            ),
        )?;
    }
    let b = wg.isa();
    // prefix = routed experts of the waves below this one.
    sop(b, format!("s_mov_b32 s{PRE}, 0"), &[PRE], &[])?;
    for k in 0..16u8 {
        op(
            b,
            format!("v_readfirstlane_b32 s{}, v{}", SM + k, MK + k),
            &[s(SM + k)],
            &[v(MK + k)],
        )?;
        sop(
            b,
            format!("s_bcnt1_i32_b32 s{CNT}, s{}", SM + k),
            &[CNT],
            &[SM + k],
        )?;
        sop(b, format!("s_cmp_gt_u32 s{WAVE}, {k}"), &[], &[WAVE])?;
        sop(
            b,
            format!("s_cselect_b32 s{CNT}, s{CNT}, 0"),
            &[CNT],
            &[CNT],
        )?;
        sop(
            b,
            format!("s_add_co_i32 s{PRE}, s{PRE}, s{CNT}"),
            &[PRE],
            &[PRE, CNT],
        )?;
    }
    op(
        b,
        format!("v_mov_b32_e32 v{VPRE}, s{PRE}"),
        &[v(VPRE)],
        &[s(PRE)],
    )?;
    op(
        b,
        format!("v_mbcnt_lo_u32_b32 v{IDX}, s{ACC}, v{VPRE}"),
        &[v(IDX)],
        &[m(ACC), v(VPRE)],
    )?;
    // owner = idx % n_ranks: restoring division, idx + (-(ranks << k)) kept when it did not wrap.
    op(
        b,
        format!("v_mov_b32_e32 v{OWN}, v{IDX}"),
        &[v(OWN)],
        &[v(IDX)],
    )?;
    for k in (0..8u32).rev() {
        sop(
            b,
            format!("s_lshl_b32 s{TST}, s{NEGR}, {k}"),
            &[TST],
            &[NEGR],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{TMP}, s{TST}, v{OWN}"),
            &[v(TMP)],
            &[s(TST), v(OWN)],
        )?;
        op(
            b,
            format!("v_min_u32_e32 v{OWN}, v{OWN}, v{TMP}"),
            &[v(OWN)],
            &[v(OWN), v(TMP)],
        )?;
    }
    Ok(())
}

/// Rewrite this lane's live gate/up and down entries.
fn tables(wg: &mut Wg) -> R {
    let b = wg.isa();
    // take = routed && owner == self_rank.
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MEQ}, s{SELF}, v{OWN}"),
        &[m(MEQ)],
        &[s(SELF), v(OWN)],
    )?;
    op(
        b,
        format!("s_and_b32 s{TAKE}, s{ACC}, s{MEQ}"),
        &[m(TAKE)],
        &[m(ACC), m(MEQ)],
    )?;
    for (dst, full, dummy) in [(LG, FG, DGU), (LD, FD, DDN)] {
        for half in 0..2u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, s{}", dst + half, dummy + half),
                &[v(dst + half)],
                &[s(dummy + half)],
            )?;
            op(
                b,
                format!(
                    "v_cndmask_b32_e64 v{}, v{}, v{}, s{TAKE}",
                    dst + half,
                    dst + half,
                    full + half
                ),
                &[v(dst + half)],
                &[v(dst + half), v(full + half), m(TAKE)],
            )?;
        }
    }
    bstore(b, LG, VOFF8, SRD_LGU, 0)?;
    bstore(b, LG + 1, VOFF8, SRD_LGU, 4)?;
    bstore(b, LD, VOFF8, SRD_LDN, 0)?;
    bstore(b, LD + 1, VOFF8, SRD_LDN, 4)
}

/// Publish `owner(e)` of every expert; barrier.
fn publish_owners(
    wg: &mut Wg,
    owners: LdsRegion<Owners, Free>,
) -> Result<LdsRegion<Owners, Published>, String> {
    let st = wg.begin_write(owners);
    let (ow, pend) = wg.ds_store(
        st,
        ds_st(format!("ds_store_b32 v{VOW}, v{OWN}"), VOW, v(OWN)),
    )?;
    let d = wg.wait(pend)?;
    let (op_,) = wg.barrier((ready(ow, d),))?;
    Ok(op_)
}

/// `slot_owner[s] = owner(ids[s])` (`-1` out of range) for `s < slots`.
fn slot_owners(wg: &mut Wg, op_: &LdsRegion<Owners, Published>) -> R {
    let b = wg.isa();
    // This lane's slot id, the slot clamped to the last one.
    op(
        b,
        format!("v_min_u32_e32 v{VT}, s{SLM1}, v{TID}"),
        &[v(VT)],
        &[s(SLM1), v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VT}, 2, v{VT}"),
        &[v(VT)],
        &[v(VT)],
    )?;
    bload(b, 1, OID, VT, SRD_IDS, 0)?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MVAL}, {}, v{OID}", lit(EXPERTS)),
        &[m(MVAL)],
        &[v(OID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{OADDR}, 2, v{OID}"),
        &[v(OADDR)],
        &[v(OID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{OADDR}, {}, v{OADDR}", lit(OWNERS_BASE)),
        &[v(OADDR)],
        &[v(OADDR)],
    )?;
    op(
        b,
        format!(
            "v_cndmask_b32_e64 v{OADDR}, {}, v{OADDR}, s{MVAL}",
            lit(OWNERS_BASE)
        ),
        &[v(OADDR)],
        &[v(OADDR), m(MVAL)],
    )?;
    wg.ds_load(
        op_,
        ds_ld(format!("ds_load_b32 v{OLD}, v{OADDR}"), v(OLD), OADDR),
    )?;
    let b = wg.isa();
    op(
        b,
        format!("v_cndmask_b32_e64 v{RES}, -1, v{OLD}, s{MVAL}"),
        &[v(RES)],
        &[v(OLD), m(MVAL)],
    )?;
    // Only lanes of slots (`tid < slots`) store.
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MSLOT}, s{SLOTS}, v{TID}"),
        &[m(MSLOT)],
        &[s(SLOTS), v(TID)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{MSLOT}"),
        &[],
        &[m(MSLOT)],
    )?;
    bstore(b, RES, TID4, SRD_SOW, 0)?;
    b.wait_all()
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
        "buffer_load_b64",
        "buffer_store_b32",
        "ds_store_b32",
        "ds_load_b32",
        "ds_load_b128",
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
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_add_*", "ds_sub_*", "ds_rsub_*", "ds_inc_*", "ds_dec_*", "ds_min_*", "ds_max_*", "ds_and_*", "ds_or_*", "ds_xor_*", "ds_mskor_*", "ds_cmpst*", "ds_cmpstore*", "ds_wrxchg*", "ds_append", "ds_consume", "ds_gws_*", "ds_bpermute*", "ds_permute*", "v_writelane*", "s_setpc*", "s_swappc*"],
        "vgpr_max": 48, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": STATIC_LDS,
    })
}
