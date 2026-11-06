// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EXPAND_LIVE (P0 template 1) wrapper for gfx1201: a standalone kernel whose only
//! content is the native task-body region [`expand_body`] (`p0_expand_gfx1201/body.rs`).
//! The same region is what a P0 pool hosts, so the wrapper reproduces exactly the pool's
//! region entry state and nothing else.
//!
//! Scope: M0, `o4` tid 16 only. `dni` (tid 18) stays a static grid because its invalid
//! experts zero-fill.
//!
//! # Kernarg ABI ([`KERNARG_BYTES`] = 64)
//! `u64` pointers: `expert_tile_ids` +0 (RO), `sorted_slot_index` +8 (RO),
//! `live_list` +16 (WO), `count_cell` +24 (WO). `u32`: `M` +32, `grouped_rows` +36,
//! `n_experts` +40, `gx` +44, `gy` +48, `capacity` +52, reserved +56 and +60.
//!
//! # Valid domain
//! `gx = ceil(M / 16)`, `gy = ceil(grouped_rows / 2)`, grid z = 1; `capacity <= gx * gy`
//! and large enough for the actual live list. The list holds stride-16-byte `u32`
//! entries `{vx, vy, vz = 0, pad = 0}` ordered y ascending, then x ascending (the
//! implicit `vcount` is 1 in the host header). A pair `y` is live iff `2*y < grouped_rows`,
//! `expert_tile_ids[(2*y) >> 4]` is signed `>= 0` and `< n_experts`, and at least one of
//! sorted slots `2*y` or (in range) `2*y+1` is signed `>= 0`; an `x` is live iff
//! `16*x < M`. Mixed live/pad pairs emit. `count_cell` is a `u32`; entries are released
//! before the final count store, and the final count is released.
//!
//! Fixed ABI contract (not validated at runtime; real M0 is well inside): `M` and
//! `n_experts` are `<= i32::MAX` (the hipcc kernel's `int`s); `grouped_rows <=
//! u32::MAX / 4` (byte indexing); `gx * gy <= u32::MAX / 16` (32-bit LiveList byte
//! offsets); `expert_tile_ids` covers `ceil(grouped_rows / 16)` ids and
//! `sorted_slot_index` covers `grouped_rows` slots; the four pointers' resources do
//! not overlap. Undersized `capacity`: no entry is written out of bounds, but the full
//! live count is still published, so a consumer's poison check on the count only holds
//! when the actual count is `<= capacity`.
//!
//! # Native region ABI
//! Inputs: `s[0:1]` the kernarg pointer, `v0` the FLAT thread id (read-only); no
//! workgroup-id input (the region is single-task; the kernel is launched as one
//! workgroup of [`emit`]'s `block` threads, grid 1). The body owns `s2..s39` and
//! `v1..v7`, no LDS. It retires all memory, restores full EXEC, executes a barrier after
//! every wave's record release and before flat-tid-0's count publication, and contains no
//! `s_endpgm`.
//!
//! # Wrapper
//! Establishes the flat `v0` exactly as the lifted pool does for every region step:
//! `s2 = readfirstlane(v0_hw)` (the wave's first flat tid), then
//! `v0 = mbcnt_lo(-1, 0) + s2`; `s2` is body-local after the entry. After the region it
//! restores EXEC, releases at device scope and ends. The finished program is verified by
//! the strict CFG wait check (`ledger_cfg::verify_emitted`, body included).
//!
//! Valid `block`: 32, 64, 128, 256 (1-D workgroup, wave32).

use crate::insn::Instruction;
use crate::plan::{region_entry_label, region_exit_label, Access, RegionAbi};
use crate::reg::{Kind, Live, RegRef};
use crate::sync::Scope;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

mod body;
pub use body::expand_body;

pub const TEMPLATE_ID: u32 = 1;
pub const SYMBOL: &str = "pm_expand_live_gfx1201";
pub const KERNARG_BYTES: u32 = 64;
/// The body's SGPR footprint (s0..s39) and VGPR footprint (v0..v7).
pub const SGPRS: u32 = 40;
pub const VGPRS: u32 = 8;
pub const LDS_BYTES: u32 = 0;
/// The region's name (entry/exit labels are `.Lexpand_entry`/`.Lexpand_exit`; the body is handed the label prefix `.Lexpand`).
pub const REGION: &str = "expand";

fn s(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 1,
    }
}
fn v(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    }
}
fn sp(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 2,
    }
}
fn op(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}

/// Split `[from, to)` into aligned pieces of at most 8 registers (as the pool does).
fn add_pieces(
    regs: &mut RegPlan,
    name: &str,
    kind: Kind,
    from: u8,
    to: u8,
    live: &Live,
) -> Result<(), String> {
    let mut at = from;
    while at < to {
        let mut piece = 8u8;
        while piece > 1 && (at % piece != 0 || at + piece > to) {
            piece /= 2;
        }
        regs.add_range(&format!("{name}{at}"), kind, at, piece, live.clone())?;
        at += piece;
    }
    Ok(())
}

/// The kernel spec of the standalone wrapper at workgroup size `block`.
pub fn kernel_spec(block: u32) -> KernelSpec {
    KernelSpec {
        kernel_id: "pm_expand_live".into(),
        variant: format!("gfx1201_b{block}"),
        arch: Arch::Gfx1201,
        symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(KERNARG_BYTES)
            .pointer_access("expert_tile_ids", 0, Access::ReadOnly)
            .pointer_access("sorted_slot_index", 8, Access::ReadOnly)
            .pointer_access("live_list", 16, Access::WriteOnly)
            .pointer_access("count_cell", 24, Access::WriteOnly)
            .hidden("M", 32, 4, "by_value")
            .hidden("grouped_rows", 36, 4, "by_value")
            .hidden("n_experts", 40, 4, "by_value")
            .hidden("gx", 44, 4, "by_value")
            .hidden("gy", 48, 4, "by_value")
            .hidden("capacity", 52, 4, "by_value")
            .hidden("reserved0", 56, 4, "by_value")
            .hidden("reserved1", 60, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: block as u16,
        group_segment_fixed_size: LDS_BYTES,
        wave32: true,
        cu_mode: false,
    }
}

/// The standalone EXPAND_LIVE kernel at workgroup size `block` (32, 64, 128 or 256).
pub fn emit(block: u32) -> Result<Emitted, String> {
    if !matches!(block, 32 | 64 | 128 | 256) {
        return Err(format!(
            "expand_live: block must be 32, 64, 128 or 256, got {block}"
        ));
    }
    let name = REGION;
    let (entry, exit) = (region_entry_label(name), region_exit_label(name));
    let mut regs = RegPlan::new(VGPRS as u16, SGPRS as u16)?;
    // Caller-owned through the region: the kernarg pointer and the flat lane id.
    regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)?;
    regs.add_range("lane", Kind::V, 0, 1, Live::Whole)?;
    // The wave base is a temporary of the prologue; its SGPR is body-local afterwards.
    regs.add_range(
        "wavebase",
        Kind::S,
        2,
        1,
        Live::Between("entry".into(), entry.clone()),
    )?;
    let inside = Live::Between(entry, exit);
    add_pieces(&mut regs, "body_s", Kind::S, 2, SGPRS as u8, &inside)?;
    add_pieces(&mut regs, "body_v", Kind::V, 1, VGPRS as u8, &inside)?;
    let mut b = Builder::new(kernel_spec(block), regs);
    b.enable_valu_schedule()?;
    // flat tid = wave first tid (hardware v0, lane 0) + lane index in the wave
    op(&mut b, "v_readfirstlane_b32 s2, v0", &[s(2)], &[v(0)])?;
    op(&mut b, "v_mbcnt_lo_u32_b32 v0, -1, 0", &[v(0)], &[])?;
    op(
        &mut b,
        "v_add_nc_u32_e32 v0, s2, v0",
        &[v(0)],
        &[v(0), s(2)],
    )?;
    b.wait_all()?;
    let abi = RegionAbi {
        name: name.into(),
        inputs: vec![("args".into(), sp(0)), ("lane".into(), v(0))],
        lds_bytes: LDS_BYTES,
        workgroup_size: block as u16,
    };
    b.region(&abi, |b| expand_body(b, block, ".Lexpand"))?;
    b.regions
        .last_mut()
        .expect("region records a proof")
        .template_id = Some(TEMPLATE_ID);
    op(&mut b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    b.release(Scope::Dev)?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    let emitted = b.finish()?;
    crate::ledger_cfg::verify_emitted(&emitted)
        .map_err(|e| format!("{SYMBOL}: CFG wait verification failed: {e}"))?;
    Ok(emitted)
}
