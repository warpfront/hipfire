// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Kind, RegRef};
use crate::sync::Scope;
use crate::Builder;

fn r(kind: Kind, base: u8, len: u8) -> RegRef {
    RegRef { kind, base, len }
}
fn s(base: u8) -> RegRef {
    r(Kind::S, base, 1)
}
fn sp(base: u8) -> RegRef {
    r(Kind::S, base, 2)
}
fn sq(base: u8) -> RegRef {
    r(Kind::S, base, 4)
}
fn v(base: u8) -> RegRef {
    r(Kind::V, base, 1)
}
fn vq(base: u8) -> RegRef {
    r(Kind::V, base, 4)
}
fn op(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
fn load(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(MemoryClass::SmemLoad))?;
    // Each route-dependent branch must also be safe on the non-fallthrough CFG path.
    b.wait_all()
}
// Route outputs are mutable GPU data, not kernargs: use VMEM, whose caches
// participate in the pool's predecessor-event acquire (global_inv). Load once
// per wave through lane 0, then broadcast into the replicated scalar scan.
fn route_load(b: &mut Builder, dst: u8, offset: u32) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    op(b, &format!("v_mov_b32_e32 v1, {offset:#x}"), &[v(1)], &[])?;
    b.push(
        Instruction::new(
            "global_load_b32 v2, v1, s[26:27]",
            vec![v(2)],
            vec![v(1), sp(26)],
        )
        .memory(MemoryClass::VmemLoad),
    )?;
    b.wait_all()?;
    op(
        b,
        &format!("v_readfirstlane_b32 s{dst}, v2"),
        &[s(dst)],
        &[v(2)],
    )?;
    full_exec(b)
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn full_exec(b: &mut Builder) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

/// EXPAND_LIVE task body, inside the caller's `Builder::region`.
///
/// `s[0:1]` is the 64-byte argument block; `v0` is the flat workgroup thread
/// index. Both are read-only. Scratch is s4..s30 and v1..v7; no LDS or dispatcher
/// registers are used. There must be exactly one producer workgroup per span.
/// Every wave scans the same immutable route, carrying an identical prefix;
/// flat thread indices partition the x coordinates, not the prefix calculation.
/// Thus neither ordering nor output bytes depend on wave execution order.
///
/// An undersized capacity never causes an out-of-bounds store: only its prefix
/// is written, but the full count is published so the consumer's count check
/// poisons the span. A valid invocation supplies capacity >= the live count.
pub fn expand_body(b: &mut Builder, block: u32, name: &str) -> Result<(), String> {
    if !matches!(block, 32 | 64 | 128 | 256) {
        return Err(format!("EXPAND_LIVE: unsupported block size {block}"));
    }
    load(b, "s_load_b128 s[4:7], s[0:1], 0x0", &[sq(4)], &[sp(0)])?;
    load(b, "s_load_b128 s[8:11], s[0:1], 0x10", &[sq(8)], &[sp(0)])?;
    load(b, "s_load_b128 s[12:15], s[0:1], 0x20", &[sq(12)], &[sp(0)])?;
    load(b, "s_load_b64 s[16:17], s[0:1], 0x30", &[sp(16)], &[sp(0)])?;
    // s29 = number of x workgroups that have any valid output rows.
    op(b, "s_add_co_u32 s29, s12, 15", &[s(29)], &[s(12)])?;
    op(b, "s_lshr_b32 s29, s29, 4", &[s(29)], &[s(29)])?;
    op(b, "s_min_u32 s29, s29, s15", &[s(29)], &[s(29), s(15)])?;
    op(b, "s_mov_b32 s18, 0", &[s(18)], &[])?; // y
    op(b, "s_mov_b32 s19, 0", &[s(19)], &[])?; // output prefix
    let head = format!(".L{name}_pairs");
    let next = format!(".L{name}_next_pair");
    let emit = format!(".L{name}_emit_pair");
    let xs = format!(".L{name}_xs");
    let xdone = format!(".L{name}_xs_done");
    let publish = format!(".L{name}_publish_count");
    b.loop_(&head, |b| {
        op(b, "s_cmp_ge_u32 s18, s16", &[], &[s(18), s(16)])?;
        branch(b, &format!("s_cbranch_scc1 {publish}"))?;
        op(b, "s_cmp_eq_u32 s29, 0", &[], &[s(29)])?;
        branch(b, &format!("s_cbranch_scc1 {publish}"))?;
        op(b, "s_lshl_b32 s20, s18, 1", &[s(20)], &[s(18)])?;
        op(b, "s_cmp_ge_u32 s20, s13", &[], &[s(20), s(13)])?;
        branch(b, &format!("s_cbranch_scc1 {publish}"))?;
        // expert_tile_ids[(2*y) >> 4]. Unsigned >= E also rejects negative
        // signed expert ids because the valid domain has 0 <= E <= INT_MAX.
        op(b, "s_lshr_b32 s21, s20, 4", &[s(21)], &[s(20)])?;
        op(b, "s_lshl_b32 s21, s21, 2", &[s(21)], &[s(21)])?;
        op(b, "s_add_co_u32 s26, s4, s21", &[s(26)], &[s(4), s(21)])?;
        op(b, "s_add_co_ci_u32 s27, s5, 0", &[s(27)], &[s(5)])?;
        route_load(b, 22, 0)?;
        op(b, "s_cmp_ge_u32 s22, s14", &[], &[s(22), s(14)])?;
        branch(b, &format!("s_cbranch_scc1 {next}"))?;
        op(b, "s_lshl_b32 s25, s20, 2", &[s(25)], &[s(20)])?;
        op(b, "s_add_co_u32 s26, s6, s25", &[s(26)], &[s(6), s(25)])?;
        op(b, "s_add_co_ci_u32 s27, s7, 0", &[s(27)], &[s(7)])?;
        route_load(b, 23, 0)?;
        op(b, "s_cmp_ge_i32 s23, 0", &[], &[s(23)])?;
        branch(b, &format!("s_cbranch_scc1 {emit}"))?;
        op(b, "s_add_co_u32 s25, s20, 1", &[s(25)], &[s(20)])?;
        op(b, "s_cmp_ge_u32 s25, s13", &[], &[s(25), s(13)])?;
        branch(b, &format!("s_cbranch_scc1 {next}"))?;
        route_load(b, 24, 4)?;
        op(b, "s_cmp_lt_i32 s24, 0", &[], &[s(24)])?;
        branch(b, &format!("s_cbranch_scc1 {next}"))?;
        b.label(&emit)?;
        op(b, "s_mov_b32 s30, 0", &[s(30)], &[])?;
        b.loop_(&xs, |b| {
            op(b, "s_cmp_ge_u32 s30, s29", &[], &[s(30), s(29)])?;
            branch(b, &format!("s_cbranch_scc1 {xdone}"))?;
            full_exec(b)?;
            op(b, "v_add_nc_u32_e32 v1, s30, v0", &[v(1)], &[s(30), v(0)])?;
            op(b, "v_add_nc_u32_e32 v3, s19, v1", &[v(3)], &[s(19), v(1)])?;
            op(b, "v_lshlrev_b32_e32 v2, 4, v3", &[v(2)], &[v(3)])?;
            op(b, "v_mov_b32_e32 v4, v1", &[v(4)], &[v(1)])?;
            op(b, "v_mov_b32_e32 v5, s18", &[v(5)], &[s(18)])?;
            op(b, "v_mov_b32_e32 v6, 0", &[v(6)], &[])?;
            op(b, "v_mov_b32_e32 v7, 0", &[v(7)], &[])?;
            op(b, "v_cmpx_gt_u32_e32 s29, v1", &[], &[s(29), v(1)])?;
            op(b, "v_cmpx_gt_u32_e32 s17, v3", &[], &[s(17), v(3)])?;
            b.push(
                Instruction::new(
                    "global_store_b128 v2, v[4:7], s[8:9]",
                    vec![],
                    vec![v(2), vq(4), sp(8)],
                )
                .memory(MemoryClass::VmemStore),
            )?;
            b.wait_all()?;
            full_exec(b)?;
            op(
                b,
                &format!("s_add_co_u32 s30, s30, {block:#x}"),
                &[s(30)],
                &[s(30)],
            )?;
            branch(b, &format!("s_branch {xs}"))
        })?;
        b.label(&xdone)?;
        op(b, "s_add_co_u32 s19, s19, s29", &[s(19)], &[s(19), s(29)])?;
        b.label(&next)?;
        op(b, "s_add_co_u32 s18, s18, 1", &[s(18)], &[s(18)])?;
        branch(b, &format!("s_branch {head}"))
    })?;
    b.label(&publish)?;
    full_exec(b)?;
    // The count store must not overtake any wave's entries. Each wave releases
    // its stores first; the convergent workgroup barrier then precedes tid0's
    // publication, including for empty routes and waves with no active x lane.
    b.release(Scope::Dev)?;
    if block > 32 {
        b.barrier(&[])?;
    }
    op(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[v(0)])?;
    op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(b, "v_mov_b32_e32 v2, s19", &[v(2)], &[s(19)])?;
    b.push(
        Instruction::new(
            "global_store_b32 v1, v2, s[10:11]",
            vec![],
            vec![v(1), v(2), sp(10)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.release(Scope::Dev)?;
    full_exec(b)?;
    b.wait_all()
}
