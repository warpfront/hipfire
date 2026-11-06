// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab-only gfx1201 persistent pool dispatcher hosting lifted hipcc bodies
//! (`peacemaker_lift::rewrite::region`) as regions, with MegaM0's P0 task ABI
//! (`release-0.4.5/megam0/p0h/m0_rt.h`); nothing here is on a product route.
//!
//! Kernarg: `M0Params` (72 bytes): `tasks, ticket, events, kernargs, trace, poison`
//! pointers at 0..48, `spin_ticks` at 48 (its low dword is the poll budget of a
//! predecessor wait), `n_tasks` at 56, `trace_on` at 60. A task is a 128-byte
//! `M0TaskDesc`: `tmpl, vx, vy, vz, gx, gy, gz, ka_off, pred_event, expected,
//! done_event, vcount`. Every workgroup (the hosted bodies' `blockDim`) loops:
//!
//! 1. wave 0 lane 0 claims `idx = ticket++` (relaxed); past `n_tasks` it stops;
//!    it loads the descriptor and, when `pred_event != ~0`, spins until
//!    `events[pred] >= expected` (acquire; on poll-budget exhaustion it sets
//!    `*poison = 1` and stops); `t_ready` is read here;
//! 2. the claim is broadcast through an LDS slot above every hosted body's LDS
//!    (two barriers; single-wave pools broadcast in registers); every wave
//!    acquires at device scope;
//! 3. for `vcount` (0 = 1) steps the template's region runs at virtual workgroup
//!    `(vx + step, vy, vz)` with `s[0:1] = kernargs + ka_off` (`t_start` before,
//!    `t_end` after); multi-wave pools barrier before every step, so no wave
//!    reuses the body's LDS while another still reads the previous step's;
//! 4. every wave releases, the workgroup barriers, and wave 0 lane 0 increments
//!    `events[done_event]` with release ordering (unless `~0`), reads
//!    `t_publish` and, with `trace_on`, writes the `M0Trace` record at `idx`.

use crate::insn::{Instruction, MemoryClass};
use crate::lds::Transition;
use crate::plan::{region_entry_label, region_exit_label, Access, RegionAbi};
use crate::reg::{Kind, Live, RegRef};
use crate::sync::{AtomicOp, GlobalAddr, Ordering, Scope, Watchdog};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, S, V};
use peacemaker_lift::rewrite::region::LiftedRegion;

pub const KERNARG_BYTES: u32 = 72;
pub const TASK_BYTES: u32 = 128;
pub const TRACE_BYTES: u32 = 48;
/// Highest SGPR count a hosted body may use (the dispatcher owns s72..s103).
pub const BODY_SGPR_LIMIT: u16 = 72;
/// The SGPRs that carry the virtual workgroup id (`ttmp9`, `ttmp7`) into a region.
pub const WG_X: u16 = 102;
pub const WG_YZ: u16 = 103;

/// One hosted template.
pub struct Hosted<'a> {
    pub tmpl: u32,
    pub region: &'a LiftedRegion,
}

// Dispatcher SGPRs (all above BODY_SGPR_LIMIT).
const GX: u8 = 72;
const PTR_A: u8 = 74;
const T_PUBLISH: u8 = 76;
const PTR_B: u8 = 78;
const HW_WG: u8 = 80;
const T_START: u8 = 82;
const T_END: u8 = 84;
const T_READY: u8 = 86;
const PARAMS: u8 = 88;
const WAVEBASE: u8 = 90;
const TEMP: u8 = 91;
const N_TASKS: u8 = 92;
const TRACE_ON: u8 = 93;
const IDX: u8 = 94;
const TMPL: u8 = 95;
const VX: u8 = 96;
const VYZ: u8 = 97;
const KA_OFF: u8 = 98;
const DONE: u8 = 99;
const VCOUNT: u8 = 100;
const STEP: u8 = 101;

fn s(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
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
fn v(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    }
}
fn vq(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 4,
    }
}
fn op(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
fn smem(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(MemoryClass::SmemLoad))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn param(b: &mut Builder, dst: u8, offset: u32) -> Result<(), String> {
    smem(
        b,
        &format!(
            "s_load_b64 s[{dst}:{}], s[{PARAMS}:{}], {offset:#x}",
            dst + 1,
            PARAMS + 1
        ),
        &[sp(dst)],
        &[sp(PARAMS)],
    )
}
fn readfirst(b: &mut Builder, dst: u8, src: u8) -> Result<(), String> {
    op(
        b,
        &format!("v_readfirstlane_b32 s{dst}, v{src}"),
        &[s(dst)],
        &[v(src)],
    )
}
fn vmov_s(b: &mut Builder, dst: u8, src: u8) -> Result<(), String> {
    op(
        b,
        &format!("v_mov_b32_e32 v{dst}, s{src}"),
        &[v(dst)],
        &[s(src)],
    )
}
fn exec_lane0(b: &mut Builder) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}
fn exec_all(b: &mut Builder) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

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

/// The pool dispatcher `symbol` of workgroup size `block` hosting `hosted`, and
/// with `copy = Some(tmpl)` the PM-emitted COPY template (`copy_body`).
pub fn pool(
    symbol: &str,
    block: u32,
    hosted: &[Hosted],
    copy: Option<u32>,
) -> Result<Emitted, String> {
    if hosted.is_empty() && copy.is_none() {
        return Err("a pool hosts at least one template".into());
    }
    if let Some(c) = copy {
        if hosted.iter().any(|h| h.tmpl == c) {
            return Err(format!("tmpl {c} is both COPY and a lifted body"));
        }
    }
    for h in hosted {
        let r = h.region;
        if r.block != block {
            return Err(format!(
                "{}: blockDim {} != pool {block}",
                r.symbol, r.block
            ));
        }
        if r.sgprs > BODY_SGPR_LIMIT {
            return Err(format!(
                "{}: {} SGPRs exceed the hosted limit {BODY_SGPR_LIMIT}",
                r.symbol, r.sgprs
            ));
        }
        if r.receipt
            .input_rewrites
            .iter()
            .any(|(_, sub, _)| sub != &format!("s{WG_X}") && sub != &format!("s{WG_YZ}"))
        {
            return Err(format!(
                "{}: converted with other workgroup-id SGPRs than s{WG_X}/s{WG_YZ}",
                r.symbol
            ));
        }
    }
    let multi = block > 32;
    let body_lds = hosted.iter().map(|h| h.region.lds_bytes).max().unwrap_or(0);
    let slot_base = body_lds.next_multiple_of(16);
    let lds_total = if multi { slot_base + 64 } else { body_lds };
    let step_label = |t: u32| format!(".Lt{t}_step");
    let first_step = step_label(
        hosted
            .first()
            .map_or_else(|| copy.expect("copy"), |h| h.tmpl),
    );
    let mut regs = RegPlan::new(256, 104)?;
    // entry: the hardware kernarg pointer and thread id, consumed before the loop
    regs.add_range(
        "hw_kernarg",
        Kind::S,
        0,
        2,
        Live::Between("entry".into(), ".Lnext".into()),
    )?;
    regs.add_range(
        "hw_tid",
        Kind::V,
        0,
        1,
        Live::Between("entry".into(), ".Lnext".into()),
    )?;
    for (name, base, len) in [
        ("gx", GX, 1),
        ("ptr_a", PTR_A, 2),
        ("t_publish", T_PUBLISH, 2),
        ("ptr_b", PTR_B, 2),
        ("hw_wg", HW_WG, 1),
        ("t_start", T_START, 2),
        ("t_end", T_END, 2),
        ("t_ready", T_READY, 2),
        ("params", PARAMS, 2),
        ("wavebase", WAVEBASE, 1),
        ("temp", TEMP, 1),
        ("n_tasks", N_TASKS, 1),
        ("trace_on", TRACE_ON, 1),
        ("idx", IDX, 1),
        ("tmpl", TMPL, 1),
        ("vx", VX, 1),
        ("vyz", VYZ, 1),
        ("ka_off", KA_OFF, 1),
        ("done", DONE, 1),
        ("vcount", VCOUNT, 1),
        ("step", STEP, 1),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    // claim/broadcast and publish temporaries (outside every region)
    add_pieces(
        &mut regs,
        "claim_v",
        Kind::V,
        1,
        18,
        &Live::Between(".Lnext".into(), first_step.clone()),
    )?;
    add_pieces(
        &mut regs,
        "publish_v",
        Kind::V,
        1,
        16,
        &Live::Between(".Lpublish".into(), ".Ldone".into()),
    )?;
    for h in hosted {
        let name = format!("t{}", h.tmpl);
        let (step, entry, exit) = (
            step_label(h.tmpl),
            region_entry_label(&name),
            region_exit_label(&name),
        );
        regs.add_range(
            &format!("{name}_args"),
            Kind::S,
            0,
            2,
            Live::Between(step.clone(), entry.clone()),
        )?;
        regs.add_range(
            &format!("{name}_lane"),
            Kind::V,
            0,
            1,
            Live::Between(step.clone(), entry.clone()),
        )?;
        regs.add_range(
            &format!("{name}_wg_x"),
            Kind::S,
            WG_X as u8,
            1,
            Live::Between(step.clone(), exit.clone()),
        )?;
        regs.add_range(
            &format!("{name}_wg_yz"),
            Kind::S,
            WG_YZ as u8,
            1,
            Live::Between(step, exit),
        )?;
    }
    if let Some(c) = copy {
        // COPY is a PM-emitted region: its inputs stay dispatcher-owned (and unwritten)
        // through the region; its own registers are s2..s15, v1..v3.
        let name = format!("t{c}");
        let (step, entry, exit) = (
            step_label(c),
            region_entry_label(&name),
            region_exit_label(&name),
        );
        let between = Live::Between(entry, exit);
        // through the exit label (live there: the region checks its inputs at its exit)
        let through = Live::Between(step, ".Lpublish".into());
        regs.add_range(&format!("{name}_args"), Kind::S, 0, 2, through.clone())?;
        regs.add_range(&format!("{name}_lane"), Kind::V, 0, 1, through.clone())?;
        regs.add_range(
            &format!("{name}_wg_x"),
            Kind::S,
            WG_X as u8,
            1,
            through.clone(),
        )?;
        regs.add_range(&format!("{name}_wg_yz"), Kind::S, WG_YZ as u8, 1, through)?;
        add_pieces(&mut regs, &format!("{name}_s"), Kind::S, 2, 16, &between)?;
        add_pieces(&mut regs, &format!("{name}_v"), Kind::V, 1, 4, &between)?;
    }
    let spec = KernelSpec {
        kernel_id: "pm_lift_pool".into(),
        variant: symbol.into(),
        arch: Arch::Gfx1201,
        symbol: symbol.into(),
        kernargs: KernargLayout::new(KERNARG_BYTES)
            .pointer_access("tasks", 0, Access::ReadOnly)
            .pointer_access("ticket", 8, Access::ReadWrite)
            .pointer_access("events", 16, Access::ReadWrite)
            .pointer_access("kernargs", 24, Access::ReadOnly)
            .pointer_access("trace", 32, Access::WriteOnly)
            .pointer_access("poison", 40, Access::WriteOnly)
            .hidden("spin_ticks", 48, 8, "by_value")
            .hidden("n_tasks", 56, 4, "by_value")
            .hidden("trace_on", 60, 4, "by_value")
            .hidden("pad", 64, 8, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: block as u16,
        group_segment_fixed_size: lds_total,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, regs);
    let slot = if multi {
        Some(b.lds_slot("claim", slot_base, 64)?)
    } else {
        None
    };
    op(
        &mut b,
        &format!("s_mov_b32 s{PARAMS}, s0"),
        &[s(PARAMS)],
        &[s(0)],
    )?;
    op(
        &mut b,
        &format!("s_mov_b32 s{}, s1", PARAMS + 1),
        &[s(PARAMS + 1)],
        &[s(1)],
    )?;
    op(
        &mut b,
        &format!("s_mov_b32 s{HW_WG}, ttmp9"),
        &[s(HW_WG)],
        &[],
    )?;
    readfirst(&mut b, WAVEBASE, 0)?;
    smem(
        &mut b,
        &format!(
            "s_load_b64 s[{N_TASKS}:{TRACE_ON}], s[{PARAMS}:{}], 0x38",
            PARAMS + 1
        ),
        &[s(N_TASKS), s(TRACE_ON)],
        &[sp(PARAMS)],
    )?;
    b.wait_all()?;
    let ptr = |reg: u8| S::<2>(reg);
    b.loop_(".Lnext", |b| {
        if multi {
            op(
                b,
                &format!("s_cmp_lg_u32 s{WAVEBASE}, 0"),
                &[],
                &[s(WAVEBASE)],
            )?;
            branch(b, "s_cbranch_scc1 .Lbcast")?;
        }
        // ---- wave 0 lane 0: claim, descriptor, predecessor
        exec_lane0(b)?;
        param(b, PTR_A, 0x8)?;
        op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        b.event_rmw(
            AtomicOp::AddU32,
            Some(V::<1>(3)),
            v(2),
            GlobalAddr {
                vaddr: V::<1>(1),
                base: ptr(PTR_A),
                offset: 0,
            },
            Scope::Dev,
            Ordering::Relaxed,
        )?;
        readfirst(b, IDX, 3)?;
        vmov_s(b, 16, IDX)?;
        op(b, "v_mov_b32_e32 v4, -1", &[v(4)], &[])?;
        op(
            b,
            &format!("s_cmp_ge_u32 s{IDX}, s{N_TASKS}"),
            &[],
            &[s(IDX), s(N_TASKS)],
        )?;
        branch(b, "s_cbranch_scc1 .Lclaimed")?;
        param(b, PTR_A, 0x0)?;
        op(
            b,
            &format!("s_lshl_b32 s{TEMP}, s{IDX}, 7"),
            &[s(TEMP)],
            &[s(IDX)],
        )?;
        vmov_s(b, 1, TEMP)?;
        for (dst, off) in [(4u8, 0u32), (8, 16), (12, 32)] {
            let o = if off == 0 {
                String::new()
            } else {
                format!(" offset:{off}")
            };
            b.push(
                Instruction::new(
                    format!(
                        "global_load_b128 v[{dst}:{}], v1, s[{PTR_A}:{}]{o}",
                        dst + 3,
                        PTR_A + 1
                    ),
                    vec![vq(dst)],
                    vec![v(1), sp(PTR_A)],
                )
                .memory(MemoryClass::VmemLoad),
            )?;
        }
        readfirst(b, TEMP, 12)?;
        op(b, &format!("s_cmp_eq_u32 s{TEMP}, -1"), &[], &[s(TEMP)])?;
        branch(b, "s_cbranch_scc1 .Lready")?;
        readfirst(b, TMPL, 13)?; // expected
        param(b, PTR_B, 0x10)?;
        op(
            b,
            &format!("s_lshl_b32 s{TEMP}, s{TEMP}, 2"),
            &[s(TEMP)],
            &[s(TEMP)],
        )?;
        vmov_s(b, 1, TEMP)?;
        smem(
            b,
            &format!("s_load_b32 s{VX}, s[{PARAMS}:{}], 0x30", PARAMS + 1),
            &[s(VX)],
            &[sp(PARAMS)],
        )?;
        b.event_wait(
            ".Lpred",
            GlobalAddr {
                vaddr: V::<1>(1),
                base: ptr(PTR_B),
                offset: 0,
            },
            S::<1>(TMPL),
            V::<1>(2),
            S::<1>(TEMP),
            Scope::Dev,
            Ordering::Acquire,
            1,
            Some(&Watchdog {
                budget: S::<1>(VX),
                timeout: ".Ltimeout".into(),
            }),
        )?;
        branch(b, "s_branch .Lready")?;
        b.label(".Ltimeout")?;
        b.acquire(Scope::Dev)?;
        param(b, PTR_B, 0x28)?;
        op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        b.push(
            Instruction::new(
                format!("global_store_b32 v1, v2, s[{PTR_B}:{}]", PTR_B + 1),
                vec![],
                vec![v(1), v(2), sp(PTR_B)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
        op(b, "v_mov_b32_e32 v4, -1", &[v(4)], &[])?;
        b.label(".Lready")?;
        b.realtime(S::<2>(T_READY))?;
        b.label(".Lclaimed")?;
        if let Some(slot) = slot {
            op(
                b,
                &format!("v_mov_b32_e32 v17, {slot_base:#x}"),
                &[v(17)],
                &[],
            )?;
            for (src, off) in [(4u8, 0u32), (8, 16), (12, 32)] {
                let o = if off == 0 {
                    String::new()
                } else {
                    format!(" offset:{off}")
                };
                b.ds_store(
                    slot,
                    Instruction::new(
                        format!("ds_store_b128 v17, v[{src}:{}]{o}", src + 3),
                        vec![],
                        vec![v(17), vq(src)],
                    )
                    .memory(MemoryClass::DsStore),
                )?;
            }
            b.ds_store(
                slot,
                Instruction::new(
                    "ds_store_b32 v17, v16 offset:48",
                    vec![],
                    vec![v(17), v(16)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        exec_all(b)?;
        if let Some(slot) = slot {
            b.label(".Lbcast")?;
            op(
                b,
                &format!("v_mov_b32_e32 v17, {slot_base:#x}"),
                &[v(17)],
                &[],
            )?;
            b.barrier(&[Transition::Ready(slot)])?;
            for (dst, off) in [(4u8, 0u32), (8, 16), (12, 32)] {
                let o = if off == 0 {
                    String::new()
                } else {
                    format!(" offset:{off}")
                };
                b.ds_load(
                    slot,
                    Instruction::new(
                        format!("ds_load_b128 v[{dst}:{}], v17{o}", dst + 3),
                        vec![vq(dst)],
                        vec![v(17)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            b.ds_load(
                slot,
                Instruction::new("ds_load_b32 v16, v17 offset:48", vec![v(16)], vec![v(17)])
                    .memory(MemoryClass::DsLoad),
            )?;
            b.wait_all()?;
            b.barrier(&[Transition::Retire(slot)])?;
        }
        // vyz = vy | vz << 16 (gfx12 ttmp7)
        op(b, "v_lshl_or_b32 v6, v7, 16, v6", &[v(6)], &[v(6), v(7)])?;
        for (dst, src) in [
            (IDX, 16u8),
            (TMPL, 4),
            (VX, 5),
            (VYZ, 6),
            (KA_OFF, 11),
            (DONE, 14),
            (VCOUNT, 15),
            (GX, 8),
        ] {
            readfirst(b, dst, src)?;
        }
        op(b, &format!("s_cmp_eq_u32 s{TMPL}, -1"), &[], &[s(TMPL)])?;
        branch(b, "s_cbranch_scc1 .Ldone")?;
        b.acquire(Scope::Dev)?;
        op(
            b,
            &format!("s_max_u32 s{VCOUNT}, s{VCOUNT}, 1"),
            &[s(VCOUNT)],
            &[s(VCOUNT)],
        )?;
        for h in hosted {
            op(
                b,
                &format!("s_cmp_eq_u32 s{TMPL}, {:#x}", h.tmpl),
                &[],
                &[s(TMPL)],
            )?;
            branch(b, &format!("s_cbranch_scc1 .Lt{}_go", h.tmpl))?;
        }
        if let Some(c) = copy {
            op(b, &format!("s_cmp_eq_u32 s{TMPL}, {c:#x}"), &[], &[s(TMPL)])?;
            branch(b, &format!("s_cbranch_scc1 .Lt{c}_go"))?;
        }
        branch(b, "s_branch .Lpublish")?;
        for h in hosted {
            let name = format!("t{}", h.tmpl);
            b.label(&format!(".Lt{}_go", h.tmpl))?;
            op(b, &format!("s_mov_b32 s{STEP}, 0"), &[s(STEP)], &[])?;
            b.realtime(S::<2>(T_START))?;
            b.wait_all()?;
            let abi = RegionAbi {
                name: name.clone(),
                inputs: vec![
                    ("args".into(), sp(0)),
                    ("lane".into(), v(0)),
                    ("workgroup_x".into(), s(WG_X as u8)),
                    ("workgroup_yz".into(), s(WG_YZ as u8)),
                ],
                lds_bytes: h.region.lds_bytes,
                workgroup_size: block as u16,
            };
            b.loop_(&step_label(h.tmpl), |b| {
                if multi {
                    // The previous step's waves may still read the body's LDS: every wave
                    // leaves step n before any wave enters step n+1.
                    b.barrier(&[])?;
                }
                op(
                    b,
                    &format!("s_add_co_i32 s{WG_X}, s{VX}, s{STEP}"),
                    &[s(WG_X as u8)],
                    &[s(VX), s(STEP)],
                )?;
                op(
                    b,
                    &format!("s_mov_b32 s{WG_YZ}, s{VYZ}"),
                    &[s(WG_YZ as u8)],
                    &[s(VYZ)],
                )?;
                smem(
                    b,
                    &format!("s_load_b64 s[0:1], s[{PARAMS}:{}], 0x18", PARAMS + 1),
                    &[sp(0)],
                    &[sp(PARAMS)],
                )?;
                b.wait_all()?;
                op(
                    b,
                    &format!("s_add_co_u32 s0, s0, s{KA_OFF}"),
                    &[s(0)],
                    &[s(0), s(KA_OFF)],
                )?;
                op(b, "s_add_co_ci_u32 s1, s1, 0", &[s(1)], &[s(1)])?;
                op(b, "v_mbcnt_lo_u32_b32 v0, -1, 0", &[v(0)], &[])?;
                op(
                    b,
                    &format!("v_add_nc_u32_e32 v0, s{WAVEBASE}, v0"),
                    &[v(0)],
                    &[v(0), s(WAVEBASE)],
                )?;
                b.wait_all()?;
                let r = h.region;
                b.foreign_region(&abi, &r.lines, r.sgprs, r.vgprs, r.lds_bytes)?;
                exec_all(b)?;
                op(
                    b,
                    &format!("s_add_co_i32 s{STEP}, s{STEP}, 1"),
                    &[s(STEP)],
                    &[s(STEP)],
                )?;
                op(
                    b,
                    &format!("s_cmp_lt_u32 s{STEP}, s{VCOUNT}"),
                    &[],
                    &[s(STEP), s(VCOUNT)],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", step_label(h.tmpl)))
            })?;
            b.realtime(S::<2>(T_END))?;
            b.wait_all()?;
            branch(b, "s_branch .Lpublish")?;
        }
        if let Some(c) = copy {
            let name = format!("t{c}");
            b.label(&format!(".Lt{c}_go"))?;
            op(b, &format!("s_mov_b32 s{STEP}, 0"), &[s(STEP)], &[])?;
            b.realtime(S::<2>(T_START))?;
            b.wait_all()?;
            let abi = RegionAbi {
                name: name.clone(),
                inputs: vec![
                    ("args".into(), sp(0)),
                    ("lane".into(), v(0)),
                    ("workgroup_x".into(), s(WG_X as u8)),
                ],
                lds_bytes: 0,
                workgroup_size: block as u16,
            };
            b.loop_(&step_label(c), |b| {
                if multi {
                    b.barrier(&[])?;
                }
                op(
                    b,
                    &format!("s_add_co_i32 s{WG_X}, s{VX}, s{STEP}"),
                    &[s(WG_X as u8)],
                    &[s(VX), s(STEP)],
                )?;
                op(
                    b,
                    &format!("s_mov_b32 s{WG_YZ}, s{VYZ}"),
                    &[s(WG_YZ as u8)],
                    &[s(VYZ)],
                )?;
                smem(
                    b,
                    &format!("s_load_b64 s[0:1], s[{PARAMS}:{}], 0x18", PARAMS + 1),
                    &[sp(0)],
                    &[sp(PARAMS)],
                )?;
                b.wait_all()?;
                op(
                    b,
                    &format!("s_add_co_u32 s0, s0, s{KA_OFF}"),
                    &[s(0)],
                    &[s(0), s(KA_OFF)],
                )?;
                op(b, "s_add_co_ci_u32 s1, s1, 0", &[s(1)], &[s(1)])?;
                op(b, "v_mbcnt_lo_u32_b32 v0, -1, 0", &[v(0)], &[])?;
                op(
                    b,
                    &format!("v_add_nc_u32_e32 v0, s{WAVEBASE}, v0"),
                    &[v(0)],
                    &[v(0), s(WAVEBASE)],
                )?;
                b.wait_all()?;
                b.region(&abi, |b| copy_body(b, block, &name))?;
                exec_all(b)?;
                op(
                    b,
                    &format!("s_add_co_i32 s{STEP}, s{STEP}, 1"),
                    &[s(STEP)],
                    &[s(STEP)],
                )?;
                op(
                    b,
                    &format!("s_cmp_lt_u32 s{STEP}, s{VCOUNT}"),
                    &[],
                    &[s(STEP), s(VCOUNT)],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", step_label(c)))
            })?;
            b.realtime(S::<2>(T_END))?;
            b.wait_all()?;
            branch(b, "s_branch .Lpublish")?;
        }
        // ---- publication
        b.label(".Lpublish")?;
        b.release(Scope::Dev)?;
        if multi {
            b.barrier(&[])?;
            op(
                b,
                &format!("s_cmp_lg_u32 s{WAVEBASE}, 0"),
                &[],
                &[s(WAVEBASE)],
            )?;
            branch(b, "s_cbranch_scc1 .Lnext_tail")?;
        }
        exec_lane0(b)?;
        op(b, &format!("s_cmp_eq_u32 s{DONE}, -1"), &[], &[s(DONE)])?;
        branch(b, "s_cbranch_scc1 .Lsignaled")?;
        param(b, PTR_B, 0x10)?;
        op(
            b,
            &format!("s_lshl_b32 s{TEMP}, s{DONE}, 2"),
            &[s(TEMP)],
            &[s(DONE)],
        )?;
        vmov_s(b, 1, TEMP)?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        b.event_signal(
            V::<1>(2),
            GlobalAddr {
                vaddr: V::<1>(1),
                base: ptr(PTR_B),
                offset: 0,
            },
            Scope::Dev,
            Ordering::Release,
        )?;
        b.label(".Lsignaled")?;
        b.realtime(S::<2>(T_PUBLISH))?;
        op(
            b,
            &format!("s_cmp_eq_u32 s{TRACE_ON}, 0"),
            &[],
            &[s(TRACE_ON)],
        )?;
        branch(b, "s_cbranch_scc1 .Ltraced")?;
        param(b, PTR_B, 0x20)?;
        op(
            b,
            &format!("s_mul_i32 s{TEMP}, s{IDX}, {TRACE_BYTES}"),
            &[s(TEMP)],
            &[s(IDX)],
        )?;
        vmov_s(b, 1, TEMP)?;
        let words = [
            IDX,
            TMPL,
            HW_WG,
            HW_WG,
            T_READY,
            T_READY + 1,
            T_START,
            T_START + 1,
            T_END,
            T_END + 1,
            T_PUBLISH,
            T_PUBLISH + 1,
        ];
        for (i, w) in words.iter().enumerate() {
            vmov_s(b, 4 + i as u8, *w)?;
        }
        op(b, "v_mov_b32_e32 v7, 0", &[v(7)], &[])?;
        for (src, off) in [(4u8, 0u32), (8, 16), (12, 32)] {
            let o = if off == 0 {
                String::new()
            } else {
                format!(" offset:{off}")
            };
            b.push(
                Instruction::new(
                    format!(
                        "global_store_b128 v1, v[{src}:{}], s[{PTR_B}:{}]{o}",
                        src + 3,
                        PTR_B + 1
                    ),
                    vec![],
                    vec![v(1), vq(src), sp(PTR_B)],
                )
                .memory(MemoryClass::VmemStore),
            )?;
        }
        // the trace record publishes nothing; release it so the next claim is clean
        b.release(Scope::Dev)?;
        b.label(".Ltraced")?;
        exec_all(b)?;
        if multi {
            b.label(".Lnext_tail")?;
        }
        b.wait_all()?;
        branch(b, "s_branch .Lnext")
    })?;
    b.label(".Ldone")?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// COPY (`M0CopyArgs {u64 dst, u64 src, u64 bytes}` at the task's kernarg block):
/// virtual workgroup `w` of `gx` copies dwords `w * block + t + k * gx * block` below
/// `bytes / 4`; virtual workgroup 0 copies the `bytes % 4` tail. `src`/`dst` must not
/// overlap; `bytes < 2^32`. The body's registers are s0..s15, v0..v3 (the region's).
fn copy_body(b: &mut Builder, block: u32, name: &str) -> Result<(), String> {
    let (wx, nb) = (WG_X as u8, block);
    smem(
        b,
        "s_load_b128 s[4:7], s[0:1], 0x0",
        &[sp(4), sp(6)],
        &[sp(0)],
    )?;
    smem(b, "s_load_b32 s8, s[0:1], 0x10", &[s(8)], &[sp(0)])?;
    op(
        b,
        &format!("s_mul_i32 s10, s{GX}, {nb:#x}"),
        &[s(10)],
        &[s(GX)],
    )?;
    op(
        b,
        &format!("s_mul_i32 s11, s{wx}, {nb:#x}"),
        &[s(11)],
        &[s(wx)],
    )?;
    op(b, "v_add_nc_u32_e32 v1, s11, v0", &[v(1)], &[s(11), v(0)])?;
    op(b, "s_lshr_b32 s9, s8, 2", &[s(9)], &[s(8)])?;
    b.wait_all()?;
    let (head, tail, end) = (
        format!(".L{name}_copy"),
        format!(".L{name}_tail"),
        format!(".L{name}_end"),
    );
    b.loop_(&head, |b| {
        // lanes past the last dword drop out of EXEC for good
        op(b, "v_cmpx_gt_u32_e32 s9, v1", &[], &[s(9), v(1)])?;
        branch(b, &format!("s_cbranch_execz {tail}"))?;
        op(b, "v_lshlrev_b32_e32 v2, 2, v1", &[v(2)], &[v(1)])?;
        b.push(
            Instruction::new(
                "global_load_b32 v3, v2, s[6:7]",
                vec![v(3)],
                vec![v(2), sp(6)],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
        b.push(
            Instruction::new(
                "global_store_b32 v2, v3, s[4:5]",
                vec![],
                vec![v(2), v(3), sp(4)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
        op(b, "v_add_nc_u32_e32 v1, s10, v1", &[v(1)], &[s(10), v(1)])?;
        b.release_store_sources()?;
        branch(b, &format!("s_branch {head}"))
    })?;
    b.label(&tail)?;
    exec_all(b)?;
    op(b, "s_and_b32 s12, s8, 3", &[s(12)], &[s(8)])?;
    // tail only when bytes % 4 != 0, by virtual workgroup 0 (s12 != 0 and wx == 0)
    op(b, "s_cmp_eq_u32 s12, 0", &[], &[s(12)])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    op(b, &format!("s_cmp_lg_u32 s{wx}, 0"), &[], &[s(wx)])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    op(b, "v_cmpx_gt_u32_e32 s12, v0", &[], &[s(12), v(0)])?;
    branch(b, &format!("s_cbranch_execz {end}"))?;
    op(b, "s_and_b32 s14, s8, -4", &[s(14)], &[s(8)])?;
    op(b, "v_add_nc_u32_e32 v2, s14, v0", &[v(2)], &[s(14), v(0)])?;
    b.push(
        Instruction::new(
            "global_load_u8 v3, v2, s[6:7]",
            vec![v(3)],
            vec![v(2), sp(6)],
        )
        .memory(MemoryClass::VmemLoad),
    )?;
    b.push(
        Instruction::new(
            "global_store_b8 v2, v3, s[4:5]",
            vec![],
            vec![v(2), v(3), sp(4)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.label(&end)?;
    b.wait_all()
}
