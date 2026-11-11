// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab-only gfx1201 persistent pool dispatcher hosting lifted hipcc bodies
//! (`peacemaker_lift::rewrite::region`) as regions, with MegaM0's P0 task ABI
//! (`release-0.4.5/megam0/p0h/m0_rt.h`); nothing here is on a product route.
//!
//! Kernarg: `M0Params` (72 bytes): `tasks, ticket, events, kernargs, trace, poison`
//! pointers at 0..48, `spin_ticks` at 48, `n_tasks` at 56, `trace_on` at 60.
//! `spin_ticks` is a count of POLLS, not of clock ticks: its low dword is how many
//! times one predecessor wait re-reads `events[pred]` (each poll is a device-scope load
//! plus an `s_sleep`) before it gives up and sets poison; the high dword is ignored. A task is a 128-byte
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
use crate::ledger::Counter;
use crate::plan::{region_entry_label, region_exit_label, Access, RegionAbi};
use crate::reg::{Kind, Live, RegRef};
use crate::sync::{AtomicOp, GlobalAddr, Ordering, Scope, Watchdog};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, S, V};
use peacemaker_lift::rewrite::region::LiftedRegion;

pub const KERNARG_BYTES: u32 = 72;
pub const TASK_BYTES: u32 = 128;
pub const TRACE_BYTES: u32 = 48;
/// SGPRs a hosted body may use without spilling (the dispatcher owns s72..s103). A body
/// using more (up to `WG_X`, the first region input above it) runs with the dispatcher
/// SGPRs below its extent parked in the lanes of one spare VGPR.
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
/// The template id a span task whose index is past the live count runs as: hosted by no
/// pool, so the task publishes its done event without entering a region.
pub const NOP_TMPL: u32 = 0xffff_fffe;
/// Every dispatcher-owned SGPR range (name, base, length).
const DISPATCHER_SGPRS: &[(&str, u8, u8)] = &[
    // s72 = gx; s[72:73] = the span bitmap base inside the span checks (before the broadcast)
    ("gx", GX, 2),
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
];

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

/// A PM-emitted template a pool hosts beside its lifted bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Native {
    /// COPY (`M0CopyArgs {u64 dst, u64 src, u64 bytes}`), MegaM0's tmpl 2.
    Copy(u32),
    /// EXPAND_LIVE (`kernels::p0_expand_gfx1201`), tmpl 1: o4's live tile list.
    Expand(u32),
}
impl Native {
    pub fn tmpl(self) -> u32 {
        match self {
            Native::Copy(t) | Native::Expand(t) => t,
        }
    }
    /// The registers the body owns inside its region: SGPRs `[2, s)` and VGPRs `[1, v)`.
    fn footprint(self) -> (u8, u8) {
        use crate::kernels::p0_expand_gfx1201 as expand;
        match self {
            Native::Copy(_) => (16, 4),
            Native::Expand(_) => (expand::SGPRS as u8, expand::VGPRS as u8),
        }
    }
    /// The body's LDS, `[0, lds)` like a lifted body's (the dispatcher slot sits above).
    fn lds(self) -> u32 {
        match self {
            Native::Copy(_) => 0,
            Native::Expand(_) => crate::kernels::p0_expand_gfx1201::LDS_BYTES,
        }
    }
    fn emit(self, b: &mut Builder, block: u32, name: &str) -> Result<(), String> {
        match self {
            Native::Copy(_) => copy_body(b, block, name),
            Native::Expand(_) => crate::kernels::p0_expand_gfx1201::expand_body(b, block, name),
        }
    }
}

/// The pool dispatcher `symbol` of workgroup size `block` hosting the lifted bodies
/// `hosted` and the PM-emitted templates `natives`.
pub fn pool(
    symbol: &str,
    block: u32,
    hosted: &[Hosted],
    natives: &[Native],
) -> Result<Emitted, String> {
    if hosted.is_empty() && natives.is_empty() {
        return Err("a pool hosts at least one template".into());
    }
    let mut tmpls: Vec<u32> = hosted.iter().map(|h| h.tmpl).collect();
    tmpls.extend(natives.iter().map(|n| n.tmpl()));
    let distinct: std::collections::BTreeSet<u32> = tmpls.iter().copied().collect();
    if distinct.len() != tmpls.len() {
        return Err(format!("a template id is hosted twice: {tmpls:?}"));
    }
    for h in hosted {
        let r = h.region;
        if r.block != block {
            return Err(format!(
                "{}: blockDim {} != pool {block}",
                r.symbol, r.block
            ));
        }
        if r.sgprs > WG_X {
            return Err(format!(
                "{}: {} SGPRs reach the region inputs at s{WG_X}",
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
    let body_lds = hosted
        .iter()
        .map(|h| h.region.lds_bytes)
        .chain(natives.iter().map(|n| n.lds()))
        .max()
        .unwrap_or(0);
    let slot_base = body_lds.next_multiple_of(16);
    let lds_total = if multi { slot_base + 64 } else { body_lds };
    let step_label = |t: u32| format!(".Lt{t}_step");
    let first_step = step_label(hosted.first().map_or_else(|| natives[0].tmpl(), |h| h.tmpl));
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
    // Bodies above BODY_SGPR_LIMIT: the dispatcher SGPRs they overlap are parked in
    // lanes of `spill_v` (above every body's and dispatcher temporary's VGPRs) from just
    // before the region entry to just after its exit; those SGPRs' ranges exclude it.
    let spilling: Vec<(u32, String, String, u16)> = hosted
        .iter()
        .filter(|h| h.region.sgprs > BODY_SGPR_LIMIT)
        .map(|h| {
            let name = format!("t{}", h.tmpl);
            (
                h.tmpl,
                region_entry_label(&name),
                region_exit_label(&name),
                h.region.sgprs,
            )
        })
        .collect();
    // unused (0) when no body spills
    let spill_v: u8 = if spilling.is_empty() {
        0
    } else {
        hosted
            .iter()
            .map(|h| h.region.vgprs)
            .max()
            .unwrap_or(0)
            .max(19)
            .try_into()
            .map_err(|_| "no VGPR left to park dispatcher SGPRs".to_string())?
    };
    for &(name, base, len) in DISPATCHER_SGPRS {
        let parked: Vec<&(u32, String, String, u16)> = spilling
            .iter()
            .filter(|(_, _, _, top)| u16::from(base) < *top)
            .collect();
        if parked.is_empty() {
            regs.add_range(name, Kind::S, base, len, Live::Whole)?;
            continue;
        }
        let mut from = "entry".to_string();
        for (i, (_, entry, exit, _)) in parked.iter().enumerate() {
            regs.add_range(
                &format!("{name}_{i}"),
                Kind::S,
                base,
                len,
                Live::Between(from, entry.clone()),
            )?;
            from = exit.clone();
        }
        regs.add_range(
            &format!("{name}_{}", parked.len()),
            Kind::S,
            base,
            len,
            Live::Between(from, ".Ldone".into()),
        )?;
    }
    for (t, ..) in &spilling {
        regs.add_range(
            &format!("park_t{t}"),
            Kind::V,
            spill_v,
            1,
            Live::Between(format!(".Lt{t}_park"), format!(".Lt{t}_unpark")),
        )?;
    }
    // claim/broadcast and publish temporaries (outside every region)
    add_pieces(
        &mut regs,
        "claim_v",
        Kind::V,
        1,
        40,
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
    for (i, n) in natives.iter().enumerate() {
        // A PM-emitted region: its inputs stay dispatcher-owned (and unwritten) through
        // the region; it owns `footprint()` inside it.
        let c = n.tmpl();
        let name = format!("t{c}");
        let (step, entry, exit) = (
            step_label(c),
            region_entry_label(&name),
            region_exit_label(&name),
        );
        let between = Live::Between(entry, exit);
        // through the exit label (live there: the region checks its inputs at its exit),
        // up to the next native template's arm
        let end = natives
            .get(i + 1)
            .map_or(".Lpublish".to_string(), |m| format!(".Lt{}_go", m.tmpl()));
        let through = Live::Between(step, end);
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
        let (s_end, v_end) = n.footprint();
        add_pieces(&mut regs, &format!("{name}_s"), Kind::S, 2, s_end, &between)?;
        add_pieces(&mut regs, &format!("{name}_v"), Kind::V, 1, v_end, &between)?;
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
            .pointer_access("poison", 40, Access::ReadWrite) // set by an atomic OR
            .hidden("spin_ticks", 48, 8, "by_value")
            .hidden("n_tasks", 56, 4, "by_value")
            .hidden("trace_on", 60, 4, "by_value")
            .pointer_access("spans", 64, Access::ReadOnly),
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
        // dwords 0..11 (v4..v15) and the span words 12..15 (v18..v21: flags, k, list id)
        for (dst, off) in [(4u8, 0u32), (8, 16), (12, 32), (18, 48)] {
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
        branch(b, "s_cbranch_scc1 .Lspan")?;
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
        branch(b, "s_branch .Lspan")?;
        b.label(".Ltimeout")?;
        b.acquire(Scope::Dev)?;
        param(b, PTR_B, 0x28)?;
        op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        // poison is an event cell: railgun EventCells admit only commutative atomics, so it
        // is set by a released (idempotent) atomic OR
        b.event_rmw(
            AtomicOp::OrB32,
            None,
            v(2),
            GlobalAddr {
                vaddr: V::<1>(1),
                base: ptr(PTR_B),
                offset: 0,
            },
            Scope::Dev,
            Ordering::Release,
        )?;
        op(b, "v_mov_b32_e32 v4, -1", &[v(4)], &[])?;
        branch(b, "s_branch .Lready")?;
        span_entry(b)?;
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
        for n in natives {
            let c = n.tmpl();
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
                // the dispatcher SGPRs below the body's extent, parked lane by lane
                let parked: Vec<u8> = DISPATCHER_SGPRS
                    .iter()
                    .filter(|(_, base, _)| u16::from(*base) < r.sgprs)
                    .flat_map(|&(_, base, len)| base..base + len)
                    .collect();
                if !parked.is_empty() {
                    b.label(&format!(".Lt{}_park", h.tmpl))?;
                    for &k in &parked {
                        op(
                            b,
                            &format!("v_writelane_b32 v{spill_v}, s{k}, {}", k - GX),
                            &[v(spill_v)],
                            &[s(k)],
                        )?;
                    }
                }
                b.foreign_region(&abi, &r.lines, r.sgprs, r.vgprs, r.lds_bytes)?;
                // provenance for `.note.pm_regions` (railgun-cert replays the conversion)
                let p = b
                    .regions
                    .last_mut()
                    .expect("foreign_region records a proof");
                p.template_id = Some(h.tmpl);
                p.lift_inputs = Some(crate::plan::RegionLiftInputs {
                    prefix: r.inputs.prefix.clone(),
                    workgroup_x: r.inputs.workgroup_x,
                    workgroup_yz: r.inputs.workgroup_yz,
                    block: r.inputs.block,
                    dynamic_lds: r.inputs.dynamic_lds,
                });
                p.source_object_sha256 = Some(r.receipt.base_object_sha256);
                p.source_symbol = Some(r.symbol.clone());
                for &k in &parked {
                    op(
                        b,
                        &format!("v_readlane_b32 s{k}, v{spill_v}, {}", k - GX),
                        &[s(k)],
                        &[v(spill_v)],
                    )?;
                }
                if !parked.is_empty() {
                    b.label(&format!(".Lt{}_unpark", h.tmpl))?;
                }
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
        for &n in natives {
            let c = n.tmpl();
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
                lds_bytes: n.lds(),
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
                b.region(&abi, |b| n.emit(b, block, &name))?;
                b.regions
                    .last_mut()
                    .expect("region records a proof")
                    .template_id = Some(c);
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
        // `t_ready` (`.Lready`) is returned through KMcnt and nothing on a path through an
        // unmatched template id (it is skipped, not refused: `.Lpublish` straight from the
        // switch) waits for it before the record below reads it; the trace block also reads
        // `t_publish` and the `PTR_B` load, so one KMcnt wait here retires all three and the
        // builder's own later wait (before the first of them it reads) is not emitted twice.
        b.wait(Counter::Km, 0)?;
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
        // `.Ltraced` is also reached with trace off straight from the release signal,
        // whose atomic may still be reading v1/v2: the linear ledger here only knows the
        // traced path, so drain every counter before the claim reuses those VGPRs.
        b.drain_all_counters()?;
        branch(b, "s_branch .Lnext")
    })?;
    b.label(".Ldone")?;
    branch(&mut b, "s_endpgm")?;
    let emitted = b.finish()?;
    // Dispatcher-only fatal verification: every instruction outside the hosted regions'
    // bodies is checked with no allowance; the bodies (foreign lifted text) are walked for
    // their memory and wait effects but their findings are not reported here (they are the
    // hosted kernel's own, listed per template by `tests/ledger_cfg_pins.rs`).
    crate::ledger_cfg::verify_dispatcher_emitted(&emitted)
        .map_err(|e| format!("{symbol}: CFG wait verification failed: {e}"))?;
    Ok(emitted)
}

/// COPY (`M0CopyArgs {u64 dst, u64 src, u64 bytes}` at the task's kernarg block):
/// virtual workgroup `w` of `gx` copies dwords `w * block + t + k * gx * block` below
/// `bytes / 4`; virtual workgroup 0 copies the `bytes % 4` tail. `src`/`dst` must not
/// overlap; `bytes < 2^32`. The body's registers are s0..s15, v0..v3 (the region's).
/// Span mode (railgunmega/DYNAMIC.md), on wave 0 lane 0 after the predecessor acquire,
/// with the descriptor in v4..v15 and its span words in v18 (flags), v19 (k), v20 (list
/// id). Static tasks (flags 0) pass straight to `.Lready`; a span task without a
/// predecessor event poisons. A span task reads its span descriptor and live entry into
/// VGPRs (v24..v39: the dispatcher's meaning-carrying SGPRs are not reused; the bitmap
/// base alone is read into s[72:73], free until the broadcast defines `gx`):
/// `list_id < n_spans`, `c = count` (after the acquire) `<= K`, `k >= c` turns the task
/// into a NOP (template `NOP_TMPL`: no region, `done` still released), the entry
/// `{vx, vy, vz}` must lie inside the header grid and its bit in the claim bitmap must be
/// clear (a returning atomic OR sets it); any failed check poisons. A valid entry replaces
/// the descriptor's `vx, vy, vz` and sets `vcount = 1`.
fn span_entry(b: &mut Builder) -> Result<(), String> {
    // v_cmp into TEMP, then branch to `to` when any lane (lane 0) holds
    fn check(b: &mut Builder, cmp: &str, uses: &[RegRef], to: &str) -> Result<(), String> {
        op(b, &format!("{cmp}"), &[s(TEMP)], uses)?;
        op(b, &format!("s_cmp_lg_u32 s{TEMP}, 0"), &[], &[s(TEMP)])?;
        branch(b, &format!("s_cbranch_scc1 {to}"))
    }
    let load = |b: &mut Builder, text: String, def: RegRef, uses: Vec<RegRef>| {
        b.push(Instruction::new(text, vec![def], uses).memory(MemoryClass::VmemLoad))
    };
    let vp = |n: u8| RegRef {
        kind: Kind::V,
        base: n,
        len: 2,
    };
    b.label(".Lspan")?;
    // reached from the no-predecessor branch and the wait's exit: the span words' load
    // may still be in flight on either (the linear ledger here only knows the wait path)
    b.wait(Counter::Load, 0)?;
    readfirst(b, TEMP, 18)?;
    op(b, &format!("s_cmp_eq_u32 s{TEMP}, 0"), &[], &[s(TEMP)])?;
    branch(b, "s_cbranch_scc1 .Lready")?;
    op(b, &format!("s_cmp_lg_u32 s{TEMP}, 1"), &[], &[s(TEMP)])?;
    branch(b, "s_cbranch_scc1 .Lspan_poison")?;
    // a span header must name its producer's done event: the count load below is then
    // ordered by that wait's acquire on every path (a host-filled span names an event the
    // host leaves satisfied, e.g. `expected = 0`)
    readfirst(b, TEMP, 12)?;
    op(b, &format!("s_cmp_eq_u32 s{TEMP}, -1"), &[], &[s(TEMP)])?;
    branch(b, "s_cbranch_scc1 .Lspan_poison")?;
    // the span table: u32 n_spans at +0, descriptor i at +32 + 32 i
    param(b, PTR_B, 0x40)?;
    b.wait_all()?;
    op(
        b,
        &format!("s_or_b32 s{TEMP}, s{PTR_B}, s{}", PTR_B + 1),
        &[s(TEMP)],
        &[sp(PTR_B)],
    )?;
    op(b, &format!("s_cmp_eq_u32 s{TEMP}, 0"), &[], &[s(TEMP)])?;
    branch(b, "s_cbranch_scc1 .Lspan_poison")?;
    op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    load(
        b,
        format!("global_load_b32 v28, v1, s[{PTR_B}:{}]", PTR_B + 1),
        v(28),
        vec![v(1), sp(PTR_B)],
    )?;
    b.wait_all()?;
    check(
        b,
        &format!("v_cmp_ge_u32_e64 s{TEMP}, v20, v28"),
        &[v(20), v(28)],
        ".Lspan_poison",
    )?;
    // descriptor `{live_list, count_cell}` -> v[32:35], `{bitmap, K, 0}` -> v[36:39]
    op(b, "v_lshl_add_u32 v29, v20, 5, 32", &[v(29)], &[v(20)])?;
    for (dst, off) in [(32u8, ""), (36, " offset:16")] {
        load(
            b,
            format!(
                "global_load_b128 v[{dst}:{}], v29, s[{PTR_B}:{}]{off}",
                dst + 3,
                PTR_B + 1
            ),
            vq(dst),
            vec![v(29), sp(PTR_B)],
        )?;
    }
    b.wait_all()?;
    // c, ordered after the predecessor acquire (the producer released entries and count)
    load(
        b,
        "global_load_b32 v28, v[34:35], off".into(),
        v(28),
        vec![vp(34)],
    )?;
    b.wait_all()?;
    check(
        b,
        &format!("v_cmp_gt_u32_e64 s{TEMP}, v28, v38"),
        &[v(28), v(38)],
        ".Lspan_poison",
    )?;
    check(
        b,
        &format!("v_cmp_ge_u32_e64 s{TEMP}, v19, v28"),
        &[v(19), v(28)],
        ".Lspan_nop",
    )?;
    op(
        b,
        "v_mad_co_u64_u32 v[30:31], null, v19, 16, v[32:33]",
        &[vp(30)],
        &[v(19), vp(32)],
    )?;
    load(
        b,
        "global_load_b128 v[24:27], v[30:31], off".into(),
        vq(24),
        vec![vp(30)],
    )?;
    b.wait_all()?;
    // inside the header grid (v8..v10 = gx, gy, gz)
    for (e, g) in [(24u8, 8u8), (25, 9), (26, 10)] {
        check(
            b,
            &format!("v_cmp_ge_u32_e64 s{TEMP}, v{e}, v{g}"),
            &[v(e), v(g)],
            ".Lspan_poison",
        )?;
    }
    // bit = vx + gx * (vy + gy * vz); claim it in the bitmap
    op(b, "v_mul_lo_u32 v27, v26, v9", &[v(27)], &[v(26), v(9)])?;
    op(
        b,
        "v_add_nc_u32_e32 v27, v27, v25",
        &[v(27)],
        &[v(27), v(25)],
    )?;
    op(b, "v_mul_lo_u32 v27, v27, v8", &[v(27)], &[v(27), v(8)])?;
    op(
        b,
        "v_add_nc_u32_e32 v27, v27, v24",
        &[v(27)],
        &[v(27), v(24)],
    )?;
    op(b, "v_lshrrev_b32_e32 v1, 5, v27", &[v(1)], &[v(27)])?;
    op(b, "v_lshlrev_b32_e32 v1, 2, v1", &[v(1)], &[v(1)])?;
    op(b, "v_and_b32_e32 v2, 31, v27", &[v(2)], &[v(27)])?;
    op(b, "v_lshlrev_b32_e64 v2, v2, 1", &[v(2)], &[v(2)])?;
    readfirst(b, GX, 36)?;
    readfirst(b, GX + 1, 37)?;
    b.event_rmw(
        AtomicOp::OrB32,
        Some(V::<1>(3)),
        v(2),
        GlobalAddr {
            vaddr: V::<1>(1),
            base: S::<2>(GX),
            offset: 0,
        },
        Scope::Dev,
        Ordering::Relaxed,
    )?;
    b.wait_all()?;
    op(b, "v_and_b32_e32 v3, v3, v2", &[v(3)], &[v(3), v(2)])?;
    check(
        b,
        &format!("v_cmp_ne_u32_e64 s{TEMP}, 0, v3"),
        &[v(3)],
        ".Lspan_poison",
    )?;
    for (d, e) in [(5u8, 24u8), (6, 25), (7, 26)] {
        op(b, &format!("v_mov_b32_e32 v{d}, v{e}"), &[v(d)], &[v(e)])?;
    }
    op(b, "v_mov_b32_e32 v15, 1", &[v(15)], &[])?;
    branch(b, "s_branch .Lready")?;
    b.label(".Lspan_nop")?;
    op(b, &format!("v_mov_b32_e32 v4, {NOP_TMPL:#x}"), &[v(4)], &[])?;
    branch(b, "s_branch .Lready")?;
    b.label(".Lspan_poison")?;
    b.acquire(Scope::Dev)?;
    param(b, PTR_B, 0x28)?;
    op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
    b.event_rmw(
        AtomicOp::OrB32,
        None,
        v(2),
        GlobalAddr {
            vaddr: V::<1>(1),
            base: S::<2>(PTR_B),
            offset: 0,
        },
        Scope::Dev,
        Ordering::Release,
    )?;
    op(b, "v_mov_b32_e32 v4, -1", &[v(4)], &[])
}

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
