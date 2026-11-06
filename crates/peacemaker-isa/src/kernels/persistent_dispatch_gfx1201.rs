// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab-only gfx1201 kernels proving the PeaceMaker persistent-megakernel
//! prerequisites; nothing here is registered in a shipping registry or reachable
//! from a product route.
//!
//! * [`dispatcher`]: a trivial persistent dispatcher. Each workgroup (one wave32,
//!   the body's geometry) claims task indices from a device-scope claim counter
//!   (`Relaxed` fetch-add, returning) until they run out, runs the F05 MQ2
//!   x-batched GEMV body ([`super::gemv_mq2_xbatch_pm_gfx1201`], the shipped
//!   `r4_b4_u1_native_anchor` module's code) as a region at virtual workgroup
//!   `task`, then publishes the task's completion with a `Release` increment of a
//!   completion counter. Kernarg: the body's 36-byte ABI (`A, x, y, M, K, B`) at 0,
//!   the queue pointer at 40 (`queue[0]` claim counter, `queue[1]` completion
//!   counter, both zero at launch), the task count `ceil(M / rows)` at 48.
//! * [`handoff`]: two (or more) workgroups hand data off through an event
//!   counter. Workgroup 0 sleeps (`delay` x `s_sleep 127`), stores `WORDS` dwords
//!   per lane, reads the realtime clock and publishes with a `Release` signal;
//!   every other workgroup first reads the same data (priming its vector cache
//!   with the pre-publication values), then `event_wait`s with `Acquire` (bounded
//!   by a watchdog), reads the data again and stores both copies, its status and
//!   its observation time.

use super::gemv_mq2_xbatch_pm_gfx1201 as mq2;
use crate::insn::{Instruction, MemoryClass};
use crate::plan::{Access, RegionAbi};
use crate::reg::{Kind, Live, RegRef};
use crate::sync::{GlobalAddr, Ordering, Scope, Watchdog};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, S, V};

pub const DISPATCH_SYMBOL: &str = "pm_persistent_mq2_xbatch_dispatch";
pub const DISPATCH_KERNARG_BYTES: u32 = 56;
pub const HANDOFF_SYMBOL: &str = "pm_event_handoff";
pub const HANDOFF_KERNARG_BYTES: u32 = 40;
/// Dwords each lane of the producer stores (and each consumer reads twice).
pub const WORDS: u32 = 16;
/// Dwords of one workgroup's output record: `post[WORDS*32]`, `pre[WORDS*32]`, then
/// `status` (1 = acquired, `0xdead` = watchdog timeout), `spins_left`, `time_lo`, `time_hi`.
pub const RECORD_DWORDS: u32 = 2 * WORDS * 32 + 4;
/// The region name of the MQ2 body inside the dispatcher.
pub const REGION: &str = "mq2";

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
fn smem(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(MemoryClass::SmemLoad))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
/// EXEC = lane 0 only / all 32 lanes (no register operands: EXEC is not a plan register).
fn exec_lane0(b: &mut Builder) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}
fn exec_all(b: &mut Builder) -> Result<(), String> {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

// SGPRs of the dispatcher, above every SGPR the body touches (s0..s43).
const S_QUEUE: u8 = 44;
const S_TASKS: u8 = 46;
const S_TASK: u8 = 47;

/// The dispatcher's kernel spec.
pub fn dispatcher_spec() -> KernelSpec {
    let body = mq2::kernel_spec(&mq2::SHIPPED);
    KernelSpec {
        kernel_id: "persistent_mq2_xbatch_dispatch".into(),
        variant: mq2::SHIPPED.tag(),
        symbol: DISPATCH_SYMBOL.into(),
        kernargs: KernargLayout::new(DISPATCH_KERNARG_BYTES)
            .pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly)
            .pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("B", 32, 4, "by_value")
            .pointer_access("queue", 40, Access::ReadWrite)
            .hidden("tasks", 48, 4, "by_value"),
        ..body
    }
}

/// The persistent dispatcher around the MQ2 body region (module docs).
pub fn dispatcher() -> Result<Emitted, String> {
    let p = mq2::SHIPPED;
    let entry = crate::plan::region_entry_label(REGION);
    let exit = crate::plan::region_exit_label(REGION);
    let next = ".Ldisp_next";
    let done = ".Ldisp_done";
    let mut regs = RegPlan::new(256, 64)?;
    // Caller-owned, live everywhere: the argument block pointer, the lane id and the
    // dispatcher's queue state (all disjoint from the body's registers).
    regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)?;
    regs.add_range("lane", Kind::V, 0, 1, Live::Whole)?;
    regs.add_range("queue", Kind::S, S_QUEUE, 2, Live::Whole)?;
    regs.add_range("tasks", Kind::S, S_TASKS, 1, Live::Whole)?;
    regs.add_range("task", Kind::S, S_TASK, 1, Live::Whole)?;
    // Dispatcher temporaries reuse body VGPRs v1..v3 outside the region: the plan proves
    // the lifetimes disjoint (`verify_lifetimes`), the ledger orders the accesses.
    for (name, base) in [("claim_offset", 1), ("claim_inc", 2), ("claim_old", 3)] {
        regs.add_range(
            name,
            Kind::V,
            base,
            1,
            Live::Between(next.into(), entry.clone()),
        )?;
    }
    for (name, base) in [("signal_offset", 1), ("signal_inc", 2)] {
        regs.add_range(
            name,
            Kind::V,
            base,
            1,
            Live::Between(exit.clone(), done.into()),
        )?;
    }
    mq2::plan_ranges(
        &mut regs,
        &p,
        Live::Between(entry.clone(), exit.clone()),
        false,
    )?;
    let mut b = Builder::new(dispatcher_spec(), regs);
    smem(
        &mut b,
        "s_load_b64 s[44:45], s[0:1], 0x28",
        &[sp(S_QUEUE)],
        &[sp(0)],
    )?;
    smem(
        &mut b,
        "s_load_b32 s46, s[0:1], 0x30",
        &[s(S_TASKS)],
        &[sp(0)],
    )?;
    b.wait_all()?;
    let queue = S::<2>(S_QUEUE);
    let abi = RegionAbi {
        name: REGION.into(),
        inputs: vec![
            ("args".into(), sp(0)),
            ("lane".into(), v(0)),
            ("virtual_workgroup".into(), s(S_TASK)),
        ],
        lds_bytes: 0,
        workgroup_size: 32,
    };
    b.loop_(next, |b| {
        // claim = atomic_add(&queue[0], 1) by lane 0, broadcast
        exec_lane0(b)?;
        op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        b.event_rmw(
            crate::sync::AtomicOp::AddU32,
            Some(V::<1>(3)),
            v(2),
            GlobalAddr {
                vaddr: V::<1>(1),
                base: queue,
                offset: 0,
            },
            Scope::Dev,
            Ordering::Relaxed,
        )?;
        op(b, "v_readfirstlane_b32 s47, v3", &[s(S_TASK)], &[v(3)])?;
        exec_all(b)?;
        op(b, "s_cmp_ge_u32 s47, s46", &[], &[s(S_TASK), s(S_TASKS)])?;
        branch(b, &format!("s_cbranch_scc1 {done}"))?;
        b.wait_all()?;
        b.region(&abi, |b| {
            mq2::emit_body(b, &p, mq2::WgId::Sgpr(S_TASK), ".Lmq2")
        })?;
        // The body leaves EXEC at lane 0 (its lane-0 store); publish this task's completion.
        exec_lane0(b)?;
        op(b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
        op(b, "v_mov_b32_e32 v2, 1", &[v(2)], &[])?;
        b.event_signal(
            V::<1>(2),
            GlobalAddr {
                vaddr: V::<1>(1),
                base: queue,
                offset: 4,
            },
            Scope::Dev,
            Ordering::Release,
        )?;
        exec_all(b)?;
        // The signal's address/data VGPRs are redefined by the next claim.
        b.release_store_sources()?;
        branch(b, &format!("s_branch {next}"))
    })?;
    b.label(done)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// The handoff kernel's spec (grid: 1 producer + N consumers, 32 threads each).
pub fn handoff_spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "pm_event_handoff".into(),
        variant: "release_acquire".into(),
        arch: Arch::Gfx1201,
        symbol: HANDOFF_SYMBOL.into(),
        kernargs: KernargLayout::new(HANDOFF_KERNARG_BYTES)
            .pointer_access("data", 0, Access::ReadWrite)
            .pointer_access("event", 8, Access::ReadWrite)
            .pointer_access("out", 16, Access::WriteOnly)
            .hidden("delay", 24, 4, "by_value")
            .hidden("budget", 28, 4, "by_value")
            .hidden("expected", 32, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// The two-workgroup release/acquire handoff (module docs). Kernarg: `data`
/// (`WORDS * 32` dwords, zero at launch), `event` (one dword, zero at launch),
/// `out` (`RECORD_DWORDS` per workgroup), `delay`, watchdog `budget`, `expected`.
pub fn handoff() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(64, 48)?;
    for (name, base, len) in [
        ("kernarg", 0u8, 2u8),
        ("data", 4, 2),
        ("event", 6, 2),
        ("out", 8, 2),
        ("delay", 10, 1),
        ("budget", 11, 1),
        ("expected", 12, 1),
        ("observed", 13, 1),
        ("wg", 14, 1),
        ("record", 16, 2),
        ("time", 18, 2),
        ("cycles_tmp", 20, 1),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    for (name, base, len) in [
        ("lane", 0u8, 1u8),
        ("lane_bytes", 1, 1),
        ("zero", 2, 1),
        ("one", 3, 1),
        ("poll", 4, 1),
        ("scalar_copy", 5, 1),
        ("pre", 8, 8),
        ("pre_hi", 16, 8),
        ("post", 24, 8),
        ("post_hi", 32, 8),
    ] {
        regs.add_range(name, Kind::V, base, len, Live::Whole)?;
    }
    let mut b = Builder::new(handoff_spec(), regs);
    smem(
        &mut b,
        "s_load_b128 s[4:7], s[0:1], 0x0",
        &[sp(4), sp(6)],
        &[sp(0)],
    )?;
    smem(
        &mut b,
        "s_load_b64 s[8:9], s[0:1], 0x10",
        &[sp(8)],
        &[sp(0)],
    )?;
    smem(
        &mut b,
        "s_load_b64 s[10:11], s[0:1], 0x18",
        &[s(10), s(11)],
        &[sp(0)],
    )?;
    smem(&mut b, "s_load_b32 s12, s[0:1], 0x20", &[s(12)], &[sp(0)])?;
    op(&mut b, "s_mov_b32 s14, ttmp9", &[s(14)], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_mov_b32_e32 v2, 0", &[v(2)], &[])?;
    op(&mut b, "v_mov_b32_e32 v3, 1", &[v(3)], &[])?;
    // record = out + wg * RECORD_DWORDS * 4
    op(
        &mut b,
        &format!("s_mul_i32 s16, s14, {:#x}", RECORD_DWORDS * 4),
        &[s(16)],
        &[s(14)],
    )?;
    op(&mut b, "s_mov_b32 s17, 0", &[s(17)], &[])?;
    b.wait_all()?;
    op(
        &mut b,
        "s_add_co_u32 s16, s8, s16",
        &[s(16)],
        &[s(8), s(16)],
    )?;
    op(
        &mut b,
        "s_add_co_ci_u32 s17, s9, s17",
        &[s(17)],
        &[s(9), s(17)],
    )?;
    let lane_bytes = V::<1>(1);
    let event = GlobalAddr {
        vaddr: V::<1>(2),
        base: S::<2>(6),
        offset: 0,
    };
    op(&mut b, "s_cmp_lg_u32 s14, 0", &[], &[s(14)])?;
    branch(&mut b, "s_cbranch_scc1 .Lconsumer")?;

    // ---- producer (workgroup 0)
    b.label(".Lproducer_delay")?;
    b.sleep(127)?;
    op(&mut b, "s_add_co_i32 s10, s10, -1", &[s(10)], &[s(10)])?;
    op(&mut b, "s_cmp_gt_i32 s10, 0", &[], &[s(10)])?;
    branch(&mut b, "s_cbranch_scc1 .Lproducer_delay")?;
    // data[j*32 + lane] = j*32 + lane + 1
    for j in 0..WORDS {
        let r = 24 + (j % 16) as u8;
        let value = j * 32 + 1;
        // inline constants print decimal (<= 64), literals hex
        let value = if value <= 64 {
            value.to_string()
        } else {
            format!("{value:#x}")
        };
        op(
            &mut b,
            &format!("v_add_nc_u32_e32 v{r}, {value}, v0"),
            &[v(r)],
            &[v(0)],
        )?;
        b.push(
            Instruction::new(
                format!("global_store_b32 v1, v{r}, s[4:5] offset:{}", j * 128),
                vec![],
                vec![lane_bytes.reg(), v(r), sp(4)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.realtime(S::<2>(18))?;
    exec_lane0(&mut b)?;
    b.event_signal(V::<1>(3), event, Scope::Dev, Ordering::Release)?;
    // record status words: time of publication
    store_status(&mut b, "s18", "s19", 1)?;
    branch(&mut b, "s_endpgm")?;

    // ---- consumers
    b.label(".Lconsumer")?;
    // prime: pre[j] = data[j*32 + lane]
    for j in 0..WORDS {
        let r = 8 + j as u8;
        b.push(
            Instruction::new(
                format!("global_load_b32 v{r}, v1, s[4:5] offset:{}", j * 128),
                vec![v(r)],
                vec![lane_bytes.reg(), sp(4)],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    for j in 0..WORDS {
        let r = 8 + j as u8;
        b.push(
            Instruction::new(
                format!(
                    "global_store_b32 v1, v{r}, s[16:17] offset:{}",
                    (WORDS * 32 + j * 32) * 4
                ),
                vec![],
                vec![lane_bytes.reg(), v(r), sp(16)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.wait_all()?;
    b.event_wait(
        ".Lspin",
        event,
        S::<1>(12),
        V::<1>(4),
        S::<1>(13),
        Scope::Dev,
        Ordering::Acquire,
        1,
        Some(&Watchdog {
            budget: S::<1>(11),
            timeout: ".Ltimeout".into(),
        }),
    )?;
    for j in 0..WORDS {
        let r = 24 + j as u8;
        b.push(
            Instruction::new(
                format!("global_load_b32 v{r}, v1, s[4:5] offset:{}", j * 128),
                vec![v(r)],
                vec![lane_bytes.reg(), sp(4)],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    for j in 0..WORDS {
        let r = 24 + j as u8;
        b.push(
            Instruction::new(
                format!("global_store_b32 v1, v{r}, s[16:17] offset:{}", j * 128),
                vec![],
                vec![lane_bytes.reg(), v(r), sp(16)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.realtime(S::<2>(18))?;
    exec_lane0(&mut b)?;
    store_status(&mut b, "s18", "s19", 1)?;
    branch(&mut b, "s_endpgm")?;

    // ---- watchdog: the event never reached `expected`
    b.label(".Ltimeout")?;
    op(&mut b, "s_mov_b32 s18, 0", &[s(18)], &[])?;
    op(&mut b, "s_mov_b32 s19, 0", &[s(19)], &[])?;
    exec_lane0(&mut b)?;
    store_status(&mut b, "s18", "s19", 0xdead)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Lane 0 writes `status, spins_left (s11), time_lo, time_hi` to the record's tail.
fn store_status(b: &mut Builder, lo: &str, hi: &str, status: u32) -> Result<(), String> {
    let tail = 2 * WORDS * 32 * 4;
    let reg = |t: &str| s(t[1..].parse().expect("sgpr"));
    op(b, &format!("v_mov_b32_e32 v8, {status:#x}"), &[v(8)], &[])?;
    op(b, "v_mov_b32_e32 v9, s11", &[v(9)], &[s(11)])?;
    op(b, &format!("v_mov_b32_e32 v10, {lo}"), &[v(10)], &[reg(lo)])?;
    op(b, &format!("v_mov_b32_e32 v11, {hi}"), &[v(11)], &[reg(hi)])?;
    let base = RegRef {
        kind: Kind::V,
        base: 8,
        len: 4,
    };
    b.push(
        Instruction::new(
            format!("global_store_b128 v2, v[8:11], s[16:17] offset:{tail}"),
            vec![],
            vec![v(2), base, sp(16)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
