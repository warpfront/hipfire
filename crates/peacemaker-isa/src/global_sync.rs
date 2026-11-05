// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder entry points of the gfx1201 global memory model (`crate::sync` has
//! the sequences and their sources) and of the task-body region ABI.
//!
//! **Region ABI.** [`Builder::region`] emits a task body as a region of the
//! caller's program, between labels `.L{name}_entry` and `.L{name}_exit`:
//!
//! * entry state: the declared inputs (caller-owned registers, e.g. the body's
//!   argument-block pointer, the lane id and the virtual workgroup coordinate)
//!   are live and are never written by the body; every memory operation of the
//!   caller is retired (empty ledger); no event observation is unacquired;
//!   the caller's physical workgroup size is the body's;
//! * inside: no `s_endpgm`, no hardware workgroup id (`ttmp7`/`ttmp9`), no
//!   branch out of the region except to its exit, and no branch from outside
//!   into it except at its entry; body-local registers are ranges live only
//!   `Between(entry, exit)`, so the shared register plan proves them disjoint
//!   from every caller range live inside the region and free outside it;
//! * exit state: every body memory operation retired; the body's global writes
//!   are still pending a release (the caller releases before it publishes
//!   completion); EXEC may be clobbered (recorded, the caller restores it).

use crate::insn::{Instruction, MemoryClass, Program};
use crate::ledger::{Counter, Reason};
use crate::plan::{RegionAbi, RegionProof};
use crate::reg::{Kind, Live, RegRef, S, V};
use crate::sync::{self, AtomicOp, GlobalAddr, Ordering, Scope, SyncKind, SyncProof, Watchdog};
use crate::Builder;

fn need_device_scope(what: &str, scope: Scope) -> Result<(), String> {
    if scope < Scope::Dev {
        return Err(format!(
            "{what}: inter-workgroup synchronization needs SCOPE_DEV or SCOPE_SYS, not {scope:?} \
             (workgroup scope is the LDS barrier)"
        ));
    }
    Ok(())
}

fn atomic_insn(
    op: AtomicOp,
    ret: Option<V<1>>,
    data: RegRef,
    addr: GlobalAddr,
    scope: Scope,
) -> Result<Instruction, String> {
    addr.check()?;
    sync::data_reg(data, op)?;
    let text = format!(
        "{} {}{}, {data}, {}{}{}{}",
        op.mnemonic(),
        ret.map(|r| format!("{}, ", r.reg())).unwrap_or_default(),
        addr.vaddr.reg(),
        addr.base.reg(),
        addr.suffix(),
        if ret.is_some() {
            " th:TH_ATOMIC_RETURN"
        } else {
            ""
        },
        scope.operand()
    );
    Ok(Instruction::new(
        text,
        ret.map(|r| vec![r.reg()]).unwrap_or_default(),
        vec![addr.vaddr.reg(), data, addr.base.reg()],
    )
    .memory(MemoryClass::VmemAtomic {
        returns: ret.is_some(),
    }))
}

fn sgpr(base: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base,
        len: 1,
    }
}

impl Builder {
    fn gfx12_untyped(&self, what: &str) -> Result<(), String> {
        self.untyped()?;
        if !self.spec.arch.gfx12() {
            return Err(format!(
                "{what}: the global memory model is implemented for gfx12 (gfx1201) only"
            ));
        }
        Ok(())
    }
    /// Emit a synchronization access (an event poll, an event RMW, a writeback or a timer
    /// read): it neither creates a release obligation nor is refused as a data load.
    fn emit_sync_access(&mut self, insn: Instruction) -> Result<usize, String> {
        self.sync_access = true;
        let result = self.emit(insn);
        self.sync_access = false;
        result.map(|()| self.program.instructions.len() - 1)
    }
    /// A 32-bit global atomic on data (a histogram bin, a reduction slot): with `ret` the
    /// pre-op value returns (LOADcnt), else it counts on STOREcnt. It is a global write:
    /// a later release covers it. Synchronization variables use [`Builder::event_rmw`].
    pub fn global_atomic(
        &mut self,
        op: AtomicOp,
        ret: Option<V<1>>,
        data: RegRef,
        addr: GlobalAddr,
        scope: Scope,
    ) -> Result<(), String> {
        self.gfx12_untyped("global_atomic")?;
        let insn = atomic_insn(op, ret, data, addr, scope)?;
        self.emit(insn)
    }
    /// The release sequence's waits (and system writeback) without a proof point.
    fn release_waits(&mut self, scope: Scope) -> Result<(), String> {
        need_device_scope("release", scope)?;
        if scope == Scope::Sys {
            // Writes back L2 lines another agent may read; counts on STOREcnt.
            self.emit_sync_access(
                Instruction::new("global_wb scope:SCOPE_SYS", vec![], vec![])
                    .memory(MemoryClass::VmemStore),
            )?;
        }
        // Unconditional: the ledger forgets stores whose sources were released
        // (`release_store_sources`), so it cannot prove STOREcnt is already zero.
        self.emit_wait(Counter::Load, 0, Reason::Barrier)?;
        self.emit_wait(Counter::Store, 0, Reason::Barrier)
    }
    /// A release fence at `scope`: every earlier global load and store is complete (and at
    /// system scope written back) before any later operation issues.
    pub fn release(&mut self, scope: Scope) -> Result<(), String> {
        self.gfx12_untyped("release")?;
        self.release_waits(scope)?;
        self.syncs.push(SyncProof {
            pc_index: self.program.instructions.len(),
            kind: SyncKind::Release,
            scope,
        });
        self.sync.release(scope);
        Ok(())
    }
    /// An acquire at `scope` completing an observation: the observing load or RMW has
    /// returned (`s_wait_loadcnt 0`), then `global_inv` drops stale vector-cache lines below
    /// `scope`, so later loads see every write released before the observed value.
    pub fn acquire(&mut self, scope: Scope) -> Result<(), String> {
        self.gfx12_untyped("acquire")?;
        need_device_scope("acquire", scope)?;
        self.emit_wait(Counter::Load, 0, Reason::Barrier)?;
        let text = format!("global_inv{}", scope.operand());
        let pc_index = self.program.instructions.len();
        self.program
            .instructions
            .push(Instruction::new(text, vec![], vec![]));
        self.syncs.push(SyncProof {
            pc_index,
            kind: SyncKind::Acquire,
            scope,
        });
        self.sync.unacquired = false;
        Ok(())
    }
    /// An RMW on a synchronization variable (event counter, claim counter) at device or
    /// system scope. `Release`/`AcqRel` release every earlier global access first;
    /// `Relaxed` is refused while a global write is pending a release at `scope` (it would
    /// publish before the data is visible). `Acquire`/`AcqRel` need `ret` and acquire after
    /// it returns. Every active lane performs the RMW: callers counting waves restrict EXEC.
    pub fn event_rmw(
        &mut self,
        op: AtomicOp,
        ret: Option<V<1>>,
        data: RegRef,
        cell: GlobalAddr,
        scope: Scope,
        ordering: Ordering,
    ) -> Result<(), String> {
        self.gfx12_untyped("event_rmw")?;
        need_device_scope("event_rmw", scope)?;
        if ordering.acquires() && ret.is_none() {
            return Err(
                "an acquiring event RMW needs its return value (th:TH_ATOMIC_RETURN)".into(),
            );
        }
        let insn = atomic_insn(op, ret, data, cell, scope)?;
        if ordering.releases() {
            self.release_waits(scope)?;
        } else if self.sync.unreleased(scope) {
            return Err(format!(
                "`{}`: Relaxed event RMW while global writes are pending a {scope:?} release \
                 (missing release: use Ordering::Release)",
                insn.text
            ));
        }
        let pc_index = self.emit_sync_access(insn)?;
        self.syncs.push(SyncProof {
            pc_index,
            kind: if ordering.releases() {
                SyncKind::ReleaseSignal
            } else {
                SyncKind::RelaxedSignal
            },
            scope,
        });
        if ordering.releases() {
            self.sync.release(scope);
        }
        if ordering.acquires() {
            self.acquire(scope)?;
        }
        Ok(())
    }
    /// `event_rmw(AddU32, None, inc, ...)`: add `inc` to an event counter.
    pub fn event_signal(
        &mut self,
        inc: V<1>,
        cell: GlobalAddr,
        scope: Scope,
        ordering: Ordering,
    ) -> Result<(), String> {
        self.event_rmw(AtomicOp::AddU32, None, inc.reg(), cell, scope, ordering)
    }
    /// Spin until the event counter at `cell` reaches `expected` (unsigned `>=`):
    ///
    /// ```text
    /// {head}:  global_load_b32 value, vaddr, base offset:o scope:X   ; the poll
    ///          s_wait_loadcnt 0x0
    ///          v_readfirstlane_b32 value_s, value
    ///          s_cmp_ge_u32 value_s, expected
    ///          s_cbranch_scc1 {head}_ready
    ///          [s_add_co_i32 budget, budget, -1 / s_cmp_eq_u32 budget, 0 / s_cbranch_scc1 timeout]
    ///          s_sleep N
    ///          s_branch {head}
    /// {head}_ready:
    ///          [Acquire: s_wait_loadcnt 0x0 / global_inv scope:X]
    /// ```
    ///
    /// The device-scope poll misses stale vector-cache lines, so the spin makes progress.
    /// With `Ordering::Relaxed` the observation stays unacquired: every later load is
    /// refused until [`Builder::acquire`]. The watchdog's timeout path is unacquired too.
    #[allow(clippy::too_many_arguments)]
    pub fn event_wait(
        &mut self,
        head: &str,
        cell: GlobalAddr,
        expected: S<1>,
        value: V<1>,
        value_s: S<1>,
        scope: Scope,
        ordering: Ordering,
        sleep: u8,
        watchdog: Option<&Watchdog>,
    ) -> Result<(), String> {
        self.gfx12_untyped("event_wait")?;
        need_device_scope("event_wait", scope)?;
        cell.check()?;
        if ordering.releases() {
            return Err("event_wait observes; release with event_signal".into());
        }
        if sleep > 127 {
            return Err(format!("s_sleep {sleep} exceeds the 7-bit sleep count"));
        }
        // The spin drains LOADcnt each trip: start from no pending loads so its back-edge
        // ledger equals its head's (the `loop_` fixpoint).
        if self.ledger.has_pending(Counter::Load) {
            self.emit_wait(Counter::Load, 0, Reason::LoopFixpoint)?;
        }
        let ready = format!("{head}_ready");
        let poll = Instruction::new(
            format!(
                "global_load_b32 {}, {}, {}{}{}",
                value.reg(),
                cell.vaddr.reg(),
                cell.base.reg(),
                cell.suffix(),
                scope.operand()
            ),
            vec![value.reg()],
            vec![cell.vaddr.reg(), cell.base.reg()],
        )
        .memory(MemoryClass::VmemLoad);
        self.fixpoint_loop(head, |b| {
            let pc_index = b.emit_sync_access(poll.clone())?;
            b.syncs.push(SyncProof {
                pc_index,
                kind: SyncKind::Poll,
                scope,
            });
            b.sync.unacquired = true;
            b.emit(Instruction::new(
                format!("v_readfirstlane_b32 {}, {}", value_s.reg(), value.reg()),
                vec![value_s.reg()],
                vec![value.reg()],
            ))?;
            b.emit(Instruction::new(
                format!("s_cmp_ge_u32 {}, {}", value_s.reg(), expected.reg()),
                vec![],
                vec![value_s.reg(), expected.reg()],
            ))?;
            let exit = format!("s_cbranch_scc1 {ready}");
            b.branch_sync(&exit)?;
            b.emit(Instruction::new(exit, vec![], vec![]))?;
            if let Some(w) = watchdog {
                let n = w.budget.reg();
                b.emit(Instruction::new(
                    format!("s_add_co_i32 {n}, {n}, -1"),
                    vec![n],
                    vec![n],
                ))?;
                b.emit(Instruction::new(
                    format!("s_cmp_eq_u32 {n}, 0"),
                    vec![],
                    vec![n],
                ))?;
                let out = format!("s_cbranch_scc1 {}", w.timeout);
                b.branch_sync(&out)?;
                b.emit(Instruction::new(out, vec![], vec![]))?;
            }
            b.emit(Instruction::new(format!("s_sleep {sleep}"), vec![], vec![]))?;
            // The back edge carries the unacquired poll to a head that only polls again:
            // it bypasses `branch_sync`'s loop-head check by construction.
            b.emit(Instruction::new(format!("s_branch {head}"), vec![], vec![]))
        })?;
        self.label(&ready)?;
        if ordering.acquires() {
            self.acquire(scope)?;
        }
        Ok(())
    }
    /// `s_sleep n`: about `64 * n` clocks (RDNA4 ISA SOPP S_SLEEP; 0 = no sleep).
    pub fn sleep(&mut self, n: u8) -> Result<(), String> {
        if n > 127 {
            return Err(format!("s_sleep {n} exceeds the 7-bit sleep count"));
        }
        self.emit(Instruction::new(format!("s_sleep {n}"), vec![], vec![]))
    }
    /// The 64-bit shader clock into `dst` (lo, hi), as LLVM's gfx12
    /// `__builtin_readcyclecounter`: HI, LO, HI again; a changed HI means LO wrapped
    /// between the reads, and the result is `(0, new hi)`.
    pub fn shader_cycles(&mut self, dst: S<2>, tmp: S<1>) -> Result<(), String> {
        self.gfx12_untyped("shader_cycles")?;
        let (lo, hi, t) = (sgpr(dst.base()), sgpr(dst.base() + 1), tmp.reg());
        for (text, def) in [
            (
                format!("s_getreg_b32 {t}, hwreg(HW_REG_SHADER_CYCLES_HI)"),
                t,
            ),
            (
                format!("s_getreg_b32 {lo}, hwreg(HW_REG_SHADER_CYCLES_LO)"),
                lo,
            ),
            (
                format!("s_getreg_b32 {hi}, hwreg(HW_REG_SHADER_CYCLES_HI)"),
                hi,
            ),
        ] {
            self.emit(Instruction::new(text, vec![def], vec![]))?;
        }
        self.emit(Instruction::new(
            format!("s_cmp_eq_u32 {t}, {hi}"),
            vec![],
            vec![t, hi],
        ))?;
        self.emit(Instruction::new(
            format!("s_cselect_b32 {lo}, {lo}, 0"),
            vec![lo],
            vec![lo],
        ))
    }
    /// The constant 100 MHz clock into `dst` (`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`):
    /// it returns through KMcnt, so the ledger waits before `dst` is read.
    pub fn realtime(&mut self, dst: S<2>) -> Result<(), String> {
        self.gfx12_untyped("realtime")?;
        self.emit_sync_access(
            Instruction::new(
                format!(
                    "s_sendmsg_rtn_b64 {}, sendmsg(MSG_RTN_GET_REALTIME)",
                    dst.reg()
                ),
                vec![dst.reg()],
                vec![],
            )
            .memory(MemoryClass::SmemLoad),
        )
        .map(|_| ())
    }
    /// Emit `body` as a task-body region (module docs: the region ABI).
    pub fn region(
        &mut self,
        abi: &RegionAbi,
        body: impl FnOnce(&mut Builder) -> Result<(), String>,
    ) -> Result<(), String> {
        self.untyped()?;
        self.flush_valu()?;
        let name = &abi.name;
        if let Some(open) = &self.open_region {
            return Err(format!("region {name} inside region {open}"));
        }
        if !self.ledger.is_empty() {
            return Err(format!(
                "region {name}: the caller's memory operations must be retired at entry"
            ));
        }
        if self.sync.unacquired {
            return Err(format!(
                "region {name}: entered with an unacquired event observation"
            ));
        }
        if abi.workgroup_size != self.spec.workgroup_size {
            return Err(format!(
                "region {name}: body geometry {} threads != caller workgroup {}",
                abi.workgroup_size, self.spec.workgroup_size
            ));
        }
        if abi.lds_bytes > self.spec.group_segment_fixed_size {
            return Err(format!(
                "region {name}: {} LDS bytes exceed the caller's {}",
                abi.lds_bytes, self.spec.group_segment_fixed_size
            ));
        }
        let (entry, exit) = (
            crate::plan::region_entry_label(name),
            crate::plan::region_exit_label(name),
        );
        self.label(&entry)?;
        let entry_pc = self.program.instructions.len() - 1;
        for (meaning, r) in &abi.inputs {
            self.regs
                .verify_access(*r, &entry, &self.labels)
                .map_err(|e| format!("region {name} input {meaning}: {e}"))?;
        }
        self.open_region = Some(name.clone());
        let result = body(self);
        self.open_region = None;
        result?;
        self.label(&exit)?;
        let exit_pc = self.program.instructions.len() - 1;
        for (meaning, r) in &abi.inputs {
            self.regs
                .verify_access(*r, &exit, &self.labels)
                .map_err(|e| format!("region {name} input {meaning} at exit: {e}"))?;
            if let Some(i) = self.program.instructions[entry_pc..exit_pc]
                .iter()
                .find(|i| i.defs.iter().any(|d| d.overlaps(*r)))
            {
                return Err(format!(
                    "region {name}: `{}` writes the input {meaning} ({r})",
                    i.text
                ));
            }
        }
        if !self.ledger.is_empty() {
            return Err(format!(
                "region {name}: the body must retire its memory operations before its exit"
            ));
        }
        let inner: Vec<&str> = self.program.instructions[entry_pc + 1..=exit_pc]
            .iter()
            .filter_map(|i| i.text.strip_suffix(':'))
            .collect();
        for i in &self.program.instructions[entry_pc..exit_pc] {
            if let Some(t) = sync::branch_target(&i.text) {
                if !inner.contains(&t) {
                    return Err(format!("region {name}: `{}` leaves the region", i.text));
                }
            }
        }
        let exec_clobbered = self.program.instructions[entry_pc..exit_pc]
            .iter()
            .any(|i| {
                let m = i.mnemonic();
                m.starts_with("v_cmpx")
                    || m.contains("saveexec")
                    || i.text
                        .split_whitespace()
                        .nth(1)
                        .is_some_and(|d| d.trim_end_matches(',').starts_with("exec"))
            });
        let between = Live::Between(entry.clone(), exit.clone());
        let locals = self
            .regs
            .ranges
            .iter()
            .filter(|r| r.live == between)
            .map(|r| format!("{} {}", r.name, r.reg()))
            .collect();
        self.regions.push(RegionProof {
            name: name.clone(),
            entry_pc,
            exit_pc,
            inputs: abi
                .inputs
                .iter()
                .map(|(m, r)| (m.clone(), r.to_string()))
                .collect(),
            locals,
            lds_bytes: abi.lds_bytes,
            workgroup_size: abi.workgroup_size,
            exec_clobbered,
            unreleased_writes_at_exit: self.sync.unreleased_dev,
        });
        Ok(())
    }
}

/// No branch outside a region targets a label strictly inside it (a region is
/// entered only at its entry label).
pub(crate) fn check_region_entries(
    program: &Program,
    regions: &[RegionProof],
) -> Result<(), String> {
    for r in regions {
        let inner: Vec<&str> = program.instructions[r.entry_pc + 1..r.exit_pc]
            .iter()
            .filter_map(|i| i.text.strip_suffix(':'))
            .collect();
        for (pc, i) in program.instructions.iter().enumerate() {
            if (r.entry_pc..r.exit_pc).contains(&pc) {
                continue;
            }
            if let Some(t) = sync::branch_target(&i.text) {
                if inner.contains(&t) {
                    return Err(format!(
                        "`{}` (pc {pc}) jumps into region {} past its entry",
                        i.text, r.name
                    ));
                }
            }
        }
    }
    Ok(())
}
