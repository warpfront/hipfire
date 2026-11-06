// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 global memory model of the RDNA builder: typed global atomics,
//! agent/system-scope release and acquire, inter-workgroup event counters,
//! `s_sleep` and timer reads, and an independent checker of the emitted
//! release/acquire sequences.
//!
//! The sequences are the AMDGPU memory model's for GFX12 (LLVM
//! `SIMemoryLegalizer` `SIGfx12CacheControl`), confirmed against AMD clang
//! 24.0.0git (ROCm `d6f6cb69`) for `__hip_atomic_*` at agent and system
//! scope on gfx1201:
//!
//! * **release, agent** (`SCOPE_DEV`): `s_wait_loadcnt 0`, `s_wait_storecnt 0`
//!   before the releasing operation. A global store completes (STOREcnt
//!   decrements) once it is visible at its scope; RDNA4 vector caches are
//!   write-through, so no writeback is needed for the device.
//! * **release, system** (`SCOPE_SYS`): `global_wb scope:SCOPE_SYS` first (it
//!   writes back L2 lines another agent may read, and counts on STOREcnt), then
//!   the same waits.
//! * **acquire, agent/system**: the observing load or returning atomic, then
//!   `s_wait_loadcnt 0` (its value has returned) and `global_inv scope:X`, which
//!   invalidates the non-coherent vector caches below scope `X` so that later
//!   default-scope loads miss stale lines. LLVM emits no wait after the
//!   invalidate; neither does the builder.
//! * **atomics**: `global_atomic_<op>` at `scope:X`. With `th:TH_ATOMIC_RETURN`
//!   the pre-op value returns through LOADcnt (SIInsertWaitcnts
//!   `getVmemWaitEventType`: a returning atomic is a VMEM read access); without
//!   it the atomic counts on STOREcnt. The builder retires a returning atomic
//!   only with `s_wait_loadcnt 0` (it is not assumed in order with loads).
//! * **timers**: gfx12 has no `s_memtime`/`s_memrealtime` (llvm-mc: "instruction
//!   not supported on this GPU (gfx1201)"). The 64-bit shader clock is
//!   `s_getreg_b32 HW_REG_SHADER_CYCLES_{HI,LO,HI}` (LLVM
//!   `__builtin_readcyclecounter`); the constant 100 MHz clock is
//!   `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, returned through KMcnt
//!   (`__builtin_readsteadycounter`, `wall_clock64`).
//!
//! The builder tracks two obligations through the emission ledger
//! ([`SyncState`]): global writes not yet covered by a release at a scope (an
//! event signal declared `Relaxed` while they exist is refused) and an event
//! observed without an acquire (any later global data load is refused until
//! `acquire`). Both follow branches: a forward branch carries its state to its
//! target label, joined with the fall-through; a back edge must not carry an
//! obligation its head did not already assume.
//!
//! [`verify`] re-derives both rules from the emitted instruction text over the
//! program's control-flow graph, given only the [`SyncProof`] points (which
//! instruction is a declared release signal, which load is an event poll), so
//! a program whose release wait or acquire invalidate was removed is rejected.

use crate::insn::Program;
use crate::reg::{RegRef, S, V};
use serde::Serialize;

/// Memory scope of a gfx12 cache-policy `scope:` operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Scope {
    /// `SCOPE_CU`: the issuing compute unit (the default; printed as nothing).
    Cu,
    /// `SCOPE_SE`: the shader engine.
    Se,
    /// `SCOPE_DEV`: the agent (every workgroup of the device).
    Dev,
    /// `SCOPE_SYS`: the system (host and peer agents).
    Sys,
}
impl Scope {
    /// The trailing ` scope:SCOPE_X` operand (empty for the default `SCOPE_CU`).
    pub fn operand(self) -> &'static str {
        match self {
            Self::Cu => "",
            Self::Se => " scope:SCOPE_SE",
            Self::Dev => " scope:SCOPE_DEV",
            Self::Sys => " scope:SCOPE_SYS",
        }
    }
    fn of_text(text: &str) -> Scope {
        if text.contains("scope:SCOPE_SYS") {
            Self::Sys
        } else if text.contains("scope:SCOPE_DEV") {
            Self::Dev
        } else if text.contains("scope:SCOPE_SE") {
            Self::Se
        } else {
            Self::Cu
        }
    }
}

/// Ordering of a synchronizing operation, as in the C++/HSA memory model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}
impl Ordering {
    pub fn releases(self) -> bool {
        matches!(self, Self::Release | Self::AcqRel)
    }
    pub fn acquires(self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel)
    }
}

/// 32-bit global atomic read-modify-write operations (gfx12 `global_atomic_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AtomicOp {
    AddU32,
    SubU32,
    /// `old >= data ? 0 : old + 1`
    IncU32,
    /// `old == 0 || old > data ? data : old - 1`
    DecU32,
    SwapB32,
    /// data is the pair `{new, compare}` (`v[n:n+1]`).
    CmpSwapB32,
    MinU32,
    MaxU32,
    AndB32,
    OrB32,
    XorB32,
}
impl AtomicOp {
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::AddU32 => "global_atomic_add_u32",
            Self::SubU32 => "global_atomic_sub_u32",
            Self::IncU32 => "global_atomic_inc_u32",
            Self::DecU32 => "global_atomic_dec_u32",
            Self::SwapB32 => "global_atomic_swap_b32",
            Self::CmpSwapB32 => "global_atomic_cmpswap_b32",
            Self::MinU32 => "global_atomic_min_u32",
            Self::MaxU32 => "global_atomic_max_u32",
            Self::AndB32 => "global_atomic_and_b32",
            Self::OrB32 => "global_atomic_or_b32",
            Self::XorB32 => "global_atomic_xor_b32",
        }
    }
    /// Data VGPRs the operation reads (cmpswap: new value and compare value).
    pub fn data_len(self) -> u8 {
        if self == Self::CmpSwapB32 {
            2
        } else {
            1
        }
    }
}

/// A global address in saddr form: `base + vaddr + offset` (gfx12 IOFFSET is a
/// signed 24-bit byte offset; the builder admits `0..2^23`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalAddr {
    pub vaddr: V<1>,
    pub base: S<2>,
    pub offset: u32,
}
impl GlobalAddr {
    pub(crate) fn check(self) -> Result<(), String> {
        if self.offset >= 1 << 23 {
            return Err(format!(
                "global offset {} exceeds the signed 24-bit field",
                self.offset
            ));
        }
        Ok(())
    }
    pub(crate) fn suffix(self) -> String {
        if self.offset == 0 {
            String::new()
        } else {
            format!(" offset:{}", self.offset)
        }
    }
}

/// Bounded spin for [`crate::Builder::event_wait`]: `budget` (caller
/// initialized) counts polls; when it reaches zero the wait branches to
/// `timeout` (a label the caller places) without acquiring.
#[derive(Clone, Debug)]
pub struct Watchdog {
    pub budget: S<1>,
    pub timeout: String,
}

/// One declared synchronization point of an emitted program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SyncProof {
    /// Instruction index in `Program::instructions`.
    pub pc_index: usize,
    pub kind: SyncKind,
    pub scope: Scope,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SyncKind {
    /// A release-ordered event increment: every earlier global access is
    /// complete (and, at system scope, written back) before it issues.
    ReleaseSignal,
    /// A relaxed event increment or relaxed RMW (claim counters): it publishes
    /// nothing, so no global write may be pending a release before it.
    RelaxedSignal,
    /// An event poll (the observing load): data loads after it need an acquire.
    Poll,
    /// The `global_inv` completing an acquire.
    Acquire,
    /// A standalone release fence; `pc_index` is its last wait, after which the
    /// ordering holds.
    Release,
}

/// The builder's synchronization obligations at one program point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncState {
    /// Global writes not covered by a device-scope release.
    pub unreleased_dev: bool,
    /// Global writes not covered by a system-scope release.
    pub unreleased_sys: bool,
    /// An event was observed (relaxed) and not yet acquired.
    pub unacquired: bool,
}
impl SyncState {
    pub fn join(self, other: SyncState) -> SyncState {
        SyncState {
            unreleased_dev: self.unreleased_dev | other.unreleased_dev,
            unreleased_sys: self.unreleased_sys | other.unreleased_sys,
            unacquired: self.unacquired | other.unacquired,
        }
    }
    /// Every obligation of `self` is already assumed by `other`.
    pub fn within(self, other: SyncState) -> bool {
        self.join(other) == other
    }
    pub fn write(&mut self) {
        self.unreleased_dev = true;
        self.unreleased_sys = true;
    }
    pub fn unreleased(self, scope: Scope) -> bool {
        match scope {
            Scope::Sys => self.unreleased_sys,
            _ => self.unreleased_dev,
        }
    }
    pub fn release(&mut self, scope: Scope) {
        self.unreleased_dev = false;
        if scope == Scope::Sys {
            self.unreleased_sys = false
        }
    }
}

/// The target label of a branch instruction (`s_branch L`, `s_cbranch_* L`).
pub fn branch_target(text: &str) -> Option<&str> {
    let mnemonic = text.split_whitespace().next()?;
    if mnemonic == "s_branch" || mnemonic.starts_with("s_cbranch_") {
        text.split_whitespace().nth(1)
    } else {
        None
    }
}

pub(crate) fn data_reg(data: RegRef, op: AtomicOp) -> Result<(), String> {
    if data.kind != crate::reg::Kind::V || data.len != op.data_len() {
        return Err(format!(
            "{} needs {} data VGPR(s), got {data}",
            op.mnemonic(),
            op.data_len()
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ checker

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Flow {
    obligations: SyncState,
    /// A store, returnless atomic or writeback issued since `s_wait_storecnt 0`.
    stores: bool,
    /// A load or returning atomic issued since `s_wait_loadcnt 0`.
    loads: bool,
    /// A `global_wb scope:SCOPE_SYS` issued since the last global write and
    /// not yet retired by `s_wait_storecnt 0`.
    wb_pending: bool,
    /// The poll of an unacquired observation has not returned yet.
    poll_in_flight: bool,
}
impl Flow {
    fn join(self, o: Flow) -> Flow {
        Flow {
            obligations: self.obligations.join(o.obligations),
            stores: self.stores | o.stores,
            loads: self.loads | o.loads,
            // a writeback must be pending on every path to count
            wb_pending: self.wb_pending & o.wb_pending,
            poll_in_flight: self.poll_in_flight | o.poll_in_flight,
        }
    }
}

fn is_label(text: &str) -> Option<&str> {
    let t = text.trim();
    t.strip_suffix(':').filter(|l| !l.contains(' '))
}

/// Re-derive the release/acquire obligations of `program` from its text:
/// a [`SyncKind::ReleaseSignal`] point needs every earlier global load and
/// store retired (`s_wait_loadcnt 0`, `s_wait_storecnt 0`) and, at system
/// scope, a system writeback after the last global write; a
/// [`SyncKind::RelaxedSignal`] point needs no global write pending a release;
/// a global data load after a [`SyncKind::Poll`] needs the poll retired and a
/// `global_inv` of device or system scope in between. Dataflow over the
/// control-flow graph (labels, `s_branch`, `s_cbranch_*`, `s_endpgm`) to a
/// fixed point; obligations join by union.
pub fn verify(program: &Program, proofs: &[SyncProof]) -> Result<(), String> {
    let insns = &program.instructions;
    let n = insns.len();
    let labels: std::collections::HashMap<&str, usize> = insns
        .iter()
        .enumerate()
        .filter_map(|(i, x)| is_label(&x.text).map(|l| (l, i)))
        .collect();
    let mut at: Vec<Option<Flow>> = vec![None; n + 1];
    at[0] = Some(Flow::default());
    let mut work: Vec<usize> = vec![0];
    let proof_at = |pc: usize| proofs.iter().filter(move |p| p.pc_index == pc);
    while let Some(pc) = work.pop() {
        if pc >= n {
            continue;
        }
        let mut f = at[pc].expect("worklist entries have a state");
        let text = insns[pc].text.trim();
        let mnemonic = text.split_whitespace().next().unwrap_or("");
        for p in proof_at(pc) {
            match p.kind {
                SyncKind::ReleaseSignal => {
                    if f.loads || f.stores {
                        return Err(format!(
                            "pc {pc} `{text}`: release signal with global accesses outstanding \
                             (missing s_wait_loadcnt/s_wait_storecnt 0 before it)"
                        ));
                    }
                    if p.scope == Scope::Sys && f.obligations.unreleased_sys {
                        return Err(format!(
                            "pc {pc} `{text}`: system-scope release without global_wb scope:SCOPE_SYS"
                        ));
                    }
                    f.obligations.release(p.scope);
                }
                // checked after the instruction (the fence's last wait), below
                SyncKind::Release => {}
                SyncKind::RelaxedSignal => {
                    if f.obligations.unreleased(p.scope) {
                        return Err(format!(
                            "pc {pc} `{text}`: relaxed signal while global writes are pending a \
                             release (missing release)"
                        ));
                    }
                }
                SyncKind::Poll | SyncKind::Acquire => {}
            }
        }
        let is_poll = proof_at(pc).any(|p| p.kind == SyncKind::Poll);
        let global = mnemonic.starts_with("global_") || mnemonic.starts_with("buffer_");
        let atomic = mnemonic.contains("_atomic_");
        let returns = atomic && text.contains("TH_ATOMIC_RETURN");
        let load = global && mnemonic.contains("_load") || returns;
        let write = global && (mnemonic.contains("_store") || atomic);
        // The model does not rely on `global_inv` covering the scalar data cache, so an SMEM
        // read is refused while an observation is unacquired (and published data is read
        // with vector loads after the acquire).
        let scalar_load = mnemonic.starts_with("s_load") || mnemonic.starts_with("s_buffer_load");
        if (load || scalar_load) && !is_poll && f.obligations.unacquired {
            return Err(format!(
                "pc {pc} `{text}`: global load after an event poll without an acquire \
                 (missing s_wait_loadcnt 0 + global_inv scope:SCOPE_DEV/SYS)"
            ));
        }
        match mnemonic {
            "s_wait_loadcnt" if text.ends_with(" 0x0") || text.ends_with(" 0") => {
                f.loads = false;
                f.poll_in_flight = false;
            }
            "s_wait_loadcnt_dscnt" => {
                let imm = text
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                if imm.is_some_and(|v| v >> 8 == 0) {
                    f.loads = false;
                    f.poll_in_flight = false;
                }
            }
            "s_wait_storecnt" if text.ends_with(" 0x0") || text.ends_with(" 0") => {
                f.stores = false;
                if f.wb_pending {
                    f.obligations.unreleased_sys = false;
                    f.wb_pending = false;
                }
            }
            "global_wb" => {
                f.stores = true;
                if Scope::of_text(text) == Scope::Sys {
                    f.wb_pending = true;
                }
            }
            "global_inv" => {
                if Scope::of_text(text) >= Scope::Dev && !f.poll_in_flight {
                    f.obligations.unacquired = false;
                }
            }
            _ => {}
        }
        if load {
            f.loads = true;
        }
        // A returning atomic counts on LOADcnt only; a returnless one on STOREcnt.
        if write && !returns {
            f.stores = true;
        }
        // Declared event/claim increments are synchronization variables, not data: they
        // publish nothing themselves, so they create no release obligation.
        let sync_variable = proof_at(pc)
            .any(|p| matches!(p.kind, SyncKind::ReleaseSignal | SyncKind::RelaxedSignal));
        if write && mnemonic != "global_wb" && !sync_variable {
            f.obligations.write();
            f.wb_pending = false;
        }
        if is_poll {
            f.obligations.unacquired = true;
            f.poll_in_flight = true;
        }
        for p in proof_at(pc).filter(|p| p.kind == SyncKind::Release) {
            if f.loads || f.stores {
                return Err(format!(
                    "pc {pc}: release fence with global accesses outstanding"
                ));
            }
            if p.scope == Scope::Sys && f.obligations.unreleased_sys {
                return Err(format!(
                    "pc {pc}: system-scope release without global_wb scope:SCOPE_SYS"
                ));
            }
            f.obligations.release(p.scope);
        }
        let mut succ: Vec<usize> = Vec::new();
        if mnemonic == "s_endpgm" {
        } else if let Some(target) = branch_target(text) {
            let t = *labels
                .get(target)
                .ok_or_else(|| format!("pc {pc}: branch to unknown label {target}"))?;
            succ.push(t);
            if mnemonic != "s_branch" {
                succ.push(pc + 1);
            }
        } else {
            succ.push(pc + 1);
        }
        for s in succ {
            let next = match at[s] {
                None => f,
                Some(old) => old.join(f),
            };
            if at[s] != Some(next) {
                at[s] = Some(next);
                work.push(s);
            }
        }
    }
    Ok(())
}
