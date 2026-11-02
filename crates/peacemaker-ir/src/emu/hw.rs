// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The hardware-fidelity ("gate") model over the in-order interpreter, enabled by `Emulator::hw_model()`.
//!
//! The plain interpreter retires every load at issue and runs the waves in one fixed interleaving, so a kernel can
//! pass it and still fault on a card. The gate model closes three holes, each failing closed:
//!
//! 1. **In-flight memory.** Every VMEM / LDS / SMEM instruction with a counter effect is issued into a per-wave
//!    pending list (issue order). A load's destination registers are written only when the matching
//!    `s_waitcnt`/`s_wait_*` retires it, using the same rule table as the static M7 replay
//!    (`passes::waits::retire`): gfx11 `vmcnt`/`lgkmcnt` retire in order, the store counter and `kmcnt` only at 0, a
//!    positive `lgkmcnt` retires nothing while an SMEM is pending (SMEM returns out of order), `loadcnt` retires
//!    partially only inside one in-order family. Touching (reading or writing) a register that a pending load
//!    will still overwrite, or redefining a VGPR that a pending store has not yet read, is an error naming the
//!    instruction and the pending one. Two loads of the same in-order class may write one register back to back
//!    (their results land in issue order).
//! 2. **LDS races.** Every LDS access is stamped with its wave and an access id. A wave's access is *fenced* when
//!    its wait counter retires it, stamped with the number of barriers the wave had arrived at by then. An access
//!    by another wave to the same byte (read-after-write, write-after-write, write-after-read; atomic against
//!    atomic is exempt) must be ordered: the first access retired before its wave arrived at barrier `k`, and the
//!    second wave had passed barrier `k` before it issued. A write still in flight at the barrier (no
//!    `lgkmcnt` wait before `s_barrier`) is therefore a race with the next epoch's accesses.
//! 3. **Schedules.** Each multi-wave workgroup runs under round-robin, "wave w runs ahead until blocked" for every
//!    wave, and two seeded-random interleavings, on a store journal so the runs start from the same memory; the
//!    stores must be bit-identical across all of them.
use super::{
    alu, control, failure_bits,
    memory::{self, LdsTouch},
    Class, Emulator, Launch, Memory, Result, State, Step, StoreImage, WriteSet,
};
use crate::{
    codec::gfx12::operand_register,
    effects::{MemClass, SrcRead},
    inst::{Abi, Wave},
    passes::waits::{family_of, is_load, operand_mask, unit_weights, wait_alu_bits},
    reg::{Kind, RegRef},
    wait::{Counter, CounterSet, N},
    Arch,
};

const NO_WAVE: u8 = u8::MAX;
const NO_ACC: u32 = u32::MAX;
const ATOMIC: u32 = 1 << 31;
const UNRETIRED: u32 = u32::MAX;
/// Workgroups per emulator that run under every schedule (see `run_hw`).
const FULL_SCHEDULE_WGS: u64 = 16;

/// What a run did, for receipts.
#[derive(Clone, Debug, Default)]
pub struct RunReport {
    /// Instructions executed by schedule 0 (round-robin).
    pub instructions: u64,
    /// Names of the schedules run and compared.
    pub schedules: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Schedule {
    RoundRobin,
    Ahead(usize),
    Random(u64),
}
impl Schedule {
    fn name(self) -> String {
        match self {
            Schedule::RoundRobin => "round-robin".into(),
            Schedule::Ahead(w) => format!("wave{w}-runs-ahead"),
            Schedule::Random(s) => format!("random(seed={s:#x})"),
        }
    }
}

/// One register dword an instruction reads and/or writes, with the bits of it that matter (d16 halves).
#[derive(Clone, Copy, Debug)]
struct Touch {
    kind: Kind,
    dword: u16,
    mask: u32,
    read: bool,
    write: bool,
}

/// Static memory-effect facts of one instruction.
pub(super) struct MemHw {
    class: MemClass,
    counters: CounterSet,
    units: [u8; N],
    load: bool,
    family: &'static str,
    /// In-order class for write-after-write tolerance between back-to-back loads.
    order: Option<&'static str>,
    def_regs: Vec<RegRef>,
    def_mask: u32,
    /// VGPR dwords a store reads after issue (until its counter retires it).
    locks: Vec<(Kind, u16)>,
}
/// Static per-instruction facts, computed once by `Emulator::hw_model`.
pub(super) struct StepHw {
    touch: Vec<Touch>,
    mem: Option<MemHw>,
    wait: Option<[Option<u8>; N]>,
    clears_store_locks: bool,
}

pub(super) fn build_step_hw(arch: Arch, wave: Wave, step: &Step, reads_at_issue: bool) -> StepHw {
    let inst = step.inst;
    let eff = &inst.effects;
    // A MUBUF access with neither OFFEN nor IDXEN (`off`) ignores VADDR, but the decoder still yields the encoded
    // VADDR field as a register (`v[0:1]`): it is not read. (IDXEN is unrepresentable; `memory.rs::buffer_shape`
    // refuses it.)
    let skip: Option<usize> = (matches!(
        inst.form,
        crate::inst::Form::Vmem(crate::inst::VmemForm::Buffer)
    ) && !inst
        .operands
        .iter()
        .any(|o| matches!(o, crate::operand::Operand::Vmem(crate::operand::VmemToken::Offen)))
        && matches!(inst.operands.get(1), Some(crate::operand::Operand::Reg(r)) if r.kind == Kind::V))
    .then_some(1);
    let operand_reg = |idx: usize| {
        if Some(idx) == skip {
            None
        } else {
            operand_register(arch, wave, inst, idx)
        }
    };
    let skipped: Option<RegRef> = skip.and_then(|i| operand_register(arch, wave, inst, i));
    let mut masks: Vec<(Kind, u16, u32)> = Vec::new();
    let mut explicit: Vec<(Kind, u16)> = Vec::new();
    for idx in 0..inst.operands.len() {
        let Some(reg) = operand_reg(idx) else {
            continue;
        };
        let mask = operand_mask(inst, arch, idx);
        for w in 0..u16::from(reg.len) {
            let key = (reg.kind, reg.base + w);
            explicit.push(key);
            match masks.iter_mut().find(|m| (m.0, m.1) == key) {
                Some(m) => m.2 |= mask,
                None => masks.push((key.0, key.1, mask)),
            }
        }
    }
    let mask_of = |key: (Kind, u16)| {
        masks
            .iter()
            .find(|m| (m.0, m.1) == key)
            .map_or(u32::MAX, |m| m.2)
    };
    let mut touch: Vec<Touch> = Vec::new();
    // The tables reserve 64-bit lane-mask slots even in wave32 (`operand_register`): an effect register that starts at
    // an explicit operand takes that operand's real width (`wave` decides it: one SGPR in wave32, a pair in wave64), or `v_add_co_u32 v, s1, ...` would "write" s2 as well.
    let exact: Vec<RegRef> = (0..inst.operands.len()).filter_map(operand_reg).collect();
    let fit = |r: &RegRef| -> RegRef {
        match exact
            .iter()
            .find(|e| e.kind == r.kind && e.base == r.base && e.len < r.len)
        {
            Some(e) => *e,
            None => *r,
        }
    };
    let covered = |r: &RegRef| skipped.as_ref() != Some(r);
    let defs: Vec<RegRef> = eff.defs.iter().filter(|r| covered(r)).map(&fit).collect();
    let uses: Vec<RegRef> = eff.uses.iter().filter(|r| covered(r)).map(&fit).collect();
    let named: Vec<(Kind, u16)> = defs
        .iter()
        .chain(uses.iter())
        .flat_map(|r| (0..u16::from(r.len)).map(move |w| (r.kind, r.base + w)))
        .collect();
    let mut add = |key: (Kind, u16), read: bool, write: bool| match touch
        .iter_mut()
        .find(|t| (t.kind, t.dword) == key)
    {
        Some(t) => {
            t.read |= read;
            t.write |= write;
        }
        None => touch.push(Touch {
            kind: key.0,
            dword: key.1,
            mask: mask_of(key),
            read,
            write,
        }),
    };
    for reg in &defs {
        for w in 0..u16::from(reg.len) {
            add((reg.kind, reg.base + w), false, true);
        }
    }
    for reg in &uses {
        for w in 0..u16::from(reg.len) {
            add((reg.kind, reg.base + w), true, false);
        }
    }
    for key in explicit {
        // An explicit operand neither table list names: touched, conservatively as a read.
        if !named.contains(&key) {
            add(key, true, false);
        }
    }
    let mem = eff.mem.as_ref().map(|m| {
        let load = is_load(m.class);
        let family = family_of(inst);
        let order = match m.class {
            MemClass::DsLoad | MemClass::DsAtomic { .. } => Some("ds"),
            MemClass::VmemLoad | MemClass::VmemAtomic { returns: true } => Some(family),
            _ => None,
        };
        MemHw {
            class: m.class,
            counters: m.counters,
            units: unit_weights(inst, arch),
            load,
            family,
            order,
            def_regs: if load {
                eff.defs.iter().copied().collect()
            } else {
                Vec::new()
            },
            def_mask: operand_mask(inst, arch, 0),
            // A store the effect table says reads its sources after issue (`SrcRead::Deferred`: LDS stores) locks them
            // until its counter retires it: the PM builder's own rule on gfx12. No lock (`reads_at_issue`) for gfx11
            // stores and for ANY object lifted from the stock compiler (`Frontend::Hipcc`), gfx1201 included: hipcc
            // redefines a DS store's data VGPR immediately and LLVM inserts no wait there. Card-proven evidence: shipped
            // gfx1201 hipcc objects (gated_delta_net_q8_fast `ds_store_b128 v39, v[17:20]` then `v_mov_b32 v17, 0`,
            // gdn_chunk_scan, gdn_chunk_kkt_solve, attention_fp8_e4m3_fa2_gqa_qresident_gfx1201,
            // gemm_mq4g256v2_residual_mmq_iu4_gfx12_symfold, sample_top_p_parallel: 6 of 38 objects of the pristine
            // kernel cache) contain exactly that pattern and run correctly.
            locks: if load || m.src_read != SrcRead::Deferred || reads_at_issue {
                Vec::new()
            } else {
                eff.uses
                    .iter()
                    .filter(|r| r.kind == Kind::V)
                    .flat_map(|r| (0..u16::from(r.len)).map(move |w| (r.kind, r.base + w)))
                    .collect()
            },
        }
    });
    let mut wait = None;
    if let Some(w) = &inst.mods.wait {
        wait = Some(w.per_counter);
    }
    let clears_store_locks = matches!(inst.op.name(arch), Some("s_wait_alu" | "s_waitcnt_depctr"))
        && wait_alu_bits(inst).is_some_and(|bits| bits >> 2 & 7 == 0);
    StepHw {
        touch,
        mem,
        wait,
        clears_store_locks,
    }
}

/// A destination dword a pending load will write: lanes it was issued under, and the values it computed.
struct RegWrite {
    kind: Kind,
    reg: u16,
    mask: u32,
    lanes: u64,
    vals: [u32; 64],
}
struct Pending {
    pc: usize,
    class: MemClass,
    counters: CounterSet,
    satisfied: CounterSet,
    units: [u8; N],
    family: &'static str,
    order: Option<&'static str>,
    defs: Vec<(Kind, u16, u32)>,
    locks: Vec<(Kind, u16)>,
    writes: Vec<RegWrite>,
    acc: Option<usize>,
}
struct Acc {
    pc: usize,
    write: bool,
    /// `signals` of the issuing wave when its counter retired the access; `UNRETIRED` while in flight.
    retire: u32,
}
#[derive(Default)]
struct WaveHw {
    pending: Vec<Pending>,
    accs: Vec<Acc>,
    /// Barriers this wave has completed (resumed from).
    passes: u32,
}

fn counter_name(c: Counter) -> &'static str {
    match c {
        Counter::Load => "loadcnt",
        Counter::Store => "storecnt",
        Counter::Ds => "dscnt",
        Counter::Km => "kmcnt",
        Counter::Sample => "samplecnt",
        Counter::Bvh => "bvhcnt",
        Counter::Exp => "expcnt",
        Counter::Vm => "vmcnt",
        Counter::Vs => "vscnt",
        Counter::Lgkm => "lgkmcnt",
    }
}
const COUNTERS: [Counter; N] = [
    Counter::Load,
    Counter::Store,
    Counter::Ds,
    Counter::Km,
    Counter::Sample,
    Counter::Bvh,
    Counter::Exp,
    Counter::Vm,
    Counter::Vs,
    Counter::Lgkm,
];
fn reg_name(kind: Kind, dword: u16) -> String {
    match kind {
        Kind::V => format!("v{dword}"),
        Kind::S => format!("s{dword}"),
        Kind::Ttmp => format!("ttmp{dword}"),
    }
}
fn pending_counter(p: &Pending) -> &'static str {
    COUNTERS
        .iter()
        .find(|c| p.counters.contains(**c) && !p.satisfied.contains(**c))
        .map_or("a wait", |c| counter_name(*c))
}

fn read_dword(st: &State, kind: Kind, n: u16) -> [u32; 64] {
    match kind {
        Kind::V => {
            let mut out = [0; 64];
            for (lane, o) in out.iter_mut().enumerate().take(st.lanes) {
                *o = st.v[usize::from(n)][lane];
            }
            out
        }
        Kind::S => [st.s[usize::from(n)]; 64],
        Kind::Ttmp => [st.t[usize::from(n)]; 64],
    }
}
fn write_dword(st: &mut State, kind: Kind, n: u16, mask: u32, lanes: u64, vals: &[u32; 64]) {
    let n = usize::from(n);
    match kind {
        Kind::V => {
            for lane in 0..st.lanes {
                if lanes >> lane & 1 != 0 {
                    let v = &mut st.v[n][lane];
                    *v = (*v & !mask) | (vals[lane] & mask);
                }
            }
        }
        Kind::S => st.s[n] = (st.s[n] & !mask) | (vals[0] & mask),
        Kind::Ttmp => st.t[n] = (st.t[n] & !mask) | (vals[0] & mask),
    }
}

impl WaveHw {
    fn complete(&mut self, st: &mut State, p: Pending) {
        for w in &p.writes {
            write_dword(st, w.kind, w.reg, w.mask, w.lanes, &w.vals);
        }
        if let Some(a) = p.acc {
            self.accs[a].retire = st.signals.min(u64::from(UNRETIRED - 1)) as u32;
        }
    }
    /// `s_wait_X n`: the dynamic form of `passes::waits::retire` (same rules, one concrete path).
    fn retire(&mut self, st: &mut State, counter: Counter, count: u8) {
        if count != 0 {
            if matches!(counter, Counter::Km | Counter::Store | Counter::Vs) {
                return;
            }
            let live =
                |p: &&Pending| p.counters.contains(counter) && !p.satisfied.contains(counter);
            if counter == Counter::Lgkm
                && self.pending.iter().filter(live).any(|p| {
                    !matches!(
                        p.class,
                        MemClass::DsLoad
                            | MemClass::DsStore
                            | MemClass::DsAtomic { .. }
                            | MemClass::FlatLoad
                            | MemClass::FlatStore
                            | MemClass::FlatAtomic { .. }
                    )
                })
            {
                return;
            }
            if counter == Counter::Load {
                let mut fam: Option<&str> = None;
                for p in self.pending.iter().filter(live) {
                    match fam {
                        None => fam = Some(p.family),
                        Some(f) if f != p.family => return,
                        _ => {}
                    }
                }
            }
        }
        let c = counter as usize;
        let mut younger = 0u32;
        let mut done = Vec::new();
        for (pos, p) in self.pending.iter_mut().enumerate().rev() {
            if !p.counters.contains(counter) || p.satisfied.contains(counter) {
                continue;
            }
            let eligible = count == 0 || younger >= u32::from(count);
            younger += u32::from(p.units[c]);
            if eligible {
                p.satisfied.insert(counter);
                if p.satisfied == p.counters {
                    done.push(pos);
                }
            }
        }
        // `done` is youngest-first: remove from the back (indices stay valid), then apply oldest-first so
        // back-to-back loads of one register land in issue order.
        let mut finished: Vec<Pending> = done
            .into_iter()
            .map(|pos| self.pending.remove(pos))
            .collect();
        finished.reverse();
        for p in finished {
            self.complete(st, p);
        }
    }
}

/// Per-workgroup LDS happens-before tracker.
struct LdsRace {
    nw: usize,
    lw_wave: Vec<u8>,
    lw_id: Vec<u32>,
    lr: Vec<u32>,
}
struct Race {
    byte: usize,
    kind: &'static str,
    other_wave: usize,
    other_acc: u32,
    why: &'static str,
}
impl LdsRace {
    fn new(len: usize, nw: usize) -> Self {
        Self {
            nw,
            lw_wave: vec![NO_WAVE; len],
            lw_id: vec![NO_ACC; len],
            lr: vec![NO_ACC; len * nw],
        }
    }
    fn access(
        &mut self,
        wave: usize,
        id: u32,
        t: &LdsTouch,
        ord: &dyn Fn(usize, u32) -> std::result::Result<(), &'static str>,
    ) -> std::result::Result<(), Race> {
        for byte in t.lo..t.lo + t.len {
            let lw = self.lw_wave[byte];
            if lw != NO_WAVE && usize::from(lw) != wave {
                let lid = self.lw_id[byte];
                if !(t.atomic && lid & ATOMIC != 0) {
                    if let Err(why) = ord(usize::from(lw), lid & !ATOMIC) {
                        return Err(Race {
                            byte,
                            kind: if t.write {
                                "write-after-write"
                            } else {
                                "read-after-write"
                            },
                            other_wave: usize::from(lw),
                            other_acc: lid & !ATOMIC,
                            why,
                        });
                    }
                }
            }
            if t.write {
                for a in 0..self.nw {
                    let r = self.lr[byte * self.nw + a];
                    // Atomics on one byte are mutually ordered by the LDS; only a plain access races with them.
                    if a != wave && r != NO_ACC && !(t.atomic && r & ATOMIC != 0) {
                        if let Err(why) = ord(a, r & !ATOMIC) {
                            return Err(Race {
                                byte,
                                kind: "write-after-read",
                                other_wave: a,
                                other_acc: r & !ATOMIC,
                                why,
                            });
                        }
                    }
                }
                self.lw_wave[byte] = wave as u8;
                self.lw_id[byte] = id | if t.atomic { ATOMIC } else { 0 };
            }
            if !t.write || t.atomic {
                self.lr[byte * self.nw + wave] = id | if t.atomic { ATOMIC } else { 0 };
            }
        }
        Ok(())
    }
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn fnv(words: &[u64]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, w| {
        (h ^ w).wrapping_mul(0x100_0000_01b3)
    })
}

struct Ctx {
    states: Vec<State>,
    hws: Vec<WaveHw>,
    lds: Vec<u8>,
    defined: WriteSet,
    race: LdsRace,
    log: Vec<LdsTouch>,
    executed: u64,
}

impl<'a> Emulator<'a> {
    /// The full gate model: in-flight loads/stores, LDS race detection, schedule exploration (see the module docs).
    /// Implies `strict_lds`. Every kernel run through the pre-card gates uses it; there is no switch to relax it.
    pub fn hw_model(self) -> Self {
        self.hw_model_with(false)
    }
    /// The gate model plus the PM builder's own store-source rule on every object: an LDS store's data VGPRs may not be
    /// redefined before `lgkmcnt`/`dscnt` retires it. Stock hipcc output violates this (it reads store data at issue,
    /// see `build_step_hw`), so no gate uses it; it names the hazard class in regression tests and in audits.
    pub fn hw_model_strict_store_sources(self) -> Self {
        self.hw_model_with(true)
    }
    fn hw_model_with(mut self, strict_store_sources: bool) -> Self {
        let arch = self.decode_arch;
        let hipcc = matches!(
            self.kernel.origin,
            crate::inst::KernelOrigin::Frontend {
                kind: crate::inst::Frontend::Hipcc,
                ..
            }
        );
        let reads_at_issue =
            !strict_store_sources && (hipcc || matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
        self.hw = Some(
            self.steps
                .iter()
                .map(|s| build_step_hw(arch, self.kernel.wave, s, reads_at_issue))
                .collect(),
        );
        self.strict_lds = true;
        self
    }
    pub(super) fn hw_enabled(&self) -> bool {
        self.hw.is_some()
    }
    /// Run one workgroup under every schedule of the gate model; leaves `mem` in schedule 0's final state.
    pub(super) fn run_hw(&self, launch: &Launch, mem: &mut Memory) -> Result<RunReport> {
        let nw = (launch.block[0]
            .checked_mul(launch.block[1])
            .and_then(|n| n.checked_mul(launch.block[2]))
            .ok_or("block size overflow")? as usize)
            .div_ceil(self.kernel.wave.lanes());
        let h = fnv(&[
            u64::from(launch.workgroup[0]),
            u64::from(launch.workgroup[1]),
            u64::from(launch.workgroup[2]),
        ]);
        let mut scheds = vec![Schedule::RoundRobin];
        if nw > 1 {
            // The first FULL_SCHEDULE_WGS workgroups an emulator runs get every schedule; later ones round-robin
            // plus a rotating run-ahead wave and one random order, so a long grid still covers every schedule while
            // the cost stays bounded. Race and in-flight detection are schedule independent and run on all of them.
            let k = self.runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if k < FULL_SCHEDULE_WGS {
                scheds.extend((0..nw).map(Schedule::Ahead));
                scheds.push(Schedule::Random(h ^ 0x9e37_79b9_7f4a_7c15));
                scheds.push(Schedule::Random(h ^ 0xc2b2_ae3d_27d4_eb4f));
            } else {
                scheds.push(Schedule::Ahead((k - FULL_SCHEDULE_WGS) as usize % nw));
                scheds.push(Schedule::Random(h ^ 0x9e37_79b9_7f4a_7c15 ^ k));
            }
        }
        let names: Vec<String> = scheds.iter().map(|s| s.name()).collect();
        if scheds.len() == 1 {
            let n = self.run_schedule(launch, mem, scheds[0])?;
            return Ok(RunReport {
                instructions: n,
                schedules: names,
            });
        }
        mem.journal_start();
        let mut first: Option<(StoreImage, u64)> = None;
        for (k, s) in scheds.iter().enumerate() {
            let r = self.run_schedule(launch, mem, *s);
            let n = match r {
                Ok(n) => n,
                Err(e) => {
                    mem.journal_stop();
                    return Err(format!("[schedule {}] {e}", names[k]));
                }
            };
            let image = mem.journal_image();
            mem.journal_undo();
            match &first {
                None => first = Some((image, n)),
                Some((base, _)) => {
                    if let Some(d) = image_diff(base, &image, mem) {
                        mem.journal_stop();
                        return Err(format!(
                            "[schedule] workgroup {:?} stores differ between schedule `{}` and `{}`: {d}",
                            launch.workgroup, names[0], names[k]
                        ));
                    }
                }
            }
        }
        mem.journal_stop();
        let (image, instructions) = first.expect("at least one schedule ran");
        mem.apply_image(&image)?;
        Ok(RunReport {
            instructions,
            schedules: names,
        })
    }

    fn run_schedule(&self, launch: &Launch, mem: &mut Memory, sched: Schedule) -> Result<u64> {
        let kernel = self.kernel;
        let Abi::Hsa { descriptor: d, .. } = &kernel.abi else {
            return Err("HSA descriptor required".into());
        };
        let n = launch.block[0]
            .checked_mul(launch.block[1])
            .and_then(|n| n.checked_mul(launch.block[2]))
            .ok_or("block size overflow")? as usize;
        if n == 0 || n > 1024 {
            return Err("workgroup must contain 1..1024 threads".into());
        }
        let nw = n.div_ceil(kernel.wave.lanes());
        let states = (0..nw)
            .map(|w| {
                State::new(self.arch, kernel, launch, w).map(|mut s| {
                    s.nan_loose = self.nan_loose;
                    s
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let lds_len = d
            .group_segment_fixed_size
            .checked_add(launch.dynamic_lds)
            .ok_or("LDS size overflow")? as usize;
        let mut ctx = Ctx {
            states,
            hws: (0..nw).map(|_| WaveHw::default()).collect(),
            lds: vec![0u8; lds_len],
            defined: WriteSet::new(lds_len),
            race: LdsRace::new(lds_len, nw),
            log: Vec::new(),
            executed: 0,
        };
        let mut cursor = nw - 1;
        let mut rng = match sched {
            Schedule::Random(s) => s | 1,
            _ => 1,
        };
        while ctx.states.iter().any(|s| !s.ended) {
            let runnable =
                |ctx: &Ctx, i: usize| !ctx.states[i].ended && ctx.states[i].barrier.is_none();
            let round_robin = |ctx: &Ctx, cursor: usize| {
                (1..=nw)
                    .map(|k| (cursor + k) % nw)
                    .find(|&i| runnable(ctx, i))
            };
            let pick = match sched {
                Schedule::RoundRobin => round_robin(&ctx, cursor),
                Schedule::Ahead(w) => {
                    if runnable(&ctx, w) {
                        Some(w)
                    } else {
                        round_robin(&ctx, cursor)
                    }
                }
                Schedule::Random(_) => {
                    let live: Vec<usize> = (0..nw).filter(|&i| runnable(&ctx, i)).collect();
                    if live.is_empty() {
                        None
                    } else {
                        Some(live[(xorshift(&mut rng) % live.len() as u64) as usize])
                    }
                }
            };
            match pick {
                Some(w) => {
                    cursor = w;
                    self.exec_one(launch, mem, &mut ctx, w)?;
                }
                None => {
                    // A gfx12 signal is an arrival, not a wait (see the legacy loop): release every wave whose
                    // phase every live wave has reached.
                    let arrivals = ctx
                        .states
                        .iter()
                        .filter(|s| !s.ended)
                        .map(|s| s.signals)
                        .min();
                    let mut progress = false;
                    if let Some(arrivals) = arrivals {
                        for (st, hw) in ctx.states.iter_mut().zip(ctx.hws.iter_mut()) {
                            if st.barrier.is_some_and(|phase| phase < arrivals) {
                                st.barrier = None;
                                hw.passes += 1;
                                progress = true;
                            }
                        }
                    }
                    if !progress {
                        return Err("workgroup barrier deadlock".into());
                    }
                }
            }
        }
        Ok(ctx.executed)
    }

    fn exec_one(
        &self,
        launch: &Launch,
        mem: &mut Memory,
        ctx: &mut Ctx,
        wave: usize,
    ) -> Result<()> {
        let pc = ctx.states[wave].pc;
        let step = self.steps.get(pc).ok_or("PC ran past kernel end")?;
        ctx.states[wave].pc += 1;
        ctx.executed += 1;
        if ctx.executed > 100_000_000 {
            return Err("instruction limit exceeded (possible divergent loop)".into());
        }
        self.exec_inner(mem, ctx, wave, pc, step).map_err(|e| {
            format!(
                "{} wg={:?} inst={pc} {}: {e}{}",
                self.kernel.symbol.0,
                launch.workgroup,
                step.name,
                failure_bits(self.arch, wave, &ctx.states[wave], step.inst)
            )
        })
    }

    fn step_name(&self, pc: usize) -> &'static str {
        self.steps.get(pc).map_or("?", |s| s.name)
    }

    fn exec_inner(
        &self,
        mem: &mut Memory,
        ctx: &mut Ctx,
        wave: usize,
        pc: usize,
        step: &Step,
    ) -> Result<()> {
        let shw = &self.hw.as_ref().expect("hw model")[pc];
        let inst = step.inst;
        let name = step.name;
        // 1. In-flight hazards: this instruction against the wave's pending loads and stores.
        for p in &ctx.hws[wave].pending {
            for t in &shw.touch {
                if let Some(d) = p
                    .defs
                    .iter()
                    .find(|d| d.0 == t.kind && d.1 == t.dword && d.2 & t.mask != 0)
                {
                    let tolerated = t.write
                        && !t.read
                        && shw
                            .mem
                            .as_ref()
                            .is_some_and(|m| m.load && m.order.is_some() && m.order == p.order);
                    if !tolerated {
                        return Err(format!(
                            "[in-flight] inst {} {} {} while the load at inst {} ({}) into it has not retired (no {} wait since it was issued) [touches: {}]",
                            pc,
                            if t.write && t.read {
                                "reads and writes"
                            } else if t.write {
                                "writes"
                            } else {
                                "reads"
                            },
                            reg_name(d.0, d.1),
                            p.pc,
                            self.step_name(p.pc),
                            pending_counter(p),
                            shw.touch
                                .iter()
                                .map(|x| format!("{}{}", reg_name(x.kind, x.dword), if x.write { "w" } else { "r" }))
                                .collect::<Vec<_>>()
                                .join(","),
                        ));
                    }
                }
                if t.write && t.kind == Kind::V && p.locks.contains(&(t.kind, t.dword)) {
                    return Err(format!(
                        "[in-flight] writes {} (in-flight store source) while the store at inst {} ({}) may not have read it (no {} wait since it was issued)",
                        reg_name(t.kind, t.dword),
                        p.pc,
                        self.step_name(p.pc),
                        pending_counter(p),
                    ));
                }
            }
        }
        // 2. Execute.
        match step.class {
            Class::Wait => {
                if let Some(wait) = &shw.wait {
                    let Ctx { states, hws, .. } = ctx;
                    for (i, w) in wait.iter().enumerate() {
                        if let Some(count) = *w {
                            hws[wave].retire(&mut states[wave], COUNTERS[i], count);
                        }
                    }
                } else if shw.clears_store_locks {
                    for p in &mut ctx.hws[wave].pending {
                        if p.counters.contains(Counter::Store) || p.counters.contains(Counter::Vs) {
                            p.locks.clear();
                        }
                    }
                }
            }
            Class::Control => control(name, inst, &mut ctx.states[wave], self.kernel)?,
            Class::Alu => alu::execute(self.arch, name, inst, &mut ctx.states[wave])?,
            Class::Memory => self.memory_step(mem, ctx, wave, pc, step, shw)?,
        }
        Ok(())
    }

    fn memory_step(
        &self,
        mem: &mut Memory,
        ctx: &mut Ctx,
        wave: usize,
        pc: usize,
        step: &Step,
        shw: &StepHw,
    ) -> Result<()> {
        let Ctx {
            states,
            hws,
            lds,
            defined,
            race,
            log,
            ..
        } = ctx;
        let inst = step.inst;
        log.clear();
        let mut snap: Vec<(Kind, u16, [u32; 64])> = Vec::new();
        let exec = states[wave].exec;
        if let Some(m) = shw.mem.as_ref().filter(|m| m.load) {
            for r in &m.def_regs {
                for w in 0..u16::from(r.len) {
                    snap.push((
                        r.kind,
                        r.base + w,
                        read_dword(&states[wave], r.kind, r.base + w),
                    ));
                }
            }
        }
        memory::execute_tracked(
            step.name,
            inst,
            &mut states[wave],
            mem,
            lds,
            Some(defined),
            Some(log),
        )?;
        // The load's destinations keep their old values until a wait retires it.
        let mut writes = Vec::with_capacity(snap.len());
        let def_mask = shw.mem.as_ref().map_or(u32::MAX, |m| m.def_mask);
        for (kind, reg, old) in snap {
            let new = read_dword(&states[wave], kind, reg);
            let lanes = if kind == Kind::V { exec } else { 1 };
            // Restore only what the load changed (exec lanes of a VGPR; the one scalar).
            write_dword(&mut states[wave], kind, reg, u32::MAX, u64::MAX, &old);
            writes.push(RegWrite {
                kind,
                reg,
                mask: def_mask,
                lanes,
                vals: new,
            });
        }
        // LDS: stamp the access and check it against every other wave's.
        let mut acc = None;
        if !log.is_empty() {
            let id = hws[wave].accs.len() as u32;
            hws[wave].accs.push(Acc {
                pc,
                write: log.iter().any(|t| t.write),
                retire: UNRETIRED,
            });
            acc = Some(id as usize);
            let passes = hws[wave].passes;
            let ord = |a: usize, aid: u32| -> std::result::Result<(), &'static str> {
                let rec = &hws[a].accs[aid as usize];
                if rec.retire == UNRETIRED {
                    return Err("still in flight when the other access issued (no lgkmcnt/dscnt wait retired it before a barrier)");
                }
                if rec.retire < passes && u64::from(rec.retire) < states[a].signals {
                    Ok(())
                } else {
                    Err("no barrier orders them (same barrier epoch)")
                }
            };
            for t in log.iter() {
                if let Err(r) = race.access(wave, id, t, &ord) {
                    let other = &hws[r.other_wave].accs[r.other_acc as usize];
                    return Err(format!(
                        "[race] LDS {} on byte {:#x}: wave {wave} ({} at inst {pc}) vs wave {} ({} at inst {} {}): {}",
                        r.kind,
                        r.byte,
                        step.name,
                        r.other_wave,
                        if other.write { "write" } else { "read" },
                        other.pc,
                        self.step_name(other.pc),
                        r.why,
                    ));
                }
            }
        }
        if let Some(m) = shw.mem.as_ref() {
            if m.counters.0 != 0 {
                let defs = if m.load {
                    m.def_regs
                        .iter()
                        .flat_map(|r| {
                            (0..u16::from(r.len)).map(move |w| (r.kind, r.base + w, m.def_mask))
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                hws[wave].pending.push(Pending {
                    pc,
                    class: m.class,
                    counters: m.counters,
                    satisfied: CounterSet::default(),
                    units: m.units,
                    family: m.family,
                    order: m.order,
                    defs,
                    locks: m.locks.clone(),
                    writes,
                    acc,
                });
            } else if !writes.is_empty() {
                // No counter covers this load: it completes at once.
                for w in &writes {
                    write_dword(&mut states[wave], w.kind, w.reg, w.mask, w.lanes, &w.vals);
                }
            }
        }
        Ok(())
    }
}

/// First difference between two schedules' stores, or `None` when bit-identical.
fn image_diff(a: &StoreImage, b: &StoreImage, mem: &Memory) -> Option<String> {
    let at = |r: u32, o: u64| {
        let reg = &mem.regions[r as usize];
        format!("{}+{o:#x} (address {:#x})", reg.name, reg.base + o)
    };
    let (mut i, mut j) = (0, 0);
    let mut differ = 0usize;
    let mut first = None;
    while i < a.0.len() || j < b.0.len() {
        let x = a.0.get(i);
        let y = b.0.get(j);
        let msg = match (x, y) {
            (Some(&(r, o, v)), Some(&(r2, o2, v2))) if (r, o) == (r2, o2) => {
                i += 1;
                j += 1;
                if v == v2 {
                    continue;
                }
                format!("{} is {v:#04x} then {v2:#04x}", at(r, o))
            }
            (Some(&(r, o, _)), Some(&(r2, o2, _))) if (r, o) < (r2, o2) => {
                i += 1;
                format!("{} stored only by the first", at(r, o))
            }
            (Some(_), Some(&(r2, o2, _))) => {
                j += 1;
                format!("{} stored only by the second", at(r2, o2))
            }
            (Some(&(r, o, _)), None) => {
                i += 1;
                format!("{} stored only by the first", at(r, o))
            }
            (None, Some(&(r2, o2, _))) => {
                j += 1;
                format!("{} stored only by the second", at(r2, o2))
            }
            (None, None) => break,
        };
        differ += 1;
        first.get_or_insert(msg);
    }
    first.map(|m| format!("{differ} byte(s) differ, first: {m}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(write: bool, atomic: bool) -> LdsTouch {
        LdsTouch {
            lo: 0,
            len: 4,
            write,
            atomic,
        }
    }
    /// Nothing is ordered: every cross-wave pair is a race unless exempt.
    fn unordered(_: usize, _: u32) -> std::result::Result<(), &'static str> {
        Err("unordered")
    }

    #[test]
    fn atomics_on_one_dword_from_two_waves_are_not_a_race() {
        // `__syncthreads_or`: every wave's lane 0 does `ds_or_b32` on one dword.
        let mut r = LdsRace::new(16, 3);
        assert!(r.access(0, 0, &touch(true, true), &unordered).is_ok());
        assert!(r.access(1, 0, &touch(true, true), &unordered).is_ok());
        assert!(r.access(2, 0, &touch(true, true), &unordered).is_ok());
    }

    #[test]
    fn an_atomic_against_a_plain_access_from_another_wave_is_a_race() {
        for (first, second, kind) in [
            (touch(true, true), touch(false, false), "read-after-write"),
            (touch(true, true), touch(true, false), "write-after-write"),
            (touch(true, false), touch(true, true), "write-after-write"),
            (touch(false, false), touch(true, true), "write-after-read"),
        ] {
            let mut r = LdsRace::new(16, 2);
            assert!(r.access(0, 0, &first, &unordered).is_ok());
            let e = r.access(1, 0, &second, &unordered).err().expect("race");
            assert_eq!((e.kind, e.other_wave), (kind, 0));
        }
    }

    #[test]
    fn ordered_accesses_and_one_wave_never_race() {
        let ordered = |_: usize, _: u32| Ok(());
        let mut r = LdsRace::new(16, 2);
        assert!(r.access(0, 0, &touch(true, false), &ordered).is_ok());
        assert!(r.access(1, 0, &touch(false, false), &ordered).is_ok());
        assert!(r.access(1, 1, &touch(true, false), &ordered).is_ok());
        let mut one = LdsRace::new(16, 2);
        assert!(one.access(0, 0, &touch(true, false), &unordered).is_ok());
        assert!(one.access(0, 1, &touch(false, false), &unordered).is_ok());
        assert!(one.access(0, 2, &touch(true, false), &unordered).is_ok());
    }
}
