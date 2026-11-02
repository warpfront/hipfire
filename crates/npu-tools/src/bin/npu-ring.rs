// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-ring: silicon harness for the persistent V9 ring (`peacemaker_npu::kernels::ring`).
//!
//! usage: npu-ring M N K --slots S [--nslots R] [--rounds N | --fit-ms B] [--mode prepublish|paced|empty-paced|empty-prepublish]
//!                 [--neg skip-done|stale-seq] [--timeout-ms T] [--allow-hang]
//!
//! One hardware context, one PDI, ONE retained instruction BO and ONE retained command BO. Every round is one
//! START_CU whose TXN runs `S` slot-runs back to back; between rounds only the declared `patch_seq` words of the
//! instruction BO change (`new seq0 = 1 + round*S`) and exactly those cache lines are flushed. Arguments are the
//! shmem arenas A, B, C and the shmem ring (`railgun::npu::{Device,HwCtx,Bo}` submit/wait only).
//!
//! * `M N K` — V9 shape (`design_v9(.., V8_DEFAULT_EPILOGUE, Control::Fast)`). The empty modes run
//!   `empty_persistent` (no GEMM, 512x512x64 V9 PDI): M N K are ignored and the shape reports 512x512x64.
//! * `--slots S` — slot-runs per round (required, >= 1). Run `j` of every round uses ring slot `j % nslots`; operands
//!   are fixed per physical slot (A/B/C offsets = slot * packed bytes) and the slot plan is fixed per design (only seq
//!   words are patched), and seq of (round r, run j) is `1 + r*S + j`.
//! * `--nslots R` — ring slots, a power of two (default: next power of two of S). prepublish needs `R >= S`.
//! * `--rounds N` — rounds to run (default 1; 2 for `--neg stale-seq`). `--fit-ms B` instead runs rounds until `B` ms
//!   of loop wall time elapsed (at least one, at most `--rounds` or 1000). More than one round needs `S % R == 0`
//!   (the retained j%R routing cannot follow the global run_index%R otherwise): everything else is rejected with exit 2,
//!   so multi-round prepublish means `S == R`.
//! * `--mode` — `prepublish` (default): every slot line + seq is published BEFORE the submit (throughput).
//!   `paced`: submit first, then run j is published only after the done line of run j-1 was observed (latency).
//!   `empty-prepublish` / `empty-paced`: the same without a GEMM (pure mechanism).
//!   The CPU producer stages the packed operands (once, immutable arenas) and flushes them; per publish it writes the
//!   slot fields, flushes, writes `seq` LAST, flushes, sfence. A paced publish refuses (host guard) a slot whose
//!   previous seq has no matching done line; paced also reports `premature_c`, C slots found touched before publish.
//! * Operand sets: internal, `min(slots in use, 4)` when M*N*K <= 2^24, else 1; physical slot s uses set `s % sets`.
//!   The CPU reference is the exact separable closed form (A[r,k]=ra(r), B[k,c]=cb(c), amplitudes 7..19, signs vary per
//!   row/col/set; C = epilogue(K*ra*cb)), O(M*N). One eager `design_v9` submit per set runs FIRST on the SAME hardware
//!   context and PDI (the persistent PDI must be byte-identical) and is compared with the CPU reference. Every run of every
//!   round is then compared byte for byte with that eager C (paced slot reuse is checked right after its done is
//!   observed, before the slot can be overwritten), so ring results are exact by transitivity (CPU == eager == ring);
//!   round 0 and the final round are additionally unpacked and compared with the CPU directly. Empty modes run no eager
//!   and report `exact=n/a`.
//! * `--timeout-ms T` — every wait / done spin deadline (default 5000). The harness always eventually publishes every
//!   seq its command polls for: after a missed done it publishes every still-unpublished run whose physical slot is
//!   free (previous seq has its done line) and keeps retrying until the deadline; it NEVER overwrites a slot whose
//!   previous seq is not done. With `--allow-hang` no rescue publish happens and a hang is reported as allowed (exit 0).
//! * `--neg skip-done` — `Neg::SkipDone(S/2)`: that run writes no done line. The producer does not wait for the
//!   skipped done (it publishes the remaining runs) and the negative is reported detected when exactly that done
//!   line is missing. Needs `S/2 + nslots >= S`.
//! * `--neg stale-seq` — last round runs WITHOUT `patch_seq` and without publishing new seqs: the retained insts still
//!   poll the old seqs that are still in the ring, so it completes at once with old done lines; reported with zero
//!   patched inventory and no new done. With `--allow-hang` the new seqs are published instead (the unpatched insts
//!   poll for old seqs forever -> an expected, allowed hang). Needs `nslots >= S`, `--rounds >= 2`, no `--fit-ms`.
//! * `--lean` — build the persistent design with `persistent_v9_lean` instead of `persistent_v9` (same operands, slots,
//!   neg, patch inventory: exactly two seq words per run). Run 0 of every submission is full, runs 1.. are lean with
//!   forced col0 S2MM4/MM2S0 restoration (no `--lean-first`). Setup prints `lean: insts_bytes_lean=.. insts_bytes_full=..`
//!   (the full length is `persistent_v9` of the same operands). GEMM modes only.
//! * `--verify-every` — additionally CPU-direct exact-check (unpack + reference) every slot after every round. The
//!   byte-by-byte eager comparison of every slot of every round is unconditional and unchanged.
//! * Ring overrun is never forced on silicon: the harness prints `NEG overrun: guard refused publish` from the host
//!   guard (`producer_may_publish`) on a scratch ring image.
//!
//! Output: setup lines, one line per round, `NEG ...` lines and finally
//! `ring: mode=.. shape=MxNxK slots=S rounds=N per_round_us=.. per_slot_us=.. busy_tops=.. wall_tops=.. rt_us p50=.. p99=..
//! min=.. firstdone_us p50=.. min=.. exact=ok/total bytes_equal_eager=ok/total premature_c=N`.
//! per_round_us = mean submit->wait-return (Instant), busy_tops = S*2MNK / that; wall_tops uses the mean of publish-start
//! ->wait-return. rt_us: paced = per-run (seq published -> done seen); prepublish = per-run done-to-done interval (the
//! first is submit->first done). firstdone_us = submit -> first observed done. Exit 0 = everything verified (negatives
//! detected), 1 = failure, 2 = usage.
use peacemaker_npu::kernels::gemm_array::{
    apply_epilogue, design_v9, ArrayDesign, V8_DEFAULT_EPILOGUE,
};
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::ring::{
    empty_persistent, patch_seq, persistent_v9, persistent_v9_lean, producer_may_publish, Neg,
    PatchKind, RingLayout, SlotPlan,
};
use railgun::npu::{clflush, Bo, Device, HwCtx, ERT_STATE_COMPLETED};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{compiler_fence, Ordering};
use std::time::{Duration, Instant};

const DONE_MAGIC: u32 = 0x454e_4f44;
const POISON: u8 = 0xA5;
const USAGE: &str = "usage: npu-ring M N K --slots S [--nslots R] [--rounds N | --fit-ms B] \
[--mode prepublish|paced|empty-paced|empty-prepublish] [--neg skip-done|stale-seq] [--timeout-ms T] [--allow-hang] [--lean] [--verify-every]";
/// `max_opc` for the hardware context (as npu-gemm uses for every array design).
const MAX_OPC: u32 = 2048;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Prepublish,
    Paced,
    EmptyPaced,
    EmptyPrepublish,
}
impl Mode {
    fn empty(self) -> bool {
        matches!(self, Mode::EmptyPaced | Mode::EmptyPrepublish)
    }
    fn paced(self) -> bool {
        matches!(self, Mode::Paced | Mode::EmptyPaced)
    }
    fn name(self) -> &'static str {
        match self {
            Mode::Prepublish => "prepublish",
            Mode::Paced => "paced",
            Mode::EmptyPaced => "empty-paced",
            Mode::EmptyPrepublish => "empty-prepublish",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NegKind {
    SkipDone,
    StaleSeq,
}

struct Cfg {
    m: usize,
    n: usize,
    k: usize,
    slots: usize,
    nslots: usize,
    rounds: Option<usize>,
    fit_ms: Option<u64>,
    mode: Mode,
    neg: Option<NegKind>,
    timeout_ms: u64,
    allow_hang: bool,
    /// `persistent_v9_lean`: run 0 of every submission full, remaining runs lean.
    lean: bool,
    /// CPU-direct exact check of every slot after EVERY round (the eager byte comparison is always every slot, every round).
    verify_every: bool,
}

fn parse(args: &[String]) -> Result<Cfg, String> {
    let mut pos = Vec::new();
    let mut slots = None;
    let mut nslots = None;
    let mut rounds = None;
    let mut fit_ms = None;
    let mut mode = Mode::Prepublish;
    let mut neg = None;
    let mut timeout_ms = 5000u64;
    let mut allow_hang = false;
    let mut lean = false;
    let mut verify_every = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--slots" => slots = Some(parse_num::<usize>("--slots", &val("--slots")?)?),
            "--nslots" => nslots = Some(parse_num::<usize>("--nslots", &val("--nslots")?)?),
            "--rounds" => rounds = Some(parse_num::<usize>("--rounds", &val("--rounds")?)?),
            "--fit-ms" => fit_ms = Some(parse_num::<u64>("--fit-ms", &val("--fit-ms")?)?),
            "--timeout-ms" => timeout_ms = parse_num::<u64>("--timeout-ms", &val("--timeout-ms")?)?,
            "--allow-hang" => allow_hang = true,
            "--lean" => lean = true,
            "--verify-every" => verify_every = true,
            "--mode" => {
                mode = match val("--mode")?.as_str() {
                    "prepublish" => Mode::Prepublish,
                    "paced" => Mode::Paced,
                    "empty-paced" => Mode::EmptyPaced,
                    "empty-prepublish" => Mode::EmptyPrepublish,
                    other => return Err(format!("--mode: unknown mode {other:?}")),
                }
            }
            "--neg" => {
                neg = Some(match val("--neg")?.as_str() {
                    "skip-done" => NegKind::SkipDone,
                    "stale-seq" => NegKind::StaleSeq,
                    other => return Err(format!("--neg: unknown negative {other:?}")),
                })
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => pos.push(a.clone()),
        }
    }
    if pos.len() != 3 {
        return Err("need exactly M N K".into());
    }
    let (m, n, k) = (
        parse_num::<usize>("M", &pos[0])?,
        parse_num::<usize>("N", &pos[1])?,
        parse_num::<usize>("K", &pos[2])?,
    );
    let slots = slots.ok_or("--slots is required")?;
    if slots == 0 {
        return Err("--slots must be >= 1".into());
    }
    let nslots = nslots.unwrap_or_else(|| slots.next_power_of_two());
    if !nslots.is_power_of_two() {
        return Err(format!("--nslots {nslots} must be a power of two"));
    }
    if matches!(mode, Mode::Prepublish | Mode::EmptyPrepublish) && nslots < slots {
        return Err(format!(
            "{} needs --nslots >= --slots ({nslots} < {slots})",
            mode.name()
        ));
    }
    if rounds == Some(0) || timeout_ms == 0 {
        return Err("--rounds/--timeout-ms must be >= 1".into());
    }
    if neg.is_some() && mode.empty() {
        return Err("--neg needs a GEMM mode (prepublish|paced)".into());
    }
    if neg == Some(NegKind::SkipDone) && slots / 2 + nslots < slots {
        return Err(format!(
            "--neg skip-done needs S/2 + nslots >= S (skipped run {} would have its slot reused)",
            slots / 2
        ));
    }
    if neg == Some(NegKind::StaleSeq) {
        if nslots < slots {
            return Err("--neg stale-seq needs --nslots >= --slots".into());
        }
        if fit_ms.is_some() {
            return Err("--neg stale-seq needs an explicit --rounds, not --fit-ms".into());
        }
        if rounds.is_some_and(|r| r < 2) {
            return Err("--neg stale-seq needs --rounds >= 2".into());
        }
    }
    let multi_round =
        rounds.is_some_and(|r| r > 1) || fit_ms.is_some() || neg == Some(NegKind::StaleSeq);
    if multi_round && slots % nslots != 0 {
        return Err(format!(
            "multi-round needs --slots divisible by --nslots (S={slots}, R={nslots}): the retained command bakes slot j%R and only seq words are patched \
             between rounds, so for S%R != 0 the global producer slot run_index%R differs from the retained j%R in later rounds (prepublish therefore needs S==R for rounds>1)"
        ));
    }
    if lean && mode.empty() {
        return Err("--lean needs a GEMM mode (prepublish|paced)".into());
    }
    Ok(Cfg {
        m,
        n,
        k,
        slots,
        nslots,
        rounds,
        fit_ms,
        mode,
        neg,
        timeout_ms,
        allow_hang,
        lean,
        verify_every,
    })
}

/// `p50`, `p99`, `min` of a sample set (`None` when empty).
fn stats(v: &mut [f64]) -> Option<(f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some((percentile(v, 0.5), percentile(v, 0.99), v[0]))
}

// ---- ring access (non-coherent CPU view: flush before reading device-written lines, after writing) ----

fn rd32(bo: &Bo, off: usize) -> u32 {
    assert!(off + 4 <= bo.len);
    unsafe { core::ptr::read_volatile(bo.host.add(off) as *const u32) }
}
fn wr32(bo: &Bo, off: usize, v: u32) {
    assert!(off + 4 <= bo.len);
    unsafe { core::ptr::write_volatile(bo.host.add(off) as *mut u32, v) }
}
fn wr64(bo: &Bo, off: usize, v: u64) {
    assert!(off + 8 <= bo.len);
    unsafe { core::ptr::write_volatile(bo.host.add(off) as *mut u64, v) }
}

/// (done_seq, slot, magic) of one done line.
fn read_done(ring: &Bo, layout: RingLayout, slot: usize) -> (u32, u32, u32) {
    let off = layout.done_line(slot);
    unsafe { clflush(ring.host.add(off), 64) };
    (rd32(ring, off), rd32(ring, off + 4), rd32(ring, off + 8))
}

/// Block until done line `slot` carries `seq` (equality, wrap-safe) and is well formed. Returns the observation time.
fn spin_done(
    ring: &Bo,
    layout: RingLayout,
    slot: usize,
    seq: u32,
    deadline: Instant,
) -> Result<Instant, String> {
    loop {
        let (s, sl, magic) = read_done(ring, layout, slot);
        if s == seq {
            let t = Instant::now();
            if sl != slot as u32 || magic != DONE_MAGIC {
                return Err(format!(
                    "malformed done line slot {slot}: seq {s:#x} slot field {sl} magic {magic:#x}"
                ));
            }
            return Ok(t);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timeout waiting for done[{slot}] seq {seq:#x} (last seen {s:#x})"
            ));
        }
        core::hint::spin_loop();
    }
}

/// CPU producer: slot fields, flush, seq last, flush, sfence; with the host overrun guard.
struct Producer {
    layout: RingLayout,
    plans: Vec<SlotPlan>,
    m: u32,
    flags: u32,
    /// Last seq published into each ring slot (0 = never).
    prev: Vec<u32>,
}

impl Producer {
    /// The command completed (or nothing ever ran): every slot is idle, so its last published seq is
    /// whatever its done line says.
    fn quiesce(&mut self, ring: &Bo) {
        for s in 0..self.layout.nslots {
            self.prev[s] = read_done(ring, self.layout, s).0;
        }
    }

    /// A slot may take a new seq only when its previous seq has its done line (or it was never used).
    fn may_publish(&self, ring: &Bo, slot: usize) -> bool {
        read_done(ring, self.layout, slot).0 == self.prev[slot]
    }

    /// Safe rescue: publish every unpublished run whose slot is free (its previous seq has its done line). Never
    /// overwrites a slot whose previous seq is not done, so a seq the command may still poll for is never replaced.
    /// Retries until `deadline`; returns the runs still pending.
    fn rescue(
        &mut self,
        ring: &Bo,
        published: &mut [bool],
        seq_of: &dyn Fn(usize) -> u32,
        deadline: Instant,
    ) -> Vec<usize> {
        loop {
            for j in 0..published.len() {
                if !published[j] && self.publish(ring, j, seq_of(j)).is_ok() {
                    published[j] = true;
                }
            }
            let pending: Vec<usize> = (0..published.len()).filter(|&j| !published[j]).collect();
            if pending.is_empty() || Instant::now() >= deadline {
                return pending;
            }
            core::hint::spin_loop();
        }
    }

    fn publish(&mut self, ring: &Bo, run: usize, seq: u32) -> Result<(), String> {
        let plan = self.plans[run];
        if !self.may_publish(ring, plan.slot) {
            return Err(format!("host guard refused publish of run {run} (seq {seq:#x}): slot {} previous seq {:#x} has no done line", plan.slot, self.prev[plan.slot]));
        }
        let base = self.layout.slot_line(plan.slot);
        wr32(ring, base + 4, 0); // prog 0 (the P1 GEMM program)
        wr32(ring, base + 8, self.m);
        wr32(ring, base + 12, self.flags);
        wr64(ring, base + 16, plan.a_off);
        wr64(ring, base + 24, plan.b_off);
        wr64(ring, base + 32, plan.c_off);
        unsafe { clflush(ring.host.add(base), 64) };
        compiler_fence(Ordering::SeqCst);
        wr32(ring, base, seq);
        unsafe {
            clflush(ring.host.add(base), 64);
            core::arch::x86_64::_mm_sfence();
        }
        self.prev[plan.slot] = seq;
        Ok(())
    }
}

/// Patch the declared seq words of the retained insts BO in place; returns the patched word count. Every word that
/// differs must be in the inventory; only those cache lines are flushed; the BO is then read back and compared.
fn patch_insts(
    bo: &Bo,
    host: &mut [u8],
    sites: &[(usize, PatchKind)],
    seq0: u32,
) -> Result<usize, String> {
    let old = host.to_vec();
    patch_seq(host, sites, seq0);
    let changed: Vec<usize> = old
        .chunks_exact(4)
        .zip(host.chunks_exact(4))
        .enumerate()
        .filter(|(_, (o, n))| o != n)
        .map(|(w, _)| w)
        .collect();
    let mut declared: Vec<usize> = sites.iter().map(|s| s.0).collect();
    declared.sort_unstable();
    declared.dedup();
    if let Some(w) = changed.iter().find(|w| declared.binary_search(w).is_err()) {
        return Err(format!("patch_seq changed undeclared insts word {w}"));
    }
    for &w in &changed {
        let v = u32::from_le_bytes(host[4 * w..4 * w + 4].try_into().unwrap());
        unsafe { core::ptr::write_volatile((bo.host as *mut u32).add(w), v) };
    }
    let mut last_line = usize::MAX;
    for &w in &changed {
        let line = 4 * w / 64;
        if line != last_line {
            unsafe { clflush(bo.host.add(line * 64), 64) };
            last_line = line;
        }
    }
    unsafe { clflush(bo.host, host.len()) };
    if bo.as_slice()[..host.len()] != *host {
        return Err("insts BO readback differs from the patched host image".into());
    }
    Ok(changed.len())
}

/// Packed-C bytes equal check helper: number of mismatching i32s of an unpacked result against the reference.
fn count_mismatch(got: &[i32], want: &[i32]) -> usize {
    got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len())
}

struct Rig<'d> {
    cmd: Bo,
    insts: Bo,
    a: Bo,
    b: Bo,
    c: Bo,
    ring: Bo,
    pdi: Bo,
    insts_host: Vec<u8>,
    /// seq0 baked into the retained insts right now (what the NPU polls for).
    insts_seq0: u32,
    ctx: HwCtx<'d>,
}

struct Plan {
    empty: bool,
    sites: Vec<(usize, PatchKind)>,
    c_bytes: usize,
    skip: Option<usize>,
    /// Distinct operand sets; physical slot s uses set `s % sets`.
    sets: usize,
    /// Eager packed C per set (empty for the empty modes).
    eager: Vec<Vec<u8>>,
}

#[derive(Default)]
struct RoundOut {
    busy_us: f64,
    wall_us: f64,
    first_done_us: Option<f64>,
    rts: Vec<f64>,
    patched: usize,
    premature: usize,
    /// Byte-equal-to-eager checks done in-loop (paced slot reuse) and how many passed.
    early_total: usize,
    early_ok: usize,
    /// Runs whose done line was observed well formed.
    done_seen: usize,
    /// Slots (by run) whose done line did not carry the expected seq after completion (nslots >= S only).
    done_missing: Vec<usize>,
    /// Stale-seq: done lines carrying the NEW expected seq / the OLD baked seq after completion.
    new_done: usize,
    old_done: usize,
    failure: Option<String>,
}

fn run_round(
    rig: &mut Rig,
    cfg: &Cfg,
    pl: &Plan,
    prod: &mut Producer,
    r: usize,
    stale: bool,
) -> Result<RoundOut, String> {
    let s_runs = cfg.slots;
    let seq0 = 1 + (r * s_runs) as u32;
    let mut out = RoundOut::default();
    prod.quiesce(&rig.ring);
    if !pl.empty {
        rig.c.as_mut_slice().fill(POISON);
        rig.c.flush();
    }
    if r > 0 && !stale {
        out.patched = patch_insts(&rig.insts, &mut rig.insts_host, &pl.sites, seq0)?;
        rig.insts_seq0 = seq0;
    }
    let expect = |j: usize| seq0 + j as u32;
    let paced = cfg.mode.paced() && !stale;
    let mut published = vec![false; s_runs];
    let t_wall0 = Instant::now();
    if stale && !cfg.allow_hang {
        // The old seqs are still in the ring: nothing is published, the unpatched insts consume them.
        published.fill(true);
    } else if !paced {
        for j in 0..s_runs {
            prod.publish(&rig.ring, j, expect(j))?;
            published[j] = true;
        }
    }
    let refs = [&rig.a, &rig.b, &rig.c, &rig.ring];
    let t_submit = Instant::now();
    let seq = rig.ctx.submit(&mut rig.cmd, &rig.insts, &refs)?;
    let mut t_pub: Vec<Option<Instant>> = vec![None; s_runs];
    let mut t_done: Vec<Option<Instant>> = vec![None; s_runs];
    let mut obs_err: Option<String> = None;
    if !(stale && !cfg.allow_hang) {
        for j in 0..s_runs {
            if paced && !published[j] {
                if !pl.empty {
                    let off = prod.plans[j].slot * pl.c_bytes;
                    unsafe { clflush(rig.c.host.add(off), 64) };
                    if rig.c.as_slice()[off..off + 64].iter().any(|&x| x != POISON) {
                        out.premature += 1;
                    }
                }
                if let Err(e) = prod.publish(&rig.ring, j, expect(j)) {
                    obs_err = Some(e);
                    break;
                }
                published[j] = true;
                t_pub[j] = Some(Instant::now());
            }
            if pl.skip == Some(j) {
                // The skipped run writes no done line: never wait for it, keep publishing the rest.
                continue;
            }
            let deadline = Instant::now() + Duration::from_millis(cfg.timeout_ms);
            match spin_done(
                &rig.ring,
                prod.layout,
                prod.plans[j].slot,
                expect(j),
                deadline,
            ) {
                Ok(t) => {
                    t_done[j] = Some(t);
                    out.done_seen += 1;
                    // A slot that a later run of this round reuses is verified now (and re-poisoned) before
                    // that run can be published and overwrite it; final occupants are verified after the wait.
                    if !pl.empty && j + cfg.nslots < s_runs {
                        let slot = prod.plans[j].slot;
                        let (off, len) = (slot * pl.c_bytes, pl.c_bytes);
                        unsafe { clflush(rig.c.host.add(off), len) };
                        out.early_total += 1;
                        out.early_ok += (rig.c.as_slice()[off..off + len]
                            == *pl.eager[slot % pl.sets])
                            as usize;
                        rig.c.as_mut_slice()[off..off + len].fill(POISON);
                        unsafe { clflush(rig.c.host.add(off), len) };
                    }
                }
                Err(e) => {
                    obs_err = Some(e);
                    break;
                }
            }
        }
    }
    if obs_err.is_some() && !cfg.allow_hang {
        // Never leave the NPU polling for a seq we owe it, and never overwrite a slot whose seq is not done yet.
        let pending = prod.rescue(
            &rig.ring,
            &mut published,
            &expect,
            Instant::now() + Duration::from_millis(cfg.timeout_ms),
        );
        if !pending.is_empty() {
            obs_err = Some(format!(
                "{}; rescue could not publish runs {pending:?} (slots still busy)",
                obs_err.take().unwrap()
            ));
        }
    }
    let waited = rig.ctx.wait(&rig.cmd, seq, cfg.timeout_ms);
    let t_end = Instant::now();
    out.busy_us = us(t_end - t_submit);
    out.wall_us = us(t_end - t_wall0);
    match (&waited, &obs_err) {
        (Ok(s), None) if *s == ERT_STATE_COMPLETED => {}
        (Ok(s), e) => {
            out.failure = Some(format!(
                "state {s} (want {ERT_STATE_COMPLETED}){}",
                e.as_ref().map(|e| format!("; {e}")).unwrap_or_default()
            ))
        }
        (Err(e), o) => {
            out.failure = Some(format!(
                "wait failed: {e}{}",
                o.as_ref().map(|o| format!("; {o}")).unwrap_or_default()
            ))
        }
    }
    if out.failure.is_some() {
        return Ok(out);
    }
    let first = t_done.iter().flatten().next().copied();
    out.first_done_us = first.map(|t| us(t - t_submit));
    if paced {
        for j in 0..s_runs {
            if let (Some(p), Some(d)) = (t_pub[j], t_done[j]) {
                out.rts.push(us(d - p));
            }
        }
    } else if !stale {
        let mut prev = t_submit;
        for t in t_done.iter().flatten() {
            out.rts.push(us(*t - prev));
            prev = *t;
        }
    }
    if cfg.nslots >= s_runs {
        for j in 0..s_runs {
            let (s, _, _) = read_done(&rig.ring, prod.layout, prod.plans[j].slot);
            if s != expect(j) {
                out.done_missing.push(j);
            }
            if s == expect(j) {
                out.new_done += 1;
            }
            if s == rig.insts_seq0 + j as u32 {
                out.old_done += 1;
            }
        }
    }
    Ok(out)
}

/// One eager `design_v9` submit per operand set on the ring's own hardware context (its PDI is the identical V9 PDI,
/// already configured); returns the packed C bytes per set.
fn eager_runs(
    dev: &Device,
    ctx: &HwCtx,
    d: &ArrayDesign,
    packs: &[[Vec<u8>; 2]],
    timeout_ms: u64,
) -> Result<Vec<Vec<u8>>, String> {
    let insts = dev.dev_bo(&d.insts)?;
    let mut cmd = dev.cmd_bo()?;
    let mut result = Vec::new();
    for (i, p) in packs.iter().enumerate() {
        let mut a = dev.shmem_bo(d.args[0].bytes)?;
        a.as_mut_slice().copy_from_slice(&p[0]);
        a.flush();
        let mut b = dev.shmem_bo(d.args[1].bytes)?;
        b.as_mut_slice().copy_from_slice(&p[1]);
        b.flush();
        let mut c = dev.shmem_bo(d.args[2].bytes)?;
        c.as_mut_slice().fill(POISON);
        c.flush();
        let seq = ctx.submit(&mut cmd, &insts, &[&a, &b, &c])?;
        let state = ctx.wait(&cmd, seq, timeout_ms)?;
        if state != ERT_STATE_COMPLETED {
            return Err(format!("eager submit set {i}: state {state}"));
        }
        c.flush();
        result.push(c.as_slice().to_vec());
    }
    Ok(result)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("npu-ring: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    match run(&cfg) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            println!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}

fn run(cfg: &Cfg) -> Result<bool, String> {
    let empty = cfg.mode.empty();
    let (m, n, k) = if empty {
        (512, 512, 64)
    } else {
        (cfg.m, cfg.n, cfg.k)
    };
    let s_runs = cfg.slots;
    let layout = RingLayout { nslots: cfg.nslots };
    let d = match catch_unwind(AssertUnwindSafe(|| {
        design_v9(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast)
    })) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("npu-ring: design_v9({m},{n},{k}) rejected this shape\n{USAGE}");
            std::process::exit(2);
        }
    };
    assert!(d.args.len() == 3, "design_v9 must have args A, B, C");
    let (a_bytes, b_bytes, c_bytes) = (d.args[0].bytes, d.args[1].bytes, d.args[2].bytes);
    let useful = d.useful_ops();
    let used = s_runs.min(cfg.nslots);
    let sets = (if (m as u64) * (n as u64) * (k as u64) <= 1 << 24 {
        used.min(4)
    } else {
        1
    })
    .max(1);
    let neg_skip =
        (cfg.neg == Some(NegKind::SkipDone)).then_some(if s_runs >= 2 { s_runs / 2 } else { 0 });
    let max_rounds = cfg.rounds.unwrap_or(match (cfg.fit_ms, cfg.neg) {
        (Some(_), _) => 1000,
        (None, Some(NegKind::StaleSeq)) => 2,
        _ => 1,
    });
    let min_rounds = if cfg.neg == Some(NegKind::StaleSeq) {
        2
    } else {
        1
    };

    // Operand sets (not for the empty modes). The silicon CPU reference is the exact separable O(M*N) closed form
    // A[r,kk] = ra(r) (constant over kk), B[kk,c] = cb(c) => C[r,c] = epilogue(K * ra(r) * cb(c)); amplitudes 7..19 and
    // signs vary per row / column / set so the int8 outputs are nonzero and non-constant. A naive 4096x1280x2560
    // reference per set would not fit the hipx window; eager byte equality per set still covers layout errors.
    let mut packs: Vec<[Vec<u8>; 2]> = Vec::new();
    let mut refs: Vec<Vec<i32>> = Vec::new();
    if !empty {
        for set in 0..sets {
            let ra: Vec<i32> = (0..m)
                .map(|r| {
                    (7 + (r * 5 + set * 3) % 13) as i32
                        * if (r * 7 + set) % 3 == 0 { -1 } else { 1 }
                })
                .collect();
            let cb: Vec<i32> = (0..n)
                .map(|c| {
                    (7 + (c * 3 + set * 7) % 13) as i32
                        * if (c * 5 + set * 2) % 4 == 1 { -1 } else { 1 }
                })
                .collect();
            let a: Vec<i8> = (0..m * k).map(|i| ra[i / k] as i8).collect();
            let b: Vec<i8> = (0..k * n).map(|i| cb[i % n] as i8).collect();
            packs.push(d.pack_in(&a, &b));
            let mut want = Vec::with_capacity(m * n);
            for &x in &ra {
                for &y in &cb {
                    want.push(apply_epilogue(
                        (k as i32).wrapping_mul(x * y),
                        V8_DEFAULT_EPILOGUE,
                    ));
                }
            }
            refs.push(want);
        }
    }

    // Persistent design.
    let slot_plans: Vec<SlotPlan> = (0..s_runs)
        .map(|j| {
            if empty {
                SlotPlan {
                    slot: j % cfg.nslots,
                    a_off: 0,
                    b_off: 0,
                    c_off: 0,
                }
            } else {
                SlotPlan {
                    slot: j % cfg.nslots,
                    a_off: ((j % cfg.nslots) * a_bytes) as u64,
                    b_off: ((j % cfg.nslots) * b_bytes) as u64,
                    c_off: ((j % cfg.nslots) * c_bytes) as u64,
                }
            }
        })
        .collect();
    let neg = match (cfg.neg, neg_skip) {
        (Some(NegKind::SkipDone), Some(js)) => Some(Neg::SkipDone(js)),
        (Some(NegKind::StaleSeq), _) => Some(Neg::SkipSeqPatch),
        _ => None,
    };
    let build = |lean: bool| {
        if lean {
            persistent_v9_lean(m, n, k, layout, &slot_plans, 1, neg)
        } else {
            persistent_v9(m, n, k, layout, &slot_plans, 1, neg)
        }
    };
    let pd = if empty {
        empty_persistent(layout, &slot_plans, 1)
    } else {
        build(cfg.lean)
    };
    // Same operands/slots/neg: the full body is the reference length for the lean report (lean run 0 is full, runs 1.. lean).
    let (insts_full_bytes, insts_lean_bytes) = if cfg.lean {
        (build(false).insts.len(), pd.insts.len())
    } else {
        (pd.insts.len(), 0)
    };
    if cfg.lean && pd.patch_sites.len() != 2 * s_runs {
        return Err(format!("lean design has {} patch sites for {s_runs} runs, expected exactly 2 seq words per run", pd.patch_sites.len()));
    }
    if pd.args.len() != 4 {
        return Err(format!(
            "persistent design has {} args, expected A, B, C, ring",
            pd.args.len()
        ));
    }
    let need = [
        if empty { 0 } else { used * a_bytes },
        if empty { 0 } else { used * b_bytes },
        if empty { 0 } else { used * c_bytes },
        layout.bytes(),
    ];
    let sizes: Vec<usize> = (0..4)
        .map(|i| round_up(pd.args[i].bytes.max(need[i]).max(4096), 4096))
        .collect();
    println!(
        "setup: mode={} variant={} shape={m}x{n}x{k} slots={s_runs} nslots={} sets={sets} pdi {} B ({}) insts {} B patch_sites {} ({} per run) arenas A {} B {} C {} ring {} B",
        cfg.mode.name(),
        if cfg.lean { "lean" } else { "full" },
        cfg.nslots,
        pd.pdi.len(),
        if empty { "empty design".to_string() } else { format!("pdi_same_as_eager={}", pd.pdi == d.pdi) },
        pd.insts.len(),
        pd.patch_sites.len(),
        pd.patch_sites.len() / s_runs,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3]
    );
    if cfg.lean {
        println!(
            "lean: insts_bytes_lean={insts_lean_bytes} insts_bytes_full={insts_full_bytes} saved={} S={s_runs} (run 0 full, runs 1..{} lean with forced col0 S2MM4/MM2S0 restore; initial run of every submission is full)",
            insts_full_bytes as i64 - insts_lean_bytes as i64,
            s_runs.saturating_sub(1)
        );
    }

    // Host overrun guard demonstration on a scratch ring image (never forced on silicon).
    let mut guard_ok = true;
    {
        let mut img = vec![0u8; layout.bytes()];
        layout.initialize(&mut img);
        let first_ok = producer_may_publish(&img, layout, 0);
        let refused = !producer_may_publish(&img, layout, cfg.nslots);
        println!(
            "NEG overrun: guard {} publish of run {} into slot 0 while done[0]=0 != seq 1 (run 0 allowed={first_ok})",
            if refused { "refused" } else { "ALLOWED (BUG)" },
            cfg.nslots
        );
        guard_ok &= refused && first_ok;
    }

    let mut dev = Device::open().map_err(|e| format!("device open failed: {e}"))?;
    dev.map_heap(64 << 20)
        .map_err(|e| format!("device heap map failed: {e}"))?;
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    // ONE hardware context + PDI for the eager baseline and the persistent command: the V9 PDI is byte-identical.
    let ctx = HwCtx::create(&dev, ntiles, MAX_OPC)?;
    if !empty && pd.pdi != d.pdi {
        return Err("persistent PDI differs from the design_v9 PDI: eager and ring cannot share one context".into());
    }
    let pdi = dev.dev_bo(&pd.pdi)?;
    ctx.config_cu(&pdi)?;
    // Eager baseline: same process, same shape PDI/context, before the first ring submit.
    let mut eager_c: Vec<Vec<u8>> = Vec::new();
    let mut checks_total = 0usize;
    let mut checks_ok = 0usize;
    if !empty {
        eager_c = eager_runs(&dev, &ctx, &d, &packs, cfg.timeout_ms)?;
        for (i, c) in eager_c.iter().enumerate() {
            let bad = count_mismatch(&d.unpack_out(c), &refs[i]);
            println!("eager set {i}: mismatches vs CPU {bad}");
            if bad != 0 {
                return Err(format!("eager baseline set {i} is not exact ({bad} mismatches); ring comparison is meaningless"));
            }
        }
    }

    let mut a = dev.shmem_bo(sizes[0])?;
    let mut b = dev.shmem_bo(sizes[1])?;
    let c = dev.shmem_bo(sizes[2])?;
    let mut ring = dev.shmem_bo(sizes[3])?;
    if !empty {
        for s in 0..used {
            let p = &packs[s % sets];
            a.as_mut_slice()[s * a_bytes..(s + 1) * a_bytes].copy_from_slice(&p[0]);
            b.as_mut_slice()[s * b_bytes..(s + 1) * b_bytes].copy_from_slice(&p[1]);
        }
    }
    a.flush();
    b.flush();
    layout.initialize(&mut ring.as_mut_slice()[..layout.bytes()]);
    ring.flush();

    let insts = dev.dev_bo(&pd.insts)?;
    let cmd = dev.cmd_bo()?;
    let mut rig = Rig {
        cmd,
        insts,
        a,
        b,
        c,
        ring,
        pdi,
        insts_host: pd.insts.clone(),
        insts_seq0: 1,
        ctx,
    };
    let _ = &rig.pdi;
    let plan = Plan {
        empty,
        sites: pd.patch_sites.clone(),
        c_bytes,
        skip: neg_skip,
        sets,
        eager: eager_c,
    };
    let mut prod = Producer {
        layout,
        plans: slot_plans,
        m: m as u32,
        flags: empty as u32,
        prev: vec![0; cfg.nslots],
    };
    let inventory = pd.patch_sites.len();

    let mut busy = Vec::new();
    let mut wall = Vec::new();
    let mut rts = Vec::new();
    let mut firstdone = Vec::new();
    let mut premature = 0usize;
    let mut neg_detected: Option<bool> = None;
    let mut all_ok = guard_ok;
    let mut rounds_done = 0usize;
    let mut last_verified_cpu = false;
    let start = Instant::now();
    while rounds_done < max_rounds {
        let r = rounds_done;
        let stale = cfg.neg == Some(NegKind::StaleSeq) && r + 1 == max_rounds;
        let o = run_round(&mut rig, cfg, &plan, &mut prod, r, stale)?;
        premature += o.premature;
        if let Some(f) = &o.failure {
            println!("round {r}: FAIL {f}");
            if cfg.allow_hang && cfg.neg == Some(NegKind::StaleSeq) && stale {
                println!("NEG stale-seq: allowed hang, no done for new seqs within {} ms, patched_words=0/{inventory} -> detected", cfg.timeout_ms);
                neg_detected = Some(true);
            } else if cfg.allow_hang {
                println!("hang allowed (--allow-hang): reporting and stopping");
            } else {
                all_ok = false;
            }
            rounds_done += 1;
            last_verified_cpu = false;
            break;
        }
        rounds_done += 1;
        println!(
            "round {r}{}: seq0={} patched_words={}/{inventory} busy_us={:.1} firstdone_us={} done_seen={}",
            if stale { " (stale-seq, insts unpatched)" } else { "" },
            1 + (r * s_runs) as u32,
            o.patched,
            o.busy_us,
            o.first_done_us.map_or("n/a".to_string(), |v| format!("{v:.1}")),
            o.done_seen
        );
        if !stale {
            busy.push(o.busy_us);
            wall.push(o.wall_us);
            if let Some(f) = o.first_done_us {
                firstdone.push(f);
            }
            rts.extend(o.rts.iter().copied());
        }
        if !empty {
            // Final occupant of every physical slot (earlier occupants were checked in-loop before reuse).
            rig.c.flush();
            let cs = rig.c.as_slice();
            for s in 0..used {
                checks_total += 1;
                checks_ok += (cs[s * c_bytes..(s + 1) * c_bytes] == *plan.eager[s % sets]) as usize;
            }
            checks_total += o.early_total;
            checks_ok += o.early_ok;
        }
        last_verified_cpu = false;
        if !empty && (r == 0 || (cfg.verify_every && !stale)) {
            cpu_verify(
                &rig.c,
                &d,
                &refs,
                used,
                sets,
                c_bytes,
                &format!("round {r}"),
                &mut all_ok,
            );
            last_verified_cpu = true;
        }
        match cfg.neg {
            Some(NegKind::SkipDone) => {
                let js = neg_skip.unwrap();
                let detected = o.done_missing == vec![js];
                println!(
                    "NEG skip-done: run {js} slot {} done line {} (missing runs {:?}); {} of {} other done lines present; producer published all remaining seqs -> {}",
                    prod.plans[js].slot,
                    if o.done_missing.contains(&js) { "missing" } else { "PRESENT" },
                    o.done_missing,
                    o.done_seen,
                    s_runs - 1,
                    if detected { "detected" } else { "NOT DETECTED" }
                );
                neg_detected = Some(neg_detected.unwrap_or(true) && detected);
            }
            Some(NegKind::StaleSeq) if stale => {
                let detected =
                    o.patched == 0 && o.new_done == 0 && o.old_done == s_runs && inventory > 0;
                println!(
                    "NEG stale-seq: patched_words={}/{inventory} (inventory requires all), new done lines {} of {s_runs}, done lines still showing old seqs {} of {s_runs} -> {}",
                    o.patched,
                    o.new_done,
                    o.old_done,
                    if detected { "detected" } else { "NOT DETECTED" }
                );
                neg_detected = Some(detected);
            }
            _ => {}
        }
        if r + 1 >= max_rounds {
            break;
        }
        if let Some(ms) = cfg.fit_ms {
            if r + 1 >= min_rounds && start.elapsed() >= Duration::from_millis(ms) {
                break;
            }
        }
    }
    if !empty && rounds_done > 1 && !last_verified_cpu && all_ok {
        cpu_verify(
            &rig.c,
            &d,
            &refs,
            used,
            sets,
            c_bytes,
            "final round",
            &mut all_ok,
        );
    }

    let cpu_total = if empty { 0 } else { cpu_counts().0 };
    let cpu_ok = if empty { 0 } else { cpu_counts().1 };
    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let (per_round, per_wall) = (mean(&busy), mean(&wall));
    let tops = |t_us: f64| {
        if empty {
            0.0
        } else {
            s_runs as f64 * useful as f64 / (t_us * 1e6)
        }
    };
    let fmt3 = |x: f64| {
        if x.is_nan() {
            "n/a".to_string()
        } else {
            format!("{x:.3}")
        }
    };
    let rt = stats(&mut rts);
    let fd = stats(&mut firstdone);
    println!(
        "ring: mode={} shape={m}x{n}x{k} slots={s_runs} rounds={rounds_done} per_round_us={} per_slot_us={} busy_tops={} wall_tops={} rt_us p50={} p99={} min={} firstdone_us p50={} min={} exact={} bytes_equal_eager={} premature_c={premature}",
        cfg.mode.name(),
        fmt3(per_round),
        fmt3(per_round / s_runs as f64),
        if per_round.is_nan() { "n/a".to_string() } else { format!("{:.4}", tops(per_round)) },
        if per_wall.is_nan() { "n/a".to_string() } else { format!("{:.4}", tops(per_wall)) },
        rt.map_or("n/a".to_string(), |s| fmt3(s.0)),
        rt.map_or("n/a".to_string(), |s| fmt3(s.1)),
        rt.map_or("n/a".to_string(), |s| fmt3(s.2)),
        fd.map_or("n/a".to_string(), |s| fmt3(s.0)),
        fd.map_or("n/a".to_string(), |s| fmt3(s.2)),
        if empty { "n/a".to_string() } else { format!("{cpu_ok}/{cpu_total}") },
        if empty { "n/a".to_string() } else { format!("{checks_ok}/{checks_total}") }
    );
    if !empty && checks_ok != checks_total {
        all_ok = false;
    }
    if premature != 0 {
        all_ok = false;
    }
    if cfg.neg.is_some() && neg_detected != Some(true) {
        println!(
            "NEG {}: not detected",
            if cfg.neg == Some(NegKind::SkipDone) {
                "skip-done"
            } else {
                "stale-seq"
            }
        );
        all_ok = false;
    }
    if !empty {
        println!("note: exact=CPU-direct checks (round 0 + final round); bytes_equal_eager covers every slot of every round against the eager C that was itself verified exact vs the CPU reference, so all rounds are CPU-exact by transitivity");
    }
    println!("result: {}", if all_ok { "PASS" } else { "FAIL" });
    Ok(all_ok)
}

use npu_tools::util::{parse_num, percentile, round_up, us};
use std::sync::atomic::AtomicUsize;
static CPU_TOTAL: AtomicUsize = AtomicUsize::new(0);
static CPU_OK: AtomicUsize = AtomicUsize::new(0);
fn cpu_counts() -> (usize, usize) {
    (
        CPU_TOTAL.load(Ordering::Relaxed),
        CPU_OK.load(Ordering::Relaxed),
    )
}

/// Exact CPU check of every slot currently in the C arena (unpack + reference compare).
#[allow(clippy::too_many_arguments)]
fn cpu_verify(
    c: &Bo,
    d: &ArrayDesign,
    refs: &[Vec<i32>],
    s_runs: usize,
    sets: usize,
    c_bytes: usize,
    what: &str,
    all_ok: &mut bool,
) {
    c.flush();
    let cs = c.as_slice();
    for j in 0..s_runs {
        let bad = count_mismatch(
            &d.unpack_out(&cs[j * c_bytes..(j + 1) * c_bytes]),
            &refs[j % sets],
        );
        CPU_TOTAL.fetch_add(1, Ordering::Relaxed);
        if bad == 0 {
            CPU_OK.fetch_add(1, Ordering::Relaxed);
        } else {
            println!(
                "{what}: slot {j} (set {}): {bad} mismatches vs CPU",
                j % sets
            );
            *all_ok = false;
        }
    }
}
