// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Persistent-ring V9 in the whole-array simulator: late publication, slot wrap, patch-and-resubmit, negatives.
//! The host side is a `Config::execute_with` producer called once per simulator wait tick (see `Producer`).
use peacemaker_npu::kernels::{
    gemm_array::{self, ArrayDesign},
    gemm_core::{Control, Epilogue},
    ring::*,
};
use peacemaker_npu::sim::config::Config;
use std::ops::Range;

const INT8: Epilogue = Epilogue::Int8 { shift: 12 };
const POISON: u8 = 0xa5;
const GUARD: u8 = 0xcc;
const DONE_MAGIC: u32 = 0x454e4f44;
const LINE: usize = 64;
/// Wait tick of the first operand write: the NPU is already polling an empty ring.
const FIRST_AT: usize = 37;
/// Wait ticks between operand writes and the (last) `seq` write of a run.
const OPERAND_TO_SEQ: usize = 3;

fn matrices(m: usize, n: usize, k: usize, seed: u32) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut a: Vec<_> = (0..m * k).map(|_| random()).collect();
    let mut b: Vec<_> = (0..k * n).map(|_| random()).collect();
    a[0] = -128;
    b[0] = -128;
    (a, b)
}
fn rd(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn wr32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes())
}
fn wr64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes())
}
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| a[i] != b[i])
}
fn assert_bytes(a: &[u8], b: &[u8], what: &str) {
    if let Some(i) = first_diff(a, b) {
        panic!(
            "{what}: first byte mismatch at {i} (lens {} {})",
            a.len(),
            b.len()
        )
    }
}
fn diff_words(a: &[u8], b: &[u8]) -> Vec<usize> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 4, 0);
    (0..a.len() / 4)
        .filter(|&i| rd(a, i * 4) != rd(b, i * 4))
        .collect()
}
/// The exact 64-byte line the NPU writes for run `g` of slot `s`.
fn done_bytes(slot: usize, seq: u32) -> [u8; LINE] {
    let mut l = [0; LINE];
    wr32(&mut l, 0, seq);
    wr32(&mut l, 4, slot as u32);
    wr32(&mut l, 8, DONE_MAGIC);
    l
}
fn done_at(ring: &[u8], l: RingLayout, slot: usize) -> [u8; LINE] {
    let o = l.done_line(slot);
    ring[o..o + LINE].try_into().unwrap()
}
/// Consumer view: run `g` is complete iff its done line carries seq `g+1`.
fn consumer_done(ring: &[u8], l: RingLayout, g: usize) -> bool {
    rd(ring, l.done_line(g % l.nslots)) == g as u32 + 1
}

struct Rig {
    m: usize,
    n: usize,
    k: usize,
    layout: RingLayout,
    eager: ArrayDesign,
    d: PersistentDesign,
    plans: Vec<SlotPlan>,
    cb: usize,
    used: usize,
}
fn plans(base: usize, runs: usize, nslots: usize, a: usize, b: usize, c: usize) -> Vec<SlotPlan> {
    (0..runs)
        .map(|i| {
            let slot = (base + i) % nslots;
            SlotPlan {
                slot,
                a_off: (slot * a) as u64,
                b_off: (slot * b) as u64,
                c_off: (slot * c) as u64,
            }
        })
        .collect()
}
fn rig(k: usize, nslots: usize, runs: usize, seq0: u32, neg: Option<Neg>) -> Rig {
    rig_shape(512, 512, k, nslots, runs, 0, seq0, neg, false)
}
/// One submission of `runs` runs starting at GLOBAL run index `base` (slot `(base+i)%nslots`, seq `seq0+i`); `lean` selects
/// `persistent_v9_lean` (run 0 FULL, later runs lean) instead of `persistent_v9`.
fn rig_shape(
    m: usize,
    n: usize,
    k: usize,
    nslots: usize,
    runs: usize,
    base: usize,
    seq0: u32,
    neg: Option<Neg>,
    lean: bool,
) -> Rig {
    let eager = gemm_array::design_v9(m, n, k, INT8, Control::Fast);
    let (a, b, c) = (
        eager.args[0].bytes,
        eager.args[1].bytes,
        eager.args[2].bytes,
    );
    let layout = RingLayout { nslots };
    let plans = plans(base, runs, nslots, a, b, c);
    let d = if lean {
        persistent_v9_lean(m, n, k, layout, &plans, seq0, neg)
    } else {
        persistent_v9(m, n, k, layout, &plans, seq0, neg)
    };
    assert_eq!(d.args.len(), 4, "persistent args = [A,B,C,ring]");
    // Arenas hold exactly the slots a plan uses (max slot + 1); unused ring slots have no arena.
    let used = plans.iter().map(|p| p.slot + 1).max().unwrap();
    for p in &plans {
        assert!(
            p.a_off as usize + a <= d.args[0].bytes
                && p.b_off as usize + b <= d.args[1].bytes
                && p.c_off as usize + c <= d.args[2].bytes,
            "plan {p:?} outside arenas"
        );
    }
    assert!(d.args[2].bytes == used * c && d.args[3].bytes == layout.bytes());
    assert_eq!((d.a_bytes(), d.b_bytes(), d.c_bytes()), (a, b, c));
    Rig {
        m,
        n,
        k,
        layout,
        eager,
        d,
        plans,
        cb: c,
        used,
    }
}
fn fresh_args(r: &Rig) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = r.d.args.iter().map(|s| vec![GUARD; s.bytes]).collect();
    v[2].fill(POISON);
    r.layout.initialize(&mut v[3]);
    v
}

/// Operands of global run `g`, its eager-V9 packed C bytes and the CPU reference.
struct Golden {
    a: Vec<u8>,
    b: Vec<u8>,
    eager_c: Vec<u8>,
    reference: Vec<i32>,
}
fn golden(r: &Rig, g: usize) -> Golden {
    let (a, b) = matrices(r.m, r.n, r.k, 0xc001d00d + g as u32 * 7919);
    let [ap, bp] = r.eager.pack_in(&a, &b);
    let [dap, dbp] = r.d.pack_in(&a, &b);
    assert_eq!(
        (&dap, &dbp),
        (&ap, &bp),
        "persistent packing == eager packing"
    );
    let mut args = vec![ap.clone(), bp.clone(), vec![POISON; r.eager.args[2].bytes]];
    Config::from_pdi(&r.eager.pdi)
        .unwrap()
        .submit(&r.eager.insts, &mut args)
        .unwrap();
    let reference = r.eager.reference(&a, &b);
    assert_eq!(
        r.eager.unpack_out(&args[2]),
        reference,
        "eager V9 == CPU reference"
    );
    Golden {
        a: ap,
        b: bp,
        eager_c: args.swap_remove(2),
        reference,
    }
}
fn goldens(r: &Rig, total: usize) -> Vec<Golden> {
    (0..total).map(|g| golden(r, g)).collect()
}
fn assert_c(r: &Rig, c: &[u8], g: &Golden, what: &str) {
    assert_bytes(
        c,
        &g.eager_c,
        &format!("{what}: persistent C != eager V9 bytes"),
    );
    assert_eq!(
        r.d.unpack_out(c),
        g.reference,
        "{what}: persistent C != CPU reference"
    );
}

/// Host publisher driven by the simulator's wait ticks. Runs are published strictly in order; each run goes through
/// stage 0 (guard open: verify+re-poison the reused C slot, write operands), a delay, stage 1 (C must still be poison,
/// slot fields, then `seq` LAST), then `gap` ticks before the next run. Every write it makes is mirrored in `shadow`.
struct Producer<'a> {
    r: &'a Rig,
    gold: &'a [Golden],
    base: usize,
    runs: usize,
    gap: usize,
    skipped: Vec<usize>,
    next: usize,
    stage: u8,
    at: usize,
    shadow: Vec<Vec<u8>>,
    last: Vec<[u8; LINE]>,
    seen: Vec<(usize, [u8; LINE])>,
    calls: usize,
    reuse_checks: usize,
    guard_waits: usize,
    /// Negative runs: a wrong old output at reuse is recorded in `bad_reuse` (global run index) instead of panicking.
    lenient: bool,
    bad_reuse: Vec<usize>,
}
impl<'a> Producer<'a> {
    fn new(
        r: &'a Rig,
        gold: &'a [Golden],
        base: usize,
        runs: usize,
        gap: usize,
        skipped: Vec<usize>,
        args: &[Vec<u8>],
    ) -> Self {
        let last = (0..r.layout.nslots)
            .map(|s| done_at(&args[3], r.layout, s))
            .collect();
        Self {
            r,
            gold,
            base,
            runs,
            gap,
            skipped,
            next: 0,
            stage: 0,
            at: FIRST_AT,
            shadow: args.to_vec(),
            last,
            seen: Vec::new(),
            calls: 0,
            reuse_checks: 0,
            guard_waits: 0,
            lenient: false,
            bad_reuse: Vec::new(),
        }
    }
    fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }
    fn put(&mut self, args: &mut [Vec<u8>], arg: usize, off: usize, bytes: &[u8]) {
        args[arg][off..off + bytes.len()].copy_from_slice(bytes);
        self.shadow[arg][off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn c_range(&self, i: usize) -> Range<usize> {
        let c = self.r.plans[i].c_off as usize;
        c..c + self.r.cb
    }
    fn must_poison(&self, j: usize) -> bool {
        self.base + j < self.r.layout.nslots || (j == self.next && self.stage >= 1)
    }
    fn call(&mut self, t: usize, args: &mut [Vec<u8>]) {
        self.calls += 1;
        let l = self.r.layout;
        for s in 0..l.nslots {
            let line = done_at(&args[3], l, s);
            if line != self.last[s] {
                self.last[s] = line;
                if rd(&line, 8) == DONE_MAGIC {
                    self.seen.push((s, line))
                }
            }
        }
        if self.calls % 16 == 1 {
            for j in self.next..self.runs {
                if self.must_poison(j) {
                    assert!(
                        args[2][self.c_range(j)].iter().all(|&b| b == POISON),
                        "run {} C written before its publication completed (tick {t})",
                        self.base + j
                    );
                }
            }
        }
        if self.next == self.runs || t < self.at {
            return;
        }
        let (i, g) = (self.next, self.base + self.next);
        let p = self.r.plans[i];
        let c = self.c_range(i);
        if self.stage == 0 {
            if !producer_may_publish(&args[3], l, g) {
                self.guard_waits += 1;
                return;
            }
            if g >= l.nslots {
                // The slot's previous output is complete (done seen by the guard): verify it, then re-poison it for reuse.
                if self.lenient {
                    if args[2][c.clone()] != self.gold[g - l.nslots].eager_c[..] {
                        self.bad_reuse.push(g - l.nslots)
                    }
                } else {
                    assert_c(
                        self.r,
                        &args[2][c.clone()],
                        &self.gold[g - l.nslots],
                        &format!("old output of run {} before reuse by run {g}", g - l.nslots),
                    )
                }
                args[2][c.clone()].fill(POISON);
                self.reuse_checks += 1;
            } else {
                assert!(
                    args[2][c.clone()].iter().all(|&b| b == POISON),
                    "fresh C slot of run {g} not poison before publication"
                )
            }
            let (a_off, b_off) = (p.a_off as usize, p.b_off as usize);
            let (a, b) = (self.gold[g].a.clone(), self.gold[g].b.clone());
            self.put(args, 0, a_off, &a);
            self.put(args, 1, b_off, &b);
            self.stage = 1;
            self.at = t + OPERAND_TO_SEQ;
        } else {
            assert!(
                args[2][c].iter().all(|&b| b == POISON),
                "run {g}: C computed from operands whose seq was not yet published (tick {t})"
            );
            let line = l.slot_line(p.slot);
            // prog, m, flags, a/b/c offsets first; seq (byte 0 of the line) last.
            let mut fields = [0u8; LINE - 4];
            wr32(&mut fields, 0, 9);
            wr32(&mut fields, 4, self.r.m as u32);
            wr32(&mut fields, 8, 0);
            wr64(&mut fields, 12, p.a_off);
            wr64(&mut fields, 20, p.b_off);
            wr64(&mut fields, 28, p.c_off);
            self.put(args, 3, line + 4, &fields);
            self.put(args, 3, line, &(g as u32 + 1).to_le_bytes());
            self.next += 1;
            self.stage = 0;
            self.at = t + self.gap;
        }
    }
    /// After the submission: operands/ring-slot lines equal exactly what this producer wrote, every run published, every
    /// non-skipped DONE of this submission captured (mid-flight or final) and well formed.
    fn finish(&self, args: &[Vec<u8>]) {
        let l = self.r.layout;
        assert_eq!(self.next, self.runs, "producer did not publish every run");
        assert_bytes(
            &args[0],
            &self.shadow[0],
            "args0 != initial + producer writes",
        );
        assert_bytes(
            &args[1],
            &self.shadow[1],
            "args1 != initial + producer writes",
        );
        assert_bytes(
            &args[3][..l.done_line(0)],
            &self.shadow[3][..l.done_line(0)],
            "ring header/slot lines != initial + producer writes",
        );
        for &(s, line) in &self.seen {
            let seq = rd(&line, 0) as usize;
            assert!(
                seq >= 1
                    && (seq - 1) % l.nslots == s
                    && rd(&line, 4) == s as u32
                    && line[12..].iter().all(|&b| b == 0),
                "malformed done line seen on slot {s}: {line:?}"
            );
        }
        for i in 0..self.runs {
            if self.skipped.contains(&i) {
                continue;
            }
            let g = self.base + i;
            let (s, want) = (g % l.nslots, done_bytes(g % l.nslots, g as u32 + 1));
            assert!(
                self.seen.iter().any(|&(ss, line)| ss == s && line == want)
                    || done_at(&args[3], l, s) == want,
                "DONE of run {g} (seq {}) never captured",
                g + 1
            );
        }
    }
}
fn run(sim: &mut Config, insts: &[u8], args: &mut Vec<Vec<u8>>, p: &mut Producer) {
    sim.submit_with(insts, args, &mut |t, a| p.call(t, a))
        .unwrap()
}

/// Final C of the last run on each slot is exact (CPU reference + eager V9 bytes), unused slots/padding are still poison.
fn check_outputs(r: &Rig, gold: &[Golden], args: &[Vec<u8>], total: usize) {
    let l = r.layout;
    for s in 0..r.used {
        let c = &args[2][s * r.cb..(s + 1) * r.cb];
        match (0..total).rev().find(|g| g % l.nslots == s) {
            Some(g) => assert_c(r, c, &gold[g], &format!("slot {s} final run {g}")),
            None => assert!(c.iter().all(|&b| b == POISON)),
        }
    }
    assert!(
        args[2][r.used * r.cb..].iter().all(|&b| b == POISON),
        "C arena padding written"
    );
}
/// Every done line equals the last non-skipped run on its slot (or is untouched zero).
fn check_done_lines(l: RingLayout, args: &[Vec<u8>], total: usize, skipped_global: &[usize]) {
    for s in 0..l.nslots {
        let want = (0..total)
            .rev()
            .find(|g| g % l.nslots == s && !skipped_global.contains(g))
            .map(|g| done_bytes(s, g as u32 + 1))
            .unwrap_or([0; LINE]);
        assert_eq!(done_at(&args[3], l, s), want, "done line of slot {s}");
    }
}
/// Consumer view for a submission WITHOUT wrap (runs <= nslots): every run whose done line is absent.
fn missing_runs(args: &[Vec<u8>], l: RingLayout, total: usize) -> Vec<usize> {
    assert!(total <= l.nslots);
    (0..total)
        .filter(|&g| !consumer_done(&args[3], l, g))
        .collect()
}
/// With wrap an older DONE is legitimately overwritten (same line, newer seq): only the latest run on each slot is visible.
fn latest_missing(args: &[Vec<u8>], l: RingLayout, total: usize) -> Vec<usize> {
    (0..l.nslots)
        .filter_map(|s| (0..total).rev().find(|g| g % l.nslots == s))
        .filter(|&g| !consumer_done(&args[3], l, g))
        .collect()
}

/// Late publication, slots reused (wrap when `runs > nslots`): operands first, `seq` last, C untouched until then.
fn late_publication(k: usize) {
    for (runs, nslots) in [(2, 2), (3, 2)] {
        let r = rig(k, nslots, runs, 1, None);
        let gold = goldens(&r, runs);
        for gap in [0, 20_000] {
            let mut args = fresh_args(&r);
            let mut sim = Config::from_pdi(&r.d.pdi).unwrap();
            let mut p = Producer::new(&r, &gold, 0, runs, gap, vec![], &args);
            run(&mut sim, &r.d.insts, &mut args, &mut p);
            p.finish(&args);
            check_outputs(&r, &gold, &args, runs);
            check_done_lines(r.layout, &args, runs, &[]);
            assert!(latest_missing(&args, r.layout, runs).is_empty());
            assert_eq!(
                p.reuse_checks,
                runs.saturating_sub(nslots),
                "every wrapped run verified its predecessor before reuse"
            );
            eprintln!("PASS ring late publish K={k} runs={runs} nslots={nslots} gap={gap}: {} ticks, producer calls={}, guard waits={}, reuse checks={}",sim.ticks,p.calls,p.guard_waits,p.reuse_checks);
        }
    }
}
#[test]
fn late_publication_wrap_k64() {
    late_publication(64)
}
#[test]
fn late_publication_wrap_k128() {
    late_publication(128)
}

/// Patch inventory: the byte diff between a seq0=1 and seq0=1+S build is exactly the declared words, patching gives the
/// from-scratch build byte for byte, and the patched stream runs correctly as the second submission on the same context.
fn resubmit_patch(k: usize) {
    let (runs, nslots) = (2, 2);
    let r1 = rig(k, nslots, runs, 1, None);
    let r2 = rig(k, nslots, runs, 1 + runs as u32, None);
    let r3 = rig(k, nslots, runs, 1 + 2 * runs as u32, None);
    assert_eq!(r1.d.pdi, r2.d.pdi);
    let mut declared: Vec<usize> = r1.d.patch_sites.iter().map(|&(w, _)| w).collect();
    declared.sort();
    declared.dedup();
    assert!(!declared.is_empty());
    assert_eq!(
        declared.len(),
        r1.d.patch_sites.len(),
        "duplicate patch words"
    );
    assert_eq!(
        r1.d.patch_sites, r2.d.patch_sites,
        "inventory independent of seq0"
    );
    for (to, seq0) in [(&r2, 1 + runs as u32), (&r3, 1 + 2 * runs as u32)] {
        assert_eq!(
            diff_words(&r1.d.insts, &to.d.insts),
            declared,
            "byte diff must be exactly the declared words (seq0 {seq0})"
        );
        let mut patched = r1.d.insts.clone();
        patch_seq(&mut patched, &r1.d.patch_sites, seq0);
        assert_bytes(&patched, &to.d.insts, "patched insts != from-scratch build");
        assert_eq!(
            diff_words(&r1.d.insts, &patched),
            declared,
            "patch touched words outside the inventory"
        );
    }
    let gold = goldens(&r1, 2 * runs);
    let mut args = fresh_args(&r1);
    let mut sim = Config::from_pdi(&r1.d.pdi).unwrap();
    let mut p = Producer::new(&r1, &gold, 0, runs, 0, vec![], &args);
    run(&mut sim, &r1.d.insts, &mut args, &mut p);
    p.finish(&args);
    check_outputs(&r1, &gold, &args, runs);
    check_done_lines(r1.layout, &args, runs, &[]);
    let mut patched = r1.d.insts.clone();
    patch_seq(&mut patched, &r1.d.patch_sites, 1 + runs as u32);
    let mut p = Producer::new(&r1, &gold, runs, runs, 20_000, vec![], &args);
    run(&mut sim, &patched, &mut args, &mut p);
    p.finish(&args);
    check_outputs(&r1, &gold, &args, 2 * runs);
    check_done_lines(r1.layout, &args, 2 * runs, &[]);
    assert!(latest_missing(&args, r1.layout, 2 * runs).is_empty());
    assert_eq!(
        p.reuse_checks, runs,
        "second submission reused both slots after verifying them"
    );
    eprintln!("PASS ring patch resubmit K={k}: {} declared words of {}, exact byte diff + exact patched stream, {} ticks",declared.len(),r1.d.insts.len()/4,sim.ticks);
}
#[test]
fn resubmit_patch_inventory_exact_k64() {
    resubmit_patch(64)
}
#[test]
fn resubmit_patch_inventory_exact_k128() {
    resubmit_patch(128)
}

/// SkipDone(j): the run still computes, its done line stays absent, the consumer pins exactly run j, later dones work.
#[test]
fn skip_done_detected_run_index_while_later_done_works() {
    for (nslots, runs, skip) in [(2, 2, 0), (2, 2, 1), (4, 3, 1)] {
        let r = rig(64, nslots, runs, 1, Some(Neg::SkipDone(skip)));
        let gold = goldens(&r, runs);
        let mut args = fresh_args(&r);
        let mut sim = Config::from_pdi(&r.d.pdi).unwrap();
        let mut p = Producer::new(&r, &gold, 0, runs, 0, vec![skip], &args);
        run(&mut sim, &r.d.insts, &mut args, &mut p);
        p.finish(&args);
        assert_eq!(
            missing_runs(&args, r.layout, runs),
            vec![skip],
            "consumer must see exactly run {skip} without DONE"
        );
        check_done_lines(r.layout, &args, runs, &[skip]);
        for g in (0..runs).filter(|&g| g != skip) {
            assert!(
                consumer_done(&args[3], r.layout, g),
                "later/earlier done of run {g}"
            )
        }
        check_outputs(&r, &gold, &args, runs);
        eprintln!("PASS ring SkipDone({skip}) nslots={nslots} runs={runs}: detected run {skip}, other dones present, all C exact");
    }
}

/// SkipSeqPatch: resubmitting the previous stream completes at once on the OLD seqs, the consumer never sees a new done,
/// and its byte diff against the required stream is the whole inventory (zero once patched); then a patched resubmit works.
#[test]
fn stale_seq_resubmit_completes_on_old_seqs() {
    let (k, runs, nslots) = (64, 2, 2);
    let r = rig(k, nslots, runs, 1, Some(Neg::SkipSeqPatch));
    assert_bytes(
        &r.d.insts,
        &rig(k, nslots, runs, 1, None).d.insts,
        "SkipSeqPatch builder is the normal stream",
    );
    let required = rig(k, nslots, runs, 1 + runs as u32, None);
    let gold = goldens(&r, 2 * runs);
    let mut args = fresh_args(&r);
    let mut sim = Config::from_pdi(&r.d.pdi).unwrap();
    let mut p = Producer::new(&r, &gold, 0, runs, 0, vec![], &args);
    run(&mut sim, &r.d.insts, &mut args, &mut p);
    p.finish(&args);
    let first_ticks = sim.ticks;
    let before = args.clone();
    assert!(consumer_done(&args[3], r.layout, 0) && consumer_done(&args[3], r.layout, 1));
    // Host skips patch_seq. The inventory requires every declared word to change; the stream actually resubmitted differs from
    // the one already executed in ZERO words, and from the required stream in exactly the declared words.
    let mut declared: Vec<usize> = r.d.patch_sites.iter().map(|&(w, _)| w).collect();
    declared.sort();
    declared.dedup();
    assert!(!declared.is_empty());
    let executed_first = r.d.insts.clone();
    let resubmitted = r.d.insts.clone();
    let (actual_diff, required_diff) = (
        diff_words(&executed_first, &resubmitted),
        diff_words(&resubmitted, &required.d.insts),
    );
    assert_eq!(actual_diff.len(), 0, "stale resubmit changes no word");
    assert_eq!(
        required_diff, declared,
        "inventory requires every declared word to change"
    );
    assert!(actual_diff.len() < required_diff.len());
    let stale_diff = required_diff;
    let mut patched = r.d.insts.clone();
    patch_seq(&mut patched, &r.d.patch_sites, 1 + runs as u32);
    assert_eq!(diff_words(&patched, &required.d.insts).len(), 0);
    // Stale resubmit with a producer that never publishes: must complete without waiting for anything.
    let mut calls = 0usize;
    let t0 = sim.ticks;
    sim.submit_with(&resubmitted, &mut args, &mut |_, _| calls += 1)
        .unwrap();
    let stale_ticks = sim.ticks - t0;
    assert!(
        stale_ticks < first_ticks,
        "stale resubmit ({stale_ticks} ticks) must not wait for publication (first {first_ticks})"
    );
    for g in runs..2 * runs {
        assert!(
            !consumer_done(&args[3], r.layout, g),
            "consumer must not see a new done for run {g}"
        );
    }
    assert!(consumer_done(&args[3], r.layout, 0) && consumer_done(&args[3], r.layout, 1));
    assert_bytes(
        &args[3],
        &before[3],
        "ring changed by stale resubmit (same seqs, same done lines)",
    );
    assert_bytes(&args[0], &before[0], "args0");
    assert_bytes(&args[1], &before[1], "args1");
    check_outputs(&r, &gold, &args, runs);
    check_done_lines(r.layout, &args, runs, &[]);
    // Recovery: patched resubmit publishes and completes runs 2 and 3.
    let mut p = Producer::new(&r, &gold, runs, runs, 0, vec![], &args);
    run(&mut sim, &patched, &mut args, &mut p);
    p.finish(&args);
    check_outputs(&r, &gold, &args, 2 * runs);
    check_done_lines(r.layout, &args, 2 * runs, &[]);
    assert!(latest_missing(&args, r.layout, 2 * runs).is_empty());
    eprintln!("PASS ring stale resubmit: {stale_ticks} ticks (first {first_ticks}, {calls} wait ticks), stale diff {} words vs required, patched 0",stale_diff.len());
}

/// Overrun guard: run j+nslots may publish only after done[j%nslots]==seq(j) (global run index); forcing it anyway makes
/// the NPU's equality POLL for seq(j) time out.
#[test]
fn overrun_guard_and_forced_overwrite_times_out_poll() {
    let l = RingLayout { nslots: 2 };
    let mut ring = vec![0u8; l.bytes()];
    l.initialize(&mut ring);
    assert!(producer_may_publish(&ring, l, 0) && producer_may_publish(&ring, l, 1));
    assert!(!producer_may_publish(&ring, l, 2) && !producer_may_publish(&ring, l, 3));
    let done = |ring: &mut [u8], s: usize, seq: u32| {
        ring[l.done_line(s)..l.done_line(s) + LINE].copy_from_slice(&done_bytes(s, seq))
    };
    done(&mut ring, 0, 1);
    assert!(
        producer_may_publish(&ring, l, 2),
        "slot 0 done seq 1 -> run 2 may publish"
    );
    assert!(
        !producer_may_publish(&ring, l, 3)
            && !producer_may_publish(&ring, l, 4)
            && !producer_may_publish(&ring, l, 5)
    );
    done(&mut ring, 0, 3);
    assert!(
        !producer_may_publish(&ring, l, 2),
        "global run index: run 2 is behind done seq 3"
    );
    assert!(producer_may_publish(&ring, l, 4), "run 4 needs done[0]==3");
    done(&mut ring, 1, 2);
    assert!(producer_may_publish(&ring, l, 3) && !producer_may_publish(&ring, l, 5));

    // One-run submission on slot 0 of a 2-slot ring, no GEMM.
    let slots = [SlotPlan {
        slot: 0,
        a_off: 0,
        b_off: 0,
        c_off: 0,
    }];
    let d = empty_persistent(l, &slots, 1);
    let args = || {
        let mut v: Vec<Vec<u8>> = d.args.iter().map(|s| vec![GUARD; s.bytes]).collect();
        l.initialize(&mut v[3]);
        v
    };
    // Control: the correct seq 1 completes, done[0]==1, and the guard then admits run 2 but not run 3.
    let mut a = args();
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.submit_with(&d.insts, &mut a, &mut |t, a| {
        if t == 5 {
            wr32(&mut a[3], l.slot_line(0), 1)
        }
    })
    .unwrap();
    assert_eq!(done_at(&a[3], l, 0), done_bytes(0, 1));
    assert!(producer_may_publish(&a[3], l, 2) && !producer_may_publish(&a[3], l, 3));
    // Forced overwrite: guard says run 2 must NOT publish (done[0]==0), the producer writes seq(2)=3 into slot 0 anyway.
    let mut a = args();
    assert!(!producer_may_publish(&a[3], l, 2));
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.tick_limit = 5_000;
    let err = sim
        .submit_with(&d.insts, &mut a, &mut |t, a| {
            if t == 5 {
                wr32(&mut a[3], l.slot_line(0), 3)
            }
        })
        .unwrap_err();
    assert!(err.contains("MASKPOLL timed out"), "{err}");
    assert_eq!(
        done_at(&a[3], l, 0),
        [0; LINE],
        "no done may be written for a poll that never matched"
    );
    eprintln!("PASS ring overrun guard table + forced overwrite -> {err}");
}

// ---------------------------------------------------------------- lean ring: run 0 FULL every submission, later runs lean
/// Sorted patch words of a design (two per run: poll value + done value).
fn declared_words(d: &PersistentDesign, runs: usize) -> Vec<usize> {
    let mut w: Vec<usize> = d.patch_sites.iter().map(|&(w, _)| w).collect();
    w.sort();
    w.dedup();
    assert_eq!(
        w.len(),
        2 * runs,
        "exactly one poll and one done patch word per run"
    );
    for i in 0..runs {
        for kind in [PatchKind::PollSeq(i), PatchKind::DoneSeq(i)] {
            assert_eq!(
                d.patch_sites.iter().filter(|&&(_, k)| k == kind).count(),
                1,
                "{kind:?}"
            )
        }
    }
    w
}
/// `rounds` submissions of `s` runs each on ONE simulator + ring (round r = global runs `r*s..(r+1)*s`, seq0 = `r*s+1`), once
/// with the lean ring and once with the full-body ring under the identical producer schedule. Later rounds re-use the first
/// round's stream through `patch_seq`, which needs the slot of run i to repeat each round (`s % nslots == 0`). `rearm`:
/// rounds after the first submit the lean-first `persistent_v9_lean_rearm` stream instead.
fn lean_rounds(
    m: usize,
    n: usize,
    k: usize,
    gold: &[Golden],
    nslots: usize,
    s: usize,
    rounds: usize,
    gap: usize,
    rearm: bool,
) {
    assert!(
        rounds == 1 || s % nslots == 0,
        "patch_seq keeps slots: rounds>1 needs S % nslots == 0"
    );
    let what = format!("{m}x{n}x{k} S={s} nslots={nslots} rounds={rounds} gap={gap} rearm={rearm}");
    let (l0, f0) = (
        rig_shape(m, n, k, nslots, s, 0, 1, None, true),
        rig_shape(m, n, k, nslots, s, 0, 1, None, false),
    );
    assert_bytes(
        &l0.d.pdi,
        &f0.d.pdi,
        &format!("{what}: lean and full ring load the same PDI"),
    );
    assert!(
        l0.d.insts.len() < f0.d.insts.len(),
        "{what}: lean ring stream ({} B) is not smaller than the full-body ring ({} B)",
        l0.d.insts.len(),
        f0.d.insts.len()
    );
    let (dl, df) = (declared_words(&l0.d, s), declared_words(&f0.d, s));
    let (mut args_l, mut args_f) = (fresh_args(&l0), fresh_args(&f0));
    let (mut sim_l, mut sim_f) = (
        Config::from_pdi(&l0.d.pdi).unwrap(),
        Config::from_pdi(&f0.d.pdi).unwrap(),
    );
    let (mut lean_insts, mut full_insts) = (l0.d.insts.clone(), f0.d.insts.clone());
    for round in 0..rounds {
        let (base, seq0) = (round * s, 1 + (round * s) as u32);
        if round > 0 {
            let (fl, ff) = (
                rig_shape(m, n, k, nslots, s, base, seq0, None, true),
                rig_shape(m, n, k, nslots, s, base, seq0, None, false),
            );
            if rearm {
                let fr = persistent_with(
                    gemm_array::design_v9(m, n, k, gemm_array::V8_DEFAULT_EPILOGUE, Control::Fast),
                    l0.layout,
                    &fl.plans,
                    seq0,
                    Bodies::LeanFirst,
                    false,
                );
                assert_bytes(
                    &fr.pdi,
                    &l0.d.pdi,
                    &format!("{what}: rearm ring loads the same PDI"),
                );
                assert_eq!(declared_words(&fr, s).len(), dl.len());
                if round > 1 {
                    patch_seq(&mut lean_insts, &fr.patch_sites, seq0);
                    assert_bytes(
                        &lean_insts,
                        &fr.insts,
                        &format!("{what}: patched rearm stream != from-scratch rearm build"),
                    );
                }
                lean_insts = fr.insts;
            } else {
                assert_eq!(
                    diff_words(&lean_insts, &fl.d.insts),
                    dl,
                    "{what}: lean byte diff between rounds must be exactly the declared words"
                );
                patch_seq(&mut lean_insts, &l0.d.patch_sites, seq0);
                assert_bytes(
                    &lean_insts,
                    &fl.d.insts,
                    &format!("{what}: patched lean stream != from-scratch lean build"),
                );
            }
            patch_seq(&mut full_insts, &f0.d.patch_sites, seq0);
            assert_bytes(
                &full_insts,
                &ff.d.insts,
                &format!("{what}: patched full stream != from-scratch full build"),
            );
            assert_eq!(df, declared_words(&ff.d, s));
        }
        let total = base + s;
        let (lt, ft) = (sim_l.ticks, sim_f.ticks);
        let mut p = Producer::new(&l0, gold, base, s, gap, vec![], &args_l);
        run(&mut sim_l, &lean_insts, &mut args_l, &mut p);
        p.finish(&args_l);
        let mut pf = Producer::new(&f0, gold, base, s, gap, vec![], &args_f);
        run(&mut sim_f, &full_insts, &mut args_f, &mut pf);
        pf.finish(&args_f);
        for (r, args, p) in [(&l0, &args_l, &p), (&f0, &args_f, &pf)] {
            check_outputs(r, gold, args, total);
            check_done_lines(r.layout, args, total, &[]);
            assert!(
                latest_missing(args, r.layout, total).is_empty(),
                "{what}: round {round}: a consumer-visible DONE is missing"
            );
            assert_eq!(
                p.reuse_checks,
                (base..total).filter(|&g| g >= nslots).count(),
                "{what}: round {round}: every reused slot verified its predecessor"
            );
            assert!(
                p.calls >= (s - 1) * gap,
                "{what}: round {round}: the producer was not genuinely late ({} wait ticks)",
                p.calls
            );
        }
        for a in 0..4 {
            assert_bytes(
                &args_l[a],
                &args_f[a],
                &format!("{what}: round {round}: lean-ring arg{a} != full-ring arg{a}"),
            );
        }
        assert!(
            sim_l.state_snapshot() == sim_f.state_snapshot(),
            "{what}: round {round}: retained state of the lean ring != full-body ring"
        );
        eprintln!("PASS lean ring {what} round {round}: all runs CPU+eager exact, DONE lines exact, state == full ring; lean {} ticks, full {} ticks, {} B vs {} B",sim_l.ticks-lt,sim_f.ticks-ft,l0.d.insts.len(),f0.d.insts.len());
    }
}
/// `(512,512,64)`: kc*waves, waves and NW all odd (nonperiodic parity edge); `(512,512,128)`: W and NW odd, J even;
/// `(1024,512,128)`: NW odd, W and J even.
fn lean_ring_matrix(m: usize, n: usize, k: usize) {
    let probe = rig_shape(m, n, k, 2, 3, 0, 1, None, true);
    let gold = goldens(&probe, 8);
    for gap in [0, 20_000] {
        lean_rounds(m, n, k, &gold, 2, 3, 1, gap, false); // S=3 > nslots=2: slot wrap inside one submission
        lean_rounds(m, n, k, &gold, 2, 4, 2, gap, false); // two retained rounds, second via patch_seq, wrap in each
        lean_rounds(m, n, k, &gold, 2, 2, 3, gap, true); // lean-first re-arm: round 1 from scratch, round 2 via patch_seq
    }
}
#[test]
fn lean_ring_512x512x128() {
    lean_ring_matrix(512, 512, 128)
}
#[test]
fn lean_ring_1024x512x128() {
    lean_ring_matrix(1024, 512, 128)
}
#[test]
fn lean_ring_512x512x64_all_odd_parity() {
    lean_ring_matrix(512, 512, 64)
}

/// Ring of an explicit `design` (`persistent_with`): round 0 lean (full run 0), later rounds lean-first, against the
/// full-body ring of the same design under the same schedule. Every run is CPU + eager exact, and the lean ring's
/// arguments and retained array state equal the full-body ring's after every round. Returns the goldens.
fn design_rounds(
    m: usize,
    n: usize,
    k: usize,
    what: &str,
    design: &dyn Fn() -> ArrayDesign,
) -> Vec<Golden> {
    let (nslots, s, rounds) = (2, 2, 3);
    let layout = RingLayout { nslots };
    let rig_of = |base: usize, seq0: u32, bodies: Bodies| -> Rig {
        let eager = design();
        let (a, b, c) = (
            eager.args[0].bytes,
            eager.args[1].bytes,
            eager.args[2].bytes,
        );
        let plans = plans(base, s, nslots, a, b, c);
        let d = persistent_with(design(), layout, &plans, seq0, bodies, false);
        Rig {
            m,
            n,
            k,
            layout,
            eager,
            d,
            plans,
            cb: c,
            used: nslots,
        }
    };
    let (l0, f0) = (rig_of(0, 1, Bodies::Lean), rig_of(0, 1, Bodies::Full));
    let gold = goldens(&l0, s * rounds);
    let (mut args_l, mut args_f) = (fresh_args(&l0), fresh_args(&f0));
    let (mut sim_l, mut sim_f) = (
        Config::from_pdi(&l0.d.pdi).unwrap(),
        Config::from_pdi(&f0.d.pdi).unwrap(),
    );
    for round in 0..rounds {
        let (base, seq0) = (round * s, 1 + (round * s) as u32);
        let (rl, rf) = (
            rig_of(
                base,
                seq0,
                if round == 0 {
                    Bodies::Lean
                } else {
                    Bodies::LeanFirst
                },
            ),
            rig_of(base, seq0, Bodies::Full),
        );
        let mut p = Producer::new(&rl, &gold, base, s, 0, vec![], &args_l);
        run(&mut sim_l, &rl.d.insts, &mut args_l, &mut p);
        p.finish(&args_l);
        let mut pf = Producer::new(&rf, &gold, base, s, 0, vec![], &args_f);
        run(&mut sim_f, &rf.d.insts, &mut args_f, &mut pf);
        pf.finish(&args_f);
        for (r, args) in [(&rl, &args_l), (&rf, &args_f)] {
            check_outputs(r, &gold, args, base + s);
            check_done_lines(r.layout, args, base + s, &[]);
        }
        for a in 0..4 {
            assert_bytes(
                &args_l[a],
                &args_f[a],
                &format!("{what} {m}x{n}x{k} round {round}: lean arg{a} != full arg{a}"),
            );
        }
        assert!(
            sim_l.state_snapshot() == sim_f.state_snapshot(),
            "{what} {m}x{n}x{k} round {round}: lean ring state != full ring"
        );
        eprintln!("PASS {what} ring {m}x{n}x{k} round {round}: exact, state == full ring");
    }
    gold
}

/// `ArrayDesign::with_a_repeat` (host A = each `mw` tile once, the shim task replays it `NW` times): the ring rounds of
/// [`design_rounds`], and every eager C equals the default (replicated-A) V9's.
fn a_repeat_rounds(m: usize, n: usize, k: usize) {
    let v9 = |arep: bool| {
        let d = gemm_array::design_v9(m, n, k, INT8, Control::Fast);
        if arep {
            d.with_a_repeat()
        } else {
            d
        }
    };
    let plain = v9(false);
    assert_eq!(v9(true).pdi, plain.pdi, "A repeat keeps the PDI");
    assert_eq!(
        v9(true).args[0].bytes * n.div_ceil(512),
        plain.args[0].bytes,
        "A repeat holds 1/NW of the replicated A"
    );
    let gold = design_rounds(m, n, k, "A-repeat", &|| v9(true));
    for (g, i) in gold.iter().zip(0..) {
        let (a, b) = matrices(m, n, k, 0xc001d00d + i as u32 * 7919);
        let [ap, bp] = plain.pack_in(&a, &b);
        let mut args = vec![ap, bp, vec![POISON; plain.args[2].bytes]];
        Config::from_pdi(&plain.pdi)
            .unwrap()
            .submit(&plain.insts, &mut args)
            .unwrap();
        assert_bytes(
            &g.eager_c,
            &args[2],
            "A-repeat eager C != default V9 eager C",
        );
    }
}
#[test]
fn a_repeat_ring_512x1280x128() {
    a_repeat_rounds(512, 1280, 128)
}
#[test]
fn a_repeat_ring_1024x1280x64_two_m_waves() {
    a_repeat_rounds(1024, 1280, 64)
}
#[test]
fn g80_ring_512x1280x128() {
    design_rounds(512, 1280, 128, "G80", &|| {
        peacemaker_npu::kernels::gemm_g80::design_g80(512, 1280, 128, INT8, Control::Fast)
    });
}

/// G80 B-resident ring (`persistent_with(.., b_shared = true)`): every run shares one B; run 0 of round 0 fills it from
/// DDR, every later run keeps it in the memtile. From run 2 on the host B is poison, so an exact C proves the lean
/// runs read the resident B and nothing from DDR.
#[test]
fn g80_b_resident_ring_512x1280x128() {
    let (m, n, k, nslots, s, rounds) = (512, 1280, 128, 2, 2, 3);
    let layout = RingLayout { nslots };
    let design = || peacemaker_npu::kernels::gemm_g80::design_g80(m, n, k, INT8, Control::Fast);
    let eager = design();
    let (ab, bb, cb) = (
        eager.args[0].bytes,
        eager.args[1].bytes,
        eager.args[2].bytes,
    );
    let plans_of = |base: usize| -> Vec<SlotPlan> {
        (0..s)
            .map(|i| {
                let slot = (base + i) % nslots;
                SlotPlan {
                    slot,
                    a_off: (slot * ab) as u64,
                    b_off: 0,
                    c_off: (slot * cb) as u64,
                }
            })
            .collect()
    };
    let rig_of = |base: usize, seq0: u32, bodies: Bodies| -> Rig {
        let plans = plans_of(base);
        Rig {
            m,
            n,
            k,
            layout,
            eager: design(),
            d: persistent_with(design(), layout, &plans, seq0, bodies, true),
            plans,
            cb,
            used: nslots,
        }
    };
    let (_, b) = matrices(m, n, k, 0xb0b);
    let r0 = rig_of(0, 1, Bodies::Lean);
    let gold: Vec<Golden> = (0..s * rounds)
        .map(|g| {
            let (a, _) = matrices(m, n, k, 0xc001d00d + g as u32 * 7919);
            let [ap, bp] = eager.pack_in(&a, &b);
            let mut args = vec![ap.clone(), bp.clone(), vec![POISON; cb]];
            Config::from_pdi(&eager.pdi)
                .unwrap()
                .submit(&eager.insts, &mut args)
                .unwrap();
            let reference = eager.reference(&a, &b);
            assert_eq!(eager.unpack_out(&args[2]), reference);
            // Only run 0 reads B from DDR. Run 1's operands are written while run 0 runs (same bytes); from run 2 on (slot of
            // run 0 reused, so run 0 is done) the producer writes poison there.
            Golden {
                a: ap,
                b: if g < 2 { bp } else { vec![0x5a; bb] },
                eager_c: args.swap_remove(2),
                reference,
            }
        })
        .collect();
    let mut args = fresh_args(&r0);
    let mut sim = Config::from_pdi(&r0.d.pdi).unwrap();
    for round in 0..rounds {
        let (base, seq0) = (round * s, 1 + (round * s) as u32);
        let r = if round == 0 {
            rig_of(0, 1, Bodies::Lean)
        } else {
            rig_of(base, seq0, Bodies::LeanFirst)
        };
        let mut p = Producer::new(&r, &gold, base, s, 0, vec![], &args);
        run(&mut sim, &r.d.insts, &mut args, &mut p);
        p.finish(&args);
        check_outputs(&r, &gold, &args, base + s);
        check_done_lines(r.layout, &args, base + s, &[]);
        if round > 0 {
            assert!(args[1].iter().all(|&x| x == 0x5a), "host B is poison");
        }
        eprintln!("PASS G80 B-resident ring round {round}: exact with poisoned host B");
    }
}

/// What a run of a (possibly broken) lean ring did wrong, judged against the CPU/eager goldens and the full-body ring state.
#[derive(Debug)]
#[allow(dead_code)]
enum Caught {
    Failed(String),
    WrongC(usize),
    BadDone(usize),
    StateMismatch,
}
fn caught(
    r: &Rig,
    gold: &[Golden],
    s: usize,
    res: Result<(), String>,
    p: &Producer,
    args: &[Vec<u8>],
    state_equal: bool,
) -> Option<Caught> {
    let l = r.layout;
    if let Err(e) = res {
        return Some(Caught::Failed(e));
    }
    if let Some(&g) = p.bad_reuse.first() {
        return Some(Caught::WrongC(g));
    }
    for slot in 0..l.nslots {
        let g = (0..s).rev().find(|g| g % l.nslots == slot).unwrap();
        if args[2][slot * r.cb..(slot + 1) * r.cb] != gold[g].eager_c[..] {
            return Some(Caught::WrongC(g));
        }
        if done_at(&args[3], l, slot) != done_bytes(slot, g as u32 + 1) {
            return Some(Caught::BadDone(g));
        }
    }
    if !state_equal {
        return Some(Caught::StateMismatch);
    }
    None
}
/// `Neg::SkipLeanRequeue(1)` drops the forced repairs of lean run 1. The shapes have even NW, W and J, so the lean body alone
/// repairs nothing the next run needs from the run before and only the ring's forced repairs keep run 1 correct: the same
/// producer schedule that the intact lean ring passes (control) must be caught on the omitted stream.
#[test]
fn lean_ring_skip_requeue_is_detected() {
    for (m, n, k) in [(512, 1024, 128), (512, 1024, 256)] {
        let (nslots, s) = (2, 3);
        let what = format!("{m}x{n}x{k} S={s} nslots={nslots}");
        let normal = rig_shape(m, n, k, nslots, s, 0, 1, None, true);
        let neg = rig_shape(
            m,
            n,
            k,
            nslots,
            s,
            0,
            1,
            Some(Neg::SkipLeanRequeue(1)),
            true,
        );
        let full = rig_shape(m, n, k, nslots, s, 0, 1, None, false);
        assert_bytes(&neg.d.pdi, &normal.d.pdi, "same PDI");
        assert!(
            neg.d.insts.len() < normal.d.insts.len(),
            "{what}: SkipLeanRequeue(1) must drop repair ops ({} vs {} B)",
            neg.d.insts.len(),
            normal.d.insts.len()
        );
        declared_words(&neg.d, s);
        let gold = goldens(&normal, s);
        // Full-body ring reference state.
        let mut args = fresh_args(&full);
        let mut sim_f = Config::from_pdi(&full.d.pdi).unwrap();
        let mut p = Producer::new(&full, &gold, 0, s, 0, vec![], &args);
        run(&mut sim_f, &full.d.insts, &mut args, &mut p);
        p.finish(&args);
        let want = sim_f.state_snapshot();
        // One attempt = lenient producer, tick limit tightened to a multiple of one whole intact run.
        let attempt = |r: &Rig, insts: &[u8], limit: usize| -> (Option<Caught>, usize) {
            let mut args = fresh_args(r);
            let mut sim = Config::from_pdi(&r.d.pdi).unwrap();
            sim.tick_limit = limit;
            let mut p = Producer::new(r, &gold, 0, s, 0, vec![], &args).lenient();
            let res = sim.submit_with(insts, &mut args, &mut |t, a| p.call(t, a));
            let state_equal = res.is_ok() && sim.state_snapshot() == want;
            (caught(r, &gold, s, res, &p, &args, state_equal), sim.ticks)
        };
        let (control, ticks) = attempt(&normal, &normal.d.insts, 10_000_000);
        assert!(
            control.is_none(),
            "{what}: intact lean ring misbehaved: {control:?}"
        );
        let (got, _) = attempt(&neg, &neg.d.insts, ticks * 2 + 4096);
        assert!(got.is_some(),"{what}: lean ring without its forced repairs behaved like the intact lean ring: the repairs are not load-bearing");
        eprintln!(
            "NEGATIVE lean ring {what}: SkipLeanRequeue(1) -> {:?} ({} B vs {} B)",
            got.unwrap(),
            neg.d.insts.len(),
            normal.d.insts.len()
        );
    }
}
