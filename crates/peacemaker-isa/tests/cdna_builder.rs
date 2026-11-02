// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Acceptance of the checked CDNA wave64 builder (`peacemaker_isa::cdna::builder`).
//!
//! Unit tests pin the wait ledger as the builder applies it (vmcnt younger
//! counts including mixed loads and stores, combined vmcnt+lgkmcnt waits and
//! their `WaitProof`s, SMEM, DS and cross-lane lgkmcnt, barrier drains, forward
//! joins, loop invariants, unreachable code), branch resolution (offsets
//! relative to the next dword across multi-dword instructions, i16 range) and
//! the register/kernarg/workgroup-ID/work-item allocators. The end-to-end tests
//! build a vector add with the builder and `cdna::ops`, link it with
//! `emit::code_object`, load it with `sim::load_code_object` and run it on the
//! CPU emulator against a bit-exact CPU reference, then remove the waits from
//! the finished instructions, re-encode the hand-assembled variants and require
//! the emulator to reject (`SimError::Wait`) or miscompute them.
//!
//! No GPU is used. Hardware-dependent rules are named in each test's doc.
use peacemaker_cdna::enc::{
    ds::DsOp,
    flat::{Flat, FlatOp, FlatSeg},
    mubuf::MubufOp,
    smem::SmemOp,
    sop::{waitcnt_fields, waitcnt_imm, Sop1Op, Sop2Op, SopcOp, Sopp, SoppOp},
    vop::{Vop1Op, Vop2Op, VopcOp},
};
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::sim::{self, Config, Launch, Memory, SimError};
use peacemaker_cdna::{decode_all, encode_all, Arch as CdnaArch, Inst, Src};
use peacemaker_isa::cdna::builder::{Builder, Lds};
use peacemaker_isa::cdna::emit::{code_object, CodeObjectVersion};
use peacemaker_isa::cdna::{hazard, ops, Item, KernArgKind, Kernel};
use peacemaker_isa::ledger::{Counter, Reason};
use peacemaker_isa::Arch;

const CDNA: CdnaArch = CdnaArch::Gfx942;
/// `s_waitcnt` field values that mean "do not wait on this counter" (gfx942
/// field maxima, `Caps::{vmcnt_max, expcnt_max, lgkmcnt_max}`).
const NO_VM: u8 = 63;
const NO_EXP: u8 = 7;
const NO_LGKM: u8 = 15;

// ----------------------------------------------------------------------------
// Helpers

fn sg(base: Src, k: u8) -> Src {
    match base {
        Src::Sgpr(n) => Src::Sgpr(n + k),
        other => panic!("not an SGPR: {other:?}"),
    }
}
fn vg(base: Src, k: u8) -> Src {
    match base {
        Src::Vgpr(n) => Src::Vgpr(n + k),
        other => panic!("not a VGPR: {other:?}"),
    }
}
fn one() -> Src {
    Src::Float(InlineFloat::One)
}
fn nop() -> Inst {
    ops::sp(SoppOp::Nop, 0)
}
fn new_builder(name: &str) -> Builder {
    Builder::new(Arch::Gfx942, name, 64).expect("builder")
}

fn waitcnt_of(inst: &Inst) -> Option<(u8, u8, u8)> {
    match inst {
        Inst::Sopp(p) if p.op == SoppOp::Waitcnt => Some(waitcnt_fields(CDNA, p.simm16)),
        _ => None,
    }
}
/// `(vmcnt, lgkmcnt)` of every `s_waitcnt` among `items`, in order.
fn waits(items: &[Item]) -> Vec<(u8, u8)> {
    items
        .iter()
        .filter_map(|i| match i {
            Item::Inst(x) => waitcnt_of(x).map(|(vm, _, lgkm)| (vm, lgkm)),
            _ => None,
        })
        .collect()
}
fn is_wait_item(item: &Item) -> bool {
    matches!(item, Item::Inst(i) if waitcnt_of(i).is_some())
}
fn is_barrier_item(item: &Item) -> bool {
    matches!(item, Item::Inst(Inst::Sopp(p)) if p.op == SoppOp::Barrier)
}
fn is_endpgm(inst: &Inst) -> bool {
    matches!(inst, Inst::Sopp(p) if p.op == SoppOp::Endpgm)
}

/// A builder with a scalar pointer pair, 32 SGPRs, 48 VGPRs and LDS, plus
/// shorthand for the instructions the ledger tests need. `v(0)` is the VMEM
/// address / DS address register, `v(1)` the store / DS data register.
struct Rig {
    b: Builder,
    ptr: Src,
    sbase: Src,
    vbase: Src,
}

fn rig() -> Rig {
    let mut b = new_builder("t");
    let ptr = b.sgprs(2, 2).unwrap();
    let sbase = b.sgprs(32, 4).unwrap();
    let vbase = b.vgprs(48, 4).unwrap();
    b.lds(1024, 16).unwrap();
    Rig {
        b,
        ptr,
        sbase,
        vbase,
    }
}

impl Rig {
    fn v(&self, k: u8) -> Src {
        vg(self.vbase, k)
    }
    fn s(&self, k: u8) -> Src {
        sg(self.sbase, k)
    }
    fn push(&mut self, i: Inst) {
        self.b.push(i).unwrap_or_else(|e| panic!("push: {e}"))
    }
    fn items(&self) -> &[Item] {
        self.b.items()
    }
    fn waits(&self) -> Vec<(u8, u8)> {
        waits(self.b.items())
    }
    fn label(&mut self, name: &str) {
        self.b
            .label(name)
            .unwrap_or_else(|e| panic!("label {name}: {e}"))
    }
    /// `global_load_dword v(dst), v(0), s[ptr]`.
    fn ld(&mut self, dst: u8) {
        let i = ops::global_load(FlatOp::LoadDword, self.v(dst), self.v(0), Some(self.ptr), 0);
        self.push(i)
    }
    /// `global_store_dword v(0), v(data), s[ptr]`.
    fn st(&mut self, data: u8) {
        let i = ops::global_store(
            FlatOp::StoreDword,
            self.v(0),
            self.v(data),
            Some(self.ptr),
            0,
        );
        self.push(i)
    }
    /// `v_add_f32 v(dst), 1.0, v(src)`: reads `v(src)`.
    fn consume(&mut self, dst: u8, src: u8) {
        let i = ops::v2(Vop2Op::AddF32, self.v(dst), one(), self.v(src));
        self.push(i)
    }
    /// `s_load_dword s(k), s[0:1], 4*k`.
    fn s_ld(&mut self, k: u8) {
        let i = ops::s_load(SmemOp::LoadDword, self.s(k), Src::Sgpr(0), 4 * i32::from(k));
        self.push(i)
    }
    /// `s_load_dwordx2 s[k:k+1], s[0:1], 4*k` (`k` even).
    fn s_ld2(&mut self, k: u8) {
        let i = ops::s_load(
            SmemOp::LoadDwordx2,
            self.s(k),
            Src::Sgpr(0),
            4 * i32::from(k),
        );
        self.push(i)
    }
    /// `s_add_u32 s(dst), s(src), 1`: reads `s(src)`.
    fn s_use(&mut self, dst: u8, src: u8) {
        let i = ops::s2(Sop2Op::AddU32, self.s(dst), self.s(src), Src::Int(1));
        self.push(i)
    }
    fn ds_ld(&mut self, dst: u8) {
        let i = ops::ds_read(DsOp::ReadB32, self.v(dst), self.v(0), 0);
        self.push(i)
    }
    fn ds_st(&mut self, data: u8) {
        let i = ops::ds_write(DsOp::WriteB32, self.v(0), self.v(data), 0);
        self.push(i)
    }
    /// `ds_bpermute_b32 v(dst), v(0), v(1)`.
    fn bperm(&mut self, dst: u8) {
        let i = ops::ds_xlane(DsOp::BpermuteB32, self.v(dst), self.v(0), self.v(1), 0);
        self.push(i)
    }
    /// `s_cmp_eq_u32 s(0), 0` then `s_cbranch_scc1 target`.
    fn cmp_branch(&mut self, target: &str) {
        let c = ops::sc(SopcOp::CmpEqU32, self.s(0), Src::Int(0));
        self.push(c);
        self.b
            .branch(SoppOp::CbranchScc1, target)
            .unwrap_or_else(|e| panic!("branch {target}: {e}"));
    }
    fn finish(self) -> Kernel {
        self.b.finish().unwrap_or_else(|e| panic!("finish: {e}"))
    }
}

fn label_dword(k: &Kernel, name: &str) -> usize {
    k.labels
        .iter()
        .find(|(l, _)| l.as_str() == name)
        .unwrap_or_else(|| panic!("label {name} missing"))
        .1
}
fn dword_positions(k: &Kernel) -> Vec<usize> {
    let mut at = 0;
    k.insts
        .iter()
        .map(|i| {
            let p = at;
            at += i.words(CDNA).unwrap().len();
            p
        })
        .collect()
}
fn count_waitcnts(k: &Kernel) -> usize {
    k.insts.iter().filter(|i| waitcnt_of(i).is_some()).count()
}

// ----------------------------------------------------------------------------
// Wait ledger: VMEM

/// A load's first use waits for it with `vmcnt = number of younger VMEM
/// operations`; the wait retires everything older too (CDNA3 ISA §4.4 p.19).
#[test]
fn vm_load_use_waits_for_the_younger_count() {
    let mut r = rig();
    r.ld(2);
    r.ld(3);
    r.ld(4);
    assert!(r.waits().is_empty(), "issuing loads needs no wait");
    r.consume(20, 3); // B: C is younger
    assert_eq!(r.waits(), vec![(1, NO_LGKM)]);
    let n = r.items().len();
    assert!(
        is_wait_item(&r.items()[n - 2]),
        "the wait sits immediately before the using instruction"
    );
    r.consume(21, 2); // A was retired by vmcnt(1)
    assert_eq!(r.waits().len(), 1);
    r.consume(22, 4); // C
    assert_eq!(r.waits(), vec![(1, NO_LGKM), (0, NO_LGKM)]);
    r.consume(23, 4); // everything retired
    assert_eq!(r.waits().len(), 2);
}

#[test]
fn vm_oldest_load_needs_all_younger_loads_outstanding_count() {
    let mut r = rig();
    r.ld(2);
    r.ld(3);
    r.ld(4);
    r.ld(5);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(3, NO_LGKM)]);
}

#[test]
fn vm_youngest_load_waits_for_zero() {
    let mut r = rig();
    r.ld(2);
    r.ld(3);
    r.consume(20, 3);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
}

/// VM_CNT retires loads and stores in issue order including mixing, so younger
/// stores count toward a load's `vmcnt` (CDNA3 ISA §4.4 p.19).
#[test]
fn vm_mixed_loads_and_stores_count_in_issue_order() {
    let mut r = rig();
    r.ld(2);
    r.st(1);
    r.ld(3);
    r.ld(4);
    r.consume(20, 2); // younger: store, B, C
    assert_eq!(r.waits(), vec![(3, NO_LGKM)]);
    r.consume(21, 4);
    assert_eq!(r.waits(), vec![(3, NO_LGKM), (0, NO_LGKM)]);

    let mut r = rig();
    r.st(1);
    r.ld(2);
    r.st(1);
    r.consume(20, 2); // older store does not count; younger store does
    assert_eq!(r.waits(), vec![(1, NO_LGKM)]);

    let mut r = rig();
    r.st(1);
    r.st(1);
    r.ld(2);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
}

// ----------------------------------------------------------------------------
// Wait ledger: combined waits and proofs

/// One instruction needing a VM load and an SMEM load gets ONE `s_waitcnt`
/// carrying both counters, and the kernel records one `WaitProof` per counter.
#[test]
fn combined_vm_and_lgkm_wait_is_one_instruction_with_one_proof_per_counter() {
    let mut r = rig();
    r.s_ld(0);
    r.s_ld(1);
    r.ld(2);
    r.ld(3);
    // v_add_f32 v20, s0, v2: SMEM (out of order -> lgkmcnt(0)) and a VM load
    // with one younger load (vmcnt(1)).
    let i = ops::v2(Vop2Op::AddF32, r.v(20), r.s(0), r.v(2));
    r.push(i);
    assert_eq!(
        r.waits(),
        vec![(1, 0)],
        "a single s_waitcnt vmcnt(1) lgkmcnt(0)"
    );
    let k = r.finish();

    assert_eq!(count_waitcnts(&k), 1);
    assert_eq!(k.waits.len(), 2, "one proof per counter");
    let vm = k
        .waits
        .iter()
        .find(|p| p.counter == Counter::Vm)
        .expect("vm proof");
    let lg = k
        .waits
        .iter()
        .find(|p| p.counter == Counter::Lgkm)
        .expect("lgkm proof");
    assert_eq!(
        k.waits.iter().filter(|p| p.counter == Counter::Vm).count(),
        1
    );
    assert_eq!(
        k.waits
            .iter()
            .filter(|p| p.counter == Counter::Lgkm)
            .count(),
        1
    );
    assert_eq!(vm.count, 1);
    assert_eq!(lg.count, 0);
    assert!(matches!(vm.reason, Reason::Use { .. }), "{:?}", vm.reason);
    assert!(matches!(lg.reason, Reason::Use { .. }), "{:?}", lg.reason);
    assert_eq!(
        vm.pc_index, lg.pc_index,
        "both proofs name the same s_waitcnt"
    );
    assert!(
        waitcnt_of(&k.insts[vm.pc_index]).is_some(),
        "pc_index points at the s_waitcnt in Kernel::insts"
    );
}

/// A wait needing one counter leaves the other field at its maximum and
/// produces exactly one proof.
#[test]
fn single_counter_waits_leave_the_other_field_unconstrained() {
    let mut r = rig();
    r.ld(2);
    r.consume(20, 2);
    r.s_ld(0);
    r.s_use(8, 0);
    assert_eq!(r.waits(), vec![(0, NO_LGKM), (NO_VM, 0)]);
    let k = r.finish();
    assert_eq!(k.waits.len(), 2);
    assert_eq!(k.waits[0].counter, Counter::Vm);
    assert_eq!(k.waits[1].counter, Counter::Lgkm);
    for p in &k.waits {
        let (vm, exp, lgkm) = waitcnt_of(&k.insts[p.pc_index]).expect("s_waitcnt");
        assert_eq!(exp, NO_EXP, "expcnt is never waited on here");
        assert!(vm == NO_VM || lgkm == NO_LGKM);
    }
}

// ----------------------------------------------------------------------------
// Wait ledger: SMEM, DS, cross-lane

/// SMEM returns out of order: any use needs `lgkmcnt(0)`, even of the oldest
/// of several outstanding loads (CDNA3 ISA §4.4 p.20).
#[test]
fn smem_use_waits_lgkmcnt_zero() {
    let mut r = rig();
    r.s_ld2(0);
    r.s_ld(2);
    r.s_ld(3);
    r.s_use(8, 0);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
    r.s_use(9, 3); // already drained
    assert_eq!(r.waits().len(), 1);

    let mut r = rig();
    r.s_ld(4);
    r.s_use(8, 4);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);

    // A loaded SGPR pair used as the base of the next scalar load.
    let mut r = rig();
    r.s_ld2(0);
    let i = ops::s_load(SmemOp::LoadDword, r.s(4), r.s(0), 0);
    r.push(i);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
}

/// DS operations retire in order: a DS load's use waits for the younger DS
/// operations, loads and stores alike.
#[test]
fn ds_load_counts_younger_ds_operations() {
    let mut r = rig();
    r.ds_ld(2);
    r.ds_ld(3);
    r.ds_ld(4);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(NO_VM, 2)]);
    r.consume(21, 3);
    assert_eq!(r.waits(), vec![(NO_VM, 2), (NO_VM, 1)]);
    r.consume(22, 4);
    assert_eq!(r.waits(), vec![(NO_VM, 2), (NO_VM, 1), (NO_VM, 0)]);

    let mut r = rig();
    r.ds_ld(2);
    r.ds_st(1);
    r.consume(20, 2); // the younger DS store counts
    assert_eq!(r.waits(), vec![(NO_VM, 1)]);

    let mut r = rig();
    r.ds_st(1);
    r.ds_ld(2);
    r.consume(20, 2); // the older DS store does not
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
}

/// `ds_bpermute_b32` returns data under LGKM_CNT in DS order like a DS load.
#[test]
fn ds_cross_lane_returns_under_lgkm_in_ds_order() {
    let mut r = rig();
    r.ds_ld(2);
    r.bperm(3);
    r.consume(20, 2); // the younger bpermute counts
    assert_eq!(r.waits(), vec![(NO_VM, 1)]);
    r.consume(21, 3);
    assert_eq!(r.waits(), vec![(NO_VM, 1), (NO_VM, 0)]);

    let mut r = rig();
    r.bperm(2);
    r.ds_ld(3);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(NO_VM, 1)]);

    let mut r = rig();
    r.bperm(2);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
}

/// SMEM may retire out of order relative to everything on LGKM_CNT, so a DS
/// load older than an SMEM load cannot be waited for with a positive count.
#[test]
fn ds_load_with_a_younger_smem_load_waits_for_zero() {
    let mut r = rig();
    r.ds_ld(2);
    r.s_ld(0);
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
}

/// The VM and LGKM counters are independent: a DS wait ignores VMEM operations.
#[test]
fn vm_and_lgkm_counts_do_not_mix() {
    let mut r = rig();
    r.ld(2);
    r.ds_ld(3);
    r.ld(4);
    r.ds_ld(5);
    r.consume(20, 3); // DS: one younger DS load; VMEM operations do not count
    assert_eq!(r.waits(), vec![(NO_VM, 1)]);
    r.consume(21, 2); // VM: one younger load; DS loads do not count
    assert_eq!(r.waits(), vec![(NO_VM, 1), (1, NO_LGKM)]);
}

// ----------------------------------------------------------------------------
// Barrier and wait_all

/// `s_barrier` does not wait for memory on gfx942 (back-off barrier), so the
/// builder drains LGKM_CNT first when a DS load is pending (the LDS WAR rule).
#[test]
fn barrier_drains_pending_ds_loads() {
    let mut r = rig();
    r.ds_ld(2);
    r.b.barrier().unwrap();
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
    let n = r.items().len();
    assert!(
        is_wait_item(&r.items()[n - 2]) && is_barrier_item(&r.items()[n - 1]),
        "s_waitcnt then s_barrier"
    );
    r.consume(20, 2); // the drained load needs nothing more
    assert_eq!(r.waits().len(), 1);
}

/// ... and when a DS store is pending (the LDS RAW rule: the barrier publishes
/// the store to the other waves).
#[test]
fn barrier_drains_pending_ds_stores() {
    let mut r = rig();
    r.ds_st(1);
    r.b.barrier().unwrap();
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);
    let n = r.items().len();
    assert!(is_wait_item(&r.items()[n - 2]) && is_barrier_item(&r.items()[n - 1]));
}

#[test]
fn barrier_drains_mixed_lgkm_with_one_wait_and_leaves_vmcnt() {
    let mut r = rig();
    r.ld(2);
    r.ds_ld(3);
    r.ds_st(1);
    r.s_ld(0);
    r.bperm(4);
    r.b.barrier().unwrap();
    assert_eq!(
        r.waits(),
        vec![(NO_VM, 0)],
        "one s_waitcnt lgkmcnt(0), vmcnt untouched"
    );
    r.consume(20, 2); // the VMEM load is still pending
    assert_eq!(r.waits(), vec![(NO_VM, 0), (0, NO_LGKM)]);
}

#[test]
fn barrier_with_nothing_pending_or_only_vmem_pending_inserts_no_wait() {
    let mut r = rig();
    r.b.barrier().unwrap();
    assert!(r.waits().is_empty());
    assert!(is_barrier_item(r.items().last().unwrap()));

    let mut r = rig();
    r.ld(2);
    r.b.barrier().unwrap();
    assert!(
        r.waits().is_empty(),
        "a VMEM load is not drained by a barrier"
    );
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
}

#[test]
fn wait_all_drains_every_counter_once() {
    let mut r = rig();
    r.ld(2);
    r.s_ld(0);
    r.ds_ld(3);
    let before = r.items().len();
    r.b.wait_all().unwrap();
    assert!(r.items().len() > before, "pending work needs a wait");
    let drain = r.items().len();
    let (mut vm, mut lgkm) = (NO_VM, NO_LGKM);
    for (v, l) in r.waits() {
        vm = vm.min(v);
        lgkm = lgkm.min(l);
    }
    assert_eq!(
        (vm, lgkm),
        (0, 0),
        "vmcnt(0) and lgkmcnt(0) are both reached"
    );
    let waits_so_far = r.waits().len();
    r.consume(20, 2);
    r.consume(21, 3);
    r.s_use(8, 0);
    assert_eq!(
        r.waits().len(),
        waits_so_far,
        "nothing is pending after wait_all"
    );
    assert_eq!(
        r.items().len(),
        drain + 3,
        "only the three consumers were appended"
    );

    // Nothing pending: wait_all emits nothing.
    let mut r = rig();
    r.consume(20, 2);
    let before = r.items().len();
    r.b.wait_all().unwrap();
    assert_eq!(r.items().len(), before);
}

/// A required count at or past the field maximum is clamped to `max - 1`
/// (vm 62, lgkm 14): the all-ones field value means "no wait".
#[test]
fn thresholds_past_the_field_maximum_saturate_below_the_all_ones_value() {
    let mut r = rig();
    r.ld(2);
    for _ in 0..61 {
        r.st(1);
    }
    r.consume(20, 2); // 61 younger: exact
    assert_eq!(r.waits(), vec![(61, NO_LGKM)]);

    let mut r = rig();
    r.ld(2);
    for _ in 0..70 {
        r.st(1);
    }
    r.consume(20, 2); // 70 younger: clamped to 62
    assert_eq!(r.waits(), vec![(62, NO_LGKM)]);

    let mut r = rig();
    r.ds_ld(2);
    for _ in 0..13 {
        r.ds_st(1);
    }
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(NO_VM, 13)]);

    let mut r = rig();
    r.ds_ld(2);
    for _ in 0..20 {
        r.ds_st(1);
    }
    r.consume(20, 2); // 20 younger: clamped to 14
    assert_eq!(r.waits(), vec![(NO_VM, 14)]);
}

// ----------------------------------------------------------------------------
// Forward joins

/// A load issued on only one path is pending on the other as "maybe": after
/// the join, waits on its counter drain to zero rather than counting.
#[test]
fn forward_join_of_different_paths_is_conservative() {
    let mut r = rig();
    r.cmp_branch("L"); // branch point: nothing pending
    r.ld(2);
    r.ld(3);
    r.label("L");
    r.consume(20, 2); // positional would be vmcnt(1)
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
}

/// Two paths that issue the same loads into the same registers join
/// positionally: the count of younger loads survives the join.
#[test]
fn forward_join_of_matching_paths_keeps_positional_counts() {
    // The same pending loads on the branch path and the fall-through path.
    let mut r = rig();
    r.ld(2);
    r.ld(3);
    r.cmp_branch("L");
    r.push(nop());
    r.label("L");
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(1, NO_LGKM)]);

    // Two different paths issuing loads with identical shapes.
    let mut r = rig();
    r.cmp_branch("else");
    r.ld(2);
    r.ld(3);
    r.b.branch(SoppOp::Branch, "join").unwrap();
    r.label("else");
    r.ld(2);
    r.ld(3);
    r.label("join");
    r.consume(20, 2);
    assert_eq!(
        r.waits(),
        vec![(1, NO_LGKM)],
        "positional: one younger load on both paths"
    );
    r.consume(21, 3);
    assert_eq!(r.waits(), vec![(1, NO_LGKM), (0, NO_LGKM)]);
    let k = r.finish();
    assert_eq!(k.waits.len(), 2);
}

/// Loads into different registers on the two paths are different slots: the
/// join falls back to the conservative drain.
#[test]
fn forward_join_of_paths_with_different_destinations_is_conservative() {
    let mut r = rig();
    r.cmp_branch("else");
    r.ld(2);
    r.ld(3);
    r.b.branch(SoppOp::Branch, "join").unwrap();
    r.label("else");
    r.ld(2);
    r.ld(5);
    r.label("join");
    r.consume(20, 2);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
}

// ----------------------------------------------------------------------------
// Loops

fn loop_header(r: &mut Rig) {
    r.push(ops::s1(Sop1Op::MovB32, r.s(1), Src::Int(0)));
}
fn loop_latch(r: &mut Rig) -> Result<(), String> {
    r.push(ops::s2(Sop2Op::AddU32, r.s(1), r.s(1), Src::Int(1)));
    r.push(ops::sc(SopcOp::CmpLtU32, r.s(1), r.s(2)));
    r.b.branch(SoppOp::CbranchScc1, "top")
}

/// A backward branch that arrives with the pending shape its label had is
/// accepted: an empty ledger, and a non-empty one that the body leaves alone.
#[test]
fn loop_with_matching_pending_shape_is_accepted() {
    // Loads waited for inside the body: empty at the label and at the branch.
    let mut r = rig();
    loop_header(&mut r);
    r.label("top");
    r.ld(2);
    r.consume(20, 2);
    loop_latch(&mut r).expect("empty shape at the label and at the back edge");
    let k = r.finish();
    assert!(k
        .insts
        .iter()
        .any(|i| matches!(i, Inst::Sopp(p) if p.op == SoppOp::CbranchScc1)));

    // A load pending before the loop and untouched by it keeps its shape.
    let mut r = rig();
    loop_header(&mut r);
    r.ld(2);
    r.label("top");
    r.push(nop());
    loop_latch(&mut r).expect("same non-empty shape at the label and at the back edge");
    r.consume(20, 2);
    assert_eq!(
        r.waits(),
        vec![(0, NO_LGKM)],
        "the load is still pending after the loop"
    );
    r.finish();

    // Draining before the back edge when the label saw nothing pending.
    let mut r = rig();
    loop_header(&mut r);
    r.label("top");
    r.ld(2);
    r.b.wait_all().unwrap();
    loop_latch(&mut r).expect("wait_all restores the empty shape");
    r.finish();
}

/// A back edge that arrives with more pending than its label is an error.
#[test]
fn loop_back_edge_with_extra_pending_load_is_rejected() {
    let mut r = rig();
    loop_header(&mut r);
    r.label("top");
    r.ld(2); // still pending at the back edge, not pending at the label
    let res = loop_latch(&mut r).and_then(|_| r.b.finish().map(|_| ()));
    assert!(res.is_err(), "loop invariant violated: {res:?}");
}

/// ... and with less pending than its label.
#[test]
fn loop_back_edge_with_fewer_pending_than_the_label_is_rejected() {
    let mut r = rig();
    loop_header(&mut r);
    r.ld(2);
    r.label("top");
    r.consume(20, 2); // retires the load the label saw pending
    let res = loop_latch(&mut r).and_then(|_| r.b.finish().map(|_| ()));
    assert!(res.is_err(), "loop invariant violated: {res:?}");
}

/// ... and with a different shape (SMEM at the back edge, VMEM at the label).
#[test]
fn loop_back_edge_with_a_different_pending_kind_is_rejected() {
    let mut r = rig();
    loop_header(&mut r);
    r.ld(2);
    r.label("top");
    r.consume(20, 2);
    r.s_ld(6);
    let res = loop_latch(&mut r).and_then(|_| r.b.finish().map(|_| ()));
    assert!(res.is_err(), "loop invariant violated: {res:?}");
}

// ----------------------------------------------------------------------------
// Unreachable code and labels

#[test]
fn push_after_an_unconditional_branch_is_unreachable() {
    let mut r = rig();
    r.b.branch(SoppOp::Branch, "end").unwrap();
    assert!(r.b.push(nop()).is_err(), "unreachable instruction");
    assert!(r
        .b
        .push(ops::s1(Sop1Op::MovB32, r.s(0), Src::Int(1)))
        .is_err());

    // The target label resumes at the branch point.
    let mut r = rig();
    r.b.branch(SoppOp::Branch, "end").unwrap();
    r.label("end");
    r.push(nop());
    let k = r.finish();
    assert!(is_endpgm(k.insts.last().unwrap()));
}

/// A label after an unconditional branch resumes with the ledger of the
/// forward branch that targets it, not the fall-through path's.
#[test]
fn label_after_unconditional_branch_resumes_with_the_branch_ledger() {
    let mut r = rig();
    r.cmp_branch("else"); // nothing pending
    r.ld(2); // pending on the fall-through path only
    r.b.branch(SoppOp::Branch, "end").unwrap();
    assert!(r.b.push(nop()).is_err());
    r.label("else");
    r.consume(20, 2); // the else path never issued the load: no wait
    assert!(
        r.waits().is_empty(),
        "resumed with the branch-point ledger, got {:?}",
        r.waits()
    );
    r.label("end"); // joins [A] (maybe) from the first path
    r.consume(21, 2);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
    r.finish();
}

/// A label that no forward branch targets, placed where control cannot
/// arrive, is an error.
#[test]
fn label_with_no_incoming_path_after_an_unconditional_branch_is_rejected() {
    let mut r = rig();
    r.b.branch(SoppOp::Branch, "end").unwrap();
    assert!(r.b.label("orphan").is_err());
}

#[test]
fn duplicate_labels_and_undefined_targets_are_rejected() {
    let mut r = rig();
    r.label("x");
    assert!(r.b.label("x").is_err(), "duplicate label");

    let mut r = rig();
    r.cmp_branch("nowhere");
    r.push(nop());
    assert!(r.b.finish().is_err(), "branch to an undefined label");
}

/// `s_waitcnt`, `s_barrier`, `s_endpgm` and branches are not pushed
/// instructions: the builder owns them.
#[test]
fn push_rejects_builder_owned_instructions() {
    let mut r = rig();
    for op in [
        SoppOp::Waitcnt,
        SoppOp::Barrier,
        SoppOp::Endpgm,
        SoppOp::Branch,
        SoppOp::CbranchScc0,
        SoppOp::CbranchExecz,
    ] {
        assert!(
            r.b.push(ops::sp(op, 0)).is_err(),
            "{op:?} must not be pushable"
        );
    }
    r.push(nop()); // the builder is still usable
}

// ----------------------------------------------------------------------------
// Branch resolution

/// Offsets are relative to the next dword (PC + 4 + simm16 * 4, CDNA3 ISA §4.2
/// p.18) and count the dwords of multi-dword instructions (literal operands
/// between branch and target).
#[test]
fn branch_offsets_are_relative_to_the_next_dword_across_multiword_instructions() {
    let mut r = rig();
    r.push(ops::s1(Sop1Op::MovB32, r.s(0), Src::Lit(0x1234_5678))); // 2 dwords
    r.label("top");
    r.push(ops::sc(SopcOp::CmpEqU32, r.s(0), r.s(1)));
    r.b.branch(SoppOp::CbranchScc1, "mid").unwrap(); // forward over 3 x 2 dwords
    r.push(ops::v2(
        Vop2Op::AddF32,
        r.v(1),
        Src::Lit(0x3f80_0000),
        r.v(2),
    )); // 2 dwords
    r.push(ops::v1(Vop1Op::MovB32, r.v(3), Src::Lit(0x4000_0000))); // 2 dwords
    r.push(ops::s1(Sop1Op::MovB32, r.s(2), Src::Lit(7))); // 2 dwords
    r.label("mid");
    r.push(ops::s2(Sop2Op::AddU32, r.s(3), r.s(3), Src::Int(1)));
    r.push(ops::sc(SopcOp::CmpLtU32, r.s(3), r.s(4)));
    r.b.branch(SoppOp::CbranchScc1, "top").unwrap(); // backward
    r.b.branch(SoppOp::Branch, "end").unwrap(); // to the next instruction
    r.label("end");
    let k = r.finish();

    let pos = dword_positions(&k);
    let branches: Vec<(usize, i16)> = k
        .insts
        .iter()
        .enumerate()
        .filter_map(|(i, x)| x.branch_offset().map(|o| (i, o)))
        .collect();
    assert_eq!(branches.len(), 3);
    for ((i, off), name) in branches.iter().zip(["mid", "top", "end"]) {
        let want = label_dword(&k, name) as i64 - (pos[*i] as i64 + 1);
        assert_eq!(i64::from(*off), want, "branch to {name}");
    }

    // The forward offset is the number of dwords skipped, not instructions.
    let (mid_i, mid_off) = branches[0];
    let j = pos
        .iter()
        .position(|&p| p == label_dword(&k, "mid"))
        .expect("label on an instruction boundary");
    let skipped: usize = k.insts[mid_i + 1..j]
        .iter()
        .map(|x| x.words(CDNA).unwrap().len())
        .sum();
    assert_eq!(mid_off as usize, skipped);
    assert!(
        skipped >= 6 && skipped > j - (mid_i + 1),
        "multi-dword instructions are counted by dwords"
    );
    assert!(branches[1].1 < 0, "the back edge is negative");
    assert_eq!(
        label_dword(&k, "end"),
        k.words.len() - 1,
        "the label precedes s_endpgm"
    );

    // The words decode back to exactly the instructions.
    let decoded = decode_all(CDNA, &k.words).unwrap();
    assert_eq!(decoded.len(), k.insts.len());
    for ((off, inst), (p, want)) in decoded.iter().zip(pos.iter().zip(&k.insts)) {
        assert_eq!(off, p);
        assert_eq!(inst, want);
    }
    assert_eq!(k.words, encode_all(CDNA, &k.insts).unwrap());
}

/// A kernel is `insts` + `s_endpgm`, named and typed.
#[test]
fn finish_appends_endpgm_and_describes_the_kernel() {
    let mut r = rig();
    r.push(nop());
    let k = r.finish();
    assert_eq!(k.name, "t");
    assert_eq!(k.arch, CDNA);
    assert!(is_endpgm(k.insts.last().unwrap()));
    assert_eq!(k.insts.iter().filter(|i| is_endpgm(i)).count(), 1);
    assert_eq!(k.words.len(), dword_positions(&k).last().unwrap() + 1);
    assert!(k.waits.is_empty());
}

fn far_branch(forward: bool, nops: usize) -> Result<Kernel, String> {
    let mut r = rig();
    if forward {
        r.cmp_branch("far");
        for _ in 0..nops {
            r.push(nop());
        }
        r.label("far");
    } else {
        r.label("back");
        for _ in 0..nops {
            r.push(nop());
        }
        r.b.branch(SoppOp::CbranchScc1, "back")?;
    }
    r.b.finish()
}

fn only_branch_offset(k: &Kernel) -> i16 {
    let offs: Vec<i16> = k.insts.iter().filter_map(|i| i.branch_offset()).collect();
    assert_eq!(offs.len(), 1);
    offs[0]
}

/// SIMM16 is signed 16 bits of dwords: +32767 and -32768 encode, one past
/// either end is an error.
#[test]
fn branch_offsets_outside_the_signed_16_bit_dword_range_are_errors() {
    let k = far_branch(true, 32767).expect("forward +32767");
    assert_eq!(only_branch_offset(&k), 32767);
    assert!(far_branch(true, 32768).is_err(), "forward +32768");

    // Back edge: offset = -(nops + 1).
    let k = far_branch(false, 32767).expect("backward -32768");
    assert_eq!(only_branch_offset(&k), -32768);
    assert!(far_branch(false, 32768).is_err(), "backward -32769");
}

// ----------------------------------------------------------------------------
// Allocators, kernargs, system registers

fn vcc_use() -> Inst {
    ops::vc(VopcOp::CmpEqU32, Src::VccLo, Src::Int(0), Src::Vgpr(1))
}

#[test]
fn new_validates_workgroup_size_and_architecture() {
    for ok in [64, 128, 256, 576, 1024] {
        let b = Builder::new(Arch::Gfx942, "k", ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        assert_eq!(b.finish().unwrap().resources.workgroup_size, ok);
    }
    for bad in [0, 32, 100, 1088, 2048] {
        assert!(
            Builder::new(Arch::Gfx942, "k", bad).is_err(),
            "workgroup size {bad}"
        );
    }
    assert!(
        Builder::new(Arch::Gfx1100, "k", 64).is_err(),
        "an RDNA arch is not a CDNA arch"
    );
}

#[test]
fn sgprs_start_after_the_kernarg_pointer_aligned_and_capped() {
    let mut b = new_builder("s");
    assert_eq!(b.kernarg_base(), Src::Sgpr(0));
    assert_eq!(b.sgprs(1, 1).unwrap(), Src::Sgpr(2));
    assert_eq!(b.sgprs(2, 2).unwrap(), Src::Sgpr(4)); // s3 padded
    assert_eq!(b.sgprs(4, 4).unwrap(), Src::Sgpr(8));
    assert_eq!(b.sgprs(3, 5).unwrap(), Src::Sgpr(15)); // any positive alignment: next multiple of 5 after s11
    assert_eq!(b.sgprs(1, 7).unwrap(), Src::Sgpr(21));
    assert!(b.sgprs(0, 1).is_err(), "zero count");
    assert!(b.sgprs(1, 0).is_err(), "zero alignment");
    let k = b.finish().unwrap();
    assert_eq!(k.resources.next_free_sgpr, 22);
    assert!(!k.resources.uses_vcc);

    // s0..s101 without VCC: 100 SGPRs after the kernarg pointer.
    let mut b = new_builder("s");
    assert_eq!(b.sgprs(100, 1).unwrap(), Src::Sgpr(2));
    assert!(b.sgprs(1, 1).is_err(), "s102 does not exist");
    assert_eq!(b.finish().unwrap().resources.next_free_sgpr, 102);
    assert!(new_builder("s").sgprs(101, 1).is_err());
}

/// VCC occupies two SGPRs of the allocation: with VCC used the cap is 100.
#[test]
fn sgpr_cap_drops_by_two_when_vcc_is_used() {
    // VCC first, then SGPRs.
    let mut b = new_builder("v");
    b.vgprs(2, 1).unwrap();
    b.push(vcc_use()).unwrap();
    assert!(b.sgprs(99, 1).is_err(), "s2..s100 would reach 101 SGPRs");
    assert_eq!(b.sgprs(98, 1).unwrap(), Src::Sgpr(2));
    assert!(b.sgprs(1, 1).is_err());
    let k = b.finish().unwrap();
    assert!(k.resources.uses_vcc);
    assert_eq!(k.resources.next_free_sgpr, 100);

    // SGPRs first: a VCC-using instruction beyond the reduced cap is rejected
    // at push, and nothing invalid is committed.
    let mut b = new_builder("v");
    b.vgprs(2, 1).unwrap();
    b.sgprs(100, 1).unwrap();
    assert!(b.push(vcc_use()).is_err());
    b.push(nop()).unwrap();
    let k = b.finish().unwrap();
    assert!(!k.resources.uses_vcc);

    // Exactly at the reduced cap is fine.
    let mut b = new_builder("v");
    b.vgprs(2, 1).unwrap();
    b.sgprs(98, 1).unwrap();
    b.push(vcc_use()).unwrap();
    assert!(b.finish().unwrap().resources.uses_vcc);
}

#[test]
fn vgprs_reserve_v0_align_and_cap_at_256() {
    let mut b = new_builder("v");
    assert_eq!(
        b.vgprs(1, 1).unwrap(),
        Src::Vgpr(1),
        "v0 is reserved for the work-item IDs"
    );
    assert_eq!(b.vgprs(2, 2).unwrap(), Src::Vgpr(2));
    assert_eq!(b.vgprs(1, 1).unwrap(), Src::Vgpr(4));
    assert_eq!(b.vgprs(4, 4).unwrap(), Src::Vgpr(8));
    assert_eq!(b.vgprs(3, 5).unwrap(), Src::Vgpr(15));
    assert!(b.vgprs(0, 1).is_err());
    assert!(b.vgprs(1, 0).is_err());

    // 256 architectural VGPRs including v0.
    let mut b = new_builder("v");
    assert_eq!(b.vgprs(255, 1).unwrap(), Src::Vgpr(1));
    assert!(b.vgprs(1, 1).is_err(), "v256 does not exist");
    let k = b.finish().unwrap();
    assert_eq!(k.resources.next_free_vgpr, 256);
    assert_eq!(k.resources.accum_offset, 256);
    assert!(
        new_builder("v").vgprs(255, 2).is_err(),
        "aligned to 2 the range ends at 257"
    );
}

/// AGPRs are a separate 256-register file; the VGPR/AGPR split of the unified
/// 512-register file is `accum_offset` (next_free_vgpr rounded up to 4, >= 4).
#[test]
fn agprs_align_cap_at_256_and_fit_the_unified_file_with_accum_offset() {
    let mut b = new_builder("a");
    assert_eq!(b.agprs(1, 1).unwrap(), Src::Agpr(0));
    assert_eq!(b.agprs(4, 4).unwrap(), Src::Agpr(4));
    assert_eq!(b.agprs(2, 2).unwrap(), Src::Agpr(8));
    assert!(b.agprs(0, 1).is_err());
    assert!(b.agprs(1, 0).is_err());
    let k = b.finish().unwrap();
    assert_eq!(k.resources.agprs, 10);
    assert_eq!(k.resources.next_free_vgpr, 1, "only v0 is in use");
    assert_eq!(k.resources.accum_offset, 4, "accum_offset is at least 4");

    // accum_offset = next_free_vgpr rounded up to a multiple of 4.
    for (vgprs, next_free, accum) in [
        (3u8, 4u16, 4u16),
        (4, 5, 8),
        (5, 6, 8),
        (7, 8, 8),
        (8, 9, 12),
    ] {
        let mut b = new_builder("a");
        b.vgprs(vgprs, 1).unwrap();
        b.agprs(4, 4).unwrap();
        let k = b.finish().unwrap();
        assert_eq!(k.resources.next_free_vgpr, next_free, "{vgprs} VGPRs");
        assert_eq!(k.resources.accum_offset, accum, "{vgprs} VGPRs");
        assert!(k.resources.accum_offset + k.resources.agprs <= 512);
    }

    // 256 AGPRs + 256 VGPRs fill the unified file exactly.
    let mut b = new_builder("a");
    b.vgprs(255, 1).unwrap();
    assert_eq!(b.agprs(255, 1).unwrap(), Src::Agpr(0));
    assert_eq!(b.agprs(1, 1).unwrap(), Src::Agpr(255));
    assert!(b.agprs(1, 1).is_err(), "a256 does not exist");
    let k = b.finish().unwrap();
    assert_eq!(k.resources.agprs, 256);
    assert_eq!(k.resources.accum_offset, 256);
    assert_eq!(k.resources.accum_offset + k.resources.agprs, 512);

    // The split also bounds VGPR allocation once AGPRs exist.
    let mut b = new_builder("a");
    b.agprs(255, 1).unwrap();
    b.agprs(1, 1).unwrap();
    assert_eq!(b.vgprs(255, 1).unwrap(), Src::Vgpr(1));
}

#[test]
fn kernarg_list_offsets_sizes_and_pointer_alignment() {
    let mut b = new_builder("k");
    assert_eq!(b.arg_ptr("a", true).unwrap(), 0);
    assert_eq!(b.arg_ptr("b", true).unwrap(), 8);
    assert_eq!(b.arg_ptr("out", false).unwrap(), 16);
    assert_eq!(b.arg_u32("n").unwrap(), 24);
    assert!(b.arg_u32("").is_err(), "empty name");
    assert!(b.arg_u32("n").is_err(), "duplicate name");
    assert!(b.arg_ptr("a", false).is_err(), "duplicate name");
    let k = b.finish().unwrap();
    let got: Vec<(&str, u32, u32, KernArgKind)> = k
        .kernargs
        .iter()
        .map(|a| (a.name.as_str(), a.offset, a.size, a.kind))
        .collect();
    assert_eq!(
        got,
        vec![
            ("a", 0, 8, KernArgKind::GlobalBuffer { readonly: true }),
            ("b", 8, 8, KernArgKind::GlobalBuffer { readonly: true }),
            ("out", 16, 8, KernArgKind::GlobalBuffer { readonly: false }),
            ("n", 24, 4, KernArgKind::ByValue),
        ]
    );
    assert_eq!(
        k.resources.kernarg_size, 28,
        "the actual end of the last argument"
    );

    // Pointers are 8-aligned after 4-byte arguments; u32s pack at 4.
    let mut b = new_builder("k");
    assert_eq!(b.arg_u32("x").unwrap(), 0);
    assert_eq!(b.arg_u32("y").unwrap(), 4);
    assert_eq!(b.arg_u32("z").unwrap(), 8);
    assert_eq!(b.arg_ptr("p", true).unwrap(), 16);
    assert_eq!(b.arg_u32("w").unwrap(), 24);
    let k = b.finish().unwrap();
    assert_eq!(k.resources.kernarg_size, 28);
    assert!(k.kernargs.iter().all(|a| a.offset % a.size == 0));

    assert_eq!(new_builder("k").finish().unwrap().resources.kernarg_size, 0);
}

#[test]
fn workgroup_ids_follow_the_kernarg_pointer_in_ascending_dimension_order() {
    let mut b = new_builder("w");
    assert_eq!(b.workgroup_id(0).unwrap(), Src::Sgpr(2));
    assert_eq!(b.workgroup_id(1).unwrap(), Src::Sgpr(3));
    assert_eq!(b.workgroup_id(2).unwrap(), Src::Sgpr(4));
    assert_eq!(
        b.workgroup_id(0).unwrap(),
        Src::Sgpr(2),
        "asking again returns the same register"
    );
    assert!(b.workgroup_id(3).is_err());
    assert_eq!(
        b.sgprs(1, 1).unwrap(),
        Src::Sgpr(5),
        "allocatable SGPRs start after the enabled IDs"
    );
    let k = b.finish().unwrap();
    assert_eq!(k.resources.workgroup_id, [true, true, true]);

    // Only Y enabled: it is the first system SGPR after the kernarg pointer.
    let mut b = new_builder("w");
    assert_eq!(b.workgroup_id(1).unwrap(), Src::Sgpr(2));
    assert_eq!(b.sgprs(1, 1).unwrap(), Src::Sgpr(3));
    assert_eq!(
        b.finish().unwrap().resources.workgroup_id,
        [false, true, false]
    );

    // A newly enabled dimension must come after the enabled lower ones: the
    // hardware lays the IDs out X, Y, Z, so enabling X after Y would move Y.
    let mut b = new_builder("w");
    assert_eq!(b.workgroup_id(2).unwrap(), Src::Sgpr(2));
    assert!(b.workgroup_id(0).is_err());
    assert!(b.workgroup_id(1).is_err());

    // IDs must be requested before any SGPR is allocated.
    let mut b = new_builder("w");
    b.sgprs(1, 1).unwrap();
    assert!(b.workgroup_id(0).is_err());

    assert_eq!(
        new_builder("w").finish().unwrap().resources.workgroup_id,
        [false; 3]
    );
}

#[test]
fn workitem_ids_live_in_the_reserved_v0() {
    let mut b = new_builder("t");
    assert_eq!(b.workitem_ids(1).unwrap(), Src::Vgpr(0));
    assert!(b.workitem_ids(0).is_err());
    assert!(b.workitem_ids(4).is_err());
    assert_eq!(
        b.vgprs(1, 1).unwrap(),
        Src::Vgpr(1),
        "v0 is never allocated"
    );
    assert_eq!(b.finish().unwrap().resources.workitem_id, 0);

    for (dims, want) in [(2u8, 1u8), (3, 2)] {
        let mut b = new_builder("t");
        assert_eq!(b.workitem_ids(dims).unwrap(), Src::Vgpr(0));
        assert_eq!(b.finish().unwrap().resources.workitem_id, want);
    }

    // v0 is reserved even when the kernel does not read the IDs.
    let mut b = new_builder("t");
    assert_eq!(b.vgprs(1, 1).unwrap(), Src::Vgpr(1));
    assert_eq!(b.finish().unwrap().resources.workitem_id, 0);
}

#[test]
fn lds_allocations_align_and_respect_the_arch_limit() {
    let mut b = new_builder("l");
    assert_eq!(
        b.lds(256, 16).unwrap(),
        Lds {
            offset: 0,
            bytes: 256
        }
    );
    assert_eq!(
        b.lds(100, 64).unwrap(),
        Lds {
            offset: 256,
            bytes: 100
        }
    );
    assert_eq!(
        b.lds(4, 4).unwrap(),
        Lds {
            offset: 356,
            bytes: 4
        }
    );
    assert!(b.finish().unwrap().resources.group_segment_fixed_size >= 360);

    let mut b = new_builder("l");
    assert_eq!(
        b.lds(65536, 16).unwrap(),
        Lds {
            offset: 0,
            bytes: 65536
        }
    );
    assert!(b.lds(1, 1).is_err(), "gfx942 has 64 KiB of LDS");
    assert!(new_builder("l").lds(65537, 1).is_err());
    assert!(new_builder("l").lds(16, 0).is_err(), "zero alignment");
}

#[test]
fn finish_reports_the_descriptor_resources() {
    let mut b = Builder::new(Arch::Gfx942, "res", 256).unwrap();
    b.arg_ptr("p", false).unwrap();
    b.workgroup_id(0).unwrap();
    b.workgroup_id(1).unwrap();
    b.workitem_ids(2).unwrap();
    let s = b.sgprs(4, 4).unwrap();
    b.vgprs(5, 1).unwrap();
    b.agprs(4, 4).unwrap();
    b.lds(512, 16).unwrap();
    b.set_fp16_ovfl(true);
    b.push(ops::s1(Sop1Op::MovB32, s, Src::Int(1))).unwrap();
    let k = b.finish().unwrap();
    let r = &k.resources;
    assert_eq!(r.next_free_sgpr, 8);
    assert_eq!(r.next_free_vgpr, 6);
    assert_eq!(r.agprs, 4);
    assert_eq!(r.accum_offset, 8);
    assert!(!r.uses_vcc);
    assert!(r.group_segment_fixed_size >= 512);
    assert_eq!(r.kernarg_size, 8);
    assert_eq!(r.workgroup_id, [true, true, false]);
    assert_eq!(r.workitem_id, 1);
    assert_eq!(r.workgroup_size, 256);
    assert!(r.fp16_ovfl);

    let b = new_builder("def");
    assert!(!b.finish().unwrap().resources.fp16_ovfl);
    let mut b = new_builder("vcc");
    b.vgprs(2, 1).unwrap();
    b.push(vcc_use()).unwrap();
    assert!(b.finish().unwrap().resources.uses_vcc);
}

// ----------------------------------------------------------------------------
// End to end: vector add on the CPU emulator

const A_BASE: u64 = 0x1000_0000;
const B_BASE: u64 = 0x2000_0000;
const O_BASE: u64 = 0x3000_0000;
const KARG_BASE: u64 = 0x7000_0000;
const POISON: u32 = 0xdead_beef;
/// Mapped words after `out[n]`: a store beyond `n` by an inactive lane lands
/// here and is caught.
const PAD: usize = 300;

/// `out[i] = a[i] + b[i]` for `i < n`; one work-item per element, workgroups
/// of 256, bounds mask by `v_cmp` + `s_and_saveexec_b64` (no branch, so the
/// finished instructions can be re-encoded after removing waits).
fn vector_add_kernel() -> Kernel {
    let mut b = Builder::new(Arch::Gfx942, "pm_vector_add", 256).unwrap();
    let a_off = b.arg_ptr("a", true).unwrap();
    let b_off = b.arg_ptr("b", true).unwrap();
    let o_off = b.arg_ptr("out", false).unwrap();
    let n_off = b.arg_u32("n").unwrap();
    assert_eq!([a_off, b_off, o_off, n_off], [0, 8, 16, 24]);
    let wg = b.workgroup_id(0).unwrap();
    let tid = b.workitem_ids(1).unwrap();
    let ab = b.sgprs(4, 4).unwrap();
    let out = b.sgprs(2, 2).unwrap();
    let n = b.sgprs(1, 1).unwrap();
    let wgbase = b.sgprs(1, 1).unwrap();
    let save = b.sgprs(2, 2).unwrap();
    let idx = b.vgprs(1, 1).unwrap();
    let off = b.vgprs(1, 1).unwrap();
    let va = b.vgprs(1, 1).unwrap();
    let vb = b.vgprs(1, 1).unwrap();
    let vr = b.vgprs(1, 1).unwrap();
    let base = b.kernarg_base();
    let program = [
        ops::s_load(SmemOp::LoadDwordx4, ab, base, a_off as i32),
        ops::s_load(SmemOp::LoadDwordx2, out, base, o_off as i32),
        ops::s_load(SmemOp::LoadDword, n, base, n_off as i32),
        // idx = workgroup_id * 256 + (v0 & 0x3ff)
        ops::s2(Sop2Op::LshlB32, wgbase, wg, Src::Int(8)),
        ops::v2(Vop2Op::AndB32, idx, Src::Lit(0x3ff), tid),
        ops::v2(Vop2Op::AddU32, idx, wgbase, idx),
        // exec &= (n > idx)
        ops::vc(VopcOp::CmpGtU32, Src::VccLo, n, idx),
        ops::s1(Sop1Op::AndSaveexecB64, save, Src::VccLo),
        ops::v2(Vop2Op::LshlrevB32, off, Src::Int(2), idx),
        ops::global_load(FlatOp::LoadDword, va, off, Some(ab), 0),
        ops::global_load(FlatOp::LoadDword, vb, off, Some(sg(ab, 2)), 0),
        ops::v2(Vop2Op::AddF32, vr, va, vb),
        ops::global_store(FlatOp::StoreDword, off, vr, Some(out), 0),
        ops::s2(Sop2Op::OrB64, Src::ExecLo, Src::ExecLo, save),
    ];
    for inst in program {
        b.push(inst).unwrap_or_else(|e| panic!("push: {e}"));
    }
    b.finish().unwrap_or_else(|e| panic!("finish: {e}"))
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
/// A random finite normal f32: exponent within 2^-20..2^20, any sign and
/// mantissa, so sums are normal and CPU/emulator denorm modes cannot differ.
fn random_f32_bits(state: &mut u64) -> u32 {
    let r = xorshift(state);
    let exp = 127 - 20 + ((r >> 32) % 41) as u32;
    ((r >> 63) as u32) << 31 | exp << 23 | (r as u32 & 0x7f_ffff)
}
fn le_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn words_of(mem: &Memory, region: &str) -> Vec<u32> {
    mem.region(region)
        .unwrap()
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Run {
    n: usize,
    result: Result<sim::Stats, SimError>,
    out: Vec<u32>,
    expected: Vec<u32>,
    pad: Vec<u32>,
}

fn run_vector_add(k: &Kernel, version: CodeObjectVersion, n: usize, seed: u64) -> Run {
    let mut state = seed | 1;
    let mut a: Vec<u32> = (0..n).map(|_| random_f32_bits(&mut state)).collect();
    let mut b: Vec<u32> = (0..n).map(|_| random_f32_bits(&mut state)).collect();
    if n >= 2 {
        // signed zeros: (+0) + (-0) = +0, (-0) + (-0) = -0
        a[0] = 0.0f32.to_bits();
        b[0] = (-0.0f32).to_bits();
        a[1] = (-0.0f32).to_bits();
        b[1] = (-0.0f32).to_bits();
    }
    let expected: Vec<u32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f32::from_bits(*x) + f32::from_bits(*y)).to_bits())
        .collect();

    let elf = code_object(&[k], version).expect("code object");
    assert!(elf.starts_with(b"\x7fELF"));
    let image = sim::load_code_object(&elf, &k.name).expect("load_code_object");

    let mut mem = Memory::default();
    mem.map("a", A_BASE, le_bytes(&a)).unwrap();
    mem.map("b", B_BASE, le_bytes(&b)).unwrap();
    mem.map("out", O_BASE, le_bytes(&vec![POISON; n + PAD]))
        .unwrap();
    let mut kernarg = vec![0u8; k.resources.kernarg_size as usize];
    for (arg, value) in k.kernargs.iter().zip([A_BASE, B_BASE, O_BASE, n as u64]) {
        let at = arg.offset as usize;
        kernarg[at..at + arg.size as usize]
            .copy_from_slice(&value.to_le_bytes()[..arg.size as usize]);
    }
    let launch = Launch {
        grid: [n.div_ceil(256) as u32, 1, 1],
        block: [256, 1, 1],
        kernarg,
        kernarg_addr: KARG_BASE,
        dynamic_lds: 0,
    };
    let result = sim::run(&image, &launch, &mut mem, &Config::default());
    let all = words_of(&mem, "out");
    Run {
        n,
        result,
        out: all[..n].to_vec(),
        expected,
        pad: all[n..].to_vec(),
    }
}

fn assert_exact(run: &Run, what: &str) {
    let stats = run
        .result
        .as_ref()
        .unwrap_or_else(|e| panic!("{what} n={}: {e}", run.n));
    assert_eq!(
        stats.workgroups,
        run.n.div_ceil(256) as u64,
        "{what} n={}",
        run.n
    );
    assert_eq!(
        stats.waves,
        4 * run.n.div_ceil(256) as u64,
        "{what} n={}",
        run.n
    );
    if let Some(i) = (0..run.n).find(|&i| run.out[i] != run.expected[i]) {
        panic!(
            "{what} n={}: out[{i}] = {:#010x}, want {:#010x}",
            run.n, run.out[i], run.expected[i]
        );
    }
    assert!(
        run.pad.iter().all(|w| *w == POISON),
        "{what} n={}: an inactive lane stored past n",
        run.n
    );
}

fn assert_wait_or_differs(run: &Run, what: &str) {
    match &run.result {
        Err(SimError::Wait(_)) => {}
        Err(other) => panic!("{what}: expected SimError::Wait or a wrong result, got {other}"),
        Ok(_) => assert!(
            run.out != run.expected || run.pad.iter().any(|w| *w != POISON),
            "{what}: ran clean and bit-exact without its waits"
        ),
    }
}

#[test]
fn vector_add_kernel_shape() {
    let k = vector_add_kernel();
    assert_eq!(k.name, "pm_vector_add");
    assert_eq!(k.arch, CDNA);
    let got: Vec<(&str, u32, u32, KernArgKind)> = k
        .kernargs
        .iter()
        .map(|a| (a.name.as_str(), a.offset, a.size, a.kind))
        .collect();
    assert_eq!(
        got,
        vec![
            ("a", 0, 8, KernArgKind::GlobalBuffer { readonly: true }),
            ("b", 8, 8, KernArgKind::GlobalBuffer { readonly: true }),
            ("out", 16, 8, KernArgKind::GlobalBuffer { readonly: false }),
            ("n", 24, 4, KernArgKind::ByValue),
        ]
    );
    let r = &k.resources;
    assert_eq!(r.kernarg_size, 28);
    assert_eq!(r.workgroup_id, [true, false, false]);
    assert_eq!(r.workitem_id, 0);
    assert_eq!(r.workgroup_size, 256);
    assert!(r.uses_vcc);
    assert_eq!(r.next_free_sgpr, 14);
    assert_eq!(r.next_free_vgpr, 6);
    assert_eq!(r.accum_offset, 8);
    assert_eq!(r.agprs, 0);
    assert!(is_endpgm(k.insts.last().unwrap()));
    assert!(k.insts.iter().all(|i| i.branch_offset().is_none()));

    // Exactly two waits: lgkmcnt(0) for n before the compare, vmcnt(0) for
    // both loads before the add (the loads already waited: one younger).
    let seen: Vec<(u8, u8, u8)> = k.insts.iter().filter_map(waitcnt_of).collect();
    assert_eq!(seen, vec![(NO_VM, NO_EXP, 0), (0, NO_EXP, NO_LGKM)]);
    assert_eq!(k.waits.len(), 2);
    assert_eq!(k.waits[0].counter, Counter::Lgkm);
    assert_eq!(k.waits[1].counter, Counter::Vm);
    for p in &k.waits {
        assert!(waitcnt_of(&k.insts[p.pc_index]).is_some());
    }

    let violations = hazard::check(CDNA, &k.insts);
    assert!(
        violations.is_empty(),
        "hazards in the finished kernel: {violations:?}"
    );
    let decoded = decode_all(CDNA, &k.words).unwrap();
    assert_eq!(decoded.len(), k.insts.len());
    assert!(decoded.iter().zip(&k.insts).all(|((_, d), i)| d == i));
}

#[test]
fn vector_add_matches_the_cpu_reference_bit_exactly_including_the_inactive_tail() {
    let k = vector_add_kernel();
    for version in [CodeObjectVersion::V5, CodeObjectVersion::V6] {
        // Exact multiples, partial waves, a partial last workgroup, n = 1
        // (three whole waves of the one workgroup are inactive).
        for n in [
            1usize,
            2,
            63,
            64,
            65,
            255,
            256,
            257,
            512,
            1000,
            1024,
            3 * 256 + 17,
        ] {
            let run = run_vector_add(&k, version, n, 0x9e37_79b9_7f4a_7c15 ^ (n as u64) << 8);
            assert_exact(&run, &format!("{version:?}"));
        }
    }
}

/// Re-encoding the unmodified finished instructions reproduces the words and
/// passes: the hand-assembly path used by the negative tests is faithful.
fn without_waits(k: &Kernel, drop_vm: bool, drop_lgkm: bool) -> Kernel {
    assert!(
        k.insts.iter().all(|i| i.branch_offset().is_none()),
        "no branch offsets to re-resolve"
    );
    let insts: Vec<Inst> = k
        .insts
        .iter()
        .filter_map(|inst| match waitcnt_of(inst) {
            None => Some(inst.clone()),
            Some((vm, exp, lgkm)) => {
                let vm = if drop_vm { NO_VM } else { vm };
                let lgkm = if drop_lgkm { NO_LGKM } else { lgkm };
                if vm == NO_VM && exp == NO_EXP && lgkm == NO_LGKM {
                    None
                } else {
                    Some(Inst::Sopp(Sopp {
                        op: SoppOp::Waitcnt,
                        simm16: waitcnt_imm(CDNA, vm, exp, lgkm).unwrap(),
                    }))
                }
            }
        })
        .collect();
    let words = encode_all(CDNA, &insts).unwrap();
    Kernel {
        insts,
        words,
        labels: Vec::new(),
        waits: Vec::new(),
        hazards: Vec::new(),
        ..k.clone()
    }
}

#[test]
fn hand_reassembled_kernel_without_changes_still_runs_exactly() {
    let k = vector_add_kernel();
    let same = without_waits(&k, false, false);
    assert_eq!(same.words, k.words);
    for n in [1usize, 257, 1000] {
        assert_exact(
            &run_vector_add(&same, CodeObjectVersion::V6, n, 7 + n as u64),
            "re-assembled",
        );
    }
}

/// Removing the waits from the finished instructions must be caught by the
/// emulator: a use of an in-flight load/SMEM result is `SimError::Wait`
/// (or, at the least, the output differs).
#[test]
fn removing_the_waits_from_the_finished_kernel_is_rejected_by_the_emulator() {
    let k = vector_add_kernel();
    let full = count_waitcnts(&k);
    assert_eq!(full, 2);
    for (drop_vm, drop_lgkm, what) in [
        (true, true, "no s_waitcnt at all"),
        (true, false, "vmcnt waits removed"),
        (false, true, "lgkmcnt waits removed"),
    ] {
        let broken = without_waits(&k, drop_vm, drop_lgkm);
        assert!(
            count_waitcnts(&broken) < full,
            "{what}: something was removed"
        );
        for version in [CodeObjectVersion::V5, CodeObjectVersion::V6] {
            for n in [1usize, 1000] {
                let run = run_vector_add(&broken, version, n, 0x1234_5678_9abc_def0 + n as u64);
                assert_wait_or_differs(&run, what);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Memory operand classes and ledger locks

/// `inst` must be rejected by `push` with nothing committed, and the builder
/// must stay usable.
fn assert_push_rejected(inst: Inst, what: &str) {
    let mut r = rig();
    assert!(r.b.push(inst).is_err(), "{what} must be rejected");
    assert!(r.items().is_empty(), "{what}: nothing invalid is committed");
    r.push(nop());
    assert_eq!(r.items().len(), 1);
}

/// SMEM data destinations are plain SGPRs: not VCC, EXEC, FLAT_SCRATCH,
/// XNACK_MASK, TTMP or M0 (the ledger tracks only SGPR/VGPR spans).
#[test]
fn smem_destination_must_be_a_plain_sgpr() {
    let mut r = rig();
    r.s_ld(0);
    r.s_ld2(2); // plain SGPR destinations are accepted
    for (dst, what) in [
        (Src::VccLo, "vcc_lo"),
        (Src::VccHi, "vcc_hi"),
        (Src::ExecLo, "exec_lo"),
        (Src::ExecHi, "exec_hi"),
        (Src::M0, "m0"),
        (Src::Ttmp(0), "ttmp0"),
        (Src::FlatScratchLo, "flat_scratch_lo"),
        (Src::XnackMaskLo, "xnack_mask_lo"),
    ] {
        assert_push_rejected(
            ops::s_load(SmemOp::LoadDword, dst, Src::Sgpr(0), 0),
            &format!("s_load_dword {what}"),
        );
    }
    for (dst, what) in [
        (Src::VccLo, "vcc"),
        (Src::ExecLo, "exec"),
        (Src::Ttmp(0), "ttmp[0:1]"),
        (Src::FlatScratchLo, "flat_scratch"),
        (Src::XnackMaskLo, "xnack_mask"),
    ] {
        assert_push_rejected(
            ops::s_load(SmemOp::LoadDwordx2, dst, Src::Sgpr(0), 0),
            &format!("s_load_dwordx2 {what}"),
        );
    }
}

/// Global and buffer load destinations / store data are plain VGPRs.
#[test]
fn vmem_data_must_be_a_plain_vgpr() {
    let r = rig();
    let (v0, ptr, rsrc) = (r.v(0), r.ptr, r.s(4));
    for (bad, what) in [
        (Src::Agpr(0), "a0"),
        (Src::Ttmp(0), "ttmp0"),
        (Src::VccLo, "vcc_lo"),
        (Src::ExecLo, "exec_lo"),
        (Src::M0, "m0"),
    ] {
        assert_push_rejected(
            ops::global_load(FlatOp::LoadDword, bad, v0, Some(ptr), 0),
            &format!("global_load_dword {what}"),
        );
        assert_push_rejected(
            ops::global_store(FlatOp::StoreDword, v0, bad, Some(ptr), 0),
            &format!("global_store_dword {what}"),
        );
        assert_push_rejected(
            ops::buffer_load(
                MubufOp::LoadDword,
                bad,
                Some(v0),
                rsrc,
                Src::Int(0),
                0,
                true,
            ),
            &format!("buffer_load_dword {what}"),
        );
        assert_push_rejected(
            ops::buffer_store(
                MubufOp::StoreDword,
                bad,
                Some(v0),
                rsrc,
                Src::Int(0),
                0,
                true,
            ),
            &format!("buffer_store_dword {what}"),
        );
    }
    let mut r = rig();
    r.ld(2); // a plain VGPR destination is accepted
    r.st(1);
}

/// DS destinations and data (including cross-lane results) are plain VGPRs.
#[test]
fn ds_data_must_be_a_plain_vgpr() {
    let r = rig();
    let (v0, v1) = (r.v(0), r.v(1));
    for (bad, what) in [
        (Src::Agpr(0), "a0"),
        (Src::Ttmp(0), "ttmp0"),
        (Src::VccLo, "vcc_lo"),
        (Src::M0, "m0"),
    ] {
        assert_push_rejected(
            ops::ds_read(DsOp::ReadB32, bad, v0, 0),
            &format!("ds_read_b32 {what}"),
        );
        assert_push_rejected(
            ops::ds_write(DsOp::WriteB32, v0, bad, 0),
            &format!("ds_write_b32 {what}"),
        );
        assert_push_rejected(
            ops::ds_xlane(DsOp::BpermuteB32, bad, v0, v1, 0),
            &format!("ds_bpermute_b32 {what}"),
        );
    }
    let mut r = rig();
    r.ds_ld(2);
    r.ds_st(1);
    r.bperm(3);
}

/// A younger write to a register a pending store reads waits for the store
/// (WAR lock); the lock counts in issue order like a load's result.
#[test]
fn pending_vmem_store_locks_its_source_registers() {
    // Data register.
    let mut r = rig();
    r.st(1);
    let w = ops::v1(Vop1Op::MovB32, r.v(1), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
    let w = ops::v1(Vop1Op::MovB32, r.v(1), Src::Int(1));
    r.push(w); // the store retired
    assert_eq!(r.waits().len(), 1);
    let k = r.finish();
    assert_eq!(k.waits.len(), 1);
    assert_eq!(k.waits[0].counter, Counter::Vm);
    assert!(
        matches!(k.waits[0].reason, Reason::War { .. }),
        "{:?}",
        k.waits[0].reason
    );

    // Address register; younger VMEM operations count.
    let mut r = rig();
    r.st(1);
    r.ld(2);
    r.ld(3);
    let w = ops::v1(Vop1Op::MovB32, r.v(0), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(2, NO_LGKM)]);

    // Mixed with an older load: only the store's position matters.
    let mut r = rig();
    r.ld(2);
    r.st(1);
    r.ld(3);
    let w = ops::v1(Vop1Op::MovB32, r.v(1), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(1, NO_LGKM)]);

    // Reading a store's sources is not a hazard.
    let mut r = rig();
    r.st(1);
    r.consume(20, 1);
    assert!(r.waits().is_empty());
}

/// Overwriting a pending load's destination waits for it (WAW).
#[test]
fn pending_load_destination_is_waited_for_before_overwrite() {
    let mut r = rig();
    r.ld(2);
    let w = ops::v1(Vop1Op::MovB32, r.v(2), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(0, NO_LGKM)]);
    let k = r.finish();
    assert!(
        matches!(k.waits[0].reason, Reason::Waw { .. }),
        "{:?}",
        k.waits[0].reason
    );

    let mut r = rig();
    r.ld(2);
    r.st(1);
    let w = ops::v1(Vop1Op::MovB32, r.v(2), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(1, NO_LGKM)], "the younger store counts");
}

/// A DS store's address and data registers are locked until it retires; a
/// younger DS operation counts toward the wait.
#[test]
fn pending_ds_store_locks_its_source_registers() {
    let mut r = rig();
    r.ds_st(1);
    let w = ops::v1(Vop1Op::MovB32, r.v(1), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(NO_VM, 0)]);

    let mut r = rig();
    r.ds_st(1);
    r.ds_ld(2);
    let w = ops::v1(Vop1Op::MovB32, r.v(1), Src::Int(0));
    r.push(w);
    assert_eq!(r.waits(), vec![(NO_VM, 1)]);
}

/// Only GLOBAL-segment `Inst::Flat` is accepted: FLAT (dual VM/LGKM counting)
/// and SCRATCH (the descriptor has no private segment) are rejected with a
/// message naming the problem, and the builder stays usable for GLOBAL.
#[test]
fn only_global_segment_flat_instructions_are_accepted() {
    let flat = |seg: FlatSeg, load: bool| {
        let r = rig();
        let (v0, v1, v2) = (r.v(0), r.v(1), r.v(2));
        // FLAT takes a 64-bit VGPR address; SCRATCH a 32-bit VGPR offset.
        let addr = Some(v0);
        Inst::Flat(Flat {
            op: if load {
                FlatOp::LoadDword
            } else {
                FlatOp::StoreDword
            },
            seg,
            vdst: if load { Some(v2) } else { None },
            addr,
            data: if load { None } else { Some(v1) },
            saddr: None,
            offset: 0,
            sc0: false,
            sc1: false,
            nt: false,
        })
    };
    for (seg, name) in [(FlatSeg::Flat, "flat"), (FlatSeg::Scratch, "scratch")] {
        for load in [true, false] {
            let mut r = rig();
            let err = r.b.push(flat(seg, load)).expect_err(&format!(
                "{name} {} must be rejected",
                if load { "load" } else { "store" }
            ));
            let lower = err.to_lowercase();
            assert!(
                lower.contains("global") || lower.contains("flat") || lower.contains("scratch"),
                "the error names the unsupported segment: {err:?}"
            );
            assert!(r.items().is_empty(), "nothing is committed");
            r.ld(2); // GLOBAL is still accepted
            r.st(1);
            assert_eq!(r.items().len(), 2);
        }
    }
}
