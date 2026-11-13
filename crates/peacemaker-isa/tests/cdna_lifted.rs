// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cdna::lifted::certify_stream` (the typed-stream core of `certify`): hazard check plus CFG
//! wait-ledger replay over decoded gfx942 instructions. Streams are built in memory from
//! typed instructions, so no compiled anchor is needed. Each negative case removes exactly
//! one thing a positive case has (a wait state, a wait, a barrier drain) and must fail.

use peacemaker_cdna::enc::{
    ds::{Ds, DsOp},
    flat::{FlatOp, FlatSeg},
    smem::SmemOp,
    sop::{waitcnt_imm, Sop1Op, Sop2Op, SopcOp, SoppOp},
    vop::{Vop1Op, VopcOp},
    vop3p::MaiOp,
};
use peacemaker_cdna::{Arch, Inst, Src};
use peacemaker_isa::arch::Arch as IsaArch;
use peacemaker_isa::cdna::lifted::certify_stream;
use peacemaker_isa::cdna::ops;
use peacemaker_isa::insn::{Instruction as Text, MemoryClass, Program};
use peacemaker_isa::ledger_cfg::{verify, verify_dispatcher};
use peacemaker_isa::reg::{Kind, RegRef};
use peacemaker_lift::cdna::Instruction;
use serde_json::Value;
use Src::{Sgpr as s, Vgpr as v};

const A: Arch = Arch::Gfx942;

/// The stream with the true dword offsets the encoder produces.
fn stream(insts: &[Inst]) -> Vec<Instruction> {
    let mut at = 0;
    insts
        .iter()
        .map(|inst| {
            let it = Instruction {
                dword_offset: at,
                inst: inst.clone(),
            };
            at += inst.words(A).expect("encodes").len();
            it
        })
        .collect()
}

fn certify(insts: &[Inst]) -> Result<Value, String> {
    certify_stream("k", &stream(insts))
}

fn end() -> Inst {
    ops::sp(SoppOp::Endpgm, 0)
}
fn wait(vm: u8, lgkm: u8) -> Inst {
    ops::sp(
        SoppOp::Waitcnt,
        waitcnt_imm(A, vm, 7, lgkm).expect("valid wait"),
    )
}
fn nop(imm: u16) -> Inst {
    ops::sp(SoppOp::Nop, imm)
}
fn use_of(src: u8) -> Inst {
    ops::v1(Vop1Op::MovB32, v(200), v(src))
}
fn load(dst: u8) -> Inst {
    ops::global_load(FlatOp::LoadDword, v(dst), v(2), None, 0)
}
fn store(data: u8) -> Inst {
    ops::global_store(FlatOp::StoreDword, v(2), v(data), None, 0)
}
fn ds_read(dst: u8) -> Inst {
    ops::ds_read(DsOp::ReadB32, v(dst), v(2), 0)
}
fn ds_write(data: u8) -> Inst {
    ops::ds_write(DsOp::WriteB32, v(2), v(data), 0)
}
fn barrier() -> Inst {
    ops::sp(SoppOp::Barrier, 0)
}
/// A 16x16x16 BF16 MFMA (XDL, 4 passes) writing v[0:3].
fn mfma() -> Inst {
    ops::mfma(MaiOp::MfmaF32_16x16x16Bf16, v(0), v(4), v(6), v(0))
}
/// Reads v1, inside the MFMA's destination.
fn read_mfma_result() -> Inst {
    ops::v1(Vop1Op::MovB32, v(100), v(1))
}

fn ok(insts: &[Inst]) -> Value {
    certify(insts).unwrap_or_else(|e| panic!("expected certification, got: {e}"))
}
fn fails(insts: &[Inst], needle: &str) {
    match certify(insts) {
        Ok(cert) => panic!("expected failure containing `{needle}`, certified: {cert}"),
        Err(e) => assert!(e.contains(needle), "`{needle}` not in: {e}"),
    }
}

#[test]
fn evidence_reports_checked_hazards_and_ledger_stats() {
    let e = ok(&[
        load(1),
        wait(0, 15),
        use_of(1),
        ds_write(1),
        wait(63, 0),
        barrier(),
        end(),
    ]);
    assert_eq!(e["checked"], true);
    assert_eq!(e["blocking"], 0);
    assert_eq!(e["findings"], 0);
    assert_eq!(e["bypasses"], 0);
    assert_eq!(e["arch"], "gfx942");
    assert_eq!(e["kernel"], "k");
    assert_eq!(e["instructions"], 7);
    assert_eq!(e["unreachable_instructions"], 0);
    assert_eq!(e["hazard"]["violations"], 0);
    assert_eq!(e["hazard"]["inserted_nops"], 0);
    assert_eq!(e["ledger"]["barriers"], 1);
    assert_eq!(e["ledger"]["waitcnt"], 2);
    assert_eq!(e["ledger"]["memory"]["vmem_load"], 1);
    assert_eq!(e["ledger"]["memory"]["ds_store"], 1);
    assert!(e["ledger"]["steps"].as_u64().unwrap() >= 7);
}

#[test]
fn missing_mfma_wait_states_fail_and_enough_pass() {
    // XDL (4 passes) write -> VALU read of the result needs 7 wait states (hazard table); an
    // adjacent read has none.
    fails(
        &[mfma(), read_mfma_result(), end()],
        "write VGPR -> VALU read/write",
    );
    // `s_nop 7` is 8 wait states.
    let e = ok(&[mfma(), nop(7), read_mfma_result(), end()]);
    assert_eq!(e["hazard"]["s_nop"], 1);
    assert_eq!(e["hazard"]["s_nop_wait_states"], 8);
    assert_eq!(e["hazard"]["mfma"], 1);
    // One wait state short of the requirement still fails (`s_nop 5` is 6).
    fails(
        &[mfma(), nop(5), read_mfma_result(), end()],
        "write VGPR -> VALU read/write",
    );
}

#[test]
fn certification_never_inserts_or_rewrites() {
    let insts = [mfma(), nop(7), read_mfma_result(), end()];
    let e = ok(&insts);
    assert_eq!(e["instructions"], insts.len());
    assert_eq!(e["hazard"]["rewritten_instructions"], 0);
}

#[test]
fn missing_vmem_wait_fails_and_the_wait_fixes_it() {
    fails(&[load(1), use_of(1), end()], "Vm wait");
    ok(&[load(1), wait(0, 15), use_of(1), end()]);
    // WAW: a pending load's destination needs its wait before it is overwritten.
    fails(
        &[load(1), ops::v1(Vop1Op::MovB32, v(1), v(0)), end()],
        "Vm wait",
    );
    ok(&[
        load(1),
        wait(0, 15),
        ops::v1(Vop1Op::MovB32, v(1), v(0)),
        end(),
    ]);
}

#[test]
fn same_vmem_read_class_waw_keeps_pending_results_across_widths_and_policies() {
    for (old_op, old_base, new_op, new_base, shared) in [
        (FlatOp::LoadDword, 1, FlatOp::LoadDwordx2, 0, 1),
        (FlatOp::LoadDwordx2, 0, FlatOp::LoadDword, 1, 1),
        (FlatOp::LoadDwordx4, 0, FlatOp::LoadDwordx2, 2, 2),
        (FlatOp::LoadUbyte, 1, FlatOp::LoadDword, 1, 1),
        (FlatOp::LoadDword, 1, FlatOp::LoadUshort, 1, 1),
    ] {
        for old_policy in 0..8 {
            for new_policy in 0..8 {
                let load_with = |op, base, policy| {
                    let mut i = ops::global_load(op, v(base), v(40), None, 0);
                    let Inst::Flat(f) = &mut i else {
                        unreachable!()
                    };
                    f.sc0 = policy & 1 != 0;
                    f.sc1 = policy & 2 != 0;
                    f.nt = policy & 4 != 0;
                    i
                };
                let older = load_with(old_op, old_base, old_policy);
                let younger = load_with(new_op, new_base, new_policy);
                ok(&[
                    older.clone(),
                    younger.clone(),
                    wait(0, 15),
                    use_of(shared),
                    end(),
                ]);
                // A partial wait retires only the older result, not the younger one.
                fails(
                    &[
                        older.clone(),
                        younger.clone(),
                        wait(1, 15),
                        use_of(shared),
                        end(),
                    ],
                    "Vm wait",
                );
                fails(&[older, younger, use_of(shared), end()], "Vm wait");
            }
        }
    }
}

#[test]
fn ordered_load_waw_is_not_an_exec_path_exemption() {
    // The conv-fuse pattern issues optional per-lane loads, switches EXEC, then
    // loads the same VGPR again. Ordering is proven without claiming disjoint
    // EXEC masks, dropping operations, or pruning a feasible branch.
    let pair = [
        ops::s1(Sop1Op::MovB64, Src::ExecLo, Src::Int(1)),
        load(1),
        ops::s1(Sop1Op::MovB64, Src::ExecLo, Src::Int(-1)),
        load(1),
    ];
    let mut p = pair.to_vec();
    p.extend([wait(0, 15), use_of(1), end()]);
    ok(&p);
    let mut p = pair.to_vec();
    p.extend([use_of(1), end()]);
    fails(&p, "Vm wait");
    // A genuine reachable WAW, including under a different EXEC, still refuses.
    fails(
        &[
            load(1),
            ops::s1(Sop1Op::MovB64, Src::ExecLo, Src::Int(1)),
            ops::v1(Vop1Op::MovB32, v(1), v(0)),
            end(),
        ],
        "Waw",
    );
    fails(&[load(1), ds_read(1), wait(0, 0), end()], "Waw");
    fails(&[ds_read(1), load(1), wait(0, 0), end()], "Waw");
}

#[test]
fn store_sources_are_not_counter_held_on_gfx942_but_counters_and_barriers_still_count() {
    let overwrite = |data: u8| ops::v1(Vop1Op::MovB32, v(data), v(0));
    // The CDNA3 ISA has no counter for "store data read" outside GWS (EXP_CNT): overwriting
    // a pending VMEM or DS store's data register needs no counter wait. hipcc output
    // (the rows16 anchor) does exactly this after a `ds_write2_b64`.
    let e = ok(&[store(1), overwrite(1), end()]);
    assert_eq!(e["ledger"]["memory"]["vmem_store"], 1);
    ok(&[ds_write(1), overwrite(1), end()]);
    ok(&[
        ops::ds_write2(DsOp::Write2B64, v(88), v(62), v(64), 0, 1),
        overwrite(65),
        end(),
    ]);
    // Still enforced: the store occupies its counter slot (a later load's wait position),
    // and a pending DS store blocks the barrier even though its data was overwritten.
    fails(
        &[ds_write(1), overwrite(1), barrier(), end()],
        "Lgkm wait 0 (Barrier)",
    );
    ok(&[ds_write(1), overwrite(1), wait(63, 0), barrier(), end()]);
    ok(&[store(5), load(1), wait(1, 15), end()]);
    fails(
        &[store(5), load(1), wait(1, 15), use_of(1), end()],
        "Vm wait",
    );
    // The ISA's wait-state rule for wide VMEM store data stays enforced by the hazard pass.
    let wide = || ops::global_store(FlatOp::StoreDwordx4, v(2), v(4), None, 0);
    fails(&[wide(), overwrite(5), end()], "> 64 bits of data");
    ok(&[wide(), nop(1), overwrite(5), end()]);
}

#[test]
fn nonzero_vmcnt_retires_by_position_in_one_in_order_stream() {
    // Loads and stores share VM_CNT in issue order: after `vmcnt(1)` only the youngest remains.
    ok(&[
        load(1),
        load(10),
        wait(1, 15),
        use_of(1),
        wait(0, 15),
        use_of(10),
        end(),
    ]);
    fails(
        &[load(1), load(10), wait(1, 15), use_of(10), end()],
        "Vm wait",
    );
    ok(&[load(1), store(5), wait(1, 15), use_of(1), end()]);
    // A wait with the no-wait maximum retires nothing.
    fails(&[load(1), wait(63, 15), use_of(1), end()], "Vm wait");
}

#[test]
fn missing_lgkm_wait_for_ds_fails_and_nonzero_lgkmcnt_retires_by_position() {
    fails(&[ds_read(1), use_of(1), end()], "Lgkm wait");
    ok(&[ds_read(1), wait(63, 0), use_of(1), end()]);
    ok(&[ds_read(1), ds_read(10), wait(63, 1), use_of(1), end()]);
    fails(
        &[ds_read(1), ds_read(10), wait(63, 1), use_of(10), end()],
        "Lgkm wait",
    );
}

#[test]
fn all_float_compares_and_ge_u64_certify_through_both_builder_encodings() {
    let compares: Vec<_> = VopcOp::ALL
        .iter()
        .copied()
        .filter(|op| op.float_compare().is_some())
        .chain(std::iter::once(VopcOp::CmpGeU64))
        .collect();
    assert_eq!(compares.len(), 65);
    for op in compares {
        for dst in [Src::VccLo, s(20)] {
            let inst = ops::vc(op, dst, v(0), v(4));
            assert_eq!(matches!(&inst, Inst::Vopc(_)), dst == Src::VccLo);
            assert_eq!(matches!(&inst, Inst::Vop3(_)), dst != Src::VccLo);
            let report = ok(&[inst, end()]);
            assert_eq!(report["checked"], true, "{op:?} -> {dst:?}");
            assert_eq!(report["findings"], 0, "{op:?} -> {dst:?}");
            assert_eq!(report["bypasses"], 0, "{op:?} -> {dst:?}");
        }
    }
}

#[test]
fn ds_or_return_and_no_return_keep_counter_and_barrier_obligations() {
    let atomic = |returns: bool| {
        Inst::Ds(Ds {
            op: if returns { DsOp::OrRtnB32 } else { DsOp::OrB32 },
            vdst: returns.then_some(v(10)),
            addr: Some(v(2)),
            data0: Some(v(3)),
            data1: None,
            offset0: 4,
            offset1: 0,
            gds: false,
        })
    };
    for returns in [false, true] {
        let report = ok(&[atomic(returns), wait(63, 0), barrier(), end()]);
        assert_eq!(
            report["ledger"]["memory"][if returns { "ds_load" } else { "ds_store" }],
            1
        );
        // Both forms modify LDS; a returnless atomic is not an ignorable no-op.
        fails(
            &[atomic(returns), barrier(), end()],
            "Lgkm wait 0 (Barrier)",
        );
        fails(
            &[atomic(returns), wait(0, 15), barrier(), end()],
            "Lgkm wait",
        );
        fails(
            &[atomic(returns), ds_read(20), wait(63, 1), barrier(), end()],
            "Barrier",
        );
    }
    fails(&[atomic(true), use_of(10), end()], "Lgkm wait");
    fails(&[atomic(true), load(10), end()], "Waw");
    ok(&[atomic(true), wait(63, 0), use_of(10), end()]);
    // A partial wait retires the older returned OR, but not the younger DS read.
    ok(&[atomic(true), ds_read(20), wait(63, 1), use_of(10), end()]);
    fails(
        &[atomic(true), ds_read(20), wait(63, 1), use_of(20), end()],
        "Lgkm wait",
    );
    // A wait on only the fallthrough cannot discharge the taken path's atomic result.
    let skip_wait = ops::sp(SoppOp::CbranchScc1, 1);
    fails(
        &[atomic(true), skip_wait, wait(63, 0), use_of(10), end()],
        "Lgkm wait",
    );
}

#[test]
fn barrier_requires_pending_ds_writes_and_reads_to_have_retired() {
    // LDS RAW: a pending DS write must retire before the workgroup is released.
    fails(&[ds_write(1), barrier(), end()], "Lgkm wait 0 (Barrier)");
    ok(&[ds_write(1), wait(63, 0), barrier(), end()]);
    // LDS WAR: a pending DS read (its result never used) too.
    fails(&[ds_read(1), barrier(), end()], "Lgkm wait 0 (Barrier)");
    ok(&[ds_read(1), wait(63, 0), barrier(), end()]);
    // A partial count that leaves a DS operation in flight does not release the barrier.
    fails(
        &[ds_write(1), ds_write(3), wait(63, 1), barrier(), end()],
        "Lgkm wait 0 (Barrier)",
    );
    ok(&[ds_write(1), ds_write(3), wait(63, 0), barrier(), end()]);
    // Global traffic is not LDS: a barrier with only VMEM in flight is not rejected by this rule.
    ok(&[store(1), barrier(), wait(0, 15), end()]);
}

#[test]
fn a_wait_on_only_one_branch_path_is_rejected() {
    // 0: load  1: s_cbranch_scc1 -> 3  2: s_waitcnt vmcnt(0)  3: use  4: end
    // The branch skips the wait (offset 1 dword over the one-dword wait).
    let skip = ops::sp(SoppOp::CbranchScc1, 1);
    fails(
        &[load(1), skip.clone(), wait(0, 15), use_of(1), end()],
        "Vm wait",
    );
    // The wait before the branch covers both paths.
    ok(&[load(1), wait(0, 15), skip, use_of(1), end()]);
}

#[test]
fn backward_branch_loops_are_walked_to_a_fixed_point() {
    // L: global_load v1; s_waitcnt vmcnt(0); use v1; s_cbranch_scc1 L; s_endpgm
    // load = 2 dwords, wait 1, use 1, branch 1: back to dword 0 from dword 5 -> -5.
    let back = ops::sp(SoppOp::CbranchScc1, (-5i16) as u16);
    let e = ok(&[load(1), wait(0, 15), use_of(1), back, end()]);
    assert_eq!(e["ledger"]["branches"], 1);
    assert_eq!(e["ledger"]["labels"], 1);
    // Without the wait the use of the load result fails on the first iteration.
    let back = ops::sp(SoppOp::CbranchScc1, (-4i16) as u16);
    fails(&[load(1), use_of(1), back, end()], "Vm wait");
}

#[test]
fn control_flow_and_effects_the_checkers_cannot_follow_fail_closed() {
    // Off an instruction boundary (into the middle of the 2-dword load).
    fails(
        &[ops::sp(SoppOp::Branch, 1), load(1), wait(0, 15), end()],
        "not an instruction boundary",
    );
    // Past the end of the code.
    fails(
        &[ops::sp(SoppOp::Branch, 5), end()],
        "not an instruction boundary",
    );
    // Fall-through off the end.
    fails(&[wait(0, 15)], "fall off the end");
    // Indirect branch.
    fails(&[ops::s1(Sop1Op::SetpcB64, s(0), s(2)), end()], "indirect");
    // Traps and messages are outside the ledger's model.
    fails(
        &[ops::sp(SoppOp::Trap, 1), end()],
        "outside the wait ledger's model",
    );
    fails(
        &[ops::sp(SoppOp::Sendmsg, 1), end()],
        "outside the wait ledger's model",
    );
    // FLAT-segment accesses count on two counters and may reach LDS.
    let mut flat = load(1);
    if let Inst::Flat(f) = &mut flat {
        f.seg = FlatSeg::Flat;
    }
    fails(&[flat, wait(0, 0), end()], "segment access");
}

fn a(n: u8) -> Src {
    Src::Agpr(n)
}
/// `global_load_dword aN, v[2:3], off`.
fn load_a(dst: u8) -> Inst {
    ops::global_load(FlatOp::LoadDword, a(dst), v(2), None, 0)
}

#[test]
fn agpr_vmem_load_is_ledger_tracked_like_a_vgpr_load() {
    // Drained: the wait retires the load and the AGPR read is accepted.
    ok(&[load_a(0), wait(0, 15), ops::acc_read(v(100), a(0)), end()]);
    // Missing wait: reading the pending AGPR destination is refused.
    fails(&[load_a(0), ops::acc_read(v(100), a(0)), end()], "Vm wait");
    // WAW on the same a0: overwriting a pending load destination needs the wait first.
    fails(&[load_a(0), ops::acc_write(a(0), v(0)), end()], "Vm wait");
    ok(&[load_a(0), wait(0, 15), ops::acc_write(a(0), v(0)), end()]);
    // A second load to the same a0 before the first drains is also a WAW.
    fails(&[load_a(0), load_a(0), wait(0, 15), end()], "Vm wait");
    // AGPR as memory store data is tracked: the store needs no counter wait for its data
    // (store sources are not counter-held) but its counter slot is still drained.
    ok(&[
        ops::global_store(FlatOp::StoreDword, v(2), a(0), None, 0),
        wait(0, 15),
        end(),
    ]);
}

#[test]
fn agpr_and_vgpr_banks_do_not_alias_in_the_ledger() {
    // A pending a0 load does not poison v0 (read or write) ...
    ok(&[
        load_a(0),
        use_of(0),
        ops::v1(Vop1Op::MovB32, v(0), v(1)),
        wait(0, 15),
        end(),
    ]);
    // ... and a pending v0 load does not poison a0.
    ok(&[
        load(0),
        ops::acc_read(v(100), a(0)),
        ops::acc_write(a(0), v(1)),
        wait(0, 15),
        end(),
    ]);
}

#[test]
fn accvgpr_moves_read_and_write_are_tracked() {
    // v_accvgpr_read_b32 reads its AGPR source: pending a1 load.
    fails(&[load_a(1), ops::acc_read(v(100), a(1)), end()], "Vm wait");
    // v_accvgpr_write_b32 reads its VGPR source: pending v1 load.
    fails(&[load(1), ops::acc_write(a(0), v(1)), end()], "Vm wait");
    // v_accvgpr_read_b32 writes its VGPR destination: pending v1 load (WAW).
    fails(&[load(1), ops::acc_read(v(1), a(0)), end()], "Vm wait");
    ok(&[load(1), wait(0, 15), ops::acc_read(v(1), a(0)), end()]);
}

#[test]
fn offsets_must_be_the_encoders() {
    let mut bad = stream(&[nop(0), nop(0), end()]);
    bad[1].dword_offset = 5;
    let e = certify_stream("k", &bad).unwrap_err();
    assert!(e.contains("instruction 1 is at dword 5"), "{e}");
    assert!(certify_stream("k", &[])
        .unwrap_err()
        .contains("no instructions"));
}

#[test]
fn unreachable_code_is_counted_not_hidden() {
    // `s_branch` over a nop to the end: the nop can only be reached by falling through the
    // unconditional branch, so it is unreachable.
    let e = ok(&[ops::sp(SoppOp::Branch, 1), nop(0), end()]);
    assert_eq!(e["unreachable_instructions"], 1);
}

// ---------------------------------------------------------------------------------------
// Block-local scalar branch proofs (pm/LEDGER-PATH-PROOF.md sections 3 and 6).
//
// Every program below is a typed stream. A *positive* proof test places the only hazard on
// an edge that a block-local scalar constant makes impossible; it must certify. A *negative*
// test makes the same hazardous edge feasible (a constant removed, a register clobbered, a
// block boundary, an unsupported writer, an unknown EXEC/VCC condition, ...); the unchanged
// strict ledger must still refuse it with the original wait diagnosis. A wrongly pruned
// edge would turn a negative into a certification, so each negative names the wrong belief
// it guards against.
// ---------------------------------------------------------------------------------------

fn dw(insts: &[Inst]) -> u16 {
    insts
        .iter()
        .map(|i| i.words(A).expect("encodes").len() as u16)
        .sum()
}
/// A conditional branch over exactly `over` (forward, to the instruction after it).
fn br(op: SoppOp, over: &[Inst]) -> Inst {
    ops::sp(op, dw(over))
}
fn flat_load_segment() -> Inst {
    let mut flat = load(1);
    if let Inst::Flat(f) = &mut flat {
        f.seg = FlatSeg::Flat;
    }
    flat
}
fn mov32(dst: u8, src: Src) -> Inst {
    ops::s1(Sop1Op::MovB32, s(dst), src)
}
fn mov_pair_zero() -> Inst {
    ops::s1(Sop1Op::MovB64, s(0), Src::Int(0))
}
fn and_exec() -> Inst {
    ops::s2(Sop2Op::AndB64, s(0), s(0), Src::ExecLo)
}
fn cselect_1_0() -> Inst {
    ops::s2(Sop2Op::CselectB32, s(0), Src::Int(1), Src::Int(0))
}
fn cmp_lg(a: Src, b: Src) -> Inst {
    ops::sc(SopcOp::CmpLgU32, a, b)
}
fn cmp_eq(a: Src, b: Src) -> Inst {
    ops::sc(SopcOp::CmpEqU32, a, b)
}
/// An unsupported SCC writer with unknown inputs.
fn scc_clobber() -> Inst {
    ops::s2(Sop2Op::AddU32, s(5), s(6), s(7))
}
/// Wait states a VALU-to-SGPR writer needs before a scalar consumer in these streams.
fn pad() -> Inst {
    nop(7)
}

/// Known SCC from a block-local constant: `s0 = 0; s_cmp_{eq,lg}_u32 s0, 0`.
fn scc(value: bool) -> Vec<Inst> {
    let cmp = if value {
        cmp_eq(s(0), Src::Int(0))
    } else {
        cmp_lg(s(0), Src::Int(0))
    };
    vec![mov32(0, Src::Int(0)), cmp]
}

/// A pending load of v1, `guard`, then `op` over a hazardous block on the *fall-through*
/// edge (a VALU overwrite of v1 before its load retires); the target is `s_endpgm`.
fn guarded_fall(guard: &[Inst], op: SoppOp) -> Vec<Inst> {
    let hazard = [ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15)];
    let mut p = vec![load(1)];
    p.extend_from_slice(guard);
    p.push(br(op, &hazard));
    p.extend(hazard);
    p.push(end());
    p
}
/// The same hazard, but on the branch *target*; the fall-through is `s_endpgm`.
fn guarded_target(guard: &[Inst], op: SoppOp) -> Vec<Inst> {
    let safe = [end()];
    let mut p = vec![load(1)];
    p.extend_from_slice(guard);
    p.push(br(op, &safe));
    p.extend(safe);
    p.extend([ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15), end()]);
    p
}

/// The exact pc889-893 shape (`L867` of the P1b wide64 anchor) with optional instructions in
/// four slots; a hazardous VALU overwrite is on the fall-through of the final
/// `s_cbranch_scc1`. With no insertion the branch is taken for every EXEC value.
fn l867(init: &[Inst], slots: [&[Inst]; 4]) -> Vec<Inst> {
    let mut guard = init.to_vec();
    guard.extend_from_slice(slots[0]);
    guard.push(and_exec());
    guard.extend_from_slice(slots[1]);
    guard.push(cselect_1_0());
    guard.extend_from_slice(slots[2]);
    guard.push(cmp_lg(s(0), Src::Int(1)));
    guard.extend_from_slice(slots[3]);
    guarded_fall(&guard, SoppOp::CbranchScc1)
}
const NONE: &[Inst] = &[];

#[test]
fn exact_l867_zero_mask_sequence_discharges_the_impossible_fallthrough() {
    // Pending VMEM load on the predecessor path; the conflicting VALU write exists only on
    // the fall-through of `s_cbranch_scc1`, which is taken for every possible EXEC.
    let e = ok(&l867(&[mov_pair_zero()], [NONE; 4]));
    assert_eq!(e["checked"], true);
    assert_eq!(e["blocking"], 0);
    assert_eq!(e["findings"], 0);
    assert_eq!(e["hazard"]["rewritten_instructions"], 0);
    assert_eq!(e["hazard"]["inserted_nops"], 0);
    assert_eq!(e["ledger"]["branches"], 1);
    // Syntactic reachability is unchanged by semantic pruning (documented meaning).
    assert_eq!(e["unreachable_instructions"], 0);
    // Same wait-free program with the proof removed is the baseline refusal.
    fails(&guarded_fall(&[], SoppOp::CbranchScc1), "Vm wait");
}

#[test]
fn zero_pair_through_overlapping_halves_and_width_variants_discharges() {
    // Alias/width handling, not a five-line recogniser: the pair is zero by two b32 movs;
    // by an all-ones pair overwritten half by half; by a b64 overwrite of a b64 -1.
    ok(&l867(
        &[mov32(0, Src::Int(0)), mov32(1, Src::Int(0))],
        [NONE; 4],
    ));
    ok(&l867(
        &[
            ops::s1(Sop1Op::MovB64, s(0), Src::Int(-1)),
            mov32(0, Src::Int(0)),
            mov32(1, Src::Int(0)),
        ],
        [NONE; 4],
    ));
    ok(&l867(
        &[ops::s1(Sop1Op::MovB64, s(0), Src::Int(-1)), mov_pair_zero()],
        [NONE; 4],
    ));
    // The 32-bit mask form: s0 = 0; s_and_b32 s0, s0, exec_lo; cselect; cmp; branch.
    let guard = [
        mov32(0, Src::Int(0)),
        ops::s2(Sop2Op::AndB32, s(0), s(0), Src::ExecLo),
        cselect_1_0(),
        cmp_lg(s(0), Src::Int(1)),
    ];
    ok(&guarded_fall(&guard, SoppOp::CbranchScc1));
    // Instructions that cannot touch s0/s1/SCC leave the proof intact; `s_mov` never
    // writes SCC.
    let harmless = [nop(0), mov32(5, s(9))];
    ok(&l867(
        &[mov_pair_zero()],
        [
            &harmless[..1],
            &harmless[..1],
            &harmless[..1],
            &harmless[1..],
        ],
    ));
}

#[test]
fn both_branch_polarities_prune_only_the_infeasible_edge() {
    for scc_value in [true, false] {
        for op in [SoppOp::CbranchScc0, SoppOp::CbranchScc1] {
            let taken = (op == SoppOp::CbranchScc1) == scc_value;
            let guard = scc(scc_value);
            // Hazard only on the impossible edge: certifies.
            let impossible = if taken {
                guarded_fall(&guard, op)
            } else {
                guarded_target(&guard, op)
            };
            ok(&impossible);
            // Swap the hazard onto the feasible edge: still refused.
            let feasible = if taken {
                guarded_target(&guard, op)
            } else {
                guarded_fall(&guard, op)
            };
            fails(&feasible, "Vm wait");
        }
    }
}

#[test]
fn identical_cselect_alternatives_keep_a_constant_but_different_ones_do_not() {
    // Unknown SCC (`s_cmp_lg_u32 s9, 0`): both alternatives are 0, so s0 is 0 either way.
    let same = [
        cmp_lg(s(9), Src::Int(0)),
        ops::s2(Sop2Op::CselectB32, s(0), Src::Int(0), Src::Int(0)),
        cmp_lg(s(0), Src::Int(0)),
    ];
    ok(&guarded_fall(&same, SoppOp::CbranchScc0));
    // Alternatives 1/0 under unknown SCC: s0 may be 1, the fall-through is feasible.
    let differ = [
        mov32(0, Src::Int(0)),
        cmp_lg(s(9), Src::Int(0)),
        cselect_1_0(),
        cmp_lg(s(0), Src::Int(0)),
    ];
    fails(&guarded_fall(&differ, SoppOp::CbranchScc0), "Vm wait");
}

#[test]
fn a_direct_entry_to_the_guarded_block_and_the_guarded_block_both_verify() {
    // First conditional branch (SCC unknown at entry) enters the overwrite directly,
    // mirroring L690 -> L872; the first load and the zero-mask guard are on the other path,
    // whose fall-through into the overwrite is impossible.
    let second = [ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15)];
    let mut guard = vec![load(1), mov_pair_zero(), and_exec(), cselect_1_0()];
    guard.push(cmp_lg(s(0), Src::Int(1)));
    guard.push(br(SoppOp::CbranchScc1, &second));
    let mut p = vec![br(SoppOp::CbranchScc1, &guard)];
    p.extend(guard.clone());
    p.extend(second.clone());
    p.push(end());
    ok(&p);
    // A load already in flight on the direct path (hoisted before the first branch) is a
    // reachable WAW at the VALU write: the consumer block is not deleted by the proof.
    let mut q = vec![load(1), br(SoppOp::CbranchScc1, &guard[1..])];
    q.extend(guard[1..].iter().cloned());
    q.extend(second);
    q.push(end());
    fails(&q, "Vm wait");
}

#[test]
fn a_branch_into_the_middle_of_the_proof_sequence_resets_the_facts() {
    // Skipping the zero MOV: the AND block starts at the branch target, whose scalar facts
    // are unknown. The same text sequence without the side entry certifies.
    let proof = |entry_over: Option<usize>| {
        let body = [and_exec(), cselect_1_0(), cmp_lg(s(0), Src::Int(1))];
        let hazard = [ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15)];
        let mut p = vec![load(1)];
        let head = [
            mov_pair_zero(),
            body[0].clone(),
            body[1].clone(),
            body[2].clone(),
        ];
        if let Some(over) = entry_over {
            // Branch (unknown SCC) over `head[..over]` into `head[over..]`.
            p.push(br(SoppOp::CbranchScc1, &head[..over]));
        }
        p.extend(head);
        p.push(br(SoppOp::CbranchScc1, &hazard));
        p.extend(hazard);
        p.push(end());
        p
    };
    ok(&proof(None));
    // Target at the first instruction of the block (offset 0): no bypass, still proven.
    ok(&proof(Some(0)));
    // Targets at the AND, the CSELECT and the CMP: each is a fresh block with no facts.
    fails(&proof(Some(1)), "Vm wait");
    fails(&proof(Some(2)), "Vm wait");
    fails(&proof(Some(3)), "Vm wait");
}

#[test]
fn facts_cross_exactly_one_edge_into_the_block_after_a_conditional_branch() {
    // s0 = 0 in the predecessor block; `s_cbranch_execz 0` makes the next instruction a branch
    // target with two in-edges, both leaving that block with s0 = 0. Over either edge the
    // compare decides its `scc0` (previously the block-local table kept both edges and this
    // refused): the hazardous fall-through does not exist.
    let hazard = [ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15)];
    let through = |hops: usize| {
        let mut p = vec![load(1), mov32(0, Src::Int(0))];
        p.extend((0..hops).map(|_| ops::sp(SoppOp::CbranchExecz, 0)));
        p.extend([cmp_lg(s(0), Src::Int(0)), br(SoppOp::CbranchScc0, &hazard)]);
        p.extend(hazard.iter().cloned());
        p.push(end());
        p
    };
    ok(&through(1));
    // One more block between the constant and the compare: a predecessor's facts are carried
    // across one edge only, so the second hop knows nothing and keeps both edges.
    fails(&through(2), "Vm wait");
}

#[test]
fn a_loop_that_changes_the_predicate_operand_keeps_no_facts_across_the_back_edge() {
    // load v1; s0 = 0; L: cmp s0,0; scc0 -> Lbody; [overwrite v1; wait]; Lbody: s0 += 1;
    // scc1 -> L; end. The first iteration skips the overwrite; the second one (s0 = 1)
    // falls into it while the first load is still pending.
    let hazard = [ops::v1(Vop1Op::MovB32, v(1), v(0)), wait(0, 15)];
    let head = [cmp_lg(s(0), Src::Int(0)), br(SoppOp::CbranchScc0, &hazard)];
    let inc = ops::s2(Sop2Op::AddU32, s(0), s(0), Src::Int(1));
    let loop_len = dw(&head) + dw(&hazard) + dw(std::slice::from_ref(&inc)) + 1;
    let back = ops::sp(SoppOp::CbranchScc1, loop_len.wrapping_neg());
    let mut p = vec![load(1), mov32(0, Src::Int(0))];
    p.extend(head);
    p.extend(hazard);
    p.push(inc);
    p.push(back);
    p.push(end());
    fails(&p, "Vm wait");
}

#[test]
fn removing_zeroing_or_clobbering_the_pair_reopens_the_impossible_edge() {
    // No zero MOV at all.
    fails(&l867(NONE, [NONE; 4]), "Vm wait");
    // Zero replaced by an unknown register.
    fails(
        &l867(&[ops::s1(Sop1Op::MovB64, s(0), s(8))], [NONE; 4]),
        "Vm wait",
    );
    // A real clobber of either half (or a vector-to-SGPR write) between MOV and AND.
    let lo = [mov32(0, s(9))];
    let hi = [mov32(1, s(9))];
    let readfirstlane = [ops::v1(Vop1Op::ReadfirstlaneB32, s(1), v(5)), pad()];
    let vcmp = [ops::vc(VopcOp::CmpEqU32, s(0), v(0), v(1)), pad()];
    for slot0 in [&lo[..], &hi[..], &readfirstlane[..], &vcmp[..]] {
        fails(
            &l867(&[mov_pair_zero()], [slot0, NONE, NONE, NONE]),
            "Vm wait",
        );
    }
    // High half overwritten by a constant that is not provably zero under the mask.
    fails(
        &l867(
            &[mov_pair_zero()],
            [&[mov32(1, Src::Int(1))][..], NONE, NONE, NONE],
        ),
        "Vm wait",
    );
}

#[test]
fn unsupported_writers_between_the_steps_of_the_certificate_forget_it() {
    let add_s0 = [ops::s2(Sop2Op::AddU32, s(0), s(6), s(7))];
    let lo = [mov32(0, s(9))];
    let rfl = [ops::v1(Vop1Op::ReadfirstlaneB32, s(0), v(5)), pad()];
    // Between CSELECT and CMP: an SGPR writer (modelled or not) with unknown inputs.
    for slot2 in [&add_s0[..], &lo[..], &rfl[..]] {
        fails(
            &l867(&[mov_pair_zero()], [NONE, NONE, slot2, NONE]),
            "Vm wait",
        );
    }
    // Between AND and CSELECT, and between CMP and the branch: an SCC writer the domain does
    // not model (or models with unknown inputs).
    let scc_writers = [
        scc_clobber(),
        ops::s2(Sop2Op::LshlB32, s(5), s(5), Src::Int(1)),
        ops::s1(Sop1Op::NotB32, s(5), s(5)),
        cmp_lg(s(9), Src::Int(3)),
    ];
    for w in &scc_writers {
        let one = std::slice::from_ref(w);
        fails(
            &l867(&[mov_pair_zero()], [NONE, one, NONE, NONE]),
            "Vm wait",
        );
        fails(
            &l867(&[mov_pair_zero()], [NONE, NONE, NONE, one]),
            "Vm wait",
        );
    }
}

#[test]
fn conditional_partial_and_overlapping_scalar_writes_keep_both_edges() {
    // All compare the believed-zero low dword and take `scc0` (taken if the value were 0).
    let believed_zero = |guard: &[Inst], reg: u8| {
        let mut g = guard.to_vec();
        g.push(cmp_lg(s(reg), Src::Int(0)));
        guarded_fall(&g, SoppOp::CbranchScc0)
    };
    // s_cmov under an unknown SCC may or may not write.
    fails(
        &believed_zero(
            &[
                mov32(0, Src::Int(0)),
                cmp_lg(s(9), Src::Int(0)),
                ops::s1(Sop1Op::CmovB32, s(0), Src::Int(1)),
            ],
            0,
        ),
        "Vm wait",
    );
    // 64-bit conditional overwrite of the pair: the low dword may become 1.
    fails(
        &believed_zero(
            &[
                mov_pair_zero(),
                cmp_lg(s(9), Src::Int(0)),
                ops::s1(Sop1Op::CmovB64, s(0), Src::Int(1)),
            ],
            0,
        ),
        "Vm wait",
    );
    // High-half overwrite: the compare reads s1.
    fails(
        &believed_zero(&[mov_pair_zero(), mov32(1, s(9))], 1),
        "Vm wait",
    );
    // Destination overlapping a source with an unknown operand.
    fails(
        &believed_zero(
            &[mov_pair_zero(), ops::s2(Sop2Op::OrB64, s(0), s(0), s(8))],
            0,
        ),
        "Vm wait",
    );
    // Different CSELECT alternatives under unknown SCC, and an unknown source.
    fails(
        &believed_zero(
            &[
                mov32(0, Src::Int(0)),
                cmp_lg(s(9), Src::Int(0)),
                cselect_1_0(),
            ],
            0,
        ),
        "Vm wait",
    );
    fails(&believed_zero(&[mov32(0, s(9))], 0), "Vm wait");
    fails(&believed_zero(&[], 0), "Vm wait");
}

#[test]
fn vector_to_sgpr_writes_and_async_scalar_loads_do_not_leave_stale_constants() {
    let believed_zero = |guard: &[Inst]| {
        let mut g = guard.to_vec();
        g.push(cmp_lg(s(0), Src::Int(0)));
        guarded_fall(&g, SoppOp::CbranchScc0)
    };
    fails(
        &believed_zero(&[
            mov32(0, Src::Int(0)),
            ops::v1(Vop1Op::ReadfirstlaneB32, s(0), v(5)),
            pad(),
        ]),
        "Vm wait",
    );
    fails(
        &believed_zero(&[
            mov_pair_zero(),
            ops::vc(VopcOp::CmpEqU32, s(0), v(0), v(1)),
            pad(),
        ]),
        "Vm wait",
    );
    // No value is learned at a wait: s0 is the loaded word, not the earlier constant.
    fails(
        &believed_zero(&[
            mov32(0, Src::Int(0)),
            ops::s_load(SmemOp::LoadDword, s(0), s(2), 0),
            wait(63, 0),
        ]),
        "Vm wait",
    );
}

#[test]
fn ambiguous_raw_64_bit_literals_are_not_trusted_in_either_extension() {
    // A raw 32-bit literal in a 64-bit SALU source: zero- vs sign-extension is unverified
    // ([UNVERIFIED] in the encoder), so the high dword is unknown. 0x8000_0000 extends to
    // high 0 (zero-extend) or 0xffff_ffff (sign-extend); one program per wrong belief.
    let lit = Src::Lit(0x8000_0000);
    let guard = |cmp: Inst| vec![ops::s1(Sop1Op::MovB64, s(0), lit), mov32(2, s(1)), cmp];
    // Believing zero-extension: s2 == 0 always, `scc1` always taken.
    let zext = guard(cmp_eq(s(2), Src::Int(0)));
    fails(&guarded_fall(&zext, SoppOp::CbranchScc1), "Vm wait");
    // Believing sign-extension: s2 != 0 always, so SCC = 1 and `scc0` is never taken.
    let sext = guard(cmp_lg(s(2), Src::Int(0)));
    fails(&guarded_target(&sext, SoppOp::CbranchScc0), "Vm wait");
    // And the two beliefs are mutually exclusive, so neither fall-through nor target can be
    // pruned in the mirrored programs either.
    fails(&guarded_target(&zext, SoppOp::CbranchScc1), "Vm wait");
    fails(&guarded_fall(&sext, SoppOp::CbranchScc0), "Vm wait");
}

#[test]
fn unknown_exec_and_vcc_branches_retain_both_edges_even_for_a_zero_mask() {
    for op in [
        SoppOp::CbranchExecz,
        SoppOp::CbranchExecnz,
        SoppOp::CbranchVccz,
        SoppOp::CbranchVccnz,
    ] {
        fails(&guarded_fall(&[], op), "Vm wait");
        fails(&guarded_target(&[], op), "Vm wait");
        // A known-zero scalar mask (SCC = 0) says nothing about EXEC/VCC.
        let zero_mask = [mov_pair_zero(), and_exec()];
        fails(&guarded_fall(&zero_mask, op), "Vm wait");
        fails(&guarded_target(&zero_mask, op), "Vm wait");
    }
    // Writing exec/vcc to a constant does not create an EXEC=0 shortcut either: there is
    // no counter retirement for an empty wave.
    let exec0 = [ops::s1(Sop1Op::MovB64, Src::ExecLo, Src::Int(0))];
    let vcc0 = [ops::s1(Sop1Op::MovB64, Src::VccLo, Src::Int(0))];
    fails(&guarded_fall(&exec0, SoppOp::CbranchExecz), "Vm wait");
    fails(&guarded_target(&exec0, SoppOp::CbranchExecnz), "Vm wait");
    fails(&guarded_fall(&vcc0, SoppOp::CbranchVccz), "Vm wait");
    fails(&guarded_target(&vcc0, SoppOp::CbranchVccnz), "Vm wait");
}

/// Every reachable hazard class must survive a decided branch that keeps its path.
fn hazard_matrix() -> Vec<(&'static str, Vec<Inst>, Vec<Inst>, &'static str)> {
    let smem = |dst: u8| ops::s_load(SmemOp::LoadDword, s(dst), s(2), 0);
    vec![
        (
            "load -> DS load",
            vec![load(1)],
            vec![ds_read(1)],
            "Vm wait",
        ),
        (
            "load -> VALU overwrite",
            vec![load(1)],
            vec![ops::v1(Vop1Op::MovB32, v(1), v(0))],
            "Vm wait",
        ),
        ("load -> use", vec![load(1)], vec![use_of(1)], "Vm wait"),
        (
            "vmcnt(max) is no wait",
            vec![load(1)],
            vec![wait(63, 15), use_of(1)],
            "Vm wait",
        ),
        (
            "partial vmcnt keeps the older load",
            vec![load(1), load(10)],
            vec![wait(1, 15), use_of(10)],
            "Vm wait",
        ),
        (
            "DS load -> overwrite",
            vec![ds_read(1)],
            vec![ops::v1(Vop1Op::MovB32, v(1), v(0))],
            "Lgkm wait",
        ),
        (
            "DS load -> use",
            vec![ds_read(1)],
            vec![use_of(1)],
            "Lgkm wait",
        ),
        (
            "partial lgkmcnt keeps the older DS read",
            vec![ds_read(1), ds_read(10)],
            vec![wait(63, 1), use_of(10)],
            "Lgkm wait",
        ),
        (
            "SMEM load -> use",
            vec![smem(4)],
            vec![mov32(5, s(4))],
            "Lgkm wait",
        ),
        (
            "SMEM load -> overwrite",
            vec![smem(4)],
            vec![mov32(4, Src::Int(0))],
            "Lgkm wait",
        ),
        (
            "SMEM then DS: nonzero lgkmcnt retires nothing",
            vec![smem(4), ds_read(1)],
            vec![wait(63, 1), use_of(1)],
            "Lgkm wait",
        ),
        (
            "DS then SMEM: nonzero lgkmcnt retires nothing",
            vec![ds_read(1), smem(4)],
            vec![wait(63, 1), mov32(5, s(4))],
            "Lgkm wait",
        ),
        (
            "AGPR load -> overwrite (same bank)",
            vec![load_a(0)],
            vec![ops::acc_write(a(0), v(0))],
            "Vm wait",
        ),
        (
            "AGPR load -> read (same bank)",
            vec![load_a(0)],
            vec![ops::acc_read(v(100), a(0))],
            "Vm wait",
        ),
        (
            "pending DS write at a barrier",
            vec![ds_write(1)],
            vec![barrier()],
            "Lgkm wait 0 (Barrier)",
        ),
        (
            "partial lgkmcnt at a barrier",
            vec![ds_write(1), ds_write(3)],
            vec![wait(63, 1), barrier()],
            "Lgkm wait 0 (Barrier)",
        ),
    ]
}

#[test]
fn reachable_memory_hazards_are_refused_straight_line_and_behind_decided_branches() {
    for (what, first, second, needle) in hazard_matrix() {
        let straight: Vec<Inst> = first
            .iter()
            .chain(&second)
            .cloned()
            .chain([end()])
            .collect();
        let ctx = |label: &str, prog: &[Inst]| match certify(prog) {
            Ok(c) => panic!("{what} ({label}): expected `{needle}`, certified: {c}"),
            Err(e) => assert!(
                e.contains(needle),
                "{what} ({label}): `{needle}` not in: {e}"
            ),
        };
        ctx("straight", &straight);
        // Provably taken branch whose *target* holds the hazard: the safe fall-through is
        // pruned, the hazard path is not.
        let mut taken = first.clone();
        taken.extend(scc(true));
        taken.push(br(SoppOp::CbranchScc1, &[end()]));
        taken.push(end());
        taken.extend(second.clone());
        taken.push(end());
        ctx("decided taken", &taken);
        // Provably not-taken branch whose *fall-through* holds the hazard.
        let mut fall = first.clone();
        fall.extend(scc(true));
        fall.push(br(SoppOp::CbranchScc0, &second));
        fall.extend(second.clone());
        fall.push(end());
        ctx("decided fall-through", &fall);
        // Unknown condition: both edges.
        let mut unknown = first.clone();
        unknown.push(br(SoppOp::CbranchScc1, &second));
        unknown.extend(second.clone());
        unknown.push(end());
        ctx("unknown", &unknown);
    }
}

#[test]
fn the_hazard_matrix_certifies_once_the_missing_wait_is_present() {
    // Positive control for the matrix: the drained versions are accepted, so every refusal
    // above is caused by exactly the missing wait/barrier drain.
    ok(&[load(1), load(10), wait(0, 15), use_of(10), end()]);
    ok(&[ds_read(1), ds_read(10), wait(63, 0), use_of(10), end()]);
    ok(&[
        ops::s_load(SmemOp::LoadDword, s(4), s(2), 0),
        wait(63, 0),
        mov32(5, s(4)),
        end(),
    ]);
    ok(&[
        ops::s_load(SmemOp::LoadDword, s(4), s(2), 0),
        ds_read(1),
        wait(63, 0),
        use_of(1),
        end(),
    ]);
    ok(&[ds_write(1), ds_write(3), wait(63, 0), barrier(), end()]);
}

#[test]
fn malformed_waits_are_refused_before_any_proof_can_use_them() {
    // `s_waitcnt` with unused bits set, or counts above the architecture maxima, never
    // reaches the ledger as an encodable instruction.
    assert!(ops::sp(SoppOp::Waitcnt, 0x3080).words(A).is_err());
    assert!(ops::sp(SoppOp::Waitcnt, 0x0080).words(A).is_err());
    assert!(waitcnt_imm(A, 64, 7, 15).is_err());
    assert!(waitcnt_imm(A, 63, 8, 15).is_err());
    assert!(waitcnt_imm(A, 63, 7, 16).is_err());
    // A barrier with a nonzero SIMM16 cannot be encoded either.
    assert!(ops::sp(SoppOp::Barrier, 1).words(A).is_err());
}

#[test]
fn unsupported_and_indirect_instructions_on_an_impossible_edge_are_still_refused() {
    // The proof only chooses an edge; every instruction is still converted and validated.
    let taken_over = |bad: Inst| {
        let mut p = scc(true);
        p.push(br(SoppOp::CbranchScc1, std::slice::from_ref(&bad)));
        p.push(bad);
        p.push(end());
        p
    };
    fails(
        &taken_over(ops::s1(Sop1Op::SetpcB64, s(0), s(2))),
        "indirect",
    );
    fails(
        &taken_over(ops::sp(SoppOp::Trap, 1)),
        "outside the wait ledger's model",
    );
    fails(
        &taken_over(ops::sp(SoppOp::Sendmsg, 1)),
        "outside the wait ledger's model",
    );
    fails(&taken_over(flat_load_segment()), "segment access");
    // A branch target off an instruction boundary is rejected even when the edge is
    // provably dead.
    let mut p = scc(false);
    p.push(ops::sp(SoppOp::CbranchScc1, 1));
    p.extend([load(1), wait(0, 15), end()]);
    fails(&p, "not an instruction boundary");
}

#[test]
fn an_async_scalar_destination_is_not_pruned_past_a_hazardous_constant_write() {
    // s_load s0 (async); s_mov s0, 0 is a WAW while it is pending; the branch below would
    // be provable from that constant. The earlier WAW is reported, not pruned past.
    let hazard = [end()];
    let p = [
        ops::s_load(SmemOp::LoadDword, s(0), s(2), 0),
        mov32(0, Src::Int(0)),
        cmp_lg(s(0), Src::Int(0)),
        br(SoppOp::CbranchScc0, &hazard),
        end(),
        end(),
    ];
    fails(&p, "Lgkm wait");
    let p64 = [
        ops::s_load(SmemOp::LoadDwordx2, s(0), s(2), 0),
        mov_pair_zero(),
        and_exec(),
        cselect_1_0(),
        cmp_lg(s(0), Src::Int(1)),
        br(SoppOp::CbranchScc1, &hazard),
        end(),
        end(),
    ];
    fails(&p64, "Lgkm wait");
    // With the wait in place the same program is hazard-free and certifies.
    let waited = [
        ops::s_load(SmemOp::LoadDword, s(0), s(2), 0),
        wait(63, 0),
        mov32(0, Src::Int(0)),
        cmp_lg(s(0), Src::Int(0)),
        br(SoppOp::CbranchScc0, &hazard),
        end(),
        end(),
    ];
    ok(&waited);
}

/// A gfx1200 text `Program` as the unadapted public entry points see it: the L867-shaped
/// scalar sequence is only text, and the second load is supported dispatcher body text.
fn l867_text_program() -> Program {
    let v1 = RegRef {
        kind: Kind::V,
        base: 1,
        len: 1,
    };
    let load = || {
        Text::new("global_load_b32 v1, v0, s[0:1]", vec![v1], vec![]).memory(MemoryClass::VmemLoad)
    };
    let ctl = |t: &str| Text::new(t, vec![], vec![]);
    Program {
        arch: IsaArch::Gfx1200,
        instructions: vec![
            load(),
            ctl("s_mov_b64 s[0:1], 0"),
            ctl("s_and_b64 s[0:1], s[0:1], exec"),
            ctl("s_cselect_b32 s0, 1, 0"),
            ctl("s_cmp_lg_u32 s0, 1"),
            ctl("s_cbranch_scc1 Lend"),
            load(),
            ctl("s_wait_loadcnt 0x0"),
            ctl("Lend:"),
            ctl("s_endpgm"),
        ],
    }
}

#[test]
fn unadapted_public_entries_and_dispatcher_mode_take_no_scalar_branch_proofs() {
    let mut p = l867_text_program();
    // `verify` with no typed control adapter is all-Both: the impossible fall-through is
    // still explored and the WAW is found.
    let e = verify(&p, &[]).unwrap_err();
    assert!(e.contains("Load wait"), "{e}");
    // Dispatcher mode over the same program: the second load is declared body text, its
    // hazard is counted (suppressed) because the fall-through was still walked, i.e. the
    // dispatcher walk consumed no strict-only scalar assumption.
    let stats = verify_dispatcher(&p, &[6..8]).unwrap();
    assert!(stats.suppressed >= 1, "{stats:?}");
    // The same hazard one pc outside the declared body span stays fatal.
    assert!(verify_dispatcher(&p, &[7..8]).is_err());
    // gfx942 still has no foreign-text front end. Keep that refusal rather than adding
    // dispatcher support or treating its body as if strict typed assumptions applied.
    p.arch = IsaArch::Gfx942;
    let error = verify_dispatcher(&p, &[6..8]).unwrap_err();
    assert!(
        error.contains("foreign text has no gfx942 front end"),
        "{error}"
    );
}
