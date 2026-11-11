// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cdna::lifted::certify_stream` (the typed-stream core of `certify`): hazard check plus CFG
//! wait-ledger replay over decoded gfx942 instructions. Streams are built in memory from
//! typed instructions, so no compiled anchor is needed. Each negative case removes exactly
//! one thing a positive case has (a wait state, a wait, a barrier drain) and must fail.

use peacemaker_cdna::enc::{
    ds::DsOp,
    flat::{FlatOp, FlatSeg},
    sop::{waitcnt_imm, Sop1Op, SoppOp},
    vop::Vop1Op,
    vop3p::MaiOp,
};
use peacemaker_cdna::{Arch, Inst, Src};
use peacemaker_isa::cdna::lifted::certify_stream;
use peacemaker_isa::cdna::ops;
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
