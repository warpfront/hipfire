// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 global memory model and task-body region ABI, on the CPU:
//!
//! * the builder (emission ledger) refuses a missing release (a `Relaxed` event
//!   signal while global writes are pending a release, a loop back edge carrying
//!   them to a head that does not), a missing acquire (a load after a relaxed
//!   observation), raw atomics/writebacks, and every region-ABI violation;
//! * the independent replay (`sync::verify`) accepts the emitted handoff and
//!   dispatcher and rejects them once a release wait, the system writeback or
//!   the acquire invalidate is removed from the instruction stream;
//! * the MQ2 body emitted as a region inside the persistent dispatcher is the
//!   standalone shipped body's instruction stream (labels renamed, `ttmp9` ->
//!   the virtual-workgroup input), and both produce the same output bytes on the
//!   PeaceMaker emulator.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::insn::{Instruction, MemoryClass, Program};
use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201 as mq2;
use peacemaker_isa::kernels::persistent_dispatch_gfx1201 as pd;
use peacemaker_isa::plan::RegionAbi;
use peacemaker_isa::reg::{Kind, Live, RegRef};
use peacemaker_isa::sync::{self, AtomicOp, GlobalAddr, Ordering, Scope, SyncKind, SyncProof};
use peacemaker_isa::{native, Arch, Builder, KernargLayout, KernelSpec, RegPlan, S, V};

fn spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "sync_test".into(),
        variant: "t".into(),
        arch: Arch::Gfx1201,
        symbol: "sync_test".into(),
        kernargs: KernargLayout::new(16),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}
fn builder() -> Builder {
    let mut regs = RegPlan::new(32, 32).unwrap();
    for (n, b, l) in [
        ("s01", 0u8, 2u8),
        ("s45", 4, 2),
        ("s67", 6, 2),
        ("s8", 8, 1),
        ("s9", 9, 1),
        ("s10", 10, 1),
    ] {
        regs.add_range(n, Kind::S, b, l, Live::Whole).unwrap();
    }
    for b in [0u8, 1, 2, 3, 6, 7] {
        regs.add_range(&format!("v{b}"), Kind::V, b, 1, Live::Whole)
            .unwrap();
    }
    regs.add_range("v45", Kind::V, 4, 2, Live::Whole).unwrap();
    Builder::new(spec(), regs)
}
fn vr(n: u8) -> RegRef {
    V::<1>(n).reg()
}
fn store(b: &mut Builder) -> Result<(), String> {
    b.push(
        Instruction::new(
            "global_store_b32 v0, v1, s[4:5]",
            vec![],
            vec![vr(0), vr(1), S::<2>(4).reg()],
        )
        .memory(MemoryClass::VmemStore),
    )
}
fn load(b: &mut Builder) -> Result<(), String> {
    b.push(
        Instruction::new(
            "global_load_b32 v2, v0, s[4:5]",
            vec![vr(2)],
            vec![vr(0), S::<2>(4).reg()],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn cell() -> GlobalAddr {
    GlobalAddr {
        vaddr: V::<1>(3),
        base: S::<2>(6),
        offset: 0,
    }
}
fn end(mut b: Builder) -> Result<peacemaker_isa::Emitted, String> {
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

// ------------------------------------------------------------------ ledger: release

#[test]
fn relaxed_signal_after_a_global_store_is_a_missing_release() {
    let mut b = builder();
    store(&mut b).unwrap();
    let e = b
        .event_signal(V::<1>(4), cell(), Scope::Dev, Ordering::Relaxed)
        .unwrap_err();
    assert!(e.contains("missing release"), "{e}");
    // A release signal orders the store first: loadcnt/storecnt 0, then the atomic.
    let mut b = builder();
    store(&mut b).unwrap();
    b.event_signal(V::<1>(4), cell(), Scope::Dev, Ordering::Release)
        .unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    let n = text.len();
    assert_eq!(
        &text[n - 3..],
        &[
            "s_wait_loadcnt 0x0",
            "s_wait_storecnt 0x0",
            "global_atomic_add_u32 v3, v4, s[6:7] scope:SCOPE_DEV"
        ]
    );
    // Afterwards a relaxed signal publishes nothing new: accepted.
    b.event_signal(V::<1>(4), cell(), Scope::Dev, Ordering::Relaxed)
        .unwrap();
    end(b).unwrap();
}

#[test]
fn device_release_does_not_cover_system_scope() {
    let mut b = builder();
    store(&mut b).unwrap();
    b.release(Scope::Dev).unwrap();
    let e = b
        .event_signal(V::<1>(4), cell(), Scope::Sys, Ordering::Relaxed)
        .unwrap_err();
    assert!(e.contains("missing release"), "{e}");
    b.event_signal(V::<1>(4), cell(), Scope::Sys, Ordering::Release)
        .unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    let n = text.len();
    assert_eq!(
        &text[n - 4..],
        &[
            "global_wb scope:SCOPE_SYS",
            "s_wait_loadcnt 0x0",
            "s_wait_storecnt 0x0",
            "global_atomic_add_u32 v3, v4, s[6:7] scope:SCOPE_SYS"
        ]
    );
    end(b).unwrap();
}

#[test]
fn forward_branch_carries_unreleased_writes_to_its_target() {
    // A store on one path only still needs the release at the join.
    let mut b = builder();
    b.push(Instruction::new(
        "s_cmp_eq_u32 s8, 0",
        vec![],
        vec![S::<1>(8).reg()],
    ))
    .unwrap();
    b.control(Instruction::new("s_cbranch_scc1 .Ljoin", vec![], vec![]))
        .unwrap();
    store(&mut b).unwrap();
    b.control(Instruction::new("s_branch .Ljoin", vec![], vec![]))
        .unwrap();
    b.label(".Lskip").unwrap();
    b.label(".Ljoin").unwrap();
    let e = b
        .event_signal(V::<1>(4), cell(), Scope::Dev, Ordering::Relaxed)
        .unwrap_err();
    assert!(e.contains("missing release"), "{e}");
}

#[test]
fn loop_back_edge_with_unreleased_writes_is_refused() {
    let mut b = builder();
    b.label(".Lhead").unwrap();
    store(&mut b).unwrap();
    b.wait_all().unwrap();
    let e = b
        .control(Instruction::new("s_cbranch_scc1 .Lhead", vec![], vec![]))
        .unwrap_err();
    assert!(e.contains("back edge"), "{e}");
}

#[test]
fn raw_atomics_and_cache_maintenance_are_refused() {
    let mut b = builder();
    for text in [
        "global_atomic_add_u32 v3, v4, s[6:7] scope:SCOPE_DEV",
        "global_wb scope:SCOPE_SYS",
        "global_inv scope:SCOPE_DEV",
    ] {
        assert!(
            b.push(Instruction::new(text, vec![], vec![])).is_err(),
            "{text}"
        );
    }
    assert!(b
        .event_signal(V::<1>(4), cell(), Scope::Se, Ordering::Release)
        .is_err());
}

#[test]
fn atomics_wait_for_their_return_and_lock_their_sources() {
    let mut b = builder();
    b.global_atomic(
        AtomicOp::CmpSwapB32,
        Some(V::<1>(2)),
        V::<2>(4).reg(),
        cell(),
        Scope::Dev,
    )
    .unwrap();
    // Redefining the data pair waits for the atomic (its sources are locked).
    b.push(Instruction::new("v_mov_b32_e32 v4, 0", vec![vr(4)], vec![]))
        .unwrap();
    // Reading the return value waits for LOADcnt to drain.
    b.push(Instruction::new(
        "v_mov_b32_e32 v1, v2",
        vec![vr(1)],
        vec![vr(2)],
    ))
    .unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    assert_eq!(
        text,
        [
            "global_atomic_cmpswap_b32 v2, v3, v[4:5], s[6:7] th:TH_ATOMIC_RETURN scope:SCOPE_DEV",
            "s_wait_loadcnt 0x0",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v1, v2"
        ]
    );
    // A data atomic is a global write: a relaxed signal after it needs a release.
    let e = b
        .event_signal(V::<1>(7), cell(), Scope::Dev, Ordering::Relaxed)
        .unwrap_err();
    assert!(e.contains("missing release"), "{e}");
    // Returnless: STOREcnt.
    let mut b = builder();
    b.global_atomic(AtomicOp::IncU32, None, vr(4), cell(), Scope::Dev)
        .unwrap();
    b.push(Instruction::new("v_mov_b32_e32 v4, 0", vec![vr(4)], vec![]))
        .unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    assert_eq!(
        text,
        [
            "global_atomic_inc_u32 v3, v4, s[6:7] scope:SCOPE_DEV",
            "s_wait_storecnt 0x0",
            "v_mov_b32_e32 v4, 0"
        ]
    );
}

// ------------------------------------------------------------------ ledger: acquire

fn wait(b: &mut Builder, ordering: Ordering) -> Result<(), String> {
    b.event_wait(
        ".Lw",
        cell(),
        S::<1>(8),
        V::<1>(5),
        S::<1>(9),
        Scope::Dev,
        ordering,
        2,
        None,
    )
}

#[test]
fn load_after_a_relaxed_observation_is_a_missing_acquire() {
    let mut b = builder();
    wait(&mut b, Ordering::Relaxed).unwrap();
    let e = load(&mut b).unwrap_err();
    assert!(e.contains("missing acquire"), "{e}");
    b.acquire(Scope::Dev).unwrap();
    load(&mut b).unwrap();
    b.wait_all().unwrap();
    end(b).unwrap();
    // Acquire ordering: the spin and the acquire, then loads are accepted.
    let mut b = builder();
    wait(&mut b, Ordering::Acquire).unwrap();
    load(&mut b).unwrap();
    b.wait_all().unwrap();
    let emitted = end(b).unwrap();
    let text = emitted.s_text;
    let spin = text.find(".Lw:").unwrap();
    let tail: Vec<&str> = text[spin..].lines().map(str::trim).take(13).collect();
    assert_eq!(
        tail,
        [
            ".Lw:",
            "global_load_b32 v5, v3, s[6:7] scope:SCOPE_DEV",
            "s_wait_loadcnt 0x0",
            "v_readfirstlane_b32 s9, v5",
            "s_cmp_ge_u32 s9, s8",
            "s_cbranch_scc1 .Lw_ready",
            "s_sleep 2",
            "s_branch .Lw",
            ".Lw_ready:",
            "s_wait_loadcnt 0x0",
            "global_inv scope:SCOPE_DEV",
            "global_load_b32 v2, v0, s[4:5]",
            "s_wait_loadcnt 0x0",
        ]
    );
    // Scalar loads are refused while unacquired too.
    let mut b = builder();
    wait(&mut b, Ordering::Relaxed).unwrap();
    let e = b
        .push(
            Instruction::new(
                "s_load_b32 s10, s[4:5], 0x0",
                vec![S::<1>(10).reg()],
                vec![S::<2>(4).reg()],
            )
            .memory(MemoryClass::SmemLoad),
        )
        .unwrap_err();
    assert!(e.contains("missing acquire"), "{e}");
}

#[test]
fn timers_and_sleep() {
    let mut b = builder();
    b.sleep(127).unwrap();
    assert!(b.sleep(128).is_err());
    b.shader_cycles(S::<2>(4), S::<1>(8)).unwrap();
    b.realtime(S::<2>(6)).unwrap();
    b.push(Instruction::new(
        "s_mov_b32 s9, s6",
        vec![S::<1>(9).reg()],
        vec![S::<1>(6).reg()],
    ))
    .unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    assert_eq!(
        text,
        [
            "s_sleep 127",
            "s_getreg_b32 s8, hwreg(HW_REG_SHADER_CYCLES_HI)",
            "s_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES_LO)",
            "s_getreg_b32 s5, hwreg(HW_REG_SHADER_CYCLES_HI)",
            "s_cmp_eq_u32 s8, s5",
            "s_cselect_b32 s4, s4, 0",
            "s_sendmsg_rtn_b64 s[6:7], sendmsg(MSG_RTN_GET_REALTIME)",
            "s_wait_kmcnt 0x0",
            "s_mov_b32 s9, s6"
        ]
    );
    end(b).unwrap();
}

// ------------------------------------------------------------------ independent replay

fn without(
    program: &Program,
    proofs: &[SyncProof],
    drop: impl Fn(usize, &str) -> bool,
) -> (Program, Vec<SyncProof>) {
    let mut map = Vec::new();
    let mut out = Program {
        arch: program.arch,
        instructions: vec![],
    };
    for (pc, i) in program.instructions.iter().enumerate() {
        map.push(out.instructions.len());
        if !drop(pc, &i.text) {
            out.instructions.push(i.clone());
        }
    }
    let proofs = proofs
        .iter()
        .map(|p| SyncProof {
            pc_index: map[p.pc_index],
            ..p.clone()
        })
        .collect();
    (out, proofs)
}

fn emitted_program(e: &peacemaker_isa::Emitted) -> Program {
    e.program.clone()
}

#[test]
fn replay_accepts_the_handoff_and_rejects_a_removed_release_or_acquire() {
    let mut b = builder();
    store(&mut b).unwrap();
    b.event_signal(V::<1>(4), cell(), Scope::Sys, Ordering::Release)
        .unwrap();
    wait(&mut b, Ordering::Acquire).unwrap();
    load(&mut b).unwrap();
    b.wait_all().unwrap();
    let e = end(b).unwrap();
    let program = emitted_program(&e);
    let proofs = &e.proof.syncs;
    assert_eq!(
        proofs.iter().map(|p| p.kind).collect::<Vec<_>>(),
        [SyncKind::ReleaseSignal, SyncKind::Poll, SyncKind::Acquire]
    );
    // The proofs index the builder's own program: recovered text must line up with them.
    for p in proofs {
        let t = &program.instructions[p.pc_index].text;
        match p.kind {
            SyncKind::ReleaseSignal => assert!(t.starts_with("global_atomic_add_u32"), "{t}"),
            SyncKind::Poll => assert!(t.starts_with("global_load_b32 v5"), "{t}"),
            SyncKind::Acquire => assert!(t.starts_with("global_inv"), "{t}"),
            _ => {}
        }
    }
    sync::verify(&program, proofs).unwrap();
    for (what, needle) in [
        ("store wait", "s_wait_storecnt 0x0"),
        ("system writeback", "global_wb scope:SCOPE_SYS"),
        ("acquire invalidate", "global_inv scope:SCOPE_DEV"),
    ] {
        let (p, q) = without(&program, proofs, |_, t| t == needle);
        let err = sync::verify(&p, &q).expect_err(what);
        assert!(
            err.contains("release") || err.contains("acquire"),
            "{what}: {err}"
        );
    }
}

#[test]
fn handoff_kernel_passes_its_own_replay_and_fails_without_its_fences() {
    let e = pd::handoff().unwrap();
    let program = emitted_program(&e);
    let proofs = &e.proof.syncs;
    sync::verify(&program, proofs).unwrap();
    for needle in ["s_wait_storecnt 0x0", "global_inv scope:SCOPE_DEV"] {
        let (p, q) = without(&program, proofs, |_, t| t == needle);
        assert!(sync::verify(&p, &q).is_err(), "{needle}");
    }
    native::assemble(&e.s_text, Arch::Gfx1201).unwrap();
}

// ------------------------------------------------------------------ region ABI

fn region_builder() -> (Builder, RegionAbi) {
    let mut regs = RegPlan::new(32, 32).unwrap();
    regs.add_range("args", Kind::S, 0, 2, Live::Whole).unwrap();
    regs.add_range("wg", Kind::S, 8, 1, Live::Whole).unwrap();
    regs.add_range("lane", Kind::V, 0, 1, Live::Whole).unwrap();
    let live = Live::Between(".Lr_entry".into(), ".Lr_exit".into());
    regs.add_range("body_s", Kind::S, 4, 2, live.clone())
        .unwrap();
    regs.add_range("body_v", Kind::V, 1, 1, live.clone())
        .unwrap();
    // reuses the body's v1 outside the region
    regs.add_range(
        "caller_v",
        Kind::V,
        1,
        1,
        Live::Between(".Lr_exit".into(), ".Lend".into()),
    )
    .unwrap();
    let abi = RegionAbi {
        name: "r".into(),
        inputs: vec![
            ("args".into(), S::<2>(0).reg()),
            ("wg".into(), S::<1>(8).reg()),
        ],
        lds_bytes: 0,
        workgroup_size: 32,
    };
    (Builder::new(spec(), regs), abi)
}

#[test]
fn region_abi_is_enforced() {
    let body_ok = |b: &mut Builder| {
        b.push(
            Instruction::new(
                "s_load_b64 s[4:5], s[0:1], 0x0",
                vec![S::<2>(4).reg()],
                vec![S::<2>(0).reg()],
            )
            .memory(MemoryClass::SmemLoad),
        )?;
        b.push(Instruction::new(
            "v_mov_b32_e32 v1, s8",
            vec![vr(1)],
            vec![S::<1>(8).reg()],
        ))?;
        b.wait_all()?;
        b.push(
            Instruction::new(
                "global_store_b32 v0, v1, s[4:5]",
                vec![],
                vec![vr(0), vr(1), S::<2>(4).reg()],
            )
            .memory(MemoryClass::VmemStore),
        )?;
        b.wait_all()
    };
    let (mut b, abi) = region_builder();
    b.region(&abi, body_ok).unwrap();
    b.push(Instruction::new("v_mov_b32_e32 v1, 0", vec![vr(1)], vec![]))
        .unwrap();
    b.label(".Lend").unwrap();
    let e = end(b).unwrap();
    let r = &e.proof.regions[0];
    assert!(r.unreleased_writes_at_exit);
    assert!(!r.exec_clobbered);
    assert_eq!(r.locals.len(), 2);

    // body-local register used outside the region: no live owner
    let (mut b, _) = region_builder();
    assert!(b
        .push(Instruction::new(
            "s_mov_b32 s4, 0",
            vec![S::<1>(4).reg()],
            vec![]
        ))
        .is_err());

    type Body = fn(&mut Builder) -> Result<(), String>;
    let cases: [(&str, Body); 5] = [
        ("s_endpgm", |b| {
            b.control(Instruction::new("s_endpgm", vec![], vec![]))
        }),
        ("workgroup id", |b| {
            b.push(Instruction::new(
                "s_mov_b32 s4, ttmp9",
                vec![S::<1>(4).reg()],
                vec![],
            ))
        }),
        ("input write", |b| {
            b.push(Instruction::new(
                "s_mov_b32 s8, 0",
                vec![S::<1>(8).reg()],
                vec![],
            ))
        }),
        ("leaves", |b| {
            b.control(Instruction::new("s_branch .Lend", vec![], vec![]))
        }),
        ("unretired", |b| {
            b.push(
                Instruction::new(
                    "s_load_b64 s[4:5], s[0:1], 0x0",
                    vec![S::<2>(4).reg()],
                    vec![S::<2>(0).reg()],
                )
                .memory(MemoryClass::SmemLoad),
            )
        }),
    ];
    for (what, body) in cases {
        let (mut b, abi) = region_builder();
        let r = b.region(&abi, body).and_then(|()| {
            b.label(".Lend")?;
            end(b).map(|_| ())
        });
        assert!(r.is_err(), "{what} accepted");
    }
    // a branch from outside into the region past its entry
    let (mut b, abi) = region_builder();
    b.control(Instruction::new("s_cbranch_scc1 .Linner", vec![], vec![]))
        .unwrap();
    b.region(&abi, |b| {
        b.label(".Linner")?;
        Ok(())
    })
    .unwrap();
    b.label(".Lend").unwrap();
    let e = end(b).unwrap_err();
    assert!(e.contains("jumps into region"), "{e}");
}

// ------------------------------------------------------------------ dispatcher

fn body_lines(text: &str, from: &str, to: &str) -> Vec<String> {
    let start = text.find(from).unwrap() + from.len();
    let stop = start + text[start..].find(to).unwrap();
    text[start..stop]
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

#[test]
fn dispatcher_region_is_the_shipped_body_and_its_proofs_close() {
    let e = pd::dispatcher().unwrap();
    let standalone = mq2::emit_text(&mq2::SHIPPED).unwrap();
    // The standalone body: after the kernel symbol label, up to s_endpgm.
    let sym = format!("{}:", mq2::SYMBOL);
    let mut solo = body_lines(&standalone, &sym, "s_endpgm");
    solo.iter_mut().for_each(|l| {
        *l = l.replace(".Lpx", ".Lmq2").replace("ttmp9", "s47");
    });
    let region = body_lines(&e.s_text, ".Lmq2_entry:", ".Lmq2_exit:");
    // Inside the dispatcher loop the body also follows its own previous iteration, so the
    // gfx12 SGPR hazard tracker adds `s_wait_alu` guards the first-and-only standalone pass
    // never needs; every other instruction is the shipped body's, in order.
    let guard = |l: &&String| l.starts_with("s_wait_alu");
    let strip = |v: &[String]| v.iter().filter(|l| !guard(l)).cloned().collect::<Vec<_>>();
    assert_eq!(
        strip(&region),
        strip(&solo),
        "region body differs from the standalone shipped body"
    );
    let extra = region.iter().filter(guard).count() - solo.iter().filter(guard).count();
    eprintln!(
        "region: {} instructions, {extra} extra hazard guards",
        region.len()
    );
    let r = &e.proof.regions[0];
    assert_eq!(r.name, "mq2");
    assert!(r.exec_clobbered && r.unreleased_writes_at_exit);
    assert!(!r.locals.is_empty());
    let kinds: Vec<SyncKind> = e.proof.syncs.iter().map(|p| p.kind).collect();
    assert_eq!(kinds, [SyncKind::RelaxedSignal, SyncKind::ReleaseSignal]);
    let program = emitted_program(&e);
    sync::verify(&program, &e.proof.syncs).unwrap();
    let (p, q) = without(&program, &e.proof.syncs, |_, t| t == "s_wait_storecnt 0x0");
    assert!(sync::verify(&p, &q).is_err());
    native::assemble(&e.s_text, Arch::Gfx1201).unwrap();
}

// ------------------------------------------------------------------ emulator differential

const KARG: u64 = 0x1_0000;
const A: u64 = 0x100_0000;
const X: u64 = 0x800_0000;
const Y: u64 = 0x900_0000;
const QUEUE: u64 = 0xa00_0000;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn mq2_inputs(m: usize, k: usize, b: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg(seed);
    let groups = k / 256;
    let mut a = vec![0u8; m * groups * 72 + 64];
    for row in 0..m {
        for g in 0..groups {
            let at = (row * groups + g) * 72;
            for h in 0..4 {
                // small f16 scales/zeros (sign, exponent 10..14, mantissa)
                let v = ((rng.next() & 0x8000)
                    | (((10 + rng.next() % 5) << 10) & 0x7c00)
                    | (rng.next() & 0x3ff)) as u16;
                a[at + 2 * h..at + 2 * h + 2].copy_from_slice(&v.to_le_bytes());
            }
            for byte in &mut a[at + 8..at + 72] {
                *byte = rng.next() as u8;
            }
        }
    }
    let x: Vec<u8> = (0..b * k)
        .flat_map(|_| ((rng.next() % 2001) as f32 / 1000.0 - 1.0).to_le_bytes())
        .collect();
    (a, x)
}

fn lift(text: &str) -> peacemaker_ir::Program {
    let elf = native::assemble(text, Arch::Gfx1201).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn run_mq2(
    text: &str,
    symbol: &str,
    grid: u32,
    m: usize,
    k: usize,
    b: usize,
    persistent: bool,
) -> Vec<u8> {
    let program = lift(text);
    let emu = Emulator::new(&program, symbol).unwrap();
    let (a, x) = mq2_inputs(m, k, b, 7);
    let mut karg = Vec::new();
    for ptr in [A, X, Y] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [m, k, b] {
        karg.extend((v as u32).to_le_bytes());
    }
    karg.resize(40, 0);
    karg.extend(QUEUE.to_le_bytes());
    karg.extend((m.div_ceil(4) as u32).to_le_bytes());
    karg.resize(pd::DISPATCH_KERNARG_BYTES as usize, 0);
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg).unwrap();
    mem.map("a".into(), A, a).unwrap();
    mem.map("x".into(), X, x).unwrap();
    mem.map("y".into(), Y, vec![0xa5; b * m * 4]).unwrap();
    mem.map("queue".into(), QUEUE, vec![0; 8]).unwrap();
    for wg in 0..grid {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KARG,
                workgroup: [wg, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("{symbol} workgroup {wg}: {e}"));
    }
    if persistent {
        let q = mem.read(QUEUE, 8).unwrap();
        let claims = u32::from_le_bytes(q[0..4].try_into().unwrap());
        let done = u32::from_le_bytes(q[4..8].try_into().unwrap());
        let tasks = m.div_ceil(4) as u32;
        assert_eq!(done, tasks, "completion counter");
        assert_eq!(
            claims,
            tasks + grid,
            "every workgroup claims once past the end"
        );
    }
    mem.read(Y, b * m * 4).unwrap().to_vec()
}

#[test]
fn dispatcher_matches_the_standalone_module_on_the_emulator() {
    let solo = mq2::emit_text(&mq2::SHIPPED).unwrap();
    let disp = pd::dispatcher().unwrap().s_text;
    for (m, k, b) in [(13usize, 512usize, 1usize), (9, 256, 3), (16, 768, 4)] {
        let want = run_mq2(&solo, mq2::SYMBOL, m.div_ceil(4) as u32, m, k, b, false);
        let got = run_mq2(&disp, pd::DISPATCH_SYMBOL, 3, m, k, b, true);
        assert_eq!(got, want, "M={m} K={k} B={b}");
        assert!(
            want.chunks(4).all(|w| w != [0xa5; 4]),
            "every output written"
        );
    }
}

// ------------------------------------------------------------------ lifted region (P0 pool)

/// The shipped MQ2 module, lifted from its committed bundle (not re-emitted), converted
/// into a region (`peacemaker_lift::rewrite::region`) and hosted by the P0 pool
/// dispatcher: every carried instruction re-encodes to its original words, and on the
/// emulator the pool writes the standalone module's bytes.
#[test]
fn lifted_shipped_mq2_module_runs_as_a_p0_pool_region_on_the_emulator() {
    use peacemaker_isa::kernels::lifted_dispatch_gfx1201 as ld;
    use peacemaker_lift::rewrite::region::{check_carried, lift_region, RegionInputs};
    let bundle = include_bytes!("../../../kernels/gemv_mq2_xbatch_pm_gfx1201.hxaco");
    let region = lift_region(
        bundle,
        mq2::SYMBOL,
        &RegionInputs {
            prefix: ".Lmq2r".into(),
            workgroup_x: ld::WG_X,
            workgroup_yz: ld::WG_YZ,
            block: 32,
            dynamic_lds: 0,
        },
    )
    .unwrap();
    assert_eq!(region.receipt.exits_converted, 1);
    assert_eq!(
        region.receipt.input_rewrites[0].2, 1,
        "one ttmp9 read rewritten"
    );
    let pool = ld::pool(
        "pm_pool_mq2",
        32,
        &[ld::Hosted {
            tmpl: 5,
            region: &region,
        }],
        None,
    )
    .unwrap();
    let elf =
        native::assemble_with_regions(&pool.s_text, Arch::Gfx1201, &pool.proof.regions).unwrap();
    assert!(check_carried(&elf, "pm_pool_mq2", &region).unwrap() > 1000);
    // `.note.pm_regions` names the region, its lift request and its source object
    let notes = peacemaker_lift::regions_note::read_regions(&elf).unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].kernel, "pm_pool_mq2");
    let [note] = notes[0].regions.as_slice() else {
        panic!("one region expected: {:?}", notes[0].regions)
    };
    assert_eq!(note.template_id, 5);
    assert!(note.entry_pc < note.exit_pc);
    assert_eq!(note.digest, region.receipt.base_object_sha256);
    assert_eq!(note.source_symbol, region.symbol);
    assert_eq!(
        (note.block, note.prefix.as_str()),
        (region.inputs.block, region.inputs.prefix.as_str())
    );
    let solo = mq2::emit_text(&mq2::SHIPPED).unwrap();
    let (m, k, b) = (21usize, 512usize, 3usize);
    let want = run_mq2(&solo, mq2::SYMBOL, m.div_ceil(4) as u32, m, k, b, false);
    // the P0 run: one task per 3 virtual workgroups, kernarg block = the 36-byte ABI
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emu = Emulator::new(&program, "pm_pool_mq2").unwrap();
    let (a, x) = mq2_inputs(m, k, b, 7);
    let mut karg = Vec::new();
    for p in [A, X, Y] {
        karg.extend(p.to_le_bytes());
    }
    for v in [m, k, b] {
        karg.extend((v as u32).to_le_bytes());
    }
    karg.resize(48, 0);
    let gx = m.div_ceil(4) as u32;
    let mut tasks = Vec::new();
    let mut x0 = 0;
    while x0 < gx {
        let n = 3.min(gx - x0);
        let mut t = [0u32; 32];
        t[..12].copy_from_slice(&[5, x0, 0, 0, gx, 1, 1, 0, u32::MAX, 0, 0, n]);
        tasks.extend(t.iter().flat_map(|w| w.to_le_bytes()));
        x0 += n;
    }
    let n_tasks = (tasks.len() / 128) as u32;
    const TASKS: u64 = 0xb00_0000;
    const PARAMS: u64 = 0xc00_0000;
    let mut params = Vec::new();
    for p in [TASKS, QUEUE, QUEUE + 16, KARG, QUEUE + 64, QUEUE + 32] {
        params.extend(p.to_le_bytes());
    }
    params.extend(1000u64.to_le_bytes());
    params.extend(n_tasks.to_le_bytes());
    params.extend(1u32.to_le_bytes());
    params.resize(ld::KERNARG_BYTES as usize, 0);
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg).unwrap();
    mem.map("a".into(), A, a).unwrap();
    mem.map("x".into(), X, x).unwrap();
    mem.map("y".into(), Y, vec![0xa5; b * m * 4]).unwrap();
    mem.map("tasks".into(), TASKS, tasks).unwrap();
    mem.map("queue".into(), QUEUE, vec![0; 64 + n_tasks as usize * 48])
        .unwrap();
    mem.map("params".into(), PARAMS, params).unwrap();
    for wg in 0..2u32 {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: PARAMS,
                workgroup: [wg, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("pool workgroup {wg}: {e}"));
    }
    assert_eq!(
        mem.read(Y, b * m * 4).unwrap(),
        &want[..],
        "pool output != standalone"
    );
    let q = mem
        .read(QUEUE, 64 + n_tasks as usize * 48)
        .unwrap()
        .to_vec();
    let word = |o: usize| u32::from_le_bytes(q[o..o + 4].try_into().unwrap());
    assert_eq!(
        word(0),
        n_tasks + 2,
        "claims: every task plus one stop per workgroup"
    );
    assert_eq!(
        word(16),
        n_tasks,
        "events[0]: one release increment per task"
    );
    assert_eq!(word(32), 0, "poison");
    for i in 0..n_tasks as usize {
        assert_eq!(word(64 + i * 48), i as u32, "trace record {i}");
    }
}
