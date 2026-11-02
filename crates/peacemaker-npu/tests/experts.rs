// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Grouped V9 experts (`kernels::experts`) in the whole-array simulator: ONE TXN, expert 0 FULL body, experts `1..` LEAN
//! bodies, optionally behind ONE persistent-ring poll / done pair around the whole group.
//!
//! Proven here: every expert's C is byte-equal to its own eager `design_v9(m, ..)` full TXN on a FRESH `Config` and equals
//! the CPU reference; the retained state after the group equals that of the same group built from all-FULL bodies (also
//! behind the ring with a late producer that publishes operands then `seq` through `submit_with`); the group stream is
//! exactly one standalone full payload plus `E - 1` standalone lean payloads with only the shim DDR patches moved by
//! the offsets, and the ring stream is the one-run empty ring with that payload cut in.
use peacemaker_npu::kernels::{
    experts::*,
    gemm_array::{self, ArrayDesign},
    gemm_core::{Control, Epilogue},
    ring::*,
};
use peacemaker_npu::sim::config::{parse_txn, Config, TxnOp};

const INT8: Epilogue = Epilogue::Int8 { shift: 12 };
const POISON: u8 = 0xa5;
const GUARD: u8 = 0xcc;
const DONE_MAGIC: u32 = 0x454e4f44;
const LINE: usize = 64;
const N: usize = 512;
const K: usize = 128;
/// Unused bytes between consecutive experts' arena regions: any stray write lands on GUARD / POISON and is caught.
const GAP: usize = 256;
/// Wait tick of the first operand write: the NPU is already polling an empty ring.
const FIRST_AT: usize = 37;
/// Wait ticks between operand writes and the `seq` write.
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
fn done_bytes(slot: usize, seq: u32) -> [u8; LINE] {
    let mut l = [0; LINE];
    wr32(&mut l, 0, seq);
    wr32(&mut l, 4, slot as u32);
    wr32(&mut l, 8, DONE_MAGIC);
    l
}

/// Experts with distinct `m`, operands laid out with gaps in a deliberately non-monotonic order (A forward, B and C reversed).
fn plans(ms: &[usize], n: usize, k: usize) -> Vec<ExpertPlan> {
    let d = gemm_array::design_v9(512, n, k, INT8, Control::Fast);
    let [a, b, c] = [0, 1, 2].map(|i| d.args[i].bytes + GAP);
    let e = ms.len();
    ms.iter()
        .enumerate()
        .map(|(i, &m)| ExpertPlan {
            m,
            a_off: (i * a) as u64,
            b_off: ((e - 1 - i) * b) as u64,
            c_off: ((e - 1 - i) * c) as u64,
        })
        .collect()
}

/// Operands of one expert, its packed bytes, the eager V9 C bytes of a FRESH `Config` and the CPU reference.
struct Gold {
    a: Vec<i8>,
    b: Vec<i8>,
    packed: [Vec<u8>; 2],
    eager_c: Vec<u8>,
    reference: Vec<i32>,
    eager: ArrayDesign,
}
fn golds(ms: &[usize], n: usize, k: usize) -> Vec<Gold> {
    ms.iter()
        .enumerate()
        .map(|(e, &m)| {
            let (a, b) = matrices(m, n, k, 0x1234abcd ^ (e as u32 * 7919 + m as u32));
            let eager = gemm_array::design_v9(m, n, k, INT8, Control::Fast);
            let packed = eager.pack_in(&a, &b);
            let mut args = vec![
                packed[0].clone(),
                packed[1].clone(),
                vec![POISON; eager.args[2].bytes],
            ];
            Config::from_pdi(&eager.pdi)
                .unwrap()
                .submit(&eager.insts, &mut args)
                .unwrap();
            let reference = eager.reference(&a, &b);
            assert_eq!(
                eager.unpack_out(&args[2]),
                reference,
                "eager V9 m={m} == CPU reference"
            );
            Gold {
                a,
                b,
                packed,
                eager_c: args.swap_remove(2),
                reference,
                eager,
            }
        })
        .collect()
}

/// Guard-filled A / B arenas, poison C arena (and, with a ring, an initialised ring arg).
fn blank_args(g: &GroupedDesign, layout: Option<RingLayout>) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = g.args.iter().map(|s| vec![GUARD; s.bytes]).collect();
    v[2].fill(POISON);
    if let Some(l) = layout {
        assert_eq!(v.len(), 4);
        assert_eq!(v[3].len(), l.bytes());
        l.initialize(&mut v[3])
    } else {
        assert_eq!(v.len(), 3)
    }
    v
}
fn put_operands(args: &mut [Vec<u8>], plans: &[ExpertPlan], gold: &[Gold]) {
    for (p, g) in plans.iter().zip(gold) {
        let (a, b) = (p.a_off as usize, p.b_off as usize);
        args[0][a..a + g.packed[0].len()].copy_from_slice(&g.packed[0]);
        args[1][b..b + g.packed[1].len()].copy_from_slice(&g.packed[1]);
    }
}
fn c_range(g: &GroupedDesign, p: &ExpertPlan, e: usize) -> std::ops::Range<usize> {
    p.c_off as usize..p.c_off as usize + g.c_bytes(e)
}

/// API contract vs eager designs, then every expert's final C (byte-equal to the eager fresh-Config bytes and to the CPU
/// reference), operands untouched, every byte outside the C regions still poison.
fn check_group(
    g: &GroupedDesign,
    plans: &[ExpertPlan],
    gold: &[Gold],
    args: &[Vec<u8>],
    layout: Option<RingLayout>,
    what: &str,
) {
    let mut want = blank_args(g, layout);
    put_operands(&mut want, plans, gold);
    assert_bytes(&args[0], &want[0], &format!("{what}: A arena modified"));
    assert_bytes(&args[1], &want[1], &format!("{what}: B arena modified"));
    let mut covered = vec![false; args[2].len()];
    for (e, (p, gl)) in plans.iter().zip(gold).enumerate() {
        let c = c_range(g, p, e);
        assert_bytes(
            &args[2][c.clone()],
            &gl.eager_c,
            &format!(
                "{what}: expert {e} (m={}) C != eager V9 bytes of a fresh Config",
                p.m
            ),
        );
        assert_eq!(
            g.unpack_out(e, &args[2][c.clone()]),
            gl.reference,
            "{what}: expert {e} C != CPU reference"
        );
        covered[c].fill(true);
    }
    assert!(
        args[2]
            .iter()
            .zip(&covered)
            .all(|(&b, &c)| c || b == POISON),
        "{what}: C arena written outside the expert regions"
    );
}
/// Packing, reference, unpacking and image of the group equal the per-expert eager design for every expert.
fn check_api(g: &GroupedDesign, plans: &[ExpertPlan], gold: &[Gold]) {
    assert_eq!(g.experts(), plans.len());
    for (e, gl) in gold.iter().enumerate() {
        assert_eq!(
            g.pack_in(e, &gl.a, &gl.b),
            gl.packed,
            "expert {e}: grouped packing == eager packing"
        );
        assert_eq!(
            g.reference(e, &gl.a, &gl.b),
            gl.reference,
            "expert {e}: grouped reference == eager reference"
        );
        assert_eq!(
            g.unpack_out(e, &gl.eager_c),
            gl.reference,
            "expert {e}: grouped unpack of the eager C == reference"
        );
        assert_eq!(
            (g.a_bytes(e), g.b_bytes(e), g.c_bytes(e)),
            (
                gl.eager.args[0].bytes,
                gl.eager.args[1].bytes,
                gl.eager.args[2].bytes
            )
        );
        assert_bytes(
            &g.pdi,
            &gl.eager.pdi,
            &format!("expert {e}: one shared PDI"),
        );
    }
}

// ---------------------------------------------------------------- no ring
fn eager_exact(ms: &[usize]) {
    let (plans, gold) = (plans(ms, N, K), golds(ms, N, K));
    let (g, f) = (
        grouped_v9(N, K, &plans, None),
        grouped_v9_all_full(N, K, &plans, None),
    );
    assert!(g.patch_sites.is_empty() && f.patch_sites.is_empty());
    assert!(
        g.insts.len() < f.insts.len(),
        "lean group ({} B) must be smaller than the all-full group ({} B)",
        g.insts.len(),
        f.insts.len()
    );
    assert_bytes(&g.pdi, &f.pdi, "lean and all-full group load the same PDI");
    check_api(&g, &plans, &gold);
    let mut args_g = blank_args(&g, None);
    put_operands(&mut args_g, &plans, &gold);
    let mut args_f = args_g.clone();
    let (mut sim_g, mut sim_f) = (
        Config::from_pdi(&g.pdi).unwrap(),
        Config::from_pdi(&f.pdi).unwrap(),
    );
    sim_g.submit(&g.insts, &mut args_g).unwrap();
    sim_f.submit(&f.insts, &mut args_f).unwrap();
    check_group(&g, &plans, &gold, &args_g, None, "lean group");
    check_group(&f, &plans, &gold, &args_f, None, "all-full group");
    for a in 0..3 {
        assert_bytes(
            &args_g[a],
            &args_f[a],
            &format!("lean group arg{a} != all-full group arg{a}"),
        )
    }
    assert!(
        sim_g.state_snapshot() == sim_f.state_snapshot(),
        "retained state after the lean group != the all-full group"
    );
    eprintln!("PASS experts E={} m={ms:?}: C == eager fresh-Config bytes == CPU, state == all-full group; {} B vs {} B, {} vs {} ticks",ms.len(),g.insts.len(),f.insts.len(),sim_g.ticks,sim_f.ticks);
}
#[test]
fn experts_e3_exact_and_state_equal() {
    eager_exact(&[64, 160, 512])
}
#[test]
fn experts_e4_exact_and_state_equal() {
    eager_exact(&[512, 64, 160, 64])
}

// ---------------------------------------------------------------- ring around the whole group
/// Operands are written at tick `FIRST_AT`, `seq` (after informational slot fields) `OPERAND_TO_SEQ + delay` ticks later;
/// the C arena must stay poison until then. Returns the final args and the simulator.
fn run_ring(
    g: &GroupedDesign,
    plans: &[ExpertPlan],
    gold: &[Gold],
    layout: RingLayout,
    slot: usize,
    seq: u32,
    delay: usize,
) -> (Vec<Vec<u8>>, Config) {
    let mut args = blank_args(g, Some(layout));
    let mut sim = Config::from_pdi(&g.pdi).unwrap();
    let mut shadow = args[3].clone();
    let (mut stage, mut at, mut calls) = (0u8, FIRST_AT, 0usize);
    sim.submit_with(&g.insts, &mut args, &mut |t, a| {
        calls += 1;
        if stage < 2 && calls % 16 == 1 {
            for (e, p) in plans.iter().enumerate() {
                assert!(
                    a[2][c_range(g, p, e)].iter().all(|&b| b == POISON),
                    "expert {e} C written before the group was published (tick {t})"
                );
            }
        }
        if stage == 2 || t < at {
            return;
        }
        if stage == 0 {
            put_operands(a, plans, gold);
            stage = 1;
            at = t + OPERAND_TO_SEQ + delay;
            return;
        }
        for (e, p) in plans.iter().enumerate() {
            assert!(
                a[2][c_range(g, p, e)].iter().all(|&b| b == POISON),
                "expert {e}: C computed from operands whose seq was not yet published (tick {t})"
            )
        }
        let line = layout.slot_line(slot);
        let p = &plans[0];
        let mut fields = [0u8; LINE - 4];
        wr32(&mut fields, 0, 9);
        wr32(&mut fields, 4, p.m as u32);
        wr32(&mut fields, 8, 0);
        wr64(&mut fields, 12, p.a_off);
        wr64(&mut fields, 20, p.b_off);
        wr64(&mut fields, 28, p.c_off);
        for arg in [&mut a[3], &mut shadow] {
            arg[line + 4..line + LINE].copy_from_slice(&fields);
            arg[line..line + 4].copy_from_slice(&seq.to_le_bytes())
        }
        stage = 2;
    })
    .unwrap();
    assert_eq!(stage, 2, "the producer never published");
    assert!(
        sim.ticks > delay,
        "the group finished ({} ticks) before the late publication ({delay})",
        sim.ticks
    );
    assert_bytes(
        &args[3][..layout.done_line(0)],
        &shadow[..layout.done_line(0)],
        "ring header / slot lines != initial + producer writes",
    );
    for s in 0..layout.nslots {
        let at = layout.done_line(s);
        let want = if s == slot {
            done_bytes(slot, seq)
        } else {
            [0; LINE]
        };
        assert_eq!(&args[3][at..at + LINE], &want[..], "done line of slot {s}");
    }
    (args, sim)
}
fn ring_group(ms: &[usize], layout: RingLayout, slot: usize, seq: u32, delay: usize) {
    let (plans, gold) = (plans(ms, N, K), golds(ms, N, K));
    let ring = Some((layout, slot, seq));
    let (g, f) = (
        grouped_v9(N, K, &plans, ring),
        grouped_v9_all_full(N, K, &plans, ring),
    );
    assert_eq!(g.args.len(), 4);
    assert_eq!(g.args[3].bytes, layout.bytes());
    check_api(&g, &plans, &gold);
    let (args_g, sim_g) = run_ring(&g, &plans, &gold, layout, slot, seq, delay);
    let (args_f, sim_f) = run_ring(&f, &plans, &gold, layout, slot, seq, delay);
    check_group(&g, &plans, &gold, &args_g, Some(layout), "lean ring group");
    check_group(
        &f,
        &plans,
        &gold,
        &args_f,
        Some(layout),
        "all-full ring group",
    );
    for a in 0..4 {
        assert_bytes(
            &args_g[a],
            &args_f[a],
            &format!("lean ring group arg{a} != all-full ring group arg{a}"),
        )
    }
    assert!(
        sim_g.state_snapshot() == sim_f.state_snapshot(),
        "retained state after the lean ring group != the all-full ring group"
    );
    eprintln!("PASS experts ring E={} m={ms:?} slot={slot} seq={seq} delay={delay}: final DONE exact, C == eager == CPU, state == all-full ring; {} B vs {} B, {} vs {} ticks",ms.len(),g.insts.len(),f.insts.len(),sim_g.ticks,sim_f.ticks);
}
#[test]
fn experts_ring_e3_late_producer() {
    ring_group(&[64, 160, 512], RingLayout { nslots: 2 }, 0, 1, 20_000)
}
#[test]
fn experts_ring_e4_late_producer_slot_seq() {
    ring_group(&[160, 512, 64, 64], RingLayout { nslots: 4 }, 3, 7, 400)
}

// ---------------------------------------------------------------- structure
/// `group` is `standalone` with every DDR patch of args 0..=2 advanced by `off[arg]` and NOTHING else different. Returns
/// the number of moved patches.
fn assert_only_offsets_differ(
    group: &[TxnOp],
    standalone: &[TxnOp],
    off: [u64; 3],
    what: &str,
) -> usize {
    assert_eq!(group.len(), standalone.len(), "{what}: op count");
    let mut moved = 0;
    for (i, (x, y)) in group.iter().zip(standalone).enumerate() {
        match (x, y) {
            (
                TxnOp::DdrPatch {
                    address: ax,
                    arg: gx,
                    plus: px,
                },
                TxnOp::DdrPatch {
                    address: ay,
                    arg: gy,
                    plus: py,
                },
            ) => {
                assert!(
                    ax == ay && gx == gy && *gx < 3,
                    "{what}: op {i}: DDR patch target changed ({x:?} vs {y:?})"
                );
                assert_eq!(
                    *px,
                    *py + off[*gx],
                    "{what}: op {i}: DDR patch of arg {gx} not moved by exactly its offset"
                );
                if off[*gx] != 0 {
                    moved += 1
                }
            }
            _ => assert_eq!(
                x, y,
                "{what}: op {i} differs from the standalone TXN and is not a shim DDR patch"
            ),
        }
    }
    moved
}
fn structure(ms: &[usize]) {
    let plans = plans(ms, N, K);
    let e = ms.len();
    let g = grouped_v9(N, K, &plans, None);
    let eager = gemm_array::design_v9(512, N, K, INT8, Control::Fast);
    // M padding: every m maps to the identical image and full TXN.
    for &m in ms {
        let d = gemm_array::design_v9(m, N, K, INT8, Control::Fast);
        assert_bytes(&d.pdi, &g.pdi, "pdi");
        assert_bytes(
            &d.insts,
            &eager.insts,
            &format!("full TXN of m={m} == padded"),
        );
    }
    let (full, lean) = (eager.insts.clone(), eager.lean_insts().unwrap());
    let (ops_g, ops_f, ops_l) = (
        parse_txn(&g.insts).unwrap(),
        parse_txn(&full).unwrap(),
        parse_txn(&lean).unwrap(),
    );
    // Byte accounting: header + one full payload + (E-1) lean payloads, nothing per-expert full.
    assert!(lean.len() < full.len());
    assert_eq!(
        g.insts.len(),
        16 + (full.len() - 16) + (e - 1) * (lean.len() - 16),
        "group bytes = 16 + F + (E-1) L"
    );
    assert_eq!(ops_g.len(), ops_f.len() + (e - 1) * ops_l.len());
    assert_bytes(&g.insts[..8], &full[..8], "header geometry");
    // Segment 0 = standalone full, segment e>=1 = standalone lean; only shim DDR patches moved by the offsets.
    let mut at = 0;
    for (i, p) in plans.iter().enumerate() {
        let (seg, name) = if i == 0 {
            (&ops_f, "full")
        } else {
            (&ops_l, "lean")
        };
        let off = [p.a_off, p.b_off, p.c_off];
        let moved = assert_only_offsets_differ(
            &ops_g[at..at + seg.len()],
            seg,
            off,
            &format!("expert {i} vs standalone {name}"),
        );
        // Every nonzero offset moved at least one patch of its arg, and a zero offset moved none.
        assert_eq!(
            moved > 0,
            off != [0; 3],
            "expert {i}: moved patches {moved} vs offsets {off:?}"
        );
        at += seg.len();
    }
    assert_eq!(at, ops_g.len());
    assert!(
        plans
            .iter()
            .skip(1)
            .any(|p| p.a_off != 0 && p.b_off != 0 && p.c_off != 0),
        "the plans must move every arg for some lean expert"
    );
    eprintln!(
        "PASS experts structure E={e}: {} B = 16 + {} (full) + {}x{} (lean)",
        g.insts.len(),
        full.len() - 16,
        e - 1,
        lean.len() - 16
    );
}
#[test]
fn experts_group_is_full_plus_lean_payloads_with_moved_patches_e3() {
    structure(&[64, 160, 512])
}
#[test]
fn experts_group_is_full_plus_lean_payloads_with_moved_patches_e4() {
    structure(&[512, 64, 160, 64])
}

/// With a ring the stream is the one-run empty ring with the no-ring payload cut in after the DRAIN sentinel poll; the
/// patch inventory is exactly `PollSeq(0)` / `DoneSeq(0)` and round-trips through `patch_seq`.
#[test]
fn experts_ring_wraps_group_once_with_patch_inventory() {
    let (ms, layout, slot, seq0) = ([64usize, 160, 512], RingLayout { nslots: 2 }, 1usize, 5u32);
    let plans = plans(&ms, N, K);
    let (g, bare) = (
        grouped_v9(N, K, &plans, Some((layout, slot, seq0))),
        grouped_v9(N, K, &plans, None),
    );
    let proto = empty_persistent(
        layout,
        &[SlotPlan {
            slot,
            a_off: 0,
            b_off: 0,
            c_off: 0,
        }],
        seq0,
    );
    assert_eq!(
        g.insts.len(),
        proto.insts.len() + bare.insts.len() - 16,
        "ring bytes = empty one-run ring + body payload"
    );
    let (ops_g, ops_p, ops_b) = (
        parse_txn(&g.insts).unwrap(),
        parse_txn(&proto.insts).unwrap(),
        parse_txn(&bare.insts).unwrap(),
    );
    let cut = {
        let mut polls = ops_p
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(o,TxnOp::MaskPoll(_,_,v) if *v==SENTINEL));
        let (i, _) = polls.next().expect("drain poll");
        assert!(polls.next().is_none());
        i + 1
    };
    let want: Vec<TxnOp> = [&ops_p[..cut], &ops_b[..], &ops_p[cut..]].concat();
    assert_eq!(
        ops_g, want,
        "ring group ops == empty ring with the body cut in once"
    );
    assert_eq!(
        ops_g
            .iter()
            .filter(|o| matches!(o, TxnOp::Sync { .. }))
            .count(),
        ops_p
            .iter()
            .filter(|o| matches!(o, TxnOp::Sync { .. }))
            .count()
            + ops_b
                .iter()
                .filter(|o| matches!(o, TxnOp::Sync { .. }))
                .count()
    );
    // Sites.
    assert_eq!(g.patch_sites.len(), 2);
    assert_eq!(g.patch_sites[0].1, PatchKind::PollSeq(0));
    assert_eq!(g.patch_sites[1].1, PatchKind::DoneSeq(0));
    for &(w, _) in &g.patch_sites {
        assert_eq!(rd(&g.insts, 4 * w), seq0)
    }
    let fresh = |s| grouped_v9(N, K, &plans, Some((layout, slot, s)));
    let other = fresh(seq0 + 9);
    assert_eq!(g.patch_sites, other.patch_sites);
    let declared: Vec<usize> = {
        let mut w: Vec<usize> = g.patch_sites.iter().map(|&(w, _)| w).collect();
        w.sort();
        w
    };
    assert_eq!(
        diff_words(&g.insts, &other.insts),
        declared,
        "byte diff between two seq0 builds is exactly the declared words"
    );
    let mut patched = g.insts.clone();
    patch_seq(&mut patched, &g.patch_sites, seq0 + 9);
    assert_bytes(
        &patched,
        &other.insts,
        "patch_seq output != from-scratch build",
    );
    assert_eq!(g.args.len(), 4);
    assert_eq!(g.args[3].bytes, layout.bytes());
    assert_eq!(&g.args[..3], &bare.args[..], "arenas unchanged by the ring");
    eprintln!(
        "PASS experts ring structure: {} B, sites {:?}",
        g.insts.len(),
        g.patch_sites
    );
}

// ---------------------------------------------------------------- boundaries
fn plan(m: usize, a: u64, b: u64, c: u64) -> ExpertPlan {
    ExpertPlan {
        m,
        a_off: a,
        b_off: b,
        c_off: c,
    }
}
#[test]
#[should_panic(expected = "at least one expert")]
fn rejects_empty_group() {
    grouped_v9(N, K, &[], None);
}
#[test]
#[should_panic(expected = "outside 1..=512")]
fn rejects_m_zero() {
    grouped_v9(N, K, &[plan(64, 0, 0, 0), plan(0, 0, 0, 0)], None);
}
#[test]
#[should_panic(expected = "outside 1..=512")]
fn rejects_m_above_one_wave() {
    grouped_v9(N, K, &[plan(513, 0, 0, 0)], None);
}
#[test]
#[should_panic(expected = "word aligned")]
fn rejects_unaligned_a_offset() {
    grouped_v9(N, K, &[plan(64, 2, 0, 0)], None);
}
#[test]
#[should_panic(expected = "word aligned")]
fn rejects_unaligned_c_offset() {
    grouped_v9(N, K, &[plan(64, 0, 0, 0), plan(64, 0, 0, 6)], None);
}
#[test]
#[should_panic(expected = "slot")]
fn rejects_ring_slot_outside_ring() {
    grouped_v9(
        N,
        K,
        &[plan(64, 0, 0, 0)],
        Some((RingLayout { nslots: 2 }, 2, 1)),
    );
}
#[test]
fn boundary_m_one_and_512_are_exact() {
    let ms = [1, 512];
    let (plans, gold) = (plans(&ms, N, K), golds(&ms, N, K));
    let g = grouped_v9(N, K, &plans, None);
    check_api(&g, &plans, &gold);
    let mut args = blank_args(&g, None);
    put_operands(&mut args, &plans, &gold);
    Config::from_pdi(&g.pdi)
        .unwrap()
        .submit(&g.insts, &mut args)
        .unwrap();
    check_group(&g, &plans, &gold, &args, None, "m=1,512");
}
