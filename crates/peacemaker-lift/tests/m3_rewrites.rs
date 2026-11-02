// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! M3 (architecture.md §7): this session's hand edits as edit scripts over the IR,
//! reproducing their objects byte for byte from the lifted base objects:
//! the profiler instrumentation (`peacemaker/profile/report.md:327`: `kt48_pmprofile.co`,
//! `silu_pmprofile.co`) and the three attn-sync modules (`perf/fp8-ng/attn-sync/
//! report.md:63,69-70`: `split-asm`, `prio-entry`, `prio-publish`, device ELF and HIP
//! bundle). Every script runs through `rewrite::apply`: each step a checked core
//! transaction, the module re-laid out and lifted back, check 11 on the final bytes.

use std::path::Path;

use peacemaker_ir::cfg::{BlockId, InstId};
use peacemaker_ir::edit::{Cursor, Edit, InstRange};
use peacemaker_ir::inst::{Frontend, Program, SymbolId};
use peacemaker_lift::rewrite::build::{self, hw, imm, label, op, s, simm16};
use peacemaker_lift::rewrite::profile::{self, Rule};
use peacemaker_lift::rewrite::{
    apply, position, Certified, EquivalenceKind, Rewrite, RewriteError, SourceMap,
};
use peacemaker_lift::{lift_object, Options};
use sha2::{Digest, Sha256};

const KT48: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201";
const SILU: &str = "gemm_mq4g256v2_fp8_silu_row_b1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn load(relative: &str, sha256: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        sha256,
        "{} is not the pinned fixture",
        path.display()
    );
    bytes
}

fn kt48_co() -> Vec<u8> {
    load(
        "tests/fixtures/kt48/hipcc.co",
        "f876e2ce520ba8d97358b34c903a48dafc5cff2e4cc552d7ffc775361cbcb6bc",
    )
}
fn kt48_hsaco() -> Vec<u8> {
    load(
        "tests/fixtures/kt48/hipcc.hsaco",
        "3f968bb9dd13a79c4a1b7c5cd805c8d8b2bd1e8dc93821230acabe19df0d0e3a",
    )
}
fn kt48_s() -> String {
    String::from_utf8(load(
        "tests/fixtures/kt48/hipcc.s",
        "a5e59478cc70261badac4bbaba7476315cedb7a668adc4b5b15e8788edd0b3a7",
    ))
    .unwrap()
}
fn f2_co() -> Vec<u8> {
    load(
        "tests/fixtures/f2/f2.co",
        "e9ab2c8a3cf82608020338845118a88324426193f2d35ece5bda97d2e1d0fbe8",
    )
}
fn f2_silu_s() -> String {
    String::from_utf8(load(
        "tests/fixtures/f2/f2.silu_row_b1.s",
        "d8d2778076dc4ba1f570567af2adb2de2dca040c7633e2c09183cdd710026eb4",
    ))
    .unwrap()
}

const HIPCC: Options = Options {
    frontend: Frontend::Hipcc,
};
const BUILDER: Options = Options {
    frontend: Frontend::Builder,
};

/// Where the original objects of this session live (only read to name the first
/// differing byte when a reproduction fails).
const ORIGINALS: &str = "/home/kaden/qcal";

/// The reproduction must be the original object: SHA-256 and length pinned; on a
/// mismatch the message names the first differing byte against the original if present.
fn assert_reproduces(bytes: &[u8], sha256: &str, len: usize, original: &str) {
    if hex(&Sha256::digest(bytes)) == sha256 && bytes.len() == len {
        return;
    }
    let detail = match std::fs::read(Path::new(ORIGINALS).join(original)) {
        Ok(want) => {
            let at = bytes
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(bytes.len().min(want.len()));
            format!(
                "first differing byte {at:#x} (ours {:?}, original {:?}), lengths {} / {}",
                bytes.get(at),
                want.get(at),
                bytes.len(),
                want.len()
            )
        }
        Err(_) => "original not available".into(),
    };
    panic!("{original} not reproduced: {detail}");
}

fn program(bytes: &[u8], options: Options) -> Program {
    lift_object(bytes, options).unwrap().program
}

fn kernel<'a>(program: &'a Program, name: &str) -> &'a peacemaker_ir::inst::Kernel {
    program.kernels.iter().find(|k| k.symbol.0 == name).unwrap()
}

fn run(
    base: &[u8],
    options: Options,
    kernel: &str,
    script: Vec<Edit>,
    claim: EquivalenceKind,
) -> Certified {
    let out = apply(
        base,
        options,
        Rewrite {
            kernel: SymbolId(kernel.into()),
            script,
            claim,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let r = &out.receipt;
    let open: Vec<&str> = r
        .new_obligations
        .iter()
        .map(|o| o.rule_id.as_str())
        .collect();
    let rewrites: Vec<String> = r
        .delay_rewrites
        .iter()
        .map(|d| format!("{:?}->{:?}", d.before, d.after))
        .collect();
    eprintln!("receipt {} -> {}: object {} ({} B), {} steps, {} claims, delay rewrites {rewrites:?}, {} re-encoded branches, new obligations {open:?}, definedness {:?}, equivalence {:?}",
        r.kernel_before, r.kernel, hex(&r.object_sha256), out.bytes.len(), r.script.len(), r.claims.len(), r.reencoded_branches, r.definedness, r.equivalence.kind);
    out
}

fn before(program: &Program, at: usize) -> Cursor {
    let (block, id) = position(kernel(program, KT48), at);
    Cursor::before(block, id)
}

fn setprio(level: u16) -> peacemaker_ir::inst::Inst {
    op("s_setprio", vec![simm16(level)]).unwrap()
}

/// prio-publish (`attn-sync/report.md:70`): `s_setprio 3` after the repeated tile
/// barrier's `global_inv` (`.s` line 4702), `s_setprio 0` at `.LBB5_68`, `.LBB5_84` and
/// `.LBB5_107`. Priority edits stay `Unproved`.
#[test]
fn prio_publish_reproduces_the_attn_sync_module() {
    let (co, hsaco) = (kt48_co(), kt48_hsaco());
    let p = program(&co, HIPCC);
    let map = SourceMap::new(kernel(&p, KT48), &kt48_s()).unwrap();
    let after_inv = map.at_line(4702).unwrap() + 1;
    assert!(map.lines[after_inv - 1].starts_with("global_inv scope:SCOPE_SE"));
    let mut script = vec![Edit::Insert {
        at: before(&p, after_inv),
        insts: vec![setprio(3)],
        claims: vec![],
    }];
    for label in [".LBB5_68", ".LBB5_84", ".LBB5_107"] {
        script.push(Edit::Insert {
            at: before(&p, map.at_label(label).unwrap()),
            insts: vec![setprio(0)],
            claims: vec![],
        });
    }
    let out = run(&co, HIPCC, KT48, script.clone(), EquivalenceKind::Unproved);
    assert_reproduces(
        &out.bytes,
        "4fa4e624d68cadfa097f3ae32e50c496b012c52ac95e4a2f1446041c1a94ed91",
        42_208,
        "perf/fp8-ng/attn-sync/obj/prio-publish.co",
    );
    assert_eq!(out.receipt.definedness, Ok(()));
    assert!(
        out.receipt.new_obligations.is_empty(),
        "{:?}",
        out.receipt.new_obligations
    );
    // The module the report names (MD5 f5388cad…) is the HIP bundle.
    let bundled = run(&hsaco, HIPCC, KT48, script, EquivalenceKind::Unproved);
    assert_reproduces(
        &bundled.bytes,
        "b81de8a6739ea43d725cf295e43aa1fddb5f3bb9904772b6d1437aa8ea02a499",
        46_304,
        "perf/fp8-ng/attn-sync/obj/prio-publish.hsaco",
    );
}

/// prio-entry (`attn-sync/report.md:69`): at the kernel entry, read `HW_REG_WAVE_HW_ID1`
/// into `s4` (overwritten by the kernel's first `s_load_b128`) and set the issue
/// priority from the wave slot: 0 → 0, 1 → 1, 2–3 → 2, ≥4 → 3. Built from core's
/// control-flow vocabulary: straight-line `Insert`, `SplitBlock`s, neutral branches and
/// jumps inserted at block ends, then `Retarget`s that skip only inserted code. The
/// kernel grows by 88 bytes, `.text` by 128: the module is re-laid out.
#[test]
fn prio_entry_reproduces_the_attn_sync_module() {
    let (co, hsaco) = (kt48_co(), kt48_hsaco());
    let p = program(&co, HIPCC);
    let k = kernel(&p, KT48);
    let e0 = k.body.layout[0];
    let next = k.body.insts.len();
    let id = |n: usize| InstId(next + n);
    let cmp = |n: i64| op("s_cmp_ge_u32", vec![s(4), imm(n)]).unwrap();
    let straight = vec![
        op("s_getreg_b32", vec![s(4), build::hwreg(hw::HW_ID1, 0, 32)]).unwrap(),
        op("s_and_b32", vec![s(4), s(4), imm(31)]).unwrap(),
        cmp(4),
        cmp(2),
        cmp(1),
        setprio(0),
        setprio(1),
        setprio(2),
        setprio(3),
    ];
    // ids next+0..=8 in `straight` order; the branches/jumps are next+9..=14.
    let (c2, c1, p0, p1, p2, p3) = (id(3), id(4), id(5), id(6), id(7), id(8));
    let at_end = |b: usize, name: &str, to: usize| Edit::Insert {
        at: Cursor::end(BlockId(b)),
        insts: vec![op(name, vec![label(BlockId(to))]).unwrap()],
        claims: vec![],
    };
    let split = |b: usize, id: InstId| Edit::SplitBlock {
        at: Cursor::before(BlockId(b), id),
    };
    let retarget = |branch: usize, to: usize| Edit::Retarget {
        branch: id(branch),
        to: BlockId(to),
    };
    let script = vec![
        Edit::Insert {
            at: Cursor::before(BlockId(0), e0),
            insts: straight,
            claims: vec![],
        },
        // B0 [getreg, and, cmp 4] B1 [cmp 2] B2 [cmp 1] B3 [prio 0] B4 [prio 1] B5 [prio 2] B6 [prio 3] B7 [kernel]
        split(0, c2),
        split(1, c1),
        split(2, p0),
        split(3, p1),
        split(4, p2),
        split(5, p3),
        split(6, e0),
        at_end(0, "s_cbranch_scc1", 1),
        at_end(1, "s_cbranch_scc1", 2),
        at_end(2, "s_cbranch_scc1", 3),
        at_end(3, "s_branch", 4),
        at_end(4, "s_branch", 5),
        at_end(5, "s_branch", 6),
        // Jumps to the join first, then each compare's branch to its priority block.
        retarget(12, 7),
        retarget(13, 7),
        retarget(14, 7),
        retarget(9, 6),
        retarget(10, 5),
        retarget(11, 4),
    ];
    let out = run(&co, HIPCC, KT48, script.clone(), EquivalenceKind::Unproved);
    assert_reproduces(
        &out.bytes,
        "ca17608496a8a03e1bf74bcd59b509ba70f0cea8c5d4f8e75f6dd3012c0994d3",
        42_336,
        "perf/fp8-ng/attn-sync/obj/prio-entry.co",
    );
    assert_eq!(out.receipt.definedness, Ok(()));
    assert!(
        out.receipt.new_obligations.is_empty(),
        "{:?}",
        out.receipt.new_obligations
    );
    let bundled = run(
        &hsaco,
        HIPCC,
        KT48,
        script.clone(),
        EquivalenceKind::Unproved,
    );
    assert_reproduces(
        &bundled.bytes,
        "46be865c1e5e21eb7b5ec5c08d553c2a056de8cba6ffb8a0ae8cfe8d96ae21f5",
        46_432,
        "perf/fp8-ng/attn-sync/obj/prio-entry.hsaco",
    );
    // Commutation is not claimable for a script that changes control flow.
    let refused = apply(
        &co,
        HIPCC,
        Rewrite {
            kernel: SymbolId(KT48.into()),
            script,
            claim: EquivalenceKind::ProvedCommutation,
        },
    );
    assert!(
        matches!(refused, Err(RewriteError::Claim { .. })),
        "{:?}",
        refused.err()
    );
}

/// split-asm (`attn-sync/report.md:48,63`): the latch's `s_barrier_signal -1` (`.s`
/// line 4695) leaves; a new one runs before the last two PV WMMAs (after `.s` line 4984)
/// behind a uniform guard: signal when the next 16 keys pass `gmax` (`s24 + 16 > s15`)
/// or on subtile 2 (`s25 == 2`). A checked `Move` is refused (the destination is inside
/// the subtile loop: not control-equivalent), so the script is `Remove` + `Insert` + the
/// guard. The barrier pairing is not statically provable; the receipt carries it open.
#[test]
fn split_asm_reproduces_the_attn_sync_module() {
    let (co, hsaco) = (kt48_co(), kt48_hsaco());
    let p = program(&co, HIPCC);
    let k = kernel(&p, KT48);
    let map = SourceMap::new(k, &kt48_s()).unwrap();
    let signal = map.at_line(4695).unwrap();
    let wmma = map.at_line(4984).unwrap() + 1;
    assert!(
        map.lines[signal].starts_with("s_barrier_signal") && map.lines[wmma].starts_with("v_wmma")
    );
    let (_, signal_id) = position(k, signal);
    let (bw, w) = position(k, wmma);
    let next = k.body.insts.len();
    let id = |n: usize| InstId(next + n);
    let b = bw.0;
    let script = vec![
        Edit::Remove {
            range: InstRange::single(signal_id),
        },
        Edit::Insert {
            at: Cursor::before(bw, w),
            insts: vec![
                op("s_add_co_i32", vec![s(4), s(24), imm(16)]).unwrap(),
                op("s_cmp_gt_i32", vec![s(4), s(15)]).unwrap(),
                op("s_cmp_eq_u32", vec![s(25), imm(2)]).unwrap(),
                op("s_barrier_signal", vec![imm(-1)]).unwrap(),
            ],
            claims: vec![],
        },
        Edit::SplitBlock {
            at: Cursor::before(bw, id(2)),
        },
        Edit::SplitBlock {
            at: Cursor::before(BlockId(b + 1), id(3)),
        },
        Edit::SplitBlock {
            at: Cursor::before(BlockId(b + 2), w),
        },
        Edit::Insert {
            at: Cursor::end(bw),
            insts: vec![op("s_cbranch_scc1", vec![label(BlockId(b + 1))]).unwrap()],
            claims: vec![],
        },
        Edit::Insert {
            at: Cursor::end(BlockId(b + 1)),
            insts: vec![op("s_cbranch_scc0", vec![label(BlockId(b + 2))]).unwrap()],
            claims: vec![],
        },
        Edit::Retarget {
            branch: id(4),
            to: BlockId(b + 2),
        },
        Edit::Retarget {
            branch: id(5),
            to: BlockId(b + 3),
        },
    ];
    let out = run(&co, HIPCC, KT48, script.clone(), EquivalenceKind::Unproved);
    assert_reproduces(
        &out.bytes,
        "158f5e4af21101bc581d0e6fde2656a456f1ac2c643114833269ecda9204d4a5",
        42_208,
        "perf/fp8-ng/attn-sync/obj/split-asm.co",
    );
    assert_eq!(out.receipt.definedness, Ok(()));
    let open: Vec<&str> = out
        .receipt
        .new_obligations
        .iter()
        .map(|o| o.rule_id.as_str())
        .collect();
    assert!(
        open.contains(&"barrier-wait-unsignalled") && open.contains(&"barrier-signal-unpaired"),
        "{open:?}"
    );
    let bundled = run(&hsaco, HIPCC, KT48, script, EquivalenceKind::Unproved);
    assert_reproduces(
        &bundled.bytes,
        "534251138e092d48c856d14db25d9802493d51f8fd62cb6e363754e31bac1eb7",
        46_304,
        "perf/fp8-ng/attn-sync/obj/split-asm.hsaco",
    );
}

fn rule(
    name: &str,
    label: Option<&str>,
    before: Option<&str>,
    after: Option<&str>,
    first_after: Option<&str>,
) -> Rule {
    Rule {
        name: name.into(),
        label: label.map(Into::into),
        before: before.map(Into::into),
        after: after.map(Into::into),
        first_after: first_after.map(Into::into),
        occurrences: None,
    }
}

/// The profiler (`peacemaker/profile/report.md:327`): KT48 full point set
/// (`profile/attn/points.json`) through the profiler client; insert-only, so it may
/// claim `ProvedCommutation`.
#[test]
fn kt48_profile_reproduces_kt48_pmprofile() {
    let co = kt48_co();
    let p = program(&co, HIPCC);
    let rules = vec![
        rule("tile", Some(".LBB5_60"), None, None, None),
        rule("fill.done", Some(".LBB5_84"), None, None, None),
        rule("sub", Some(".LBB5_87"), None, None, None),
        rule(
            "softmax",
            None,
            Some("v_dual_mul_f32 v222, v194, v214"),
            None,
            None,
        ),
        rule("pv", Some(".LBB5_85"), None, None, None),
        rule("sub.end", Some(".LBB5_86"), None, None, None),
        rule(
            "bar.pre_drain",
            None,
            Some("s_wait_loadcnt_dscnt 0x0"),
            None,
            None,
        ),
        rule("bar.pre_signal", None, Some("s_barrier_signal"), None, None),
        rule(
            "bar.post_signal",
            None,
            None,
            Some("s_barrier_signal"),
            None,
        ),
        rule("bar.pre_wait", None, Some("s_barrier_wait"), None, None),
        rule("bar.post_wait", None, None, Some("s_barrier_wait"), None),
        rule("epilogue", Some(".LBB5_107"), None, None, None),
    ];
    let plan = profile::plan(&p, &SymbolId(KT48.into()), &kt48_s(), &rules).unwrap();
    assert_eq!(
        (
            plan.registers.scratch_vgpr,
            plan.registers.pointer,
            plan.registers.timestamp,
            plan.registers.exec_save
        ),
        (239, 28, 30, 31)
    );
    let out = run(
        &co,
        HIPCC,
        KT48,
        plan.script,
        EquivalenceKind::ProvedCommutation,
    );
    assert_reproduces(
        &out.bytes,
        "9761e1e64f9d190dd8879905b570a5e5263628de1fb613e3e92998fa43c91036",
        44_280,
        "peacemaker/profile/attn/kt48_pmprofile.co",
    );
    assert_eq!(
        out.receipt.definedness,
        Ok(()),
        "check 11: the straight-line profiler insertion passes"
    );
    assert!(!out.receipt.new_obligations.iter().any(|o| o.rule_id == "definedness-partial-write"),
        "v239 lanes 0..7 are written before the header store under EXEC=0xff; later records rewrite lanes 0..1/3 under equally narrow masks");
    assert_eq!(out.receipt.delay_rewrites.len(), 2);
}

/// The profiler on the F2 builder module (`profile/f2/points.json`), whose silu kernel
/// sits mid-`.text`: every later kernel moves.
#[test]
fn silu_profile_reproduces_silu_pmprofile() {
    let co = f2_co();
    let p = program(&co, BUILDER);
    let first = "s_add_co_i32 s90, s94, s85";
    let rules = vec![
        rule("kblock", None, Some(first), None, None),
        rule(
            "w0.pre",
            None,
            Some("s_wait_loadcnt_dscnt"),
            None,
            Some(first),
        ),
        rule(
            "w0.post",
            None,
            None,
            Some("s_wait_loadcnt_dscnt"),
            Some(first),
        ),
        rule("bar.pre_signal", None, Some("s_barrier_signal"), None, None),
        rule(
            "bar.post_signal",
            None,
            None,
            Some("s_barrier_signal"),
            None,
        ),
        rule("bar.post_wait", None, None, Some("s_barrier_wait"), None),
        rule(
            "epilogue",
            Some(".Lfp8_ratio_256x128x8_row_silu_epilogue"),
            None,
            None,
            None,
        ),
    ];
    let plan = profile::plan(&p, &SymbolId(SILU.into()), &f2_silu_s(), &rules).unwrap();
    assert_eq!(
        (
            plan.registers.scratch_vgpr,
            plan.registers.pointer,
            plan.registers.timestamp,
            plan.registers.exec_save
        ),
        (191, 4, 6, 7)
    );
    let out = run(
        &co,
        BUILDER,
        SILU,
        plan.script,
        EquivalenceKind::ProvedCommutation,
    );
    assert_reproduces(
        &out.bytes,
        "b677527e75cd2efe2df769b70a6c3808c7304a17177e4f661830fa14af65c041",
        284_696,
        "peacemaker/profile/f2/silu_pmprofile.co",
    );
    assert_eq!(out.receipt.definedness, Ok(()));
    assert!(!out.receipt.new_obligations.iter().any(|o| o.rule_id == "definedness-partial-write"),
        "v191 lanes 0..7 are written before the header store under EXEC=0xff; later records rewrite lanes 0..1/3 under equally narrow masks");
}
