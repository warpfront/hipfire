// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The task-graph certificate (`railgun_cert::task_graph`) on the real PM pool
//! objects of MegaM0's P0 megakernel, and byte mutations of them.
//!
//! Fixtures (`tests/fixtures/task_graph/`, generator in its README): the pool
//! dispatchers `p0_r6_pool_b{32,64,128,256,320}.co` (each hosting lifted hipcc bodies as
//! regions, b256 also the native COPY region and the SGPR-parking template 5) and the
//! standalone objects they were lifted from, named `standalone/<sha256>.co`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use object::{Object, ObjectSection, ObjectSymbol};
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::inst::{Arch, Frontend};
use peacemaker_lift::regions_note::read_regions;
use peacemaker_lift::{lift_object, Options};
use railgun_cert::task_graph::certify_pool;
use railgun_cert::{KernelRecord, Receipt};
use railgun_corpus::receipt::{
    TaskGridOffsets, SEALED_BOUND_KERNARG, TASK_PROTO_ACQUIRE, TASK_PROTO_BOUNDED,
    TASK_PROTO_CLAIM, TASK_PROTO_COMPLETE, TASK_PROTO_ONCE, TASK_PROTO_REGION_WG,
    TASK_PROTO_RELEASE, TASK_PROTO_REQUIRED,
};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/task_graph")
}

fn pool(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(format!("p0_r6_pool_{name}.co"))).expect("fixture")
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_disk(digest: &[u8; 32]) -> Option<Vec<u8>> {
    std::fs::read(dir().join("standalone").join(format!("{}.co", hex(digest)))).ok()
}

fn kernel<'a>(r: &'a Receipt, symbol: &str) -> &'a KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .expect("record")
}

fn rules(k: &KernelRecord) -> Vec<&str> {
    k.unresolved
        .iter()
        .map(|u| u.rule.as_str())
        .filter(|r| r.starts_with("task_graph-"))
        .collect()
}

fn explain(k: &KernelRecord) -> String {
    k.unresolved
        .iter()
        .map(|u| format!("{}: {}", u.rule, u.detail))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Hosted template ids of each fixture pool (MegaM0's rows-6 table).
/// The pools every obligation is proven for.
const POOLS: &[(&str, &str, &[u32])] = &[
    ("b32", "pm_pool_b32", &[8, 10, 11, 17]),
    ("b64", "pm_pool_b64", &[6]),
    ("b128", "pm_pool_b128", &[7, 12, 16, 18]),
    ("b256", "pm_pool_b256", &[2, 3, 4, 5, 9, 13, 14, 15]),
    ("b320", "pm_pool_b320", &[19]),
];

/// Run the obligations on a fixture pool: the refusals, and the proof the pool
/// would have had if those refusals did not stand.
fn prove(name: &str, symbol: &str) -> (Vec<String>, Option<railgun_cert::task_graph::PoolProof>) {
    prove_bytes(&pool(name), symbol)
}

fn prove_bytes(
    bytes: &[u8],
    symbol: &str,
) -> (Vec<String>, Option<railgun_cert::task_graph::PoolProof>) {
    let lifted = lift_object(
        bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let notes = read_regions(bytes).expect("note");
    let note = notes.iter().find(|n| n.kernel == symbol).expect("note");
    let index = lifted
        .program
        .kernels
        .iter()
        .position(|k| k.symbol.0 == symbol)
        .expect("kernel");
    let mut partial = None;
    let result = railgun_cert::task_graph::prove_pool(
        &lifted.program,
        index,
        bytes,
        note,
        &from_disk,
        &mut partial,
    );
    let rules = match result {
        Ok(proof) => {
            assert!(partial.is_none());
            return (Vec::new(), Some(proof));
        }
        Err(rs) => rs
            .iter()
            .map(|r| format!("{}: {}", r.rule, r.detail))
            .collect(),
    };
    (rules, partial)
}

/// The loaded hidden grid words each real template carries (record offset = the
/// standalone kernarg offset); every template not listed loads none and so serializes
/// without the field (its receipt is byte-identical to before).
#[test]
fn real_templates_carry_their_hidden_grid_words() {
    use railgun_corpus::receipt::HiddenGridRecord;
    let h = |record_offset: u32, kind: &str| HiddenGridRecord {
        record_offset,
        kind: kind.into(),
    };
    let want: &[(&str, u32, Vec<HiddenGridRecord>)] = &[
        ("b32", 17, vec![h(100, "block_count_y")]),
        (
            "b256",
            4,
            vec![h(140, "group_size_x"), h(142, "group_size_y")],
        ),
        ("b256", 13, vec![h(80, "block_count_x")]),
        (
            "b256",
            3,
            vec![h(76, "group_size_x"), h(78, "group_size_y")],
        ),
        (
            "b256",
            14,
            vec![h(28, "group_size_x"), h(30, "group_size_y")],
        ),
        (
            "b256",
            15,
            vec![h(116, "group_size_x"), h(118, "group_size_y")],
        ),
    ];
    for &(name, symbol, _) in POOLS {
        let (refusals, partial) = prove(name, symbol);
        assert!(refusals.is_empty(), "{name}: {refusals:#?}");
        for t in &partial.expect("proof").arena.templates {
            let expect = want
                .iter()
                .find(|(n, id, _)| *n == name && *id == t.template)
                .map(|(_, _, v)| v.clone())
                .unwrap_or_default();
            assert_eq!(t.hidden_grid, expect, "{name} t{}", t.template);
            let json = serde_json::to_string(t).expect("json");
            assert_eq!(
                json.contains("hidden_grid"),
                !expect.is_empty(),
                "{name} t{}: {json}",
                t.template
            );
            // the footprint requirement `hidden_group_size_x == block` names the same word
            if name == "b256" && t.template == 14 {
                let f = t
                    .wg_footprints
                    .iter()
                    .find(|f| f.record_offset == 0)
                    .unwrap();
                assert_eq!(
                    f.requires,
                    [railgun_corpus::receipt::WgRequirementRecord {
                        offset: 28,
                        mask: 0xffff,
                        value: 256,
                        op: Default::default()
                    }]
                );
                assert!(t.hidden_grid.iter().any(|g| g.record_offset == 28));
            }
        }
    }
}

/// Every obligation holds on the OR-poison pools: a conditional record per pool.
#[test]
fn real_pools_are_certified() {
    for &(name, symbol, templates) in POOLS {
        let (refusals, partial) = prove(name, symbol);
        assert!(refusals.is_empty(), "{name}: {refusals:#?}");
        let proof = partial.expect("the proof");
        // item 2: the arena
        let arena = &proof.arena;
        assert_eq!(arena.arena_arg_offset, 24, "M0Params.kernargs");
        let ids: Vec<u32> = arena.templates.iter().map(|t| t.template).collect();
        assert_eq!(ids, templates, "{name}");
        for t in &arena.templates {
            assert!(
                !t.pointer_loads.is_empty(),
                "{name}: template {} loads no pointer",
                t.template
            );
        }
        // item 4: the protocol, derived from the bytes
        let p = &proof.protocol;
        // the six required properties and the optional seventh: the dispatcher runs
        // each region at (header[vx] + step, header[vy] | header[vz] << 16)
        let mut want: Vec<String> = TASK_PROTO_REQUIRED.iter().map(|s| s.to_string()).collect();
        want.push(TASK_PROTO_REGION_WG.to_string());
        want.sort();
        assert_eq!(p.proved, want, "{name}");
        assert_eq!(
            p.grid,
            Some(TaskGridOffsets {
                vx_offset: 4,
                vy_offset: 8,
                vz_offset: 12,
                gx_offset: 16,
                gy_offset: 20,
                gz_offset: 24,
                vcount_offset: 44,
            }),
            "{name}"
        );
        assert!(
            proof
                .evidence
                .iter()
                .any(|e| e.contains("region_wg_from_header: wg_x(s102)")),
            "{name}: {:#?}",
            proof.evidence
        );
        assert_eq!(
            [
                p.tasks_arg_offset,
                p.ticket_arg_offset,
                p.events_arg_offset,
                p.n_tasks_arg_offset
            ],
            [0, 8, 16, 56]
        );
        assert_eq!(
            [
                p.desc_stride,
                p.template_offset,
                p.ka_off_offset,
                p.pred_event_offset,
                p.expected_offset,
                p.done_event_offset,
                p.event_stride
            ],
            [128, 0, 28, 32, 36, 40, 4]
        );
        assert_eq!(p.none_event, u32::MAX);
        for name in [
            TASK_PROTO_ACQUIRE,
            TASK_PROTO_BOUNDED,
            TASK_PROTO_CLAIM,
            TASK_PROTO_COMPLETE,
            TASK_PROTO_ONCE,
            TASK_PROTO_RELEASE,
        ] {
            assert!(p.proved.iter().any(|x| x == name), "{name}");
        }
        // the receipt: a conditional record, well-formed under the object's digest
        let receipt = certify_pool(&pool(name), &from_disk).expect("certify");
        let k = kernel(&receipt, symbol);
        assert_eq!(k.status, "conditional", "{name}:\n{}", explain(k));
        assert!(k.unresolved.is_empty(), "{name}:\n{}", explain(k));
        assert_eq!(k.descriptor_arenas, [proof.arena.clone()]);
        assert_eq!(k.task_protocol.as_ref(), Some(&proof.protocol));
        let want: &[&str] = if arena.templates.iter().any(|t| !t.sealed_tables.is_empty()) {
            &["descriptor_arena", "sealed_table"]
        } else {
            &["descriptor_arena"]
        };
        assert_eq!(k.required_witnesses, want, "{name}");
        k.validate_case(
            &receipt.object_sha256,
            &receipt.device_elf_sha256,
            &receipt.arch,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// b128 hosts two MoE bodies (templates 16, 18) whose standalone A1 record is
/// `conditional` with a sealed table (the expert pointers come from a
/// loader-written table): each template of the pool record carries it, at the
/// record offset of its table argument with the entry-count scalar's record
/// offset as bound, and the pool record requires the `sealed_table` witness
/// besides `descriptor_arena`.
#[test]
fn b128_carries_the_sealed_tables_of_its_two_moe_bodies() {
    let (refusals, partial) = prove("b128", "pm_pool_b128");
    assert!(refusals.is_empty(), "{refusals:#?}");
    let proof = partial.expect("the proof");
    for t in &proof.arena.templates {
        let sealed = !t.sealed_tables.is_empty();
        assert_eq!(
            sealed,
            [16, 18].contains(&t.template),
            "template {}",
            t.template
        );
        if sealed {
            assert_eq!(t.sealed_tables.len(), 1, "template {}", t.template);
            let s = &t.sealed_tables[0];
            assert_eq!(s.table_arg_offset, 0, "template {}", t.template);
            assert_eq!(
                s.bound_kind, SEALED_BOUND_KERNARG,
                "template {}",
                t.template
            );
            // The table argument is loaded from the record, so it is a pointer load.
            assert!(
                t.pointer_loads.iter().any(|p| p.record_offset == 0),
                "template {}",
                t.template
            );
            // The bound is a scalar of the record, not any pointer field.
            assert!(
                t.pointer_loads
                    .iter()
                    .all(|p| u64::from(p.record_offset) + 8 <= s.bound_value
                        || s.bound_value + 4 <= u64::from(p.record_offset)),
                "template {}",
                t.template
            );
        }
    }
    let receipt = certify_pool(&pool("b128"), &from_disk).expect("certify");
    let k = kernel(&receipt, "pm_pool_b128");
    assert_eq!(k.status, "conditional", "{}", explain(k));
    assert_eq!(k.required_witnesses, ["descriptor_arena", "sealed_table"]);
    k.validate_case(
        &receipt.object_sha256,
        &receipt.device_elf_sha256,
        &receipt.arch,
    )
    .unwrap();
}

/// The standalone sealed-table proofs assume the region is entered with the
/// kernel-entry EXEC (every lane). The dispatcher narrows EXEC to lane 0 for
/// its claim code and restores it before each region: a pool that keeps EXEC
/// narrowed (lane 0, zero or unknown) at a region entry is refused.
#[test]
fn a_region_entered_without_the_kernel_entry_exec_is_refused() {
    for (name, symbol) in [("b32", "pm_pool_b32"), ("b128", "pm_pool_b128")] {
        let base = pool(name);
        let at = locate(&base, symbol);
        let restore = words_of("s_mov_b32 exec_lo, -1")[0];
        let pcs: Vec<u32> = instructions(&base, symbol)
            .into_iter()
            .filter(|(pc, dwords, m)| {
                m == "s_mov_b32" && *dwords == 1 && {
                    let o = at.file_at + *pc as usize;
                    u32::from_le_bytes(base[o..o + 4].try_into().unwrap()) == restore
                }
            })
            .map(|(pc, _, _)| pc)
            .collect();
        // One restore per path the dispatcher takes to a region: mutating the
        // first leaves some region entered with the narrowed EXEC.
        assert!(!pcs.is_empty(), "{name}: the dispatcher restores EXEC");
        let variants: &[&str] = if name == "b32" {
            &[
                "s_mov_b32 exec_lo, 1",
                "s_mov_b32 exec_lo, 0",
                "s_mov_b32 exec_lo, 5",
            ]
        } else {
            &["s_mov_b32 exec_lo, 1"]
        };
        for narrowed in variants {
            let mut bytes = base.clone();
            patch(&mut bytes, &at, pcs[0], &words_of(narrowed));
            let r = certified(&bytes, symbol);
            expect_refused(&r, symbol, "task_graph-region-exec");
            let detail = explain(kernel(&r, symbol));
            assert!(
                detail.contains("not the kernel-entry EXEC"),
                "{name} {narrowed}: {detail}"
            );
        }
        // Dropping the restore altogether (a nop) is the same.
        if name == "b32" {
            let mut bytes = base.clone();
            nop_out(&mut bytes, symbol, &at, pcs[0], 1);
            expect_refused(&certified(&bytes, symbol), symbol, "task_graph-region-exec");
        }
    }
}

#[test]
fn native_copy_region_is_analysed_from_its_own_instructions() {
    let (_, partial) = prove("b256", "pm_pool_b256");
    let proof = partial.expect("proof");
    let copy = proof
        .arena
        .templates
        .iter()
        .find(|t| t.template == 2)
        .expect("COPY template");
    // M0CopyArgs { u64 dst, u64 src, u64 bytes }: dst is written, src read
    let loads: BTreeMap<u32, Vec<String>> = copy
        .pointer_loads
        .iter()
        .map(|p| (p.record_offset, p.classes.clone()))
        .collect();
    assert_eq!(loads[&0], ["vmem_store".to_string()]);
    assert_eq!(loads[&8], ["vmem_load".to_string()]);
}

// ---------------------------------------------------------------------------
// mutations
// ---------------------------------------------------------------------------

struct Located {
    /// file offset of the kernel's first code byte
    file_at: usize,
    /// its address in the object's text (what a lifted instruction's `pc` counts from)
    addr: u64,
}

fn locate(elf: &[u8], symbol: &str) -> Located {
    let file = object::File::parse(elf).expect("elf");
    let sym = file
        .symbols()
        .find(|s| s.name() == Ok(symbol) && !s.is_undefined())
        .expect("symbol");
    let section = file.section_by_index(sym.section_index().unwrap()).unwrap();
    let (start, _) = section.file_range().expect("file range");
    Located {
        file_at: (start + sym.address() - section.address()) as usize,
        addr: sym.address(),
    }
}

fn words_of(line: &str) -> Vec<u32> {
    let inst = peacemaker_lift::text::parse_line(line, Arch::Gfx1201).expect("parse");
    gfx12::encode(&inst).expect("encode").to_vec()
}

fn patch(elf: &mut [u8], at: &Located, pc: u32, words: &[u32]) {
    for (i, w) in words.iter().enumerate() {
        let o = at.file_at + pc as usize + 4 * i;
        elf[o..o + 4].copy_from_slice(&w.to_le_bytes());
    }
}

/// `(pc, dwords, mnemonic)` of every instruction of the kernel, in layout order.
fn instructions(elf: &[u8], symbol: &str) -> Vec<(u32, usize, String)> {
    let lifted = lift_object(
        elf,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let k = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .expect("kernel");
    let base = locate(elf, symbol).addr as u32;
    let mut out = Vec::new();
    for (i, &id) in k.body.layout.iter().enumerate() {
        let inst = k.body.insts.get(id).unwrap();
        let pc = inst.prov.pc.unwrap();
        let next = k
            .body
            .layout
            .get(i + 1)
            .and_then(|&n| k.body.insts.get(n))
            .and_then(|n| n.prov.pc)
            .unwrap_or(pc + 4);
        out.push((
            pc - base,
            ((next - pc) / 4) as usize,
            inst.op.name(Arch::Gfx1201).unwrap_or("").to_owned(),
        ));
    }
    out
}

fn nop_out(elf: &mut [u8], symbol: &str, at: &Located, pc: u32, dwords: usize) {
    let nop = words_of("s_nop 0")[0];
    patch(elf, at, pc, &vec![nop; dwords]);
    let _ = symbol;
}

fn region_range(elf: &[u8], symbol: &str, template: u32) -> (u64, u64) {
    let notes = read_regions(elf).expect("note");
    let n = notes.iter().find(|n| n.kernel == symbol).expect("kernel");
    let r = n
        .regions
        .iter()
        .find(|r| r.template_id == template)
        .expect("region");
    (r.entry_pc, r.exit_pc)
}

fn certified(bytes: &[u8], symbol: &str) -> Receipt {
    let _ = symbol;
    certify_pool(bytes, &from_disk).expect("certify")
}

fn expect_refused(receipt: &Receipt, symbol: &str, rule: &str) {
    let k = kernel(receipt, symbol);
    assert_eq!(k.status, "unknown", "must stay unknown:\n{}", explain(k));
    assert!(k.task_protocol.is_none() && k.descriptor_arenas.is_empty());
    assert!(
        rules(k).contains(&rule),
        "expected {rule}, got:\n{}",
        explain(k)
    );
}

#[test]
fn a_flipped_region_byte_is_refused() {
    let symbol = "pm_pool_b32";
    let mut bytes = pool("b32");
    let at = locate(&bytes, symbol);
    let (entry, _) = region_range(&bytes, symbol, 10);
    // the region's first instruction is `s_wait_alu 0x0`: make it `s_wait_alu 0x1`
    let w = words_of("s_wait_alu 0x0")[0];
    patch(&mut bytes, &at, entry as u32, &[w ^ 1]);
    let r = certified(&bytes, symbol);
    expect_refused(&r, symbol, "task_graph-region-bytes");
}

#[test]
fn a_missing_standalone_source_is_refused() {
    let symbol = "pm_pool_b128";
    let bytes = pool("b128");
    let notes = read_regions(&bytes).unwrap();
    let digest = notes[0]
        .regions
        .iter()
        .find(|r| r.template_id == 16)
        .unwrap()
        .digest;
    let r = certify_pool(&bytes, &|d: &[u8; 32]| {
        if *d == digest {
            None
        } else {
            from_disk(d)
        }
    })
    .expect("certify");
    expect_refused(&r, symbol, "task_graph-region-source-missing");
}

#[test]
fn a_standalone_object_with_the_wrong_digest_is_refused() {
    let symbol = "pm_pool_b32";
    let bytes = pool("b32");
    let notes = read_regions(&bytes).unwrap();
    let digest = notes[0].regions[0].digest;
    let r = certify_pool(&bytes, &|d: &[u8; 32]| {
        if *d == digest {
            // another object entirely
            std::fs::read(dir().join(format!("p0_r6_pool_b128.co"))).ok()
        } else {
            from_disk(d)
        }
    })
    .expect("certify");
    expect_refused(&r, symbol, "task_graph-region-source-digest");
}

#[test]
fn removing_the_pred_acquire_fence_is_refused() {
    for (name, symbol) in [("b32", "pm_pool_b32"), ("b128", "pm_pool_b128")] {
        let mut bytes = pool(name);
        let at = locate(&bytes, symbol);
        let insts = instructions(&bytes, symbol);
        // the dispatcher's first `global_inv` is the acquire after the predecessor poll
        let (pc, dwords, _) = insts
            .iter()
            .find(|(_, _, m)| m == "global_inv")
            .cloned()
            .expect("global_inv");
        nop_out(&mut bytes, symbol, &at, pc, dwords);
        let r = certified(&bytes, symbol);
        let k = kernel(&r, symbol);
        assert_eq!(k.status, "unknown", "{name}");
        assert!(k.task_protocol.is_none(), "{name}");
        let rs = rules(k);
        assert!(
            rs.contains(&"task_graph-protocol-entry")
                || rs.contains(&"task_graph-shell-unresolved")
                || rs.contains(&"task_graph-shell-access"),
            "{name}: {}",
            explain(k)
        );
    }
}

#[test]
fn removing_the_done_increments_release_waits_is_refused() {
    for (name, symbol) in [("b32", "pm_pool_b32"), ("b128", "pm_pool_b128")] {
        let mut bytes = pool(name);
        let at = locate(&bytes, symbol);
        let insts = instructions(&bytes, symbol);
        // the one `global_atomic_add_u32` without return is the done increment; its
        // release is the two counter waits right before it
        let pos = insts
            .iter()
            .rposition(|(_, _, m)| m == "global_atomic_add_u32")
            .expect("atomic");
        for (pc, dwords, m) in insts[pos.saturating_sub(4)..pos].iter() {
            if m == "s_wait_loadcnt" || m == "s_wait_storecnt" {
                nop_out(&mut bytes, symbol, &at, *pc, *dwords);
            }
        }
        let r = certified(&bytes, symbol);
        let k = kernel(&r, symbol);
        assert_eq!(k.status, "unknown", "{name}");
        assert!(k.task_protocol.is_none(), "{name}");
        let rs = rules(k);
        assert!(
            rs.contains(&"task_graph-release") || rs.contains(&"task_graph-shell-access"),
            "{name}: {}",
            explain(k)
        );
    }
}

#[test]
fn a_branch_into_the_middle_of_a_region_is_refused() {
    for (name, symbol, template) in [("b32", "pm_pool_b32", 10), ("b128", "pm_pool_b128", 12)] {
        let mut bytes = pool(name);
        let at = locate(&bytes, symbol);
        let (entry, exit) = region_range(&bytes, symbol, template);
        let insts = instructions(&bytes, symbol);
        // retarget a dispatcher `s_branch` (outside every region) at the region's
        // second instruction
        let notes = read_regions(&bytes).unwrap();
        let inside = |pc: u32| {
            notes[0]
                .regions
                .iter()
                .any(|r| (r.entry_pc..r.exit_pc).contains(&u64::from(pc)))
        };
        let (pc, _, _) = insts
            .iter()
            .find(|(pc, _, m)| m == "s_branch" && !inside(*pc))
            .cloned()
            .expect("s_branch");
        let target = entry as u32 + 4;
        assert!(u64::from(target) < exit);
        let offset = (i64::from(target) - i64::from(pc) - 4) / 4;
        let old = u32::from_le_bytes(
            bytes[at.file_at + pc as usize..at.file_at + pc as usize + 4]
                .try_into()
                .unwrap(),
        );
        patch(
            &mut bytes,
            &at,
            pc,
            &[(old & 0xffff_0000) | (offset as i16 as u16 as u32)],
        );
        let r = certified(&bytes, symbol);
        expect_refused(&r, symbol, "task_graph-region-structure");
    }
}

/// The poison contract: the watchdog's poison is a set-only atomic OR with release. A plain
/// store or a swap in its place (same length: the OR's three dwords) is refused.
#[test]
fn a_plain_store_or_a_swap_as_poison_is_refused() {
    for (what, line, rule) in [
        (
            "plain store",
            "global_store_b32 v1, v2, s[78:79]",
            "task_graph-poison-plain-store",
        ),
        (
            "swap",
            "global_atomic_swap_b32 v1, v2, s[78:79] scope:SCOPE_DEV",
            "task_graph-poison-swap",
        ),
    ] {
        for (name, symbol) in [("b32", "pm_pool_b32"), ("b128", "pm_pool_b128")] {
            let mut bytes = pool(name);
            let at = locate(&bytes, symbol);
            let insts = instructions(&bytes, symbol);
            let (pc, dwords, _) = insts
                .iter()
                .find(|(_, _, m)| m == "global_atomic_or_b32")
                .cloned()
                .expect("the poison OR");
            let words = words_of(line);
            assert_eq!(words.len(), dwords, "{what}: same length as the OR");
            patch(&mut bytes, &at, pc, &words);
            let r = certified(&bytes, symbol);
            let k = kernel(&r, symbol);
            assert_eq!(k.status, "unknown", "{name} {what}");
            assert!(k.task_protocol.is_none(), "{name} {what}");
            assert!(
                rules(k).contains(&rule),
                "{name} {what}: expected {rule}, got:\n{}",
                explain(k)
            );
        }
    }
}

/// Region entry needs every counter drained: remove the drain in front of a region and the
/// pool is refused. `wide = false` removes the counter waits right before the entry;
/// `wide = true` removes every dispatcher counter wait before it (a pool whose earlier waits
/// already leave the counters drained at that entry needs the earlier ones gone too).
#[test]
fn a_region_entered_without_the_counter_drain_is_refused() {
    for (name, symbol, template, wide) in [
        ("b32", "pm_pool_b32", 10, true),
        ("b32", "pm_pool_b32", 11, true),
        ("b128", "pm_pool_b128", 12, true),
        ("b256", "pm_pool_b256", 13, true),
    ] {
        let mut bytes = pool(name);
        let at = locate(&bytes, symbol);
        let (entry, _) = region_range(&bytes, symbol, template);
        let notes = read_regions(&bytes).expect("note");
        let regions = &notes.iter().find(|n| n.kernel == symbol).unwrap().regions;
        let in_region = |pc: u32| {
            regions
                .iter()
                .any(|r| (r.entry_pc..r.exit_pc).contains(&u64::from(pc)))
        };
        let insts = instructions(&bytes, symbol);
        let drain: Vec<(u32, usize)> = insts
            .iter()
            .filter(|(pc, _, m)| {
                let near = u64::from(*pc) + 0x20 >= entry;
                u64::from(*pc) < entry
                    && (near || (wide && !in_region(*pc)))
                    && [
                        "s_wait_loadcnt",
                        "s_wait_storecnt",
                        "s_wait_dscnt",
                        "s_wait_kmcnt",
                    ]
                    .iter()
                    .any(|w| m.starts_with(w))
            })
            .map(|(pc, d, _)| (*pc, *d))
            .collect();
        assert!(
            drain.len() >= 2,
            "{name} t{template}: the drain before the entry: {drain:?}"
        );
        for (pc, dwords) in drain {
            nop_out(&mut bytes, symbol, &at, pc, dwords);
        }
        let r = certified(&bytes, symbol);
        let k = kernel(&r, symbol);
        assert_eq!(k.status, "unknown", "{name} t{template}");
        assert!(k.task_protocol.is_none(), "{name} t{template}");
        assert!(
            rules(k).contains(&"task_graph-region-drain"),
            "{name} t{template}: {}",
            explain(k)
        );
    }
}

// ---------------------------------------------------------------------------
// the SGPR park around a region that overwrites dispatcher registers
// ---------------------------------------------------------------------------

/// b256's gated-delta body (template 5) writes s72, s73 and s74..s79 of the dispatcher range
/// (its 80-SGPR working set). The dispatcher parks s72 and s74..s79 in lanes 0 and 2..7 of
/// v119 before the entry (`v_writelane_b32 v119, sK, lane`) and restores them after the exit
/// (`v_readlane_b32 sK, v119, lane`); s73 is dead (no path reads it before writing it).
const PARK_POOL: (&str, &str) = ("b256", "pm_pool_b256");

fn park_refusals(bytes: &[u8]) -> Vec<String> {
    let receipt = certified(bytes, PARK_POOL.1);
    expect_refused(&receipt, PARK_POOL.1, "task_graph-region-sgpr-park");
    kernel(&receipt, PARK_POOL.1)
        .unresolved
        .iter()
        .filter(|u| u.rule == "task_graph-region-sgpr-park")
        .map(|u| u.detail.clone())
        .collect()
}

#[test]
fn a_parked_region_proves_its_restore_values() {
    let (refusals, partial) = prove(PARK_POOL.0, PARK_POOL.1);
    assert!(refusals.is_empty(), "{refusals:#?}");
    let proof = partial.expect("proof");
    // the grid x register s72 is parked and restored across template 5: the header-derived
    // value reaches the next trip's entry, so the optional property still holds
    assert!(proof
        .protocol
        .proved
        .iter()
        .any(|p| p == TASK_PROTO_REGION_WG));
    let bytes = pool(PARK_POOL.0);
    for (line, n) in [
        ("v_writelane_b32 v119, s72, 0", 1),
        ("v_writelane_b32 v119, s79, 7", 1),
        ("v_readlane_b32 s72, v119, 0", 1),
        ("v_readlane_b32 s79, v119, 7", 1),
    ] {
        assert_eq!(find(&bytes, PARK_POOL.1, line).len(), n, "{line}");
    }
}

/// The region overwrites the parking VGPR (here by turning a one-dword `s_nop` inside it
/// into `v_mov_b32 v119, 0`): the restore would read the region's value.
#[test]
fn a_region_that_writes_the_parking_vgpr_is_refused() {
    let (symbol, mut bytes) = (PARK_POOL.1, pool(PARK_POOL.0));
    let (entry, exit) = region_range(&bytes, symbol, 5);
    let at = locate(&bytes, symbol);
    let nop = find(&bytes, symbol, "s_nop 0")
        .into_iter()
        .find(|&pc| (entry..exit).contains(&u64::from(pc)))
        .expect("an s_nop inside template 5");
    let write = words_of("v_mov_b32_e32 v119, 0");
    assert_eq!(write.len(), 1);
    patch(&mut bytes, &at, nop, &write);
    let r = park_refusals(&bytes);
    assert!(
        r.iter().any(|d| d.contains("s74")
            && d.contains("v119")
            && d.contains("which the region writes")),
        "{r:#?}"
    );
}

/// One restore removed: the dispatcher's next trip parks (reads) the clobbered register.
#[test]
fn a_missing_restore_of_one_register_is_refused() {
    let (symbol, mut bytes) = (PARK_POOL.1, pool(PARK_POOL.0));
    let at = locate(&bytes, symbol);
    let pc = find(&bytes, symbol, "v_readlane_b32 s76, v119, 4");
    assert_eq!(pc.len(), 1);
    nop_out(&mut bytes, symbol, &at, pc[0], 2);
    let r = park_refusals(&bytes);
    assert!(
        r.iter()
            .any(|d| d.contains("reads s76") && d.contains("not restored")),
        "{r:#?}"
    );
}

/// A restore from another lane than the one the register was parked in.
#[test]
fn a_restore_from_the_wrong_lane_is_refused() {
    let symbol = PARK_POOL.1;
    let bytes = mutate(
        PARK_POOL.0,
        symbol,
        "v_readlane_b32 s75, v119, 3",
        "v_readlane_b32 s75, v119, 4",
        0,
    );
    let r = park_refusals(&bytes);
    assert!(
        r.iter()
            .any(|d| d.contains("restores s75 from v119 lane 4") && d.contains("v119 lane 3")),
        "{r:#?}"
    );
}

/// A register the region writes and the dispatcher restores, but never parked.
#[test]
fn a_restore_of_a_register_that_was_never_parked_is_refused() {
    let (symbol, mut bytes) = (PARK_POOL.1, pool(PARK_POOL.0));
    let at = locate(&bytes, symbol);
    let pc = find(&bytes, symbol, "v_writelane_b32 v119, s77, 5");
    assert_eq!(pc.len(), 1);
    nop_out(&mut bytes, symbol, &at, pc[0], 2);
    let r = park_refusals(&bytes);
    assert!(
        r.iter()
            .any(|d| d.contains("s77") && d.contains("no lane on every path")),
        "{r:#?}"
    );
}

// ---------------------------------------------------------------------------
// region_wg_from_header
// ---------------------------------------------------------------------------

/// The file offsets (kernel-relative pcs) of every instruction that encodes exactly `line`.
fn find(elf: &[u8], symbol: &str, line: &str) -> Vec<u32> {
    let at = locate(elf, symbol);
    let words = words_of(line);
    instructions(elf, symbol)
        .into_iter()
        .filter(|(pc, dwords, _)| {
            *dwords == words.len() && {
                let o = at.file_at + *pc as usize;
                words.iter().enumerate().all(|(i, w)| {
                    u32::from_le_bytes(elf[o + 4 * i..o + 4 * i + 4].try_into().unwrap()) == *w
                })
            }
        })
        .map(|(pc, _, _)| pc)
        .collect()
}

/// `from` -> `to` (the same length) at the `nth` place the dispatcher encodes `from`.
fn mutate(name: &str, symbol: &str, from: &str, to: &str, nth: usize) -> Vec<u8> {
    let mut bytes = pool(name);
    let at = locate(&bytes, symbol);
    let pcs = find(&bytes, symbol, from);
    assert!(pcs.len() > nth, "{name}: no `{from}` (found {pcs:?})");
    let words = words_of(to);
    assert_eq!(
        words.len(),
        words_of(from).len(),
        "`{to}` is as long as `{from}`"
    );
    patch(&mut bytes, &at, pcs[nth], &words);
    bytes
}

/// The pool still proves its six required properties (the mutation lives in the
/// optional seventh) but not `region_wg_from_header`, for a stated reason.
fn expect_wg_withheld(bytes: &[u8], symbol: &str, why: &str, what: &str) {
    let (refusals, partial) = prove_bytes(bytes, symbol);
    assert!(refusals.is_empty(), "{what}: {refusals:#?}");
    let proof = partial.expect("the proof");
    let mut want: Vec<String> = TASK_PROTO_REQUIRED.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(proof.protocol.proved, want, "{what}");
    assert!(proof.protocol.grid.is_none(), "{what}");
    let note = proof
        .evidence
        .iter()
        .find(|e| e.contains("region_wg_from_header NOT proved"))
        .unwrap_or_else(|| panic!("{what}: no withheld note in {:#?}", proof.evidence));
    assert!(note.contains(why), "{what}: expected `{why}` in:\n{note}");
    // the record still validates
    proof.protocol.validate(symbol).expect("validates");
}

/// The certified receipt of a pool whose `region_wg_from_header` is withheld stays
/// `conditional` and carries the record without the property.
fn expect_conditional_without_wg(bytes: &[u8], symbol: &str, what: &str) {
    let receipt = certify_pool(bytes, &from_disk).expect("certify");
    let k = kernel(&receipt, symbol);
    assert_eq!(k.status, "conditional", "{what}:\n{}", explain(k));
    let record = k.task_protocol.as_ref().expect("record");
    assert!(
        !record.proved.iter().any(|p| p == TASK_PROTO_REGION_WG) && record.grid.is_none(),
        "{what}"
    );
}

/// Every wg_x derivation the property rests on is a mutation target: each one must
/// withhold the property on the single-wave pool and the small multi-wave pool (whose
/// header reaches the regions through the LDS slot); the large multi-wave and the
/// native-COPY pools take the two mutations that differ there (the step add and the
/// counter start).
#[test]
fn region_wg_is_withheld_when_the_dispatcher_does_not_derive_it_from_the_header() {
    let pools = [
        ("b32", "pm_pool_b32"),
        ("b64", "pm_pool_b64"),
        ("b128", "pm_pool_b128"),
        ("b256", "pm_pool_b256"),
    ];
    // (from, to, expected reason)
    let cases: &[(&str, &str, &str)] = &[
        // s102 = vx without + step
        (
            "s_add_co_i32 s102, s96, s101",
            "s_mov_b32 s102, s96",
            "wg_x register s102 is not `header[vx] + step",
        ),
        // step counter initialised to 1: the first step never runs
        (
            "s_mov_b32 s101, 0",
            "s_mov_b32 s101, 1",
            "wg_x register s102",
        ),
        // the counter steps by 2: steps are skipped
        (
            "s_add_co_i32 s101, s101, 1",
            "s_add_co_i32 s101, s101, 2",
            "step loop",
        ),
        // one step too many: the compare admits step == bound
        (
            "s_cmp_lt_u32 s101, s100",
            "s_cmp_le_u32 s101, s100",
            "wg_x register s102",
        ),
        // s103 from the wrong header field (vx for vy): two fields on one offset
        (
            "v_lshl_or_b32 v6, v7, 16, v6",
            "v_lshl_or_b32 v6, v7, 16, v5",
            "two header fields resolve to offset 4",
        ),
        // s103 packed with the wrong shift
        (
            "v_lshl_or_b32 v6, v7, 16, v6",
            "v_lshl_or_b32 v6, v7, 8, v6",
            "wg_yz register s103",
        ),
        // s103 = vx, not the packed vy | vz << 16
        (
            "s_mov_b32 s103, s97",
            "s_mov_b32 s103, s96",
            "wg_yz register s103 is not",
        ),
        // the loop bound is the done event, not vcount
        (
            "v_readfirstlane_b32 s100, v15",
            "v_readfirstlane_b32 s100, v14",
            "two header fields resolve to offset 40",
        ),
    ];
    for &(name, symbol) in &pools {
        let n = if name == "b32" || name == "b64" {
            cases.len()
        } else {
            2
        };
        for &(from, to, why) in &cases[..n] {
            let bytes = mutate(name, symbol, from, to, 0);
            expect_wg_withheld(&bytes, symbol, why, &format!("{name}: {from} -> {to}"));
        }
    }
}

/// The counter written anywhere else in the loop loses the induction fact: a second
/// increment (the `s_wait_alu` ahead of the latch's add turned into one).
#[test]
fn a_second_write_of_the_step_counter_withholds_region_wg() {
    for (name, symbol) in [("b32", "pm_pool_b32"), ("b128", "pm_pool_b128")] {
        let base = pool(name);
        let add = find(&base, symbol, "s_add_co_i32 s101, s101, 1")[0];
        let insts = instructions(&base, symbol);
        let at = insts.iter().position(|(pc, _, _)| *pc == add).unwrap();
        let (wait_pc, dwords, m) = insts[at - 1].clone();
        assert_eq!((dwords, m.as_str()), (1, "s_wait_alu"), "{name}");
        let mut bytes = base.clone();
        let loc = locate(&bytes, symbol);
        patch(
            &mut bytes,
            &loc,
            wait_pc,
            &words_of("s_add_co_i32 s101, s101, 1"),
        );
        expect_wg_withheld(&bytes, symbol, "step loop", name);
        expect_conditional_without_wg(&bytes, symbol, name);
    }
}

/// `s_max_u32 s100, s100, 1` is redundant for the property: the loop is a do-while
/// (the region runs, then the compare), so `vcount == 0` runs one step either way
/// and the steps are `[0, max(vcount, 1))` with or without it. Removing it keeps
/// the proof (and the same offsets), a bound that is not the header's vcount does
/// not.
#[test]
fn the_max_one_on_the_loop_bound_does_not_change_what_is_proved() {
    for (name, symbol) in [("b32", "pm_pool_b32"), ("b64", "pm_pool_b64")] {
        let (refusals, partial) = prove(name, symbol);
        assert!(refusals.is_empty(), "{name}: {refusals:#?}");
        let with = partial.expect("proof").protocol;
        let mut bytes = pool(name);
        let at = locate(&bytes, symbol);
        let pcs = find(&bytes, symbol, "s_max_u32 s100, s100, 1");
        assert_eq!(pcs.len(), 1, "{name}");
        patch(&mut bytes, &at, pcs[0], &words_of("s_nop 0"));
        let (refusals, partial) = prove_bytes(&bytes, symbol);
        assert!(refusals.is_empty(), "{name}: {refusals:#?}");
        assert_eq!(partial.expect("proof").protocol, with, "{name}");
        // ... but a constant bound is no header field
        let bytes = mutate(
            name,
            symbol,
            "s_max_u32 s100, s100, 1",
            "s_mov_b32 s100, 2",
            0,
        );
        expect_wg_withheld(&bytes, symbol, "wg_x register s102", name);
    }
}
