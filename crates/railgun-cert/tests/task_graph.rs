// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The task-graph certificate (`railgun_cert::task_graph`) on the real PM pool
//! objects of MegaM0's P0 megakernel, and byte mutations of them.
//!
//! Fixtures (`tests/fixtures/task_graph/`, generator in its README): the pool
//! dispatchers `p0_r6_pool_b{32,128,256}.co` (each hosting lifted hipcc bodies as
//! regions, b256 also the native COPY region) and the standalone objects they were
//! lifted from, named `standalone/<sha256>.co`.

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
    TASK_PROTO_ACQUIRE, TASK_PROTO_BOUNDED, TASK_PROTO_CLAIM, TASK_PROTO_COMPLETE, TASK_PROTO_ONCE,
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
    ("b256", "pm_pool_b256", &[2, 4, 9, 13, 14, 15]),
];

/// b128 hosts two MoE bodies (templates 16, 18) whose standalone A1 record is not
/// proven (their expert pointers are loaded from tables): the pool stays unknown, with
/// exactly those two refusals, and nothing else (the dispatcher's shell and protocol,
/// the other two regions, are proven).
/// Run the obligations on a fixture pool: the refusals, and the proof the pool would
/// have had if those refusals did not stand.
fn prove(name: &str, symbol: &str) -> (Vec<String>, Option<railgun_cert::task_graph::PoolProof>) {
    let bytes = pool(name);
    let lifted = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let notes = read_regions(&bytes).expect("note");
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
        &bytes,
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
        let mut want: Vec<String> = TASK_PROTO_REQUIRED.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(p.proved, want);
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
        assert_eq!(k.required_witnesses, ["descriptor_arena"]);
        k.validate_case(
            &receipt.object_sha256,
            &receipt.device_elf_sha256,
            &receipt.arch,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// b128 hosts two MoE bodies (templates 16, 18) whose standalone A1 record is not
/// proven (their expert pointers are loaded from tables): two more refusals, nothing else
/// (the dispatcher's shell and protocol and the other two regions are proven).
#[test]
fn b128_is_refused_on_its_two_unproven_moe_bodies_and_the_poison_store() {
    let (refusals, _) = prove("b128", "pm_pool_b128");
    let kinds: Vec<&str> = refusals
        .iter()
        .map(|r| r.split(':').next().unwrap())
        .collect();
    assert_eq!(
        kinds,
        ["task_graph-region-arena", "task_graph-region-arena"],
        "{refusals:#?}"
    );
    assert!(refusals[0].contains("template 16:") && refusals[1].contains("template 18:"));
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
