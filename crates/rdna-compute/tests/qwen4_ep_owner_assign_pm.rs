// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next EP balanced-ownership assignment builder module (gfx1201): the PM
//! twin of the host reference `qwen4_ep_owner_map` / `Gpu::qwen4_ep_owner_assign`.
//! The emitted program runs on the PeaceMaker CPU interpreter (one 512-thread
//! workgroup, two barriers) and its output bytes (live pointer tables and slot
//! owners) must equal the host reference for every input. No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::qwen4_ep_owner_assign_pm as owners;
use peacemaker_isa::Arch;
use rdna_compute::moe::{qwen4_ep_owner_map, QWEN4_EP_OWNER_EXPERTS};
use std::sync::LazyLock;
#[macro_use]
#[path = "../../peacemaker-isa/tests/support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const EXPERTS: usize = QWEN4_EP_OWNER_EXPERTS;

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    let (a, b) = (owners::emit(ARCH).unwrap(), owners::emit(ARCH).unwrap());
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert!(a.s_text.contains(&format!("{}:", owners::symbol(ARCH))));
    assert_eq!(a.shape.barriers, 2);
    assert!(
        a.shape.next_free_vgpr <= 48 && a.shape.next_free_sgpr <= 104,
        "{:?}",
        a.shape
    );
    let (emitted, text, _) = owners::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert!(text.contains(&format!("{}:", owners::symbol(ARCH))));
    assert_eq!(owners::module(ARCH), "qwen4_ep_owner_assign_pm_gfx1201");
    assert_eq!(owners::symbol(ARCH), "qwen4_ep_owner_assign_pm_gfx1201");
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(owners::emit(arch).is_err(), "{arch:?}");
        assert!(owners::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Kernarg ABI of `Gpu::qwen4_ep_owner_assign`: six pointers, two u64 dummies, three i32 (76 bytes);
/// 512-thread workgroup, static LDS, no atomics, no scratch, no flat / global access.
#[test]
fn kernarg_layout_matches_the_launcher() {
    let text = owners::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let offsets: Vec<u32> = metadata
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse::<u32>().unwrap())
        })
        .collect();
    assert_eq!(offsets, [0, 8, 16, 24, 32, 40, 48, 56, 64, 68, 72]);
    assert!(text.contains(".amdhsa_kernarg_size 76\n"), "kernarg size");
    assert!(
        text.contains(&format!(
            ".amdhsa_group_segment_fixed_size {}\n",
            owners::STATIC_LDS
        )),
        "LDS"
    );
    assert_eq!(owners::STATIC_LDS, 2112);
    assert!(
        text.contains(".max_flat_workgroup_size: 512"),
        "workgroup size"
    );
    assert!(!text.contains("atomic"));
    for l in text.lines().map(str::trim_start) {
        assert!(
            !(l.starts_with("scratch_") || l.starts_with("flat_") || l.starts_with("global_")),
            "{l}"
        );
    }
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = owners::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = owners::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle"
    );
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-ep-owner-assign-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

// ---- CPU interpreter vs the host reference ------------------------------------

static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| {
    let text = owners::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
    .program
});

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const TAIL: usize = 16;

/// One launch. Returns `(live_gate_up, live_down, slot_owner)` and asserts the slot-owner region's
/// untouched tail (`slot_owner` holds exactly `max(slots, 1)` words plus a sentinel tail).
fn run(
    ids: &[i32],
    ranks: usize,
    self_rank: usize,
    full_gu: &[u64],
    full_dn: &[u64],
    dummy_gu: u64,
    dummy_dn: u64,
) -> (Vec<u64>, Vec<u64>, Vec<i32>) {
    let slots = ids.len();
    let emu = Emulator::new(&PROGRAM, &owners::symbol(ARCH)).unwrap();
    let (kp, ip, fg, fd, lg, ld, so) = (
        0x1_0000u64,
        0x10_0000u64,
        0x20_0000u64,
        0x30_0000u64,
        0x40_0000u64,
        0x50_0000u64,
        0x60_0000u64,
    );
    let words = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let mut kern = Vec::new();
    for p in [ip, fg, fd, lg, ld, so] {
        kern.extend(p.to_le_bytes());
    }
    kern.extend(dummy_gu.to_le_bytes());
    kern.extend(dummy_dn.to_le_bytes());
    for x in [slots as i32, ranks as i32, self_rank as i32] {
        kern.extend(x.to_le_bytes());
    }
    assert_eq!(kern.len(), 76);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    // The ids buffer holds exactly `slots` words (a zero-slot launch maps one unread word).
    let mut id_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    if slots == 0 {
        id_bytes = vec![0; 4];
    }
    mem.map("ids".into(), ip, id_bytes).unwrap();
    mem.map("full_gu".into(), fg, words(full_gu)).unwrap();
    mem.map("full_dn".into(), fd, words(full_dn)).unwrap();
    mem.map("live_gu".into(), lg, vec![0xAA; 8 * EXPERTS])
        .unwrap();
    mem.map("live_dn".into(), ld, vec![0xAA; 8 * EXPERTS])
        .unwrap();
    mem.map("slot_owner".into(), so, vec![0x5A; 4 * slots.max(1) + TAIL])
        .unwrap();
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: kp,
            workgroup: [0, 0, 0],
            block: [EXPERTS as u32, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .unwrap_or_else(|e| panic!("slots {slots} ranks {ranks} self {self_rank}: {e}"));
    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let u64s = |b: Vec<u8>| -> Vec<u64> {
        b.chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let so_bytes = region("slot_owner");
    assert!(
        so_bytes[4 * slots.max(1)..].iter().all(|&b| b == 0x5A),
        "slot_owner written past slots {slots}"
    );
    if slots == 0 {
        assert!(
            so_bytes.iter().all(|&b| b == 0x5A),
            "slot_owner written with no slots"
        );
    }
    let slot_owner = so_bytes[..4 * slots]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (u64s(region("live_gu")), u64s(region("live_dn")), slot_owner)
}

/// Distinct, recognizable table entries (never equal to a dummy or to each other).
fn tables() -> (Vec<u64>, Vec<u64>, u64, u64) {
    let gu = (0..EXPERTS as u64)
        .map(|e| 0x7000_0000_0000_0000 | (e << 24) | 0x11)
        .collect();
    let dn = (0..EXPERTS as u64)
        .map(|e| 0x6000_0000_0000_0000 | (e << 20) | 0x22)
        .collect();
    (gu, dn, 0x5555_AAAA_1234_5678, 0x0DDB_A11C_0FFE_E000)
}

/// The kernel's bytes must equal the reference for `(ids, ranks, self_rank)`.
fn check(name: &str, ids: &[i32], ranks: usize, self_rank: usize) {
    let (gu, dn, dgu, ddn) = tables();
    let (live_gu, live_dn, slot_owner) = run(ids, ranks, self_rank, &gu, &dn, dgu, ddn);
    let (owner, slots) = qwen4_ep_owner_map(ids, EXPERTS, ranks);
    for e in 0..EXPERTS {
        let take = owner[e] == Some(self_rank);
        assert_eq!(
            live_gu[e],
            if take { gu[e] } else { dgu },
            "{name} ranks {ranks} self {self_rank}: live_gate_up[{e}] ids {ids:?}"
        );
        assert_eq!(
            live_dn[e],
            if take { dn[e] } else { ddn },
            "{name} ranks {ranks} self {self_rank}: live_down[{e}] ids {ids:?}"
        );
    }
    assert_eq!(
        slot_owner, slots,
        "{name} ranks {ranks} self {self_rank}: slot owners, ids {ids:?}"
    );
}

fn check_all_ranks(name: &str, ids: &[i32], cases: &mut usize) {
    for ranks in [2usize, 3, 4] {
        for self_rank in 0..ranks {
            check(name, ids, ranks, self_rank);
            *cases += 1;
        }
    }
}

const ODD_IDS: [i32; 13] = [
    -1,
    -2,
    -513,
    i32::MIN,
    i32::MAX,
    512,
    513,
    1000,
    0x7fff_fe00,
    1 << 20,
    65_535,
    511,
    0,
];

/// Random launches: ranks 2, 3 and 4, every self rank, slots 10 / 40 / 80 (decode, verify and
/// few-row shapes) and awkward counts, uniform ids with repeats, narrow ranges, negative and
/// out-of-range ids mixed in.
#[test]
fn emulated_random_launches_equal_the_host_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut cases = 0usize;
    for &slots in &[10usize, 40, 80, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 79] {
        for trial in 0..6 {
            let ids: Vec<i32> = (0..slots)
                .map(|_| match (trial, rng.below(12)) {
                    (_, 0) => -1,
                    (_, 1) => ODD_IDS[rng.below(ODD_IDS.len())],
                    (0..=2, _) => rng.below(EXPERTS) as i32,
                    (3, _) => rng.below(24) as i32,
                    (4, _) => 480 + rng.below(32) as i32,
                    _ => (rng.below(16) * 32 + rng.below(2) * 31) as i32,
                })
                .collect();
            check_all_ranks(&format!("slots {slots} trial {trial}"), &ids, &mut cases);
        }
    }
    println!("emulated_random_launches_equal_the_host_reference: {cases} launches");
    assert_eq!(cases, 15 * 6 * 9);
}

/// Every id repeated and out of range, boundary experts, all experts routed (idx runs to 511), a
/// single expert, no routed expert, no slot at all, and slot counts past the decode shapes.
#[test]
fn emulated_edge_launches_equal_the_host_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut cases = 0usize;
    check_all_ranks("no slots", &[], &mut cases);
    check_all_ranks("one slot", &[300], &mut cases);
    check_all_ranks("one negative slot", &[-4], &mut cases);
    check_all_ranks(
        "all out of range",
        &[-1, 512, 513, i32::MIN, i32::MAX, -512, 1 << 30],
        &mut cases,
    );
    check_all_ranks("all the same", &[77; 40], &mut cases);
    check_all_ranks("expert 0 only", &[0; 80], &mut cases);
    check_all_ranks("expert 511 only", &[511; 33], &mut cases);
    check_all_ranks(
        "wave boundaries",
        &[31, 32, 63, 64, 95, 96, 255, 256, 479, 480, 511, 0],
        &mut cases,
    );
    check_all_ranks(
        "descending",
        &(0..80).map(|i| 511 - 5 * i).collect::<Vec<i32>>(),
        &mut cases,
    );
    // All 512 experts routed: idx(e) = e, owner(e) = e % ranks.
    let mut perm: Vec<i32> = (0..EXPERTS as i32).collect();
    for i in (1..perm.len()).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    check_all_ranks("all 512 routed", &perm, &mut cases);
    // 200 and 512 slots: several trips, partial last trip, duplicates across trips.
    for slots in [100usize, 129, 200, 256, 511, 512] {
        let ids: Vec<i32> = (0..slots)
            .map(|_| {
                if rng.below(9) == 0 {
                    -3
                } else {
                    rng.below(EXPERTS) as i32
                }
            })
            .collect();
        check_all_ranks(&format!("{slots} slots"), &ids, &mut cases);
    }
    println!("emulated_edge_launches_equal_the_host_reference: {cases} launches");
}

/// Random shapes: slots 1..=512, ranks 2..=4, any self rank, ids uniform / narrow / clustered on
/// wave boundaries with negative and out-of-range ids mixed in.
#[test]
fn emulated_random_shapes_equal_the_host_reference() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let cases = 300usize;
    for case in 0..cases {
        let slots = 1 + rng.below(if case % 3 == 0 { 512 } else { 96 });
        let ranks = 2 + rng.below(3);
        let self_rank = rng.below(ranks);
        let span = [EXPERTS, 40, 96, EXPERTS][rng.below(4)];
        let base = rng.below(EXPERTS - span.min(EXPERTS - 1));
        let ids: Vec<i32> = (0..slots)
            .map(|_| match rng.below(16) {
                0 => -1 - rng.below(1000) as i32,
                1 => ODD_IDS[rng.below(ODD_IDS.len())],
                2 => 512 + rng.below(100) as i32,
                3 => (rng.below(16) * 32 + [0, 31][rng.below(2)]) as i32,
                _ => (base + rng.below(span)) as i32,
            })
            .collect();
        check(
            &format!("shape #{case} slots {slots}"),
            &ids,
            ranks,
            self_rank,
        );
    }
    println!("emulated_random_shapes_equal_the_host_reference: {cases} launches");
}

/// Every instruction of the module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    let mut missing = std::collections::BTreeSet::new();
    for k in &PROGRAM.kernels {
        for id in &k.body.layout {
            let i = k.body.insts.get(*id).unwrap();
            let row = peacemaker_ir::isa::lookup(PROGRAM.target.arch, i.op, i.form).unwrap();
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                missing.insert(row.name);
            }
        }
    }
    assert!(missing.is_empty(), "no interpreter semantics: {missing:?}");
}

mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The symbol certifies: parse-back, independent wait replay, the committed shape contract,
    /// the static LDS bound, and M7's lift identity plus wait/hazard/barrier/window analyses of the
    /// linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = owners::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-ep-owner-assign-cert-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let symbol = owners::symbol(ARCH);
        let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol)
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{symbol}: {m7}"
        );
        let path = format!(
            "{}/../peacemaker-isa/kernels/qwen4_ep_owner_assign_pm.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name()
        );
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json, owners::contract(ARCH, &emitted[0]), "{path} is stale");
        let contract = serde_json::from_slice(&raw).unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join("manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{symbol}: {e}"));
    }
}
