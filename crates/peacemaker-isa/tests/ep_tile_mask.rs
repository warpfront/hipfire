// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next owner-routed chunk MoE tile mask builder module (gfx1100 and gfx1201):
//! `ep_tile_mask_pm_<arch>`. The emitted program runs on the PeaceMaker CPU
//! interpreter (256-thread workgroups, one per 256 tiles) and its output
//! (`tile_ids_owned`) must equal the host reference for every input on both
//! architectures and for 2..=8 ranks; the source tile ids are never written. No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::ep_tile_mask as mask;
use peacemaker_isa::Arch;
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 2] = [Arch::Gfx1100, Arch::Gfx1201];
const BLOCK: usize = mask::BLOCK as usize;
/// Rank counts every interpreter sweep covers (the product supports 2..=8).
const RANK_COUNTS: [usize; 5] = [2, 3, 4, 6, 8];

/// The host reference of the kernel's law.
fn reference(tile_ids: &[i32], owner_table: &[i32], rank: i32) -> Vec<i32> {
    tile_ids
        .iter()
        .map(|&id| {
            let live = id >= 0 && (id as usize) < owner_table.len();
            if live && owner_table[id as usize] == rank {
                id
            } else {
                -1
            }
        })
        .collect()
}

#[test]
fn emission_is_deterministic_and_targets_gfx1100_and_gfx1201_only() {
    for arch in ARCHES {
        let (a, b) = (mask::emit(arch).unwrap(), mask::emit(arch).unwrap());
        assert_eq!(a.s_text, b.s_text, "{arch:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{arch:?}");
        assert!(a.s_text.contains(&format!("{}:", mask::symbol(arch))));
        assert_eq!(a.shape.barriers, 0);
        assert!(
            a.shape.next_free_vgpr <= 16 && a.shape.next_free_sgpr <= 104,
            "{arch:?} {:?}",
            a.shape
        );
        let (emitted, text, _) = mask::emit_module(arch).unwrap();
        assert_eq!(emitted.len(), 1);
        assert!(text.contains(&format!("{}:", mask::symbol(arch))));
        assert_eq!(
            mask::module(arch),
            format!("ep_tile_mask_pm_{}", arch.name())
        );
        assert_eq!(
            mask::symbol(arch),
            format!("ep_tile_mask_pm_{}", arch.name())
        );
    }
    assert_eq!(mask::symbol(Arch::Gfx1100), "ep_tile_mask_pm_gfx1100");
    assert_eq!(mask::symbol(Arch::Gfx1201), "ep_tile_mask_pm_gfx1201");
    for arch in [Arch::Gfx1151, Arch::Gfx942] {
        assert!(mask::emit(arch).is_err(), "{arch:?}");
        assert!(mask::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// The gfx1201 text is the pre-gfx1100 text: gfx12 spellings only, no gfx11 system-SGPR read.
#[test]
fn arch_specific_spellings() {
    let g12 = mask::emit(Arch::Gfx1201).unwrap().s_text;
    assert!(
        g12.contains("ttmp9") && g12.contains("s_sub_co_i32") && g12.contains(", null offen"),
        "gfx1201 spellings"
    );
    let g11 = mask::emit(Arch::Gfx1100).unwrap().s_text;
    assert!(
        !g11.contains("ttmp9") && !g11.contains("s_sub_co_i32") && !g11.contains("null offen"),
        "gfx1100 carries a gfx12 spelling"
    );
    assert!(
        g11.contains("s_sub_i32") && g11.contains(", 0 offen"),
        "gfx1100 spellings"
    );
}

/// Kernarg ABI of `Gpu::ep_tile_mask`: three pointers, three i32 (36 bytes); 256-thread workgroup,
/// no LDS, no atomics, no scratch, no flat / global / ds access.
#[test]
fn kernarg_layout_matches_the_launcher() {
    for arch in ARCHES {
        let text = mask::emit_module(arch).unwrap().1;
        let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
        let offsets: Vec<u32> = metadata
            .lines()
            .filter_map(|l| {
                l.trim()
                    .strip_prefix(".offset:")
                    .map(|v| v.trim().parse::<u32>().unwrap())
            })
            .collect();
        assert_eq!(offsets, [0, 8, 16, 24, 28, 32], "{arch:?}");
        assert!(
            text.contains(".amdhsa_kernarg_size 36\n"),
            "{arch:?} kernarg size"
        );
        assert!(
            text.contains(".amdhsa_group_segment_fixed_size 0\n"),
            "{arch:?} LDS"
        );
        assert!(
            text.contains(".max_flat_workgroup_size: 256"),
            "{arch:?} workgroup size"
        );
        assert!(!text.contains("atomic"));
        for l in text.lines().map(str::trim_start) {
            assert!(
                !(l.starts_with("scratch_")
                    || l.starts_with("flat_")
                    || l.starts_with("global_")
                    || l.starts_with("ds_")),
                "{arch:?} {l}"
            );
        }
    }
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for arch in ARCHES {
        let module = mask::module(arch);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = mask::emit_module(arch).unwrap().1;
        assert!(
            text_section(&link(&text, arch.name(), &module)) == text_section(&committed),
            "{module}: fresh emission differs from the committed bundle"
        );
    }
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-ep-tile-mask-{}",
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

fn lift(arch: Arch) -> peacemaker_ir::Program {
    let text = mask::emit_module(arch).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
    .program
}
static PROGRAM_GFX1100: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(Arch::Gfx1100));
static PROGRAM_GFX1201: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(Arch::Gfx1201));
fn program(arch: Arch) -> &'static peacemaker_ir::Program {
    match arch {
        Arch::Gfx1100 => &PROGRAM_GFX1100,
        Arch::Gfx1201 => &PROGRAM_GFX1201,
        a => panic!("{a:?}"),
    }
}

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

/// One full grid over `tile_ids`. Returns `tile_ids_owned` and asserts the output's untouched tail and
/// the input's immutability.
fn run(arch: Arch, tile_ids: &[i32], owner_table: &[i32], rank: i32) -> Vec<i32> {
    let tiles = tile_ids.len();
    let emu = Emulator::new(program(arch), &mask::symbol(arch)).unwrap();
    let (kp, ip, op_, tp) = (0x1_0000u64, 0x10_0000u64, 0x20_0000u64, 0x30_0000u64);
    let words = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let mut kern = Vec::new();
    for p in [ip, op_, tp] {
        kern.extend(p.to_le_bytes());
    }
    for x in [tiles as i32, owner_table.len() as i32, rank] {
        kern.extend(x.to_le_bytes());
    }
    assert_eq!(kern.len(), mask::KERNARG_BYTES as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map("tile_ids".into(), ip, words(tile_ids)).unwrap();
    mem.map("owned".into(), op_, vec![0x5A; 4 * tiles + TAIL])
        .unwrap();
    mem.map("table".into(), tp, words(owner_table)).unwrap();
    for wg in 0..tiles.div_ceil(BLOCK) {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: kp,
                workgroup: [wg as u32, 0, 0],
                block: [BLOCK as u32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("{arch:?} tiles {tiles} rank {rank} wg {wg}: {e}"));
    }
    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let owned = region("owned");
    assert!(
        owned[4 * tiles..].iter().all(|&b| b == 0x5A),
        "{arch:?} owned written past tiles {tiles}"
    );
    assert_eq!(
        region("tile_ids"),
        words(tile_ids),
        "{arch:?} tile ids modified"
    );
    owned[..4 * tiles]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check(arch: Arch, name: &str, tile_ids: &[i32], owner_table: &[i32], rank: i32) {
    assert_eq!(
        run(arch, tile_ids, owner_table, rank),
        reference(tile_ids, owner_table, rank),
        "{arch:?} {name}: tiles {} rank {rank} ids {tile_ids:?}",
        tile_ids.len()
    );
}

/// The static `e % ranks` split (what the product installs) plus awkward tables.
fn table(experts: usize, ranks: usize) -> Vec<i32> {
    (0..experts).map(|e| (e % ranks) as i32).collect()
}

const ODD_IDS: [i32; 9] = [
    -1,
    -2,
    i32::MIN,
    i32::MAX,
    512,
    513,
    1 << 20,
    0x4000_0000,
    511,
];

/// Tile counts around the workgroup size, ids sorted in runs like the stable grouping (a run of
/// tiles per expert, a -1 sentinel tail), every rank of 2, 3, 4, 6 and 8 (tiles of the rank itself
/// are kept, every other tile and the sentinel tail map to -1).
#[test]
fn emulated_grouped_launches_equal_the_host_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut cases = 0usize;
    for arch in ARCHES {
        for &tiles in &[1usize, 2, 3, 255, 256, 257, 511, 512, 513, 1000, 1500] {
            for ranks in RANK_COUNTS {
                let owner = table(512, ranks);
                let trials = if tiles > 600 { 2 } else { 4 };
                for trial in 0..trials {
                    // Ascending expert runs of 1..=8 tiles, then the -1 tail.
                    let live = tiles - rng.below(tiles / 3 + 1);
                    let mut ids = Vec::with_capacity(tiles);
                    let mut e = rng.below(8) as i32;
                    while ids.len() < live {
                        let run = 1 + rng.below(if trial == 0 { 1 } else { 8 });
                        for _ in 0..run.min(live - ids.len()) {
                            ids.push(e);
                        }
                        e += 1 + rng.below(3) as i32;
                        if e >= 512 {
                            e = rng.below(8) as i32;
                        }
                    }
                    ids.resize(tiles, -1);
                    for rank in 0..ranks as i32 {
                        check(
                            arch,
                            &format!("grouped tiles {tiles} ranks {ranks} trial {trial}"),
                            &ids,
                            &owner,
                            rank,
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("emulated_grouped_launches_equal_the_host_reference: {cases} launches");
}

/// Negative, out-of-range and huge ids mixed with live ones; an all-sentinel grid; one expert; every
/// id owned by one rank; a table that is not a modulo split (arbitrary owners, including an owner no
/// rank holds); a one-entry table; owners up to rank 7.
#[test]
fn emulated_edge_launches_equal_the_host_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for arch in ARCHES {
        for ranks in RANK_COUNTS {
            let owner = table(512, ranks);
            for rank in 0..ranks as i32 {
                check(arch, "all sentinel", &[-1; 300], &owner, rank);
                check(arch, "one tile", &[7], &owner, rank);
                check(arch, "one sentinel tile", &[-1], &owner, rank);
                check(arch, "expert 0 and 511", &[0, 511, 0, 511], &owner, rank);
                check(arch, "same expert", &[300; 300], &owner, rank);
                let odd: Vec<i32> = (0..400)
                    .map(|i| {
                        if i % 3 == 0 {
                            ODD_IDS[rng.below(ODD_IDS.len())]
                        } else {
                            rng.below(512) as i32
                        }
                    })
                    .collect();
                check(arch, "odd ids", &odd, &owner, rank);
            }
        }
        // Owners that are not a modulo split (-1 .. 8 inclusive: no owner, ranks 0..8, one past the last rank).
        let arbitrary: Vec<i32> = (0..512).map(|_| rng.below(10) as i32 - 1).collect();
        for rank in -1..9 {
            let ids: Vec<i32> = (0..777)
                .map(|_| {
                    if rng.below(7) == 0 {
                        -1
                    } else {
                        rng.below(512) as i32
                    }
                })
                .collect();
            check(arch, "arbitrary owners", &ids, &arbitrary, rank);
        }
        // A short table: ids at or past its length are masked.
        let short = [1, 0, 1];
        for rank in 0..2 {
            check(
                arch,
                "short table",
                &[0, 1, 2, 3, 4, -1, 1 << 30],
                &short,
                rank,
            );
        }
        // All experts of one rank (the top rank of eight, then a rank that owns nothing).
        check(
            arch,
            "one rank owns all",
            &(0..512).collect::<Vec<i32>>(),
            &vec![7; 512],
            7,
        );
        check(
            arch,
            "the other rank owns all",
            &(0..512).collect::<Vec<i32>>(),
            &vec![7; 512],
            0,
        );
    }
}

/// Random shapes: tiles 1..=3100 (the production capacity bound is 3,040 tiles), ranks 2..=8.
#[test]
fn emulated_random_shapes_equal_the_host_reference() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let cases = 40usize;
    for arch in ARCHES {
        for case in 0..cases {
            let tiles = 1 + rng.below(if case % 4 == 0 { 3100 } else { 700 });
            let ranks = 2 + rng.below(7);
            let rank = rng.below(ranks) as i32;
            let owner = table(512, ranks);
            let ids: Vec<i32> = (0..tiles)
                .map(|_| match rng.below(10) {
                    0 => -1 - rng.below(1000) as i32,
                    1 => ODD_IDS[rng.below(ODD_IDS.len())],
                    _ => rng.below(512) as i32,
                })
                .collect();
            check(arch, &format!("shape #{case}"), &ids, &owner, rank);
        }
    }
    println!("emulated_random_shapes_equal_the_host_reference: {cases} launches per arch");
}

/// Every instruction of the module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    for arch in ARCHES {
        let program = program(arch);
        let mut missing = std::collections::BTreeSet::new();
        for k in &program.kernels {
            for id in &k.body.layout {
                let i = k.body.insts.get(*id).unwrap();
                let row = peacemaker_ir::isa::lookup(program.target.arch, i.op, i.form).unwrap();
                if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                    missing.insert(row.name);
                }
            }
        }
        assert!(
            missing.is_empty(),
            "{arch:?}: no interpreter semantics: {missing:?}"
        );
    }
}

#[cfg(feature = "toolchain")]
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
        for arch in ARCHES {
            let (emitted, text, _) = mask::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-ep-tile-mask-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            let symbol = mask::symbol(arch);
            let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}: {m7}"
            );
            let path = format!(
                "{}/kernels/ep_tile_mask_pm.{}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                arch.name()
            );
            let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(json, mask::contract(arch, &emitted[0]), "{path} is stale");
            let contract = serde_json::from_slice(&raw).unwrap();
            certify(
                &toolchain,
                &build,
                &s,
                arch.name(),
                &dir.join("manifest.json"),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }
}
