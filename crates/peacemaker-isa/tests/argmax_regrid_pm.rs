// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1201 row argmax `argmax_f32_batched_regrid_pm_gfx1201`, the byte twin of
//! the hipcc `argmax_f32_batched_regrid` (`kernels/src/select_regrid.hip`). The emitted program
//! runs on the PeaceMaker CPU interpreter (1024-thread workgroups, one per row, LDS and
//! workgroup barrier included) and its index must equal the host reference of the HIP law for
//! every row; the committed image equals a fresh emission. No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::argmax_regrid_pm as am;
use peacemaker_isa::Arch;
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const BLOCK: u32 = am::BLOCK;
const TAIL: usize = 16;

/// The HIP law: `v > -1e30` qualifies; larger value wins, equal values the lower
/// `rev8(i & 255)`, then the lower index; no qualifying element -> 0.
fn reference(row: &[f32]) -> i32 {
    let key = |i: usize| ((i as u32 & 255) as u8).reverse_bits() as u64 * (1 << 32) + i as u64;
    let mut best: Option<(f32, usize)> = None;
    for (i, &v) in row.iter().enumerate() {
        if !(v > -1e30f32) {
            continue;
        }
        best = match best {
            None => Some((v, i)),
            Some((bv, bi)) => {
                if v > bv || (v == bv && key(i) < key(bi)) {
                    Some((v, i))
                } else {
                    Some((bv, bi))
                }
            }
        };
    }
    best.map_or(0, |(_, i)| i as i32)
}

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    let a = am::emit_module(ARCH).unwrap();
    let b = am::emit_module(ARCH).unwrap();
    assert_eq!(a.1, b.1);
    assert_eq!(a.2.s_text_sha256, b.2.s_text_sha256);
    assert!(a.0[0].shape.barriers > 0);
    assert!(a.0[0].shape.next_free_vgpr <= 32 && a.0[0].shape.next_free_sgpr <= 104);
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(am::emit(arch).is_err(), "{arch:?}");
    }
    assert_eq!(am::symbol(ARCH), "argmax_f32_batched_regrid_pm_gfx1201");
    assert_eq!(am::module(ARCH), "argmax_regrid_pm_gfx1201");
}

/// Kernarg ABI of `Gpu::argmax_f32_batched_regrid_pm`: data, result, i32 n (20 bytes).
#[test]
fn kernarg_layout_matches_the_launcher() {
    let text = am::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let offsets: Vec<u32> = metadata
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse::<u32>().unwrap())
        })
        .collect();
    assert_eq!(offsets, [0, 8, 16]);
    assert!(text.contains(".amdhsa_kernarg_size 20\n"));
    assert!(text.contains(".amdhsa_group_segment_fixed_size 256\n"));
    assert!(text.contains(".max_flat_workgroup_size: 1024"));
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
    let module = am::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = am::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle"
    );
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-argmax-regrid-{}",
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
    let text = am::emit_module(ARCH).unwrap().1;
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
    fn f(&mut self) -> f32 {
        (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
    }
}

/// One launch over `rows` rows of `n` floats (row-major in `data`); returns the batch results
/// and asserts the guard words around the result and the data were never touched.
fn run(data: &[f32], n: usize, rows: usize) -> Vec<i32> {
    assert_eq!(data.len(), n * rows);
    let emu = Emulator::new(&PROGRAM, &am::symbol(ARCH)).unwrap();
    let (kp, dp, rp) = (0x1_0000u64, 0x10_0000u64, 0x4000_0000u64);
    let mut kern = Vec::new();
    kern.extend(dp.to_le_bytes());
    kern.extend(rp.to_le_bytes());
    kern.extend((n as i32).to_le_bytes());
    assert_eq!(kern.len(), am::KERNARG_BYTES as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map(
        "data".into(),
        dp,
        data.iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
    )
    .unwrap();
    mem.map("result".into(), rp, vec![0x5A; 4 * rows + TAIL])
        .unwrap();
    for row in 0..rows {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: kp,
                workgroup: [row as u32, 0, 0],
                block: [BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("n {n} row {row}: {e}"));
    }
    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let res = region("result");
    assert!(
        res[4 * rows..].iter().all(|&b| b == 0x5A),
        "n {n}: result written past the batch"
    );
    res[..4 * rows]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check(name: &str, data: &[f32], n: usize, rows: usize) {
    let got = run(data, n, rows);
    for r in 0..rows {
        let want = reference(&data[r * n..(r + 1) * n]);
        assert_eq!(got[r], want, "{name}: n {n} row {r}");
    }
}

fn denormals(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let m = (rng.next() % 8) as u32;
            let sign = if rng.next() & 1 == 0 { 0 } else { 0x8000_0000 };
            f32::from_bits(sign | m)
        })
        .collect()
}

#[test]
fn emulated_edge_rows_equal_the_host_reference() {
    let mut rng = Rng(0xA11CE);
    for n in [
        1usize, 2, 3, 7, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 5000,
    ] {
        // Ties everywhere.
        check("all-equal", &vec![0.5; n * 3], n, 3);
        check("all-zero", &vec![0.0; n * 2], n, 2);
        check(
            "neg-zero-vs-zero",
            &(0..2 * n)
                .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            n,
            2,
        );
        // No qualifying element -> 0.
        check("all-nan", &vec![f32::NAN; n], n, 1);
        check("all-neg-inf", &vec![f32::NEG_INFINITY; n], n, 1);
        check("all-below", &vec![-2e30; n], n, 1);
        check("exactly -1e30", &vec![-1e30; n], n, 1);
        check(
            "+inf ties",
            &(0..n)
                .map(|i| if i % 5 == 0 { f32::INFINITY } else { 1.0 })
                .collect::<Vec<_>>(),
            n,
            1,
        );
        check("denormals", &denormals(&mut rng, n), n, 1);
        // Random with NaN / inf sprinkled at random positions.
        for _ in 0..3 {
            let mut row: Vec<f32> = (0..n).map(|_| rng.f()).collect();
            for _ in 0..(n / 4 + 1) {
                let p = rng.below(n);
                row[p] = match rng.below(5) {
                    0 => f32::NAN,
                    1 => f32::NEG_INFINITY,
                    2 => f32::INFINITY,
                    3 => -0.0,
                    _ => 0.0,
                };
            }
            check("random+special", &row, n, 1);
        }
        // Coarse values so the max repeats at many strides.
        let coarse: Vec<f32> = (0..n).map(|_| (rng.below(4)) as f32).collect();
        check("coarse", &coarse, n, 1);
    }
}

#[test]
fn emulated_vocab_rows_equal_the_host_reference() {
    let mut rng = Rng(0xBEEF);
    for n in [151_936usize, 248_320, 262_144] {
        let mut row: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        check("random", &row, n, 1);
        row.iter_mut().for_each(|x| *x = 1.0);
        check("all-equal", &row, n, 1);
        row[n - 1] = f32::NAN;
        row[n / 2] = f32::INFINITY;
        check("inf-mid", &row, n, 1);
    }
}

/// Rows 1..=64 of random shapes, the grid the launcher admits for `batch_size`.
#[test]
fn emulated_batched_launches_equal_the_host_reference() {
    let mut rng = Rng(0xC0FFEE);
    for rows in [1usize, 2, 5, 8, 16, 33, 64] {
        let n = 200 + rng.below(5000);
        let data: Vec<f32> = (0..rows * n)
            .map(|_| (rng.below(64)) as f32 / 8.0)
            .collect();
        check("batch", &data, n, rows);
    }
}

/// Every instruction of the module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    let mut missing = std::collections::BTreeSet::new();
    for k in &PROGRAM.kernels {
        for id in &k.body.layout {
            let i = k.body.insts.get(*id).unwrap();
            let row = peacemaker_ir::isa::lookup_inst(PROGRAM.target.arch, &i)
                .ok()
                .unwrap();
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                missing.insert(row.name);
            }
        }
    }
    assert!(missing.is_empty(), "no interpreter semantics: {missing:?}");
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
        for arch in [ARCH] {
            let (emitted, text, _) = am::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-argmax-regrid-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            let symbol = am::symbol(arch);
            let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}: {m7}"
            );
            let path = format!(
                "{}/kernels/argmax_regrid_pm.{}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                arch.name()
            );
            let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(json, am::contract(arch, &emitted[0]), "{path} is stale");
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
