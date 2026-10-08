// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The native code-object writer against the ROCm oracle it replaces: every
//! PM-emitted kernel symbol on gfx1100, gfx1151 and gfx1201, alone and in its
//! product module, must link to the code object `llvm-mc` + `ld.lld -shared`
//! produce and bundle to the `clang-offload-bundler -bundle-align=4096` output,
//! equal field by field and byte by byte in everything except the one place
//! the writer differs on purpose: the `.comment` identification string (the
//! oracle's is the ROCm linker's, the native one is the peacemaker native-emit
//! stamp). The comparison and its exact normalization rules live in
//! `support/native_compare.rs` (`compare_code_objects`, `compare_bundles`),
//! shared with `hipfire-rip`'s `qsa` test; the strict whole-file comparison is
//! kept there as `strict_whole_file_difference`.
#![cfg(feature = "toolchain")]
use hipfire_isa::kernels::{fp8_gemm, gdn_scan, iu4_gemm, iu4_v2b, iu4_v2b_a4, iu4_v2c, qsa_gather, qwen4_mq6_x4, qwen4_mq6_x4_gfx11, qwen4_moe_sym};
use hipfire_isa::kernels::gemm_uk::{Chain, Iu8, MmaKind};
use hipfire_isa::toolchain::{oracle_assemble_link_bundle, Toolchain};
use hipfire_isa::{native, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V, reg::Live};
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, MmaIu4, Workgroup};
use std::collections::BTreeMap;

#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];
/// The host entry spelling of the committed `.hxaco` bundles.
const HOST: &str = "host-x86_64-unknown-linux-gnu-";

/// One assembler unit: a single symbol or a product module.
struct Unit { name: String, arch: Arch, text: String, symbols: Vec<String> }

fn single(units: &mut Vec<Unit>, e: Emitted) {
    let symbol = e.s_text.lines().find_map(|l| l.trim().strip_prefix(".amdhsa_kernel ")).unwrap().trim().to_owned();
    units.push(Unit { name: symbol.clone(), arch: e.proof.arch, text: e.s_text, symbols: vec![symbol] });
}
fn module(units: &mut Vec<Unit>, name: String, arch: Arch, text: String) {
    let symbols = text.lines().filter_map(|l| l.trim().strip_prefix(".amdhsa_kernel ")).map(|s| s.trim().to_owned()).collect();
    units.push(Unit { name: format!("{name} (module)"), arch, text, symbols });
}

/// The `gemm_uk` IU8 chain probe kernel.
fn iu8_chain<T: MmaIu4, const W: u8>(arch: Arch) -> Emitted where Iu8: MmaKind<T, Input = V<W>> {
    let mut regs = RegPlan::new(32, 8).unwrap();
    let a = regs.v::<W>("a", 0, Live::Whole).unwrap();
    let x = regs.v::<W>("x", 4, Live::Whole).unwrap();
    let c0 = regs.v::<8>("c0", 8, Live::Whole).unwrap();
    let c1 = regs.v::<8>("c1", 16, Live::Whole).unwrap();
    let seed = regs.v::<8>("seed", 24, Live::Whole).unwrap();
    let mut b = Builder::new(KernelSpec { kernel_id: "iu8_chain".into(), variant: "probe".into(),
        arch, symbol: "iu8_chain".into(), kernargs: KernargLayout::new(8), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false }, regs);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    let chain = Chain::<2, Iu8>::new([c0, c1], seed).unwrap();
    chain.step(&mut wg, a, [x, x], true).unwrap();
    chain.step(&mut wg, a, [x, x], false).unwrap();
    let end = wg.exit(".Lend").unwrap();
    wg.end(end).unwrap();
    b.finish().unwrap()
}

/// Every symbol the builder emits on the three targets (every axis
/// combination its families accept), every product module, and the
/// `peacemaker profile` DIAGNOSTIC builds of the committed point sets.
fn corpus() -> Vec<Unit> {
    let mut units = Vec::new();
    for arch in ARCHES {
        // IU4 GEMM families, including the bf16 SiLU and the fused GDN entries.
        for fold in [iu4_gemm::Fold::K128, iu4_gemm::Fold::K256Shared, iu4_gemm::Fold::K256Pow2] {
            for tile in [iu4_gemm::Tile::T128x128x8, iu4_gemm::Tile::T256x128x16] {
                for cacc in [iu4_gemm::Cacc::One, iu4_gemm::Cacc::Two] {
                    for act in [iu4_gemm::ALayout::Token, iu4_gemm::ALayout::Slab] {
                        for epi in [iu4_gemm::Epi::Set, iu4_gemm::Epi::Add, iu4_gemm::Epi::GateUpSilu, iu4_gemm::Epi::GateUpSiluBf16, iu4_gemm::Epi::QkvzaGdn] {
                            if let Ok(e) = iu4_gemm::emit(iu4_gemm::Spec { fold, tile, cacc, epi, act, arch }) { single(&mut units, e) }
                        }
                        if let Ok((_, text, proof)) = iu4_gemm::emit_module(fold, tile, cacc, act, arch) { module(&mut units, proof.module, arch, text) }
                    }
                }
            }
        }
        for epi in iu4_v2c::Epi::ALL { if let Ok(e) = iu4_v2c::emit(iu4_v2c::Spec { arch, epi }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = iu4_v2c::module(arch, &iu4_v2c::Epi::ALL) { module(&mut units, proof.module, arch, text) }
        for epi in iu4_v2b::Epi::ALL { if let Ok(e) = iu4_v2b::emit(iu4_v2b::Spec { arch, epi }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = iu4_v2b::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        for epi in iu4_v2b_a4::Epi::ALL { if let Ok(e) = iu4_v2b_a4::emit(iu4_v2b_a4::Spec { arch, epi }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = iu4_v2b_a4::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        // Qwen4 MoE: the anchors, NT4/NT8 expert runs and the NT4 row repeats.
        for spec in qwen4_moe_sym::module_specs(arch) { if let Ok(e) = qwen4_moe_sym::emit(spec) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = qwen4_moe_sym::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        for kind in qsa_gather::Kind::ALL { if let Ok(e) = qsa_gather::emit(qsa_gather::Spec { arch, kind }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = qsa_gather::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        // Qwen4 MQ6 X4 trunk GEMM (gfx1201 only): the W4/W8 entries and their module.
        for wr in qwen4_mq6_x4::Spec::ALL { if let Ok(e) = qwen4_mq6_x4::emit(qwen4_mq6_x4::Spec { arch, wr }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = qwen4_mq6_x4::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        // Qwen4 MQ6 trunk GEMM U3 twin (gfx1151 only): the six entries and their module.
        for (wr, kind) in qwen4_mq6_x4_gfx11::Spec::ALL { if let Ok(e) = qwen4_mq6_x4_gfx11::emit(qwen4_mq6_x4_gfx11::Spec { arch, wr, kind }) { single(&mut units, e) } }
        if let Ok((_, text, proof)) = qwen4_mq6_x4_gfx11::emit_module(arch) { module(&mut units, proof.module, arch, text) }
        // F2 FP8 GEMM: every symbol, the per-scale modules and the
        // `--scale both --epi all` product module.
        let epis = [fp8_gemm::Epi::Set, fp8_gemm::Epi::Add, fp8_gemm::Epi::GateUpSilu, fp8_gemm::Epi::Qkv, fp8_gemm::Epi::Qkvza];
        let mut both = Vec::new();
        for act_scale in [fp8_gemm::ActScale::Row, fp8_gemm::ActScale::K128] {
            for epi in [fp8_gemm::Epi::Set, fp8_gemm::Epi::Add, fp8_gemm::Epi::GateUpSilu, fp8_gemm::Epi::GateUpSiluBf16, fp8_gemm::Epi::Qkv, fp8_gemm::Epi::Qkvza, fp8_gemm::Epi::QkvzaGdn] {
                if let Ok(e) = fp8_gemm::emit(fp8_gemm::Spec { arch, act_scale, epi }) {
                    if epis.contains(&epi) { both.push(e.clone()) }
                    single(&mut units, e);
                }
            }
            if let Ok((_, text, proof)) = fp8_gemm::emit_module(arch, act_scale, &epis) { module(&mut units, format!("{}-{act_scale:?}", proof.module), arch, text) }
        }
        if !both.is_empty() {
            let (text, proof) = fp8_gemm::module(&both).unwrap();
            module(&mut units, proof.module, arch, text);
        }
        if let Ok(e) = gdn_scan::emit(arch) { single(&mut units, e) }
    }
    single(&mut units, iu8_chain::<Gfx1100, 4>(Arch::Gfx1100));
    single(&mut units, iu8_chain::<Gfx1151, 4>(Arch::Gfx1151));
    single(&mut units, iu8_chain::<Gfx1201, 2>(Arch::Gfx1201));
    // The CLI-only `fold_magic` probe (gfx1201).
    let probe = std::env::temp_dir().join(format!("hipfire-native-fold-magic-{}", std::process::id()));
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_hipfire-isa"))
        .args(["emit", "--kernel", "fold_magic", "--arch", "gfx1201", "--out"]).arg(probe.with_extension("s"))
        .arg("--proof").arg(probe.with_extension("json")).status().unwrap();
    assert!(status.success());
    let text = std::fs::read_to_string(probe.with_extension("s")).unwrap();
    for ext in ["s", "json"] { std::fs::remove_file(probe.with_extension(ext)).unwrap() }
    units.push(Unit { name: "fold_magic".into(), arch: Arch::Gfx1201, text, symbols: vec!["fold_magic".into()] });
    let points = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/profile");
    let mut profiled = Vec::new();
    for entry in std::fs::read_dir(points).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(".points.json") { continue }
        let cfg: hipfire_isa::profile::Config = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let Some(u) = units.iter().find(|u| u.symbols == [cfg.symbol.clone()]) else { continue };
        // A point set whose anchors the kernel no longer has instruments nothing.
        let Ok((text, map)) = hipfire_isa::profile::instrument(&u.text, &cfg, u.arch) else { continue };
        profiled.push(Unit { name: map.profiled_symbol.clone(), arch: u.arch, text, symbols: vec![map.profiled_symbol] });
    }
    // Every target gets a profile build: where no committed point set still
    // instruments, one point at the first label of a single-symbol kernel
    // that accepts it.
    for arch in ARCHES {
        if profiled.iter().any(|u| u.arch == arch) { continue }
        'kernels: for u in units.iter().filter(|u| u.arch == arch && u.symbols.len() == 1) {
            for label in u.text.lines().filter_map(|l| l.trim().strip_suffix(':')).filter(|l| l.starts_with(".L") && !l.ends_with("_end")) {
                let cfg: hipfire_isa::profile::Config = serde_json::from_value(serde_json::json!({
                    "symbol": u.symbols[0], "rules": [{ "name": "point", "label": label }] })).unwrap();
                if let Ok((text, map)) = hipfire_isa::profile::instrument(&u.text, &cfg, arch) {
                    profiled.push(Unit { name: map.profiled_symbol.clone(), arch, text, symbols: vec![map.profiled_symbol] });
                    break 'kernels;
                }
            }
        }
    }
    profiled.sort_by(|a, b| a.name.cmp(&b.name));
    units.extend(profiled);
    units
}

#[test]
fn native_code_objects_and_bundles_equal_the_rocm_oracle_for_every_symbol() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let _ = require_rocm_tool!("clang-offload-bundler");
    let _ = require_rocm_tool!("llvm-objdump");
    let _ = require_rocm_tool!("llvm-readobj");
    let toolchain = Toolchain { host_target: HOST.into(), ..Toolchain::default() };
    let dir = std::env::temp_dir().join(format!("hipfire-native-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Per arch: single-symbol objects and modules (built, identical), and each
    // distinct symbol's verdict (identical only if every unit holding it is).
    let mut objects: BTreeMap<&str, [usize; 4]> = BTreeMap::new();
    let mut symbols: BTreeMap<&str, BTreeMap<String, bool>> = BTreeMap::new();
    let mut failures = Vec::new();
    let units = corpus();
    for (i, unit) in units.iter().enumerate() {
        let arch = unit.arch.name();
        let s = dir.join(format!("u{i}.s"));
        std::fs::write(&s, &unit.text).unwrap();
        let oracle = oracle_assemble_link_bundle(&toolchain, &s, &dir.join(format!("u{i}.hxaco")), arch).unwrap();
        let (oracle_elf, oracle_bundle) = (std::fs::read(&oracle.elf).unwrap(), std::fs::read(&oracle.hsaco).unwrap());
        let verdict = native::assemble(&unit.text, unit.arch).map(|elf| {
            let bundle = native::bundle(&elf, unit.arch, HOST);
            native_compare::compare_native_to_oracle(&elf, &bundle, &oracle_elf, &oracle_bundle)
        });
        let identical = matches!(verdict, Ok(None));
        match verdict {
            Ok(None) => {}
            Ok(Some(why)) => failures.push(format!("{arch} {}: {why}", unit.name)),
            Err(e) => failures.push(format!("{arch} {}: native: {e}", unit.name)),
        }
        let row = objects.entry(arch).or_default();
        let column = if unit.symbols.len() == 1 && !unit.name.ends_with("(module)") { 0 } else { 2 };
        row[column] += 1;
        row[column + 1] += usize::from(identical);
        for symbol in &unit.symbols { *symbols.entry(arch).or_default().entry(symbol.clone()).or_insert(true) &= identical }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    println!("| arch | symbols | byte-identical symbols | single-symbol objects (identical) | modules (identical) |\n|---|---:|---:|---:|---:|");
    for (arch, [singles, singles_ok, modules, modules_ok]) in &objects {
        let s = &symbols[arch];
        println!("| {arch} | {} | {} | {singles} ({singles_ok}) | {modules} ({modules_ok}) |", s.len(), s.values().filter(|&&ok| ok).count());
    }
    println!("symbols:");
    for (arch, s) in &symbols { for (symbol, ok) in s { println!("  {arch} {symbol} {}", if *ok { "identical" } else { "DIFFERS" }) } }
    assert!(failures.is_empty(), "{} of {} units differ:\n{}", failures.len(), units.len(), failures.join("\n"));
}
