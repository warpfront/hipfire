// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emission of the MQ4-only landing object `kernels/mq_i4_moe_gemv_mq4only_pm_gfx1100.hxaco`:
//! the module `mq_i4_moe_gemv_pm_gfx1100` restricted to the two existing `Mq4Sym` entries
//! `mq_i4_moe_gemv_mq4sym_{r256_gate_up,r128_down}_pm_gfx1100` of the qualified generator
//! `kernels/mq_i4_moe_gemv_gfx1100.rs` (included below with `#[path]`; it is deliberately not
//! in the kernel registry). The `Mq2I4` (W2) entries are not emitted and do not land.
//!
//! ```text
//! cargo run -p peacemaker-isa --features toolchain --example mq_i4_moe_gemv_mq4only_gfx1100_emit -- \
//!     <output-dir> [--host-target TRIPLE]
//! ```
//!
//! Each entry is the generator's own `emit(spec)` and the module is `iu4_gemm::module` over just those two.
//! The generator carries the 96-byte kernarg (`n_experts@88`, zero `reserved@92`) and the unsigned `expert_id >=
//! n_experts` exit before `expert_id * 8`; the committed four-symbol `kernels/mq_i4_moe_gemv_pm_gfx1100.hxaco` is
//! the older 88-byte W2 image and is neither regenerated nor landed. Files written to `<output-dir>`, stem
//! `mq_i4_moe_gemv_mq4only_pm_gfx1100`: `.s`, `.proof.json`, `.co` (`native::assemble`), `.hxaco`
//! (`native::bundle`, host entry `host-x86_64-unknown-linux-gnu` unless `--host-target` is given), `.m7.json`
//! (`--features toolchain`), and `mq_i4_moe_gemv.gfx1100.mq4sym_{r256_gate_up,r128_down}.contract.json`.
//! Prints the SHA-256 of the object and of the bundle. Nothing here runs a GPU.

#![allow(dead_code)]

use std::{fs, path::PathBuf};

use sha2::{Digest, Sha256};

// Facade for the emitter's `crate::` / `super::` dependencies: the file is written against the library's
// module tree, so the example re-exports exactly those names at its own crate root.
use peacemaker_isa::kernels::iu4_gemm;
use peacemaker_isa::{
    arch, insn, native, plan, reg, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
mod kernels {
    pub use peacemaker_isa::kernels::common;
}
#[path = "../src/kernels/mq_i4_moe_gemv_gfx1100.rs"]
mod mq_i4_moe_gemv_gfx1100;

use mq_i4_moe_gemv_gfx1100::{contract, emit, module_specs, Codec, Spec};

const USAGE: &str =
    "usage: mq_i4_moe_gemv_mq4only_gfx1100_emit <output-dir> [--host-target TRIPLE]";
const STEM: &str = "mq_i4_moe_gemv_mq4only_pm_gfx1100";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut dir, mut host) = (None::<PathBuf>, native::DEFAULT_HOST_TARGET.to_owned());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(v) = arg.strip_prefix("--host-target=") {
            host = v.to_owned()
        } else if arg == "--host-target" {
            host = args.next().ok_or(USAGE)?
        } else if arg.starts_with("--") || dir.is_some() {
            return Err(USAGE.into());
        } else {
            dir = Some(PathBuf::from(arg))
        }
    }
    let dir = dir.ok_or(USAGE)?;
    fs::create_dir_all(&dir)?;
    let target = Arch::Gfx1100;

    let specs: Vec<Spec> = module_specs(target)
        .into_iter()
        .filter(|s| s.codec == Codec::Mq4Sym)
        .collect();
    if specs.len() != 2 {
        return Err(format!("expected the two Mq4Sym entries, found {}", specs.len()).into());
    }
    let emitted: Vec<Emitted> = specs.iter().map(|s| emit(*s)).collect::<Result<_, _>>()?;
    let (text, proof): (String, iu4_gemm::ModuleProof) =
        iu4_gemm::module(&emitted, &Spec::module(target))?;

    let source = dir.join(format!("{STEM}.s"));
    fs::write(&source, &text)?;
    fs::write(
        dir.join(format!("{STEM}.proof.json")),
        serde_json::to_vec_pretty(&proof)?,
    )?;
    let elf = native::assemble(&text, target)?;
    let object = dir.join(format!("{STEM}.co"));
    fs::write(&object, &elf)?;
    let bundle = native::bundle(&elf, target, &host);
    let bundle_path = dir.join(format!("{STEM}.hxaco"));
    fs::write(&bundle_path, &bundle)?;

    let mut summary = Vec::new();
    for (spec, e) in specs.iter().zip(&emitted) {
        let file = dir.join(format!(
            "mq_i4_moe_gemv.{}.{}.contract.json",
            target.name(),
            spec.tag()
        ));
        fs::write(&file, serde_json::to_vec_pretty(&contract(*spec, e))?)?;
        summary.push(serde_json::json!({
            "symbol": spec.symbol(), "contract": file, "vgpr": e.shape.next_free_vgpr, "sgpr": e.shape.next_free_sgpr,
            "waits": e.shape.waits, "hazard_guards": e.proof.hazards.len(), "s_text_sha256": e.proof.s_text_sha256,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "module": Spec::module(target), "source": source, "object": object, "bundle": bundle_path,
            "s_text_sha256": proof.s_text_sha256, "object_sha256": sha(&elf), "bundle_sha256": sha(&bundle),
            "object_bytes": elf.len(), "bundle_bytes": bundle.len(), "entries": summary,
        }))?
    );

    m7(&dir, &object, &specs)
}

/// M7 byte obligations of every symbol of the linked object, written to `<stem>.m7.json`.
#[cfg(feature = "toolchain")]
fn m7(
    dir: &std::path::Path,
    object: &std::path::Path,
    specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut receipts = std::collections::BTreeMap::new();
    for spec in specs {
        let symbol = spec.symbol();
        let receipt = peacemaker_isa::pm_check::m7(object, spec.arch.name(), &symbol)?;
        receipts.insert(symbol, receipt);
    }
    let receipts = serde_json::to_vec_pretty(&receipts)?;
    fs::write(dir.join(format!("{STEM}.m7.json")), &receipts)?;
    println!("{}", String::from_utf8_lossy(&receipts));
    Ok(())
}

#[cfg(not(feature = "toolchain"))]
fn m7(
    _dir: &std::path::Path,
    _object: &std::path::Path,
    _specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    Err(format!("M7 certification requires --features toolchain (pm_check is feature-gated); no {STEM}.m7.json was written").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<&str> {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("s_delay_alu"))
            .collect()
    }
    fn mnemonic(line: &str) -> &str {
        line.split_whitespace().next().unwrap_or("")
    }
    fn field(text: &str, key: &str) -> Vec<u32> {
        text.lines()
            .filter_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().parse().unwrap())
            .collect()
    }
    /// Whether an instruction line names scalar register `n` (a single `sN` operand or inside an `s[a:b]` range).
    fn touches_sgpr(line: &str, n: u32) -> bool {
        let ops = line.split_once(' ').map_or("", |(_, rest)| rest);
        ops.split(", ").any(|o| {
            let o = o.trim();
            match o
                .strip_prefix("s[")
                .and_then(|r| r.strip_suffix(']'))
                .and_then(|r| r.split_once(':'))
            {
                Some((a, b)) => {
                    (a.parse::<u32>().unwrap()..=b.parse::<u32>().unwrap()).contains(&n)
                }
                None => o == format!("s{n}"),
            }
        })
    }

    /// The landing set is exactly the two `Mq4Sym` entries (W2 is never emitted by this example).
    #[test]
    fn landing_set_is_the_two_mq4sym_entries() {
        let specs: Vec<Spec> = module_specs(Arch::Gfx1100)
            .into_iter()
            .filter(|s| s.codec == Codec::Mq4Sym)
            .collect();
        assert_eq!(
            specs.iter().map(Spec::symbol).collect::<Vec<_>>(),
            [
                "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
                "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100"
            ]
        );
    }

    /// The 96-byte kernarg: the 88-byte R5 block, `n_experts@88` and the zero `reserved@92`, in the metadata, the
    /// descriptor and the scalar loads (one `s_load_b32` of `n_experts` into the spare `s4`; the first kernarg block is
    /// `s[80:87]`, past the workgroup ids the gfx1100 `FeatureUserSGPRInit16Bug` padding moves to `s14` / `s15`).
    #[test]
    fn kernarg_is_96_bytes_with_n_experts_and_reserved() {
        assert_eq!(mq_i4_moe_gemv_gfx1100::KERNARG_BYTES, 96);
        for spec in module_specs(Arch::Gfx1100) {
            let e = emit(spec).unwrap();
            let symbol = spec.symbol();
            assert!(e.s_text.contains(".kernarg_segment_size: 96\n"), "{symbol}");
            assert!(e.s_text.contains(".amdhsa_kernarg_size 96\n"), "{symbol}");
            let offsets = field(&e.s_text, ".offset:");
            let mut want: Vec<u32> = (0..7).map(|i| 8 * i).collect();
            want.extend((0..10).map(|i| 56 + 4 * i));
            assert_eq!(offsets, want, "{symbol}: kernarg offsets");
            assert_eq!(
                field(&e.s_text, ".size:").len(),
                17,
                "{symbol}: kernarg sizes"
            );
            let names: Vec<&str> = e
                .s_text
                .lines()
                .filter_map(|l| l.trim().strip_prefix(".name: "))
                .take(17)
                .collect();
            assert_eq!(&names[15..], ["n_experts", "reserved"], "{symbol}");
            for load in [
                "s_load_b256 s[80:87], s[0:1], null",
                "s_load_b256 s[16:23], s[0:1], 0x20",
                "s_load_b128 s[24:27], s[0:1], 0x40",
                "s_load_b64 s[28:29], s[0:1], 0x50",
                "s_load_b32 s4, s[0:1], 0x58",
            ] {
                assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
            }
        }
    }

    /// The `n_experts` guard: the word lives in the spare `s4` (touched only by its load and the compare, so every other
    /// register assignment is unchanged; the SGPR ceiling is 88: the kernarg block is relocated to `s[80:87]`); `s_cmp_ge_u32 s34, s4` plus the existing exit
    /// branch follow the offsets guard and precede `expert_id * 8` and the expert-table `s_load_b64`.
    #[test]
    fn n_experts_guard_precedes_the_table_index() {
        for spec in module_specs(Arch::Gfx1100) {
            let e = emit(spec).unwrap();
            let symbol = spec.symbol();
            let l = lines(&e.s_text);
            let exit = format!(".Lmqi4moe_{}_exit", spec.tag());
            assert_eq!(e.shape.next_free_sgpr, 88, "{symbol}: sgpr ceiling");
            let load = l
                .iter()
                .position(|t| *t == "s_load_b32 s4, s[0:1], 0x58")
                .unwrap_or_else(|| panic!("{symbol}: n_experts load"));
            let cmp = l
                .iter()
                .position(|t| *t == "s_cmp_ge_u32 s34, s4")
                .unwrap_or_else(|| panic!("{symbol}: n_experts compare"));
            let touching: Vec<usize> = l
                .iter()
                .enumerate()
                .filter(|(_, t)| touches_sgpr(t, 4))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                touching,
                [load, cmp],
                "{symbol}: s4 is touched only by the load and the compare"
            );
            assert_eq!(
                l[cmp + 1],
                format!("s_cbranch_scc1 {exit}"),
                "{symbol}: compare feeds the existing exit"
            );
            let offsets_guard = l
                .iter()
                .position(|t| t.starts_with("s_cmp_gt_u32 s33, s51"))
                .unwrap();
            assert_eq!(
                l[offsets_guard + 1],
                format!("s_cbranch_scc1 {exit}"),
                "{symbol}: offsets guard"
            );
            assert!(
                cmp > offsets_guard + 1
                    && l[offsets_guard + 2..cmp]
                        .iter()
                        .all(|t| t.starts_with("s_waitcnt")),
                "{symbol}: n_experts guard follows the offsets guard with at most a wait between"
            );
            let scale_eid = l
                .iter()
                .position(|t| *t == "s_lshl_b32 s51, s34, 3")
                .unwrap();
            let table_load = l
                .iter()
                .position(|t| t.starts_with("s_load_b64 s[38:39], s[36:37]"))
                .unwrap();
            assert!(
                cmp + 1 < scale_eid && scale_eid < table_load,
                "{symbol}: guard before expert_id*8 and the expert-table load"
            );
            let guards = l
                .iter()
                .filter(|t| mnemonic(t) == "s_cbranch_scc1" && t.ends_with(&exit))
                .count();
            assert_eq!(
                guards, 6,
                "{symbol}: row, expert index, empty, >8 slots, offsets>10C, expert id >= n_experts"
            );
            let raw: Vec<&str> = e.s_text.lines().map(str::trim).collect();
            let (lo, hi) = (
                raw.iter()
                    .position(|t| *t == "s_load_b32 s4, s[0:1], 0x58")
                    .unwrap(),
                raw.iter()
                    .position(|t| *t == "s_cmp_ge_u32 s34, s4")
                    .unwrap(),
            );
            assert!(
                raw[lo..hi].iter().any(|t| t.starts_with("s_waitcnt")),
                "{symbol}: no load-completion wait before the compare"
            );
        }
    }
}
