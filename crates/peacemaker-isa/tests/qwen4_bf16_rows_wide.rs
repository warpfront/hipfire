// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Structural and row-exact CPU tests of the BF16 fixed-eight rowtile twins,
//! module `bf16_rows_wide_pm_gfx1151` (gfx1151 only): the ordinary x4 entry
//! `gemv_bf16_xf32_rows_x4_t8_pm_gfx1151` and the k4 HC entry
//! `gemv_bf16_xf32_k4_rows_t8_pm_gfx1151`. Emitter metadata, ABI, resources,
//! proof obligations (barrier/EXEC discipline, row masking, early returns),
//! the per-row source DAG, instruction counts, weight-load reuse, the
//! committed bundle, and the `rowtiles` CPU oracles at every R=1..64.
//!
//! The `rowtiles` tests execute the entries against independent finite-input
//! DAG transcriptions; they are not an actual-singleton-object certificate.
//! HC's explicit refusal and structural/ownership tests are not HC numerical
//! proof; raw/activated HC bytes, extreme inputs and actual-singleton parity
//! remain mandatory hardware gates.
use peacemaker_isa::kernels::qwen4_bf16_rows_wide;
use peacemaker_isa::Arch;

const ARCH: Arch = Arch::Gfx1151;
const MODULE: &str = "bf16_rows_wide_pm_gfx1151";
const ORDINARY: &str = "gemv_bf16_xf32_rows_x4_t8_pm_gfx1151";
const HC: &str = "gemv_bf16_xf32_k4_rows_t8_pm_gfx1151";

/// Ordinary ABI: `A0..A3 x y0..y3` pointers, then `M0 M1 M2 M3 K R` i32.
const X4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 8),
    (48, 8),
    (56, 8),
    (64, 8),
    (72, 4),
    (76, 4),
    (80, 4),
    (84, 4),
    (88, 4),
    (92, 4),
];
/// HC k4 ABI: `W X Y` pointers, then `M K scale hc_act R` 4-byte values.
const K4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 4),
    (28, 4),
    (32, 4),
    (36, 4),
    (40, 4),
];

fn instructions(text: &str) -> Vec<&str> {
    let body = &text[..text.find(".section .rodata").unwrap()];
    body.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && !l.starts_with('.') && !l.starts_with(';') && !l.ends_with(':')
        })
        .collect()
}

fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(lines: &[&str], m: &str) -> usize {
    lines.iter().filter(|l| mnemonic(l) == m).count()
}
fn count_exact(lines: &[&str], line: &str) -> usize {
    lines.iter().filter(|l| **l == line).count()
}
fn operands(line: &str) -> Vec<&str> {
    line.strip_prefix(mnemonic(line))
        .unwrap()
        .split(',')
        .map(str::trim)
        .collect()
}
/// `v40` -> 40.
fn vreg(op: &str) -> u32 {
    op.split_whitespace()
        .next()
        .unwrap()
        .strip_prefix('v')
        .unwrap_or_else(|| panic!("not a VGPR: {op}"))
        .parse()
        .unwrap()
}
/// The first instruction after `i`, ignoring scheduling hints.
fn after<'a>(lines: &[&'a str], i: usize) -> &'a str {
    lines[i + 1..]
        .iter()
        .copied()
        .find(|l| mnemonic(l) != "s_delay_alu" && mnemonic(l) != "s_nop")
        .unwrap()
}
fn position(lines: &[&str], pred: impl Fn(&str) -> bool) -> usize {
    lines
        .iter()
        .position(|l| pred(l))
        .expect("instruction present")
}
fn kernel_field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

/// `(index, [acc, x, w])` of every `v_fmac_f32_e32` with three VGPR operands, in program order.
fn fmacs(lines: &[&str]) -> Vec<(usize, [u32; 3])> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| mnemonic(l) == "v_fmac_f32_e32")
        .map(|(i, l)| {
            let o = operands(l);
            (i, [vreg(o[0]), vreg(o[1]), vreg(o[2])])
        })
        .collect()
}

/// Mnemonic families no entry of the module may contain.
const FORBIDDEN: [&str; 11] = [
    "scratch_",
    "flat_",
    "buffer_atomic_",
    "global_atomic_",
    "v_readlane",
    "v_writelane",
    "s_setpc",
    "s_swappc",
    "v_wmma",
    "v_cvt_",
    "v_fma_mix",
];

/// The last EXEC write before every `s_barrier` (program order) is the all-ones restore.
fn assert_barriers_run_with_full_exec(lines: &[&str], what: &str, want: usize) {
    let mut last_exec: Option<&str> = None;
    let mut barriers = 0;
    for &l in lines {
        if mnemonic(l) == "s_barrier" {
            assert_eq!(
                last_exec,
                Some("s_mov_b32 exec_lo, -1"),
                "{what}: barrier #{barriers} with a narrowed EXEC"
            );
            barriers += 1;
        } else if l.contains("exec_lo") || mnemonic(l) == "v_cmpx_eq_u32_e32" {
            last_exec = Some(l);
        }
    }
    assert_eq!(barriers, want, "{what}: barrier count");
}

fn offsets(lines: &[&str], m: &str, keep: impl Fn(&str) -> bool) -> Vec<u32> {
    lines
        .iter()
        .filter(|l| mnemonic(l) == m && keep(l))
        .map(|l| {
            l.split("offset:")
                .nth(1)
                .map_or(0, |o| o.trim().parse().unwrap())
        })
        .collect()
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

fn t8(hc: bool) -> peacemaker_isa::Emitted {
    qwen4_bf16_rows_wide::emit_rowtiles(ARCH, hc).unwrap()
}

#[test]
fn t8_names_module_and_entry_symbols() {
    assert_eq!(qwen4_bf16_rows_wide::module(ARCH), MODULE);
    let (emitted, text, proof) = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 2);
    assert_eq!(proof.module, MODULE);
    assert_eq!(proof.kernels.len(), 2);
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 2);
    for symbol in [ORDINARY, HC] {
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
    assert!(!text.contains("rows_wide_r16"));
    assert!(!text.contains("rows_wide_r32"));
}

#[test]
fn t8_emission_is_deterministic_and_refuses_other_arches() {
    for hc in [false, true] {
        let (ea, ta, pa) = (
            t8(hc),
            t8(hc).s_text.clone(),
            t8(hc).proof.s_text_sha256.clone(),
        );
        let eb = t8(hc);
        assert_eq!(
            (ea.s_text.len(), ta.len()),
            (eb.s_text.len(), eb.s_text.len())
        );
        assert_eq!(ea.s_text, eb.s_text);
        assert_eq!(pa, eb.proof.s_text_sha256);
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            assert!(
                qwen4_bf16_rows_wide::emit_rowtiles(arch, hc).is_err(),
                "{arch:?}: cannot borrow Halo singleton-object proof"
            );
            assert!(
                qwen4_bf16_rows_wide::emit_module(arch).is_err(),
                "{arch:?}: module is exact-gfx1151 only"
            );
        }
    }
}

#[test]
fn t8_kernarg_abi_and_launch_geometry() {
    for (hc, args, bytes, block, lds, reads, writes) in [
        (false, X4_ARGS, 96u32, 32u16, 0u32, 5usize, 4usize),
        (true, K4_ARGS, 48u32, 128u16, 128u32, 2usize, 1usize),
    ] {
        let e = t8(hc);
        let t = &e.s_text;
        assert_eq!(
            kernel_field(t, ".offset:"),
            args.iter().map(|a| a.0).collect::<Vec<_>>()
        );
        assert_eq!(
            kernel_field(t, ".size:"),
            args.iter().map(|a| a.1).collect::<Vec<_>>()
        );
        assert_eq!(
            args.last().map(|&(o, n)| o + n),
            Some(if hc { 44 } else { 96 })
        );
        assert!(t.contains(&format!(".max_flat_workgroup_size: {block}\n")));
        assert!(t.contains(&format!(".group_segment_fixed_size: {lds}\n")));
        assert!(t.contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")));
        assert!(t.contains(".amdhsa_system_sgpr_workgroup_id_x 1\n"));
        assert!(t.contains(".amdhsa_system_sgpr_workgroup_id_y 1\n"));
        assert!(t.contains(".amdhsa_wavefront_size32 1\n"));
        assert_eq!(t.matches(".actual_access: read_only\n").count(), reads);
        assert_eq!(t.matches(".actual_access: write_only\n").count(), writes);
        assert_eq!((e.shape.next_free_vgpr, e.shape.next_free_sgpr), (80, 80));
        assert_eq!((e.proof.next_free_vgpr, e.proof.next_free_sgpr), (80, 80));
        for field in [
            ".private_segment_fixed_size: 0\n",
            ".vgpr_spill_count: 0\n",
            ".sgpr_spill_count: 0\n",
        ] {
            assert!(t.contains(field), "{field}");
        }
    }
}

#[test]
fn t8_entries_hold_forbidden_mnemonic_and_spill_contract() {
    for hc in [false, true] {
        let e = t8(hc);
        let what = if hc { HC } else { ORDINARY };
        let lines = instructions(&e.s_text);
        for pattern in FORBIDDEN {
            let prefix = pattern.trim_end_matches('*');
            assert!(
                !lines.iter().any(|l| mnemonic(l).starts_with(prefix)),
                "{what}: emission contains forbidden {prefix}*"
            );
        }
        assert_eq!(e.shape.next_free_vgpr, 80, "{what}: vgpr high-water");
        assert_eq!(e.shape.next_free_sgpr, 80, "{what}: sgpr high-water");
        assert!(e.s_text.contains(".wavefront_size: 32\n"), "{what}");
        assert_eq!(e.proof.next_free_vgpr, 80, "{what}: proof vgpr");
        assert_eq!(e.proof.next_free_sgpr, 80, "{what}: proof sgpr");
    }
}

/// The runtime embeds one certified module: its `.text` must be exactly what the builder emits today
/// (`peacemaker-isa emit --kernel qwen4_bf16_rows_wide --epi all --arch gfx1151 ... --bundle`).
#[test]
fn t8_committed_bundle_equals_native_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let path = format!("{root}/{MODULE}.hxaco");
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    assert!(
        text_section(&elf) == text_section(&committed),
        "{MODULE}: native emission differs from the committed bundle"
    );
}

mod rowtiles {
    use super::*;
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    use peacemaker_ir::inst::Frontend;
    use peacemaker_ir::Program;

    const ORDINARY: &str = "gemv_bf16_xf32_rows_x4_t8_pm_gfx1151";
    const HC: &str = "gemv_bf16_xf32_k4_rows_t8_pm_gfx1151";
    const KARG: u64 = 0x1_0000;
    const X: u64 = 0x1_0000_0000;
    const GUARD: usize = 64;
    const POISON: u32 = 0x7fc0_a5a5;

    fn emit(hc: bool) -> peacemaker_isa::Emitted {
        qwen4_bf16_rows_wide::emit_rowtiles(ARCH, hc).unwrap()
    }

    fn program(hc: bool) -> Program {
        let elf = peacemaker_isa::native::assemble(&emit(hc).s_text, ARCH).unwrap();
        peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program
    }

    #[test]
    fn frozen_symbols_abi_resources_and_arch_refusal() {
        for hc in [false, true] {
            let e = emit(hc);
            let t = &e.s_text;
            let symbol = if hc { HC } else { ORDINARY };
            let (args, bytes, block, lds, reads, writes) = if hc {
                (K4_ARGS, 48, 128, 128, 2, 1)
            } else {
                (X4_ARGS, 96, 32, 0, 5, 4)
            };
            assert_eq!(t.matches(&format!("{symbol}:\n")).count(), 1);
            assert_eq!(t.matches(".amdhsa_kernel ").count(), 1);
            assert!(t.contains(&format!(".amdhsa_kernel {symbol}\n")));
            assert!(t.contains(&format!(".amdhsa_kernarg_size {bytes}\n")));
            assert!(t.contains(&format!(".kernarg_segment_size: {bytes}\n")));
            assert_eq!(
                kernel_field(t, ".offset:"),
                args.iter().map(|a| a.0).collect::<Vec<_>>()
            );
            assert_eq!(
                kernel_field(t, ".size:"),
                args.iter().map(|a| a.1).collect::<Vec<_>>()
            );
            assert_eq!(
                args.last().map(|&(o, n)| o + n),
                Some(if hc { 44 } else { 96 })
            );
            assert!(t.contains(&format!(".max_flat_workgroup_size: {block}\n")));
            assert!(t.contains(&format!(".group_segment_fixed_size: {lds}\n")));
            assert!(t.contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")));
            assert!(t.contains(".amdhsa_system_sgpr_workgroup_id_x 1\n"));
            assert!(t.contains(".amdhsa_system_sgpr_workgroup_id_y 1\n"));
            assert!(t.contains(".amdhsa_wavefront_size32 1\n"));
            assert!(t.contains(".wavefront_size: 32\n"));
            assert!(t.contains(".workgroup_processor_mode: 1\n"));
            assert_eq!(t.matches(".actual_access: read_only\n").count(), reads);
            assert_eq!(t.matches(".actual_access: write_only\n").count(), writes);
            assert_eq!((e.shape.next_free_vgpr, e.shape.next_free_sgpr), (80, 80));
            assert_eq!((e.proof.next_free_vgpr, e.proof.next_free_sgpr), (80, 80));
            for field in [
                ".private_segment_fixed_size: 0\n",
                ".vgpr_spill_count: 0\n",
                ".sgpr_spill_count: 0\n",
            ] {
                assert!(t.contains(field), "{symbol}: {field}");
            }
            let again = emit(hc);
            assert_eq!(t, &again.s_text);
            assert_eq!(e.proof.s_text_sha256, again.proof.s_text_sha256);
            for arch in [Arch::Gfx1100, Arch::Gfx1201] {
                assert!(
                    qwen4_bf16_rows_wide::emit_rowtiles(arch, hc).is_err(),
                    "{arch:?}: cannot borrow Halo singleton-object proof"
                );
            }
        }
    }

    #[test]
    fn fixed_eight_row_dag_direct_loads_and_hc_quarters() {
        for hc in [false, true] {
            let e = emit(hc);
            let lines = instructions(&e.s_text);
            let fm = fmacs(&lines);
            // Fast and scalar paths both exist, independent of total R.
            for r in 0..8u32 {
                for j in 0..8u32 {
                    assert_eq!(
                        fm[(8 * r + j) as usize].1,
                        [40 + 4 * r + (j & 3), 24 + j, 16 + j]
                    );
                }
                assert_eq!(fm[64 + r as usize].1, [40 + 4 * r, 24, 16]);
            }
            assert_eq!(count(&lines, "buffer_load_b128"), 1);
            assert_eq!(count(&lines, "buffer_load_u16"), 1);
            assert_eq!(count(&lines, "global_load_b32"), 72);
            assert_eq!(count(&lines, "global_load_b128"), 0);
            assert_eq!(count(&lines, "ds_swizzle_b32"), 8);
            assert_eq!(count(&lines, "ds_bpermute_b32"), 32);
            assert_eq!(count(&lines, "ds_load_b32"), 0, "X is direct-global S1");
            assert_eq!(count(&lines, "global_store_b32"), 8);
            for forbidden in FORBIDDEN {
                assert!(
                    !lines.iter().any(|l| l.starts_with(forbidden)),
                    "{forbidden}"
                );
            }
            assert!(lines.contains(&"s_lshl_b32 s41, s3, 3"));
            assert!(lines.contains(&"s_cmp_gt_u32 s31, 64"));
            assert!(lines.contains(&"s_sub_u32 s45, s31, s41"));
            assert!(lines.contains(&"s_min_u32 s40, s45, 8"));
            // COUNT (s40) keeps exactly ONE definition: the domain min above.
            assert_eq!(count_exact(&lines, "s_min_u32 s40, s45, 8"), 1);
            assert_eq!(
                lines
                    .iter()
                    .filter(|l| {
                        let m = mnemonic(l);
                        !(m.starts_with("s_cmp")
                            || m.starts_with("s_cbranch")
                            || m.starts_with("s_branch"))
                            && matches!(operands(l).first().map(|s| s.trim()), Some("s40"))
                    })
                    .count(),
                1,
                "COUNT s40 must have exactly one definition",
            );
            // All dead-tile / row-base guards retained.
            assert!(lines.contains(&"s_cmp_ge_u32 s60, s34"));
            assert!(lines.contains(&"s_cmp_eq_u32 s31, 0"));
            assert!(lines.contains(&"s_cmp_le_i32 s30, 0"));
            assert!(lines.contains(&"s_cmp_ge_u32 s41, s31"));
            // Three fold operations per physical row, with the singleton's operand orientation.
            let reduce = fm[71].0;
            // Later adds (e.g. the activation's `1.0 + x`) take literal operands, so only the
            // fold entries are parsed as VGPR triples.
            let adds: Vec<_> = lines[reduce + 1..]
                .iter()
                .filter(|l| mnemonic(l) == "v_add_f32_e32")
                .collect();
            for r in 0..8 {
                let a = 40 + 4 * r as u32;
                let folds: Vec<Vec<u32>> = adds[8 * r..8 * r + 3]
                    .iter()
                    .map(|l| operands(l).iter().map(|o| vreg(o)).collect())
                    .collect();
                assert_eq!(
                    folds,
                    &[
                        vec![a + 1, a + 1, a],
                        vec![a + 3, a + 3, a + 2],
                        vec![a + 3, a + 3, a + 1]
                    ]
                );
            }
            if !hc {
                assert_eq!(count(&lines, "s_barrier"), 0);
                assert_eq!(count(&lines, "ds_store_b32"), 0);
                assert_eq!(count(&lines, "ds_load_b128"), 0);
                assert_eq!(e.shape.barriers, 0);
                continue;
            }
            assert!(lines.contains(&"s_and_b32 s45, s30, 31"));
            assert!(lines.contains(&"s_lshr_b32 s42, s30, 2"));
            assert!(lines.contains(&"s_mul_i32 s57, s43, s42"));
            assert!(lines.contains(&"s_lshl_b32 s45, s43, 2"));
            assert_eq!(
                offsets(&lines, "ds_store_b32", |_| true),
                (0..8).map(|r| 16 * r).collect::<Vec<_>>()
            );
            assert_eq!(count(&lines, "ds_load_b128"), 8);
            assert_eq!(e.shape.barriers, 1);
            assert_eq!(e.proof.barriers.len(), 1);
            assert_barriers_run_with_full_exec(&lines, "HC t8 quarter publication", 1);
            let barrier = position(&lines, |l| mnemonic(l) == "s_barrier");
            assert!(lines[..barrier]
                .iter()
                .all(|l| mnemonic(l) != "ds_load_b128"));
            assert!(lines[barrier + 1..]
                .iter()
                .all(|l| mnemonic(l) != "ds_store_b32"));
            assert!(lines[barrier + 1..].contains(&"s_cmp_lg_u32 s43, 0"));
            for fold in [
                "s_add_f32 s69, s64, s65",
                "s_add_f32 s70, s66, s67",
                "s_add_f32 s68, s69, s70",
                "s_cmp_eq_u32 s61, 0",
                "s_mul_f32 s68, s44, s68",
            ] {
                assert_eq!(count_exact(&lines, fold), 8, "{fold}");
            }
            for (op, per_row) in [
                ("s_bfe_u32", 3),
                ("s_addk_i32", 3),
                ("s_rndne_f32", 1),
                ("s_fmamk_f32", 2),
                ("v_exp_f32_e32", 1),
                ("v_ldexp_f32", 1),
                ("v_rcp_f32_e32", 1),
                ("v_div_scale_f32", 2),
                ("v_div_fmas_f32", 1),
                ("v_div_fixup_f32", 1),
                ("v_fma_f32", 2),
            ] {
                assert_eq!(count(&lines, op), 8 * per_row, "{op}: full HC expansion");
            }
        }
    }

    #[test]
    fn hc_raw_and_activated_cpu_bytes_are_explicitly_unsupported() {
        // Emulator::new validates the whole kernel, not just the selected branch.
        // Even hc_act=0 requires three unsupported s_add_f32 instructions.
        // Never replace SALU with VALU, skip on arbitrary errors, or call this numerical PASS.
        let p = program(true);
        let err = match Emulator::new(&p, HC) {
            Ok(_) => panic!("HC semantics changed: replace this gap assertion with actual raw/activated byte oracles"),
            Err(err) => err,
        };
        assert_eq!(err, "no semantics for s_add_f32");
        eprintln!(
            "HC RAW/ACTIVATED CPU BYTE COVERAGE: UNSUPPORTED ({err}); \
             all R=1..64 require actual-singleton hardware byte gates, including HC extremes"
        );
    }

    #[test]
    fn hc_quarter_diagnostic_assembles_and_lifts() {
        let symbol = format!("gemv_bf16_xf32_k4_rows_t8_quarters_pm_{}", ARCH.name());
        let emitted = qwen4_bf16_rows_wide::emit_hc_quarter_diagnostic(ARCH).unwrap();
        assert!(emitted.s_text.contains(&format!("{symbol}:\n")));
        let elf = peacemaker_isa::native::assemble(&emitted.s_text, ARCH).unwrap();
        assert!(!text_section(&elf).is_empty(), "diagnostic .text is empty");
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        assert!(
            program.kernels.iter().any(|k| k.symbol.0 == symbol),
            "lifted program is missing {symbol}"
        );
        #[cfg(feature = "toolchain")]
        {
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-qwen4-bf16-rows-wide-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let co = dir.join("hc_quarter_diagnostic.co");
            std::fs::write(&co, &elf).unwrap();
            let m7 = peacemaker_isa::pm_check::m7(&co, ARCH.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
    }

    #[test]
    fn every_width_has_unique_rows_outputs_and_hc_partial_owners() {
        for rows in 1usize..=64 {
            for ms in [[1, 2, 3, 9], [0, 7, 0, 8], [0, 0, 0, 1], [0; 4]] {
                let mut output = [
                    vec![0u8; rows * ms[0]],
                    vec![0u8; rows * ms[1]],
                    vec![0u8; rows * ms[2]],
                    vec![0u8; rows * ms[3]],
                ];
                for gy in 0..rows.div_ceil(8) {
                    let base = 8 * gy;
                    let count = (rows - base).min(8);
                    for gx in 0..ms.iter().sum::<usize>() {
                        let mut col = gx;
                        let matrix = (0..4)
                            .find(|&i| {
                                if col < ms[i] {
                                    true
                                } else {
                                    col -= ms[i];
                                    false
                                }
                            })
                            .unwrap();
                        for local in 0..count {
                            output[matrix][(base + local) * ms[matrix] + col] += 1;
                        }
                    }
                    // HC quarter slots are row-local and shared only by the four K owners.
                    let mut partials = [0u8; 128];
                    for q in 0..4 {
                        for local in 0..count {
                            let offset = 16 * local + 4 * q;
                            for byte in &mut partials[offset..offset + 4] {
                                *byte += 1;
                            }
                        }
                    }
                    assert!(partials[..16 * count].iter().all(|&n| n == 1));
                    assert!(partials[16 * count..].iter().all(|&n| n == 0));
                }
                assert!(
                    output.iter().flatten().all(|&n| n == 1),
                    "R={rows} M={ms:?}"
                );
                assert!(8 * rows.div_ceil(8) >= rows, "first dead y tile");
            }
            for k in [32usize, 96, 1024, 1056] {
                let mut terms = vec![0u8; k];
                for q in 0..4 {
                    for term in &mut terms[q * (k / 4)..(q + 1) * (k / 4)] {
                        *term += 1;
                    }
                }
                assert!(terms.iter().all(|&n| n == 1), "four fixed K quarters");
            }
        }
    }

    // Finite normal operands, plus both signed zero encodings. Exponents and mantissas
    // vary by row/column/K so a wrong stride, term order or chain fold cannot hide in all-ones data.
    fn sample(index: usize, salt: u32) -> f32 {
        let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
        x ^= x >> 16;
        x = x.wrapping_mul(0x85eb_ca6b);
        x ^= x >> 13;
        if index % 29 == 0 {
            return f32::from_bits(x & 0x8000_0000);
        }
        f32::from_bits((x & 0x807f_ffff) | ((119 + (x >> 25) % 12) << 23))
    }

    /// Independent transcription of the documented Halo singleton DAG, not its emitter helpers.
    /// Deliberately finite-input only: host arithmetic does not certify GPU NaN/denormal rules.
    fn dot(x: &[f32], weights: &[u16]) -> u32 {
        assert_eq!(x.len(), weights.len());
        let mut lane = [0.0f32; 32];
        for (l, result) in lane.iter_mut().enumerate() {
            let mut a = [0.0f32; 4];
            if x.len() % 8 == 0 {
                for group in (0..x.len()).step_by(256) {
                    for j in 0..8 {
                        let k = group + 8 * l + j;
                        if k < x.len() {
                            let w = f32::from_bits(u32::from(weights[k]) << 16);
                            a[j & 3] = x[k].mul_add(w, a[j & 3]);
                        }
                    }
                }
            } else {
                for k in (l..x.len()).step_by(32) {
                    let w = f32::from_bits(u32::from(weights[k]) << 16);
                    a[0] = x[k].mul_add(w, a[0]);
                }
            }
            a[1] += a[0];
            a[3] += a[2];
            a[3] += a[1];
            *result = a[3];
        }
        let before = lane;
        for l in 0..32 {
            lane[l] += before[l | 16];
        }
        for distance in [8, 4, 2, 1] {
            let before = lane;
            for l in 0..32 {
                lane[l] += before[if l + distance < 32 { l + distance } else { l }];
            }
        }
        lane[0].to_bits()
    }

    fn put32(args: &mut [u8], offset: usize, value: u32) {
        args[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(args: &mut [u8], offset: usize, value: u64) {
        args[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn wbase(i: usize) -> u64 {
        0x2_0000_0000 + i as u64 * 0x100_0000
    }
    fn ybase(i: usize) -> u64 {
        0x3_0000_0000 + i as u64 * 0x100_0000
    }

    fn run(emu: &Emulator<'_>, rows: usize, k: usize, ms: [usize; 4], salt: u32) {
        let mut mem = Memory::default();
        let mut args = vec![0u8; 96];
        let x: Vec<_> = (0..rows * k).map(|i| sample(i, salt)).collect();
        // Exact input extents, no mapped tail: inactive-row/element loads are errors.
        mem.map(
            "x".into(),
            X,
            x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        put64(&mut args, 32, X);
        let mut expected: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());
        for i in 0..4 {
            put32(&mut args, 72 + 4 * i, ms[i] as u32);
            if ms[i] == 0 {
                continue;
            } // Null pointers for every empty x4 position.
            let w: Vec<u16> = (0..ms[i] * k)
                .map(|j| (sample(j, salt.wrapping_add(17 + i as u32 * 97)).to_bits() >> 16) as u16)
                .collect();
            mem.map(
                format!("w{i}"),
                wbase(i),
                w.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .unwrap();
            let padded_rows = rows.div_ceil(8) * 8 + 8;
            let output: Vec<u8> = (0..(2 * GUARD / 4 + padded_rows * ms[i]))
                .flat_map(|_| POISON.to_le_bytes())
                .collect();
            mem.map(format!("y{i}"), ybase(i), output).unwrap();
            put64(&mut args, 8 * i, wbase(i));
            put64(&mut args, 40 + 8 * i, ybase(i) + GUARD as u64);
            for row in 0..rows {
                for col in 0..ms[i] {
                    expected[i].push(dot(&x[row * k..(row + 1) * k], &w[col * k..(col + 1) * k]));
                }
            }
        }
        put32(&mut args, 88, k as u32);
        put32(&mut args, 92, rows as u32);
        mem.map("kernarg".into(), KARG, args).unwrap();
        let before: Vec<_> = mem.regions.iter().map(|r| r.bytes.clone()).collect();
        // Overlaunch both axes by one to exercise uniform dead-workgroup refusal.
        for gy in 0..=rows.div_ceil(8) as u32 {
            for gx in 0..=ms.iter().sum::<usize>() as u32 {
                emu.run_workgroup(
                    &Launch {
                        kernarg_ptr: KARG,
                        workgroup: [gx, gy, 0],
                        block: [32, 1, 1],
                        dynamic_lds: 0,
                    },
                    &mut mem,
                )
                .unwrap_or_else(|err| panic!("R={rows} K={k} M={ms:?} wg=({gx},{gy}): {err}"));
            }
        }
        for (region, original) in mem.regions.iter().zip(before) {
            if !region.name.starts_with('y') {
                assert_eq!(region.bytes, original, "{} mutated", region.name);
                assert_eq!(region.written.count(), 0, "{} was written", region.name);
                continue;
            }
            let i: usize = region.name[1..].parse().unwrap();
            let end = GUARD + 4 * rows * ms[i];
            assert_eq!(region.written.count(), 4 * rows * ms[i]);
            for b in 0..region.bytes.len() {
                assert_eq!(
                    region.written.contains(b),
                    (GUARD..end).contains(&b),
                    "R={rows} K={k} M={ms:?} {} byte {b}: write ownership",
                    region.name
                );
                if !(GUARD..end).contains(&b) {
                    assert_eq!(
                        region.bytes[b], original[b],
                        "{} guard/inactive row",
                        region.name
                    );
                }
            }
            for (word, (&want, got)) in expected[i]
                .iter()
                .zip(region.bytes[GUARD..end].chunks_exact(4))
                .enumerate()
            {
                assert_eq!(
                    u32::from_le_bytes(got.try_into().unwrap()),
                    want,
                    "R={rows} K={k} M={ms:?} matrix={i} row={} col={}",
                    word / ms[i],
                    word % ms[i]
                );
            }
        }
    }

    #[test]
    fn ordinary_every_width_scalar_k_and_every_divisible8_lane_tail() {
        let p = program(false);
        let emu = Emulator::new(&p, ORDINARY).unwrap();
        for rows in 1..=64 {
            for k in [
                1, 2, 3, 7, 9, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513,
            ] {
                run(&emu, rows, k, [1, 0, 0, 0], 0x53a9);
            }
            // Every legal partial K256 lane mask, both before and after a full group.
            for tail in 1..=32 {
                for full in [0, 256] {
                    run(&emu, rows, full + 8 * tail, [0, 1, 0, 0], 0x7bd3);
                }
            }
            for k in [768, 1024, 1032] {
                run(&emu, rows, k, [0, 0, 1, 0], 0xb517);
            }
        }
    }

    #[test]
    fn ordinary_every_width_x4_differing_heights_and_all_empty_live_masks() {
        let p = program(false);
        let emu = Emulator::new(&p, ORDINARY).unwrap();
        for rows in 1..=64 {
            for mask in 0..16 {
                let ms = std::array::from_fn(|i| {
                    if mask & (1 << i) != 0 {
                        [1, 3, 7, 9][i]
                    } else {
                        0
                    }
                });
                for k in [33, 264] {
                    run(&emu, rows, k, ms, 0x1d35 + mask);
                }
            }
            // Include M=2 and M=8 as well as ragged heights and all four differing strides.
            run(&emu, rows, 520, [2, 8, 3, 1], 0x2a79);
        }
    }

    #[test]
    fn ordinary_invalid_domains_return_without_touching_unmapped_tensors() {
        let p = program(false);
        let emu = Emulator::new(&p, ORDINARY).unwrap();
        for (rows, k) in [(0, 33), (65, 33), (u32::MAX, 33), (1, 0), (64, u32::MAX)] {
            let mut args = vec![0u8; 96];
            put32(&mut args, 72, 1);
            put32(&mut args, 88, k);
            put32(&mut args, 92, rows);
            let original = args.clone();
            let mut mem = Memory::default();
            mem.map("kernarg".into(), KARG, args).unwrap();
            for gy in [0, 1, 8] {
                emu.run_workgroup(
                    &Launch {
                        kernarg_ptr: KARG,
                        workgroup: [0, gy, 0],
                        block: [32, 1, 1],
                        dynamic_lds: 0,
                    },
                    &mut mem,
                )
                .unwrap();
            }
            assert_eq!(mem.regions[0].bytes, original);
            assert_eq!(mem.regions[0].written.count(), 0);
        }
    }
}
