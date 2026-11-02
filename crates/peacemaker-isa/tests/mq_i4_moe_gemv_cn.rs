// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 K1-Cn multi-lane (2..8 decode lanes) slab decode GEMV module `mq_i4_moe_gemv_cn_pm_gfx1151`
//! (`mq_i4_moe_gemv_cn_{mq3i4,mq4sym}_{r256_gate_up,r128_down}`, one codec per symbol): symbols, K1's frozen 96-byte kernarg
//! ABI (`n_experts@88`, `reserved@92`), the unsigned `expert_ids[u] >= n_experts` exit (emitted order and, on the peacemaker CPU
//! interpreter, invalid ids `-1`, `i32::MIN`, `n`, `n + 1` against valid `n - 1`), the fixed LDS (8 slots x one column phase),
//! exact-arch gating, the code-end guard, the schedule census (every weight
//! load precedes the first barrier and none follows it, no weight load sits in a slot branch, the X staging and `vmcnt`
//! ladder of the two-phase gate), the per-slot instruction census of K1's numerical tree (dequantization once per group,
//! the partial/chain add per live slot, the fold and the bpermute reduction per live slot), the derived shape contract, the
//! committed eight-entry narrow + `_t8` production bundle and contracts, and (feature `toolchain`) narrow diagnostic wait replay and M7.
//!
//! Executed `_t8` consumers cover R1..64 with first/last production output blocks,
//! independent C1 narrow-K1 references, descriptor-z/absolute-slot addressing,
//! exact output bytes/write masks, sparse guard exits, clipped X, and count8
//! output tails (including padded strides and the unaligned gate split).
//! This is CPU evidence only: waits are no-ops and ended waves are excluded from
//! barrier rendezvous. It proves neither hardware wait/barrier timing, native
//! floating-point equivalence, capture/replay, throughput, nor product admission.
use peacemaker_isa::emit::CODE_END_TAIL;
use peacemaker_isa::kernels::mq_i4_moe_gemv_cn::{self as cn, Codec, Kind, Spec};
use peacemaker_isa::{native, Arch};

const SYMBOLS: [&str; 4] = [
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq3i4_r128_down",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down",
];
const PRODUCT_SYMBOLS: [&str; 8] = [
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq3i4_r128_down",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down",
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
    "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
];
const MODULE: &str = "mq_i4_moe_gemv_cn_pm_gfx1151";
/// K1's frozen ABI: `(offset, size, name)` in kernarg order.
const ARGS: [(u32, u32, &str); 17] = [
    (0, 8, "expert_weight_ptrs"),
    (8, 8, "expert_ids"),
    (16, 8, "expert_offsets"),
    (24, 8, "slot_ids"),
    (32, 8, "Xrot"),
    (40, 8, "Y0"),
    (48, 8, "Y1"),
    (56, 4, "M"),
    (60, 4, "K"),
    (64, 4, "U"),
    (68, 4, "C"),
    (72, 4, "x_row_div"),
    (76, 4, "x_src_rows"),
    (80, 4, "x_row_stride"),
    (84, 4, "y_slot_stride"),
    (88, 4, "n_experts"),
    (92, 4, "reserved"),
];
const S_CODE_END: [u8; 4] = 0xbf9f_0000u32.to_le_bytes();

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(lines: &[&str], mnemonic_name: &str) -> usize {
    lines
        .iter()
        .filter(|l| mnemonic(l) == mnemonic_name)
        .count()
}
/// Operand `n` (0-based) of an instruction line, split on `", "`.
fn operand(line: &str, n: usize) -> String {
    line.trim()
        .split_once(' ')
        .map(|(_, rest)| rest.split(", ").nth(n).unwrap_or("").trim().to_owned())
        .unwrap_or_default()
}
fn vreg(token: &str) -> u32 {
    token
        .trim_start_matches('v')
        .parse()
        .unwrap_or_else(|_| panic!("not a VGPR: {token}"))
}
fn imm_offset(line: &str) -> u32 {
    line.split("offset:")
        .nth(1)
        .map_or(0, |v| v.split_whitespace().next().unwrap().parse().unwrap())
}
fn is_weight_load(line: &str) -> bool {
    matches!(
        mnemonic(line),
        "global_load_b64" | "global_load_b32" | "global_load_u16"
    )
}
fn arch_specs() -> Vec<Spec> {
    cn::module_specs()
}
fn gate(spec: Spec) -> bool {
    spec.kind == Kind::GateUp
}
fn mq4(spec: Spec) -> bool {
    spec.codec == Codec::Mq4Sym
}

/// Mirror RegistryCli: all narrow entries, then all rowtiles, in the same image.
/// `cn::emit_module` deliberately remains the four-entry diagnostic API.
fn product_module() -> (
    Vec<peacemaker_isa::Emitted>,
    String,
    peacemaker_isa::kernels::iu4_gemm::ModuleProof,
) {
    let mut emitted = cn::module_specs()
        .into_iter()
        .map(cn::emit)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    emitted.extend(
        cn::module_specs()
            .into_iter()
            .map(cn::emit_rowtiles)
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    );
    let (text, proof) =
        peacemaker_isa::kernels::iu4_gemm::module(&emitted, &Spec::module()).unwrap();
    (emitted, text, proof)
}

/// The trimmed lines of the kernel body (after `.Lcn_<tag>_body:` up to `_exit:`).
fn kernel_lines(text: &str, spec: Spec) -> Vec<&str> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let at = |name: &str| {
        lines
            .iter()
            .position(|l| *l == format!(".Lcn_{}_{name}:", spec.tag()))
            .unwrap_or_else(|| panic!("label {name}"))
    };
    let (body, exit) = (at("body"), at("exit"));
    assert!(body < exit);
    lines[body + 1..exit].to_vec()
}
/// The lines of `body` strictly between the labels `from:` and `to:` (full label names without the colon).
fn between<'a>(body: &[&'a str], spec: Spec, from: &str, to: &str) -> Vec<&'a str> {
    let at = |name: &str| {
        body.iter()
            .position(|l| *l == format!(".Lcn_{}_{name}:", spec.tag()))
            .unwrap_or_else(|| panic!("label {name}"))
    };
    body[at(from) + 1..at(to)].to_vec()
}

#[test]
fn diagnostic_module_exports_are_the_four_frozen_symbols_in_order() {
    let specs = cn::module_specs();
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(cn::SYMBOLS, SYMBOLS);
    assert_eq!(
        specs.iter().map(Spec::tag).collect::<Vec<_>>(),
        [
            "mq3i4_r256_gate_up",
            "mq3i4_r128_down",
            "mq4sym_r256_gate_up",
            "mq4sym_r128_down"
        ]
    );
    assert_eq!(Spec::module(), MODULE);
    let (emitted, text, proof) = cn::emit_module().unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, Arch::Gfx1151));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: not global"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));
}

#[test]
fn production_module_exports_narrow_and_rowtiles_in_registry_order() {
    let (emitted, text, proof) = product_module();
    assert_eq!(emitted.len(), 8);
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, Arch::Gfx1151));
    assert_eq!(proof.kernels.len(), 8);
    let exports: Vec<&str> = text
        .lines()
        .filter_map(|line| line.trim().strip_prefix(".amdhsa_kernel "))
        .collect();
    assert_eq!(exports, PRODUCT_SYMBOLS);
    assert_eq!(text.matches(".globl ").count(), 8);
    for symbol in PRODUCT_SYMBOLS {
        assert!(text.contains(&format!(".globl {symbol}\n")), "{symbol}");
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}");
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));

    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    let strings = section(&elf, ".strtab");
    let mut exports: Vec<String> = section(&elf, ".symtab")
        .chunks_exact(24)
        .filter(|symbol| symbol[4] == 0x12 && u16_at(symbol, 6) != 0)
        .map(|symbol| {
            let start = u32_at(symbol, 0);
            let end = start + strings[start..].iter().position(|&byte| byte == 0).unwrap();
            String::from_utf8(strings[start..end].to_vec()).unwrap()
        })
        .collect();
    exports.sort();
    let mut expected = PRODUCT_SYMBOLS.map(str::to_owned);
    expected.sort();
    assert_eq!(exports, expected, "native production object exports");
}

/// The total-row tile ABI changes neither the 96-byte layout nor the cap8 bank.
#[test]
fn rowtiles_keep_the_frozen_abi_resources_and_derive_their_own_contracts() {
    for spec in arch_specs() {
        let e = cn::emit_rowtiles(spec).unwrap();
        let again = cn::emit_rowtiles(spec).unwrap();
        let narrow = cn::emit(spec).unwrap();
        let symbol = format!("{}_t8", spec.symbol());
        assert_eq!(e.s_text, again.s_text, "{symbol}: deterministic source");
        assert_eq!(e.proof.s_text_sha256, again.proof.s_text_sha256);
        assert_eq!(e.proof.variant, format!("{}_t8", spec.tag()));
        for marker in [
            format!(".amdhsa_kernel {symbol}\n"),
            format!(".name: {symbol}\n"),
            ".amdhsa_kernarg_size 96\n".to_owned(),
            ".kernarg_segment_size: 96\n".to_owned(),
            ".amdhsa_wavefront_size32 1\n".to_owned(),
            ".amdhsa_system_sgpr_workgroup_id_y 1\n".to_owned(),
            ".amdhsa_system_sgpr_workgroup_id_z 1\n".to_owned(),
            ".private_segment_fixed_size: 0\n".to_owned(),
            ".sgpr_spill_count: 0\n".to_owned(),
            ".vgpr_spill_count: 0\n".to_owned(),
        ] {
            assert!(e.s_text.contains(&marker), "{symbol}: {marker}");
        }
        assert!(!narrow
            .s_text
            .contains(".amdhsa_system_sgpr_workgroup_id_z 1\n"));
        assert_eq!(field(&e.s_text, ".offset:"), ARGS.map(|arg| arg.0));
        assert_eq!(field(&e.s_text, ".size:"), ARGS.map(|arg| arg.1));
        let names: Vec<&str> = e
            .s_text
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".name: "))
            .take(ARGS.len())
            .collect();
        assert_eq!(names, ARGS.map(|arg| arg.2));
        for (name, access) in [
            ("expert_weight_ptrs", "read_only"),
            ("expert_ids", "read_only"),
            ("expert_offsets", "read_only"),
            ("slot_ids", "read_only"),
            ("Xrot", "read_only"),
            ("Y0", "write_only"),
            ("Y1", "write_only"),
        ] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access");
        }
        assert!(
            !e.s_text.contains("s[0:1], 0x5c"),
            "{symbol}: reserved is unread"
        );
        assert_eq!(field(&e.s_text, ".max_flat_workgroup_size:"), [256]);
        assert_eq!(
            field(&e.s_text, ".group_segment_fixed_size:"),
            [spec.kind.lds_bytes()]
        );
        assert_eq!(
            field(&e.s_text, ".amdhsa_group_segment_fixed_size"),
            [spec.kind.lds_bytes()]
        );
        assert_eq!(e.shape.next_free_vgpr, narrow.shape.next_free_vgpr);
        assert_eq!(e.shape.next_free_sgpr, narrow.shape.next_free_sgpr);
        assert!(e.shape.next_free_vgpr <= 128, "{symbol}");
        assert_eq!(
            field(&e.s_text, ".amdhsa_next_free_vgpr"),
            [u32::from(e.shape.next_free_vgpr)]
        );
        assert_eq!(
            field(&e.s_text, ".vgpr_count:"),
            [u32::from(e.shape.next_free_vgpr)]
        );
        assert_eq!(field(&e.s_text, ".sgpr_count:"), [101]);
        assert_eq!(e.shape.barriers, if gate(spec) { 3 } else { 1 });
        assert!(
            e.s_text.contains("s_mul_i32 s41, s4, 0x788"),
            "{symbol}: descriptor stride"
        );

        let contract = cn::contract_rowtiles(spec, &e);
        assert_eq!(contract, cn::contract_rowtiles(spec, &again));
        assert_eq!(contract["symbol"].as_str(), Some(symbol.as_str()));
        assert_eq!(
            contract["vgpr_max"].as_u64(),
            Some(u64::from(e.shape.next_free_vgpr))
        );
        assert_eq!(
            contract["sgpr_max"].as_u64(),
            Some(u64::from(e.shape.next_free_sgpr))
        );
        assert_eq!(contract["group_segment_fixed_bytes"], spec.kind.lds_bytes());
        assert_eq!(contract["launch_dynamic_lds_bytes"], 0);
        for flag in [
            "require_wave32",
            "require_zero_spills",
            "require_zero_private",
        ] {
            assert_eq!(contract[flag], true, "{symbol}: {flag}");
        }
        assert_eq!(
            contract["forbidden"],
            cn::contract(spec, &narrow)["forbidden"]
        );
        for (name, n) in contract["counts"].as_object().unwrap() {
            let have = e
                .s_text
                .lines()
                .filter(|line| mnemonic(line) == name)
                .count() as u64;
            assert_eq!(n.as_u64(), Some(have), "{symbol}: {name}");
        }
        assert_eq!(
            contract["counts"]["s_barrier"],
            if gate(spec) { 3 } else { 1 }
        );
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1201, Arch::Gfx942] {
        for codec in Codec::ALL {
            for kind in Kind::ALL {
                assert!(cn::emit_rowtiles(Spec { arch, codec, kind }).is_err());
            }
        }
    }
}

/// Emission is bit-deterministic, and every architecture but gfx1151 is refused (no implicit alias).
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in arch_specs() {
        let (a, b) = (cn::emit(spec).unwrap(), cn::emit(spec).unwrap());
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
    }
    let (_, ta, pa) = cn::emit_module().unwrap();
    let (_, tb, pb) = cn::emit_module().unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1201, Arch::Gfx942] {
        for codec in Codec::ALL {
            for kind in Kind::ALL {
                let spec = Spec { arch, codec, kind };
                assert!(
                    spec.validate().is_err() && cn::emit(spec).is_err(),
                    "{arch:?} {codec:?} {kind:?}"
                );
            }
        }
    }
}

/// Wave32, eight waves per workgroup, grid y = the expert, fixed LDS of 8 slots x one column phase (40,960 B gate = 1,280
/// F32 per slot, 20,480 B down = 640), no scratch/spills/private, and the register budget the schedule targets (<= 128 VGPRs).
#[test]
fn launch_resources_match_the_frozen_geometry() {
    assert_eq!(
        (
            cn::KERNARG_BYTES,
            cn::WORKGROUP_SIZE,
            cn::ROWS_PER_WORKGROUP,
            cn::MAX_SLOTS
        ),
        (96, 256, 8, 8)
    );
    for spec in arch_specs() {
        let lds = if gate(spec) { 40960u32 } else { 20480 };
        let e = cn::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert_eq!(spec.kind.lds_bytes(), lds);
        assert!(
            e.s_text
                .contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")),
            "{symbol}: descriptor LDS"
        );
        assert!(
            e.s_text
                .contains(&format!(".group_segment_fixed_size: {lds}\n")),
            "{symbol}: metadata LDS"
        );
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [256],
            "{symbol}: block 256"
        );
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{symbol}: grid y is the expert"
        );
        assert!(
            e.s_text.contains(".private_segment_fixed_size: 0\n"),
            "{symbol}: private segment"
        );
        assert!(
            e.s_text.contains(".sgpr_spill_count: 0\n")
                && e.s_text.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: spills"
        );
        let vgprs = field(&e.s_text, ".amdhsa_next_free_vgpr");
        assert_eq!(vgprs.len(), 1, "{symbol}");
        assert!(vgprs[0] <= 128, "{symbol}: {} VGPRs", vgprs[0]);
        assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0], "{symbol}");
        assert!(
            e.s_text.contains(&format!(".vgpr_count: {}\n", vgprs[0])),
            "{symbol}: metadata VGPR count"
        );
        assert!(
            !e.s_text
                .lines()
                .any(|l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")),
            "{symbol}: scratch/flat"
        );
        assert_eq!(
            e.shape.barriers,
            if gate(spec) { 3 } else { 1 },
            "{symbol}: gate publish/retire/publish, down publish"
        );
    }
}

/// K1's 96-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the first loads.
#[test]
fn kernarg_abi_matches_the_frozen_96_byte_layout() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 96\n"),
            "{symbol}: kernarg size"
        );
        assert!(
            e.s_text.contains(".amdhsa_kernarg_size 96\n"),
            "{symbol}: descriptor kernarg size"
        );
        assert_eq!(
            field(&e.s_text, ".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(&e.s_text, ".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
        let names: Vec<String> = e
            .s_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix(".name: "))
            .take(ARGS.len())
            .map(str::to_owned)
            .collect();
        assert_eq!(
            names,
            ARGS.iter().map(|a| a.2.to_owned()).collect::<Vec<_>>(),
            "{symbol}: kernarg names"
        );
        let (last_off, last_size, _) = *ARGS.last().unwrap();
        assert_eq!(
            last_off + last_size,
            cn::KERNARG_BYTES,
            "{symbol}: arguments cover the kernarg block"
        );
        for (name, access) in [
            ("expert_weight_ptrs", "read_only"),
            ("expert_ids", "read_only"),
            ("expert_offsets", "read_only"),
            ("slot_ids", "read_only"),
            ("Xrot", "read_only"),
            ("Y0", "write_only"),
            ("Y1", "write_only"),
        ] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access {access}");
        }
        // The kernarg is loaded by five scalar loads covering offsets 0..92 with no over-read (`reserved@92` is never read);
        // `n_experts` lands in its own SGPR (s100, the declared ceiling is 101 SGPRs, `sgpr_max` of the contract).
        for load in [
            "s_load_b256 s[8:15], s[0:1], null",
            "s_load_b256 s[16:23], s[0:1], 0x20",
            "s_load_b128 s[24:27], s[0:1], 0x40",
            "s_load_b64 s[28:29], s[0:1], 0x50",
            "s_load_b32 s100, s[0:1], 0x58",
        ] {
            assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
        }
        assert!(
            !e.s_text.contains("s[0:1], 0x5c"),
            "{symbol}: reserved@92 is never read"
        );
        assert_eq!(
            field(&e.s_text, ".sgpr_count:"),
            [101],
            "{symbol}: sgpr count"
        );
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_n_experts"),
            "{symbol}: typed n_experts ownership"
        );
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_ptrs"),
            "{symbol}: typed kernarg ownership"
        );
    }
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap()) as usize
}
fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}
fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}
/// The bytes of the named section.
fn section(elf: &[u8], name: &str) -> Vec<u8> {
    let (shoff, entsize, count, strndx) = (
        u64_at(elf, 40),
        u16_at(elf, 58),
        u16_at(elf, 60),
        u16_at(elf, 62),
    );
    let header = |i: usize| shoff + i * entsize;
    let names = u64_at(elf, header(strndx) + 24);
    for i in 0..count {
        let at = names + u32_at(elf, header(i));
        let end = at + elf[at..].iter().position(|&c| c == 0).unwrap();
        if &elf[at..end] == name.as_bytes() {
            let (offset, size) = (u64_at(elf, header(i) + 24), u64_at(elf, header(i) + 32));
            return elf[offset..offset + size].to_vec();
        }
    }
    panic!("no section {name}")
}
fn function_sizes(elf: &[u8]) -> Vec<usize> {
    section(elf, ".symtab")
        .chunks(24)
        .filter(|s| s[4] & 0xf == 2)
        .map(|s| u64_at(s, 16))
        .collect()
}

/// The module (and each entry) carries the gfx11 `s_code_end` guard exactly once at the end, the native writer lays it
/// out after the last kernel (128-byte alignment fill then 96 `s_code_end` dwords) without moving any kernel, symbol
/// size or descriptor.
#[test]
fn code_end_tail_is_present_once_and_changes_nothing_else() {
    let (emitted, text, _) = cn::emit_module().unwrap();
    for t in emitted
        .iter()
        .map(|e| e.s_text.as_str())
        .chain([text.as_str()])
    {
        assert!(t.ends_with(CODE_END_TAIL));
        assert_eq!(t.matches(".p2alignl").count(), 1);
        assert_eq!(t.matches(".fill").count(), 1);
    }
    let bare = text
        .strip_suffix(CODE_END_TAIL)
        .expect("module ends with the guard");
    let guarded = native::assemble(&text, Arch::Gfx1151).unwrap();
    let plain = native::assemble(bare, Arch::Gfx1151).unwrap();
    let (guarded_text, bare_text) = (section(&guarded, ".text"), section(&plain, ".text"));
    assert_eq!(
        guarded_text[..bare_text.len()],
        bare_text[..],
        "the guard only appends"
    );
    assert!(
        guarded_text[bare_text.len()..]
            .chunks(4)
            .all(|w| w == S_CODE_END),
        "alignment fill and guard are s_code_end"
    );
    assert!(
        guarded_text.len() >= bare_text.len() + 96 * 4 && guarded_text.len() % 128 == 0,
        "128-byte aligned code end plus 96 guard dwords"
    );
    assert!(
        guarded_text
            .chunks(4)
            .rev()
            .take(96)
            .all(|w| w == S_CODE_END),
        "the last 96 dwords are s_code_end"
    );
    assert_eq!(function_sizes(&guarded), function_sizes(&plain));
    assert_eq!(section(&guarded, ".rodata"), section(&plain, ".rodata"));
    assert_eq!(function_sizes(&guarded).len(), SYMBOLS.len());
}

/// Weight loads of the row: every one precedes the first `s_barrier` and none follows it; none sits inside a slot branch
/// (they come after the last prologue guard and before the first compute label, and the body between the first and the last
/// weight load holds no branch or label); the phase-0 X loads precede every weight load; the first explicit
/// `vmcnt(<weight loads>)` separates the last weight load from the first `ds_store_b128`; the post-barrier `vmcnt` ladder
/// is the exact per-group descending sequence for phase 0 (the loads of a group are the oldest not yet consumed) and the
/// single `vmcnt(0)` of the gate's second phase.
#[test]
fn every_weight_load_precedes_the_barrier_and_the_vmcnt_ladder_is_exact() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        let weights = if g { 20 } else { 10 };
        let x_phase = if g { 10 } else { 5 };
        let phases = if g { 2 } else { 1 };
        let idx = |pred: &dyn Fn(&str) -> bool| -> Vec<usize> {
            body.iter()
                .enumerate()
                .filter(|(_, l)| pred(l))
                .map(|(i, _)| i)
                .collect()
        };
        let weight_ix = idx(&is_weight_load);
        let x_ix = idx(&|l| mnemonic(l) == "global_load_b128");
        let barrier_ix = idx(&|l| mnemonic(l) == "s_barrier");
        let store_ix = idx(&|l| mnemonic(l) == "ds_store_b128");
        assert_eq!(weight_ix.len(), weights, "{ctx}: weight loads");
        assert_eq!(x_ix.len(), x_phase * phases, "{ctx}: X staging loads");
        assert_eq!(store_ix.len(), x_phase * phases, "{ctx}: X stores");
        assert_eq!(barrier_ix.len(), if g { 3 } else { 1 }, "{ctx}: barriers");
        assert!(
            x_ix[..x_phase]
                .iter()
                .all(|x| weight_ix.iter().all(|w| x < w)),
            "{ctx}: phase-0 X loads precede every weight load"
        );
        assert!(
            weight_ix.iter().all(|w| *w < barrier_ix[0]),
            "{ctx}: every weight load precedes the first barrier"
        );
        assert!(
            body[barrier_ix[0]..].iter().all(|l| !is_weight_load(l)),
            "{ctx}: no weight load after the first barrier"
        );
        // not inside a slot branch: the weight loads sit between the prologue and the first compute label with no branch or label between them
        let (first, last) = (weight_ix[0], *weight_ix.last().unwrap());
        assert!(
            body[first..=last]
                .iter()
                .all(|l| !l.starts_with("s_cbranch") && !l.starts_with(".L") && *l != "s_branch"),
            "{ctx}: weight loads are straight-line"
        );
        assert!(
            store_ix[..x_phase]
                .iter()
                .all(|s| *s < barrier_ix[0] && *s > last),
            "{ctx}: phase-0 X stores sit between the loads and the first barrier"
        );
        // the explicit first wait
        let first_wait = idx(&|l| *l == format!("s_waitcnt vmcnt({weights})"));
        assert_eq!(
            first_wait.len(),
            1,
            "{ctx}: the explicit wait for the X loads"
        );
        assert!(
            first_wait[0] > last && first_wait[0] < store_ix[0],
            "{ctx}: explicit wait placement"
        );
        // the waits after the first barrier, in order
        let waits: Vec<u32> = body[barrier_ix[0]..]
            .iter()
            .filter_map(|l| l.strip_prefix("s_waitcnt vmcnt("))
            .filter_map(|r| r.split(')').next())
            .map(|n| n.parse().unwrap())
            .collect();
        let expect: Vec<u32> = if g {
            // phase 0: groups 0..9 (W3) / 0..4 (MQ4 gate, two loads per group) consume the oldest loads; then vmcnt(0) for the phase-1 X loads
            let mut v: Vec<u32> = (10..20).rev().collect();
            v.push(0);
            v
        } else if q {
            (0..10).rev().collect()
        } else {
            vec![8, 6, 4, 2, 0]
        };
        assert_eq!(waits, expect, "{ctx}: vmcnt ladder");
        if g {
            // phase 1: the retire barrier sits right after the second set of X loads, `vmcnt(0)` after it, the stores after that
            assert!(
                x_ix[x_phase..]
                    .iter()
                    .all(|x| *x > barrier_ix[0] && *x < barrier_ix[1]),
                "{ctx}: phase-1 X loads between the publish and the retire barrier"
            );
            let zero_wait = idx(&|l| l == "s_waitcnt vmcnt(0)");
            assert_eq!(zero_wait.len(), 1, "{ctx}");
            assert!(
                zero_wait[0] > barrier_ix[1]
                    && store_ix[x_phase..]
                        .iter()
                        .all(|s| *s > zero_wait[0] && *s < barrier_ix[2]),
                "{ctx}: phase-1 stores after vmcnt(0), before the publish barrier"
            );
        }
        // one lane-0 store per live slot region; no other memory ops; one endpgm (the exit label's, outside the body)
        assert_eq!(
            count(&body, "global_store_b32"),
            8,
            "{ctx}: one Y store per slot region"
        );
        assert_eq!(count(&body, "v_cmpx_eq_u32_e32"), 1, "{ctx}");
        assert!(
            !body.iter().any(|l| mnemonic(l).starts_with("flat_")
                || mnemonic(l).starts_with("scratch_")
                || mnemonic(l).starts_with("global_atomic")),
            "{ctx}: no other memory ops"
        );
    }
}

/// X staging addresses (immediates and offset registers of the `global_load_b128` and LDS stores) and the weight-load
/// addresses (group stride, packet / header / payload immediates) follow the formats; every `ds_load_b128` offset follows
/// `slot*slot_bytes + 512*g_local` (W3 and MQ4 down) / `+ 1024*g_local` and `+16` (MQ4 gate) in group-major slot order.
#[test]
fn addresses_follow_the_formats() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        // X loads: per phase `j = 0..` immediates 512 j (gate j >= 8: second register, 512 (j - 8))
        let x: Vec<&&str> = body
            .iter()
            .filter(|l| mnemonic(l) == "global_load_b128")
            .collect();
        let phase_expect: Vec<u32> = if g {
            (0..8).map(|j| 512 * j).chain([0, 512]).collect()
        } else {
            (0..5).map(|j| 512 * j).collect()
        };
        for ph in x.chunks(phase_expect.len()) {
            assert_eq!(
                ph.iter().map(|l| imm_offset(l)).collect::<Vec<_>>(),
                phase_expect,
                "{ctx}: X load immediates"
            );
            if g {
                let regs: Vec<String> = ph.iter().map(|l| operand(l, 1)).collect();
                assert!(
                    regs[..8].iter().all(|r| *r == regs[0])
                        && regs[8] == regs[9]
                        && regs[8] != regs[0],
                    "{ctx}: two X offset registers (+4096 for j >= 8)"
                );
            }
        }
        let stores: Vec<u32> = body
            .iter()
            .filter(|l| mnemonic(l) == "ds_store_b128")
            .map(|l| imm_offset(l))
            .collect();
        let per_phase: Vec<u32> = (0..phase_expect.len() as u32).map(|j| 512 * j).collect();
        assert_eq!(
            stores,
            per_phase
                .iter()
                .chain(if g { per_phase.iter() } else { [].iter() })
                .copied()
                .collect::<Vec<_>>(),
            "{ctx}: LDS stores"
        );
        // weight loads
        let groups = spec.groups();
        if !q {
            let b64: Vec<u32> = body
                .iter()
                .filter(|l| mnemonic(l) == "global_load_b64")
                .map(|l| imm_offset(l))
                .collect();
            assert_eq!(
                b64,
                (0..groups).map(|k| 48 * k).collect::<Vec<_>>(),
                "{ctx}: W3 packet windows"
            );
            let scales: Vec<u32> = body
                .iter()
                .filter(|l| mnemonic(l) == "global_load_u16")
                .map(|l| imm_offset(l))
                .collect();
            assert_eq!(
                scales,
                if g {
                    vec![]
                } else {
                    (0..groups).map(|k| 2 * k).collect::<Vec<_>>()
                },
                "{ctx}: W3 scales"
            );
            if g {
                // the row's 40 scale bytes at +40*row: ten SGPRs
                assert!(
                    body.contains(&"s_load_b256 s[88:95], s[50:51], null")
                        && body.contains(&"s_load_b64 s[96:97], s[50:51], 0x20"),
                    "{ctx}: gate scale loads"
                );
            }
        } else {
            let loads: Vec<&&str> = body.iter().filter(|l| is_weight_load(l)).collect();
            let (hdr, pay): (Vec<u32>, Vec<u32>) = (
                loads.iter().step_by(2).map(|l| imm_offset(l)).collect(),
                loads
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .map(|l| imm_offset(l))
                    .collect(),
            );
            let (stride, pay_off) = if g { (136, 8) } else { (68, 4) };
            assert_eq!(
                hdr,
                (0..groups).map(|k| stride * k).collect::<Vec<_>>(),
                "{ctx}: MQ4 headers"
            );
            assert_eq!(
                pay,
                (0..groups)
                    .map(|k| stride * k + pay_off)
                    .collect::<Vec<_>>(),
                "{ctx}: MQ4 payloads"
            );
            let payload_mnemonic = if g {
                "global_load_b32"
            } else {
                "global_load_u16"
            };
            assert_eq!(
                body.iter()
                    .filter(|l| mnemonic(l) == payload_mnemonic)
                    .count(),
                if g {
                    2 * groups as usize
                } else {
                    groups as usize
                },
                "{ctx}: payload-class loads"
            );
        }
        // LDS X reads, in program order: group-major, slot ascending
        let slot_bytes = spec.kind.slot_bytes();
        let lds: Vec<u32> = body
            .iter()
            .filter(|l| mnemonic(l) == "ds_load_b128")
            .map(|l| imm_offset(l))
            .collect();
        let per_phase_groups = groups / spec.kind.phases();
        let mut expect = Vec::new();
        for k in 0..groups {
            let gl = k % per_phase_groups;
            for i in 0..8u32 {
                if q && g {
                    expect.extend([i * slot_bytes + 1024 * gl, i * slot_bytes + 1024 * gl + 16])
                } else {
                    expect.push(i * slot_bytes + 512 * gl)
                }
            }
        }
        assert_eq!(lds, expect, "{ctx}: LDS X reads");
    }
}

/// The ops of one slot region of a group (between slot gates) as `(mul, fmac, add)` plus the operand check of K1's DAG.
/// Returns the destination accumulator register of its chain add.
fn partial_region(region: &[&str], spec: Spec, ctx: &str) -> u32 {
    let (mul, fmac, add): (Vec<&&str>, Vec<&&str>, Vec<&&str>) = (
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_mul_f32_e32")
            .collect(),
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_fmac_f32_e32")
            .collect(),
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_add_f32_e32")
            .collect(),
    );
    let nw = if mq4(spec) && gate(spec) { 8 } else { 4 };
    assert_eq!(
        (mul.len(), fmac.len(), add.len()),
        (1, nw - 1, 1),
        "{ctx}: per-slot value ops equal C1's per-group partial"
    );
    let dot = operand(mul[0], 0);
    // W3: p = w0*x0, fmac(w_j, x_j) j = 1..; MQ4: p = w1*x1, fmac(w0, x0), fmac(w_j, x_j) j = 2..
    let (w0, x0) = (
        vreg(&operand(if mq4(spec) { fmac[0] } else { mul[0] }, 1)),
        vreg(&operand(if mq4(spec) { fmac[0] } else { mul[0] }, 2)),
    );
    let first = usize::from(mq4(spec));
    assert_eq!(
        (vreg(&operand(mul[0], 1)), vreg(&operand(mul[0], 2))),
        (w0 + first as u32, x0 + first as u32),
        "{ctx}: first product"
    );
    let rest: Vec<&&str> = if mq4(spec) {
        fmac[1..].to_vec()
    } else {
        fmac.clone()
    };
    for (n, f) in rest.iter().enumerate() {
        let j = if mq4(spec) { n + 2 } else { n + 1 } as u32;
        assert_eq!(
            (operand(f, 0), vreg(&operand(f, 1)), vreg(&operand(f, 2))),
            (dot.clone(), w0 + j, x0 + j),
            "{ctx}: fmac term {j}"
        );
    }
    assert_eq!(
        operand(add[0], 0),
        operand(add[0], 1),
        "{ctx}: acc = acc + p (acc is src0)"
    );
    assert_eq!(operand(add[0], 2), dot, "{ctx}: the partial is src1");
    vreg(&operand(add[0], 0))
}

/// Per-group census: the dequantization appears ONCE per group (the same counts as C1's per-group unpack) and every live-slot
/// region carries exactly one partial (C1's per-group mul/fmac/chain-add counts) into `acc[slot][g & 3]` (down: chain 0), with
/// scalar slot gates between regions; no weight load or LDS write inside any group.
#[test]
fn per_group_census_matches_the_frozen_dag() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        let chains = spec.kind.chains() as u32;
        let nw = if q && g { 8 } else { 4 };
        // the accumulators: the first block of `8*chains` consecutive v_mov ..., 0 after the first weight load
        let first_load = body.iter().position(|l| is_weight_load(l)).unwrap();
        let acc: Vec<u32> = body[first_load..]
            .iter()
            .filter(|l| l.starts_with("v_mov_b32_e32 v") && l.ends_with(", 0"))
            .map(|l| vreg(&operand(l, 0)))
            .take(8 * chains as usize)
            .collect();
        assert_eq!(
            acc.len(),
            8 * chains as usize,
            "{ctx}: accumulators start at +0"
        );
        assert!(
            acc.windows(2).all(|w| w[1] == w[0] + 1),
            "{ctx}: consecutive accumulators"
        );
        let batches = if q && g { 2 } else { 1 };
        let bsz = 8 / batches;
        for k in 0..spec.groups() {
            let chain = if g { k & 3 } else { 0 };
            for batch in 0..batches {
                let (lw, ld) = (format!("g{k}_b{batch}_lw"), format!("g{k}_b{batch}_ld"));
                let section = between(&body, spec, &lw, &ld);
                // dequantization: before the explicit lgkmcnt(0), once per group (batch 0 only)
                let wait = section
                    .iter()
                    .position(|l| *l == "s_waitcnt lgkmcnt(0)")
                    .unwrap_or_else(|| panic!("{ctx} g{k}: explicit lgkmcnt(0)"));
                let (deq, compute) = section.split_at(wait);
                if batch == 0 {
                    if q {
                        for (m, n) in [
                            ("v_bfe_u32", nw),
                            ("v_cvt_f32_u32_e32", nw),
                            ("v_fma_f32", nw),
                            ("v_cvt_f32_f16_e64", 2),
                            ("v_lshrrev_b32_e32", 1),
                        ] {
                            assert_eq!(count(deq, m), n, "{ctx} g{k}: dequant {m}");
                        }
                    } else {
                        for (m, n) in [
                            ("v_lshrrev_b64", 1),
                            ("v_bfe_u32", 4),
                            ("v_lshl_add_u32", 4),
                            ("v_cvt_f32_i32_e32", 4),
                            ("v_cvt_f32_f16_e64", 1),
                            ("v_mul_f32_e32", 4),
                        ] {
                            assert_eq!(count(deq, m), n, "{ctx} g{k}: dequant {m}");
                        }
                    }
                } else {
                    assert!(deq.iter().all(|l| !l.starts_with("v_")), "{ctx} g{k} b{batch}: no dequantization in later batches (weights are dequantized once)");
                }
                assert!(
                    compute.iter().all(|l| !is_weight_load(l)
                        && mnemonic(l) != "ds_store_b128"
                        && mnemonic(l) != "global_load_b128"),
                    "{ctx} g{k}: no weight/LDS-write inside a slot region"
                );
                // slot regions: split on the scalar slot gates
                let regions: Vec<&[&str]> = compute[1..]
                    .split(|l| mnemonic(l) == "s_cbranch_scc0")
                    .collect();
                assert_eq!(
                    regions.len(),
                    bsz,
                    "{ctx} g{k} b{batch}: one region per slot of the batch"
                );
                for (r, region) in regions.iter().enumerate() {
                    let slot = (batch * bsz) as usize + r;
                    let dst = partial_region(region, spec, &format!("{ctx} g{k} slot{slot}"));
                    assert_eq!(
                        dst,
                        acc[0] + slot as u32 * chains + chain,
                        "{ctx} g{k} slot{slot}: chain add into acc[slot][{chain}]"
                    );
                }
            }
        }
    }
}

/// The epilogue per live slot region: gate fold `(a0+a1)+(a2+a3)` (acc as src0), the five snapshot shuffle steps
/// (`ds_bpermute` of a0, `exch = a0 + exch`, `a0 = cndmask(a0, exch, mask_k)`), one lane-0 store of a0; the shuffle tables
/// once; and nothing else on the value path (no packed math, dot, fma_mix, mad, VOPD, exp).
#[test]
fn epilogue_census_matches_the_frozen_dag() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let g = gate(spec);
        let chains = spec.kind.chains() as u32;
        let first_load = body.iter().position(|l| is_weight_load(l)).unwrap();
        let acc0 = body[first_load..]
            .iter()
            .filter(|l| l.starts_with("v_mov_b32_e32 v") && l.ends_with(", 0"))
            .map(|l| vreg(&operand(l, 0)))
            .next()
            .unwrap();
        let epi = between(&body, spec, "epi", "red_done");
        assert_eq!(
            count(&epi, "v_cmp_gt_u32_e64"),
            5,
            "{ctx}: five shuffle masks"
        );
        let regions: Vec<&[&str]> = epi.split(|l| mnemonic(l) == "s_cbranch_scc0").collect();
        assert_eq!(regions.len(), 8, "{ctx}: one reduction region per slot");
        for (slot, region) in regions.iter().enumerate() {
            let a0 = acc0 + slot as u32 * chains;
            let adds: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "v_add_f32_e32")
                .collect();
            let fold = if g { 3 } else { 0 };
            assert_eq!(
                adds.len(),
                fold + 5,
                "{ctx} slot{slot}: fold + 5 shuffle adds"
            );
            if g {
                let (f1, f2, f3) = (adds[0], adds[1], adds[2]);
                assert_eq!(
                    (vreg(&operand(f1, 0)), operand(f1, 1), vreg(&operand(f1, 2))),
                    (a0, format!("v{a0}"), a0 + 1),
                    "{ctx} slot{slot}: a0 = a0 + a1"
                );
                assert_eq!(
                    (vreg(&operand(f2, 0)), operand(f2, 1), vreg(&operand(f2, 2))),
                    (a0 + 2, format!("v{}", a0 + 2), a0 + 3),
                    "{ctx} slot{slot}: a2 = a2 + a3"
                );
                assert_eq!(
                    (vreg(&operand(f3, 0)), operand(f3, 1), vreg(&operand(f3, 2))),
                    (a0, format!("v{a0}"), a0 + 2),
                    "{ctx} slot{slot}: a0 = a0 + a2"
                );
            }
            let perm: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "ds_bpermute_b32")
                .collect();
            let selects: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 0) == format!("v{a0}"))
                .collect();
            assert_eq!(
                (perm.len(), selects.len()),
                (5, 5),
                "{ctx} slot{slot}: five shuffle steps"
            );
            let exch = operand(perm[0], 0);
            for k in 0..5 {
                assert_eq!(
                    (operand(perm[k], 0), operand(perm[k], 2)),
                    (exch.clone(), format!("v{a0}")),
                    "{ctx} slot{slot}: bpermute {k} reads a0"
                );
                let step = adds[fold + k];
                assert_eq!(
                    (operand(step, 0), operand(step, 1), operand(step, 2)),
                    (exch.clone(), format!("v{a0}"), exch.clone()),
                    "{ctx} slot{slot}: exch = a0 + exch ({k})"
                );
                assert_eq!(
                    (
                        operand(selects[k], 0),
                        operand(selects[k], 1),
                        operand(selects[k], 2)
                    ),
                    (format!("v{a0}"), format!("v{a0}"), exch.clone()),
                    "{ctx} slot{slot}: a0 = cndmask(a0, exch, mask) ({k})"
                );
            }
            // the five masks are used in order 16, 8, 4, 2, 1 (the same SGPRs for every slot)
            let masks: Vec<String> = selects.iter().map(|l| operand(l, 3)).collect();
            let expect: Vec<String> = (72..77).map(|m| format!("s{m}")).collect();
            assert_eq!(masks, expect, "{ctx} slot{slot}: shuffle masks");
        }
        // the shuffle tables: offsets 16, 8, 4, 2, 1 (mask lane < 32 - off, address (lane + off) * 4)
        let tables: Vec<&&str> = epi
            .iter()
            .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0")
            .collect();
        assert_eq!(
            tables.iter().map(|l| operand(l, 2)).collect::<Vec<_>>(),
            ["16", "8", "4", "2", "1"],
            "{ctx}: shuffle offsets"
        );
        // the stores: lane 0 only, one per live slot region, the slot's final a0, into the selected Y pointer
        let stores = between(&body, spec, "red_done", "st_done");
        assert_eq!(count(&stores, "v_cmpx_eq_u32_e32"), 1, "{ctx}: lane 0 only");
        let store_lines: Vec<&&str> = stores
            .iter()
            .filter(|l| mnemonic(l) == "global_store_b32")
            .collect();
        assert_eq!(store_lines.len(), 8, "{ctx}: one store per slot region");
        for (slot, s) in store_lines.iter().enumerate() {
            assert_eq!(
                operand(s, 1),
                format!("v{}", acc0 + slot as u32 * chains),
                "{ctx} slot{slot}: stores a0 of its slot"
            );
            assert_eq!(operand(s, 2), "s[52:53]", "{ctx}: selected Y pointer");
        }
        assert!(
            !body.iter().any(|l| {
                let m = mnemonic(l);
                m.starts_with("v_pk_")
                    || m.starts_with("v_dot")
                    || m.starts_with("v_wmma")
                    || m.starts_with("v_swmmac")
                    || m.starts_with("v_fma_mix")
                    || m.starts_with("v_mad")
                    || m.starts_with("v_fmamk")
                    || m.starts_with("v_fmaak")
                    || m.starts_with("v_dual")
                    || m.starts_with("v_exp")
            }),
            "{ctx}: forbidden value-path instruction"
        );
    }
}

/// Unpack ties of one group of each codec: exact operand patterns of K1's dequantization.
#[test]
fn unpack_matches_k1() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        if mq4(spec) {
            let fma = body
                .iter()
                .find(|l| mnemonic(l) == "v_fma_f32" && operand(l, 0) == operand(l, 2))
                .unwrap_or_else(|| panic!("{ctx}: fma"));
            assert_eq!(
                operand(fma, 0),
                operand(fma, 2),
                "{ctx}: w = fma(d, q, z) in place"
            );
        } else {
            let at = body
                .iter()
                .position(|l| mnemonic(l) == "v_lshrrev_b64")
                .unwrap();
            let pkt = format!(
                "v{}",
                operand(body[at], 0)
                    .trim_start_matches("v[")
                    .split(':')
                    .next()
                    .unwrap()
            );
            for (i, bfe) in body[at + 1..]
                .iter()
                .filter(|l| mnemonic(l) == "v_bfe_u32")
                .take(4)
                .enumerate()
            {
                assert_eq!(
                    (operand(bfe, 1), operand(bfe, 2), operand(bfe, 3)),
                    (pkt.clone(), (3 * i).to_string(), "3".into()),
                    "{ctx}: W3 code {i}: bfe(pkt, 3i, 3)"
                );
            }
            let lsa = body
                .iter()
                .find(|l| mnemonic(l) == "v_lshl_add_u32")
                .unwrap();
            assert_eq!(
                (operand(lsa, 2), operand(lsa, 3)),
                ("1".into(), "-7".into()),
                "{ctx}: W3 grid: (c << 1) + -7"
            );
        }
    }
}

/// The shape contract derives deterministically from emission, covers all four entries, forbids scratch/flat/WMMA/mix/dot/pk
/// and the gfx12 wait family (but not LDS), pins the register ceilings to the plan and the fixed LDS.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let c = cn::contract(spec, &e);
        assert_eq!(c, cn::contract(spec, &cn::emit(spec).unwrap()));
        assert_eq!(c["symbol"].as_str(), Some(spec.symbol().as_str()));
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (
                Some(u64::from(e.shape.next_free_vgpr)),
                Some(u64::from(e.shape.next_free_sgpr))
            )
        );
        assert!(c["vgpr_max"].as_u64().unwrap() <= 128, "{}", spec.symbol());
        assert_eq!(
            c["group_segment_fixed_bytes"].as_u64(),
            Some(u64::from(spec.kind.lds_bytes()))
        );
        assert_eq!(c["launch_dynamic_lds_bytes"].as_u64(), Some(0));
        for flag in [
            "require_wave32",
            "require_zero_spills",
            "require_zero_private",
        ] {
            assert_eq!(c[flag], true, "{flag}");
        }
        let forbidden: Vec<&str> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_str().unwrap())
            .collect();
        for f in [
            "scratch_*",
            "flat_*",
            "v_wmma*",
            "v_swmmac*",
            "v_fma_mix*",
            "v_dot*",
            "v_pk_*",
            "s_wait_loadcnt*",
            "s_wait_alu*",
        ] {
            assert!(forbidden.contains(&f), "{f}");
        }
        assert!(
            !forbidden
                .iter()
                .any(|f| f.starts_with("ds_") || f.starts_with("s_barrier")),
            "this kernel stages X in LDS"
        );
        for (name, n) in c["counts"].as_object().unwrap() {
            let have = e.s_text.lines().filter(|l| mnemonic(l) == name).count() as u64;
            assert_eq!(n.as_u64(), Some(have), "{} counts {name}", spec.symbol());
        }
        assert_eq!(
            c["counts"]["s_barrier"].as_u64(),
            Some(if gate(spec) { 3 } else { 1 }),
            "{}",
            spec.symbol()
        );
    }
}

/// The shipped eight-entry production bundle is exactly the RegistryCli composition,
/// byte for byte (native writer, no ROCm), including all narrow and `_t8` contracts.
#[test]
fn committed_bundle_and_contracts_equal_fresh_emission() {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let path = format!("{root}/kernels/{MODULE}.hxaco");
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel mq_i4_moe_gemv_cn --arch gfx1151 --epi all ... --bundle)"));
    let (emitted, text, _) = product_module();
    assert_eq!(emitted.len(), PRODUCT_SYMBOLS.len());
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    let fresh = native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
    let entries = [false, true]
        .into_iter()
        .flat_map(|tiled| arch_specs().into_iter().map(move |spec| (tiled, spec)));
    for ((tiled, spec), e) in entries.zip(&emitted) {
        let tag = if tiled {
            format!("{}_t8", spec.tag())
        } else {
            spec.tag()
        };
        let file = format!(
            "{}/kernels/mq_i4_moe_gemv_cn.gfx1151.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            tag
        );
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            if tiled {
                cn::contract_rowtiles(spec, e)
            } else {
                cn::contract(spec, e)
            },
            "{file}: committed contract differs from the derived contract"
        );
    }
}

/// The module and every entry assemble with the native writer (no ROCm), and the bundle wraps the object.
#[test]
fn native_writer_assembles_every_entry_and_the_module() {
    let (emitted, text, _) = cn::emit_module().unwrap();
    for e in &emitted {
        let elf = native::assemble(&e.s_text, Arch::Gfx1151)
            .unwrap_or_else(|err| panic!("{}: {err}", e.proof.kernel_id));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    assert!(native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET).len() > elf.len());
}

/// The expert-id range check is the FIRST use of the loaded id (after the `n_experts` kernarg load) and is an UNSIGNED
/// `s_cmp_ge_u32` immediately followed by the uniform branch to the exit label; before it the kernel has issued no
/// barrier, no LDS access, no global access and no scalar load but the kernarg loads, the `offsets[u]` pair load and the
/// `expert_ids[u]` load itself; the id shift, the weight-pointer table load, the slot-id loads and every X / weight load
/// follow it; the exit is shared with the `wg_y`, `M`, `K` and slot-count guards.
#[test]
fn expert_id_guard_is_the_first_use_of_the_id_before_any_table_read_prefetch_or_barrier() {
    const EID: &str = "s34";
    const NEXP: &str = "s100";
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let (tag, symbol) = (spec.tag(), spec.symbol());
        let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
        let exit_branch = format!("s_cbranch_scc1 .Lcn_{tag}_exit");
        let pos = |pred: &dyn Fn(&str) -> bool| {
            lines
                .iter()
                .position(|l| pred(*l))
                .unwrap_or_else(|| panic!("{symbol}: line not found"))
        };
        let uses_eid = |l: &str| {
            l.split_whitespace()
                .skip(1)
                .any(|t| t.trim_end_matches(',') == EID)
        };
        let nexp_load = pos(&|l: &str| l == "s_load_b32 s100, s[0:1], 0x58");
        let eid_load = pos(&|l: &str| l.starts_with("s_load_b32 s34, "));
        let cmp = (eid_load + 1..lines.len())
            .find(|&i| uses_eid(lines[i]))
            .unwrap();
        assert_eq!(
            lines[cmp],
            format!("s_cmp_ge_u32 {EID}, {NEXP}"),
            "{symbol}: the first use of the id is the unsigned n_experts compare"
        );
        assert!(
            nexp_load < eid_load && eid_load < cmp,
            "{symbol}: n_experts load, id load, compare order"
        );
        assert_eq!(
            lines[cmp + 1],
            exit_branch,
            "{symbol}: uniform exit right after the compare"
        );
        assert!(
            !lines[..cmp]
                .iter()
                .any(|l| uses_eid(l) && !l.starts_with("s_load_b32 s34, ")),
            "{symbol}: id used before the compare"
        );
        // Nothing before the compare touches memory but the kernarg loads, the offsets pair and the expert_ids[u] load.
        for l in &lines[..cmp] {
            let m = mnemonic(l);
            assert!(
                !(m == "s_barrier"
                    || m.starts_with("global_")
                    || m.starts_with("ds_load")
                    || m.starts_with("ds_store")
                    || m.starts_with("flat_")
                    || m.starts_with("scratch_")),
                "{symbol}: `{l}` before the id guard"
            );
            if m.starts_with("s_load_") {
                let base = l.split(", ").nth(1).unwrap_or_default();
                assert!(
                    base == "s[0:1]" || base == "s[36:37]",
                    "{symbol}: `{l}` is neither a kernarg load nor the offsets / expert_ids load"
                );
            }
        }
        // The id shift, the weight-pointer table load, the slot-id loads and every X / weight load follow the guard.
        let shift = pos(&|l: &str| l == "s_lshl_b32 s41, s34, 3");
        assert!(cmp < shift, "{symbol}: id shift after the guard");
        let table_loads: Vec<usize> = (0..lines.len())
            .filter(|&i| lines[i].starts_with("s_load_b64 s[38:39], "))
            .collect();
        assert_eq!(
            table_loads.len(),
            1,
            "{symbol}: one weight-pointer table load"
        );
        assert!(
            cmp < table_loads[0],
            "{symbol}: weight-pointer table load after the guard"
        );
        for i in (0..lines.len())
            .filter(|&i| (57..=65).any(|r| lines[i].starts_with(&format!("s_load_b32 s{r}, "))))
        {
            assert!(cmp < i, "{symbol}: slot-id load before the guard")
        }
        for i in (0..lines.len()).filter(|&i| mnemonic(lines[i]).starts_with("global_load")) {
            assert!(cmp < i, "{symbol}: X / weight load before the guard")
        }
        // Shared exit: wg_y, M, K, slot-count (two), offsets and expert-id guards all branch to it before the body.
        let body = pos(&|l: &str| l == format!(".Lcn_{tag}_body:"));
        assert_eq!(
            lines[..body].iter().filter(|l| **l == exit_branch).count(),
            7,
            "{symbol}: wg_y, M, K, n == 0, n > 8, offsets, expert-id exits"
        );
        assert!(cmp < body, "{symbol}: guard in the prologue");
        assert!(
            lines[..body].contains(&"s_cmp_ge_u32 s3, s24"),
            "{symbol}: wg_y guard unchanged"
        );
    }
}

// ------------------------------------------------------------------------------------------------ CPU interpreter
// The real emitted code objects, lifted and run on the peacemaker CPU interpreter: invalid expert ids exit with no store
// and no dereference, valid ids are untouched by the guard and equal, bit for bit, the incumbent K1 narrow entry (the
// kernel Cn replaces, `mq_i4_moe_gemv.rs`, itself guarded) on the same descriptors. Tensor data are random full-mantissa as
// in K1's wide-emulator test; only experts 0 and `E_N - 1` are mapped, and the weight-pointer table is exactly `E_N`
// entries, so a table read at or past `E_N` (or an expert dereference through a null table slot) faults.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_moe_gemv as k1;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;
const E_KERNARG: u64 = 0x1_0000;
const E_KERNARG_REF: u64 = 0x2_0000;
const E_TABLE: u64 = 0x10_0000;
const E_IDS: u64 = 0x18_0000;
const E_OFFS: u64 = 0x20_0000;
const E_SLOTS: u64 = 0x28_0000;
const E_EXPERTS: u64 = 0x1000_0000;
const E_STRIDE: u64 = 0x40_0000;
const E_X: u64 = 0x8000_0000;
const E_Y0: u64 = 0x9000_0000;
const E_Y1: u64 = 0xa000_0000;
const E_R0: u64 = 0xb000_0000;
const E_R1: u64 = 0xc000_0000;
/// The extent of the weight-pointer table (`n_experts`) of every case.
const E_N: u32 = 8;
/// Two decode lanes (`C`): twenty original slots; the one expert serves the original slots below (its `n = 3`).
const LANES: u32 = 2;
const SLOT_IDS: [u32; 3] = [3, 7, 12];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Normal f16 scale with exponent field 8..=20: `-8 * sc` is then exactly the symmetric MQ4 zero.
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

/// Planar W3 tensor: per row `K*3/8` code bytes (128 3-bit codes per 48-byte group), then the f16 scale plane.
fn w3_tensor(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut out = vec![0u8; m * row_bytes + 2 * m * groups];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            for j in 0..128 {
                let code = rng.below(8) as u32;
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        out[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
            }
            let off = m * row_bytes + 2 * (r * groups + g);
            out[off..off + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
        }
    }
    out
}

/// MQ4: gate G256 groups of 136 B (`[sc0 zp0 sc1 zp1]`, 128 payload bytes at +8), down G128 groups of 68 B (`[sc zp]`, 64 at +4).
fn mq4_tensor(rng: &mut Rng, kind: Kind, m: usize, k: usize) -> Vec<u8> {
    let (cols, stride, halves, payload_at, payload) = match kind {
        Kind::GateUp => (256, 136, 2, 8, 128),
        Kind::Down => (128, 68, 1, 4, 64),
    };
    let groups = k / cols;
    let mut out = vec![0u8; m * groups * stride];
    for r in 0..m {
        for g in 0..groups {
            let base = (r * groups + g) * stride;
            for h in 0..halves {
                let sc = scale_bits(rng);
                out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                out[base + 4 * h + 2..base + 4 * h + 4]
                    .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
            }
            for byte in &mut out[base + payload_at..base + payload_at + payload] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

/// Full-mantissa activations over a wide exponent band, random signs.
fn x_rows(rng: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k * 4);
    for _ in 0..rows * k {
        let (s, e, m) = (
            (rng.next() & 1) as u32,
            118 + rng.below(19) as u32,
            (rng.next() & 0x7f_ffff) as u32,
        );
        out.extend(((s << 31) | (e << 23) | m).to_le_bytes());
    }
    out
}

struct Rig {
    cn: peacemaker_ir::Program,
    k1: peacemaker_ir::Program,
}
fn lift_program(text: &str) -> peacemaker_ir::Program {
    let elf = native::assemble(text, Arch::Gfx1151).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}
impl Rig {
    fn new() -> Self {
        let (_, cn_text, _) = cn::emit_module().unwrap();
        let (_, k1_text, _) = k1::emit_module(Arch::Gfx1151).unwrap();
        Rig {
            cn: lift_program(&cn_text),
            k1: lift_program(&k1_text),
        }
    }
}

fn rowtile_rig() -> Rig {
    let (_, text, _) = product_module();
    let (_, k1_text, _) = k1::emit_module(Arch::Gfx1151).unwrap();
    Rig {
        cn: lift_program(&text),
        k1: lift_program(&k1_text),
    }
}

/// One entry's deterministic data: the X rows and one expert tensor of the entry's codec.
struct Data {
    spec: Spec,
    m: usize,
    k: usize,
    xdiv: u32,
    xsrc: u32,
    ys: usize,
    x: Vec<u8>,
    tensor: Vec<u8>,
}
fn data(spec: Spec) -> Data {
    // gate: `C` X rows (slot / 10), Y rows `M/2` per slot; down: `10*C` X rows (slot / 1), Y rows `M` per slot.
    let (m, xdiv, xrows, ys) = match spec.kind {
        Kind::GateUp => (1280usize, 10u32, LANES as usize, 640usize),
        Kind::Down => (2560, 1, 10 * LANES as usize, 2560),
    };
    let k = spec.kind.k() as usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ k as u64 ^ ((spec.codec as u64) << 40));
    let tensor = match spec.codec {
        Codec::Mq3I4 => w3_tensor(&mut rng, m, k),
        Codec::Mq4Sym => mq4_tensor(&mut rng, spec.kind, m, k),
    };
    let x = x_rows(&mut rng, xrows, k);
    Data {
        spec,
        m,
        k,
        xdiv,
        xsrc: xrows as u32,
        ys,
        x,
        tensor,
    }
}

/// The Y words and the number of Y bytes the launches wrote.
struct Out {
    y: Vec<u8>,
    written: usize,
}

/// Everything but the kernarg blocks: experts `0` and `E_N - 1` mapped (the same tensor), the exactly-`E_N`-entry weight table,
/// the one-expert descriptors (`expert_ids = [id]`, `offsets = [0, 3]`, `slot_ids = SLOT_IDS`), X, and the Cn (`y0`/`y1`)
/// and K1 reference (`r0`/`r1`) outputs, all sentinel-filled with guard words on both sides.
fn memory(d: &Data, id: i32) -> Result<Memory, String> {
    let mut mem = Memory::default();
    let mut table = vec![0u8; E_N as usize * 8];
    for expert in [0u32, E_N - 1] {
        let base = E_EXPERTS + u64::from(expert) * E_STRIDE;
        assert!((d.tensor.len() as u64) < E_STRIDE);
        table[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
        mem.map(format!("e{expert}"), base, d.tensor.clone())?;
    }
    mem.map("table".into(), E_TABLE, table)?;
    mem.map("ids".into(), E_IDS, id.to_le_bytes().to_vec())?;
    mem.map(
        "offs".into(),
        E_OFFS,
        [0u32, SLOT_IDS.len() as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    )?;
    mem.map(
        "slots".into(),
        E_SLOTS,
        SLOT_IDS.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    mem.map("x".into(), E_X, d.x.clone())?;
    let ybytes = 10 * LANES as usize * d.ys * 4 + 2 * GUARD;
    let gate = d.spec.kind == Kind::GateUp;
    for (name, base, live) in [
        ("y0", E_Y0, true),
        ("y1", E_Y1, gate),
        ("r0", E_R0, true),
        ("r1", E_R1, gate),
    ] {
        if live {
            mem.map(name.into(), base, vec![SENTINEL; ybytes])?;
        }
    }
    Ok(mem)
}

/// The 96-byte K1 / Cn kernarg block (identical ABI): `n_experts@88`, zero `reserved@92`.
fn kernarg(d: &Data, y0: u64, y1: u64, n_experts: u32) -> Vec<u8> {
    let mut karg = Vec::with_capacity(cn::KERNARG_BYTES as usize);
    for ptr in [
        E_TABLE,
        E_IDS,
        E_OFFS,
        E_SLOTS,
        E_X,
        y0 + GUARD as u64,
        if d.spec.kind == Kind::GateUp {
            y1 + GUARD as u64
        } else {
            0
        },
    ] {
        karg.extend(ptr.to_le_bytes());
    }
    // M, K, U, C, x_row_div, x_src_rows, x_row_stride, y_slot_stride, n_experts, reserved.
    for v in [
        d.m as u32,
        d.k as u32,
        1,
        LANES,
        d.xdiv,
        d.xsrc,
        d.k as u32,
        d.ys as u32,
        n_experts,
        0,
    ] {
        karg.extend(v.to_le_bytes());
    }
    assert_eq!(karg.len(), cn::KERNARG_BYTES as usize);
    karg
}

fn collect(mem: &Memory, names: [&str; 2]) -> Out {
    let (mut y, mut written) = (Vec::new(), 0);
    for r in mem
        .regions
        .iter()
        .filter(|r| r.name == names[0] || r.name == names[1])
    {
        y.extend_from_slice(&r.bytes);
        written += r.written.count();
    }
    Out { y, written }
}

/// Cn: workgroups `(0, 0)` and `(M/8 - 1, 0)` (first and last row block; for the gate the second lands in `Y1`) of the one
/// expert `expert_ids[0] = id`, `n_experts` as given.
fn run_cn(rig: &Rig, d: &Data, id: i32, n_experts: u32) -> Result<Out, String> {
    let mut mem = memory(d, id)?;
    mem.map(
        "kernarg".into(),
        E_KERNARG,
        kernarg(d, E_Y0, E_Y1, n_experts),
    )?;
    let emu = Emulator::new(&rig.cn, &d.spec.symbol())?;
    for wg in [[0u32, 0, 0], [d.m as u32 / 8 - 1, 0, 0]] {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG,
                workgroup: wg,
                block: [256, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    Ok(collect(&mem, ["y0", "y1"]))
}

/// The incumbent K1 narrow entry of the same codec and kind on the same descriptors (valid ids only), writing `r0`/`r1`: one
/// wave32 workgroup per output row, the sixteen rows Cn's two workgroups cover.
fn run_k1(rig: &Rig, d: &Data, id: i32) -> Result<Out, String> {
    let mut mem = memory(d, id)?;
    mem.map("kernarg".into(), E_KERNARG_REF, kernarg(d, E_R0, E_R1, E_N))?;
    let codec = match d.spec.codec {
        Codec::Mq3I4 => k1::Codec::Mq3I4,
        Codec::Mq4Sym => k1::Codec::Mq4Sym,
    };
    let kind = match d.spec.kind {
        Kind::GateUp => k1::Kind::GateUp,
        Kind::Down => k1::Kind::Down,
    };
    let emu = Emulator::new(
        &rig.k1,
        &k1::Spec::narrow(Arch::Gfx1151, codec, kind).symbol(),
    )?;
    for row in (0..8).chain(d.m as u32 - 8..d.m as u32) {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG_REF,
                workgroup: [row, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    Ok(collect(&mem, ["r0", "r1"]))
}

type TileGroups = Vec<Vec<(u32, Vec<u32>)>>;
#[derive(Clone)]
struct TileCase {
    spec: Spec,
    rows: u32,
    m: usize,
    k: usize,
    x_stride: u32,
    y_stride: u32,
    x_src_rows: u32,
    n_experts: u32,
    x: std::sync::Arc<[u8]>,
    tensors: Vec<std::sync::Arc<[u8]>>,
    descriptor: Vec<u8>,
    groups: TileGroups,
    bucket: usize,
}
fn tile_case(spec: Spec, rows: u32, m: usize, x_pad: u32, y_pad: usize) -> TileCase {
    let k = spec.kind.k() as usize;
    let x_stride = spec.kind.k() + x_pad;
    let gate = spec.kind == Kind::GateUp;
    let y_stride = (if gate { m / 2 } else { m }) as u32 + y_pad as u32;
    let x_src_rows = if gate { rows } else { 10 * rows };
    let mut rng =
        Rng(0x9e37_79b9_7f4a_7c15 ^ k as u64 ^ ((spec.codec as u64) << 40) ^ ((m as u64) << 24));
    let n_tiles = rows.div_ceil(8) as usize;
    let mut tensors = Vec::new();
    for _ in 0..2usize {
        tensors.push(
            match spec.codec {
                Codec::Mq3I4 => w3_tensor(&mut rng, m, k),
                Codec::Mq4Sym => mq4_tensor(&mut rng, spec.kind, m, k),
            }
            .into(),
        );
    }
    let mut x = vec![0u8; x_src_rows as usize * x_stride as usize * 4];
    let xr = x_rows(&mut rng, x_src_rows as usize, k);
    for r in 0..x_src_rows as usize {
        x[r * x_stride as usize * 4..r * x_stride as usize * 4 + k * 4]
            .copy_from_slice(&xr[r * k * 4..(r + 1) * k * 4]);
        for b in x[r * x_stride as usize * 4 + k * 4..(r + 1) * x_stride as usize * 4].iter_mut() {
            *b = SENTINEL;
        }
    }
    let mut groups: TileGroups = Vec::new();
    for tile in 0..n_tiles {
        let mut tile_groups: Vec<(u32, Vec<u32>)> = Vec::new();
        let start = tile * 8;
        for token in 0..8u32 {
            let r = start as u32 + token;
            if r >= rows {
                break;
            }
            let rank = if r % 2 == 0 { 0u32 } else { 9u32 };
            let slot = 10 * r + rank;
            let expert = if tile % 2 == 0 { 0u32 } else { E_N - 1 };
            if let Some((_, slots)) = tile_groups.iter_mut().find(|(e, _)| *e == expert) {
                slots.push(slot);
            } else {
                tile_groups.push((expert, vec![slot]));
            }
        }
        groups.push(tile_groups);
    }
    let bucket = usize::from(spec.codec == Codec::Mq4Sym);
    let descriptor = tile_descriptors(bucket, &groups);
    TileCase {
        spec,
        rows,
        m,
        k,
        x_stride,
        y_stride,
        x_src_rows,
        n_experts: E_N,
        x: x.into(),
        tensors,
        descriptor,
        groups,
        bucket,
    }
}
fn tile_descriptors(bucket: usize, groups: &TileGroups) -> Vec<u8> {
    let mut words = vec![0u32; 482 * groups.len().max(1)];
    for (tile, tile_groups) in groups.iter().enumerate() {
        let base = tile * 482;
        let off = base + 160 + bucket * 81;
        let count: usize = tile_groups.iter().map(|(_, s)| s.len()).sum();
        for o in 0..81usize {
            words[off + o] = count as u32;
        }
        let mut position = 0usize;
        for (gi, (expert, slots)) in tile_groups.iter().enumerate() {
            words[base + bucket * 80 + gi] = *expert;
            words[off + gi] = position as u32;
            for (j, slot) in slots.iter().enumerate() {
                words[base + 322 + bucket * 80 + position + j] = *slot;
            }
            position += slots.len();
        }
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn encode_kernarg(ptrs: [u64; 7], values: [u32; 10]) -> Vec<u8> {
    let mut out = Vec::with_capacity(96);
    for p in ptrs {
        out.extend(p.to_le_bytes());
    }
    for v in values {
        out.extend(v.to_le_bytes());
    }
    assert_eq!(out.len(), 96);
    out
}
fn tile_kernarg(case: &TileCase) -> Vec<u8> {
    let ids = E_IDS + GUARD as u64 + case.bucket as u64 * 80 * 4;
    let offs = E_IDS + GUARD as u64 + (160 + 81 * case.bucket as u64) * 4;
    let slots = E_IDS + GUARD as u64 + (322 + 80 * case.bucket as u64) * 4;
    let gate = case.spec.kind == Kind::GateUp;
    encode_kernarg(
        [
            E_TABLE,
            ids,
            offs,
            slots,
            E_X,
            E_Y0 + GUARD as u64,
            if gate { E_Y1 + GUARD as u64 } else { 0 },
        ],
        [
            case.m as u32,
            case.k as u32,
            80,
            case.rows,
            if gate { 10 } else { 1 },
            case.x_src_rows,
            case.x_stride,
            case.y_stride,
            case.n_experts,
            0,
        ],
    )
}
fn tile_memory(case: &TileCase) -> Result<Memory, String> {
    let mut mem = Memory::default();
    let mut table = vec![0u8; case.n_experts as usize * 8];
    for (tensor, expert) in case.tensors.iter().zip([0, case.n_experts - 1]) {
        let base = E_EXPERTS + u64::from(expert) * E_STRIDE;
        assert!((tensor.len() as u64) < E_STRIDE);
        table[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
        mem.map(format!("e{expert}"), base, tensor.to_vec())?;
    }
    mem.map("table".into(), E_TABLE, table)?;
    let mut descriptor = vec![SENTINEL; GUARD];
    descriptor.extend_from_slice(&case.descriptor);
    descriptor.extend_from_slice(&[SENTINEL; GUARD]);
    mem.map("desc".into(), E_IDS, descriptor)?;
    if case.x_src_rows != 0 {
        let extent = case.x_src_rows as usize * case.x_stride as usize * 4;
        mem.map("x".into(), E_X, case.x[..extent].to_vec())?;
    }
    let gate = case.spec.kind == Kind::GateUp;
    let total = 10 * case.rows as usize * case.y_stride as usize * 4 + 2 * GUARD;
    mem.map("y0".into(), E_Y0, vec![SENTINEL; total])?;
    if gate {
        mem.map("y1".into(), E_Y1, vec![SENTINEL; total])?;
    }
    Ok(mem)
}
fn run_t8(rig: &Rig, case: &TileCase, workgroups: &[[u32; 3]]) -> Result<Memory, String> {
    let mut mem = tile_memory(case)?;
    mem.map("kernarg".into(), E_KERNARG, tile_kernarg(case))?;
    let before = mem.clone();
    let emu = Emulator::new(&rig.cn, &format!("{}_t8", case.spec.symbol()))?;
    for wg in workgroups {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG,
                workgroup: *wg,
                block: [256, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    assert_immutable(&before, &mem);
    Ok(mem)
}
fn run_k1_singleton(
    rig: &Rig,
    case: &TileCase,
    token: u32,
    rank: u32,
    expert: u32,
    output_rows: &[u32],
) -> Result<Vec<[u8; 4]>, String> {
    assert!(token < case.rows && rank < 10);
    let gate = case.spec.kind == Kind::GateUp;
    let xdiv = if gate { 10u32 } else { 1u32 };
    let mut mem = Memory::default();
    let mut table = vec![0u8; case.n_experts as usize * 8];
    let base = E_EXPERTS + u64::from(expert) * E_STRIDE;
    table[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
    mem.map("table".into(), E_TABLE, table)?;
    let tensor = &case.tensors[usize::from(expert != 0)];
    mem.map(format!("e{expert}"), base, tensor.to_vec())?;
    mem.map("ids".into(), E_IDS, (expert).to_le_bytes().to_vec())?;
    mem.map(
        "offs".into(),
        E_OFFS,
        [0u32, 1u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    )?;
    mem.map("slots".into(), E_SLOTS, rank.to_le_bytes().to_vec())?;
    let k = case.k as u32;
    let xrows_full = if gate { 1 } else { 10 };
    let ys = if gate { case.m / 2 } else { case.m } as u32;
    let ybytes = 10 * ys as usize * 4 + 2 * GUARD;
    mem.map("r0".into(), E_R0, vec![SENTINEL; ybytes])?;
    if gate {
        mem.map("r1".into(), E_R1, vec![SENTINEL; ybytes])?;
    }
    let first = if gate { token } else { 10 * token };
    let local_xrows = case.x_src_rows.saturating_sub(first).min(xrows_full);
    if local_xrows != 0 {
        let mut x = Vec::with_capacity(local_xrows as usize * case.k * 4);
        for row in first..first + local_xrows {
            let start = row as usize * case.x_stride as usize * 4;
            x.extend_from_slice(&case.x[start..start + case.k * 4]);
        }
        mem.map("x".into(), E_X, x)?;
    }
    let codec = match case.spec.codec {
        Codec::Mq3I4 => k1::Codec::Mq3I4,
        Codec::Mq4Sym => k1::Codec::Mq4Sym,
    };
    let kind = match case.spec.kind {
        Kind::GateUp => k1::Kind::GateUp,
        Kind::Down => k1::Kind::Down,
    };
    let karg = encode_kernarg(
        [
            E_TABLE,
            E_IDS,
            E_OFFS,
            E_SLOTS,
            E_X,
            E_R0 + GUARD as u64,
            if gate { E_R1 + GUARD as u64 } else { 0 },
        ],
        [
            case.m as u32,
            k,
            1,
            1,
            xdiv,
            local_xrows,
            k,
            ys,
            case.n_experts,
            0,
        ],
    );
    mem.map("kernarg".into(), E_KERNARG_REF, karg)?;
    let before = mem.clone();
    let emu = Emulator::new(
        &rig.k1,
        &k1::Spec::narrow(Arch::Gfx1151, codec, kind).symbol(),
    )?;
    for row in output_rows {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG_REF,
                workgroup: [*row, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    assert_immutable(&before, &mem);
    let r0 = mem
        .read(E_R0 + GUARD as u64, 10 * ys as usize * 4)?
        .to_vec();
    let r1 = if gate {
        mem.read(E_R1 + GUARD as u64, 10 * ys as usize * 4)?
            .to_vec()
    } else {
        Vec::new()
    };
    let at = |slot: u32, row: u32| -> [u8; 4] {
        if gate {
            let mi = case.m as u32 / 2;
            let (buf, rr) = if row < mi {
                (&r0, row)
            } else {
                (&r1, row - mi)
            };
            let o = (slot as usize * ys as usize + rr as usize) * 4;
            [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]
        } else {
            let o = (slot as usize * ys as usize + row as usize) * 4;
            [r0[o], r0[o + 1], r0[o + 2], r0[o + 3]]
        }
    };
    let words: Vec<_> = output_rows.iter().map(|row| at(rank, *row)).collect();
    for (which, name) in [(0, "r0"), (1, "r1")] {
        if which == 1 && !gate {
            continue;
        }
        let mut expected = vec![SENTINEL; ybytes];
        let mut written = vec![false; ybytes];
        for (&row, word) in output_rows.iter().zip(&words) {
            let is_y1 = gate && row >= case.m as u32 / 2;
            if usize::from(is_y1) != which {
                continue;
            }
            let local = if is_y1 { row - case.m as u32 / 2 } else { row };
            let offset = GUARD + 4 * (rank as usize * ys as usize + local as usize);
            expected[offset..offset + 4].copy_from_slice(word);
            written[offset..offset + 4].fill(true);
        }
        assert_output_map(&mem, name, &expected, &written);
    }
    Ok(words)
}
fn assert_immutable(before: &Memory, after: &Memory) {
    for b in &before.regions {
        let a = after
            .regions
            .iter()
            .find(|r| r.name == b.name)
            .unwrap_or_else(|| panic!("region {} missing after", b.name));
        let input = !matches!(b.name.as_str(), "y0" | "y1" | "r0" | "r1");
        if input {
            assert!(a.bytes == b.bytes, "input region {} changed", b.name);
            for i in 0..b.bytes.len() {
                assert!(
                    !a.written.contains(i),
                    "input region {} byte {i} written",
                    b.name
                );
            }
        }
    }
}
fn assert_output_map(mem: &Memory, region: &str, expected: &[u8], expected_written: &[bool]) {
    let r = mem
        .regions
        .iter()
        .find(|r| r.name == region)
        .unwrap_or_else(|| panic!("region {region} missing"));
    assert_eq!(r.bytes.len(), expected.len(), "{region}: length");
    assert_eq!(r.bytes, expected, "{region}: bytes");
    assert_eq!(
        r.written.count(),
        expected_written.iter().filter(|b| **b).count(),
        "{region}: written count"
    );
    for (i, w) in expected_written.iter().enumerate() {
        assert_eq!(r.written.contains(i), *w, "{region}: byte {i}");
    }
}

/// Scatter independent C1 words; all other bytes and write bits stay untouched.
fn assert_tile_reference(
    rig: &Rig,
    tile: &TileCase,
    mem: &Memory,
    wgs: &[[u32; 3]],
    require_nonzero: bool,
) {
    let gate = tile.spec.kind == Kind::GateUp;
    let len = 10 * tile.rows as usize * tile.y_stride as usize * 4 + 2 * GUARD;
    let mut expected = [
        vec![SENTINEL; len],
        vec![SENTINEL; if gate { len } else { 0 }],
    ];
    let mut written = [vec![false; len], vec![false; if gate { len } else { 0 }]];
    let mut varied = std::collections::HashSet::new();
    let mut live = 0;
    for &[x, u, z] in wgs {
        let (expert, slots) = &tile.groups[z as usize][u as usize];
        let rows: Vec<u32> = (8 * x..(8 * x + 8).min(tile.m as u32)).collect();
        for &slot in slots {
            let words = run_k1_singleton(rig, tile, slot / 10, slot % 10, *expert, &rows)
                .unwrap_or_else(|e| panic!("{} slot {slot}: K1: {e}", tile.spec.tag()));
            for (&row, word) in rows.iter().zip(words) {
                let value = f32::from_le_bytes(word);
                assert!(
                    value.is_finite(),
                    "{} slot {slot} row {row}",
                    tile.spec.tag()
                );
                varied.insert(value.to_bits());
                let (which, local_row) = if gate && row >= tile.m as u32 / 2 {
                    (1, row - tile.m as u32 / 2)
                } else {
                    (0, row)
                };
                let at = GUARD + 4 * (slot as usize * tile.y_stride as usize + local_row as usize);
                expected[which][at..at + 4].copy_from_slice(&word);
                assert!(written[which][at..at + 4].iter().all(|v| !v));
                written[which][at..at + 4].fill(true);
                live += 1;
            }
        }
    }
    assert!(live > 0);
    if require_nonzero {
        assert!(
            varied.iter().any(|v| f32::from_bits(*v) != 0.0),
            "zero-only K1"
        );
        assert!(varied.len() > 1, "constant K1");
    }
    assert_output_map(mem, "y0", &expected[0], &written[0]);
    if gate {
        assert_output_map(mem, "y1", &expected[1], &written[1]);
    }
}

/// Invalid ids `-1`, `i32::MIN` (whose `<< 3` is 0: unguarded it would alias table entry 0 and run the expert through it),
/// `n` and `n + 1` (one past a table with exactly `n` entries, whose weight-pointer read would fault) exit with no store; the
/// valid id `n - 1` produces exactly the bytes the permissive `n_experts = u32::MAX` run produces AND the bytes of the
/// incumbent K1 narrow entry (an independent reference), and writes the 48 lane-0 words of the two workgroups (8 waves x 3
/// slots each). All four contracts (both codecs x gate / down).
#[test]
fn out_of_range_expert_ids_exit_with_no_store_or_dereference_and_valid_ids_are_unchanged() {
    let rig = Rig::new();
    let n = E_N as i32;
    for spec in arch_specs() {
        let d = data(spec);
        let case = spec.tag();
        for id in [-1, i32::MIN, n, n + 1] {
            let out = run_cn(&rig, &d, id, E_N).unwrap_or_else(|e| panic!("{case}: id {id}: interpreter fault (an expert or table read escaped the guard): {e}"));
            assert_eq!(out.written, 0, "{case}: id {id} stored");
            assert!(
                out.y.iter().all(|&b| b == SENTINEL),
                "{case}: id {id} changed Y"
            );
        }
        let valid = run_cn(&rig, &d, n - 1, E_N)
            .unwrap_or_else(|e| panic!("{case}: valid id {}: {e}", n - 1));
        assert_eq!(
            valid.written,
            2 * 8 * SLOT_IDS.len() * 4,
            "{case}: valid id writes one lane-0 word per wave and live slot of both workgroups"
        );
        for w in valid
            .y
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .filter(|&w| w != 0xa5a5_a5a5)
        {
            assert!(
                f32::from_bits(w).is_finite(),
                "{case}: non-finite output word {w:#x}"
            );
        }
        let open = run_cn(&rig, &d, n - 1, u32::MAX).unwrap();
        assert!(
            valid.y == open.y && valid.written == open.written,
            "{case}: the guard changed a valid launch"
        );
        let reference =
            run_k1(&rig, &d, n - 1).unwrap_or_else(|e| panic!("{case}: K1 reference: {e}"));
        assert_eq!(
            reference.written, valid.written,
            "{case}: K1 reference written bytes"
        );
        assert!(
            valid.y == reference.y,
            "{case}: Cn differs from the incumbent K1 on the same valid launch"
        );
    }
}
#[test]
fn rowtiles_all_rows_match_independent_k1_singletons() {
    let rig = rowtile_rig();
    // Reuse identical maximum-width inputs and independent references across R.
    let mut fixtures: Vec<_> = arch_specs()
        .into_iter()
        .map(|spec| {
            let m = if spec.kind == Kind::GateUp {
                1280
            } else {
                2560
            };
            (
                tile_case(spec, 64, m, 0, 0),
                std::collections::HashMap::<(u32, u32, u32, u32), [u8; 4]>::new(),
            )
        })
        .collect();
    for rows in [9u32].into_iter().chain((1..=64).filter(|r| *r != 9)) {
        // All four R9 cases precede expansion to every remaining R exactly once.
        for (maximum, cache) in &mut fixtures {
            let mut tile = maximum.clone();
            let spec = tile.spec;
            let case = spec.tag();
            let m = tile.m;
            let gate = spec.kind == Kind::GateUp;
            let ys = tile.y_stride;
            let xgs = [0, (m as u32).div_ceil(8) - 1];
            tile.rows = rows;
            tile.x_src_rows = if gate { rows } else { 10 * rows };
            tile.groups.truncate(rows.div_ceil(8) as usize);
            tile.groups.last_mut().unwrap()[0]
                .1
                .retain(|slot| *slot < 10 * rows);
            tile.descriptor = tile_descriptors(tile.bucket, &tile.groups);
            let n_tiles = rows.div_ceil(8);
            let mut wgs: Vec<[u32; 3]> = Vec::new();
            for xg in xgs {
                for z in (0..n_tiles).rev() {
                    wgs.push([xg, 0, z]);
                }
            }
            let mem = run_t8(&rig, &tile, &wgs)
                .unwrap_or_else(|e| panic!("{case}: R{rows}: t8 fault: {e}"));
            let mut exp0 = vec![SENTINEL; 10 * rows as usize * ys as usize * 4 + 2 * GUARD];
            let mut exp1 = if gate {
                Some(vec![
                    SENTINEL;
                    10 * rows as usize * ys as usize * 4 + 2 * GUARD
                ])
            } else {
                None
            };
            let mut w0 = vec![false; exp0.len()];
            let mut w1 = if gate {
                Some(vec![false; exp0.len()])
            } else {
                None
            };
            let mut live = 0usize;
            let mut ref_nonzero = 0usize;
            let mut ref_varied = std::collections::HashSet::new();
            for token in 0..rows {
                let rank = if token % 2 == 0 { 0u32 } else { 9u32 };
                let slot = 10 * token + rank;
                let expert = if (token / 8) % 2 == 0 { 0u32 } else { E_N - 1 };
                for xg in xgs {
                    let out_rows: Vec<u32> = (0..8u32)
                        .map(|w| xg * 8 + w)
                        .filter(|r| (*r as usize) < m)
                        .collect();
                    if !cache.contains_key(&(token, rank, expert, out_rows[0])) {
                        let words = run_k1_singleton(&rig, &tile, token, rank, expert, &out_rows)
                            .unwrap_or_else(|e| panic!("{case}: R{rows} token {token}: K1: {e}"));
                        for (&row, word) in out_rows.iter().zip(words) {
                            cache.insert((token, rank, expert, row), word);
                        }
                    }
                    let refs: Vec<_> = out_rows
                        .iter()
                        .map(|row| cache[&(token, rank, expert, *row)])
                        .collect();
                    for (row, word) in out_rows.iter().zip(refs.iter()) {
                        let w = u32::from_le_bytes(*word);
                        assert!(
                            f32::from_bits(w).is_finite(),
                            "{case}: R{rows} token {token} row {row}: non-finite {w:#x}"
                        );
                        if w != 0 {
                            ref_nonzero += 1;
                        }
                        ref_varied.insert(w);
                        let o = (slot as usize * ys as usize
                            + if gate {
                                let mi = m as u32 / 2;
                                if *row < mi {
                                    *row
                                } else {
                                    *row - mi
                                }
                            } else {
                                *row
                            } as usize)
                            * 4;
                        if gate {
                            let mi = m as u32 / 2;
                            if *row < mi {
                                exp0[GUARD + o..GUARD + o + 4].copy_from_slice(word);
                                for b in 0..4 {
                                    w0[GUARD + o + b] = true;
                                }
                            } else if let (Some(e1), Some(ww1)) = (exp1.as_mut(), w1.as_mut()) {
                                e1[GUARD + o..GUARD + o + 4].copy_from_slice(word);
                                for b in 0..4 {
                                    ww1[GUARD + o + b] = true;
                                }
                            }
                        } else {
                            exp0[GUARD + o..GUARD + o + 4].copy_from_slice(word);
                            for b in 0..4 {
                                w0[GUARD + o + b] = true;
                            }
                        }
                        live += 1;
                    }
                }
            }
            assert!(live > 0, "{case}: R{rows}: no live words");
            assert!(ref_nonzero > 0, "{case}: R{rows}: zero-only reference");
            assert!(ref_varied.len() > 1, "{case}: R{rows}: constant reference");
            assert_output_map(&mem, "y0", &exp0, &w0);
            if let (Some(e1), Some(ww1)) = (exp1.as_ref(), w1.as_ref()) {
                assert_output_map(&mem, "y1", e1, ww1);
            }
        }
    }
}
#[test]
fn rowtiles_output_tails_stage_all_slots_without_loading_tail_weights() {
    let rig = rowtile_rig();
    for spec in arch_specs() {
        let gate = spec.kind == Kind::GateUp;
        let m = if gate { 1282 } else { 2563 };
        for (x_pad, y_pad) in [(0, 0), (4, 5)] {
            let tile = tile_case(spec, 8, m, x_pad, y_pad);
            let mut blocks = vec![(m as u32).div_ceil(8) - 1];
            if gate {
                blocks.push(m as u32 / 2 / 8);
            }
            for x in blocks {
                let wgs = [[x, 0, 0]];
                let mem = run_t8(&rig, &tile, &wgs).unwrap_or_else(|e| {
                    panic!("{} tail block{x}, padding{x_pad}/{y_pad}: {e}", spec.tag())
                });
                assert_tile_reference(&rig, &tile, &mem, &wgs, true);
            }
        }
        if gate {
            for zero_phase in 0..2 {
                let mut tile = tile_case(spec, 8, m, 0, 0);
                // Actual stage_x_phase uses col = phase*1280 + lane*4 + j*128.
                let half = tile.k / 2;
                let x = std::sync::Arc::make_mut(&mut tile.x);
                for row in 0..tile.x_src_rows as usize {
                    let start = (row * tile.x_stride as usize + zero_phase * half) * 4;
                    x[start..start + half * 4].fill(0);
                }
                let wgs = [[(m as u32).div_ceil(8) - 1, 0, 0]];
                let mem = run_t8(&rig, &tile, &wgs)
                    .unwrap_or_else(|e| panic!("{} zero phase{zero_phase}: {e}", spec.tag()));
                assert_tile_reference(&rig, &tile, &mem, &wgs, true);
            }
        }
    }
}

#[test]
fn rowtiles_invalid_descriptors_exit_without_downstream_dereference() {
    let rig = rowtile_rig();
    for spec in arch_specs() {
        let gate = spec.kind == Kind::GateUp;
        let m = if gate { 1280 } else { 2560 };
        let base_tile = tile_case(spec, 64, m, 0, 0);
        // Only map the particular fields that must precede the targeted exit.
        // In particular, offsets/IDs never make slots or the table readable.
        let run_sparse = |karg: Vec<u8>, wg: [u32; 3], desc: Option<(i32, u32, u32, Vec<u32>)>| {
            let mut mem = Memory::default();
            let len = 640 * base_tile.y_stride as usize * 4 + 2 * GUARD;
            mem.map("y0".into(), E_Y0, vec![SENTINEL; len]).unwrap();
            if gate {
                mem.map("y1".into(), E_Y1, vec![SENTINEL; len]).unwrap();
            }
            if let Some((id, start, end, slots)) = desc {
                let z = u64::from(wg[2]) * 1928;
                let pointer = |at| u64::from_le_bytes(karg[at..at + 8].try_into().unwrap()) + z;
                mem.map("ids".into(), pointer(8), id.to_le_bytes().to_vec())
                    .unwrap();
                mem.map(
                    "offsets".into(),
                    pointer(16),
                    [start, end]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                )
                .unwrap();
                if !slots.is_empty() {
                    mem.map(
                        "slots".into(),
                        pointer(24) + u64::from(start) * 4,
                        slots.into_iter().flat_map(u32::to_le_bytes).collect(),
                    )
                    .unwrap();
                }
            }
            mem.map("kernarg".into(), E_KERNARG, karg).unwrap();
            let before = mem.clone();
            let emu = Emulator::new(&rig.cn, &format!("{}_t8", spec.symbol())).unwrap();
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: E_KERNARG,
                    workgroup: wg,
                    block: [256, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{} sparse {wg:?}: {e}", spec.tag()));
            assert_immutable(&before, &mem);
            let poison = vec![SENTINEL; len];
            let unwritten = vec![false; len];
            assert_output_map(&mem, "y0", &poison, &unwritten);
            if gate {
                assert_output_map(&mem, "y1", &poison, &unwritten);
            }
        };
        for (offset, value) in [
            (68, 0),
            (68, 65),
            (64, 79),
            (64, 81),
            (60, base_tile.k as u32 + 128),
        ] {
            let mut karg = tile_kernarg(&base_tile);
            karg[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            run_sparse(karg, [0, 0, 0], None);
        }
        for wg in [[(m as u32).div_ceil(8), 0, 0], [0, 80, 0], [0, 0, 8]] {
            run_sparse(tile_kernarg(&base_tile), wg, None);
        }
        for z in [0, 7] {
            run_sparse(tile_kernarg(&base_tile), [0, 0, z], Some((0, 0, 0, vec![])));
        }
        for (start, end) in [(0, 9), (8, 7), (81, 82), (79, 81)] {
            run_sparse(
                tile_kernarg(&base_tile),
                [0, 0, 0],
                Some((0, start, end, vec![])),
            );
        }
        {
            let mut karg = tile_kernarg(&base_tile);
            karg[68..72].copy_from_slice(&1u32.to_le_bytes());
            run_sparse(karg, [0, 0, 0], Some((0, 8, 11, vec![])));
        }
        for id in [-1, i32::MIN, E_N as i32, E_N as i32 + 1] {
            run_sparse(
                tile_kernarg(&base_tile),
                [0, 0, 0],
                Some((id, 0, 1, vec![])),
            );
        }
        {
            let mut karg = tile_kernarg(&base_tile);
            karg[88..92].copy_from_slice(&0u32.to_le_bytes());
            run_sparse(karg, [0, 0, 0], Some((0, 0, 1, vec![])));
        }
        for position in 0..8 {
            for bad in [640, u32::MAX] {
                let mut slots: Vec<u32> = (0..8).collect();
                slots[position] = bad;
                run_sparse(tile_kernarg(&base_tile), [0, 0, 0], Some((0, 0, 8, slots)));
            }
        }
        // Source clipping zero-fills contributions, but still writes live Y.
        // The singleton reference uses the same bounds with token-local X.
        for source_rows in [if gate { 7 } else { 79 }, 0] {
            let mut tile = tile_case(spec, 8, m, 0, 0);
            tile.x_src_rows = source_rows;
            let wgs = [[0, 0, 0], [(m as u32).div_ceil(8) - 1, 0, 0]];
            let mem = run_t8(&rig, &tile, &wgs)
                .unwrap_or_else(|e| panic!("{} X rows{source_rows}: {e}", spec.tag()));
            assert_tile_reference(&rig, &tile, &mem, &wgs, source_rows != 0);
            assert_eq!(mem.regions.iter().any(|r| r.name == "x"), source_rows != 0);
        }
        // Ten experts, eight slots each: the last list is exactly [72,80).
        // Only the two launched experts have readable table entries/tensors.
        let mut distinct = base_tile.clone();
        distinct.n_experts = 10;
        distinct.groups[7] = (0..10u32)
            .map(|expert| (expert, (56..64).map(|token| 10 * token + expert).collect()))
            .collect();
        distinct.descriptor = tile_descriptors(distinct.bucket, &distinct.groups);
        let last = (m as u32).div_ceil(8) - 1;
        let wgs = [[0, 0, 7], [0, 9, 7], [last, 0, 7], [last, 9, 7]];
        let mem = run_t8(&rig, &distinct, &wgs)
            .unwrap_or_else(|e| panic!("{} terminal offset80: {e}", spec.tag()));
        assert_tile_reference(&rig, &distinct, &mem, &wgs, true);
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check, toolchain::IsaShapeContract};

    /// Every symbol's waits replay independently of the builder's ledger (entry and module text), the contract
    /// parses as an `IsaShapeContract`, and M7 lifts the natively linked object byte-exactly with no obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = product_module();
        let entries = [false, true]
            .into_iter()
            .flat_map(|tiled| arch_specs().into_iter().map(move |spec| (tiled, spec)));
        for ((tiled, spec), e) in entries.zip(&emitted) {
            ledger_replay::replay_waits(&e.s_text, Arch::Gfx1151)
                .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
            assert!(e.s_text.contains(".kernarg_segment_size: 96\n"));
            serde_json::from_value::<IsaShapeContract>(if tiled {
                cn::contract_rowtiles(spec, e)
            } else {
                cn::contract(spec, e)
            })
            .unwrap();
        }
        ledger_replay::replay_waits(&text, Arch::Gfx1151).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4cn-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, Arch::Gfx1151).unwrap()).unwrap();
        for symbol in PRODUCT_SYMBOLS {
            let m7 =
                pm_check::m7(&co, "gfx1151", symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
