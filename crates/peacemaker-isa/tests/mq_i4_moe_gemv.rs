// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Common K1 diagnostics retain four narrow C8 entries and eight C16/C32 twins per architecture:
//! frozen arithmetic, instruction/resource census, 96-byte ABI, native assembly, replay and M7.
//! These diagnostic modules are not the production export list. The gfx1201 product image contains
//! four narrow entries followed by four fixed-eight-row `_t8` entries. The committed common gfx1151
//! image remains a twelve-entry diagnostic/reference image; Halo production tiles live in the Cn
//! image, covered by `mq_i4_moe_gemv_cn.rs`. No product path requires C16/C32 exports.
use peacemaker_isa::kernels::mq_i4_moe_gemv::{self, Cap, Codec, Kind, Spec};
use peacemaker_isa::Arch;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];
fn module(arch: Arch) -> String {
    format!("mq_i4_moe_gemv_pm_{}", arch.name())
}
/// Every narrow (C8) entry of every supported architecture, module order within an architecture.
fn all_specs() -> Vec<Spec> {
    ARCHES
        .into_iter()
        .flat_map(mq_i4_moe_gemv::narrow_specs)
        .collect()
}
/// The eight diagnostic wide twins (C16 then C32) of every supported architecture.
fn wide_specs() -> Vec<Spec> {
    ARCHES
        .into_iter()
        .flat_map(mq_i4_moe_gemv::module_specs)
        .filter(|s| s.cap != Cap::C8)
        .collect()
}
/// All twelve entries of every architecture's diagnostic module.
fn module_entries() -> Vec<Spec> {
    ARCHES
        .into_iter()
        .flat_map(mq_i4_moe_gemv::module_specs)
        .collect()
}

/// Match the CLI's gfx1201 product composition, independently of diagnostic `emit_module`.
fn gfx1201_product_image() -> (Vec<peacemaker_isa::Emitted>, String) {
    let specs = mq_i4_moe_gemv::narrow_specs(Arch::Gfx1201);
    let mut emitted: Vec<_> = specs
        .iter()
        .map(|&s| mq_i4_moe_gemv::emit(s).unwrap())
        .collect();
    emitted.extend(
        specs
            .into_iter()
            .map(|s| mq_i4_moe_gemv::emit_rowtiles(s).unwrap()),
    );
    let (text, _) =
        peacemaker_isa::kernels::iu4_gemm::module(&emitted, &module(Arch::Gfx1201)).unwrap();
    (emitted, text)
}
const SYMBOLS: [&str; 4] = [
    "mq_i4_moe_gemv_mq3i4_r256_gate_up",
    "mq_i4_moe_gemv_mq3i4_r128_down",
    "mq_i4_moe_gemv_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_mq4sym_r128_down",
];
const WIDE_SYMBOLS: [&str; 8] = [
    "mq_i4_moe_gemv_mq3i4_r256_gate_up_c16",
    "mq_i4_moe_gemv_mq3i4_r128_down_c16",
    "mq_i4_moe_gemv_mq4sym_r256_gate_up_c16",
    "mq_i4_moe_gemv_mq4sym_r128_down_c16",
    "mq_i4_moe_gemv_mq3i4_r256_gate_up_c32",
    "mq_i4_moe_gemv_mq3i4_r128_down_c32",
    "mq_i4_moe_gemv_mq4sym_r256_gate_up_c32",
    "mq_i4_moe_gemv_mq4sym_r128_down_c32",
];
/// sha256 of the narrow entries' emitted assembly text with the `n_experts` expert-id guard (`(arch, tag, s_text sha256)`).
const NARROW_S_TEXT_SHA256: [(&str, &str, &str); 8] = [
    (
        "gfx1151",
        "mq3i4_r256_gate_up",
        "ce700f0af513b3b4e31a92ee99bddb1f39f98ea0ff4bffb176b5b12048db536d",
    ),
    (
        "gfx1151",
        "mq3i4_r128_down",
        "c2a324faebd4080266d29782452c8e59df0bfe9a150c3824c89b356fe72833f3",
    ),
    (
        "gfx1151",
        "mq4sym_r256_gate_up",
        "5a07b4785de3c0a6b1d51545535cb2c8edbeedcb8c032f0fa264949312649309",
    ),
    (
        "gfx1151",
        "mq4sym_r128_down",
        "eada4939d88454898a6fed6e3018dc401677cd316355248500d843497b715c1d",
    ),
    (
        "gfx1201",
        "mq3i4_r256_gate_up",
        "462b0c44874780c951e1eb1266b46c1209eddca03ae4ad946f35c73f0dd37dfc",
    ),
    (
        "gfx1201",
        "mq3i4_r128_down",
        "e59607054c54fc46095457dc5735e316991db98a16e005d596a5a3cd15a42724",
    ),
    (
        "gfx1201",
        "mq4sym_r256_gate_up",
        "6d768d276e0a3083b265ae337309f0eb3e77b36d4cf5a87cbfafe2c008cd8edc",
    ),
    (
        "gfx1201",
        "mq4sym_r128_down",
        "f91bde250e722d8cd1068726b1a67d2276cabf52cb20564457c2aef3455a2e39",
    ),
];
/// Frozen ABI (DESIGN §4.2): `(offset, size, name)` in kernarg order.
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
const SLOTS: usize = 8;

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}
fn spec(arch: Arch, codec: Codec, kind: Kind) -> Spec {
    Spec::narrow(arch, codec, kind)
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
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

#[test]
fn diagnostic_module_exports_are_the_twelve_frozen_symbols_in_order() {
    for arch in ARCHES {
        let specs = mq_i4_moe_gemv::module_specs(arch);
        assert_eq!(
            &specs[..4],
            [
                spec(arch, Codec::Mq3I4, Kind::GateUp),
                spec(arch, Codec::Mq3I4, Kind::Down),
                spec(arch, Codec::Mq4Sym, Kind::GateUp),
                spec(arch, Codec::Mq4Sym, Kind::Down),
            ]
        );
        assert_eq!(mq_i4_moe_gemv::narrow_specs(arch), specs[..4]);
        assert_eq!(specs.len(), SYMBOLS.len() + WIDE_SYMBOLS.len());
        assert_eq!(
            specs[..4].iter().map(Spec::symbol).collect::<Vec<_>>(),
            SYMBOLS
        );
        assert_eq!(
            specs[4..].iter().map(Spec::symbol).collect::<Vec<_>>(),
            WIDE_SYMBOLS
        );
        assert!(
            specs[4..8].iter().all(|s| s.cap == Cap::C16)
                && specs[8..].iter().all(|s| s.cap == Cap::C32)
        );
        assert_eq!(Spec::module(arch), module(arch));
        let (emitted, text, proof) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        let all: Vec<&str> = SYMBOLS.iter().chain(&WIDE_SYMBOLS).copied().collect();
        assert_eq!(emitted.len(), all.len());
        assert_eq!(
            (proof.module.as_str(), proof.arch),
            (module(arch).as_str(), arch)
        );
        assert_eq!(proof.kernels.len(), all.len());
        assert_eq!(text.matches(".amdhsa_kernel ").count(), all.len());
        for symbol in all {
            assert!(
                text.contains(&format!(".amdhsa_kernel {symbol}\n")),
                "{arch:?} {symbol}: missing from module source"
            );
            assert!(
                text.contains(&format!(".globl {symbol}\n")),
                "{arch:?} {symbol}: not global"
            );
            assert!(
                text.contains(&format!(".name: {symbol}\n")),
                "{arch:?} {symbol}: missing from module metadata"
            );
        }
    }
}

#[test]
fn gfx1201_product_exports_are_narrow_then_fixed_eight_row_tiles() {
    let (emitted, text) = gfx1201_product_image();
    let expected: Vec<_> = SYMBOLS
        .iter()
        .map(|s| (*s).to_owned())
        .chain(SYMBOLS.iter().map(|s| format!("{s}_t8")))
        .collect();
    assert_eq!(emitted.len(), expected.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), expected.len());
    let actual: Vec<_> = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .collect();
    assert_eq!(actual, expected);
    for symbol in &expected {
        assert!(text.contains(&format!(".globl {symbol}\n")), "{symbol}");
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}");
    }
    for symbol in WIDE_SYMBOLS {
        assert!(
            !text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
    }
}

/// Tile entries keep the frozen ABI but declare the architecture-local fixed-eight schedule.
#[test]
fn rowtiles_keep_the_96_byte_abi_and_bounded_resources() {
    for spec in all_specs() {
        let e = mq_i4_moe_gemv::emit_rowtiles(spec).unwrap();
        let c = mq_i4_moe_gemv::contract_rowtiles(spec, &e);
        let symbol = if spec.arch == Arch::Gfx1151 {
            format!(
                "mq_i4_moe_gemv_cn_{}_{}_t8",
                spec.codec.tag(),
                spec.kind.tag()
            )
        } else {
            format!("{}_t8", spec.symbol())
        };
        assert_eq!(c["symbol"], symbol);
        assert!(e.s_text.contains(&format!(".amdhsa_kernel {symbol}\n")));
        assert!(e.s_text.contains(".amdhsa_kernarg_size 96\n"));
        assert_eq!(field(&e.s_text, ".kernarg_segment_size:"), [96]);
        assert_eq!(field(&e.s_text, ".offset:"), ARGS.map(|a| a.0));
        assert_eq!(field(&e.s_text, ".size:"), ARGS.map(|a| a.1));
        let names: Vec<_> = e
            .s_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix(".name: "))
            .take(ARGS.len())
            .collect();
        assert_eq!(names, ARGS.map(|a| a.2));
        assert_eq!(field(&e.s_text, ".max_flat_workgroup_size:"), [256]);
        assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"));
        assert!(e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_z 1"));
        for key in [
            ".private_segment_fixed_size:",
            ".sgpr_spill_count:",
            ".vgpr_spill_count:",
        ] {
            assert_eq!(field(&e.s_text, key), [0], "{symbol}: {key}");
        }
        let lds = if spec.arch == Arch::Gfx1151 {
            if spec.kind == Kind::GateUp {
                40_960
            } else {
                20_480
            }
        } else {
            8 * 32 * 4 * u32::from(spec.lane_elems())
        };
        assert_eq!(field(&e.s_text, ".group_segment_fixed_size:"), [lds]);
        assert_eq!(c["group_segment_fixed_bytes"], lds);
        assert_eq!(c["launch_dynamic_lds_bytes"], 0);
        assert_eq!(c["require_wave32"], true);
        assert_eq!(c["require_zero_spills"], true);
        assert_eq!(c["require_zero_private"], true);
        assert_eq!(c["vgpr_max"], e.shape.next_free_vgpr);
        assert_eq!(c["sgpr_max"], e.shape.next_free_sgpr);
        assert_eq!(
            field(&e.s_text, ".vgpr_count:"),
            [u32::from(e.shape.next_free_vgpr)]
        );
        assert_eq!(
            field(&e.s_text, ".amdhsa_next_free_vgpr "),
            [u32::from(e.shape.next_free_vgpr)]
        );
        // emit::kernel_assembly includes reserved VCC in metadata, not in
        // the logical SGPR high-water mark or .amdhsa_next_free_sgpr.
        let sgpr_count = field(&e.s_text, ".sgpr_count:");
        let reserve_vcc = field(&e.s_text, ".amdhsa_reserve_vcc ");
        assert!(matches!(reserve_vcc.as_slice(), [0] | [1]));
        let vcc_sgprs = 2 * reserve_vcc[0];
        assert_eq!(
            field(&e.s_text, ".amdhsa_next_free_sgpr "),
            [u32::from(e.shape.next_free_sgpr)]
        );
        assert_eq!(sgpr_count, [u32::from(e.shape.next_free_sgpr) + vcc_sgprs]);
        // next_free_vgpr is the logical high-water mark; native::descriptor::encode
        // rounds it up to eight-VGPR blocks for GRANULATED_WORKITEM_VGPR_COUNT.
        assert!(e.shape.next_free_vgpr > 0 && e.shape.next_free_vgpr <= 256);
        assert!(e.shape.next_free_sgpr > 0 && e.shape.next_free_sgpr <= 106);
        assert!(
            !e.s_text.lines().any(|l| {
                let m = mnemonic(l);
                m.starts_with("scratch_") || m.starts_with("flat_")
            }),
            "{symbol}: scratch/flat"
        );
        // The obsolete diagnostic cap must not enlarge the tile's physical slot bank.
        for cap in Cap::ALL {
            let alternate = Spec { cap, ..spec };
            let same = mq_i4_moe_gemv::emit_rowtiles(alternate).unwrap();
            assert_eq!(e.s_text, same.s_text, "{symbol}: cap {cap:?}");
            assert_eq!(c, mq_i4_moe_gemv::contract_rowtiles(alternate, &same));
        }
    }
    for spec in mq_i4_moe_gemv::narrow_specs(Arch::Gfx1100) {
        assert!(mq_i4_moe_gemv::emit_rowtiles(spec).is_err());
    }
}

/// The wide twins are additive: every narrow entry's emitted assembly (and so its machine code) is byte for byte the
/// text pinned in `NARROW_S_TEXT_SHA256` (the 96-byte, `n_experts`-guarded narrow emission; the wide twins add no byte to it).
#[test]
fn narrow_entries_are_unchanged_by_the_wide_twins() {
    for (arch_name, tag, sha) in NARROW_S_TEXT_SHA256 {
        let arch: Arch = arch_name.parse().unwrap();
        let spec = mq_i4_moe_gemv::module_specs(arch)
            .into_iter()
            .find(|s| s.tag() == tag)
            .unwrap();
        assert_eq!(spec.cap, Cap::C8);
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        assert_eq!(
            e.proof.s_text_sha256, sha,
            "{arch_name} {tag}: the narrow emission changed"
        );
    }
}

/// Emission is bit-deterministic per architecture, and every other architecture (gfx1100) is refused
/// (the builder is exact-gfx1151 / gfx1201: no implicit alias).
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in module_entries() {
        let (a, b) = (
            mq_i4_moe_gemv::emit(spec).unwrap(),
            mq_i4_moe_gemv::emit(spec).unwrap(),
        );
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
    }
    for arch in ARCHES {
        let (_, ta, pa) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        let (_, tb, pb) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256), "{arch:?}");
    }
    for arch in [Arch::Gfx1100] {
        for spec in mq_i4_moe_gemv::module_specs(arch) {
            assert!(mq_i4_moe_gemv::emit(spec).is_err(), "{arch:?} {spec:?}");
        }
        assert!(mq_i4_moe_gemv::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// The architecture-specific prologue and waits: gfx1201 reads the workgroup ids from `ttmp9` / `ttmp7[15:0]` and
/// waits with the split counters, gfx1151 reads them from the first two system SGPRs (`s2`, `s3`, no `ttmp`) and
/// waits with `s_waitcnt`; the arithmetic, shuffle and store spellings are shared.
#[test]
fn workgroup_ids_and_waits_follow_the_architecture() {
    for spec in all_specs() {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = format!("{:?} {}", spec.arch, spec.symbol());
        if spec.arch.gfx12() {
            assert!(
                t.contains("s_mov_b32 s30, ttmp9") && t.contains("s_and_b32 s31, ttmp7, 0xffff"),
                "{symbol}: gfx12 workgroup ids"
            );
            assert!(
                !t.lines()
                    .any(|l| mnemonic(l) == "s_waitcnt" || mnemonic(l) == "s_waitcnt_depctr"),
                "{symbol}: gfx12 has no s_waitcnt"
            );
        } else {
            assert!(
                t.contains("s_mov_b32 s30, s2") && t.contains("s_mov_b32 s31, s3"),
                "{symbol}: gfx11 workgroup ids"
            );
            assert!(
                !t.contains("ttmp"),
                "{symbol}: gfx11 never names a trap temporary"
            );
            assert!(
                t.lines().any(|l| mnemonic(l) == "s_waitcnt"),
                "{symbol}: gfx11 waits with s_waitcnt"
            );
            assert!(
                !t.lines().any(|l| mnemonic(l).starts_with("s_wait_loadcnt")
                    || mnemonic(l).starts_with("s_wait_dscnt")
                    || mnemonic(l).starts_with("s_wait_kmcnt")
                    || mnemonic(l).starts_with("s_barrier_")),
                "{symbol}: gfx12-only waits"
            );
            assert!(
                t.contains("s_sub_i32 s35, s33, s32") && !t.contains("s_sub_co_i32"),
                "{symbol}: gfx11 scalar subtract"
            );
        }
    }
}

/// Every narrow diagnostic entry is wave32, one wave per workgroup, with no scratch, spills, LDS or barriers.
#[test]
fn every_narrow_entry_is_wave32_without_scratch_or_lds() {
    for arch in ARCHES {
        let (emitted, text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        for (e, symbol) in emitted.iter().zip(SYMBOLS) {
            assert_eq!(e.proof.arch, arch, "{symbol}");
            assert!(
                e.s_text.contains(".amdhsa_wavefront_size32 1"),
                "{symbol}: wave32"
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
            assert!(
                e.s_text.contains(".group_segment_fixed_size: 0\n"),
                "{symbol}: no LDS"
            );
            assert!(
                e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
                "{symbol}: grid y is the expert"
            );
            assert_eq!(e.shape.barriers, 0, "{symbol}: barriers");
            let ds = e.s_text.lines().find(|l| {
                mnemonic(l).starts_with("ds_load")
                    || mnemonic(l).starts_with("ds_store")
                    || mnemonic(l).starts_with("ds_swizzle")
            });
            assert!(ds.is_none(), "{symbol}: LDS access {ds:?}");
            assert!(
                !e.s_text.lines().any(
                    |l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")
                ),
                "{symbol}: scratch/flat"
            );
        }
        assert!(text.contains(&format!("amdgcn-amd-amdhsa--{}", arch.name())));
    }
}

/// The 96-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the
/// first loads, and the typed ownership of the registers that carry it.
#[test]
fn kernarg_abi_matches_the_frozen_96_byte_layout() {
    assert_eq!(mq_i4_moe_gemv::KERNARG_BYTES, 96);
    assert_eq!(mq_i4_moe_gemv::WORKGROUP_SIZE, 32);
    assert_eq!(mq_i4_moe_gemv::MAX_SLOTS as usize, SLOTS);
    for spec in all_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
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
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [32],
            "{symbol}: block 32"
        );
        let (last_off, last_size, _) = *ARGS.last().unwrap();
        assert_eq!(
            last_off + last_size,
            mq_i4_moe_gemv::KERNARG_BYTES,
            "{symbol}: arguments cover the kernarg block"
        );
        // Tables and X are read-only, Y0/Y1 write-only (LLVM key order: the first key carries the list dash).
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
        // The kernarg is loaded as five scalar loads covering offsets 0..92 with no over-read (the zero SMEM offset is `0x0` on gfx12, `null` on gfx11); `reserved@92` is never read.
        let zero = if spec.arch.gfx12() { "0x0" } else { "null" };
        for load in [
            format!("s_load_b256 s[8:15], s[0:1], {zero}"),
            "s_load_b256 s[16:23], s[0:1], 0x20".into(),
            "s_load_b128 s[24:27], s[0:1], 0x40".into(),
            "s_load_b64 s[28:29], s[0:1], 0x50".into(),
            "s_load_b32 s80, s[0:1], 0x58".into(),
        ] {
            assert!(e.s_text.contains(&load), "{symbol}: kernarg load {load}");
        }
        // Typed ownership: the register plan declares the C8 slot state and every access is inside a declared range.
        let plan: Vec<&str> = e.proof.reg_plan.iter().map(|r| r.name.as_ref()).collect();
        for owner in [
            "kernel_arguments_ptrs",
            "kernel_arguments_x_y_m_k",
            "kernel_arguments_u_c_div_src",
            "kernel_arguments_strides",
            "x_descriptor",
            "y_select",
            "accumulators",
            "x_values",
            "weights",
            "slot_x_offsets",
            "slot_y_addresses",
            "shuffle_masks",
            "shuffle_addresses",
        ] {
            assert!(
                plan.contains(&owner),
                "{symbol}: reg plan misses {owner}: {plan:?}"
            );
        }
    }
}

/// Honest register ceilings of the C8 state: eight slots of accumulators (four chains per slot for gate)
/// and X registers, whole-granule VGPR reservation, SGPR bookkeeping, the contract equals the plan.
#[test]
fn register_ceilings_reflect_c8_slot_state() {
    for spec in all_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let symbol = spec.symbol();
        let (nw, chains) = (spec.lane_elems() as u16, spec.kind.chains() as u16);
        let floor = SLOTS as u16 * (chains + nw) + 2 * SLOTS as u16;
        assert!(
            e.shape.next_free_vgpr >= floor
                && e.shape.next_free_vgpr % 8 == 0
                && e.shape.next_free_vgpr <= 256,
            "{symbol}: vgpr {}",
            e.shape.next_free_vgpr
        );
        assert_eq!(e.shape.next_free_sgpr, 81, "{symbol}: sgpr");
        assert!(
            e.s_text
                .contains(&format!(".vgpr_count: {}\n", e.shape.next_free_vgpr)),
            "{symbol}: metadata vgpr count"
        );
        let c = mq_i4_moe_gemv::contract(spec, &e);
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (
                Some(u64::from(e.shape.next_free_vgpr)),
                Some(u64::from(e.shape.next_free_sgpr))
            ),
            "{symbol}"
        );
    }
}

/// The frozen numerical tree per symbol: weight/dot/accumulate chain per group instance, the gate
/// four-chain fold vs the down single chain, five snapshot shuffle steps per slot with out-of-range lanes
/// unchanged (sum into scratch, then a mask select), eight lane-0 stores, `Mq4Sym` affine fma vs `Mq3I4`
/// signed `2c-7` grid (never an XOR rebias).
#[test]
fn instruction_census_matches_the_frozen_dag() {
    for spec in all_specs() {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let nw = spec.lane_elems() as usize;
        let planar = spec.codec == Codec::Mq3I4;
        let gate = spec.kind == Kind::GateUp;
        // One slot-id load in the prologue; weight-class loads per group instance by codec/kind.
        let (u8s, u16s, b32s) = match (spec.codec, spec.kind) {
            (Codec::Mq3I4, _) => (2 * inst, inst, 1),
            (Codec::Mq4Sym, Kind::GateUp) => (0, 0, 2 * inst + 1),
            (Codec::Mq4Sym, Kind::Down) => (0, inst, inst + 1),
        };
        assert_eq!(
            count(&t, "global_load_u8"),
            u8s,
            "{symbol}: packet byte loads"
        );
        assert_eq!(count(&t, "global_load_u16"), u16s, "{symbol}: u16 loads");
        assert_eq!(count(&t, "global_load_b32"), b32s, "{symbol}: dword loads");
        assert_eq!(
            count(&t, "buffer_load_b128"),
            inst * SLOTS * nw / 4,
            "{symbol}: X loads"
        );
        assert_eq!(count(&t, "global_store_b32"), SLOTS, "{symbol}: stores");
        assert_eq!(count(&t, "v_bfe_u32"), nw * inst, "{symbol}: code extracts");
        assert_eq!(
            count(&t, "v_cvt_f32_i32_e32"),
            if planar { 4 * inst } else { 0 },
            "{symbol}: signed grid converts"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_u32_e32"),
            2 + if planar { 0 } else { nw * inst },
            "{symbol}: unsigned converts (2 are the slot divide)"
        );
        assert_eq!(
            count(&t, "v_fma_f32"),
            if planar { 0 } else { nw * inst },
            "{symbol}: affine fma"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_f16_e64"),
            inst * if planar { 1 } else { 2 },
            "{symbol}: header converts"
        );
        assert_eq!(
            count(&t, "v_lshl_or_b32"),
            if planar { inst } else { 0 },
            "{symbol}: W3 packet combine"
        );
        // First dot term per slot per group (plus 4 weight multiplies per group for Mq3I4, plus the divide's one multiply).
        assert_eq!(
            count(&t, "v_mul_f32_e32"),
            SLOTS * inst + if planar { 4 * inst } else { 0 } + 1,
            "{symbol}: multiplies"
        );
        assert_eq!(
            count(&t, "v_fmac_f32_e32"),
            (nw - 1) * SLOTS * inst,
            "{symbol}: fused steps"
        );
        let fold = if gate { 3 } else { 0 };
        assert_eq!(
            count(&t, "v_add_f32_e32"),
            SLOTS * inst + SLOTS * (fold + 5),
            "{symbol}: group-partial, fold and shuffle adds"
        );
        assert_eq!(
            count(&t, "ds_bpermute_b32"),
            5 * SLOTS,
            "{symbol}: shuffle steps"
        );
        assert_eq!(
            count(&t, "v_readlane_b32"),
            2 * SLOTS,
            "{symbol}: slot broadcasts"
        );
        assert_eq!(
            count(&t, "v_cmpx_eq_u32_e32"),
            1,
            "{symbol}: lane-0 store mask"
        );
        for forbidden in [
            "ds_swizzle",
            "ds_load",
            "ds_store",
            "v_fma_mix_f32",
            "v_xor_b32",
            "flat_",
            "scratch_",
            "v_wmma",
            "global_atomic",
            "buffer_atomic",
        ] {
            let hit = t.lines().find(|l| mnemonic(l).starts_with(forbidden));
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
        if planar {
            assert!(
                t.lines().any(|l| l.trim().starts_with("v_add_nc_u32_e32 v")
                    && l.trim().ends_with(", -7, v24")),
                "{symbol}: 2c-7 grid"
            );
        }
        // Left-associated chain: p = RN(w0*x0), then fma(w_j, x_j, p) j = 1..nw-1 with adjacent registers.
        // (the builder's `s_delay_alu` issue hints and waits sit between dependent instructions and are not part of the tree)
        let lines: Vec<&str> = t
            .lines()
            .map(str::trim)
            .filter(|l| {
                !mnemonic(l).starts_with("s_delay_alu")
                    && !mnemonic(l).starts_with("s_wait")
                    && !mnemonic(l).starts_with("s_nop")
            })
            .collect();
        let first_fmac = lines
            .iter()
            .position(|l| mnemonic(l) == "v_fmac_f32_e32")
            .unwrap();
        let mul = lines[first_fmac - 1];
        assert_eq!(mnemonic(mul), "v_mul_f32_e32", "{symbol}: first dot term");
        // Mq3I4: p = RN(w0*x0), fma(w1,x1,p), fma(w2,x2,p)...; Mq4Sym (production contraction of the old MQ4 kernels):
        // p = RN(w1*x1), fma(w0,x0,p), then fma(w_j,x_j,p) for j = 2.. ascending. Adjacent registers per element.
        let order: Vec<u32> = if planar {
            (0..nw as u32).collect()
        } else {
            [1, 0].into_iter().chain(2..nw as u32).collect()
        };
        let (dot, wf, xf) = (
            vreg(&operand(mul, 0)),
            vreg(&operand(mul, 1)),
            vreg(&operand(mul, 2)),
        );
        let (w0, x0) = (wf - order[0], xf - order[0]);
        for (k, &j) in order.iter().enumerate().skip(1) {
            let l = lines[first_fmac + k - 1];
            assert_eq!(mnemonic(l), "v_fmac_f32_e32", "{symbol}: fused step {k}");
            assert_eq!(
                (
                    vreg(&operand(l, 0)),
                    vreg(&operand(l, 1)),
                    vreg(&operand(l, 2))
                ),
                (dot, w0 + j, x0 + j),
                "{symbol}: fused step {k} (element {j}) operands"
            );
        }
        // The partial is added separately: `acc = acc + p`.
        let add = lines[first_fmac + nw - 1];
        assert_eq!(
            (
                mnemonic(add),
                vreg(&operand(add, 0)),
                vreg(&operand(add, 1)),
                vreg(&operand(add, 2))
            ),
            (
                "v_add_f32_e32",
                vreg(&operand(add, 0)),
                vreg(&operand(add, 0)),
                dot
            ),
            "{symbol}: group partial add"
        );
        // Shuffle offsets 16, 8, 4, 2, 1 in that order.
        let offsets: Vec<u32> = lines
            .iter()
            .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0")
            .map(|l| operand(l, 2).parse().unwrap())
            .collect();
        assert_eq!(offsets, [16, 8, 4, 2, 1], "{symbol}: shuffle offsets");
        // Out-of-range lanes stay unchanged: per slot and step the sum goes to the exchange register, and v_cndmask
        // keeps the old accumulator where the step's compare mask is clear.
        let exch = vreg(&operand(
            lines
                .iter()
                .find(|l| mnemonic(l) == "ds_bpermute_b32")
                .unwrap(),
            0,
        ));
        let selects = lines
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_cndmask_b32_e64"
                    && operand(l, 0) == operand(l, 1)
                    && vreg(&operand(l, 2)) == exch
            })
            .count();
        assert_eq!(selects, 5 * SLOTS, "{symbol}: accumulator selects");
        let sums = lines
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_add_f32_e32"
                    && vreg(&operand(l, 0)) == exch
                    && vreg(&operand(l, 2)) == exch
            })
            .count();
        assert_eq!(sums, 5 * SLOTS, "{symbol}: shuffle sums into scratch");
        assert!(
            !lines.iter().any(|l| mnemonic(l) == "v_add_f32_e32"
                && operand(l, 0) == operand(l, 1)
                && vreg(&operand(l, 2)) == exch),
            "{symbol}: unconditional shuffle add"
        );
        if spec.codec == Codec::Mq4Sym {
            // w = fma(d, float(q), z): one (d, z) pair across every fma, weight in place.
            let fmas: Vec<&&str> = lines
                .iter()
                .filter(|l| mnemonic(l) == "v_fma_f32")
                .collect();
            let (d, z) = (operand(fmas[0], 1), operand(fmas[0], 3));
            assert!(
                fmas.iter().all(|l| operand(l, 0) == operand(l, 2)
                    && operand(l, 1) == d
                    && operand(l, 3) == z),
                "{symbol}: affine weight fma(d, q, z)"
            );
        }
    }
}

/// Read-once C8: every weight-class load of a group sits before that group's first slot branch and
/// never inside one; the number of weight-class loads does not depend on the slot count (it is the same
/// per group instance for one and for eight slots), and every slot branch targets a join label of the
/// same group (or the epilogue ladders), so a skipped slot can never skip a weight load.
#[test]
fn weights_are_loaded_once_per_group_outside_every_slot_branch() {
    for spec in all_specs() {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let mut inside = false;
        let (mut gated, mut weight_loads_outside) = (0usize, 0usize);
        for line in t.lines().map(str::trim) {
            if line.ends_with(':') && !line.contains(' ') {
                inside = false;
                continue;
            }
            let m = mnemonic(line);
            if m == "s_cbranch_scc0" && (line.ends_with("_xdone") || line.ends_with("_done")) {
                inside = true;
                gated += 1;
                continue;
            }
            if inside {
                assert!(
                    !m.starts_with("global_load") && !m.starts_with("s_load"),
                    "{symbol}: weight/scalar load inside a slot branch: {line}"
                );
            } else if m.starts_with("global_load") {
                weight_loads_outside += 1
            }
        }
        // 7 X-ladder + 7 compute-ladder gates per group instance, 7 + 7 in the reduce and store ladders.
        assert_eq!(gated, 14 * inst + 14, "{symbol}: slot gates");
        // Weight-class loads (plus the single slot-id load) are exactly the per-instance counts, never multiplied by the slot count.
        let per_group = match spec.codec {
            Codec::Mq3I4 => 3,
            Codec::Mq4Sym => 2,
        }; // Mq3I4: two packet bytes + scale; Mq4Sym: header dword + payload
        assert_eq!(
            weight_loads_outside,
            per_group * inst + 1,
            "{symbol}: weight loads per group (read once)"
        );
        // No loop over slots around a group: the group instance count is the unrolled one (4 loop + 3 tail groups, or 1), not slots * groups.
        assert_eq!(
            count(&t, "s_cbranch_scc0"),
            14 * inst + 14 + if spec.kind == Kind::GateUp { 3 } else { 0 },
            "{symbol}: scc0 branches"
        );
        assert_eq!(
            count(&t, "s_cbranch_scc1"),
            8,
            "{symbol}: scc1 branches (6 guards, loop entry skip, back edge)"
        );
    }
}

/// The shape contract derives deterministically from emission: counts of the characteristic
/// instructions, the forbidden families (LDS and scratch, no `buffer_*` ban: X goes through a raw
/// descriptor), register ceilings equal to the plan, no private or LDS memory.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in all_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let c = mq_i4_moe_gemv::contract(spec, &e);
        assert_eq!(
            c,
            mq_i4_moe_gemv::contract(spec, &mq_i4_moe_gemv::emit(spec).unwrap())
        );
        assert_eq!(c["symbol"], spec.symbol().as_str());
        assert_eq!(c["counts"]["s_endpgm"], 1);
        assert_eq!(c["counts"]["global_store_b32"], SLOTS);
        assert_eq!(c["counts"]["ds_bpermute_b32"], 5 * SLOTS);
        assert_eq!(c["counts"]["v_readlane_b32"], 2 * SLOTS);
        assert_eq!(
            (
                c["launch_dynamic_lds_bytes"].as_u64(),
                c["group_segment_fixed_bytes"].as_u64()
            ),
            (Some(0), Some(0))
        );
        assert_eq!(
            (
                c["require_wave32"].as_bool(),
                c["require_zero_spills"].as_bool(),
                c["require_zero_private"].as_bool()
            ),
            (Some(true), Some(true), Some(true))
        );
        let forbidden: Vec<&str> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for family in [
            "scratch_*",
            "flat_*",
            "ds_load*",
            "ds_store*",
            "ds_swizzle*",
            "s_barrier*",
            "v_wmma*",
            "v_fma_mix*",
        ] {
            assert!(forbidden.contains(&family), "{family}");
        }
        // The other architecture's wait family is forbidden: `s_waitcnt*` on gfx12, the counter waits on gfx11 (where `s_waitcnt` is the wait).
        if spec.arch.gfx12() {
            assert!(forbidden.contains(&"s_waitcnt*"))
        } else {
            for family in [
                "s_wait_loadcnt*",
                "s_wait_storecnt*",
                "s_wait_dscnt*",
                "s_wait_kmcnt*",
                "s_wait_alu*",
            ] {
                assert!(forbidden.contains(&family), "{family}");
            }
            assert!(!forbidden.contains(&"s_waitcnt*"))
        }
        assert!(!forbidden.contains(&"buffer_*") && !forbidden.contains(&"v_readlane*"));
    }
}

/// Production `emit` is the reference `Mq3I4` network (two `global_load_u8` at offsets 0 and 1 plus `v_lshl_or_b32`);
/// `emit_optimized` retains the single `global_load_u16` packet network (byte-exact on GPU, slower by timing, not production).
/// Both read the same 16 bits at the same address into the same register, the packet never crosses the 48-byte
/// group, every other VALU instruction (all arithmetic) is identical, and `Mq4Sym` is byte-identical.
#[test]
fn optimized_w3_packet_is_the_same_sixteen_bits_as_the_reference_two_byte_network() {
    // Lane l reads group bytes (3l)>>1 and +1: at most byte 47 of 48.
    assert!((0..32u32).all(|l| (3 * l) / 2 + 2 <= 48));
    for spec in all_specs() {
        let (opt, reference) = (
            mq_i4_moe_gemv::emit_optimized(spec).unwrap().s_text,
            mq_i4_moe_gemv::emit(spec).unwrap().s_text,
        );
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        if spec.codec == Codec::Mq4Sym {
            assert_eq!(opt, reference, "{symbol}: Mq4Sym is network-independent");
            continue;
        }
        assert_eq!(
            (
                count(&reference, "global_load_u8"),
                count(&reference, "global_load_u16"),
                count(&reference, "v_lshl_or_b32")
            ),
            (2 * inst, inst, inst),
            "{symbol}: reference network"
        );
        assert_eq!(
            (
                count(&opt, "global_load_u8"),
                count(&opt, "global_load_u16"),
                count(&opt, "v_lshl_or_b32")
            ),
            (0, 2 * inst, 0),
            "{symbol}: optimized network"
        );
        let loads = |t: &str, m: &str| -> Vec<String> {
            t.lines()
                .map(str::trim)
                .filter(|l| mnemonic(l) == m)
                .map(str::to_owned)
                .collect()
        };
        let bytes = loads(&reference, "global_load_u8");
        let (first, second): (Vec<&String>, Vec<&String>) =
            bytes.iter().partition(|l| !l.contains("offset:1"));
        assert_eq!(
            (first.len(), second.len()),
            (inst, inst),
            "{symbol}: byte loads at offsets 0 and 1"
        );
        let packet_dst = operand(first[0], 0);
        let optimized_packet: Vec<String> = loads(&opt, "global_load_u16")
            .into_iter()
            .filter(|l| operand(l, 0) == packet_dst)
            .collect();
        assert_eq!(
            optimized_packet.len(),
            inst,
            "{symbol}: one u16 packet load per group"
        );
        for ((lo, hi), wide) in first.iter().zip(&second).zip(&optimized_packet) {
            // Same destination register and address as the offset-0 byte load, no offset; the offset-1 byte shares its address operands.
            assert_eq!(
                operand(lo, 0),
                operand(wide, 0),
                "{symbol}: packet register"
            );
            assert_eq!(
                operand(lo, 1),
                operand(wide, 1),
                "{symbol}: packet address register"
            );
            assert_eq!(
                wide.split_once(' ').unwrap().1,
                lo.split_once(' ').unwrap().1,
                "{symbol}: packet address and base, no offset"
            );
            assert!(
                operand(hi, 2).ends_with(" offset:1") && operand(hi, 1) == operand(lo, 1),
                "{symbol}: second byte is +1"
            );
        }
        // All arithmetic is the same: the VALU stream differs only by the removed packet combine.
        let valu = |t: &str, skip_combine: bool| -> Vec<String> {
            t.lines()
                .map(str::trim)
                .filter(|l| {
                    mnemonic(l).starts_with("v_")
                        && !(skip_combine && mnemonic(l) == "v_lshl_or_b32")
                })
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(
            valu(&reference, true),
            valu(&opt, false),
            "{symbol}: arithmetic is identical"
        );
    }
}

#[test]
fn codec_and_kind_constants_match_the_frozen_format() {
    assert_eq!(
        (Codec::Mq3I4.tag(), Codec::Mq4Sym.tag()),
        ("mq3i4", "mq4sym")
    );
    assert_eq!(
        (Kind::GateUp.tag(), Kind::Down.tag()),
        ("r256_gate_up", "r128_down")
    );
    assert_eq!((Kind::GateUp.chains(), Kind::Down.chains()), (4, 1));
    assert_eq!(mq_i4_moe_gemv::GROUP, 128);
    // The Mq4Sym gate lane owns eight elements of a G256 group; every other lane owns four.
    for spec in all_specs() {
        let wide = spec.codec == Codec::Mq4Sym && spec.kind == Kind::GateUp;
        assert_eq!(spec.lane_elems(), if wide { 8 } else { 4 }, "{spec:?}");
        assert_eq!(
            spec.group_instances(),
            if spec.kind == Kind::GateUp { 7 } else { 1 },
            "{spec:?}"
        );
    }
}

/// Every diagnostic entry and the diagnostic module assemble independently of committed artifacts.
#[test]
fn native_writer_assembles_every_diagnostic_entry_and_module() {
    for arch in ARCHES {
        let (emitted, text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        for e in &emitted {
            let elf = peacemaker_isa::native::assemble(&e.s_text, arch)
                .unwrap_or_else(|err| panic!("{arch:?} {}: {err}", e.proof.kernel_id));
            assert!(elf.starts_with(b"\x7fELF"));
        }
        let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
        let bundle =
            peacemaker_isa::native::bundle(&elf, arch, peacemaker_isa::native::DEFAULT_HOST_TARGET);
        assert!(bundle.len() > elf.len());
    }
}

/// Fresh source/native identity of the production image, not the twelve-entry diagnostic module.
#[test]
fn committed_gfx1201_product_bundle_equals_narrow_plus_tiles() {
    let (_, text) = gfx1201_product_image();
    assert_committed_bundle(Arch::Gfx1201, &text);
}

/// The common Halo artifact is deliberately retained as a diagnostic/reference image.
/// Halo's production narrow-plus-tiles artifact belongs to the Cn test suite.
#[test]
fn committed_gfx1151_diagnostic_bundle_equals_legacy_module() {
    let (_, text, _) = mq_i4_moe_gemv::emit_module(Arch::Gfx1151).unwrap();
    assert_committed_bundle(Arch::Gfx1151, &text);
}

fn assert_committed_bundle(arch: Arch, text: &str) {
    let module = module(arch);
    let path = format!(
        "{}/../../kernels/{module}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel mq_i4_moe_gemv --arch {} ... --bundle)", arch.name()));
    let elf = peacemaker_isa::native::assemble(text, arch).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, arch, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{module}: fresh emission differs from the committed bundle {path}"
    );
}

#[test]
fn native_writer_assembles_gfx1201_product_entries_and_image() {
    let (emitted, text) = gfx1201_product_image();
    for e in emitted {
        let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1201)
            .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1201).unwrap();
    let bundle = peacemaker_isa::native::bundle(
        &elf,
        Arch::Gfx1201,
        peacemaker_isa::native::DEFAULT_HOST_TARGET,
    );
    assert!(bundle.len() > elf.len());
}

/// The wide twins' interface: the narrow 96-byte kernarg ABI field for field, block 256, the static X tile as the only
/// LDS (4096 B; `Mq4Sym` gate G256: 8192 B), wave32, no scratch / spills / flat, barrier spellings per architecture.
#[test]
fn wide_entries_keep_the_abi_and_declare_block_256_and_the_x_tile() {
    for spec in wide_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let symbol = format!("{:?} {}", spec.arch, spec.symbol());
        let t = &e.s_text;
        assert_eq!(spec.workgroup_size(), 256, "{symbol}");
        assert_eq!(
            spec.lds_bytes(),
            if spec.lane_elems() == 8 { 8192 } else { 4096 },
            "{symbol}"
        );
        assert!(
            t.contains(".kernarg_segment_size: 96\n") && t.contains(".amdhsa_kernarg_size 96\n"),
            "{symbol}: kernarg size"
        );
        assert_eq!(
            field(t, ".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(t, ".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
        let names: Vec<String> = t
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
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [256],
            "{symbol}: block 256"
        );
        assert!(
            t.contains(&format!(
                ".group_segment_fixed_size: {}\n",
                spec.lds_bytes()
            )),
            "{symbol}: static LDS"
        );
        assert!(
            t.contains(&format!(
                ".amdhsa_group_segment_fixed_size {}\n",
                spec.lds_bytes()
            )),
            "{symbol}: descriptor LDS"
        );
        assert!(
            t.contains(".amdhsa_wavefront_size32 1")
                && t.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{symbol}: wave32 / grid y"
        );
        assert!(
            t.contains(".private_segment_fixed_size: 0\n")
                && t.contains(".sgpr_spill_count: 0\n")
                && t.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: no private memory"
        );
        assert!(
            !t.lines().any(|l| mnemonic(l).starts_with("scratch_")
                || mnemonic(l).starts_with("flat_")
                || mnemonic(l).starts_with("ds_swizzle")),
            "{symbol}: scratch/flat/swizzle"
        );
        let plan: Vec<&str> = e.proof.reg_plan.iter().map(|r| r.name.as_ref()).collect();
        for owner in [
            "kernel_arguments_ptrs",
            "x_descriptor",
            "y_select",
            "accumulators",
            "x_values",
            "weights",
            "slot_x_offsets",
            "slot_y_addresses",
            "x_tile_store_address",
            "x_tile_read_base",
            "wave",
            "row_block",
        ] {
            assert!(plan.contains(&owner), "{symbol}: reg plan misses {owner}");
        }
        // Workgroup ids and barriers follow the architecture; the first loads are the narrow kernarg loads.
        if spec.arch.gfx12() {
            assert!(
                t.contains("s_mov_b32 s81, ttmp9") && t.contains("s_and_b32 s31, ttmp7, 0xffff"),
                "{symbol}: gfx12 workgroup ids"
            );
            assert!(
                count(t, "s_barrier_signal") > 0
                    && count(t, "s_barrier_signal") == count(t, "s_barrier_wait")
                    && count(t, "s_barrier") == 0,
                "{symbol}: gfx12 barrier"
            );
            assert!(
                t.contains("s_barrier_signal -1") && t.contains("s_barrier_wait 0xffff"),
                "{symbol}: gfx12 barrier operands"
            );
        } else {
            assert!(
                t.contains("s_mov_b32 s81, s2")
                    && t.contains("s_mov_b32 s31, s3")
                    && !t.contains("ttmp"),
                "{symbol}: gfx11 workgroup ids"
            );
            assert!(
                count(t, "s_barrier") > 0
                    && count(t, "s_barrier_signal") == 0
                    && count(t, "s_barrier_wait") == 0,
                "{symbol}: gfx11 barrier"
            );
        }
        let zero = if spec.arch.gfx12() { "0x0" } else { "null" };
        for load in [
            format!("s_load_b256 s[8:15], s[0:1], {zero}"),
            "s_load_b256 s[16:23], s[0:1], 0x20".into(),
            "s_load_b128 s[24:27], s[0:1], 0x40".into(),
            "s_load_b64 s[28:29], s[0:1], 0x50".into(),
        ] {
            assert!(t.contains(&load), "{symbol}: kernarg load {load}");
        }
        assert!(
            t.contains("s_load_b32 s84, s[0:1], 0x58"),
            "{symbol}: n_experts kernarg load"
        );
    }
}

/// Honest register and LDS ceilings of the wide state (design §4): `4*C` gate / `C` down accumulators per lane, one weight
/// group and one X vector; whole-granule VGPR reservation, SGPR bookkeeping, the contract equals the plan.
#[test]
fn wide_register_ceilings_hold_the_design_budget() {
    for spec in wide_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let symbol = format!("{:?} {}", spec.arch, spec.symbol());
        let (nw, chains, cap) = (
            spec.lane_elems() as u16,
            spec.kind.chains() as u16,
            spec.cap.slots() as u16,
        );
        let floor = cap * chains + 2 * nw;
        let ceiling = match (spec.kind, spec.cap) {
            (Kind::GateUp, Cap::C16) => 128,
            (Kind::GateUp, _) => 192,
            (Kind::Down, _) => 80,
        };
        assert!(
            e.shape.next_free_vgpr >= floor
                && e.shape.next_free_vgpr % 8 == 0
                && e.shape.next_free_vgpr <= ceiling,
            "{symbol}: vgpr {} (ceiling {ceiling})",
            e.shape.next_free_vgpr
        );
        assert!(
            e.shape.next_free_sgpr <= 88,
            "{symbol}: sgpr {}",
            e.shape.next_free_sgpr
        );
        assert!(
            e.s_text
                .contains(&format!(".vgpr_count: {}\n", e.shape.next_free_vgpr)),
            "{symbol}: metadata vgpr count"
        );
        let c = mq_i4_moe_gemv::contract(spec, &e);
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (
                Some(u64::from(e.shape.next_free_vgpr)),
                Some(u64::from(e.shape.next_free_sgpr))
            ),
            "{symbol}"
        );
        assert_eq!(
            c["group_segment_fixed_bytes"].as_u64(),
            Some(u64::from(spec.lds_bytes())),
            "{symbol}"
        );
    }
}

/// The frozen numerical tree of the wide twins: the narrow per-slot chains (weights reconstructed once per group, so the
/// weight-side counts do not depend on the capacity), `C` stores, five snapshot shuffle steps per slot, the X tile
/// staged `tiles` times per group with two barriers each.
#[test]
fn wide_instruction_census_matches_the_frozen_dag() {
    for spec in wide_specs() {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = format!("{:?} {}", spec.arch, spec.symbol());
        let inst = spec.group_instances() as usize;
        let (nw, cap) = (spec.lane_elems() as usize, spec.cap.slots() as usize);
        let tiles = cap / 8;
        let planar = spec.codec == Codec::Mq3I4;
        let gate = spec.kind == Kind::GateUp;
        let (u8s, u16s, b32s) = match (spec.codec, spec.kind) {
            (Codec::Mq3I4, _) => (2 * inst, inst, 1),
            (Codec::Mq4Sym, Kind::GateUp) => (0, 0, 2 * inst + 1),
            (Codec::Mq4Sym, Kind::Down) => (0, inst, inst + 1),
        };
        assert_eq!(
            count(&t, "global_load_u8"),
            u8s,
            "{symbol}: packet byte loads (capacity independent)"
        );
        assert_eq!(count(&t, "global_load_u16"), u16s, "{symbol}: u16 loads");
        assert_eq!(count(&t, "global_load_b32"), b32s, "{symbol}: dword loads");
        assert_eq!(
            count(&t, "buffer_load_b128"),
            inst * tiles * nw / 4,
            "{symbol}: X loads, one per (group, tile)"
        );
        assert_eq!(
            count(&t, "ds_store_b128"),
            inst * tiles * nw / 4,
            "{symbol}: X tile stores"
        );
        assert_eq!(
            count(&t, "ds_load_b128"),
            inst * cap * nw / 4,
            "{symbol}: X tile reads"
        );
        assert_eq!(count(&t, "global_store_b32"), cap, "{symbol}: stores");
        let barriers = if spec.arch.gfx12() {
            count(&t, "s_barrier_signal")
        } else {
            count(&t, "s_barrier")
        };
        assert_eq!(
            barriers,
            2 * tiles * inst,
            "{symbol}: two barriers per (group, tile)"
        );
        assert_eq!(count(&t, "v_bfe_u32"), nw * inst, "{symbol}: code extracts");
        assert_eq!(
            count(&t, "v_cvt_f32_i32_e32"),
            if planar { 4 * inst } else { 0 },
            "{symbol}: signed grid converts"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_u32_e32"),
            2 + if planar { 0 } else { nw * inst },
            "{symbol}: unsigned converts (2 are the slot divide)"
        );
        assert_eq!(
            count(&t, "v_fma_f32"),
            if planar { 0 } else { nw * inst },
            "{symbol}: affine fma"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_f16_e64"),
            inst * if planar { 1 } else { 2 },
            "{symbol}: header converts"
        );
        assert_eq!(
            count(&t, "v_lshl_or_b32"),
            if planar { inst } else { 0 },
            "{symbol}: W3 packet combine"
        );
        assert_eq!(
            count(&t, "v_mul_f32_e32"),
            cap * inst + if planar { 4 * inst } else { 0 } + 1,
            "{symbol}: multiplies"
        );
        assert_eq!(
            count(&t, "v_fmac_f32_e32"),
            (nw - 1) * cap * inst,
            "{symbol}: fused steps"
        );
        assert_eq!(
            count(&t, "v_add_f32_e32"),
            cap * inst + cap * (if gate { 3 } else { 0 } + 5),
            "{symbol}: group-partial, fold and shuffle adds"
        );
        assert_eq!(
            count(&t, "ds_bpermute_b32"),
            5 * cap,
            "{symbol}: shuffle steps"
        );
        assert_eq!(
            count(&t, "v_readlane_b32"),
            inst * tiles + cap,
            "{symbol}: producer slot offsets and store addresses"
        );
        assert_eq!(count(&t, "v_readfirstlane_b32"), 1, "{symbol}: wave index");
        assert_eq!(
            count(&t, "v_cmpx_eq_u32_e32"),
            1,
            "{symbol}: lane-0 store mask"
        );
        for forbidden in [
            "ds_swizzle",
            "v_fma_mix_f32",
            "v_xor_b32",
            "flat_",
            "scratch_",
            "v_wmma",
            "global_atomic",
            "buffer_atomic",
        ] {
            let hit = t.lines().find(|l| mnemonic(l).starts_with(forbidden));
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
        let lines: Vec<&str> = t
            .lines()
            .map(str::trim)
            .filter(|l| {
                !mnemonic(l).starts_with("s_delay_alu")
                    && !mnemonic(l).starts_with("s_wait")
                    && !mnemonic(l).starts_with("s_nop")
                    && !mnemonic(l).starts_with("ds_load")
            })
            .collect();
        let first_fmac = lines
            .iter()
            .position(|l| mnemonic(l) == "v_fmac_f32_e32")
            .unwrap();
        let mul = lines[first_fmac - 1];
        assert_eq!(mnemonic(mul), "v_mul_f32_e32", "{symbol}: first dot term");
        // Mq3I4: p = RN(w0*x0), fma(w_j,x_j,p) j = 1..; Mq4Sym: p = RN(w1*x1), fma(w0,x0,p), then j = 2.. ascending.
        let order: Vec<u32> = if planar {
            (0..nw as u32).collect()
        } else {
            [1, 0].into_iter().chain(2..nw as u32).collect()
        };
        let (dot, wf, xf) = (
            vreg(&operand(mul, 0)),
            vreg(&operand(mul, 1)),
            vreg(&operand(mul, 2)),
        );
        let (w0, x0) = (wf - order[0], xf - order[0]);
        for (k, &j) in order.iter().enumerate().skip(1) {
            let l = lines[first_fmac + k - 1];
            assert_eq!(mnemonic(l), "v_fmac_f32_e32", "{symbol}: fused step {k}");
            assert_eq!(
                (
                    vreg(&operand(l, 0)),
                    vreg(&operand(l, 1)),
                    vreg(&operand(l, 2))
                ),
                (dot, w0 + j, x0 + j),
                "{symbol}: fused step {k} (element {j}) operands"
            );
        }
        let add = lines[first_fmac + nw - 1];
        assert_eq!(
            (
                mnemonic(add),
                operand(add, 0),
                operand(add, 1),
                vreg(&operand(add, 2))
            ),
            ("v_add_f32_e32", operand(add, 0), operand(add, 0), dot),
            "{symbol}: group partial add"
        );
        let offsets: Vec<u32> = lines
            .iter()
            .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0")
            .map(|l| operand(l, 2).parse().unwrap())
            .collect();
        assert_eq!(offsets, [16, 8, 4, 2, 1], "{symbol}: shuffle offsets");
        // Four exchange registers; each shuffle sum goes to its own exchange register and a select keeps out-of-range lanes.
        let exch: Vec<u32> = lines
            .iter()
            .filter(|l| mnemonic(l) == "ds_bpermute_b32")
            .map(|l| vreg(&operand(l, 0)))
            .collect();
        let base = *exch.iter().min().unwrap();
        assert!(
            exch.iter().all(|&r| (base..base + 4).contains(&r)),
            "{symbol}: exchange registers"
        );
        let selects = lines
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_cndmask_b32_e64"
                    && operand(l, 0) == operand(l, 1)
                    && (base..base + 4).contains(&vreg(&operand(l, 2)))
            })
            .count();
        assert_eq!(selects, 5 * cap, "{symbol}: accumulator selects");
        let sums = lines
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_add_f32_e32"
                    && (base..base + 4).contains(&vreg(&operand(l, 0)))
                    && operand(l, 0) == operand(l, 2)
            })
            .count();
        assert_eq!(sums, 5 * cap, "{symbol}: shuffle sums into scratch");
        if spec.codec == Codec::Mq4Sym {
            let fmas: Vec<&&str> = lines
                .iter()
                .filter(|l| mnemonic(l) == "v_fma_f32")
                .collect();
            let (d, z) = (operand(fmas[0], 1), operand(fmas[0], 3));
            assert!(
                fmas.iter().all(|l| operand(l, 0) == operand(l, 2)
                    && operand(l, 1) == d
                    && operand(l, 3) == z),
                "{symbol}: affine weight fma(d, q, z)"
            );
        }
    }
}

/// Read-once and uniform-barrier structure of the wide twins: every weight-class global load sits in the group's
/// valid-row region before any tile, no X / LDS access sits outside its guarded phase, tiles past the expert's slots are
/// skipped as a whole with their barriers, and no barrier sits inside any divergent region (tail waves reach every barrier).
#[test]
fn wide_weights_load_once_per_group_and_barriers_are_never_inside_a_branch_region() {
    for spec in wide_specs() {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = format!("{:?} {}", spec.arch, spec.symbol());
        let (inst, tiles) = (
            spec.group_instances() as usize,
            spec.cap.slots() as usize / 8,
        );
        // (region suffix, instruction class that may sit inside it)
        let regions = ["_wdone", "_pdone", "_sdone", "_udone", "_cdone"];
        let mut open: Option<&str> = None;
        let (mut seen_slot_load, mut weight_loads, mut gend_branches) = (false, 0usize, 0usize);
        for line in t.lines().map(str::trim) {
            let m = mnemonic(line);
            if line.ends_with(':') && !line.contains(' ') {
                if let Some(r) = open {
                    if line.ends_with(&format!("{r}:")) {
                        open = None
                    }
                }
                continue;
            }
            if m == "s_cbranch_scc0" {
                if line.ends_with("_gend") {
                    gend_branches += 1;
                    continue;
                }
                if let Some(r) = regions.into_iter().find(|r| line.ends_with(r)) {
                    if open.is_none() {
                        open = Some(r)
                    }
                }
                continue;
            }
            if m.starts_with("s_barrier") {
                assert!(
                    open.is_none(),
                    "{symbol}: barrier inside the {open:?} region"
                );
            }
            if m.starts_with("global_load") {
                if !seen_slot_load {
                    seen_slot_load = true;
                    assert!(open.is_none(), "{symbol}: slot-id load inside a region");
                    continue;
                }
                assert_eq!(
                    open,
                    Some("_wdone"),
                    "{symbol}: weight load outside the valid-row weight region: {line}"
                );
                weight_loads += 1;
            }
            if m == "buffer_load_b128" {
                assert_eq!(
                    open,
                    Some("_pdone"),
                    "{symbol}: X load outside the producer region"
                );
            }
            if m == "ds_store_b128" {
                assert_eq!(
                    open,
                    Some("_sdone"),
                    "{symbol}: LDS store outside the store region"
                );
            }
            if m == "ds_load_b128" {
                assert_eq!(
                    open,
                    Some("_cdone"),
                    "{symbol}: LDS read outside the consume region"
                );
            }
        }
        assert!(open.is_none(), "{symbol}: unterminated region");
        let per_group = match spec.codec {
            Codec::Mq3I4 => 3,
            Codec::Mq4Sym => 2,
        };
        assert_eq!(
            weight_loads,
            per_group * inst,
            "{symbol}: weight loads per group (capacity independent)"
        );
        assert_eq!(
            gend_branches,
            (tiles - 1) * inst,
            "{symbol}: one uniform tile skip per tile past the first"
        );
    }
}

/// The wide contract: the narrow counts' vocabulary plus the LDS / barrier mnemonics, the static X tile as the group
/// segment, and no ban on `ds_load*` / `ds_store*` / `s_barrier*` (the swizzle, scratch and flat bans stay).
#[test]
fn wide_shape_contract_declares_the_x_tile_and_allows_lds() {
    for spec in wide_specs() {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let c = mq_i4_moe_gemv::contract(spec, &e);
        let (cap, nw, tiles) = (
            spec.cap.slots() as u64,
            spec.lane_elems() as u64,
            spec.cap.slots() as u64 / 8,
        );
        let inst = spec.group_instances() as u64;
        assert_eq!(c["symbol"], spec.symbol().as_str());
        assert_eq!(c["counts"]["global_store_b32"], cap);
        assert_eq!(c["counts"]["ds_bpermute_b32"], 5 * cap);
        assert_eq!(c["counts"]["ds_load_b128"], inst * cap * nw / 4);
        assert_eq!(c["counts"]["ds_store_b128"], inst * tiles * nw / 4);
        assert_eq!(
            c["counts"][if spec.arch.gfx12() {
                "s_barrier_signal"
            } else {
                "s_barrier"
            }],
            2 * tiles * inst
        );
        assert_eq!(
            (
                c["launch_dynamic_lds_bytes"].as_u64(),
                c["group_segment_fixed_bytes"].as_u64()
            ),
            (Some(0), Some(u64::from(spec.lds_bytes())))
        );
        let forbidden: Vec<&str> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for family in [
            "scratch_*",
            "flat_*",
            "ds_swizzle*",
            "v_wmma*",
            "v_fma_mix*",
        ] {
            assert!(forbidden.contains(&family), "{family}");
        }
        for family in ["ds_load*", "ds_store*", "s_barrier*"] {
            assert!(!forbidden.contains(&family), "{family}");
        }
        assert_eq!(
            c,
            mq_i4_moe_gemv::contract(spec, &mq_i4_moe_gemv::emit(spec).unwrap())
        );
    }
}

/// The capacity axis: symbols and tags carry the suffix, the narrow entry none; slots, block size and LDS per entry.
#[test]
fn cap_axis_names_and_resources() {
    assert_eq!(
        (Cap::C8.suffix(), Cap::C16.suffix(), Cap::C32.suffix()),
        ("", "_c16", "_c32")
    );
    assert_eq!(
        (Cap::C8.slots(), Cap::C16.slots(), Cap::C32.slots()),
        (8, 16, 32)
    );
    for spec in module_entries() {
        assert_eq!(
            spec.symbol(),
            format!(
                "mq_i4_moe_gemv_{}_{}{}",
                spec.codec.tag(),
                spec.kind.tag(),
                spec.cap.suffix()
            )
        );
        assert_eq!(
            spec.tag(),
            format!(
                "{}_{}{}",
                spec.codec.tag(),
                spec.kind.tag(),
                spec.cap.suffix()
            )
        );
        assert_eq!(
            spec.workgroup_size(),
            if spec.cap == Cap::C8 { 32 } else { 256 }
        );
        assert_eq!(
            spec.lds_bytes(),
            if spec.cap == Cap::C8 {
                0
            } else {
                8 * 128 * 4 * (spec.lane_elems() as u32 / 4)
            }
        );
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, native, pm_check,
        toolchain::{build, certify, oracle_assemble_link_bundle, IsaShapeContract, Toolchain},
    };

    const HOST: &str = "host-x86_64-unknown-linux-gnu-";

    /// Every diagnostic symbol's waits and hazards replay independently of the builder's ledger.
    /// M7 lifts all twelve entries, including retained C16/C32 diagnostics, without committed fixtures.
    #[test]
    fn every_diagnostic_export_passes_wait_hazard_replay_and_m7() {
        for arch in ARCHES {
            let (emitted, text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch)
                    .unwrap_or_else(|err| panic!("{arch:?} {}: {err}", e.proof.variant));
                assert!(e.s_text.contains(".kernarg_segment_size: 96\n"));
            }
            ledger_replay::replay_waits(&text, arch).unwrap();
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-mqi4moe-m7-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let co = dir.join("module.co");
            std::fs::write(&co, native::assemble(&text, arch).unwrap()).unwrap();
            for symbol in mq_i4_moe_gemv::module_specs(arch).iter().map(Spec::symbol) {
                let symbol = symbol.as_str();
                let m7 = pm_check::m7(&co, arch.name(), symbol)
                    .unwrap_or_else(|e| panic!("{arch:?} {symbol}: {e}"));
                assert_eq!(
                    (m7["lift"].as_str(), &m7["obligations"]),
                    (Some("byte-exact"), &serde_json::json!({})),
                    "{arch:?} {symbol}"
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Fixed-eight production entries also owe replay and M7; legacy diagnostics above remain separate.
    #[test]
    fn rowtiles_pass_wait_replay_and_m7() {
        for spec in all_specs() {
            let e = mq_i4_moe_gemv::emit_rowtiles(spec).unwrap();
            ledger_replay::replay_waits(&e.s_text, spec.arch)
                .unwrap_or_else(|err| panic!("{spec:?}: {err}"));
            let contract = mq_i4_moe_gemv::contract_rowtiles(spec, &e);
            let symbol = contract["symbol"].as_str().unwrap();
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-mqi4moe-rowtiles-m7-{}-{}-{}",
                spec.arch.name(),
                spec.tag(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let co = dir.join("tile.co");
            std::fs::write(&co, native::assemble(&e.s_text, spec.arch).unwrap()).unwrap();
            let m7 = pm_check::m7(&co, spec.arch.name(), symbol)
                .unwrap_or_else(|err| panic!("{spec:?} {symbol}: {err}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{spec:?} {symbol}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Native objects/bundles against ROCm for all diagnostic entries/modules and the actual
    /// gfx1201 narrow-plus-tiles production entries/image. Only the `.comment` stamp may differ.
    #[test]
    fn native_diagnostics_and_gfx1201_product_equal_the_rocm_oracle() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let toolchain = Toolchain {
            host_target: HOST.into(),
            ..Toolchain::default()
        };
        let mut failures = Vec::new();
        let mut total = 0;
        for arch in ARCHES {
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-mqi4moe-identity-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let (emitted, module_text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
            let mut units: Vec<(String, String)> = emitted
                .iter()
                .zip(mq_i4_moe_gemv::module_specs(arch))
                .map(|(e, s)| (format!("{} {}", arch.name(), s.symbol()), e.s_text.clone()))
                .collect();
            units.push((format!("{} (diagnostic module)", module(arch)), module_text));
            if arch == Arch::Gfx1201 {
                let (product, product_text) = gfx1201_product_image();
                units.extend(product.into_iter().map(|e| {
                    (
                        format!("{} {} (product)", arch.name(), e.proof.variant),
                        e.s_text,
                    )
                }));
                units.push((format!("{} (product image)", module(arch)), product_text));
            }
            total += units.len();
            for (i, (name, text)) in units.iter().enumerate() {
                let s = dir.join(format!("u{i}.s"));
                std::fs::write(&s, text).unwrap();
                let oracle = oracle_assemble_link_bundle(
                    &toolchain,
                    &s,
                    &dir.join(format!("u{i}.hxaco")),
                    arch.name(),
                )
                .unwrap();
                let (oracle_elf, oracle_bundle) = (
                    std::fs::read(&oracle.elf).unwrap(),
                    std::fs::read(&oracle.hsaco).unwrap(),
                );
                match native::assemble(text, arch) {
                    Ok(elf) => {
                        let bundle = native::bundle(&elf, arch, HOST);
                        if let Some(why) = native_compare::compare_native_to_oracle(
                            &elf,
                            &bundle,
                            &oracle_elf,
                            &oracle_bundle,
                        ) {
                            failures.push(format!("{name}: {why}"))
                        }
                    }
                    Err(e) => failures.push(format!("{name}: native: {e}")),
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert!(
            failures.is_empty(),
            "{} of {total} units differ from the oracle:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Diagnostic certification stays source-derived for all twelve symbols; no obsolete
    /// production C16/C32 contract artifact is required.
    #[test]
    fn diagnostic_module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-mqi4moe-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            for (spec, e) in mq_i4_moe_gemv::module_specs(arch).into_iter().zip(&emitted) {
                let symbol = spec.symbol();
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                    .unwrap_or_else(|err| panic!("{arch:?} {symbol}: {err}"));
                assert_eq!(
                    (m7["lift"].as_str(), &m7["obligations"]),
                    (Some("byte-exact"), &serde_json::json!({})),
                    "{arch:?} {symbol}"
                );
                let tag = spec.tag();
                let contract: IsaShapeContract =
                    serde_json::from_value(mq_i4_moe_gemv::contract(spec, e)).unwrap();
                certify(
                    &toolchain,
                    &build,
                    &s,
                    arch.name(),
                    &dir.join(format!("{tag}.manifest.json")),
                    Some(&contract),
                    "test",
                    "test",
                )
                .unwrap_or_else(|err| panic!("{arch:?} {symbol}: {err}"));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
