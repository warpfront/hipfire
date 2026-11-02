// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 (and, via the gfx11 arms, gfx1100 / gfx1151) pointer-addressed E8P-I4 F4 read-once C1–C8 lane-union decode GEMV module
//! `e8p_i4_moe_gemv_pm_gfx1201` (`e8p_i4_moe_gemv_{r256_gate_up,r128_down}`, K-E2): symbols, the frozen
//! 96-byte kernarg ABI (identical to `mq_i4_moe_gemv`, `n_experts@88` and a zero word at 92), exact-arch gating, the 1,536-byte table-stage LDS
//! (resource, publication, address bounds), the guard-before-LDS order, the weight/table load census and
//! the read-once structure, the frozen F4 tree and the decode model against the K-E0/K-E1 definitions,
//! exact `u16` word/scale access bounds (zero tail pad), isolation from the MQ4/W3 module, native
//! assembly, and (feature `toolchain`) wait replay, M7, certification and native identity.
use peacemaker_isa::kernels::e8p_i4_moe_gemv::{self, Kind, Spec};
use peacemaker_isa::kernels::mq_i4_moe_gemv;
use peacemaker_isa::kernels::qwen4_moe_sym::e8p::{
    self as k_e1, build_lut256u32, build_mask128u32,
};
use peacemaker_isa::Arch;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1201";
const SYMBOLS: [&str; 2] = ["e8p_i4_moe_gemv_r256_gate_up", "e8p_i4_moe_gemv_r128_down"];
/// Frozen ABI: `(offset, size, name)` in kernarg order.
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
fn spec(kind: Kind) -> Spec {
    Spec { arch: ARCH, kind }
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
        .split_whitespace()
        .next()
        .unwrap()
        .trim_start_matches('v')
        .parse()
        .unwrap_or_else(|_| panic!("not a VGPR: {token}"))
}
fn offset_of(line: &str) -> u32 {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix("offset:"))
        .map_or(0, |v| v.parse().unwrap())
}
fn parse_lit(token: &str) -> u32 {
    token.strip_prefix("0x").map_or_else(
        || token.parse::<i32>().unwrap() as u32,
        |h| u32::from_str_radix(h, 16).unwrap(),
    )
}
/// Instruction lines without the builder's issue hints and waits (not part of the tree).
fn insns(t: &str) -> Vec<&str> {
    t.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty()
                && !mnemonic(l).starts_with("s_delay_alu")
                && !mnemonic(l).starts_with("s_wait_")
                && !mnemonic(l).starts_with("s_nop")
        })
        .collect()
}

/// First workgroup-id SGPR of a gfx11 arm: the DECLARED user SGPRs (`Arch::declared_user_sgprs`; gfx1100's
/// `FeatureUserSGPRInit16Bug` pads user + system to 16: x / y are `s14` / `s15`, with z (the `_t8` entries)
/// `s13` / `s14` / `s15`; gfx1151 has no padding: `s2` / `s3` / `s4`).
fn wg_x(arch: Arch, z: bool) -> u32 {
    u32::from(arch.declared_user_sgprs(2, if z { 3 } else { 2 }))
}

/// SGPRs the plan declares on `arch`.
fn sgprs(arch: Arch) -> u32 {
    u32::from(if arch == Arch::Gfx1100 {
        e8p_i4_moe_gemv::SGPRS_GFX1100
    } else {
        e8p_i4_moe_gemv::SGPRS
    })
}

/// The gfx1151 source of an entry as the gfx1100 arm spells it: the only gfx1100-specific SGPR deltas are the
/// workgroup ids (`s2` / `s3` / `s4` -> the declared-user placement of [`wg_x`]; `s4` is only an id in the `_t8` entries,
/// in the narrow entries it is the unchanged `n_experts` register), the kernarg pointer block `s[8:15]` -> `s[88:95]`,
/// and the descriptor / metadata SGPR counts.
fn gfx1151_sgprs_as_gfx1100(text: &str, z: bool) -> String {
    let x = wg_x(Arch::Gfx1100, z);
    let map = |n: u32| match n {
        2 => x,
        3 => x + 1,
        4 if z => x + 2,
        8..=15 => n + 80,
        _ => n,
    };
    let rename = |line: &str| {
        let b = line.as_bytes();
        let mut out = String::new();
        let mut i = 0;
        while i < b.len() {
            let sgpr_start = b[i] == b's'
                && (i == 0
                    || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'));
            if sgpr_start && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
                let end = (i + 1..b.len())
                    .find(|&j| !b[j].is_ascii_digit())
                    .unwrap_or(b.len());
                out += &format!("s{}", map(line[i + 1..end].parse().unwrap()));
                i = end;
            } else if sgpr_start && i + 1 < b.len() && b[i + 1] == b'[' {
                let close = line[i..].find(']').unwrap() + i;
                let (lo, hi) = line[i + 2..close].split_once(':').unwrap();
                let (lo, hi): (u32, u32) = (lo.parse().unwrap(), hi.parse().unwrap());
                assert_eq!(map(hi) - map(lo), hi - lo, "range {line}");
                out += &format!("s[{}:{}]", map(lo), map(hi));
                i = close + 1;
            } else {
                out.push(b[i] as char);
                i += 1;
            }
        }
        out
    };
    let (user, ceiling) = (
        u32::from(Arch::Gfx1100.declared_user_sgprs(2, if z { 3 } else { 2 })),
        sgprs(Arch::Gfx1100),
    );
    text.lines()
        .map(|l| {
            let l = if l.starts_with('\t') && !l.starts_with("\t.") {
                rename(l)
            } else if l == "\t.amdhsa_user_sgpr_count 2" {
                format!("\t.amdhsa_user_sgpr_count {user}")
            } else if l == format!("\t.amdhsa_next_free_sgpr {}", sgprs(Arch::Gfx1151)) {
                format!("\t.amdhsa_next_free_sgpr {ceiling}")
            } else if l == format!("    .sgpr_count: {}", sgprs(Arch::Gfx1151)) {
                format!("    .sgpr_count: {ceiling}")
            } else {
                l.to_owned()
            };
            format!("{l}\n")
        })
        .collect()
}
fn gate(spec: Spec) -> bool {
    spec.kind == Kind::GateUp
}

#[test]
fn module_exports_are_the_two_frozen_symbols_in_order() {
    let specs = e8p_i4_moe_gemv::module_specs(ARCH);
    assert_eq!(specs, [spec(Kind::GateUp), spec(Kind::Down)]);
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(Spec::module(ARCH), MODULE);
    let (emitted, text, proof) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
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
    assert_eq!(
        (Kind::GateUp.tag(), Kind::Down.tag()),
        ("r256_gate_up", "r128_down")
    );
    assert_eq!((Kind::GateUp.chains(), Kind::Down.chains()), (4, 1));
    assert_eq!(
        (
            spec(Kind::GateUp).group_instances(),
            spec(Kind::Down).group_instances()
        ),
        (7, 1)
    );
    assert_eq!(
        (e8p_i4_moe_gemv::GROUP, e8p_i4_moe_gemv::LANE_ELEMS),
        (128, 4)
    );
}

/// Emission is bit-deterministic, and every architecture other than gfx1201 / gfx1100 / gfx1151 is refused.
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let (a, b) = (
            e8p_i4_moe_gemv::emit(spec).unwrap(),
            e8p_i4_moe_gemv::emit(spec).unwrap(),
        );
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
    }
    let (_, ta, pa) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    let (_, tb, pb) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx942] {
        for spec in e8p_i4_moe_gemv::module_specs(arch) {
            assert!(e8p_i4_moe_gemv::emit(spec).is_err(), "{arch:?} {spec:?}");
        }
        assert!(e8p_i4_moe_gemv::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// The gfx11 arms: the two `_pm_<arch>` symbols, the 96-byte ABI, wave32 and the 1,536-byte table stage, the
/// system-SGPR workgroup ids (`s2` / `s3`, gfx1100: `s14` / `s15` after the 16-SGPR padding; no `ttmp`), gfx11 `s_barrier` (no gfx12 split barrier or wait
/// counters), the `.l` convert shared with gfx1201, and a successful native assembly of the module.
#[test]
fn gfx1100_arm_emits_the_suffixed_symbols_with_gfx11_ids_and_barrier() {
    gfx11_arm(Arch::Gfx1100)
}

#[test]
fn gfx1151_arm_emits_the_suffixed_symbols_with_gfx11_ids_and_barrier() {
    gfx11_arm(Arch::Gfx1151)
}

fn gfx11_arm(arch: Arch) {
    let name = arch.name();
    let symbols = [
        format!("e8p_i4_moe_gemv_r256_gate_up_pm_{name}"),
        format!("e8p_i4_moe_gemv_r128_down_pm_{name}"),
    ];
    assert_eq!(Spec::module(arch), format!("e8p_i4_moe_gemv_pm_{name}"));
    let specs = e8p_i4_moe_gemv::module_specs(arch);
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), symbols);
    assert_eq!(
        spec(Kind::GateUp).symbol(),
        SYMBOLS[0],
        "the gfx1201 symbol is unchanged"
    );
    let (emitted, text, proof) = e8p_i4_moe_gemv::emit_module(arch).unwrap();
    assert_eq!(
        (proof.module.as_str(), proof.arch),
        (format!("e8p_i4_moe_gemv_pm_{name}").as_str(), arch)
    );
    assert_eq!(text.matches(".amdhsa_kernel ").count(), symbols.len());
    for symbol in &symbols {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n"))
                && text.contains(&format!(".globl {symbol}\n")),
            "{symbol}"
        );
    }
    for (spec, e) in specs.iter().zip(&emitted) {
        let symbol = spec.symbol();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 96\n"),
            "{symbol}: kernarg size"
        );
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".group_segment_fixed_size: 1536\n"),
            "{symbol}: static LDS metadata"
        );
        assert!(
            e.s_text.contains(".amdhsa_group_segment_fixed_size 1536\n"),
            "{symbol}: static LDS descriptor"
        );
        let lines: Vec<&str> = insns(&e.s_text);
        assert!(!e.s_text.contains("ttmp"), "{symbol}: ttmp register");
        let x = wg_x(arch, false);
        let (row, uidx) = (
            lines
                .iter()
                .position(|l| *l == format!("s_mov_b32 s30, s{x}")),
            lines
                .iter()
                .position(|l| *l == format!("s_mov_b32 s31, s{}", x + 1)),
        );
        assert!(
            row.is_some() && uidx.is_some(),
            "{symbol}: s{x}/s{} workgroup id prologue",
            x + 1
        );
        assert!(
            e.s_text.contains(&format!(
                ".amdhsa_user_sgpr_count {}\n",
                arch.declared_user_sgprs(2, 2)
            )),
            "{symbol}: declared user SGPR count"
        );
        assert!(
            !lines.contains(&"s_and_b32 s31, ttmp7, 0xffff"),
            "{symbol}: gfx12 y mask"
        );
        assert_eq!(
            count(&e.s_text, "s_barrier"),
            1,
            "{symbol}: one gfx11 workgroup barrier"
        );
        for banned in [
            "s_barrier_signal",
            "s_barrier_wait",
            "s_wait_loadcnt",
            "s_wait_dscnt",
            "s_wait_kmcnt",
            "s_wait_storecnt",
            "s_wait_alu",
            "s_sub_co_i32",
            "s_add_co_i32",
        ] {
            assert_eq!(
                count(&e.s_text, banned),
                0,
                "{symbol}: gfx12 spelling {banned}"
            );
        }
        assert!(count(&e.s_text, "s_waitcnt") > 0, "{symbol}: gfx11 waits");
        assert!(
            e.s_text.contains("v_cvt_f32_f16_e64 v") && e.s_text.contains(".l\n"),
            "{symbol}: .l convert spelling"
        );
        let c = e8p_i4_moe_gemv::contract(*spec, e);
        assert_eq!(c["symbol"], symbol.as_str());
        assert!(
            !c["forbidden"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "s_waitcnt*"),
            "{symbol}: gfx11 needs s_waitcnt"
        );
    }
    let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
    let bundle =
        peacemaker_isa::native::bundle(&elf, arch, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"));
}

/// The single builder-forced hazard guard: gfx1100 has `VALUTransUseHazard` (a transcendental result read by a
/// dependent VALU within the window), gfx1151 does not, so the builder inserts this wait on gfx1100 only.
const TRANS_USE_WAIT: &str = "s_waitcnt_depctr depctr_va_vdst(0)";

/// The gfx1151 arm is the gfx1100 arm with only the arch-forced deltas: every emitted source (each entry and the
/// module), after replacing `gfx1151` with `gfx1100` and removing the gfx1100-only `VALUTransUseHazard` wait
/// (one per entry, the `v_rcp_iflag_f32` -> dependent-VALU guard of the `x_row_div` division), is byte-identical
/// to the gfx1100 text; the shape contracts agree except for the symbol name and that wait's count; the register
/// plan agrees (so the `n_experts@88` load into `s4` and its guard are the same instructions); and neither text
/// leaks the other arch's token. gfx1100's `FeatureUserSGPRInit16Bug` padding is the one further delta: its
/// workgroup ids and kernarg pointer block are renamed (`gfx1151_sgprs_as_gfx1100`: ids `s2` / `s3` -> `s14` / `s15`,
/// `s[8:15]` -> `s[88:95]`, the declared user count and the SGPR ceiling), and nothing else differs.
#[test]
fn gfx1151_arm_is_the_gfx1100_arm_modulo_the_arch_forced_deltas() {
    fn strip(text: &str) -> String {
        text.lines()
            .filter(|l| l.trim() != TRANS_USE_WAIT)
            .map(|l| format!("{l}\n"))
            .collect()
    }
    let (e1100, t1100, p1100) = e8p_i4_moe_gemv::emit_module(Arch::Gfx1100).unwrap();
    let (e1151, t1151, p1151) = e8p_i4_moe_gemv::emit_module(Arch::Gfx1151).unwrap();
    assert!(
        !t1100.contains("gfx1151") && !t1151.contains("gfx1100"),
        "arch tokens must not leak across the modules"
    );
    assert_ne!(t1100, t1151);
    assert_eq!(
        count(&t1151, "s_waitcnt_depctr"),
        0,
        "gfx1151 has no VALUTransUseHazard"
    );
    assert_eq!(
        t1100.lines().filter(|l| l.trim() == TRANS_USE_WAIT).count(),
        e1100.len(),
        "one trans-use wait per gfx1100 entry"
    );
    assert_eq!(
        gfx1151_sgprs_as_gfx1100(&strip(&t1151), false).replace("gfx1151", "gfx1100"),
        strip(&t1100),
        "module source"
    );
    assert_eq!(
        t1100.matches("gfx1100").count(),
        t1151.matches("gfx1151").count(),
        "same number of arch tokens"
    );
    assert_eq!(p1100.kernels.len(), p1151.kernels.len());
    for (a, b) in e1100.iter().zip(&e1151) {
        assert_eq!(
            gfx1151_sgprs_as_gfx1100(&strip(&b.s_text), false).replace("gfx1151", "gfx1100"),
            strip(&a.s_text),
            "{}: entry source",
            b.proof.variant
        );
        assert_eq!(
            (a.shape.next_free_vgpr, a.shape.next_free_sgpr),
            (b.shape.next_free_vgpr, sgprs(Arch::Gfx1100) as u16),
            "{}: register plan",
            b.proof.variant
        );
        assert_eq!(b.shape.next_free_sgpr as u32, sgprs(Arch::Gfx1151));
        assert_eq!(
            a.proof.barriers.len(),
            b.proof.barriers.len(),
            "{}: barriers",
            b.proof.variant
        );
    }
    for (spec, (a, b)) in e8p_i4_moe_gemv::module_specs(Arch::Gfx1100)
        .into_iter()
        .zip(e1100.iter().zip(&e1151))
    {
        let spec1151 = Spec {
            arch: Arch::Gfx1151,
            kind: spec.kind,
        };
        let (mut ca, cb) = (
            e8p_i4_moe_gemv::contract(spec, a),
            e8p_i4_moe_gemv::contract(spec1151, b),
        );
        assert_eq!(
            ca["counts"]["s_waitcnt_depctr"], 1,
            "{:?}: gfx1100 trans-use wait",
            spec.kind
        );
        assert!(
            cb["counts"].get("s_waitcnt_depctr").is_none(),
            "{:?}: gfx1151 has none",
            spec.kind
        );
        ca["counts"]
            .as_object_mut()
            .unwrap()
            .remove("s_waitcnt_depctr");
        assert_eq!(ca["sgpr_max"], sgprs(Arch::Gfx1100));
        assert_eq!(cb["sgpr_max"], sgprs(Arch::Gfx1151));
        ca["sgpr_max"] = cb["sgpr_max"].clone();
        assert_eq!(
            cb.to_string().replace("gfx1151", "gfx1100"),
            ca.to_string(),
            "{:?}: shape contract",
            spec.kind
        );
    }
}

/// The committed images and byproducts under `kernels/` are the emitter's output: the module source (`.s`, gfx11
/// arms), the natively linked + bundled `.hxaco` (every arm), and the shape contracts (gfx11 arms) re-emit
/// byte-for-byte, so a change of the emitter that moves any committed arch shows up here.
#[test]
fn committed_images_and_byproducts_are_byte_identical_to_the_emission() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    for arch in [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151] {
        let module = Spec::module(arch);
        // The committed module is the CLI row image: the two narrow entries
        // followed by their two fixed-row-tile `_t8` twins (never the narrow
        // subset `emit_module` alone returns).
        let (emitted, text, proof) = {
            let mut entries = Vec::new();
            for tiled in [false, true] {
                for spec in e8p_i4_moe_gemv::module_specs(arch) {
                    entries.push(if tiled {
                        e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap()
                    } else {
                        e8p_i4_moe_gemv::emit(spec).unwrap()
                    });
                }
            }
            let (text, proof) =
                peacemaker_isa::kernels::iu4_gemm::module(&entries, &module).unwrap();
            (entries, text, proof)
        };
        let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
        let bundle =
            peacemaker_isa::native::bundle(&elf, arch, peacemaker_isa::native::DEFAULT_HOST_TARGET);
        let committed = std::fs::read(dir.join(format!("{module}.hxaco")))
            .unwrap_or_else(|e| panic!("{module}.hxaco: {e}"));
        assert!(
            committed == bundle,
            "{module}.hxaco differs from the emitted bundle"
        );
        if arch.gfx12() {
            continue;
        }
        let source = std::fs::read_to_string(dir.join(format!("{module}.s")))
            .unwrap_or_else(|e| panic!("{module}.s: {e}"));
        assert!(source == text, "{module}.s differs from the emitted source");
        let committed_proof: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(format!("{module}.proof.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(
            committed_proof,
            serde_json::to_value(&proof).unwrap(),
            "{module}.proof.json"
        );
        for (spec, e) in e8p_i4_moe_gemv::module_specs(arch)
            .into_iter()
            .zip(&emitted)
        {
            let file = dir.join(format!(
                "e8p_i4_moe_gemv.{}.{}.contract.json",
                arch.name(),
                spec.tag()
            ));
            let committed: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display())),
            )
            .unwrap();
            assert_eq!(
                committed,
                e8p_i4_moe_gemv::contract(spec, e),
                "{}",
                file.display()
            );
        }
    }
}

/// The 88-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the first
/// loads, and the typed ownership of the registers that carry it.
#[test]
fn kernarg_abi_matches_the_frozen_96_byte_layout() {
    assert_eq!(
        (
            e8p_i4_moe_gemv::KERNARG_BYTES,
            e8p_i4_moe_gemv::WORKGROUP_SIZE,
            e8p_i4_moe_gemv::MAX_SLOTS as usize
        ),
        (96, 32, SLOTS)
    );
    // Byte-identical to the incumbent's ABI.
    assert_eq!(
        (
            e8p_i4_moe_gemv::KERNARG_BYTES,
            e8p_i4_moe_gemv::WORKGROUP_SIZE,
            e8p_i4_moe_gemv::MAX_SLOTS
        ),
        (
            mq_i4_moe_gemv::KERNARG_BYTES,
            mq_i4_moe_gemv::WORKGROUP_SIZE,
            mq_i4_moe_gemv::MAX_SLOTS
        )
    );
    let incumbent = mq_i4_moe_gemv::emit(mq_i4_moe_gemv::Spec::narrow(
        ARCH,
        mq_i4_moe_gemv::Codec::Mq3I4,
        mq_i4_moe_gemv::Kind::GateUp,
    ))
    .unwrap();
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let e = e8p_i4_moe_gemv::emit(spec).unwrap();
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
            field(&e.s_text, ".offset:"),
            field(&incumbent.s_text, ".offset:"),
            "{symbol}: incumbent offsets"
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
            e8p_i4_moe_gemv::KERNARG_BYTES,
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
        // Five scalar loads covering offsets 0..92, no over-read: the 88-byte block of the incumbent, then `n_experts`.
        for load in [
            "s_load_b256 s[8:15], s[0:1], 0x0",
            "s_load_b256 s[16:23], s[0:1], 0x20",
            "s_load_b128 s[24:27], s[0:1], 0x40",
            "s_load_b64 s[28:29], s[0:1], 0x50",
            "s_load_b32 s4, s[0:1], 0x58",
        ] {
            assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
        }
        let plan: Vec<&str> = e.proof.reg_plan.iter().map(|r| r.name.as_ref()).collect();
        for owner in [
            "kernel_arguments_ptrs",
            "kernel_arguments_x_y_m_k",
            "kernel_arguments_u_c_div_src",
            "kernel_arguments_strides",
            "kernel_arguments_n_experts",
            "x_descriptor",
            "y_select",
            "accumulators",
            "x_values",
            "weights",
            "slot_x_offsets",
            "slot_y_addresses",
            "shuffle_masks",
            "shuffle_addresses",
            "word",
            "scale",
            "lut_address",
            "mask_address",
            "lut_value",
            "mask_value",
            "shift_addend",
            "operand",
            "half_shift",
            "lds_store_address",
            "table_literals",
        ] {
            assert!(
                plan.contains(&owner),
                "{symbol}: reg plan misses {owner}: {plan:?}"
            );
        }
    }
}

/// Whether an instruction line names scalar register `n` (a single `sN` operand or inside an `s[a:b]` range).
fn touches_sgpr(line: &str, n: u32) -> bool {
    let ops = line.trim().split_once(' ').map_or("", |(_, rest)| rest);
    ops.split(", ").any(|o| {
        let o = o.trim();
        match o
            .strip_prefix("s[")
            .and_then(|r| r.strip_suffix(']'))
            .and_then(|r| r.split_once(':'))
        {
            Some((a, b)) => (a.parse::<u32>().unwrap()..=b.parse::<u32>().unwrap()).contains(&n),
            None => o == format!("s{n}"),
        }
    })
}

/// The `n_experts` guard on every architecture: the word is loaded once into the spare `s4` (no other instruction
/// defines or reads it but the compare), the unsigned compare `s_cmp_ge_u32 s34, s4` of the expert id plus the
/// existing exit branch sit after the offsets guard and before `expert_id * 8`, the expert-table `s_load_b64` and the
/// LDS stage / barrier, and the SGPR ceiling is the unchanged 88 on gfx1151 / gfx1201 (gfx1100 relocates the kernarg
/// pointer block past the padded workgroup-id SGPRs: 96; `s4` stays the spare, no id lives there).
#[test]
fn n_experts_guard_precedes_the_table_index_and_the_stage_on_both_arches() {
    for arch in [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151] {
        for spec in e8p_i4_moe_gemv::module_specs(arch) {
            let e = e8p_i4_moe_gemv::emit(spec).unwrap();
            let symbol = spec.symbol();
            let lines = insns(&e.s_text);
            let exit = format!(".Le8pi4moe_{}_exit", spec.tag());
            assert_eq!(
                (u32::from(e.shape.next_free_sgpr), sgprs(arch)),
                (sgprs(arch), sgprs(arch)),
                "{symbol}: sgpr ceiling"
            );
            let touching: Vec<usize> = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| touches_sgpr(l, 4))
                .map(|(i, _)| i)
                .collect();
            let load = lines
                .iter()
                .position(|l| *l == "s_load_b32 s4, s[0:1], 0x58")
                .unwrap_or_else(|| panic!("{symbol}: n_experts load"));
            let cmp = lines
                .iter()
                .position(|l| *l == "s_cmp_ge_u32 s34, s4")
                .unwrap_or_else(|| panic!("{symbol}: n_experts compare"));
            assert_eq!(
                touching,
                [load, cmp],
                "{symbol}: s4 is touched only by the load and the compare"
            );
            assert_eq!(
                lines[cmp + 1],
                format!("s_cbranch_scc1 {exit}"),
                "{symbol}: compare feeds the existing exit"
            );
            let offsets_guard = lines
                .iter()
                .position(|l| l.starts_with("s_cmp_gt_u32 s33, s51"))
                .unwrap();
            assert_eq!(
                lines[offsets_guard + 1],
                format!("s_cbranch_scc1 {exit}"),
                "{symbol}: offsets guard"
            );
            assert!(
                offsets_guard < cmp,
                "{symbol}: n_experts guard follows the offsets guard"
            );
            let scale_eid = lines
                .iter()
                .position(|l| *l == "s_lshl_b32 s51, s34, 3")
                .unwrap();
            let table_load = lines
                .iter()
                .position(|l| l.starts_with("s_load_b64 s[38:39], s[36:37]"))
                .unwrap();
            let first_ds = lines
                .iter()
                .position(|l| mnemonic(l).starts_with("ds_"))
                .unwrap();
            let barrier = lines
                .iter()
                .position(|l| mnemonic(l).starts_with("s_barrier"))
                .unwrap();
            assert!(
                cmp + 1 < scale_eid && scale_eid < table_load,
                "{symbol}: guard before expert_id*8 and the expert-table load"
            );
            assert!(
                table_load < first_ds && first_ds < barrier,
                "{symbol}: table load / stage / barrier order"
            );
            // The loaded word is complete before the compare: a wait sits between the load and the compare.
            let raw: Vec<&str> = e.s_text.lines().map(str::trim).collect();
            let (l, c) = (
                raw.iter()
                    .position(|l| *l == "s_load_b32 s4, s[0:1], 0x58")
                    .unwrap(),
                raw.iter()
                    .position(|l| *l == "s_cmp_ge_u32 s34, s4")
                    .unwrap(),
            );
            assert!(
                raw[l..c]
                    .iter()
                    .any(|t| t.starts_with("s_waitcnt") || t.starts_with("s_wait_kmcnt")),
                "{symbol}: no load-completion wait before the compare"
            );
            // Exactly one guard branch to the exit sits between the offsets guard and the table index.
            let between = lines[offsets_guard + 2..scale_eid]
                .iter()
                .filter(|l| mnemonic(l).starts_with("s_cbranch"))
                .count();
            assert_eq!(
                between, 1,
                "{symbol}: one branch between the offsets guard and expert_id*8"
            );
        }
    }
}

/// Evaluate an emitted scalar compare `s_cmp_<cond>_<ty> a, b` (`a` the routed expert id, `b` `n_experts`) on
/// concrete values, honouring the instruction's own signedness. Only the `ge`/`lt`/`gt`/`le` conditions of the
/// `u32`/`i32` forms the kernels can emit are modelled.
fn scalar_compare(line: &str, a: u32, b: u32) -> bool {
    let name = mnemonic(line);
    let (cond, ty) = name
        .strip_prefix("s_cmp_")
        .and_then(|r| r.split_once('_'))
        .unwrap_or_else(|| panic!("not a scalar compare: {line}"));
    let (x, y) = match ty {
        "u32" => (i64::from(a), i64::from(b)),
        "i32" => (i64::from(a as i32), i64::from(b as i32)),
        other => panic!("unmodelled compare type {other}: {line}"),
    };
    match cond {
        "ge" => x >= y,
        "gt" => x > y,
        "le" => x <= y,
        "lt" => x < y,
        other => panic!("unmodelled compare condition {other}: {line}"),
    }
}

/// CPU model of the emitted `n_experts` guard (the exact compare instruction the builder placed, evaluated on
/// concrete values, plus the emitted order): for every extent `n`, the invalid ids `-1` (the router's non-finite
/// marker), `i32::MIN`, `n` and `n + 1` take the exit branch and `n - 1` (and `0`, when `n > 0`) falls through; a
/// signed compare would let `-1` / `i32::MIN` through, so the model fails any signed spelling. Before the guard the
/// entry block issues no vector / global / buffer / LDS / scratch memory instruction and no store at all, so an
/// invalid id never dereferences the weight table or an expert tensor and never writes `Y0` / `Y1`.
#[test]
fn n_experts_guard_cpu_model_rejects_minus_one_min_n_and_n_plus_one_and_passes_n_minus_one() {
    for arch in [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151] {
        for spec in e8p_i4_moe_gemv::module_specs(arch) {
            let e = e8p_i4_moe_gemv::emit(spec).unwrap();
            let symbol = spec.symbol();
            let lines = insns(&e.s_text);
            let cmp = lines
                .iter()
                .position(|l| *l == "s_cmp_ge_u32 s34, s4")
                .unwrap_or_else(|| panic!("{symbol}: n_experts compare"));
            assert_eq!(
                mnemonic(lines[cmp + 1]),
                "s_cbranch_scc1",
                "{symbol}: SCC set (id >= n) takes the exit"
            );
            for n in [1u32, 2, 5, 512, i32::MAX as u32] {
                for id in [u32::MAX, i32::MIN as u32, n, n + 1] {
                    assert!(
                        scalar_compare(lines[cmp], id, n),
                        "{symbol}: id {id:#x} against n_experts {n} must exit"
                    );
                }
                for id in [n - 1, 0] {
                    assert!(
                        !scalar_compare(lines[cmp], id, n),
                        "{symbol}: id {id:#x} against n_experts {n} must not exit"
                    );
                }
            }
            // n_experts == 0: every id, the sentinel included, exits.
            for id in [0u32, 1, u32::MAX, i32::MIN as u32] {
                assert!(
                    scalar_compare(lines[cmp], id, 0),
                    "{symbol}: n = 0, id {id:#x}"
                );
            }
            let forbidden = |m: &str| {
                m.starts_with("global_")
                    || m.starts_with("buffer_")
                    || m.starts_with("flat_")
                    || m.starts_with("scratch_")
                    || m.starts_with("ds_")
                    || m.starts_with("s_store")
                    || m.starts_with("s_buffer_store")
            };
            assert!(
                lines[..cmp].iter().all(|l| !forbidden(mnemonic(l))),
                "{symbol}: a memory instruction precedes the n_experts guard"
            );
        }
    }
}

/// One CPU launch of the grid `[M, U]` (one slot per expert descriptor `u`, slot `u` reading `xs[u]`): a workgroup
/// `(row, u)` is dropped when the emitted `n_experts` compare `guard` fires on `ids[u]`, otherwise it stores the
/// `kernel_row` model word to `y[u][row]`. A passing id indexes the weight table with `get(..).expect(..)`, so a
/// dereference past the `n`-entry table is a test failure, not a silent read.
#[allow(clippy::too_many_arguments)]
fn cpu_guarded_launch(
    guard: &str,
    n: u32,
    ids: &[u32],
    table: &[(Vec<u8>, Vec<u16>)],
    xs: &[Vec<f32>],
    four_chain: bool,
    k: usize,
    m: usize,
) -> Vec<Vec<u32>> {
    assert_eq!(
        table.len(),
        n as usize,
        "the table has exactly n_experts entries"
    );
    let mut y = vec![vec![0xa5a5_a5a5u32; m]; ids.len()];
    for (u, &id) in ids.iter().enumerate() {
        for row in 0..m {
            if scalar_compare(guard, id, n) {
                continue;
            }
            let (plane, scales) = table
                .get(id as usize)
                .expect("a passing expert id must index the n-entry table");
            y[u][row] = kernel_row(four_chain, k, plane, scales, row, &xs[u..u + 1])[0].to_bits();
        }
    }
    y
}

/// CPU NaN-containment fixture on every K1 arch (gfx1201, gfx1100, gfx1151), no emulator: the router's fail-closed
/// output for a non-finite row is expert id `-1` with weight `+0`, and the activation rows of the failed lanes
/// may be NaN. Launching the grid with the emitted guard compare over the ids `[n-1, -1, i32::MIN, n, n+1, 0]`
/// (slots 1..=4 carry all-NaN activations): every invalid-id slot keeps its sentinel (no store, no table
/// dereference), the two valid slots (`n-1` and `0`) are finite and bit-equal to the launch that never saw the
/// invalid descriptors, and the same valid expert fed a NaN row does produce NaN (the fixture has teeth: the
/// guard, not the data, is what keeps the NaN out).
#[test]
fn n_experts_nan_containment_cpu_fixture_on_every_arch() {
    for arch in [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151] {
        for spec in e8p_i4_moe_gemv::module_specs(arch) {
            let e = e8p_i4_moe_gemv::emit(spec).unwrap();
            let symbol = spec.symbol();
            let lines = insns(&e.s_text);
            let guard = *lines
                .iter()
                .find(|l| **l == "s_cmp_ge_u32 s34, s4")
                .unwrap_or_else(|| panic!("{symbol}: n_experts compare"));
            let four_chain = spec.kind.chains() == 4;
            let (k, m, groups) = (256usize, 2usize, 2usize);
            let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ u64::from(arch as u8 as u32 + 1));
            let n = 5u32;
            let table: Vec<(Vec<u8>, Vec<u16>)> = (0..n)
                .map(|_| {
                    let words: Vec<u16> = (0..m * k / 8)
                        .map(|_| {
                            ((rng.next() & 0xff)
                                | ((rng.next() & 0x7f) << 8)
                                | ((rng.next() & 1) << 15)) as u16
                        })
                        .collect();
                    let scales: Vec<u16> = (0..m * groups)
                        .map(|_| 0x0400 + rng.below(0x7bff - 0x0400 + 1) as u16)
                        .collect();
                    (words.iter().flat_map(|w| w.to_le_bytes()).collect(), scales)
                })
                .collect();
            let finite = |rng: &mut Rng| -> Vec<f32> { (0..k).map(|_| rng.f32()).collect() };
            let nan_row = vec![f32::NAN; k];
            let (x_hi, x_lo) = (finite(&mut rng), finite(&mut rng));
            let ids = [n - 1, u32::MAX, i32::MIN as u32, n, n + 1, 0];
            let xs = [
                x_hi.clone(),
                nan_row.clone(),
                nan_row.clone(),
                nan_row.clone(),
                nan_row.clone(),
                x_lo.clone(),
            ];
            let y = cpu_guarded_launch(guard, n, &ids, &table, &xs, four_chain, k, m);
            for u in [1usize, 2, 3, 4] {
                assert!(
                    y[u].iter().all(|&w| w == 0xa5a5_a5a5),
                    "{symbol}: id {:#x} stored through the guard",
                    ids[u]
                );
            }
            let baseline = cpu_guarded_launch(
                guard,
                n,
                &[n - 1, 0],
                &table,
                &[x_hi.clone(), x_lo.clone()],
                four_chain,
                k,
                m,
            );
            for (u, b) in [(0usize, 0usize), (5, 1)] {
                assert_eq!(
                    y[u], baseline[b],
                    "{symbol}: valid slot {u} changed by the invalid ones"
                );
                assert!(
                    y[u].iter()
                        .all(|&w| f32::from_bits(w).is_finite() && w != 0xa5a5_a5a5),
                    "{symbol}: valid slot {u} not a finite written word"
                );
            }
            // Teeth: the unguarded model on a valid expert with a NaN row yields NaN.
            let (plane, scales) = &table[(n - 1) as usize];
            let poisoned = kernel_row(four_chain, k, plane, scales, 0, &[nan_row])[0];
            assert!(
                poisoned.is_nan(),
                "{symbol}: a NaN activation must poison an unguarded row"
            );
            // n_experts == 0: the whole grid is dropped.
            let none =
                cpu_guarded_launch(guard, 0, &[0, 1, u32::MAX], &[], &xs[..3], four_chain, k, m);
            assert!(
                none.iter().flatten().all(|&w| w == 0xa5a5_a5a5),
                "{symbol}: n = 0 stored"
            );
        }
    }
}

/// Resources: wave32, one wave per workgroup, no scratch/spills/private/dynamic LDS, exactly the
/// 1,536-byte static table stage (the K-E1 stage size), one workgroup barrier that publishes it.
#[test]
fn resources_are_wave32_with_exactly_the_1536_byte_stage() {
    assert_eq!(
        (
            e8p_i4_moe_gemv::STAGE_BYTES,
            e8p_i4_moe_gemv::LUT_BYTES,
            e8p_i4_moe_gemv::MASK_BYTES
        ),
        (1536, 1024, 512)
    );
    assert_eq!(e8p_i4_moe_gemv::STAGE_BYTES, k_e1::STAGE_BYTES);
    let (emitted, text, _) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    for (e, symbol) in emitted.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
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
            e.s_text.contains(".group_segment_fixed_size: 1536\n"),
            "{symbol}: static LDS metadata"
        );
        assert!(
            e.s_text.contains(".amdhsa_group_segment_fixed_size 1536\n"),
            "{symbol}: static LDS descriptor"
        );
        assert!(
            e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{symbol}: grid y is the expert"
        );
        assert_eq!(
            e.proof.barriers.len(),
            1,
            "{symbol}: the one table-publication barrier"
        );
        assert_eq!(count(&e.s_text, "s_barrier_signal"), 1, "{symbol}");
        assert_eq!(count(&e.s_text, "s_barrier_wait"), 1, "{symbol}");
        assert_eq!(e.proof.lds_slots.len(), 1, "{symbol}: one typed LDS slot");
        let slot = &e.proof.lds_slots[0];
        assert_eq!(
            (slot.name.as_str(), slot.base, slot.len),
            ("e8p_tables", 0, 1536),
            "{symbol}"
        );
        assert!(
            !e.s_text
                .lines()
                .any(|l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")),
            "{symbol}: scratch/flat"
        );
        for forbidden in [
            "ds_swizzle",
            "ds_read",
            "ds_write",
            "ds_load_b64",
            "ds_load_b128",
            "ds_store_b64",
            "v_bcnt",
            "v_wmma",
            "global_atomic",
            "buffer_atomic",
        ] {
            let hit = e
                .s_text
                .lines()
                .find(|l| mnemonic(l).starts_with(forbidden));
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
}

/// Honest register ceilings: eight slots of accumulators (four chains per slot for gate) and X
/// registers, whole-granule VGPR reservation, the incumbent's 80 SGPRs plus eight literal staging SGPRs,
/// the contract equals the plan (VGPR 112 gate / 88 down, SGPR 88).
#[test]
fn register_ceilings_reflect_c8_slot_state() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let e = e8p_i4_moe_gemv::emit(spec).unwrap();
        let symbol = spec.symbol();
        let chains = u16::from(spec.kind.chains());
        let floor = SLOTS as u16 * (chains + 4) + 2 * SLOTS as u16;
        assert!(
            e.shape.next_free_vgpr >= floor
                && e.shape.next_free_vgpr % 8 == 0
                && e.shape.next_free_vgpr <= 256,
            "{symbol}: vgpr {}",
            e.shape.next_free_vgpr
        );
        assert_eq!(
            e.shape.next_free_vgpr,
            if gate(spec) { 112 } else { 88 },
            "{symbol}: vgpr"
        );
        assert_eq!(
            (e.shape.next_free_sgpr, e8p_i4_moe_gemv::SGPRS),
            (88, 88),
            "{symbol}: sgpr"
        );
        assert!(
            e.s_text
                .contains(&format!(".vgpr_count: {}\n", e.shape.next_free_vgpr)),
            "{symbol}: metadata vgpr count"
        );
        let c = e8p_i4_moe_gemv::contract(spec, &e);
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

/// Guards and the LDS stage order: the six early-exit branches all precede the first DS access and the
/// barrier (a rejected workgroup returns before any LDS traffic); the LDS store address is derived from
/// the work-item id before the first branch (the form `pm_check::lds_bounds` evaluates); the table reads
/// use immediate-masked addresses; every table access stays inside the 1,536-byte allocation; the LUT
/// and MASK literals equal the K-E1 tables.
#[test]
fn guards_precede_the_lds_stage_and_every_lds_access_is_bounded() {
    let lut = build_lut256u32();
    let mask = build_mask128u32();
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let t = e8p_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let lines = insns(&t);
        let exit = format!(".Le8pi4moe_{}_exit", spec.tag());
        let first_branch = lines
            .iter()
            .position(|l| mnemonic(l).starts_with("s_cbranch"))
            .unwrap();
        let first_ds = lines
            .iter()
            .position(|l| mnemonic(l).starts_with("ds_store"))
            .unwrap();
        let barrier = lines
            .iter()
            .position(|l| mnemonic(l) == "s_barrier_signal")
            .unwrap();
        let guards: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "s_cbranch_scc1" && l.ends_with(&exit))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            guards.len(),
            6,
            "{symbol}: row, expert, empty, >8 slots, offsets>10C, expert id >= n_experts guards"
        );
        assert!(
            guards.iter().all(|&g| g < first_ds && g < barrier),
            "{symbol}: a guard sits after the LDS stage"
        );
        assert!(
            lines[..first_branch]
                .iter()
                .all(|l| !mnemonic(l).starts_with("ds_")),
            "{symbol}: DS before the first branch"
        );
        // Store address `4 * tid`, derived in the entry block before the first branch.
        let derive = lines[..first_branch]
            .iter()
            .find(|l| {
                mnemonic(l) == "v_lshlrev_b32_e32" && operand(l, 1) == "2" && operand(l, 2) == "v0"
            })
            .expect("4*tid before the first branch");
        let addr = operand(derive, 0);
        let stores: Vec<&&str> = lines
            .iter()
            .filter(|l| mnemonic(l).starts_with("ds_store"))
            .collect();
        assert_eq!(
            stores.len(),
            12,
            "{symbol}: 4 MASK stores + 8 LUT chunk stores"
        );
        assert!(
            stores.iter().all(|l| mnemonic(l) == "ds_store_b32"
                && l.split_whitespace().nth(1).unwrap().trim_end_matches(',') == addr),
            "{symbol}: store addresses"
        );
        // MASK entries `tid + 32j` first (offset 1024 + 128j), then LUT chunk c (offset 128c).
        let offsets: Vec<u32> = stores.iter().map(|l| offset_of(l)).collect();
        assert_eq!(
            offsets,
            [1024, 1152, 1280, 1408, 0, 128, 256, 384, 512, 640, 768, 896],
            "{symbol}: store offsets"
        );
        // Max store end: lane 31 -> 4*31 + offset + 4.
        assert_eq!(
            offsets.iter().map(|o| 4 * 31 + o + 4).max().unwrap(),
            e8p_i4_moe_gemv::STAGE_BYTES,
            "{symbol}"
        );
        // The LUT literals in staging order are the K-E1 LUT, lane by lane (chunk c, batch b, lane 8b + q).
        let literals: Vec<u32> = lines
            .iter()
            .filter_map(|l| {
                let r: u32 = operand(l, 0).strip_prefix('s')?.parse().ok()?;
                (mnemonic(l) == "s_mov_b32" && (80..88).contains(&r))
                    .then(|| parse_lit(&operand(l, 1)))
            })
            .collect();
        assert_eq!(literals, lut, "{symbol}: LUT literals");
        let lanes: Vec<u32> = lines
            .iter()
            .filter(|l| mnemonic(l) == "v_writelane_b32")
            .map(|l| operand(l, 2).parse().unwrap())
            .collect();
        assert_eq!(
            lanes,
            (0..256u32).map(|i| i % 32).collect::<Vec<_>>(),
            "{symbol}: writelane lanes"
        );
        // Table reads: masked address defined by the immediately preceding `v_and_b32_e32 vR, M, vX`.
        let mut last_mask = std::collections::HashMap::new();
        let (mut lut_reads, mut mask_reads, mut max_end) = (0, 0, 0u32);
        for l in &lines {
            if mnemonic(l) == "v_and_b32_e32" {
                let m = operand(l, 1);
                if m == "0x3fc" || m == "0x1fc" {
                    last_mask.insert(operand(l, 0), parse_lit(&m));
                }
            } else if mnemonic(l) == "ds_load_b32" {
                let addr_reg = operand(l, 1).split_whitespace().next().unwrap().to_owned();
                let m = *last_mask
                    .get(&addr_reg)
                    .unwrap_or_else(|| panic!("{symbol}: unmasked table address {l}"));
                let off = offset_of(l);
                let end = m + off + 4;
                match (m, off) {
                    (0x3fc, 0) => lut_reads += 1,
                    (0x1fc, 1024) => mask_reads += 1,
                    other => panic!("{symbol}: table read {other:?}"),
                }
                assert!(
                    end <= e8p_i4_moe_gemv::STAGE_BYTES,
                    "{symbol}: {l} ends at {end}"
                );
                max_end = max_end.max(end);
            }
        }
        let inst = spec.group_instances() as usize;
        assert_eq!(
            (lut_reads, mask_reads, max_end),
            (inst, inst, 1536),
            "{symbol}: table reads"
        );
        // Both tables fit their regions: the LUT index mask spans 256 words, the MASK index mask 128.
        assert_eq!((lut.len() * 4, mask.len() * 4), (1024, 512));
    }
}

/// Table identity: the stage writes the K-E1 tables, and the arithmetic MASK generator the stage emits
/// (xor/shift parity, no `v_bcnt_u32_b32`) equals `build_mask128u32` for every entry.
#[test]
fn staged_tables_are_the_k_e1_tables() {
    let (lut, mask) = e8p_i4_moe_gemv::tables();
    assert_eq!((lut, mask), (build_lut256u32(), build_mask128u32()));
    for s in 0..128u32 {
        assert_eq!(
            e8p_i4_moe_gemv::stage_mask_model(s),
            mask[s as usize],
            "mask[{s}]"
        );
    }
    // Every non-lane-7 nibble of a MASK is 0 or 0xF and lane 7 is the parity of popcount(sign7).
    for (s, &m) in mask.iter().enumerate() {
        assert_eq!(
            m >> 28,
            if (s as u32).count_ones() & 1 == 1 {
                0xf
            } else {
                0
            }
        );
    }
}

/// Load census: per group instance exactly one exact `u16` word load, one exact `u16` scale load and two
/// table `ds_load_b32` (no `u8`/`b32` weight load, no tail over-read), plus the single slot-id load; X goes
/// through the raw descriptor; the arithmetic census of the frozen tree and the decode network.
#[test]
fn load_and_instruction_census_matches_the_frozen_dag() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let t = e8p_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let gate = gate(spec);
        assert_eq!(
            count(&t, "global_load_u16"),
            2 * inst,
            "{symbol}: word + scale u16 loads"
        );
        assert_eq!(count(&t, "global_load_u8"), 0, "{symbol}: no byte loads");
        assert_eq!(
            count(&t, "global_load_b32"),
            1,
            "{symbol}: only the slot-id dword load"
        );
        assert_eq!(
            count(&t, "ds_load_b32"),
            2 * inst,
            "{symbol}: LUT + MASK lookups"
        );
        assert_eq!(count(&t, "ds_store_b32"), 12, "{symbol}");
        assert_eq!(
            count(&t, "buffer_load_b128"),
            inst * SLOTS,
            "{symbol}: X loads"
        );
        assert_eq!(count(&t, "global_store_b32"), SLOTS, "{symbol}: stores");
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
        assert_eq!(count(&t, "v_writelane_b32"), 256, "{symbol}: LUT staging");
        assert_eq!(
            count(&t, "v_cmpx_eq_u32_e32"),
            1,
            "{symbol}: lane-0 store mask"
        );
        // Decode network per group: bfe_i32 shift bit + 4 element extracts, add3, one xor, converts, multiplies.
        assert_eq!(
            count(&t, "v_bfe_i32"),
            5 * inst,
            "{symbol}: shift bit + four signed nibble extracts"
        );
        assert_eq!(count(&t, "v_add3_u32"), inst, "{symbol}: x + y + n");
        assert_eq!(
            count(&t, "v_xor_b32_e32"),
            12 + inst,
            "{symbol}: stage parity xors + one table xor per group"
        );
        assert_eq!(
            count(&t, "v_bfi_b32"),
            4,
            "{symbol}: stage MASK sign7|parity bit"
        );
        assert_eq!(
            count(&t, "v_lshl_or_b32"),
            12,
            "{symbol}: stage MASK nibble spread only"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_i32_e32"),
            4 * inst,
            "{symbol}: odd-integer converts"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_f16_e64"),
            inst,
            "{symbol}: exact f16 scale convert"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_u32_e32"),
            2,
            "{symbol}: only the slot divide"
        );
        assert_eq!(count(&t, "v_fma_f32"), 0, "{symbol}: no affine fma");
        assert_eq!(
            count(&t, "v_mul_f32_e32"),
            13 * inst + 1,
            "{symbol}: 4 weights + 8 first dot terms + scale quarter per group (+ divide)"
        );
        assert_eq!(
            count(&t, "v_fmac_f32_e32"),
            3 * SLOTS * inst,
            "{symbol}: fused steps"
        );
        let fold = if gate { 3 } else { 0 };
        assert_eq!(
            count(&t, "v_add_f32_e32"),
            SLOTS * inst + SLOTS * (fold + 5),
            "{symbol}: group-partial, fold and shuffle adds"
        );
        assert!(count(&t, "v_and_b32_e32") >= 3 * inst, "{symbol}");
        // Decode constants: masked table addresses, shift addend, ones.
        for (needle, n) in [
            (", 0x3fc, ", inst),
            (", 0x1fc, ", inst),
            (", 0xeeeeeeef, ", inst),
        ] {
            assert_eq!(
                t.lines()
                    .filter(|l| mnemonic(l).starts_with("v_and_b32") && l.contains(needle))
                    .count(),
                n,
                "{symbol}: {needle}"
            );
        }
        assert_eq!(
            t.lines()
                .filter(|l| mnemonic(l).starts_with("v_and_b32") && l.contains(", 0x11111111, "))
                .count(),
            inst + 4,
            "{symbol}: ones mask"
        );
        // dq = f32(d) * 0.25 once per group (literal bits), then w = dq * float(v): no FP16 d/4.
        assert_eq!(
            t.lines()
                .filter(|l| mnemonic(l) == "v_mul_f32_e32" && operand(l, 1) == "0x3e800000")
                .count(),
            inst,
            "{symbol}: exact quarter multiply"
        );
        for forbidden in [
            "v_cvt_f16_f32",
            "v_mul_f16",
            "v_fma_mix_f32",
            "v_bcnt_u32_b32",
            "v_wmma",
            "flat_",
            "scratch_",
            "global_atomic",
            "buffer_atomic",
        ] {
            let hit = t.lines().find(|l| mnemonic(l).starts_with(forbidden));
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
    }
}

/// The frozen F4 tree per symbol: `p = RN(w0*x0)`, `fma(w_j, x_j, p)` j = 1..3 on adjacent registers, the
/// partial added separately; shuffle offsets 16, 8, 4, 2, 1 with out-of-range lanes unchanged; gate fold
/// `(a0+a1)+(a2+a3)` vs the down single chain; one lane-0 store per slot.
#[test]
fn f4_tree_matches_the_frozen_dag() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let t = e8p_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let lines = insns(&t);
        let first_fmac = lines
            .iter()
            .position(|l| mnemonic(l) == "v_fmac_f32_e32")
            .unwrap();
        let mul = lines[first_fmac - 1];
        assert_eq!(mnemonic(mul), "v_mul_f32_e32", "{symbol}: first dot term");
        let (dot, wf, xf) = (
            vreg(&operand(mul, 0)),
            vreg(&operand(mul, 1)),
            vreg(&operand(mul, 2)),
        );
        for j in 1..4u32 {
            let l = lines[first_fmac + j as usize - 1];
            assert_eq!(mnemonic(l), "v_fmac_f32_e32", "{symbol}: fused step {j}");
            assert_eq!(
                (
                    vreg(&operand(l, 0)),
                    vreg(&operand(l, 1)),
                    vreg(&operand(l, 2))
                ),
                (dot, wf + j, xf + j),
                "{symbol}: fused step {j} operands"
            );
        }
        let add = lines[first_fmac + 3];
        assert_eq!(
            (
                mnemonic(add),
                vreg(&operand(add, 1)),
                vreg(&operand(add, 2))
            ),
            ("v_add_f32_e32", vreg(&operand(add, 0)), dot),
            "{symbol}: group partial add"
        );
        let offsets: Vec<u32> = lines
            .iter()
            .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0")
            .map(|l| operand(l, 2).parse().unwrap())
            .collect();
        assert_eq!(offsets, [16, 8, 4, 2, 1], "{symbol}: shuffle offsets");
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
        // Four chains per slot for gate: the accumulators of consecutive groups differ; down uses one.
        let accs: Vec<String> = lines
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_add_f32_e32"
                    && (vreg(&operand(l, 2)) == dot || vreg(&operand(l, 2)) == dot + 1)
                    && operand(l, 0) == operand(l, 1)
            })
            .map(|l| operand(l, 0))
            .collect();
        let slot0: Vec<&String> = accs.iter().step_by(SLOTS).collect();
        assert_eq!(slot0.len(), spec.group_instances() as usize, "{symbol}");
        let distinct = slot0.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(distinct, if gate(spec) { 4 } else { 1 }, "{symbol}: chains");
        if gate(spec) {
            for g in 0..7usize {
                assert_eq!(
                    slot0[g],
                    slot0[g & 3],
                    "{symbol}: group {g} feeds chain {}",
                    g & 3
                );
            }
            assert_eq!(
                slot0
                    .iter()
                    .take(4)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                4,
                "{symbol}: first four groups use distinct chains"
            );
        }
    }
}

/// Read-once C8: every weight-class and table load of a group sits before that group's first slot branch
/// and never inside one; the number of weight-class loads does not depend on the slot count, and every slot
/// branch targets a join label of the same group (or the epilogue ladders).
#[test]
fn weights_and_tables_are_read_once_outside_every_slot_branch() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let t = e8p_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let mut inside = false;
        let (mut gated, mut loads_outside) = (0usize, 0usize);
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
                    !m.starts_with("global_load")
                        && !m.starts_with("s_load")
                        && !m.starts_with("ds_load")
                        && !m.starts_with("ds_store"),
                    "{symbol}: load inside a slot branch: {line}"
                );
            } else if m.starts_with("global_load") || m.starts_with("ds_load") {
                loads_outside += 1
            }
        }
        assert_eq!(gated, 14 * inst + 14, "{symbol}: slot gates");
        // Word + scale + two table lookups per group, plus the single slot-id load: never multiplied by the slot count.
        assert_eq!(
            loads_outside,
            4 * inst + 1,
            "{symbol}: weight/table loads (read once)"
        );
        assert_eq!(
            count(&t, "s_cbranch_scc0"),
            14 * inst + 14 + if gate(spec) { 3 } else { 0 },
            "{symbol}: scc0 branches"
        );
        assert_eq!(
            count(&t, "s_cbranch_scc1"),
            8,
            "{symbol}: scc1 branches (6 guards, loop entry skip, back edge)"
        );
    }
}

/// The shape contract derives deterministically from emission: instruction counts, the forbidden
/// families (scratch/flat/swizzle/bcnt/WMMA/legacy waits; DS load/store are required here), register
/// ceilings equal to the plan, the 1,536-byte static LDS and no private or dynamic LDS memory.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in e8p_i4_moe_gemv::module_specs(ARCH) {
        let e = e8p_i4_moe_gemv::emit(spec).unwrap();
        let c = e8p_i4_moe_gemv::contract(spec, &e);
        assert_eq!(
            c,
            e8p_i4_moe_gemv::contract(spec, &e8p_i4_moe_gemv::emit(spec).unwrap())
        );
        assert_eq!(c["symbol"], spec.symbol().as_str());
        assert_eq!(c["counts"]["s_endpgm"], 1);
        assert_eq!(c["counts"]["global_store_b32"], SLOTS);
        assert_eq!(c["counts"]["ds_bpermute_b32"], 5 * SLOTS);
        assert_eq!(c["counts"]["ds_store_b32"], 12);
        assert_eq!(
            c["counts"]["ds_load_b32"],
            2 * spec.group_instances() as usize
        );
        assert_eq!(c["counts"]["s_barrier_signal"], 1);
        assert_eq!(
            (
                c["launch_dynamic_lds_bytes"].as_u64(),
                c["group_segment_fixed_bytes"].as_u64()
            ),
            (Some(0), Some(1536))
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
            "ds_swizzle*",
            "v_bcnt*",
            "v_wmma*",
            "v_fma_mix*",
            "s_waitcnt*",
        ] {
            assert!(forbidden.contains(&family), "{family}");
        }
        assert!(
            !forbidden.contains(&"ds_load*")
                && !forbidden.contains(&"ds_store*")
                && !forbidden.contains(&"s_barrier*")
                && !forbidden.contains(&"buffer_*")
        );
    }
}

// ----------------------------------------------------------------------- models

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(
            ((self.below(40) as u32 + 100) << 23)
                | (self.next() as u32 & 0x7f_ffff)
                | ((self.next() as u32 & 1) << 31),
        )
    }
}

/// The decode network model equals the K-E1 `decode8` (the frozen twin's `4*codeword`) for every one of the
/// 65,536 wire words and both lane halves, including its use of the staged tables.
#[test]
fn unpack_model_equals_decode8_for_all_65536_words() {
    let (lut, mask) = e8p_i4_moe_gemv::tables();
    for word in 0..=u16::MAX {
        let (idx, sign7, shift) = (
            (word & 0xff) as u8,
            ((word >> 8) & 0x7f) as u8,
            word >> 15 == 1,
        );
        let v = k_e1::decode8(idx, sign7, shift);
        for half in 0..2u32 {
            let m = e8p_i4_moe_gemv::unpack_model(&lut, &mask, word, half);
            for i in 0..4 {
                assert_eq!(
                    m[i],
                    i32::from(v[4 * half as usize + i]),
                    "word {word:#06x} half {half} lane {i}"
                );
            }
        }
        // The shared nibble-network word equals the K-E1 reference unpack (nibble o = (v - 1) / 2).
        assert_eq!(
            k_e1::unpack_swar(&lut, &mask, idx, sign7, shift),
            k_e1::unpack_word_ref(idx, sign7, shift),
            "word {word:#06x}"
        );
    }
}

/// FP16 -> F32 of every finite bit pattern equals the independent formula; the writer-guarded range
/// `d >= 0x0400` keeps `(f32(d)*0.25)*v` exact for every odd `|v| <= 11` (checked against `f64`), so no FP16 `d/4`.
#[test]
fn f16_convert_and_scaled_product_are_exact_in_the_guarded_range() {
    for bits in 0..=u16::MAX {
        let (e, f) = (u32::from((bits >> 10) & 0x1f), f64::from(bits & 0x3ff));
        if e == 0x1f {
            continue;
        }
        let magnitude = if e == 0 {
            f * 2f64.powi(-24)
        } else {
            (1.0 + f / 1024.0) * 2f64.powi(e as i32 - 15)
        };
        let value = if bits >> 15 == 1 {
            -magnitude
        } else {
            magnitude
        };
        assert_eq!(
            f64::from(e8p_i4_moe_gemv::f16_bits_to_f32(bits)),
            value,
            "f16 {bits:#06x}"
        );
    }
    let (lut, mask) = e8p_i4_moe_gemv::tables();
    for d in 0x0400..=0x7bffu16 {
        let dq = e8p_i4_moe_gemv::f16_bits_to_f32(d) * 0.25;
        assert_eq!(
            f64::from(dq),
            f64::from(e8p_i4_moe_gemv::f16_bits_to_f32(d)) * 0.25,
            "d {d:#06x}"
        );
        for v in (-11..=11i32).step_by(2) {
            let w = dq * v as f32;
            assert_eq!(
                f64::from(w),
                f64::from(dq) * f64::from(v),
                "d {d:#06x} v {v}"
            );
        }
    }
    // The weights the kernel produces are `RN(dq * v)` of the twin's integer lanes.
    let w = e8p_i4_moe_gemv::weights_model(&lut, &mask, 0x8123, 0x3c00, 1);
    let v = k_e1::decode8(0x23, 0x01, true);
    for i in 0..4 {
        assert_eq!(w[i], 0.25 * f32::from(v[4 + i]));
    }
}

/// The frozen wave tree (the K-E0 twin's `tree_row_core`): lane `l` owns `g*128 + 4l..+3`, `p = RN(w0*x0)`
/// then three fmas, partial added into `acc[g&3]` (R256) or the single chain (R128), fold, shuffle-down.
fn twin_row(four_chain: bool, groups: usize, x: &[f32], weight: impl Fn(usize) -> f32) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let j = g * 128 + 4 * lane;
            let mut p = weight(j) * x[j];
            for i in 1..4 {
                p = weight(j + i).mul_add(x[j + i], p);
            }
            acc[if four_chain { g & 3 } else { 0 }] += p;
        }
        *slot = if four_chain {
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            acc[0]
        };
    }
    for off in [16usize, 8, 4, 2, 1] {
        let snapshot = lanes;
        for l in 0..32 - off {
            lanes[l] = snapshot[l] + snapshot[l + off];
        }
    }
    lanes[0]
}

/// The kernel order, from the wire: lane `l` of group `g` reads the u16 word at group byte `2*(l>>1)`
/// (little-endian bytes of the plane), the group's FP16 scale, decodes through the staged tables and keeps
/// half `l & 1`; every slot runs its own chains over the same decoded weights.
fn kernel_row(
    four_chain: bool,
    k: usize,
    plane: &[u8],
    scales: &[u16],
    row: usize,
    xs: &[Vec<f32>],
) -> Vec<f32> {
    let (lut, mask) = e8p_i4_moe_gemv::tables();
    let groups = k / 128;
    let mut out = Vec::new();
    for x in xs {
        let mut lanes = [0.0f32; 32];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut acc = [0.0f32; 4];
            for g in 0..groups {
                let at = row * (k / 4) + e8p_i4_moe_gemv::word_byte(lane as u32, g as u32) as usize;
                let word = u16::from_le_bytes([plane[at], plane[at + 1]]);
                let w = e8p_i4_moe_gemv::weights_model(
                    &lut,
                    &mask,
                    word,
                    scales[row * groups + g],
                    (lane & 1) as u32,
                );
                let j = g * 128 + 4 * lane;
                let mut p = w[0] * x[j];
                for i in 1..4 {
                    p = w[i].mul_add(x[j + i], p);
                }
                acc[if four_chain { g & 3 } else { 0 }] += p;
            }
            *slot = if four_chain {
                (acc[0] + acc[1]) + (acc[2] + acc[3])
            } else {
                acc[0]
            };
        }
        for off in [16usize, 8, 4, 2, 1] {
            let snapshot = lanes;
            for l in 0..32 - off {
                lanes[l] = snapshot[l] + snapshot[l + off];
            }
        }
        out.push(lanes[0]);
    }
    out
}

/// Bit-for-bit, no tolerance: for random planar tensors (scales in the guarded range including both ends),
/// every one of up to eight slots of the kernel model equals the K-E0 twin (decode8 whole word -> `v`,
/// `RN((f32(d)*0.25)*v)`, `tree_row_core`) on the same data, for both rotations.
#[test]
fn kernel_order_model_equals_the_lane_pair_twin_bit_for_bit() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (four_chain, k) in [
        (true, 2560usize),
        (true, 256),
        (false, 640),
        (false, 128),
        (false, 256),
    ] {
        let (m, groups) = (3usize, k / 128);
        let words: Vec<u16> = (0..m * k / 8)
            .map(|_| {
                ((rng.next() & 0xff) | ((rng.next() & 0x7f) << 8) | ((rng.next() & 1) << 15)) as u16
            })
            .collect();
        let scales: Vec<u16> = (0..m * groups)
            .map(|i| match i % 7 {
                0 => 0x0400,
                1 => 0x7bff,
                _ => 0x0400 + rng.below(0x7bff - 0x0400 + 1) as u16,
            })
            .collect();
        let plane: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(plane.len(), m * k / 4);
        let xs: Vec<Vec<f32>> = (0..SLOTS)
            .map(|_| (0..k).map(|_| rng.f32()).collect())
            .collect();
        for row in 0..m {
            let got = kernel_row(four_chain, k, &plane, &scales, row, &xs);
            for (s, x) in xs.iter().enumerate() {
                let want = twin_row(four_chain, groups, x, |j| {
                    let (g, c) = (j / 128, j % 128);
                    let w = words[row * k / 8 + g * 16 + c / 8];
                    let (idx, sign7, shift) =
                        ((w & 0xff) as u8, ((w >> 8) & 0x7f) as u8, w >> 15 == 1);
                    let v = k_e1::decode8(idx, sign7, shift)[c % 8];
                    (e8p_i4_moe_gemv::f16_bits_to_f32(scales[row * groups + g]) * 0.25)
                        * f32::from(v)
                });
                assert_eq!(got[s].to_bits(), want.to_bits(), "k {k} row {row} slot {s}");
            }
        }
    }
}

/// Exact `u16` access bounds, zero tail pad: every weight word read lies inside its row's `K/4` code bytes,
/// the lanes of a pair read the same word, a row's 32 lanes x all groups cover its code bytes exactly, the
/// scale reads lie inside the scale plane, and the last byte read of the tensor is its last byte (no
/// over-read, so no tail padding is required).
#[test]
fn word_and_scale_reads_are_exact_with_zero_tail_pad() {
    for (m, k) in [
        (1280u32, 2560u32),
        (2560, 640),
        (1, 128),
        (3, 256),
        (7, 384),
    ] {
        let tensor = m * (k / 4) + m * (k / 64);
        let groups = k / 128;
        let mut max_end = 0u32;
        for r in [0u32, 1, m / 2, m - 1].into_iter().filter(|&r| r < m) {
            let mut covered = vec![false; (k / 4) as usize];
            for g in 0..groups {
                for lane in 0..32u32 {
                    let at = e8p_i4_moe_gemv::word_byte(lane, g);
                    assert_eq!(at % 2, 0);
                    assert!(
                        at >= 32 * g && at + 2 <= 32 * (g + 1),
                        "r {r} g {g} lane {lane}: word leaves its group"
                    );
                    if lane % 2 == 1 {
                        assert_eq!(at, e8p_i4_moe_gemv::word_byte(lane - 1, g));
                    }
                    covered[at as usize] = true;
                    covered[at as usize + 1] = true;
                }
                let scale_at = m * (k / 4) + 2 * (r * groups + g);
                assert!(
                    scale_at >= m * (k / 4) && scale_at + 2 <= tensor,
                    "r {r} g {g}: scale outside the scale plane"
                );
                max_end = max_end.max(scale_at + 2);
            }
            assert!(
                covered.iter().all(|&c| c),
                "row {r}: code bytes not covered exactly"
            );
            let w_end = e8p_i4_moe_gemv::word_end(k, r);
            assert_eq!(
                w_end,
                (r + 1) * (k / 4),
                "row {r}: last word byte is the row's last code byte"
            );
            assert!(w_end <= m * (k / 4));
        }
        assert_eq!(
            e8p_i4_moe_gemv::scale_end(m, k, m - 1),
            tensor,
            "m {m} k {k}: last scale byte is the tensor's last byte"
        );
        assert_eq!(max_end.max(e8p_i4_moe_gemv::scale_end(m, k, m - 1)), tensor);
    }
}

// ----------------------------------------------------------------------- isolation

/// Isolation: the E8P module is its own code object (distinct module and symbols, own label prefix, own
/// table LDS); the MQ4/W3 module is unchanged in character (no LDS, no barrier, no E8P text), and neither
/// emission depends on the other's state.
#[test]
fn e8p_module_is_isolated_from_the_mq_i4_moe_gemv_module() {
    let (_, e8p_text, e8p_proof) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    let before = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    let after = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!(
        (before.1.as_str(), before.2.s_text_sha256.as_str()),
        (after.1.as_str(), after.2.s_text_sha256.as_str()),
        "MQ4/W3 module is deterministic"
    );
    let (_, e8p_again, proof_again) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!(
        (&e8p_text, &e8p_proof.s_text_sha256),
        (&e8p_again, &proof_again.s_text_sha256),
        "E8P module independent of the MQ4/W3 emissions"
    );
    assert_ne!(Spec::module(ARCH), mq_i4_moe_gemv::Spec::module(ARCH));
    assert_ne!(e8p_proof.s_text_sha256, before.2.s_text_sha256);
    // The narrow (C8) entries are the unchanged no-LDS ones; the module's wide twins stage X through LDS by design.
    let narrow = mq_i4_moe_gemv::module_specs(ARCH)
        .into_iter()
        .map(|s| s.cap == mq_i4_moe_gemv::Cap::C8);
    for (e, _) in before
        .0
        .iter()
        .zip(narrow)
        .filter(|(_, is_narrow)| *is_narrow)
    {
        assert!(
            e.s_text.contains(".group_segment_fixed_size: 0\n") && !e.s_text.contains("e8p"),
            "{}: MQ4/W3 entry touched",
            e.proof.variant
        );
        assert_eq!(e.proof.barriers.len(), 0);
        assert!(!e
            .s_text
            .lines()
            .any(|l| mnemonic(l).starts_with("ds_load") || mnemonic(l).starts_with("ds_store")));
    }
    assert!(before.0.iter().all(|e| !e.s_text.contains("e8p")));
    assert!(!before.1.contains("e8p_i4_moe_gemv") && !before.1.contains(".Le8pi4moe_"));
    assert!(!e8p_text.contains("mq_i4_moe_gemv") && !e8p_text.contains(".Lmqi4moe_"));
    for symbol in SYMBOLS {
        assert!(!before.1.contains(symbol));
    }
    // Same numerical ABI names, different symbols and different kernel ids.
    assert_eq!(
        e8p_proof
            .kernels
            .iter()
            .map(|k| k.kernel_id.as_str())
            .collect::<Vec<_>>(),
        ["e8p_i4_moe_gemv", "e8p_i4_moe_gemv"]
    );
}

/// The module and every entry assemble with the native writer (no ROCm), and the bundle wraps the object.
#[test]
fn native_writer_assembles_every_entry_and_the_module() {
    let (emitted, text, _) = e8p_i4_moe_gemv::emit_module(ARCH).unwrap();
    for e in &emitted {
        let elf = peacemaker_isa::native::assemble(&e.s_text, ARCH)
            .unwrap_or_else(|err| panic!("{}: {err}", e.proof.kernel_id));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let bundle =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.len() > elf.len());
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, native, pm_check,
        toolchain::{build, certify, oracle_assemble_link_bundle, IsaShapeContract, Toolchain},
    };

    const HOST: &str = "host-x86_64-unknown-linux-gnu-";
    const ARCHES: [Arch; 3] = [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151];

    /// Waits/hazards replay independently of the builder's ledger, the ABI is unchanged under replay, the
    /// all-lane LDS bound holds over the 1,536-byte allocation, and M7 lifts the natively linked object
    /// byte-exactly with no outstanding obligations (no ROCm tools).
    #[test]
    fn every_export_passes_wait_hazard_replay_lds_bounds_and_m7() {
        for arch in ARCHES {
            wait_hazard_replay_lds_bounds_and_m7(arch)
        }
    }

    fn wait_hazard_replay_lds_bounds_and_m7(arch: Arch) {
        let (emitted, text, _) = e8p_i4_moe_gemv::emit_module(arch).unwrap();
        let symbols: Vec<String> = e8p_i4_moe_gemv::module_specs(arch)
            .iter()
            .map(Spec::symbol)
            .collect();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, arch)
                .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
            assert!(e.s_text.contains(".kernarg_segment_size: 96\n"));
        }
        ledger_replay::replay_waits(&text, arch).unwrap();
        for (spec, e) in e8p_i4_moe_gemv::module_specs(arch)
            .into_iter()
            .zip(&emitted)
        {
            let end =
                pm_check::lds_bounds(&e.s_text, &spec.symbol(), 1, e8p_i4_moe_gemv::STAGE_BYTES)
                    .unwrap_or_else(|err| panic!("{}: {err}", spec.symbol()));
            assert_eq!(end, e8p_i4_moe_gemv::STAGE_BYTES, "{}", spec.symbol());
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-e8pi4moe-m7-{}-{}",
            arch.name(),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, arch).unwrap()).unwrap();
        for symbol in &symbols {
            let m7 = pm_check::m7(&co, arch.name(), symbol.as_str())
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Native code object and offload bundle against the ROCm oracle for each entry alone and the module:
    /// equal in everything except the intentionally different `.comment` stamp.
    #[test]
    fn native_code_objects_and_bundles_equal_the_rocm_oracle_for_all_entries_and_the_module() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let toolchain = Toolchain {
                host_target: HOST.into(),
                ..Toolchain::default()
            };
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-e8pi4moe-identity-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let (emitted, module_text, _) = e8p_i4_moe_gemv::emit_module(arch).unwrap();
            let mut units: Vec<(String, String)> = emitted
                .iter()
                .zip(e8p_i4_moe_gemv::module_specs(arch))
                .map(|(e, spec)| (spec.symbol(), e.s_text.clone()))
                .collect();
            units.push((format!("{} (module)", Spec::module(arch)), module_text));
            let mut failures = Vec::new();
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
            assert!(
                failures.is_empty(),
                "{} of {} units differ from the oracle:\n{}",
                failures.len(),
                units.len(),
                failures.join("\n")
            );
        }
    }

    /// Every symbol certifies against the ROCm oracle build: parse-back, wait replay, its shape contract
    /// (static LDS 1,536), the all-lane LDS bound and M7 of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) = e8p_i4_moe_gemv::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-e8pi4moe-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            for (spec, e) in e8p_i4_moe_gemv::module_specs(arch)
                .into_iter()
                .zip(&emitted)
            {
                let symbol = spec.symbol();
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                    .unwrap_or_else(|err| panic!("{symbol}: {err}"));
                assert_eq!(
                    (m7["lift"].as_str(), &m7["obligations"]),
                    (Some("byte-exact"), &serde_json::json!({})),
                    "{symbol}"
                );
                let tag = spec.tag();
                let contract: IsaShapeContract =
                    serde_json::from_value(e8p_i4_moe_gemv::contract(spec, e)).unwrap();
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
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// CPU-only rowtile contract tests. These execute emitted/lifted ISA, not a coordinate-only
/// simulation; agreement with the incumbent and the independent F4 transcription is not
/// hardware qualification. Keep the narrow tests above as a separate, unchanged oracle.
mod rowtiles {
    use super::*;
    use peacemaker_ir::emu::{Emulator, Launch, Memory, Region};
    use peacemaker_ir::inst::Frontend;

    const ARCHES: [Arch; 3] = [Arch::Gfx1201, Arch::Gfx1151, Arch::Gfx1100];
    /// Arches whose transcendental (`v_rcp_iflag_f32`, ...) numerics the interpreter models. gfx1100 is
    /// refused by `peacemaker-ir` (`emu/alu.rs`: "unqualified numerical architecture"), so numerical
    /// oracles cannot execute its kernels; its guard and structural coverage stays on `ARCHES`.
    const NUMERIC_ARCHES: [Arch; 2] = [Arch::Gfx1201, Arch::Gfx1151];
    const KA: u64 = 0x1_0000;
    const TABLE: u64 = 0x10_0000;
    // z=1 crosses the low-dword boundary: descriptor additions must carry.
    const DESC: u64 = 0x2_ffff_ff00;
    const X: u64 = 0x8000_0000;
    const Y: [u64; 2] = [0x9000_0000, 0xa000_0000];
    const EXPERT: u64 = 0x1000_0000;
    const EXPERT_STRIDE: u64 = 0x10_0000;
    const NEXP: usize = 640;
    const WORDS: usize = 482;
    const GUARD: usize = 64;
    const POISON: u8 = 0xa5;

    fn lift(bytes: &[u8]) -> peacemaker_ir::Program {
        peacemaker_lift::lift_object(
            bytes,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program
    }

    fn region<'a>(mem: &'a Memory, name: &str) -> &'a Region {
        mem.regions.iter().find(|r| r.name == name).unwrap()
    }

    fn put(mem: &mut Memory, name: &str, base: u64, bytes: Vec<u8>) {
        if let Some(r) = mem.regions.iter_mut().find(|r| r.name == name) {
            r.bytes = bytes;
        } else {
            mem.map(name.into(), base, bytes).unwrap();
        }
    }

    fn u32s(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn launch(emu: &Emulator<'_>, mem: &mut Memory, wg: [u32; 3], block: u32) {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KA,
                workgroup: wg,
                block: [block, 1, 1],
                dynamic_lds: 0,
            },
            mem,
        )
        .unwrap_or_else(|e| panic!("workgroup {wg:?}, block {block}: {e}"));
    }

    struct Fixture {
        spec: Spec,
        c: usize,
        m: usize,
        k: usize,
        bucket: usize,
        ys: usize,
        xrs: usize,
        xsrc: usize,
        route: Vec<u32>,
        xs: Vec<f32>,
        weights: Vec<(Vec<u8>, Vec<u16>)>,
        mem: Memory,
    }

    impl Fixture {
        fn new(spec: Spec, c: usize, m: usize, k: usize, mode: usize) -> Self {
            let mut rng = Rng((0x91e1_7283 ^ (c * 193 + m * 17 + k + mode) as u64) | 1);
            let ys = if gate(spec) { m / 2 } else { m } + 3;
            let xrs = k + 4;
            let xsrc = if gate(spec) { c } else { 10 * c };
            // Identical, disjoint, and partially shared routes; experts are unique within
            // each token. Permute ids so first occurrence differs from sorted order.
            let route: Vec<u32> = (0..10 * c)
                .map(|s| {
                    let token = s / 10;
                    let rank = (s % 10 * 3 + 7) % 10;
                    (match mode % 3 {
                        0 => rank,
                        1 => token * 10 + rank,
                        _ => (token % 3) * 10 + rank,
                    }) as u32
                })
                .collect();
            let bucket = mode % 2;
            let mut mem = Memory::default();
            let mut table = vec![0u8; NEXP * 8];
            let mut weights = vec![(Vec::new(), Vec::new()); NEXP];
            for &id in &route {
                if !weights[id as usize].0.is_empty() {
                    continue;
                }
                let plane: Vec<u8> = (0..m * k / 8)
                    .flat_map(|_| (rng.next() as u16).to_le_bytes())
                    .collect();
                let scales: Vec<u16> = (0..m * k / 128)
                    .map(|i| match i % 7 {
                        0 => 0x0400,
                        1 => 0x7bff,
                        _ => 0x0400 + rng.below(0x7bff - 0x0400 + 1) as u16,
                    })
                    .collect();
                let mut tensor = plane.clone();
                tensor.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
                let addr = EXPERT + u64::from(id) * EXPERT_STRIDE;
                table[id as usize * 8..id as usize * 8 + 8].copy_from_slice(&addr.to_le_bytes());
                mem.map(format!("expert{id}"), addr, tensor).unwrap();
                weights[id as usize] = (plane, scales);
            }
            mem.map("table".into(), TABLE, table).unwrap();
            let xs: Vec<f32> = (0..xsrc * xrs)
                .map(|i| if i % xrs >= k { f32::NAN } else { rng.f32() })
                .collect();
            mem.map(
                "x".into(),
                X,
                xs.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            )
            .unwrap();
            // Reserve poison for all eight physical tiles, not merely the logical tail.
            for (name, addr) in [("y0", Y[0]), ("y1", Y[1])] {
                mem.map(name.into(), addr, vec![POISON; GUARD * 2 + 640 * ys * 4])
                    .unwrap();
            }
            let mut f = Self {
                spec,
                c,
                m,
                k,
                bucket,
                ys,
                xrs,
                xsrc,
                route,
                xs,
                weights,
                mem,
            };
            let args = f.args(80, c, xsrc);
            put(&mut f.mem, "kernarg", KA, args);
            f
        }

        fn args(&self, u: usize, c: usize, xsrc: usize) -> Vec<u8> {
            let b = self.bucket as u64;
            let mut bytes = Vec::new();
            for p in [
                TABLE,
                DESC + b * 320,
                DESC + (160 + b * 81) * 4,
                DESC + (322 + b * 80) * 4,
                X,
                Y[0] + GUARD as u64,
                if gate(self.spec) {
                    Y[1] + GUARD as u64
                } else {
                    0
                },
            ] {
                bytes.extend(p.to_le_bytes());
            }
            for v in [
                self.m,
                self.k,
                u,
                c,
                if gate(self.spec) { 10 } else { 1 },
                xsrc,
                self.xrs,
                self.ys,
                NEXP,
                0,
            ] {
                bytes.extend((v as u32).to_le_bytes());
            }
            assert_eq!(bytes.len(), 96);
            bytes
        }

        /// Current union law: first-occurrence experts, ascending absolute slots.
        /// The other bucket stays poisoned to detect accidental bucket/base selection.
        fn descriptors(&self, ranges: impl IntoIterator<Item = std::ops::Range<usize>>) -> Vec<u8> {
            let mut words = Vec::new();
            for range in ranges {
                let mut d = vec![u32::MAX; WORDS];
                let (ids, offs, slots) = (
                    self.bucket * 80,
                    160 + self.bucket * 81,
                    322 + self.bucket * 80,
                );
                let mut order = Vec::new();
                for s in range.clone() {
                    let id = self.route[s];
                    if !order.contains(&id) {
                        order.push(id);
                    }
                }
                let mut p = 0;
                for (u, &id) in order.iter().enumerate() {
                    d[ids + u] = id;
                    d[offs + u] = p as u32;
                    for s in range.clone().filter(|&s| self.route[s] == id) {
                        d[slots + p] = s as u32;
                        p += 1;
                    }
                }
                d[offs + order.len()..offs + 81].fill(p as u32);
                words.extend(d);
            }
            u32s(&words)
        }

        fn run(&self, emu: &Emulator<'_>, xsrc: usize) -> Memory {
            let mut mem = self.mem.clone();
            let descriptors = self.descriptors(
                (0..self.c.div_ceil(8)).map(|t| 80 * t..(80 * t + 80).min(10 * self.c)),
            );
            put(&mut mem, "descriptor", DESC, descriptors);
            put(&mut mem, "kernarg", KA, self.args(80, self.c, xsrc));
            for z in 0..self.c.div_ceil(8) as u32 {
                for y in 0..80 {
                    for x in 0..self.m.div_ceil(8) as u32 {
                        launch(emu, &mut mem, [x, y, z], 256);
                    }
                }
            }
            mem
        }

        fn singleton(&self, emu: &Emulator<'_>, xsrc: usize) -> Memory {
            let mut mem = self.mem.clone();
            for token in 0..self.c {
                let desc = self.descriptors([10 * token..10 * token + 10]);
                put(&mut mem, "descriptor", DESC, desc);
                put(&mut mem, "kernarg", KA, self.args(10, self.c, xsrc));
                for y in 0..10 {
                    for x in 0..self.m as u32 {
                        launch(emu, &mut mem, [x, y, 0], 32);
                    }
                }
            }
            mem
        }

        fn compare(&self, got: &Memory, singleton: &Memory, xsrc: usize) {
            let out_rows = if gate(self.spec) { self.m / 2 } else { self.m };
            for (part, name) in ["y0", "y1"].into_iter().enumerate() {
                let (a, b) = (region(got, name), region(singleton, name));
                assert_eq!(
                    a.bytes, b.bytes,
                    "{:?} C={} M={} K={} {name}",
                    self.spec, self.c, self.m, self.k
                );
                for byte in 0..a.bytes.len() {
                    let payload = byte >= GUARD && byte < GUARD + 10 * self.c * self.ys * 4;
                    let j = byte.saturating_sub(GUARD) / 4;
                    let live = payload && j % self.ys < out_rows && (part == 0 || gate(self.spec));
                    assert_eq!(
                        a.written.contains(byte),
                        live,
                        "{name} byte {byte}: tile ownership"
                    );
                    assert_eq!(
                        b.written.contains(byte),
                        live,
                        "{name} byte {byte}: singleton ownership"
                    );
                    if !live {
                        assert_eq!(a.bytes[byte], POISON, "{name} poison byte {byte}");
                    }
                }
                if part == 1 && !gate(self.spec) {
                    continue;
                }
                for slot in 0..10 * self.c {
                    let xr = slot / if gate(self.spec) { 10 } else { 1 };
                    let zero = vec![0.0; self.k];
                    let x = if xr < xsrc {
                        &self.xs[xr * self.xrs..xr * self.xrs + self.k]
                    } else {
                        &zero
                    };
                    let (plane, scales) = &self.weights[self.route[slot] as usize];
                    for r in 0..out_rows {
                        let row = part * out_rows + r;
                        let want = twin_row(gate(self.spec), self.k / 128, x, |j| {
                            let at = row * self.k / 4 + 2 * (j / 8);
                            let word = u16::from_le_bytes([plane[at], plane[at + 1]]);
                            let v = k_e1::decode8(
                                word as u8,
                                ((word >> 8) & 127) as u8,
                                word >> 15 != 0,
                            )[j % 8];
                            (e8p_i4_moe_gemv::f16_bits_to_f32(
                                scales[row * (self.k / 128) + j / 128],
                            ) * 0.25)
                                * f32::from(v)
                        });
                        let at = GUARD + (slot * self.ys + r) * 4;
                        let actual = u32::from_le_bytes(a.bytes[at..at + 4].try_into().unwrap());
                        assert_eq!(
                            actual,
                            want.to_bits(),
                            "{:?} C={} slot={slot} row={row}: independent singleton DAG",
                            self.spec,
                            self.c
                        );
                    }
                }
            }
            for mem in [got, singleton] {
                for r in &mem.regions {
                    if r.name != "y0" && r.name != "y1" {
                        assert_eq!(r.written.count(), 0, "{} input modified", r.name);
                    }
                }
            }
        }
    }

    #[test]
    fn every_logical_width_matches_committed_singleton_and_independent_dag() {
        use sha2::{Digest, Sha256};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        for arch in NUMERIC_ARCHES {
            let path = dir.join(format!("{}.hxaco", Spec::module(arch)));
            let object = std::fs::read(&path).unwrap();
            // Record the actual oracle object, never silently substitute another architecture.
            eprintln!(
                "E8P rowtiles singleton={} sha256={:x}",
                path.display(),
                Sha256::digest(&object)
            );
            let incumbent = lift(&object);
            for spec in e8p_i4_moe_gemv::module_specs(arch) {
                let e = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
                let program = lift(&peacemaker_isa::native::assemble(&e.s_text, arch).unwrap());
                let candidate = Emulator::new(&program, &format!("{}_t8", spec.symbol())).unwrap();
                let singleton = Emulator::new(&incumbent, &spec.symbol()).unwrap();
                // Widths are independent: sweep them concurrently (a panic in any
                // width fails the scope, so coverage and failure reporting are unchanged).
                std::thread::scope(|s| {
                    for c in 1usize..=64 {
                        let (candidate, singleton) = (&candidate, &singleton);
                        s.spawn(move || {
                            let m = if gate(spec) {
                                [2, 6, 14, 16, 18][c % 5]
                            } else {
                                [1, 3, 7, 8, 9][c % 5]
                            };
                            let groups = if gate(spec) {
                                [2, 4, 6, 8, 10][c % 5]
                            } else {
                                [1, 2, 3, 4, 5, 7, 8, 9][c % 8]
                            };
                            let f = Fixture::new(spec, c, m, 128 * groups, c);
                            let got = f.run(candidate, f.xsrc);
                            let want = f.singleton(singleton, f.xsrc);
                            f.compare(&got, &want, f.xsrc);
                        });
                    }
                });
                // Buffer-descriptor OOB rows must read zero, not poisoned stride padding.
                let f = Fixture::new(spec, 9, if gate(spec) { 18 } else { 9 }, 256, 0);
                let xsrc = if gate(spec) { 4 } else { 40 };
                f.compare(
                    &f.run(&candidate, xsrc),
                    &f.singleton(&singleton, xsrc),
                    xsrc,
                );
            }
        }
    }

    fn untouched(mem: &Memory) {
        for r in &mem.regions {
            assert_eq!(r.written.count(), 0, "{} unexpectedly written", r.name);
            if r.name == "y0" || r.name == "y1" {
                assert!(r.bytes.iter().all(|&b| b == POISON), "{} poison", r.name);
            }
        }
    }

    #[test]
    fn invalid_guards_exit_without_dereferencing_poisoned_inputs() {
        for arch in ARCHES {
            for spec in e8p_i4_moe_gemv::module_specs(arch) {
                let e = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
                let program = lift(&peacemaker_isa::native::assemble(&e.s_text, arch).unwrap());
                let emu = Emulator::new(&program, &format!("{}_t8", spec.symbol())).unwrap();
                let f = Fixture::new(spec, 9, if gate(spec) { 18 } else { 9 }, 256, 0);
                // Domain/dead workgroup guards must not even touch descriptors.
                // The weight table, X and all weights are deliberately unmapped.
                for (c, u, wg) in [
                    (0, 80, [0, 0, 0]),
                    (65, 80, [0, 0, 0]),
                    (9, 0, [0, 0, 0]),
                    (9, 79, [0, 0, 0]),
                    (9, 81, [0, 0, 0]),
                    (9, 80, [0, 0, 2]),
                    (9, 80, [f.m.div_ceil(8) as u32, 0, 1]),
                    (9, 80, [0, 80, 1]),
                ] {
                    let mut mem = f.mem.clone();
                    mem.regions
                        .retain(|r| r.name == "kernarg" || r.name.starts_with('y'));
                    put(&mut mem, "kernarg", KA, f.args(u, c, f.xsrc));
                    launch(&emu, &mut mem, wg, 256);
                    untouched(&mem);
                }
                // z=1 makes offset81 legal under the obsolete 10*C (=90) bound,
                // but illegal under the fixed descriptor capacity80.
                for (id, start, end, slot, bad_slot) in [
                    (0, 0, 0, 80, false),   // empty capacity entry
                    (0, 0, 9, 80, false),   // nine slots
                    (0, 9, 1, 80, false),   // negative count underflows
                    (0, 80, 81, 80, false), // local offsets past80
                    (NEXP as u32, 0, 1, 80, false),
                    (NEXP as u32 + 1, 0, 1, 80, false),
                    (u32::MAX, 0, 1, 80, false),
                    (i32::MIN as u32, 0, 1, 80, false),
                    (0, 0, 1, 90, true), // first absolute tail slot
                    (0, 0, 1, u32::MAX, true),
                ] {
                    let mut mem = f.mem.clone();
                    mem.regions.retain(|r| {
                        r.name == "kernarg"
                            || r.name.starts_with('y')
                            || (bad_slot && r.name == "table")
                    });
                    let mut desc = vec![u32::MAX; 2 * WORDS];
                    let base = WORDS;
                    desc[base] = id;
                    desc[base + 160] = start;
                    desc[base + 161] = end;
                    desc[base + 322] = slot;
                    put(&mut mem, "descriptor", DESC, u32s(&desc));
                    launch(&emu, &mut mem, [0, 0, 1], 256);
                    untouched(&mem);
                }
                // One bad slot among otherwise valid slots rejects uniformly, not
                // merely the lane that happened to load it (all waves share a barrier).
                for bad_lane in 0..8 {
                    let mut mem = f.mem.clone();
                    mem.regions
                        .retain(|r| !r.name.starts_with("expert") && r.name != "x");
                    let mut desc = vec![u32::MAX; 2 * WORDS];
                    desc[WORDS] = 0;
                    desc[WORDS + 160] = 0;
                    desc[WORDS + 161] = 8;
                    desc[WORDS + 322..WORDS + 330].fill(80);
                    desc[WORDS + 322 + bad_lane] = 90;
                    put(&mut mem, "descriptor", DESC, u32s(&desc));
                    launch(&emu, &mut mem, [0, 0, 1], 256);
                    untouched(&mem);
                }
            }
        }
    }

    #[test]
    fn perturbing_one_absolute_input_row_isolates_its_outputs_across_tiles() {
        for arch in NUMERIC_ARCHES {
            for spec in e8p_i4_moe_gemv::module_specs(arch) {
                let e = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
                let program = lift(&peacemaker_isa::native::assemble(&e.s_text, arch).unwrap());
                let emu = Emulator::new(&program, &format!("{}_t8", spec.symbol())).unwrap();
                let mut f = Fixture::new(spec, 17, if gate(spec) { 2 } else { 1 }, 256, 0);
                let before = f.run(&emu, f.xsrc);
                let victim = if gate(spec) { 8 } else { 80 };
                f.xs[victim * f.xrs..victim * f.xrs + f.k].fill(0.0);
                put(
                    &mut f.mem,
                    "x",
                    X,
                    f.xs.iter()
                        .flat_map(|x| x.to_bits().to_le_bytes())
                        .collect(),
                );
                let after = f.run(&emu, f.xsrc);
                let out_rows = if gate(spec) { f.m / 2 } else { f.m };
                for name in ["y0", "y1"]
                    .into_iter()
                    .take(if gate(spec) { 2 } else { 1 })
                {
                    let (a, b) = (region(&before, name), region(&after, name));
                    let mut changed = 0;
                    for slot in 0..10 * f.c {
                        let victim_slot = slot / if gate(spec) { 10 } else { 1 } == victim;
                        for row in 0..out_rows {
                            let at = GUARD + (slot * f.ys + row) * 4;
                            if victim_slot {
                                let bits =
                                    u32::from_le_bytes(b.bytes[at..at + 4].try_into().unwrap());
                                assert_eq!(bits << 1, 0, "{name} victim slot {slot}: zero row");
                                changed += usize::from(a.bytes[at..at + 4] != b.bytes[at..at + 4]);
                            } else {
                                assert_eq!(
                                    a.bytes[at..at + 4],
                                    b.bytes[at..at + 4],
                                    "{name} unrelated absolute slot {slot}"
                                );
                            }
                        }
                    }
                    assert!(
                        changed > 0,
                        "{name}: isolation fixture must change live outputs"
                    );
                }
            }
        }
    }

    /// The gfx1151 `_t8` entries are the gfx1100 ones modulo the arch-forced deltas and the gfx1100 SGPR placement
    /// (`FeatureUserSGPRInit16Bug`: ids `s2` / `s3` / `s4` -> `s13` / `s14` / `s15`, kernarg pointer block
    /// `s[8:15]` -> `s[88:95]`, the declared user count 13 and the SGPR ceiling).
    #[test]
    fn gfx1151_rowtiles_are_the_gfx1100_rowtiles_modulo_the_sgpr_placement() {
        fn strip(text: &str) -> String {
            text.lines()
                .filter(|l| l.trim() != TRANS_USE_WAIT)
                .map(|l| format!("{l}\n"))
                .collect()
        }
        for spec in e8p_i4_moe_gemv::module_specs(Arch::Gfx1100) {
            let a = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
            let b = e8p_i4_moe_gemv::emit_rowtiles(Spec {
                arch: Arch::Gfx1151,
                kind: spec.kind,
            })
            .unwrap();
            assert_eq!(
                gfx1151_sgprs_as_gfx1100(&strip(&b.s_text), true).replace("gfx1151", "gfx1100"),
                strip(&a.s_text),
                "{spec:?}: _t8 source"
            );
            assert_eq!(
                (
                    u32::from(a.shape.next_free_sgpr),
                    u32::from(b.shape.next_free_sgpr)
                ),
                (sgprs(Arch::Gfx1100), sgprs(Arch::Gfx1151))
            );
        }
    }

    #[test]
    fn emission_abi_resources_isa_and_barrier_order_are_arch_specific() {
        for arch in ARCHES {
            for spec in e8p_i4_moe_gemv::module_specs(arch) {
                let e = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
                let again = e8p_i4_moe_gemv::emit_rowtiles(spec).unwrap();
                let narrow = e8p_i4_moe_gemv::emit(spec).unwrap();
                let t = &e.s_text;
                assert_eq!(t, &again.s_text);
                assert_eq!(e.proof.arch, arch);
                assert!(t.contains(&format!(".amdhsa_kernel {}_t8\n", spec.symbol())));
                assert_eq!(
                    field(t, ".offset:"),
                    ARGS.iter().map(|a| a.0).collect::<Vec<_>>()
                );
                assert_eq!(
                    field(t, ".size:"),
                    ARGS.iter().map(|a| a.1).collect::<Vec<_>>()
                );
                for needle in [
                    ".kernarg_segment_size: 96\n",
                    ".amdhsa_kernarg_size 96\n",
                    ".amdhsa_wavefront_size32 1",
                    ".amdhsa_system_sgpr_workgroup_id_z 1",
                    ".private_segment_fixed_size: 0\n",
                    ".sgpr_spill_count: 0\n",
                    ".vgpr_spill_count: 0\n",
                ] {
                    assert!(t.contains(needle), "{spec:?}: {needle}");
                }
                assert_eq!(field(t, ".max_flat_workgroup_size:"), [256]);
                assert_eq!(field(t, ".group_segment_fixed_size:"), [1536]);
                assert_eq!(e.shape.next_free_vgpr, narrow.shape.next_free_vgpr);
                assert_eq!(e.shape.next_free_sgpr, narrow.shape.next_free_sgpr);
                let lines = insns(t);
                let pos = |needle: &str| {
                    lines
                        .iter()
                        .position(|l| *l == needle)
                        .unwrap_or_else(|| panic!("{spec:?}: {needle}"))
                };
                assert!(t.contains("s_load_b32 s6, s[0:1], 0x58"));
                assert!(t.contains("v_readfirstlane_b32 s5, v"));
                let barrier = if arch.gfx12() {
                    "s_barrier_wait 0xffff"
                } else {
                    "s_barrier"
                };
                let publish = pos(barrier);
                let row_guard = pos("s_cmp_ge_u32 s30, s22");
                assert!(
                    publish < row_guard,
                    "inactive output waves must publish tables"
                );
                for guard in [
                    "s_cmp_eq_u32 s25, 0",
                    "s_cmp_gt_u32 s25, 64",
                    "s_cmp_eq_u32 s24, 0x50",
                    "s_cmp_gt_u32 s35, 8",
                    "s_mov_b32 s51, 0x50",
                    "s_cmp_gt_u32 s33, s51",
                    "s_cmp_ge_u32 s34, s6",
                    "s_cmp_ge_u32 s51, s52",
                ] {
                    assert!(pos(guard) < publish, "{spec:?}: guard {guard} after stage");
                }
                assert!(t.contains("s_mul_i32 s51, s7, 0x788"));
                for op in [
                    "global_load_u16",
                    "buffer_load_b128",
                    "ds_load_b32",
                    "v_bfe_i32",
                    "v_add3_u32",
                    "v_cvt_f32_i32_e32",
                    "v_cvt_f32_f16_e64",
                    "ds_bpermute_b32",
                    "global_store_b32",
                ] {
                    assert_eq!(
                        count(t, op),
                        count(&narrow.s_text, op),
                        "{spec:?}: singleton {op} census"
                    );
                }
                assert_eq!(count(t, "global_store_b32"), 8);
                assert_eq!(
                    count(t, "global_load_u16"),
                    2 * spec.group_instances() as usize
                );
                assert_eq!(u32::from(e.shape.next_free_sgpr), sgprs(arch));
                assert_eq!(e.shape.next_free_vgpr, if gate(spec) { 112 } else { 88 });
                assert!(lines[..publish]
                    .iter()
                    .all(|l| mnemonic(l) != "ds_load_b32"));
                assert!(lines[publish + 1..]
                    .iter()
                    .all(|l| mnemonic(l) != "ds_store_b32"));
                assert!(lines[..publish]
                    .iter()
                    .any(|l| mnemonic(l) == "ds_store_b32"));
                if arch.gfx12() {
                    assert!(t.contains("s_lshr_b32 s7, ttmp7, 16"));
                    assert!(t.contains("s_lshl_b32 s30, ttmp9, 3"));
                    assert_eq!(
                        (count(t, "s_barrier_signal"), count(t, "s_barrier_wait")),
                        (1, 1)
                    );
                    assert_eq!(count(t, "s_barrier"), 0);
                } else {
                    // Enabled workgroup ids x / y / z follow the declared user SGPRs (gfx1100: padded to 13).
                    let x = wg_x(arch, true);
                    assert!(t.contains(&format!("s_mov_b32 s7, s{}", x + 2)));
                    assert!(t.contains(&format!("s_lshl_b32 s30, s{x}, 3")));
                    assert!(t.contains(&format!("s_mov_b32 s31, s{}", x + 1)));
                    assert!(t.contains(&format!(
                        ".amdhsa_user_sgpr_count {}\n",
                        arch.declared_user_sgprs(2, 3)
                    )));
                    assert!(!t.contains("ttmp"));
                    assert_eq!(count(t, "s_barrier"), 1);
                    for banned in [
                        "s_barrier_signal",
                        "s_barrier_wait",
                        "s_wait_loadcnt",
                        "s_wait_dscnt",
                        "s_wait_kmcnt",
                        "s_wait_storecnt",
                        "s_add_co_i32",
                        "s_sub_co_i32",
                    ] {
                        assert_eq!(count(t, banned), 0, "{spec:?}: {banned}");
                    }
                }
                // Compare the ordered arithmetic operands, not just an operation count.
                // Stage scheduling/addressing may differ; each slot's FP DAG may not.
                let fp = |text: &str| -> Vec<String> {
                    insns(text)
                        .into_iter()
                        .filter(|l| {
                            matches!(
                                mnemonic(l),
                                "v_mul_f32_e32"
                                    | "v_fmac_f32_e32"
                                    | "v_add_f32_e32"
                                    | "v_cvt_f32_f16_e64"
                                    | "v_cvt_f32_i32_e32"
                                    | "ds_bpermute_b32"
                            )
                        })
                        .map(str::to_owned)
                        .collect()
                };
                assert_eq!(
                    fp(t),
                    fp(&narrow.s_text),
                    "{spec:?}: singleton F4 operand DAG"
                );
                for prefix in [
                    "scratch_",
                    "flat_",
                    "global_atomic",
                    "buffer_atomic",
                    "v_wmma",
                    "v_mul_f16",
                    "v_fma_mix",
                ] {
                    assert!(
                        !lines.iter().any(|l| mnemonic(l).starts_with(prefix)),
                        "{spec:?}: {prefix}"
                    );
                }
                assert!(peacemaker_isa::native::assemble(t, arch)
                    .unwrap()
                    .starts_with(b"\x7fELF"));
                #[cfg(feature = "toolchain")]
                peacemaker_isa::ledger_replay::replay_waits(t, arch).unwrap();
                assert_eq!(
                    e8p_i4_moe_gemv::contract(spec, &e)["symbol"],
                    format!("{}_t8", spec.symbol())
                );
                let (combined, _) = peacemaker_isa::kernels::iu4_gemm::module(
                    &[narrow, e],
                    &format!("e8p_rowtiles_test_{}_{}", arch.name(), spec.tag()),
                )
                .unwrap();
                let labels: Vec<&str> = combined
                    .lines()
                    .map(str::trim)
                    .filter(|l| l.starts_with(".L") && l.ends_with(':'))
                    .collect();
                let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
                assert_eq!(
                    labels.len(),
                    unique.len(),
                    "{spec:?}: narrow/t8 label collision"
                );
                assert!(peacemaker_isa::native::assemble(&combined, arch)
                    .unwrap()
                    .starts_with(b"\x7fELF"));
            }
        }
        for spec in e8p_i4_moe_gemv::module_specs(Arch::Gfx942) {
            assert!(e8p_i4_moe_gemv::emit_rowtiles(spec).is_err());
        }
    }
}
