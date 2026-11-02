// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2 E8P gfx1151 (`src/kernels/qwen4_moe_sym_gfx1100/`, one arch-parameterised module): the gfx1151 twin of
//! `qwen4_moe_sym_gfx1100_e8p_p0.rs` (the rc5 64-byte ABI with `n_experts@60` and its expert-range guard), plus the
//! cross-arch law of the Halo port.
//!
//! * Names, ABI, the 1,536 B LDS stage, the builder gates (wait ledger, hazards, LDS typestate and barrier, VOPD
//!   pairing, register plan) and the mutation set hold for gfx1151 exactly as for gfx1100.
//! * The gfx1151 source is the gfx1100 source with only arch-forced deltas: the arch name, and the builder's
//!   `VALUTransUseHazard` waits (`s_waitcnt_depctr depctr_va_vdst(0)` after a transcendental), a gfx1100-only
//!   hazard (`Gfx11Hazards::for_arch`; the installed ROCm LLVM inserts it for `-mcpu=gfx1100` and not for
//!   `-mcpu=gfx1151`), and gfx1100's `FeatureUserSGPRInit16Bug` SGPR renumbering (workgroup ids s14/s15, kernarg block
//!   s88:95, E8P table literals s8..s13/s84/s85, user-SGPR count 14). The VGPR plan, LDS tables, operand/fold sequence and ABI are identical.
//! * The committed `kernels/qwen4_moe_iu4_sym_e8p_pm_{gfx1100,gfx1151}.hxaco` candidates and the six
//!   `kernels/qwen4_moe_{arch}_negative/<negative>/` images are the emitter's output, byte for byte (native
//!   assembler + HIP bundle), so any change of the shared module that moves either arch shows up here.
#![allow(dead_code, unused_imports)]

#[path = "../src/kernels/qwen4_moe_sym_gfx1100/mod.rs"]
mod gfx11;

use gfx11::{Codec, Kind, Negative, Spec};
use peacemaker_isa::kernels::iu4_gemm;
use peacemaker_isa::{native, Arch};

const TRANS_WAIT: &str = "s_waitcnt_depctr depctr_va_vdst(0)";

fn spec(arch: Arch, codec: Codec, kind: Kind, acc: bool, negative: Negative) -> Spec {
    Spec {
        arch,
        codec,
        kind,
        acc,
        negative,
    }
}

fn s1151(codec: Codec, kind: Kind, acc: bool, negative: Negative) -> Spec {
    spec(Arch::Gfx1151, codec, kind, acc, negative)
}

#[test]
fn e8p_gfx1151_names_and_abi() {
    assert_eq!(
        Codec::E8p.module(Arch::Gfx1151),
        "qwen4_moe_iu4_sym_e8p_pm_gfx1151"
    );
    assert_eq!(
        s1151(Codec::E8p, Kind::GateUp, false, Negative::None).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1151"
    );
    assert_eq!(
        s1151(Codec::E8p, Kind::Down, false, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_e8p_pm_gfx1151"
    );
    assert_eq!(
        s1151(Codec::E8p, Kind::GateUp, true, Negative::None).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1151_acc"
    );
    assert_eq!(
        s1151(Codec::E8p, Kind::Down, true, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_e8p_pm_gfx1151_acc"
    );
    assert_eq!(
        s1151(Codec::E8p, Kind::Down, true, Negative::None).variant(),
        "down_e8p_acc"
    );
    assert_eq!(gfx11::ARCHES, [Arch::Gfx1100, Arch::Gfx1151]);
    assert_eq!(gfx11::KERNARG_BYTES, 64);
    let layout = s1151(Codec::E8p, Kind::Down, false, Negative::None).kernargs();
    assert_eq!(layout.size, 64);
    assert_eq!(
        layout
            .args
            .last()
            .map(|a| (a.name.as_str(), a.offset, a.size)),
        Some(("n_experts", 60, 4))
    );
    assert_eq!(Codec::ALL.len(), 3, "the legacy module set is unchanged");
    assert_eq!(
        Negative::ALL.len(),
        4,
        "the legacy negative set is unchanged"
    );
    for kind in Kind::ALL {
        let (a, b) = (
            spec(Arch::Gfx1100, Codec::E8p, kind, false, Negative::None),
            s1151(Codec::E8p, kind, false, Negative::None),
        );
        assert_eq!(
            serde_json::to_value(a.kernargs()).unwrap(),
            serde_json::to_value(b.kernargs()).unwrap(),
            "{kind:?}: kernarg ABI"
        );
        assert_eq!(
            (a.threads(), a.lds_bytes(), a.grid(2560, 1536)),
            (b.threads(), b.lds_bytes(), b.grid(2560, 1536)),
            "{kind:?}: launch geometry"
        );
    }
}

#[test]
fn only_gfx11_targets_are_admitted() {
    for arch in [Arch::Gfx1201, Arch::Gfx942] {
        for kind in Kind::ALL {
            let s = spec(arch, Codec::E8p, kind, false, Negative::None);
            assert!(gfx11::emit(s).is_err(), "{arch:?} {kind:?}");
            assert!(
                gfx11::validate_launch(s, 2560, 2560, 1, 1536, 1536).is_err(),
                "{arch:?} {kind:?}"
            );
        }
    }
}

#[test]
fn launch_admission_is_arch_independent() {
    let launches = [
        (2560u32, 2560u32, 1u32, 1536u32, 1536u32),
        (2560, 640, 10, 32, 17),
        (2560, 2560, 1, 0, 1536),
        (100, 2560, 1, 1536, 1536),
        (2560, 2559, 1, 1536, 1536),
    ];
    for kind in Kind::ALL {
        for acc in [false, true] {
            for (m, k, x_row_div, m_total, x_src_rows) in launches {
                let r = |arch| {
                    gfx11::validate_launch(
                        spec(arch, Codec::E8p, kind, acc, Negative::None),
                        m,
                        k,
                        x_row_div,
                        m_total,
                        x_src_rows,
                    )
                };
                let (a, b) = (r(Arch::Gfx1100), r(Arch::Gfx1151));
                assert_eq!(
                    a.is_ok(),
                    b.is_ok(),
                    "{kind:?} acc {acc} {:?}",
                    (m, k, x_row_div, m_total, x_src_rows)
                );
                if let (Err(a), Err(b)) = (a, b) {
                    assert_eq!(a.replace("gfx1100", "gfx1151"), b)
                }
            }
        }
    }
}

#[test]
fn e8p_gfx1151_entries_and_mutations_build_with_the_lds_stage() {
    for kind in Kind::ALL {
        for acc in [false, true] {
            for negative in std::iter::once(Negative::None).chain(Negative::E8P) {
                let s = s1151(Codec::E8p, kind, acc, negative);
                let e = gfx11::emit(s).unwrap_or_else(|err| panic!("{s:?}: {err}"));
                assert_eq!(e.proof.arch, Arch::Gfx1151, "{s:?}");
                assert!(
                    e.s_text.contains(".group_segment_fixed_size: 1536"),
                    "{s:?}: missing the 1,536 B static LDS"
                );
                assert!(
                    e.s_text.contains("s_barrier"),
                    "{s:?}: missing the table publication barrier"
                );
                assert!(e.s_text.contains("ds_load_b32"), "{s:?}: no LDS table read");
                assert!(
                    e.s_text.contains(&s.symbol()),
                    "{s:?}: symbol missing from the source"
                );
                assert!(
                    !e.s_text.contains("gfx1100"),
                    "{s:?}: a gfx1100 name leaked into the gfx1151 source"
                );
            }
        }
    }
}

/// The rc5 expert-range guard on every gfx11 arch (MQ4 and E8P, BF16 and `_acc`, every mutation): `n_experts`
/// (kernarg 0x3c) is loaded into s5 and the routed expert id s23 is range-checked with ONE unsigned
/// `s_cmp_ge_u32 s23, s5` branching to the no-store exit before the `EID * 8` (or the `WrongExpertPointer`
/// `s_mov_b32 s4, 0`) pointer index, the expert-pointer load, the first buffer load and the table barrier. The
/// instruction sequence of the guard is the gfx1100 one (the builder's only gfx1151 delta is the trans-use wait).
#[test]
fn expert_guard_is_one_unsigned_compare_before_pointer_load_barrier_and_loads_on_every_gfx11_arch()
{
    for arch in gfx11::ARCHES {
        for codec in [Codec::Mq4, Codec::E8p] {
            let negatives: Vec<Negative> = std::iter::once(Negative::None)
                .chain([
                    Negative::WrongScaleRow,
                    Negative::ReversedEpoch,
                    Negative::WrongPackSelector,
                    Negative::WrongExpertPointer,
                    Negative::WrongLut,
                    Negative::WrongSignMask,
                    Negative::WrongFoldEpoch,
                ])
                .filter(|n| n.applies(codec))
                .collect();
            for kind in Kind::ALL {
                for acc in [false, true] {
                    for &negative in &negatives {
                        if acc && negative == Negative::WrongPackSelector {
                            continue;
                        }
                        let s = spec(arch, codec, kind, acc, negative);
                        let e = gfx11::emit(s).unwrap_or_else(|err| panic!("{s:?}: {err}"));
                        let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
                        let at = |p: &str| lines.iter().position(|l| l.starts_with(p));
                        assert!(
                            e.s_text.contains(".kernarg_segment_size: 64"),
                            "{s:?}: kernarg segment"
                        );
                        assert_eq!(
                            lines
                                .iter()
                                .filter(|l| l.starts_with("s_cmp_ge_u32 s23, s5"))
                                .count(),
                            1,
                            "{s:?}: one unsigned guard"
                        );
                        assert!(
                            !lines.iter().any(|l| l.starts_with("s_cmp_lt_i32 s23, 0")),
                            "{s:?}: old signed compare"
                        );
                        let load = at("s_load_b32 s5, s[0:1], 0x3c")
                            .unwrap_or_else(|| panic!("{s:?}: no n_experts load"));
                        let cmp = at("s_cmp_ge_u32 s23, s5").unwrap();
                        assert!(load < cmp, "{s:?}: n_experts is loaded before the guard");
                        assert!(
                            lines[cmp + 1].starts_with("s_cbranch_scc1"),
                            "{s:?}: the guard branches to the exit"
                        );
                        let pointer_index = if negative == Negative::WrongExpertPointer {
                            at("s_mov_b32 s4, 0")
                        } else {
                            at("s_lshl_b32 s4, s23, 3")
                        }
                        .expect("expert pointer index");
                        let pointer_load =
                            at("s_load_b64 s[58:59], s[6:7], null").expect("expert pointer load");
                        let first_buffer = at("buffer_load").expect("first buffer load");
                        for (what, i) in [
                            ("pointer index", pointer_index),
                            ("pointer load", pointer_load),
                            ("buffer load", first_buffer),
                        ] {
                            assert!(cmp < i, "{s:?}: guard must precede the {what}");
                        }
                        if codec == Codec::E8p {
                            let barrier = at("s_barrier").expect("table barrier");
                            assert!(cmp < barrier, "{s:?}: guard must precede the table barrier");
                        }
                    }
                }
            }
        }
    }
}

/// Evaluate an emitted scalar compare `s_cmp_<cond>_<ty> a, b` (`a` the routed expert id, `b` `n_experts`) on
/// concrete values, honouring the instruction's own signedness.
fn scalar_compare(line: &str, a: u32, b: u32) -> bool {
    let name = line.split_whitespace().next().unwrap_or("");
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

/// CPU model of the emitted guard on every gfx11 arch: the exact compare the builder placed, evaluated on concrete
/// values, rejects `-1` (the router's non-finite marker), `i32::MIN`, `n` and `n + 1` and passes `n - 1`; a signed
/// spelling would admit `-1` / `i32::MIN` and fails the model. No vector / global / buffer / LDS / scratch memory
/// instruction and no store precedes the guard, so a rejected tile never dereferences the expert table or a tensor
/// and never writes `Y`.
#[test]
fn expert_guard_cpu_model_rejects_minus_one_min_n_and_n_plus_one_and_passes_n_minus_one() {
    let forbidden = |m: &str| {
        m.starts_with("global_")
            || m.starts_with("buffer_")
            || m.starts_with("flat_")
            || m.starts_with("scratch_")
            || m.starts_with("ds_")
            || m.starts_with("s_store")
    };
    for arch in gfx11::ARCHES {
        for codec in [Codec::Mq4, Codec::E8p] {
            for kind in Kind::ALL {
                for acc in [false, true] {
                    let s = spec(arch, codec, kind, acc, Negative::None);
                    let e = gfx11::emit(s).unwrap_or_else(|err| panic!("{s:?}: {err}"));
                    let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
                    let cmp = lines
                        .iter()
                        .position(|l| l.starts_with("s_cmp_ge_u32 s23, s5"))
                        .unwrap_or_else(|| panic!("{s:?}: no guard compare"));
                    assert!(
                        lines[cmp + 1].starts_with("s_cbranch_scc1"),
                        "{s:?}: SCC set (id >= n) exits"
                    );
                    for n in [1u32, 2, 5, 512, i32::MAX as u32] {
                        for id in [u32::MAX, i32::MIN as u32, n, n + 1] {
                            assert!(
                                scalar_compare(lines[cmp], id, n),
                                "{s:?}: id {id:#x} against n_experts {n} must exit"
                            );
                        }
                        for id in [n - 1, 0] {
                            assert!(
                                !scalar_compare(lines[cmp], id, n),
                                "{s:?}: id {id:#x} against n_experts {n} must not exit"
                            );
                        }
                    }
                    for id in [0u32, 1, u32::MAX, i32::MIN as u32] {
                        assert!(
                            scalar_compare(lines[cmp], id, 0),
                            "{s:?}: n = 0, id {id:#x}"
                        );
                    }
                    assert!(
                        lines[..cmp]
                            .iter()
                            .all(|l| !forbidden(l.split_whitespace().next().unwrap_or(""))),
                        "{s:?}: a memory instruction precedes the guard"
                    );
                }
            }
        }
    }
}

#[test]
fn legacy_gfx1151_entries_stay_lds_free_and_mutations_are_codec_scoped() {
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            let e = gfx11::emit(s1151(codec, kind, false, Negative::None))
                .expect("legacy entry builds");
            assert!(
                e.s_text.contains(".group_segment_fixed_size: 0"),
                "{codec:?} {kind:?}"
            );
            assert!(
                !e.s_text.contains("s_barrier") && !e.s_text.contains("ds_load_b32"),
                "{codec:?} {kind:?}"
            );
            for bad in [
                Negative::WrongLut,
                Negative::WrongSignMask,
                Negative::WrongFoldEpoch,
            ] {
                assert!(
                    gfx11::emit(s1151(codec, kind, false, bad)).is_err(),
                    "{codec:?} accepted {bad:?}"
                );
            }
        }
    }
    for kind in Kind::ALL {
        assert!(gfx11::emit(s1151(Codec::E8p, kind, false, Negative::WrongPackSelector)).is_err());
    }
}

#[test]
fn e8p_gfx1151_mutations_change_the_source() {
    for kind in Kind::ALL {
        let base = gfx11::emit(s1151(Codec::E8p, kind, false, Negative::None))
            .unwrap()
            .s_text;
        for negative in Negative::E8P {
            let mutated = gfx11::emit(s1151(Codec::E8p, kind, false, negative))
                .unwrap()
                .s_text;
            assert_ne!(
                base, mutated,
                "{kind:?} {negative:?} left the source unchanged"
            );
        }
    }
}

#[test]
fn wrong_fold_epoch_gfx1151_drops_only_the_activation_sum() {
    for kind in Kind::ALL {
        for acc in [false, true] {
            let base = gfx11::emit(s1151(Codec::E8p, kind, acc, Negative::None))
                .unwrap()
                .s_text;
            let bad = gfx11::emit(s1151(Codec::E8p, kind, acc, Negative::WrongFoldEpoch))
                .unwrap()
                .s_text;
            let adds = |t: &str| t.matches("v_add_nc_u32_e32").count();
            let movs = |t: &str| t.matches("v_mov_b32_e32").count();
            assert_eq!(
                adds(&base),
                adds(&bad) + 1,
                "{kind:?} acc {acc}: exactly one `s - MAGIC` add is replaced"
            );
            assert_eq!(
                movs(&bad),
                movs(&base) + 1,
                "{kind:?} acc {acc}: by exactly one `-MAGIC` move"
            );
            assert_eq!(
                base.lines().count(),
                bad.lines().count(),
                "{kind:?} acc {acc}: instruction count unchanged"
            );
        }
    }
}

/// gfx1100 `FeatureUserSGPRInit16Bug` renumbering, as the gfx1151 SGPR number: the descriptor declares 14 user SGPRs, so
/// the workgroup ids are s14/s15 (gfx1151: s2/s3) and the kernarg `s_load_b256` block moves s8:15 -> s88:95; the E8P
/// table literals (s88:95 on gfx1151) move to the freed s8..s13 and s84,s85.
fn sgpr_1100_to_1151(n: u32, e8p: bool) -> u32 {
    match n {
        14 | 15 => n - 12,
        88..=95 => n - 80,
        8..=13 if e8p => n + 80,
        84 | 85 if e8p => n + 10,
        _ => n,
    }
}

/// `text` with every scalar register number mapped by `f` (`sN` and `s[N:M]` operands; not `s_load`, `v14`, `.sgpr_count`).
fn map_sgprs(text: &str, f: impl Fn(u32) -> u32) -> String {
    let b = text.as_bytes();
    let (mut out, mut i) = (String::with_capacity(text.len()), 0);
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'.';
    while i < b.len() {
        let starts = b[i] == b's' && (i == 0 || !word(b[i - 1]));
        if starts && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j >= b.len() || !word(b[j]) {
                out += &format!("s{}", f(text[i + 1..j].parse().unwrap()));
                i = j;
                continue;
            }
        } else if starts && text[i..].starts_with("s[") {
            let close = i + text[i..].find(']').unwrap();
            let (lo, hi) = text[i + 2..close].split_once(':').unwrap();
            out += &format!("s[{}:{}]", f(lo.parse().unwrap()), f(hi.parse().unwrap()));
            i = close + 1;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// The source with its arch name swapped, its gfx1100-only trans-use waits removed, the gfx1100 SGPR renumbering undone
/// (`gfx1100` only) and the SGPR-count directives (`.amdhsa_user_sgpr_count`, `.amdhsa_next_free_sgpr`, `.sgpr_count`)
/// blanked: they are the SGPR rule's own deltas.
fn arch_neutral(text: &str, arch: Arch, e8p: bool) -> String {
    let text = text.replace(arch.name(), "ARCH");
    let text = if arch == Arch::Gfx1100 {
        map_sgprs(&text, |n| sgpr_1100_to_1151(n, e8p))
    } else {
        text
    };
    text.lines()
        .filter(|l| l.trim() != TRANS_WAIT)
        .map(|l| {
            let t = l.trim();
            match [
                ".amdhsa_user_sgpr_count",
                ".amdhsa_next_free_sgpr",
                ".sgpr_count:",
            ]
            .into_iter()
            .find(|d| t.starts_with(d))
            {
                Some(d) => format!("{d} N"),
                None => l.to_owned(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every codec/kind/acc/mutation entry: gfx1151 differs from gfx1100 only by the arch name and by the
/// `VALUTransUseHazard` waits gfx1151 does not have; registers, LDS, kernargs and every other instruction agree.
#[test]
fn gfx1151_source_is_the_gfx1100_source_minus_arch_forced_deltas() {
    let mut dropped = 0usize;
    let mut entries = 0usize;
    for codec in Codec::ALL.into_iter().chain([Codec::E8p]) {
        let negatives: Vec<Negative> = if codec == Codec::E8p {
            std::iter::once(Negative::None)
                .chain(Negative::E8P)
                .collect()
        } else {
            vec![Negative::None]
        };
        for negative in negatives {
            for kind in Kind::ALL {
                for acc in [false, true] {
                    let a = gfx11::emit(spec(Arch::Gfx1100, codec, kind, acc, negative)).unwrap();
                    let b = gfx11::emit(s1151(codec, kind, acc, negative)).unwrap();
                    let id = format!("{codec:?} {kind:?} acc {acc} {negative:?}");
                    if codec == Codec::Mq4 && !acc && negative == Negative::None {
                        // The round-2 gfx1100 card winners (gate/up lockstep, down K=640 pipeline)
                        // are exact-gfx1100 only: gfx1151 keeps the plain loop, barrier-free.
                        assert!(a.s_text.contains("s_barrier"), "{id}");
                        assert!(!b.s_text.contains("s_barrier"), "{id}");
                        entries += 1;
                        continue;
                    }
                    let (wa, wb) = (
                        a.s_text.matches(TRANS_WAIT).count(),
                        b.s_text.matches(TRANS_WAIT).count(),
                    );
                    assert_eq!(wb, 0, "{id}: gfx1151 has no VALUTransUseHazard wait");
                    assert_eq!(
                        arch_neutral(&a.s_text, Arch::Gfx1100, codec == Codec::E8p),
                        arch_neutral(&b.s_text, Arch::Gfx1151, codec == Codec::E8p),
                        "{id}: unexpected delta beyond the trans-use waits and the gfx1100 SGPR renumbering"
                    );
                    assert_eq!(
                        a.s_text.lines().count(),
                        b.s_text.lines().count() + wa,
                        "{id}"
                    );
                    // gfx1100 declares user SGPRs padded to 14 and keeps its kernarg block at s88:95, so its
                    // SGPR high-water mark is at least 96 (E8P: already 96 on both).
                    assert_eq!(
                        a.shape.next_free_sgpr,
                        b.shape.next_free_sgpr.max(96),
                        "{id}: SGPR high-water mark"
                    );
                    assert_eq!(
                        (a.shape.next_free_vgpr, a.shape.wmma, a.shape.vmem),
                        (b.shape.next_free_vgpr, b.shape.wmma, b.shape.vmem),
                        "{id}: register plan and operation census"
                    );
                    assert_eq!(a.shape.barriers, b.shape.barriers, "{id}");
                    dropped += wa;
                    entries += 1;
                }
            }
        }
    }
    // 3 legacy codecs x 4 entries + E8P (candidate + 6 mutations) x 4 entries.
    assert!(
        entries == 12 + 28 && dropped > 0,
        "{entries} entries, {dropped} dropped trans-use waits"
    );
}

/// The module of one codec/negative on `arch`, in the committed order (BF16 gate/up, BF16 down, `_acc` pair).
fn module_source(arch: Arch, codec: Codec, negative: Negative) -> String {
    let mut entries = Vec::new();
    for acc in [false, true] {
        for kind in Kind::ALL {
            entries.push(gfx11::emit(spec(arch, codec, kind, acc, negative)).unwrap());
        }
    }
    let (text, _proof) = iu4_gemm::module(&entries, &codec.module(arch)).unwrap();
    text
}

fn image(arch: Arch, text: &str) -> Vec<u8> {
    let elf = native::assemble(text, arch).unwrap();
    native::bundle(&elf, arch, native::DEFAULT_HOST_TARGET)
}

/// The committed E8P candidates and negatives re-emit byte for byte, for both gfx11 arms.
#[test]
fn committed_e8p_images_are_byte_identical_to_the_emission() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    for arch in gfx11::ARCHES {
        let module = Codec::E8p.module(arch);
        let want = image(arch, &module_source(arch, Codec::E8p, Negative::None));
        let file = dir.join(format!("{module}.hxaco"));
        let committed = std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        assert!(
            committed == want,
            "{} differs from the emitted bundle",
            file.display()
        );
        for negative in Negative::E8P {
            let file = dir.join(format!(
                "qwen4_moe_{}_negative/{}/{module}.hxaco",
                arch.name(),
                negative.tag()
            ));
            let committed =
                std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
            let want = image(arch, &module_source(arch, Codec::E8p, negative));
            assert!(
                committed == want,
                "{} differs from the emitted bundle",
                file.display()
            );
        }
    }
}

/// The emitted image carries the `s_code_end` guard: at least 96 trailing `0xbf9f0000` dwords and a 128-byte-multiple `.text`.
#[test]
fn e8p_images_end_with_the_code_end_guard() {
    for arch in gfx11::ARCHES {
        let elf = native::assemble(&module_source(arch, Codec::E8p, Negative::None), arch).unwrap();
        let text = text_section(&elf);
        assert_eq!(
            text.len() % 128,
            0,
            "{arch:?}: .text not a whole instruction-cache line count"
        );
        let guard = text
            .rchunks(4)
            .take_while(|w| *w == 0xbf9f_0000u32.to_le_bytes())
            .count();
        assert!(guard >= 96, "{arch:?}: {guard} trailing s_code_end dwords");
    }
}

fn text_section(elf: &[u8]) -> Vec<u8> {
    let u16_at = |at: usize| u16::from_le_bytes(elf[at..at + 2].try_into().unwrap()) as usize;
    let u32_at = |at: usize| u32::from_le_bytes(elf[at..at + 4].try_into().unwrap()) as usize;
    let u64_at = |at: usize| u64::from_le_bytes(elf[at..at + 8].try_into().unwrap()) as usize;
    let (shoff, entsize, count, strndx) = (u64_at(40), u16_at(58), u16_at(60), u16_at(62));
    let header = |i: usize| shoff + i * entsize;
    let names = u64_at(header(strndx) + 24);
    for i in 0..count {
        let at = names + u32_at(header(i));
        let end = at + elf[at..].iter().position(|&c| c == 0).unwrap();
        if &elf[at..end] == b".text" {
            let (offset, size) = (u64_at(header(i) + 24), u64_at(header(i) + 32));
            return elf[offset..offset + size].to_vec();
        }
    }
    panic!("no .text section")
}
