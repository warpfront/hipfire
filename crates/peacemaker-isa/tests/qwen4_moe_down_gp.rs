// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 grouped-position down plane module `qwen4_moe_down_gp_pm_gfx1201`: three exports (the grouped A4 producer
//! and the NT4 / NT8 down twins), their relation to the incumbent entries they are byte-identical twins of (the same
//! instructions except where a record lives), and the committed bundle.
use peacemaker_isa::kernels::{moe_rotate_i4_pm, qwen4_moe_down_gp, qwen4_moe_sym};
use peacemaker_isa::{Arch, Emitted};

const ARCH: Arch = Arch::Gfx1201;

fn down(nt: u8) -> qwen4_moe_sym::Spec {
    qwen4_moe_sym::Spec {
        arch: ARCH,
        kind: qwen4_moe_sym::Kind::Down,
        nt,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
    }
}

/// Instruction lines (mnemonic + operands) of the body of `symbol`, comments and directives dropped.
fn body(e: &Emitted, symbol: &str) -> Vec<String> {
    let start = e
        .s_text
        .lines()
        .position(|l| l == format!("{symbol}:"))
        .unwrap_or_else(|| panic!("{symbol}: label"));
    e.s_text
        .lines()
        .skip(start + 1)
        .take_while(|l| !l.starts_with(".amdhsa_kernel") && !l.starts_with(".Lfunc_end"))
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.starts_with('.'))
        .map(str::to_owned)
        .collect()
}

fn count(lines: &[String], mnemonic: &str) -> usize {
    lines
        .iter()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}

#[test]
fn module_exports_are_the_three_gp_symbols() {
    assert_eq!(
        qwen4_moe_down_gp::module_name(ARCH),
        "qwen4_moe_down_gp_pm_gfx1201"
    );
    assert_eq!(
        qwen4_moe_down_gp::symbols(ARCH),
        [
            "moe_rotate_i4_gp_pm_gfx1201",
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
        ]
    );
    let (emitted, text, proof) = qwen4_moe_down_gp::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 3);
    assert_eq!(proof.module, "qwen4_moe_down_gp_pm_gfx1201");
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 3);
    for symbol in qwen4_moe_down_gp::symbols(ARCH) {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}");
    }
    // The incumbent exports are not part of this module (they stay the oracle anchors elsewhere).
    for anchor in [
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
        "moe_rotate_i4_pm_gfx1201",
    ] {
        assert!(
            !text.contains(&format!(".amdhsa_kernel {anchor}\n")),
            "{anchor}"
        );
    }
}

#[test]
fn emission_is_deterministic_and_exact_gfx1201_only() {
    let (_, ta, pa) = qwen4_moe_down_gp::emit_module(ARCH).unwrap();
    let (_, tb, pb) = qwen4_moe_down_gp::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(qwen4_moe_down_gp::emit_module(arch).is_err(), "{arch:?}");
    }
    // Only the gfx1201 MQ4 down NT4 / NT8 entries have a twin.
    for spec in [
        down(1),
        qwen4_moe_sym::Spec {
            kind: qwen4_moe_sym::Kind::GateUp,
            ..down(4)
        },
        qwen4_moe_sym::Spec {
            codec: qwen4_moe_sym::WeightCodec::Mq3I4,
            ..down(1)
        },
    ] {
        assert!(qwen4_moe_sym::emit_down_gp(spec).is_err());
    }
}

/// The producer twin is the incumbent's instruction stream with the record index `g * P + row` in place of
/// `g * L + sorted[row]`: exactly the multiply's second operand and the add's addend differ.
#[test]
fn grouped_producer_differs_from_the_incumbent_only_in_the_record_index() {
    let inc = moe_rotate_i4_pm::emit(moe_rotate_i4_pm::Spec { arch: ARCH }).unwrap();
    let gp = moe_rotate_i4_pm::emit_gp(moe_rotate_i4_pm::Spec { arch: ARCH }).unwrap();
    let a = body(&inc, "moe_rotate_i4_pm_gfx1201");
    let b = body(&gp, "moe_rotate_i4_gp_pm_gfx1201");
    assert_eq!(a.len(), b.len());
    let differing: Vec<(&String, &String)> = a.iter().zip(&b).filter(|(x, y)| x != y).collect();
    assert_eq!(
        differing.len(),
        2,
        "record index multiply and add only: {differing:?}"
    );
    assert!(
        differing[0].0.starts_with("s_mul_i32 s44, ")
            && differing[0].1.starts_with("s_mul_i32 s44, ")
    );
    assert!(
        differing[1].0.starts_with("s_add_co_i32 s44, ")
            || differing[1].0.starts_with("s_add_i32 s44, ")
    );
    assert!(
        differing[0].0.ends_with("s43") && differing[0].1.ends_with("s19"),
        "L -> P"
    );
    assert_eq!(gp.shape.next_free_vgpr, inc.shape.next_free_vgpr);
}

/// Each down twin keeps every MMA, weight/activation load, fold and store of the incumbent entry and only
/// swaps the flat-slot gather (an f32-reciprocal division per tile) for a position multiply.
#[test]
fn down_twins_keep_the_fold_and_replace_only_the_gather() {
    for nt in [4u8, 8] {
        let inc = qwen4_moe_sym::emit(down(nt)).unwrap();
        let gp = qwen4_moe_sym::emit_down_gp(down(nt)).unwrap();
        let isym = down(nt).symbol();
        let gsym = isym.replacen("_sym_pm_", "_sym_gp_pm_", 1);
        assert!(gp.s_text.contains(&format!(".globl {gsym}\n")), "{gsym}");
        let (a, b) = (body(&inc, &isym), body(&gp, &gsym));
        for mnemonic in [
            "v_wmma_i32_16x16x32_iu4",
            "buffer_load_b64",
            "buffer_load_b32",
            "buffer_store_b128",
            "v_dual_fmac_f32",
            "v_dual_mul_f32",
            "ds_swizzle_b32",
            "v_xor_b32_e32",
            "s_wait_loadcnt",
        ] {
            assert_eq!(
                count(&a, mnemonic),
                count(&b, mnemonic),
                "nt{nt} {mnemonic}"
            );
        }
        assert!(
            count(&a, "v_rcp_iflag_f32_e32") >= usize::from(nt),
            "gather divide in nt{nt}"
        );
        assert_eq!(
            count(&b, "v_rcp_iflag_f32_e32"),
            0,
            "nt{nt}: no gather divide"
        );
        assert!(b.len() < a.len(), "nt{nt}: shorter prologue");
        assert_eq!(
            gp.shape.next_free_vgpr, inc.shape.next_free_vgpr,
            "nt{nt} VGPRs"
        );
        assert_eq!(gp.shape.barriers, inc.shape.barriers);
        // The plane stride is the launch's m_total (s21) times the 72-byte record, not x_src_rows (s22).
        assert!(
            b.iter().any(|l| l.starts_with("s_mul_i32 s4, s21, 0x48")),
            "nt{nt} stride"
        );
        assert!(!b.iter().any(|l| l.starts_with("s_mul_i32 s4, s22, 0x48")));
    }
}

fn bundle_path() -> String {
    format!(
        "{}/../../kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The committed `kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco` must be exactly what the builder emits today.
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel qwen4_moe_down_gp_pm --arch gfx1201 ... --bundle)")
    });
    let (_, text, _) = qwen4_moe_down_gp::emit_module(ARCH).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "fresh emission differs from the committed bundle {path}"
    );
}
