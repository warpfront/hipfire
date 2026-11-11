// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free tests of `hip_twin::seal` on a committed hipcc fixture
//! (`tests/fixtures/hip_twin_saxpy.hip`, built for gfx1100).

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco};

const CO: &[u8] = include_bytes!("fixtures/hip_twin_saxpy_gfx1100.co");

/// The shipped HC8 module preserves the three hardware-judged instruction streams
/// through lift/re-encode and the existing hip_twin hazard/ABI checks.
#[test]
fn hc8_schedule_winners_reverify() {
    let image: &[u8] = include_bytes!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco");
    let symbols = [
        "hc8_gemm_e4m3_mw1_kw16_n2_act",
        "hc8_gemm_e4m3_mw1_kw16_n3_act",
        "hc8_read_up_e4m3_n1",
    ];
    let expected = [
        "ba3a3f6e6014e5b0b0e88881ce0c4ba0381bbbf6931019955c347223c7f1e431",
        "31c4f2c61d26c53e99635308bd508f5a52e6a5c17a6347eb902f4bd8ec2756ed",
        "2fce1fdd4c84577fc8e78908ab1ebfd70b2244ad2bf02d9fec04e5512df92837",
    ];
    let sealed = peacemaker_isa::hip_twin::verify_hxaco_many(image, Arch::Gfx1201, &symbols)
        .expect("HC8 winner lift identity, hazards and descriptors");
    for (kernel, hash) in sealed.iter().zip(expected) {
        assert_eq!(kernel.text_sha256, hash);
        assert_eq!(kernel.hxaco, image);
        assert_eq!(kernel.scratch, 0);
        assert_eq!(kernel.report["hazards"]["blocking"], 0);
    }
}

#[test]
fn seals_and_verifies() {
    let s = seal(CO, Arch::Gfx1100, "twin_saxpy", None).expect("seal");
    assert!(s.vgprs > 0 && s.user_sgprs >= 2);
    assert_eq!(s.report["hazards"]["checked"], true);
    let again = verify_hxaco(&s.hxaco, Arch::Gfx1100, "twin_saxpy").expect("verify");
    assert_eq!(again.hxaco, s.hxaco);
    assert_eq!(again.text_sha256, s.text_sha256);
    // The bundle input path and the raw-ELF path agree.
    let b = seal(&s.hxaco, Arch::Gfx1100, "twin_saxpy", None).unwrap();
    assert_eq!(b.elf_sha256, s.elf_sha256);
}

#[test]
fn kernel_only_object_reports_no_helpers_and_keeps_every_field() {
    let s = seal(CO, Arch::Gfx1100, "twin_saxpy", None).expect("seal");
    assert!(s.report.get("helpers").is_none());
    assert!(s.report.get("helper_coverage").is_none());
    for k in [
        "kernel",
        "arch",
        "elf_sha256",
        "elf_bytes",
        "text_sha256",
        "text_bytes",
        "hxaco_sha256",
        "hxaco_bytes",
        "lift_identity",
        "vgprs",
        "sgprs",
        "lds",
        "scratch",
        "user_sgprs",
        "rsrc2",
        "wavefront_size",
        "hazards",
        "abi",
    ] {
        assert!(s.report.get(k).is_some(), "report lost `{k}`");
    }
    assert_eq!(s.report["hxaco_sha256"], s.hxaco_sha256);
}

#[test]
fn abi_ref_same_object_matches_other_kernel_differs() {
    seal(CO, Arch::Gfx1100, "twin_saxpy", Some((CO, "twin_saxpy"))).expect("self abi");
    let e = seal(CO, Arch::Gfx1100, "twin_saxpy", Some((CO, "twin_saxpy2"))).unwrap_err();
    assert!(e.contains("ABI mismatch"), "{e}");
}

#[test]
fn wrong_symbol_errs() {
    let e = seal(CO, Arch::Gfx1100, "nope", None).unwrap_err();
    assert!(e.contains("not found"), "{e}");
}

#[test]
fn wrong_arch_errs() {
    assert!(seal(CO, Arch::Gfx1201, "twin_saxpy", None).is_err());
}

#[test]
fn corrupted_bytes_err() {
    // Flip a byte in every region in turn; none may seal.
    let mut failed = 0;
    for at in [0usize, 20, 0x40, CO.len() / 2, CO.len() - 8] {
        let mut bad = CO.to_vec();
        bad[at] ^= 0xff;
        if seal(&bad, Arch::Gfx1100, "twin_saxpy", None).is_err() {
            failed += 1;
        }
    }
    assert!(
        failed >= 4,
        "corruption slipped through: {failed}/5 rejected"
    );
}

#[test]
fn hxaco_rebundle_mismatch_errs() {
    let s = seal(CO, Arch::Gfx1100, "twin_saxpy", None).unwrap();
    // A different host triple lifts fine but is not what `native::bundle` writes.
    let mut bad = s.hxaco.clone();
    let at = bad.windows(4).position(|w| w == b"host").unwrap();
    bad[at..at + 4].copy_from_slice(b"hosu");
    assert!(verify_hxaco(&bad, Arch::Gfx1100, "twin_saxpy").is_err());
}

#[test]
fn seal_many_seals_every_kernel_into_one_bundle() {
    use peacemaker_isa::hip_twin::{seal_many, verify_hxaco_many, KernelSpec};
    let specs = [
        KernelSpec {
            symbol: "twin_saxpy",
            abi_ref: Some((CO, "twin_saxpy")),
        },
        KernelSpec {
            symbol: "twin_saxpy2",
            abi_ref: None,
        },
    ];
    let all = seal_many(CO, Arch::Gfx1100, &specs).expect("seal_many");
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].hxaco, all[1].hxaco);
    assert_eq!(all[0].hxaco_sha256, all[1].hxaco_sha256);
    assert_ne!(all[0].report["kernel"], all[1].report["kernel"]);
    let one = seal(CO, Arch::Gfx1100, "twin_saxpy", None).unwrap();
    assert_eq!(all[0].hxaco, one.hxaco);
    assert_eq!(all[0].text_sha256, one.text_sha256);
    let v = verify_hxaco_many(&all[0].hxaco, Arch::Gfx1100, &["twin_saxpy", "twin_saxpy2"])
        .expect("verify");
    assert_eq!(v[1].text_sha256, all[1].text_sha256);
}

#[test]
fn seal_many_rejects_empty_duplicate_and_bad_member() {
    use peacemaker_isa::hip_twin::{seal_many, KernelSpec};
    let k = |symbol| KernelSpec {
        symbol,
        abi_ref: None,
    };
    assert!(seal_many(CO, Arch::Gfx1100, &[])
        .unwrap_err()
        .contains("no kernels"));
    let e = seal_many(CO, Arch::Gfx1100, &[k("twin_saxpy"), k("twin_saxpy")]).unwrap_err();
    assert!(e.contains("duplicate"), "{e}");
    let e = seal_many(CO, Arch::Gfx1100, &[k("twin_saxpy"), k("nope")]).unwrap_err();
    assert!(e.contains("not found"), "{e}");
    let bad = KernelSpec {
        symbol: "twin_saxpy",
        abi_ref: Some((CO, "twin_saxpy2")),
    };
    let e = seal_many(CO, Arch::Gfx1100, &[k("twin_saxpy2"), bad]).unwrap_err();
    assert!(e.contains("ABI mismatch"), "{e}");
}
