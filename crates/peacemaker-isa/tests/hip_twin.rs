// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free tests of `hip_twin::seal` on a committed hipcc fixture
//! (`tests/fixtures/hip_twin_saxpy.hip`, built for gfx1100).

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco};

const CO: &[u8] = include_bytes!("fixtures/hip_twin_saxpy_gfx1100.co");

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
