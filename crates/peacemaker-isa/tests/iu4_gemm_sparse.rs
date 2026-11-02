// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The 2:4-sparse `iu4_gemm` kernels (`Mma::S24X64`, `Mma::S24X32`): naming, validation, the emitted instruction
//! structure, the block API's refusals and pinned digests. The numerical oracle is `iu4_gemm_sparse_emu.rs`.
use peacemaker_isa::kernels::iu4_gemm::{
    self, blocks, ALayout, Cacc, Epi, Fold, Gen, Mma, Spec, Tile, Wg, END,
};
use peacemaker_isa::{Arch, Builder};
use sha2::{Digest, Sha256};

fn spec(epi: Epi, mma: Mma) -> Spec {
    Spec {
        mma,
        ..Spec::control(epi)
    }
}

const SPARSE: [Mma; 2] = [Mma::S24X64, Mma::S24X32];

fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}

#[test]
fn mma_parses_and_names() {
    for (s, m) in [
        ("dense", Mma::Dense),
        ("s24x64", Mma::S24X64),
        ("s24x32", Mma::S24X32),
    ] {
        assert_eq!(s.parse::<Mma>().unwrap(), m);
        assert_eq!(m.name(), s);
    }
    assert!("s24x16".parse::<Mma>().unwrap_err().contains("unknown mma"));
}

#[test]
fn dense_naming_is_unchanged_and_sparse_appends_its_suffix() {
    let dense = spec(Epi::Set, Mma::Dense);
    assert_eq!(dense.variant_name(), "k128-128x128x8-cacc1-set");
    assert_eq!(
        dense.symbol(),
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1"
    );
    for (m, v, s) in [
        (Mma::S24X64, "s24x64", "_s24x64"),
        (Mma::S24X32, "s24x32", "_s24x32"),
    ] {
        let sp = spec(Epi::Add, m);
        assert_eq!(sp.variant_name(), format!("k128-128x128x8-cacc1-add-{v}"));
        assert_eq!(
            sp.symbol(),
            format!("gemm_mq4g256v2_residual_mmq_iu4_full_add_b1{s}")
        );
    }
    // from_knobs keeps its signature and returns Dense.
    let k = Spec::from_knobs("k128", "128x128x8", "1", "set", "token", Arch::Gfx1201).unwrap();
    assert_eq!(k.mma, Mma::Dense);
}

#[test]
fn validate_names_the_unsupported_axis() {
    for mma in SPARSE {
        let ok = spec(Epi::Set, mma);
        ok.validate().unwrap();
        spec(Epi::Add, mma).validate().unwrap();
        let name = mma.name();
        let cases: [(Spec, &str); 8] = [
            (
                Spec {
                    fold: Fold::K256Shared,
                    ..ok
                },
                "needs Fold::K128",
            ),
            (
                Spec {
                    fold: Fold::K256Pow2,
                    ..ok
                },
                "needs Fold::K128",
            ),
            (
                Spec {
                    tile: Tile::T256x128x16,
                    ..ok
                },
                "needs Tile::T128x128x8",
            ),
            (
                Spec {
                    cacc: Cacc::Two,
                    ..ok
                },
                "needs Cacc::One",
            ),
            (
                Spec {
                    act: ALayout::Slab,
                    ..ok
                },
                "needs ALayout::Token",
            ),
            (
                Spec {
                    epi: Epi::GateUpSilu,
                    ..ok
                },
                "supports Epi::Set and Epi::Add only",
            ),
            (
                Spec {
                    epi: Epi::AddSigmoid,
                    ..ok
                },
                "supports Epi::Set and Epi::Add only",
            ),
            (
                Spec {
                    epi: Epi::QkvzaGdn,
                    ..ok
                },
                "supports Epi::Set and Epi::Add only",
            ),
        ];
        for (s, want) in cases {
            let e = s.validate().unwrap_err();
            assert!(e.contains(want) && e.contains(name), "{e}");
            assert!(iu4_gemm::emit(s).is_err());
        }
        let e = Spec {
            epi: Epi::GateUpSiluBf16,
            ..ok
        }
        .validate()
        .unwrap_err();
        assert!(e.contains("supports Epi::Set and Epi::Add only"), "{e}");
        let e = Spec {
            arch: Arch::Gfx1151,
            ..ok
        }
        .validate()
        .unwrap_err();
        assert!(e.contains("gfx1201 only"), "{e}");
    }
}

/// The K-loop's instruction structure: per K128 block one SWMMAC per accumulator per slab (x64) or per sub-block
/// (x32), no dense WMMA, magic seeds for every accumulator; the kernel has four blocks (two trip blocks, two peeled).
#[test]
fn sparse_kernels_replace_every_wmma_with_a_swmmac() {
    let dense = iu4_gemm::emit(spec(Epi::Set, Mma::Dense)).unwrap().s_text;
    assert_eq!(count(&dense, "v_wmma_i32_16x16x32_iu4"), 4 * 32);
    assert_eq!(count(&dense, "v_swmmac_i32_16x16x64_iu4"), 0);
    for (mma, mnemonic, per_block) in [
        (Mma::S24X64, "v_swmmac_i32_16x16x64_iu4", 16),
        (Mma::S24X32, "v_swmmac_i32_16x16x32_iu4", 32),
    ] {
        for epi in [Epi::Set, Epi::Add] {
            let e = iu4_gemm::emit(spec(epi, mma)).unwrap();
            let t = &e.s_text;
            assert_eq!(count(t, mnemonic), 4 * per_block, "{mma:?}");
            assert_eq!(count(t, "v_wmma_i32_16x16x32_iu4"), 0, "{mma:?}");
            // 64 seed moves per block (8 accumulators x 8 VGPRs).
            let seeds = t
                .lines()
                .filter(|l| {
                    let l = l.trim_start();
                    l.starts_with("v_mov_b32_e32 v")
                        && l.rsplit(", v")
                            .next()
                            .and_then(|r| r.trim().parse::<u32>().ok())
                            .is_some_and(|r| (128..136).contains(&r))
                })
                .count();
            assert_eq!(seeds, 4 * 64, "{mma:?}");
            assert_eq!(e.proof.variant, spec(epi, mma).variant_name());
            assert!(t.contains(&format!(".globl {}", spec(epi, mma).symbol())));
            // The hot loop counts SWMMAC with the matrix instructions, not as VALU slots.
            let census = iu4_gemm::hot_loop_census(t);
            assert_eq!(
                census.get(mnemonic).copied(),
                Some(2 * per_block as u32),
                "{mma:?}"
            );
            assert!(census.get("v_wmma_i32_16x16x32_iu4").is_none());
        }
    }
}

#[test]
fn sparse_x32_uses_both_index_keys_x64_none() {
    let x64 = iu4_gemm::emit(spec(Epi::Set, Mma::S24X64)).unwrap().s_text;
    assert!(!x64.contains("index_key"));
    let x32 = iu4_gemm::emit(spec(Epi::Set, Mma::S24X32)).unwrap().s_text;
    let keyed = x32.matches("index_key:1").count();
    assert_eq!(keyed, 4 * 16);
    for t in [&x64, &x32] {
        assert!(t
            .lines()
            .filter(|l| l.contains("v_swmmac_"))
            .all(|l| l.contains("neg_lo:[1,1,0]")));
        assert!(!t.contains("clamp"));
    }
}

#[test]
fn sparse_kernels_assemble_with_the_native_assembler() {
    for mma in SPARSE {
        for epi in [Epi::Set, Epi::Add] {
            let t = iu4_gemm::emit(spec(epi, mma)).unwrap().s_text;
            peacemaker_isa::native::assemble(&t, Arch::Gfx1201)
                .unwrap_or_else(|e| panic!("{mma:?} {epi:?}: {e}"));
        }
    }
}

/// Dense blocks refuse a sparse generator and the reverse; the refusal names the right block family.
#[test]
fn block_families_refuse_the_other_spec() {
    for (mma, dense_block) in [
        (Mma::S24X64, true),
        (Mma::S24X32, true),
        (Mma::Dense, false),
    ] {
        let g = Gen::new(spec(Epi::Set, mma)).unwrap();
        let mut b = Builder::new(g.kernel_spec(), g.plan().unwrap());
        let mut wg = Wg::new(&mut b).unwrap();
        let lds = blocks::declare_lds(&mut wg, &g).unwrap();
        let end = wg.exit(END).unwrap();
        let lds = blocks::prologue(&mut wg, &g, &end, lds).unwrap();
        let e = if dense_block {
            blocks::block(&mut wg, &g, "x", 0, true, ".Lnext", true, lds).err()
        } else {
            blocks::sparse_block(&mut wg, &g, "x", 0, true, ".Lnext", true, lds).err()
        }
        .expect("the wrong block family must be refused");
        assert!(
            e.contains(if dense_block {
                "use the sparse blocks"
            } else {
                "use the dense blocks"
            }),
            "{e}"
        );
    }
}

/// The sparse fine steps (`sparse_block_begin`, `sparse_fragments`,
/// `sparse_wmma_group`, `sparse_scale_loads`, `sparse_scale_tail`) refuse a
/// dense spec; x64 has one SWMMAC bundle per slab (no sub-block 1, no `_s1`
/// scale tail); the x32 sub-block is 0 or 1.
#[test]
fn sparse_fine_steps_refuse_the_wrong_spec() {
    for mma in [Mma::Dense, Mma::S24X64, Mma::S24X32] {
        let g = Gen::new(spec(Epi::Set, mma)).unwrap();
        let mut b = Builder::new(g.kernel_spec(), g.plan().unwrap());
        let mut wg = Wg::new(&mut b).unwrap();
        let lds = blocks::declare_lds(&mut wg, &g).unwrap();
        let end = wg.exit(END).unwrap();
        let (ra, rw, rd, rs) = blocks::prologue(&mut wg, &g, &end, lds).unwrap();
        let begin = blocks::sparse_block_begin(
            &mut wg,
            &g,
            "x",
            0,
            ".Lnext",
            true,
            ra.cur_index(),
            rd.cur_index(),
        );
        if mma == Mma::Dense {
            let frag = blocks::sparse_fragments(&mut wg, &g, &ra, &rw);
            let wmma = blocks::sparse_wmma_group(wg.isa(), &g, 0, true, false);
            let loads = blocks::sparse_scale_loads(&mut wg, &g, ".Lx0", &rs);
            let tail = blocks::sparse_scale_tail(&mut wg, &g, ".Lx1", &rs);
            for (what, r) in [
                ("block_begin", begin),
                ("fragments", frag),
                ("wmma", wmma),
                ("scale_loads", loads),
                ("scale_tail", tail),
            ] {
                let e = r.expect_err(what);
                assert!(e.contains("the spec is dense"), "{what}: {e}");
            }
            continue;
        }
        // The steps in the block's order: head, fragments, scale loads (the
        // SWMMAC groups' emission is the twin's and the emulator's to check:
        // here only their refusals, which emit nothing).
        begin.unwrap_or_else(|e| panic!("{mma:?} block_begin: {e}"));
        blocks::sparse_fragments(&mut wg, &g, &ra, &rw).unwrap();
        assert!(
            blocks::sparse_wmma_group(wg.isa(), &g, 2, false, false).is_err(),
            "sub-block 2 is not a sub-block"
        );
        if mma == Mma::S24X64 {
            // x64's one bundle covers the slab: there is no sub-block 1.
            let e = blocks::sparse_wmma_group(wg.isa(), &g, 1, false, true)
                .expect_err("x64 sub-block 1");
            assert!(e.contains("covers the whole K64 slab"), "{e}");
        }
        blocks::sparse_scale_loads(&mut wg, &g, ".Lx0", &rs).unwrap();
        let tail = blocks::sparse_scale_tail(&mut wg, &g, ".Lx1", &rs);
        if mma == Mma::S24X64 {
            let e = tail.expect_err("x64 scale tail");
            assert!(e.contains("no sub-block 1"), "{e}");
        } else {
            tail.unwrap();
        }
    }
}

/// Digests of the sparse kernels' `s_text`, pinned so a schedule change is deliberate.
const PINNED: &[(&str, &str)] = &[
    (
        "k128-128x128x8-cacc1-set-s24x64",
        "588df0624ab772e15cfddb4adbd19d99a444d06e8423ee889cbaa4f951b98d3e",
    ),
    (
        "k128-128x128x8-cacc1-add-s24x64",
        "35c2f78dc01209744964b42aaa5f4737c321ed1b665f520804fcfdcd6991df2f",
    ),
    (
        "k128-128x128x8-cacc1-set-s24x32",
        "e6c062d95eef8d2e3d8b1ec1c0f782c8baa6bb83ea7984b4b858093f0641a70c",
    ),
    (
        "k128-128x128x8-cacc1-add-s24x32",
        "84ababbc25317e5b38cbb5ce0c66d0cc88a8c0c19dfdf11b176bdd7df2d55d0a",
    ),
];

#[test]
fn iu4_gemm_sparse_emit_digests_pinned() {
    let mut got = Vec::new();
    for mma in SPARSE {
        for epi in [Epi::Set, Epi::Add] {
            let s = spec(epi, mma);
            let t = iu4_gemm::emit(s).unwrap().s_text;
            got.push((
                s.variant_name(),
                format!("{:x}", Sha256::digest(t.as_bytes())),
            ));
        }
    }
    if std::env::var_os("IU4_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, PINNED, "the sparse iu4_gemm emitted bytes changed");
}
