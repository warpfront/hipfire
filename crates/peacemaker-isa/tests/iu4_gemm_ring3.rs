// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ring depth 3 of the `iu4_gemm` K-loop (`Tune { ring: RingDepth::Three }`): legality (the 27B prefill gate/up
//! node must be legal; everything the generator does not implement is refused with the numbers), every legal
//! variant emits and assembles, the depth-2 kernel is untouched, and depth 3 is scheduling only: its result bits
//! equal depth 2's on the CPU emulator at K covering every arm of the entry branch (`r = (K/256 - 1) mod 3`) and a
//! loop trip count of zero and more (`q = (K/256 - 1) / 3`).
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{
    emit, ALayout, Cacc, Epi, Mma, RingDepth, Spec, Tile, Tune, WaitPlace,
};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

const WAITS: [WaitPlace; 3] = [WaitPlace::AfterMma, WaitPlace::MidMma, WaitPlace::BeforeMma];

fn three(spec: Spec) -> Spec {
    Spec {
        tune: Tune {
            ring: RingDepth::Three,
            ..spec.tune
        },
        ..spec
    }
}

/// The 27B prefill gate/up node (`--fold k128 --tile 128x128x8 --cacc 1 --epi silu --alayout slab`).
fn node() -> Spec {
    Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::GateUpSilu)
    }
}

fn bitwise_eq(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// Bytes of the kernel's `.text` (instruction count is fixed-width per opcode; the assembled object is the truth).
fn text_bytes(spec: Spec) -> usize {
    let e = emit(spec).unwrap();
    let elf = peacemaker_isa::native::assemble(&e.s_text, spec.arch).unwrap();
    elf_text_bytes(&elf)
}

/// `.text` size of an ELF64 code object.
fn elf_text_bytes(elf: &[u8]) -> usize {
    let u16at = |o: usize| usize::from(u16::from_le_bytes(elf[o..o + 2].try_into().unwrap()));
    let u32at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
    let u64at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64at(0x28), u16at(0x3a), u16at(0x3c), u16at(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let stroff = u64at(sh(shstrndx) + 0x18);
    for i in 0..shnum {
        let name = u32at(sh(i));
        let nm = &elf[stroff + name..];
        let end = nm.iter().position(|&c| c == 0).unwrap();
        if &nm[..end] == b".text" {
            return u64at(sh(i) + 0x20);
        }
    }
    panic!("no .text");
}

#[test]
fn the_27b_node_is_legal_at_depth_3_and_depth_2_is_unchanged() {
    let s3 = three(node());
    s3.validate().unwrap();
    assert_eq!(s3.variant_name(), "k128-128x128x8-cacc1-slab-silu-r3");
    // The product symbol (the ABI) is the shipped one.
    assert_eq!(s3.symbol(), node().symbol());
    let (d2, d3) = (
        peacemaker_isa::kernels::iu4_gemm::Gen::new(node()).unwrap(),
        peacemaker_isa::kernels::iu4_gemm::Gen::new(s3).unwrap(),
    );
    // Same VGPRs (the second prefetch set takes magic8's registers), one more A/W slab of LDS.
    assert_eq!(d2.vgprs(), 187);
    assert_eq!(d3.vgprs(), 187);
    let (e2, e3) = (emit(node()).unwrap(), emit(s3).unwrap());
    assert_eq!(e2.shape.next_free_vgpr, 187);
    assert_eq!(e3.shape.next_free_vgpr, 187);
    // The rolled loop body is three groups where depth 2's is one: three times the WMMAs and fold packets.
    let (c2, c3) = (
        peacemaker_isa::kernels::iu4_gemm::hot_loop_census(&e2.s_text),
        peacemaker_isa::kernels::iu4_gemm::hot_loop_census(&e3.s_text),
    );
    let wmma = |c: &std::collections::BTreeMap<String, u32>| {
        c.iter()
            .filter(|(k, _)| k.starts_with("v_wmma_"))
            .map(|(_, n)| *n)
            .sum::<u32>()
    };
    assert_eq!(wmma(&c2), 64);
    assert_eq!(wmma(&c3), 3 * 64);
    // The dynamic launch size is the shipped one (the product launcher passes 20480); the third slab is static LDS.
    assert_eq!(node().lds_launch(), 20480);
    assert_eq!(node().lds_fixed(), 0);
    assert_eq!(s3.lds_launch(), 20480);
    assert_eq!(s3.lds_fixed(), 8192);
    assert_eq!(s3.lds_total(), 28672);
    println!(
        "27B gate_up: VGPRs {} -> {}, LDS {} -> {} B (dynamic {} + static {}), .text {} -> {} B",
        d2.vgprs(),
        d3.vgprs(),
        node().lds_total(),
        s3.lds_total(),
        s3.lds_launch(),
        s3.lds_fixed(),
        text_bytes(node()),
        text_bytes(s3)
    );
}

/// Everything that has a depth-3 generator emits, assembles and is deterministic; the three wait placements
/// differ from each other.
#[test]
fn every_legal_depth_3_variant_emits_and_assembles() {
    for (epi, act, tile) in [
        (Epi::Set, ALayout::Token, Tile::T128x128x8),
        (Epi::Set, ALayout::Slab, Tile::T128x128x8),
        (Epi::Add, ALayout::Token, Tile::T128x128x8),
        (Epi::Add, ALayout::Slab, Tile::T128x128x8),
        (Epi::AddSigmoid, ALayout::Token, Tile::T128x128x8),
        (Epi::GateUpSilu, ALayout::Token, Tile::T128x128x8),
        (Epi::GateUpSilu, ALayout::Slab, Tile::T128x128x8),
        (Epi::GateUpSiluBf16, ALayout::Slab, Tile::T128x128x8),
        (Epi::GateUpSiluBf16, ALayout::Token, Tile::T128x128x8),
        (Epi::Set, ALayout::Token, Tile::T256x128x16),
        (Epi::Add, ALayout::Token, Tile::T256x128x16),
        (Epi::GateUpSilu, ALayout::Token, Tile::T256x128x16),
    ] {
        let mut texts = Vec::new();
        for wait in WAITS {
            let s = Spec {
                act,
                tile,
                tune: Tune {
                    ring: RingDepth::Three,
                    wait,
                    ..Tune::default()
                },
                ..Spec::control(epi)
            };
            let e = emit(s).unwrap_or_else(|e| panic!("{} : {e}", s.variant_name()));
            peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
            assert_eq!(emit(s).unwrap().s_text, e.s_text, "{}", s.variant_name());
            texts.push(e.s_text);
        }
        assert!(texts[0] != texts[1] && texts[1] != texts[2] && texts[0] != texts[2]);
        // The depth-2 kernel of the same spec is a different program.
        let d2 = emit(Spec {
            act,
            tile,
            ..Spec::control(epi)
        })
        .unwrap()
        .s_text;
        assert!(d2 != texts[0]);
    }
}

/// Static plus dynamic LDS covers the depth-3 layout for every legal depth-3 spec, the dynamic launch size and
/// every depth-2 spec's descriptor (`group_segment_fixed_size: 0`) are the shipped ones, and the depth-3 object
/// declares the extra slab as static LDS in its descriptor.
#[test]
fn depth_3_declares_its_extra_lds_as_static_and_keeps_the_launch_size() {
    let mut checked = 0;
    for tile in [Tile::T128x128x8, Tile::T256x128x16] {
        for epi in [
            Epi::Set,
            Epi::Add,
            Epi::AddSigmoid,
            Epi::GateUpSilu,
            Epi::GateUpSiluBf16,
            Epi::QkvzaGdn,
        ] {
            for act in [ALayout::Token, ALayout::Slab] {
                let base = Spec {
                    act,
                    tile,
                    ..Spec::control(epi)
                };
                if base.validate().is_err() {
                    continue;
                }
                // Depth 2: no static LDS, the shipped launch size.
                assert_eq!(base.lds_fixed(), 0);
                assert_eq!(base.lds_total(), base.lds_launch());
                let d2 = emit(base).unwrap().s_text;
                assert!(
                    d2.contains(
                        ".group_segment_fixed_size: 0
"
                    ),
                    "{}",
                    base.variant_name()
                );
                let s3 = three(base);
                if s3.validate().is_err() {
                    continue;
                }
                let l = s3.layout();
                assert_eq!(s3.lds_launch(), base.lds_launch(), "{}", s3.variant_name());
                assert!(
                    s3.lds_fixed() + s3.lds_launch() >= l.end,
                    "{}: {} + {} < {}",
                    s3.variant_name(),
                    s3.lds_fixed(),
                    s3.lds_launch(),
                    l.end
                );
                assert!(l.fixed > 0 && l.end > base.layout().end);
                let d3 = emit(s3).unwrap().s_text;
                assert!(
                    d3.contains(&format!(".group_segment_fixed_size: {}\n", l.fixed)),
                    "{}",
                    s3.variant_name()
                );
                checked += 1;
            }
        }
    }
    // 12 dense variants with a depth-3 generator.
    assert_eq!(checked, 12);
}

#[test]
fn depth_3_refuses_what_it_does_not_generate() {
    // The fused GDN projection re-lays out the K-loop's LDS in its epilogue.
    let gdn = three(Spec::control(Epi::QkvzaGdn)).validate().unwrap_err();
    assert!(gdn.contains("ring depth 3") && gdn.contains("GDN"), "{gdn}");
    assert!(emit(three(Spec::control(Epi::QkvzaGdn))).is_err());
    // The depth-3 schedule composes the dense fine steps only; the sparse ones have no depth-3 composition.
    for mma in [Mma::S24X64, Mma::S24X32] {
        let e = three(Spec {
            mma,
            ..Spec::control(Epi::Set)
        })
        .validate()
        .unwrap_err();
        assert!(e.contains("ring depth 3") && e.contains("dense"), "{e}");
    }
    // Everything else the base spec refuses is still refused first.
    let two_acc = three(Spec {
        cacc: Cacc::Two,
        ..Spec::control(Epi::Set)
    })
    .validate()
    .unwrap_err();
    assert!(two_acc.contains("Cacc::Two"), "{two_acc}");
}

/// Depth 3 is scheduling only: the result bits equal depth 2's (and the CPU fold's) at K = 256 (no arm group, no
/// trip), 512 (r = 1), 768 (r = 2), 1024 (r = 0, one trip) and 5120 (r = 1, six trips). M = 136, N = 40 cover a
/// second, partial row tile and a partial token tile.
fn identity(epi: Epi, act: ALayout, tile: Tile, wait: WaitPlace, ks: &[usize]) {
    for &k in ks {
        let p = Problem::synth(136, 40, k, 0x9e37_79b9_7f4a_7c15 ^ k as u64);
        let base = Spec {
            act,
            tile,
            ..Spec::control(epi)
        };
        let want = run_gemm(base, &p).unwrap();
        let got = run_gemm(
            three(Spec {
                tune: Tune {
                    wait,
                    ..Tune::default()
                },
                ..base
            }),
            &p,
        )
        .unwrap();
        assert_eq!(
            bitwise_eq(&got, &want),
            0,
            "{epi:?} {act:?} {tile:?} wait {} K={k}",
            wait.name()
        );
        if wait == WaitPlace::AfterMma {
            assert_eq!(
                bitwise_eq(&got, &reference(&p, epi == Epi::Add)),
                0,
                "{epi:?} {act:?} {tile:?} K={k} vs the CPU fold"
            );
        }
    }
}

const KS: [usize; 5] = [256, 512, 768, 1024, 5120];

#[test]
fn depth_3_is_bit_identical_set_token() {
    require_rocm_tool!("llvm-mc");
    identity(
        Epi::Set,
        ALayout::Token,
        Tile::T128x128x8,
        WaitPlace::AfterMma,
        &KS,
    );
}

#[test]
fn depth_3_is_bit_identical_set_slab() {
    require_rocm_tool!("llvm-mc");
    identity(
        Epi::Set,
        ALayout::Slab,
        Tile::T128x128x8,
        WaitPlace::AfterMma,
        &KS,
    );
}

#[test]
fn depth_3_is_bit_identical_add() {
    require_rocm_tool!("llvm-mc");
    identity(
        Epi::Add,
        ALayout::Token,
        Tile::T128x128x8,
        WaitPlace::AfterMma,
        &KS,
    );
}

#[test]
fn depth_3_is_bit_identical_with_every_wait_placement() {
    require_rocm_tool!("llvm-mc");
    for wait in [WaitPlace::MidMma, WaitPlace::BeforeMma] {
        identity(
            Epi::Set,
            ALayout::Slab,
            Tile::T128x128x8,
            wait,
            &[256, 512, 768, 1024],
        );
    }
}

#[test]
fn depth_3_is_bit_identical_on_the_16_wave_tile() {
    require_rocm_tool!("llvm-mc");
    identity(
        Epi::Set,
        ALayout::Token,
        Tile::T256x128x16,
        WaitPlace::AfterMma,
        &[256, 512, 768, 1024],
    );
}

/// The certifier the tuner applies to every candidate: M7's lift/emit identity and whole-program analyses (wait and
/// hazard replay over the loop back-edges and the entry branch's joins, barrier pairing and the barrier LDS-drain
/// rule). The depth-3 LDS allocation is bounded by the emulator runs above (every LDS access of every executed arm
/// is checked against the launch size).
#[cfg(feature = "toolchain")]
#[test]
fn depth_3_kernels_pass_m7() {
    for (epi, act, tile) in [
        (Epi::GateUpSilu, ALayout::Slab, Tile::T128x128x8),
        (Epi::GateUpSiluBf16, ALayout::Slab, Tile::T128x128x8),
        (Epi::Set, ALayout::Token, Tile::T128x128x8),
        (Epi::Add, ALayout::Slab, Tile::T128x128x8),
        (Epi::AddSigmoid, ALayout::Token, Tile::T128x128x8),
        (Epi::Set, ALayout::Token, Tile::T256x128x16),
    ] {
        for wait in WAITS {
            let s = Spec {
                act,
                tile,
                tune: Tune {
                    ring: RingDepth::Three,
                    wait,
                    ..Tune::default()
                },
                ..Spec::control(epi)
            };
            let e = emit(s).unwrap();
            let elf = peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
            let path = std::env::temp_dir().join(format!(
                "iu4_ring3_{}_{}.elf",
                std::process::id(),
                s.variant_name()
            ));
            std::fs::write(&path, &elf).unwrap();
            let receipt = peacemaker_isa::pm_check::m7(&path, "gfx1201", &s.symbol())
                .unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
            std::fs::remove_file(&path).ok();
            println!("{} {receipt}", s.variant_name());
        }
    }
}
