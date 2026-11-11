// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `argmax_pair_merge_pm_gfx1201` emission checks (CPU only): deterministic text, symbol and
//! module names, gfx1201-only, 64-bit global pointer access, no buffer/LDS/scratch, register
//! limits. The GPU byte oracle is `crates/rdna-compute/tests/x1_argmax_pair_merge_gpu.rs`.
use peacemaker_isa::kernels::argmax_pair_merge_pm as pm;
use peacemaker_isa::Arch;

fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.trim().split_whitespace().next() == Some(mnemonic))
        .count()
}

#[test]
fn emission_is_deterministic_and_gfx1201_only() {
    let (a, b) = (
        pm::emit(Arch::Gfx1201).unwrap(),
        pm::emit(Arch::Gfx1201).unwrap(),
    );
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(pm::symbol(Arch::Gfx1201), "argmax_pair_merge_pm_gfx1201");
    assert_eq!(pm::module(Arch::Gfx1201), "argmax_pair_merge_pm_gfx1201");
    assert!(a.s_text.contains("argmax_pair_merge_pm_gfx1201:"));
    assert_eq!(a.shape.barriers, 0);
    assert!(
        a.shape.next_free_vgpr <= 24 && a.shape.next_free_sgpr <= 104,
        "{:?}",
        a.shape
    );
    let (emitted, text, _) = pm::emit_module(Arch::Gfx1201).unwrap();
    assert_eq!(emitted.len(), 1);
    assert!(text.contains("argmax_pair_merge_pm_gfx1201:"));
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(pm::emit(arch).is_err(), "{arch:?}");
        assert!(pm::emit_module(arch).is_err(), "{arch:?}");
    }
}

#[test]
fn instruction_mix_is_64_bit_global_only() {
    let e = pm::emit(Arch::Gfx1201).unwrap();
    let t = &e.s_text;
    assert_eq!(pm::KERNARG_BYTES, 52);
    assert_eq!(pm::BLOCK, 256);
    // i0, i1, v0, v1 loads; one store; all through 64-bit `v[lo:hi]` addresses.
    assert_eq!(count(t, "global_load_b32"), 4);
    assert_eq!(count(t, "global_store_b32"), 1);
    for l in t.lines().filter(|l| l.trim().starts_with("global_")) {
        assert!(l.contains("v[") && l.trim().ends_with("off"), "{l}");
    }
    for forbidden in ["buffer_", "ds_", "flat_", "scratch_", "s_barrier"] {
        assert!(
            !t.lines().any(|l| l.trim().starts_with(forbidden)),
            "{forbidden}"
        );
    }
    assert!(t.contains("ttmp9"));
    assert!(t.contains("v_cmp_class_f32") && t.contains("0x1f8"));
    // The full-width element index: signed 64-bit mad + 64-bit shift (no 32-bit stride narrowing).
    assert!(t.contains("v_mad_co_i64_i32") && t.contains("v_lshlrev_b64"));
}

#[test]
fn shape_contract_names_the_symbol() {
    let e = pm::emit(Arch::Gfx1201).unwrap();
    let c = pm::contract(Arch::Gfx1201, &e);
    assert_eq!(c["symbol"], "argmax_pair_merge_pm_gfx1201");
    assert_eq!(c["counts"]["global_load_b32"], 4);
    assert_eq!(c["counts"]["global_store_b32"], 1);
}

// ---- CPU interpreter over the assembled bytes ------------------------------------

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use std::sync::LazyLock;

static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| {
    let text = pm::emit_module(Arch::Gfx1201).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1201).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
    .program
});

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The hipcc law.
fn reference(v0: f32, v1: f32, i0: i32, i1: i32, w0: i32) -> i32 {
    let (f0, f1) = (v0.is_finite(), v1.is_finite());
    if f0 && (!f1 || v0 >= v1) {
        i0
    } else if f1 {
        w0 + i1
    } else {
        0
    }
}

const TAIL: usize = 16;

fn bytes_f32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}
fn bytes_i32(x: &[i32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One launch over `rows` rows. `peer_base` places the peer shard (a base near 2^32 forces the
/// 64-bit carry). Returns the output words; asserts the guard words stay untouched.
#[allow(clippy::too_many_arguments)]
fn run(
    s0: &[f32],
    s1: &[f32],
    top0: &[i32],
    top1: &[i32],
    w0: usize,
    w1: usize,
    alias: bool,
    peer_base: u64,
) -> Vec<i32> {
    let rows = top0.len();
    let emu = Emulator::new(&PROGRAM, &pm::symbol(Arch::Gfx1201)).unwrap();
    let (kp, l0p, i0p, outp) = (0x1_0000u64, 0x10_0000u64, 0x20_0000u64, 0x30_0000u64);
    let l1p = peer_base;
    let i1p = peer_base + 0x80_0000;
    let out = if alias { i0p } else { outp };
    let mut kern = Vec::new();
    for p in [l0p, i0p, l1p, i1p, out] {
        kern.extend(p.to_le_bytes());
    }
    for x in [rows as i32, w0 as i32, w1 as i32] {
        kern.extend(x.to_le_bytes());
    }
    assert_eq!(kern.len(), pm::KERNARG_BYTES as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map("l0".into(), l0p, bytes_f32(s0)).unwrap();
    let mut i0b = bytes_i32(top0);
    i0b.extend(vec![0x5A; 4 * TAIL]);
    mem.map("i0".into(), i0p, i0b).unwrap();
    mem.map("l1".into(), l1p, bytes_f32(s1)).unwrap();
    mem.map("i1".into(), i1p, bytes_i32(top1)).unwrap();
    if !alias {
        mem.map("out".into(), outp, vec![0x5A; 4 * (rows + TAIL)])
            .unwrap();
    }
    for wg in 0..rows.div_ceil(256) {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: kp,
                workgroup: [wg as u32, 0, 0],
                block: [pm::BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("wg {wg}: {e}"));
    }
    let region = mem
        .regions
        .iter()
        .find(|r| r.name == if alias { "i0" } else { "out" })
        .unwrap()
        .bytes
        .clone();
    assert!(
        region[4 * rows..].iter().all(|&b| b == 0x5A),
        "output written past the rows"
    );
    region[..4 * rows]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn sweep(w0: usize, w1: usize, peer_base: u64, seed: u64) {
    let special = [
        f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0000),
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -3.5,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        f32::MIN,
    ];
    let mut rng = Rng(seed);
    let (mut s0, mut s1) = (Vec::new(), Vec::new());
    let (mut top0, mut top1, mut want) = (Vec::new(), Vec::new(), Vec::new());
    let mut r = 0usize;
    // Every ordered pair of special values, three passes with fresh indices (> 256 rows).
    for _ in 0..3 {
        for &a in &special {
            for &b in &special {
                let (i0, i1) = (rng.below(w0), rng.below(w1));
                // Distinct finite filler so a wrongly addressed read changes the answer.
                s0.extend((0..w0).map(|k| 100.0 + (r * 7 + k) as f32));
                s1.extend((0..w1).map(|k| 200.0 + (r * 5 + k) as f32));
                s0[r * w0 + i0] = a;
                s1[r * w1 + i1] = b;
                top0.push(i0 as i32);
                top1.push(i1 as i32);
                want.push(reference(a, b, i0 as i32, i1 as i32, w0 as i32));
                r += 1;
            }
        }
    }
    assert!(r > 256);
    for alias in [false, true] {
        let got = run(&s0, &s1, &top0, &top1, w0, w1, alias, peer_base);
        for k in 0..r {
            assert_eq!(
                got[k], want[k],
                "w0 {w0} w1 {w1} alias {alias} peer_base {peer_base:#x} row {k}"
            );
        }
    }
}

#[test]
fn emulated_bytes_follow_the_finite_tie_and_address_laws() {
    // Small odd widths, a peer shard whose 64-bit address crosses 2^32, and the EP split ratio.
    sweep(7, 9, 0x4000_0000, 1);
    sweep(7, 9, 0xFFFF_FFC0, 2);
    sweep(1, 1, 0x1_0000_0000, 3);
    sweep(50, 48, 0xFFFF_F000, 4);
}
