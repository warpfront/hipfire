// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 R128 A4 producers `moe_rotate_i4_pm_gfx1201` (record `g * L + sorted[row]`) and its grouped-position twin
//! `moe_rotate_i4_gp_pm_gfx1201` (record `g * P + row`), run to completion on the PeaceMaker emulator: the emitted
//! kernels are assembled, lifted and executed unmodified over the whole `[P, K/128]` grid on real kernargs and exactly
//! sized buffers, and every record byte is compared with the Rust reference of `support/moe_rotate_ref.rs` (the CPU
//! twin of the quantizer). Pad rows (`sorted[row] < 0`) and slots no row names stay at their sentinel.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::moe_rotate_i4_pm::{self as mr, Spec};
use peacemaker_isa::{native, Arch};

#[path = "support/moe_rotate_ref.rs"]
mod reference;
use reference::{bf16_rne, record, signs, Rng};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const KARG: u64 = 0x1_0000;
const H: u64 = 0x100_0000;
const SORTED: u64 = 0x200_0000;
const XQ: u64 = 0x300_0000;
const S1: u64 = 0x400_0000;
const S2: u64 = 0x401_0000;
const SENTINEL: u8 = 0xa5;
const REC: usize = 72;

struct Problem {
    k: usize,
    p: usize,
    l: usize,
    sorted: Vec<i32>,
    h: Vec<u16>,
}

/// `p` rows of `k` bf16 columns; a pad row, a zero row, a large-magnitude row and slots of `l = p + 3` that no row names.
fn problem(k: usize, p: usize, seed: u64) -> Problem {
    let l = p + 3;
    let mut rng = Rng(seed | 1);
    let mut slots: Vec<i32> = (0..l as i32).collect();
    for i in (1..slots.len()).rev() {
        slots.swap(i, rng.below(i + 1));
    }
    let mut next = 0;
    let sorted: Vec<i32> = (0..p)
        .map(|row| {
            if row == 2 || row == p - 1 {
                return -1;
            }
            next += 1;
            slots[next - 1]
        })
        .collect();
    let mut h = vec![0u16; p * k];
    for row in 0..p {
        for c in 0..k {
            let u = (rng.next() >> 40) as f32 / (1u32 << 24) as f32 - 0.5;
            let scale = match row {
                3 => 0.0,
                4 => 3000.0,
                _ => 2.0 + (c % 7) as f32,
            };
            h[row * k + c] = bf16_rne(u * scale);
        }
    }
    Problem { k, p, l, sorted, h }
}

fn run(spec: Spec, gp: bool, pr: &Problem) -> Result<Vec<u8>, String> {
    let emitted = if gp {
        mr::emit_gp(spec)
    } else {
        mr::emit(spec)
    }?;
    let elf = native::assemble(&emitted.s_text, ARCH)?;
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?
    .program;
    let symbol = if gp {
        mr::symbol_gp(ARCH)
    } else {
        spec.symbol()
    };
    let emu = Emulator::new(&program, &symbol)?;
    let groups = pr.k / 128;
    let records = if gp { groups * pr.p } else { groups * pr.l };
    let mut karg = Vec::new();
    for ptr in [H, SORTED, XQ, S1, S2] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [pr.k, pr.p, pr.l] {
        karg.extend((v as u32).to_le_bytes());
    }
    karg.resize(mr::KERNARG_BYTES as usize, 0);
    let bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (s1, s2) = (signs(mr::SIGN_SEEDS.0), signs(mr::SIGN_SEEDS.1));
    let f = |s: &[f32; 128]| bytes(&s.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg)?;
    mem.map(
        "h".into(),
        H,
        pr.h.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )?;
    mem.map(
        "sorted".into(),
        SORTED,
        bytes(&pr.sorted.iter().map(|&x| x as u32).collect::<Vec<_>>()),
    )?;
    mem.map("xq".into(), XQ, vec![SENTINEL; records * REC])?;
    mem.map("s1".into(), S1, f(&s1))?;
    mem.map("s2".into(), S2, f(&s2))?;
    for row in 0..pr.p {
        for g in 0..groups {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [row as u32, g as u32, 0],
                    block: [mr::WORKGROUP_SIZE, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("workgroup ({row},{g}): {e}"))?;
        }
    }
    Ok(mem.read(XQ, records * REC)?.to_vec())
}

fn check(gp: bool, pr: &Problem, got: &[u8]) {
    let groups = pr.k / 128;
    let (s1, s2) = (signs(mr::SIGN_SEEDS.0), signs(mr::SIGN_SEEDS.1));
    let mut expect = vec![None; got.len() / REC];
    for row in 0..pr.p {
        let slot = pr.sorted[row];
        if slot < 0 {
            continue;
        }
        for g in 0..groups {
            let rec = record(&pr.h[row * pr.k + g * 128..][..128], &s1, &s2);
            let idx = if gp {
                g * pr.p + row
            } else {
                g * pr.l + slot as usize
            };
            assert!(
                expect[idx].replace(rec).is_none(),
                "record {idx} named twice"
            );
        }
    }
    let mut written = 0;
    for (idx, want) in expect.iter().enumerate() {
        let have = &got[idx * REC..][..REC];
        match want {
            Some(w) => {
                written += 1;
                assert_eq!(have, w, "record {idx} (gp={gp})");
            }
            None => assert!(
                have.iter().all(|&b| b == SENTINEL),
                "record {idx} (gp={gp}) was written but no live row names it"
            ),
        }
    }
    assert!(written >= groups * (pr.p - 2));
}

#[test]
fn sorted_slot_producer_matches_the_reference_on_the_interpreter() {
    require_rocm_tool!("llvm-mc");
    for (k, p, seed) in [(128usize, 7usize, 3u64), (256, 12, 5), (512, 9, 7)] {
        let pr = problem(k, p, seed);
        let got =
            run(Spec { arch: ARCH }, false, &pr).unwrap_or_else(|e| panic!("K={k} P={p}: {e}"));
        check(false, &pr, &got);
    }
}

#[test]
fn grouped_position_twin_matches_the_reference_on_the_interpreter() {
    require_rocm_tool!("llvm-mc");
    for (k, p, seed) in [(128usize, 7usize, 3u64), (256, 12, 5), (512, 9, 7)] {
        let pr = problem(k, p, seed);
        let got =
            run(Spec { arch: ARCH }, true, &pr).unwrap_or_else(|e| panic!("K={k} P={p}: {e}"));
        check(true, &pr, &got);
    }
}
