// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 row-local R128 / K640 A4 producer `mq_i4_a4_r128_k640` on the PeaceMaker emulator at production-scale
//! extents. The device buffers are *sparse*: only the bytes a correct kernel may touch (the one `h` row, the one `sorted`
//! entry and the one 72-byte record per workgroup) are mapped, at their true 64-bit addresses, so any address that is
//! wrong by even one 4 GiB multiple (the old `mad64` aliasing: `s_mul_hi_u32` read the clobbered low product, wrong once
//! `g*L + slot` exceeds `2^32 / 72^2 ~ 828,505`) is an unmapped access and fails the run. Every record is compared
//! bitwise with the CPU reference of `support/moe_rotate_ref.rs`.
//!
//! `A4_OLD_S=<file.s>` runs the same launches on a previously emitted kernel text and reports its outcome per slot
//! (A/B evidence for the fix; not asserted).
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_a4::{self as a4, Spec};
use peacemaker_isa::{native, Arch};

#[path = "support/moe_rotate_ref.rs"]
mod reference;
use reference::{bf16_rne, record, signs, Rng};

const ARCH: Arch = Arch::Gfx1201;
const KARG: u64 = 0x1_0000;
const S1: u64 = 0x400_0000;
const S2: u64 = 0x401_0000;
/// Buffer bases above 4 GiB so a dropped / spurious carry into the high dword lands outside every mapping.
const H: u64 = 0x10_0000_0000;
const SORTED: u64 = 0x20_0000_0000;
const XQ: u64 = 0x30_0000_0000;
const SENTINEL: u8 = 0xa5;
const REC: u64 = 72;
const ROW_BYTES: u64 = 1280;
/// Smallest record index the old kernel mis-addressed: `72 * 72 * x >= 2^32`.
const OLD_THRESHOLD: u64 = (1 << 32) / (72 * 72) + 1;

struct Case {
    p: u32,
    l: u32,
    /// `(row, slot)` live rows; every other row would be a pad row (never launched).
    rows: Vec<(u32, i32)>,
}

fn h_row(row: u32) -> Vec<u16> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (row as u64 + 1));
    (0..640)
        .map(|c| {
            let u = (rng.next() >> 40) as f32 / (1u32 << 24) as f32 - 0.5;
            bf16_rne(u * (2.0 + (c % 7) as f32))
        })
        .collect()
}

/// Runs workgroups `5 * row + g` for every live row / group; returns the 72 bytes at every expected record,
/// or the first emulator fault.
fn run(text: &str, case: &Case) -> Result<Vec<((u32, u32), Vec<u8>)>, String> {
    let elf = native::assemble(text, ARCH)?;
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?
    .program;
    let emu = Emulator::new(&program, &Spec { arch: ARCH }.symbol())?;
    let mut karg = Vec::new();
    for ptr in [H, SORTED, XQ, S1, S2] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [case.p, case.l] {
        karg.extend(v.to_le_bytes());
    }
    karg.resize(a4::KERNARG_BYTES as usize, 0);
    let f =
        |s: &[f32; 128]| -> Vec<u8> { s.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect() };
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg)?;
    mem.map("s1".into(), S1, f(&signs(a4::SIGN_SEEDS.0)))?;
    mem.map("s2".into(), S2, f(&signs(a4::SIGN_SEEDS.1)))?;
    for &(row, slot) in &case.rows {
        let r = row as u64;
        mem.map(
            format!("h{row}"),
            H + r * ROW_BYTES,
            h_row(row).iter().flat_map(|x| x.to_le_bytes()).collect(),
        )?;
        mem.map(
            format!("sorted{row}"),
            SORTED + 4 * r,
            slot.to_le_bytes().to_vec(),
        )?;
        for g in 0..5u64 {
            mem.map(
                format!("xq{row}.{g}"),
                XQ + REC * (g * case.l as u64 + slot as u64),
                vec![SENTINEL; REC as usize],
            )?;
        }
    }
    let mut out = Vec::new();
    for &(row, slot) in &case.rows {
        for g in 0..5u32 {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [5 * row + g, 0, 0],
                    block: [a4::WORKGROUP_SIZE, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("row {row} slot {slot} g {g}: {e}"))?;
            let at = XQ + REC * (g as u64 * case.l as u64 + slot as u64);
            out.push(((row, g), mem.read(at, REC as usize)?.to_vec()));
        }
    }
    Ok(out)
}

fn expect(case: &Case) -> Vec<((u32, u32), Vec<u8>)> {
    let (s1, s2) = (signs(a4::SIGN_SEEDS.0), signs(a4::SIGN_SEEDS.1));
    case.rows
        .iter()
        .flat_map(|&(row, _)| {
            let h = h_row(row);
            (0..5u32)
                .map(|g| {
                    (
                        (row, g),
                        record(&h[g as usize * 128..][..128], &s1, &s2).to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Slots straddling `OLD_THRESHOLD` on both sides, up to the u32-record-index limit of a 70 M-slot launch, on rows
/// from 0 up to a grid-extreme row (`5 * row` above 2^24 and `row * 1280` above 2^32).
fn production_case() -> Case {
    let l: u32 = 70_000_000;
    let t = OLD_THRESHOLD as i32;
    let slots = [
        0,
        1,
        t / 5 - 1,
        t / 5,
        t / 4,
        t - 1,
        t,
        t + 1,
        5_000_000,
        59_652_322,
        59_652_323,
        59_652_324,
        l as i32 - 1,
    ];
    let mut rows: Vec<(u32, i32)> = slots
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as u32, s))
        .collect();
    // Grid-extreme workgroups: h byte offset row*1280 past 2^32 (row > 3,355,443) and 5*row past 2^24.
    for (row, slot) in [
        (3_355_443u32, 828_507),
        (3_355_444, 69_999_998),
        (3_999_999, 12_345_678),
        (4_000_000, 2),
    ] {
        rows.push((row, slot));
    }
    Case {
        p: 4_000_001,
        l,
        rows,
    }
}

#[test]
fn fixed_producer_is_in_bounds_and_bitwise_at_production_extents() {
    let text = a4::emit(Spec { arch: ARCH }).unwrap().s_text;
    let case = production_case();
    let got = run(&text, &case).unwrap_or_else(|e| panic!("{e}"));
    for (g, w) in got.iter().zip(expect(&case)) {
        assert_eq!(g.0, w.0);
        assert_eq!(g.1, w.1, "record of (row, group) {:?}", g.0);
    }
}
/// Every record index (`g * L + slot`, `g <= 4`) below the old kernel's threshold: `L = 165_000` puts the largest at
/// `4 * 165_000 + 164_999 = 824_999`.
fn below_threshold_case() -> Case {
    let l = 165_000u32;
    let slots = [0, 1, 7, 4_321, 82_500, 164_998, 164_999];
    Case {
        p: 64,
        l,
        rows: slots
            .iter()
            .enumerate()
            .map(|(i, &s)| (3 * i as u32, s))
            .collect(),
    }
}

#[test]
fn fixed_producer_is_bitwise_below_the_old_threshold() {
    let text = a4::emit(Spec { arch: ARCH }).unwrap().s_text;
    let case = below_threshold_case();
    assert!(4 * case.l as u64 + case.l as u64 - 1 < OLD_THRESHOLD);
    let got = run(&text, &case).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(got, expect(&case));
}

/// A/B evidence only: `A4_OLD_S` names a kernel text emitted before the fix. Below the threshold the old kernel's
/// records must equal the reference (hence the fixed kernel's); above it each launch is reported.
#[test]
fn old_kernel_outcome_report() {
    let Ok(path) = std::env::var("A4_OLD_S") else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    let low = below_threshold_case();
    assert_eq!(run(&text, &low).unwrap(), expect(&low));
    eprintln!(
        "OLD below threshold (L={}): bitwise-equal to the reference",
        low.l
    );
    let base = production_case();
    let want = expect(&base);
    for (i, &(row, slot)) in base.rows.iter().enumerate() {
        let one = Case {
            p: base.p,
            l: base.l,
            rows: vec![(row, slot)],
        };
        let verdict = match run(&text, &one) {
            Err(e) => format!("FAULT {}", e.split(" | ").next().unwrap_or("")),
            Ok(got) if got == want[i * 5..i * 5 + 5] => "bitwise-equal".to_string(),
            Ok(_) => "CORRUPT".to_string(),
        };
        eprintln!("OLD row {row} slot {slot} (L={}): {verdict}", base.l);
    }
}
