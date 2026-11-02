// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Regression: the gfx1100 K3 down producer `mq_i4_a4_r128_k640_gfx1100` must address its `xq` record
//! `xq + 72 * (g * L + slot)` with an exact 64-bit product.
//!
//! The first emission computed it with `mad64(XQ_PTR, ARG_XQ, T0, 72)` whose `x` register IS the scratch
//! `T0` the low product is written to: `s_mul_i32 T0, T0, 72` clobbered `x` before `s_mul_hi_u32 T1, T0, 72`
//! read it, so the pointer gained `floor(low * 72 / 2^32) << 32` (a +4 GiB store) as soon as the record
//! index `g * L + slot` reached 828,505 (`72 * 72 * x >= 2^32`). Every earlier gate ran at tiny L; the real
//! S1 window (`T = 16896`, `L = 168,960` slots) at group 4 and slot >= 152,665 stored 4 GiB past `xq` on the
//! XTX (`HipError(700)`, X1-ROUTE). Here the emitted kernel runs on the PeaceMaker emulator with the real
//! kernargs (`P = 2048`, `L = 168960`, `xq` = exactly `5 * L * 72` bytes, nothing mapped around it) at the
//! faulting workgroups (`p = 320`, `g = 4`) and both sides of the threshold; every record must be written at
//! `g * L + slot` and equal the CPU reference bytes. A negative control swaps the high-word multiply back
//! behind the low one (the original instruction order) and must hit unmapped memory.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::{native, Arch};

#[path = "../src/kernels/mq_i4_a4_gfx1100.rs"]
#[allow(dead_code)]
mod k3;
#[path = "support/moe_rotate_ref.rs"]
mod reference;
use reference::{bf16_rne, record, signs, Rng};

const ARCH: Arch = Arch::Gfx1100;
const KARG: u64 = 0x1_0000;
const H: u64 = 0x1000_0000;
const SORTED: u64 = 0x2000_0000;
const XQ: u64 = 0x3000_0000;
const S1: u64 = 0x2800_0000;
const S2: u64 = 0x2800_1000;
const REC: usize = 72;
/// FN S1 window: 16896 rows x top-10 slots; one 2048-row panel.
const L: usize = 168_960;
const P: usize = 2048;
/// First record index whose `x * 72` high word the clobbered operand corrupts.
const FIRST_BAD_INDEX: usize = 828_505;

struct Case {
    row: usize,
    slot: usize,
}

fn cases() -> Vec<Case> {
    // g = 4 starts at record 4 * L = 675,840: slot 152,665 is the first index >= 828,505.
    assert_eq!(4 * L + 152_665, FIRST_BAD_INDEX + 0);
    vec![
        Case { row: 0, slot: 7 },
        Case {
            row: 5,
            slot: 152_664,
        },
        Case {
            row: 320,
            slot: 152_665,
        },
        Case {
            row: 321,
            slot: 160_000,
        },
        Case {
            row: P - 1,
            slot: L - 1,
        },
    ]
}

fn problem() -> (Vec<i32>, Vec<u16>) {
    let mut sorted = vec![-1i32; P];
    for c in cases() {
        sorted[c.row] = c.slot as i32;
    }
    let mut rng = Rng(0x5eed_1100);
    let mut h = vec![0u16; P * 640];
    for row in 0..P {
        for c in 0..640 {
            let u = (rng.next() >> 40) as f32 / (1u32 << 24) as f32 - 0.5;
            h[row * 640 + c] = bf16_rne(u * (2.0 + (c % 7) as f32));
        }
    }
    (sorted, h)
}

fn program(swap_xq_site: bool) -> peacemaker_ir::Program {
    let emitted = k3::emit(k3::Spec { arch: ARCH }).expect("emit K3");
    let elf = native::assemble(&emitted.s_text, ARCH).expect("assemble K3");
    let mut program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .expect("lift K3")
    .program;
    if swap_xq_site {
        let arch = program.target.arch;
        let kernel = program
            .kernels
            .iter_mut()
            .find(|k| k.symbol.0 == "mq_i4_a4_r128_k640_gfx1100")
            .expect("K3 kernel");
        let name = |kernel: &peacemaker_ir::inst::Kernel, i: usize| -> &'static str {
            let inst = kernel.body.insts.get(kernel.body.layout[i]).expect("inst");
            peacemaker_ir::isa::lookup(arch, inst.op, inst.form)
                .expect("opcode row")
                .name
        };
        // Fixed order at each of the three mad64 sites: s_mul_hi_u32 then s_mul_i32. The third is the xq
        // pointer (x == T0); putting the low product first restores the original, aliasing, order.
        let pairs: Vec<usize> = (0..kernel.body.layout.len() - 1)
            .filter(|&i| name(kernel, i) == "s_mul_hi_u32" && name(kernel, i + 1) == "s_mul_i32")
            .collect();
        assert_eq!(pairs.len(), 3, "sort, h and xq pointer sites");
        kernel.body.layout.swap(pairs[2], pairs[2] + 1);
    }
    program
}

fn run(program: &peacemaker_ir::Program, sorted: &[i32], h: &[u16]) -> Result<Memory, String> {
    let emu = Emulator::new(program, "mq_i4_a4_r128_k640_gfx1100")?
        .hw_model()
        .gfx1100_as_gfx1151();
    let mut karg = Vec::new();
    for ptr in [H, SORTED, XQ, S1, S2] {
        karg.extend(ptr.to_le_bytes());
    }
    karg.extend((P as u32).to_le_bytes());
    karg.extend((L as u32).to_le_bytes());
    karg.resize(k3::KERNARG_BYTES as usize, 0);
    let f =
        |s: &[f32; 128]| -> Vec<u8> { s.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect() };
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg)?;
    mem.map(
        "h".into(),
        H,
        h.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )?;
    mem.map(
        "sorted".into(),
        SORTED,
        sorted.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )?;
    // Exactly 5 * L records: any store past the end, or 4 GiB away, is unmapped.
    mem.map("xq".into(), XQ, vec![0u8; 5 * L * REC])?;
    mem.map("s1".into(), S1, f(&signs(43)))?;
    mem.map("s2".into(), S2, f(&signs(1043)))?;
    for c in cases() {
        for g in 0..5u32 {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [c.row as u32 * 5 + g, 0, 0],
                    block: [k3::WORKGROUP_SIZE, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("workgroup (p {}, g {g}): {e}", c.row))?;
        }
    }
    Ok(mem)
}

#[test]
fn xq_records_land_at_g_l_plus_slot_across_the_828505_threshold() {
    let (sorted, h) = problem();
    let mem = run(&program(false), &sorted, &h).expect("fixed K3 runs the real-window workgroups");
    let (s1, s2) = (signs(43), signs(1043));
    let written = mem
        .regions
        .iter()
        .find(|r| r.name == "xq")
        .unwrap()
        .written
        .count();
    assert_eq!(
        written,
        cases().len() * 5 * REC,
        "exactly the 5 records of each row, nothing else"
    );
    for c in cases() {
        for g in 0..5 {
            let index = g * L + c.slot;
            let got = mem.read(XQ + (index * REC) as u64, REC).unwrap();
            let want = record(
                &h[c.row * 640 + g * 128..c.row * 640 + (g + 1) * 128],
                &s1,
                &s2,
            );
            assert_eq!(
                got,
                &want[..],
                "row {} slot {} g {g}: record index {index}",
                c.row,
                c.slot
            );
        }
    }
}

#[test]
fn the_original_instruction_order_stores_4_gib_away() {
    let (sorted, h) = problem();
    let err = run(&program(true), &sorted, &h).expect_err("the aliasing order must fault");
    // The first hazardous workgroup in the case order: p = 320, g = 4 (record 828,505).
    assert!(
        err.contains("p 320, g 4") && err.contains("unmapped memory access"),
        "{err}"
    );
}

/// The committed product object `kernels/mq_i4_a4_pm_gfx1100.hxaco` is what the builder emits today (fresh module
/// emission, native assembly and bundling are byte-equal to it), and it follows the gfx1100
/// `FeatureUserSGPRInit16Bug` rule: 15 declared user SGPRs (kernarg pointer padded) + the workgroup id x = 16, the
/// workgroup id being `s15` and the kernarg block `s[48:55]`.
#[test]
fn committed_bundle_equals_fresh_emission_and_declares_sixteen_user_and_system_sgprs() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let committed = std::fs::read(format!("{root}/mq_i4_a4_pm_gfx1100.hxaco")).unwrap();
    let (_, text, _) = k3::emit_module(ARCH).unwrap();
    let elf = native::assemble(&text, ARCH).unwrap();
    let fresh = native::bundle(&elf, ARCH, native::DEFAULT_HOST_TARGET);
    assert!(
        fresh == committed,
        "fresh bundle differs from the committed bundle"
    );
    assert!(text.contains("\t.amdhsa_user_sgpr_count 15\n"));
    assert!(text.contains("\ts_mov_b32 s20, s15\n"));
    assert!(text.contains("\ts_load_b256 s[48:55], s[0:1], null\n"));
    assert!(!text.contains("s[8:15]"));
}
