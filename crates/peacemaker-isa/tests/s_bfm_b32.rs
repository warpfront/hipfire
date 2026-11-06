// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU emulation of scalar `s_bfm_b32` (gfx1201): the probe kernel is assembled, lifted and run per input record
//! and compared with a bit-by-bit model independent of the host law in the probe helper.
use peacemaker_ir::{
    emu::{Emulator, Launch, Memory},
    inst::Frontend,
};
use peacemaker_isa::Arch;

#[path = "support/s_bfm_probe.rs"]
mod s_bfm_probe;

/// Independent model: bit `i` is set iff `offset <= i < offset + width` (field bits past bit 31 wrap out).
fn bitwise_model(input: [u32; 3]) -> [u32; 2] {
    let (width, offset) = (input[0] & 31, input[1] & 31);
    let mut mask = 0u32;
    for i in 0..32 {
        if i >= offset && i < offset + width {
            mask |= 1 << i;
        }
    }
    [mask, u32::from(input[2] != 0)]
}

#[test]
fn host_law_matches_bitwise_model_and_inputs_cover_everything() {
    let inputs = s_bfm_probe::inputs();
    assert_eq!(inputs.len(), 8192);
    for &i in &inputs {
        assert_eq!(s_bfm_probe::expected(i), bitwise_model(i), "{i:?}");
    }
    let low: std::collections::HashSet<_> = inputs
        .iter()
        .map(|i| (i[0] & 31, i[1] & 31, i[2]))
        .collect();
    assert_eq!(low.len(), 32 * 32 * 2);
    assert!(inputs.iter().any(|i| i[0] >> 5 != 0 && i[0] >> 5 != u32::MAX >> 5));
    assert_eq!(s_bfm_probe::expected([0, 7, 1]), [0, 1]);
    assert_eq!(s_bfm_probe::expected([32, 0, 0]), [0, 0]);
    assert_eq!(s_bfm_probe::expected([16, 0, 0]), [0xffff, 0]);
    assert_eq!(s_bfm_probe::expected([31, 31, 1]), [0x8000_0000, 1]);
}

#[test]
fn emulated_probe_equals_the_independent_model_over_all_inputs() {
    let emitted = s_bfm_probe::emit_probe().unwrap();
    assert!(emitted.s_text.contains("s_bfm_b32"));
    assert!(emitted.s_text.contains(&format!("{}:", s_bfm_probe::SYMBOL)));
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emulator = Emulator::new(&program, s_bfm_probe::SYMBOL).unwrap();
    let inputs = s_bfm_probe::inputs();
    let (ip, op, kp) = (0x10_0000u64, 0x4000_0000u64, 0x4010_0000u64);
    let mut mem = Memory::default();
    mem.map(
        "inputs".into(),
        ip,
        inputs
            .iter()
            .flat_map(|r| r.iter().flat_map(|w| w.to_le_bytes()))
            .collect(),
    )
    .unwrap();
    mem.map("outputs".into(), op, vec![0xAA; inputs.len() * 8])
        .unwrap();
    let mut kern = Vec::new();
    kern.extend_from_slice(&ip.to_le_bytes());
    kern.extend_from_slice(&op.to_le_bytes());
    mem.map("kernarg".into(), kp, kern).unwrap();
    for i in 0..inputs.len() {
        emulator
            .run_workgroup(
                &Launch {
                    kernarg_ptr: kp,
                    workgroup: [i as u32, 0, 0],
                    block: [s_bfm_probe::BLOCK, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("record {i} {:?}: {e}", inputs[i]));
    }
    let out = &mem.regions.iter().find(|r| r.name == "outputs").unwrap().bytes;
    for (i, input) in inputs.iter().enumerate() {
        let got = [
            u32::from_le_bytes(out[8 * i..8 * i + 4].try_into().unwrap()),
            u32::from_le_bytes(out[8 * i + 4..8 * i + 8].try_into().unwrap()),
        ];
        assert_eq!(got, bitwise_model(*input), "record {i} {input:?}");
    }
}
