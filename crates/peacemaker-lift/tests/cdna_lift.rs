// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 typed lift (`peacemaker_lift::cdna`): identity on the committed rows16 hipcc
//! object and fail-closed negatives. Nothing is compiled by the tests.

use peacemaker_cdna::enc::sop::{Sopp, SoppOp};
use peacemaker_cdna::Inst;
use peacemaker_ir::envelope::Bundle;
use peacemaker_lift::bundle::{BundleCodec, HIP_DEVICE_TRIPLE_PREFIX};
use peacemaker_lift::cdna::{checked, lift_object, Instruction, Lifted};

static ANCHOR: &[u8] =
    include_bytes!("../../../kernels/gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co");
static MOE_ANCHOR: &[u8] =
    include_bytes!("../../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co");

/// The committed anchors are hipcc `--genco` offload bundles; the raw gfx942 device ELF is the
/// single entry whose triple is exactly `hipv4-amdgcn-amd-amdhsa--gfx942` or that followed by
/// `:feature` suffixes (the `BundleCodec::device_entry` rule, which has no gfx942 `Arch`).
fn device_elf(bundle_bytes: &[u8]) -> Vec<u8> {
    let (bundle, payloads) = Bundle::read(bundle_bytes).expect("anchor is a readable bundle");
    let exact = format!("{HIP_DEVICE_TRIPLE_PREFIX}gfx942");
    let hits: Vec<usize> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == exact || e.triple.starts_with(&format!("{exact}:")))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one gfx942 device entry: {:?}",
        bundle.entries
    );
    payloads[hits[0]].to_vec()
}

fn anchor() -> Vec<u8> {
    device_elf(ANCHOR)
}

fn lifted_anchor() -> Lifted {
    lift_object(&anchor()).expect("anchor lifts")
}

fn sopp(op: SoppOp, simm16: u16) -> Inst {
    Inst::Sopp(Sopp { op, simm16 })
}

#[test]
fn bundle_is_rejected() {
    let mut bytes = b"__CLANG_OFFLOAD_BUNDLE__".to_vec();
    bytes.extend_from_slice(&[0; 64]);
    let err = lift_object(&bytes).unwrap_err();
    assert!(err.contains("bundle"), "{err}");
}

#[test]
fn garbage_is_rejected() {
    assert!(lift_object(b"not an elf").is_err());
    assert!(lift_object(&[]).is_err());
}

#[test]
fn gfx12_object_is_not_gfx942() {
    let bytes = include_bytes!("fixtures/kt48/hipcc.co");
    let err = lift_object(bytes).unwrap_err();
    assert!(err.contains("gfx942"), "{err}");
}

#[test]
fn anchor_identity_and_typed_kernels() {
    let bytes = anchor();
    let lifted = lift_object(&bytes).expect("anchor lifts");
    assert!(!lifted.kernels.is_empty());
    for kernel in &lifted.kernels {
        assert!(!kernel.instructions.is_empty(), "{}", kernel.image.name);
        assert_eq!(kernel.instructions[0].dword_offset, 0);
        let last = kernel.instructions.last().unwrap();
        assert!(
            matches!(&last.inst, Inst::Sopp(p) if matches!(p.op, SoppOp::Endpgm | SoppOp::Branch)),
            "{}: last instruction {:?}",
            kernel.image.name,
            last.inst
        );
    }
    assert_eq!(checked(&lifted).expect("checked"), bytes);
}

/// The FP8 grouped MoE object (FNUZ pair convert, LDS stage/read, BF16 MFMA, `ds_bpermute`,
/// 64-bit VALU address math, helper kernel): every kernel lifts to typed instructions and
/// re-emits byte for byte.
#[test]
fn moe_anchor_identity_and_typed_kernels() {
    let bytes = device_elf(MOE_ANCHOR);
    let lifted = lift_object(&bytes).expect("MoE anchor lifts");
    let mut names: Vec<&str> = lifted
        .kernels
        .iter()
        .map(|k| k.image.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "gemm_fp8b128_moe_grouped_bf16mfma",
            "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
            "gemm_fp8b128_moe_tile_desc_pad16",
        ]
    );
    for kernel in &lifted.kernels {
        assert!(!kernel.instructions.is_empty(), "{}", kernel.image.name);
        assert_eq!(kernel.instructions[0].dword_offset, 0);
        let last = kernel.instructions.last().unwrap();
        assert!(
            matches!(&last.inst, Inst::Sopp(p) if matches!(p.op, SoppOp::Endpgm | SoppOp::Branch)),
            "{}: last instruction {:?}",
            kernel.image.name,
            last.inst
        );
        if kernel.image.name.ends_with("tile_desc_pad16") {
            continue;
        }
        let text: Vec<String> = kernel
            .instructions
            .iter()
            .map(|i| i.inst.to_string())
            .collect();
        for needle in [
            "v_cvt_pk_f32_fp8",
            "v_mfma_f32_16x16x16_bf16",
            "ds_write_b128",
            "ds_read2_b64",
            "ds_bpermute_b32",
            "v_pk_mul_f32",
            "v_lshl_add_u64",
        ] {
            assert!(
                text.iter().any(|l| l.contains(needle)),
                "{}: no `{needle}` in the typed instructions",
                kernel.image.name
            );
        }
    }
    assert_eq!(checked(&lifted).expect("checked"), bytes);
}

#[test]
fn missing_terminator_is_rejected() {
    let mut lifted = lifted_anchor();
    lifted.kernels[0].instructions.pop();
    let err = checked(&lifted).unwrap_err();
    assert!(!err.is_empty());
}

#[test]
fn branch_off_the_code_is_rejected() {
    let mut lifted = lifted_anchor();
    let insts = &mut lifted.kernels[0].instructions;
    for item in insts.iter_mut() {
        item.dword_offset += 1;
    }
    insts.insert(
        0,
        Instruction {
            dword_offset: 0,
            inst: sopp(SoppOp::Branch, 0x7fff),
        },
    );
    let err = checked(&lifted).unwrap_err();
    assert!(err.contains("branch target"), "{err}");
}

#[test]
fn indirect_jump_is_rejected() {
    use peacemaker_cdna::enc::sop::{Sop1, Sop1Op};
    use peacemaker_cdna::Src;
    let mut lifted = lifted_anchor();
    let insts = &mut lifted.kernels[0].instructions;
    for item in insts.iter_mut() {
        item.dword_offset += 1;
    }
    insts.insert(
        0,
        Instruction {
            dword_offset: 0,
            inst: Inst::Sop1(Sop1 {
                op: Sop1Op::SetpcB64,
                sdst: None,
                ssrc0: Some(Src::Sgpr(0)),
            }),
        },
    );
    let err = checked(&lifted).unwrap_err();
    assert!(
        err.contains("s_setpc_b64") || err.contains("indirect"),
        "{err}"
    );
}

#[test]
fn wave32_descriptor_is_rejected() {
    let mut lifted = lifted_anchor();
    lifted.kernels[0].image.descriptor.kernel_code_properties.0 |= 1 << 10;
    let err = checked(&lifted).unwrap_err();
    assert!(err.contains("wave"), "{err}");
}

#[test]
fn metadata_wave_mismatch_is_rejected() {
    let mut lifted = lifted_anchor();
    lifted.kernels[0].image.metadata.parsed.wavefront_size = 32;
    assert!(checked(&lifted).is_err());
}

#[test]
fn vgpr_allocation_mismatch_is_rejected() {
    let mut lifted = lifted_anchor();
    lifted.kernels[0].image.metadata.parsed.vgpr_count = 512;
    assert!(checked(&lifted).is_err());
}

#[test]
fn occupancy_padded_vgpr_reservation_is_valid() {
    let mut lifted = lifted_anchor();
    let rsrc1 = &mut lifted.kernels[0].image.descriptor.compute_pgm_rsrc1.0;
    let granule = *rsrc1 & 0x3f;
    assert!(granule < 0x3f);
    *rsrc1 = (*rsrc1 & !0x3f) | (granule + 1);
    checked(&lifted).expect("a descriptor may reserve more VGPRs than the code uses");
}

#[test]
fn metadata_cannot_understate_typed_vgpr_references() {
    let mut lifted = lifted_anchor();
    lifted.kernels[0].image.metadata.parsed.vgpr_count = 1;
    let error = checked(&lifted).unwrap_err();
    assert!(error.contains("beyond .vgpr_count"), "{error}");
}

#[test]
fn wrong_machine_is_rejected() {
    let mut bytes = anchor();
    // e_flags lives at ELF64 header offset 48; set the machine byte to gfx90a (0x3f).
    bytes[48] = 0x3f;
    let err = lift_object(&bytes).unwrap_err();
    assert!(err.contains("gfx942"), "{err}");
}

#[test]
fn unknown_opcode_in_code_is_rejected() {
    let mut bytes = anchor();
    let lifted = lift_object(&bytes).unwrap();
    let k = &lifted.kernels[0];
    // Find the kernel's first dword in the file and overwrite it with an unassigned encoding.
    let needle = &k.image.code[..16.min(k.image.code.len())];
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("kernel code is in the file");
    bytes[at..at + 4].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
    assert!(lift_object(&bytes).is_err());
}

#[test]
fn checked_rejects_mutated_envelope_target() {
    let mut lifted = lifted_anchor();
    lifted.envelope.header.e_flags = (lifted.envelope.header.e_flags & !0xff) | 0x3f;
    let err = checked(&lifted).unwrap_err();
    assert!(err.contains("gfx942"), "{err}");
}

#[test]
fn checked_rejects_mutated_envelope_abi() {
    let mut lifted = lifted_anchor();
    lifted.envelope.header.abi_version = 1;
    let err = checked(&lifted).unwrap_err();
    assert!(err.contains("code object v4"), "{err}");
    let mut lifted = lifted_anchor();
    lifted.envelope.header.os_abi = 0;
    assert!(checked(&lifted).is_err());
}
