// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 (CDNA3, wave64) lift: raw device ELF -> typed instructions -> the same ELF.
//!
//! Every kernel's code range is decoded by the `peacemaker-cdna` codec into typed
//! [`peacemaker_cdna::Inst`]s and re-encoded from those typed fields; the words must equal
//! the input word for word. The ELF (header, sections, notes, descriptors, metadata,
//! padding) is carried by the shared [`Envelope`] codec, so [`checked`] re-emits the whole
//! module from the envelope plus the re-encoded kernel code. [`lift_object`] proves the
//! identity of that re-emission with the input bytes before it returns.
//!
//! Fail-closed: an unknown/unsupported encoding, an indirect jump, a branch that does not
//! land on an instruction boundary, code that can fall off its end, a kernel with no
//! reachable `s_endpgm`, a bundle, a non-gfx942 target or a descriptor/metadata
//! disagreement is an error. No executable bytes are ever carried opaquely.

use std::collections::HashMap;

use peacemaker_cdna::enc::sop::{Sop1Op, SoppOp};
use peacemaker_cdna::{Arch, Inst};
use peacemaker_ir::envelope::Envelope;

use crate::bundle::{COMPRESSED_MAGIC, MAGIC};
use crate::elf::{EnvelopeCodec, KernelImage, KernelParts};

/// `EF_AMDGPU_MACH` mask and `EF_AMDGPU_MACH_AMDGCN_GFX942` (LLVM AMDGPUUsage).
const EF_AMDGPU_MACH: u32 = 0xff;
const EF_AMDGPU_MACH_AMDGCN_GFX942: u32 = 0x4c;
/// `ELFOSABI_AMDGPU_HSA`.
const ELFOSABI_AMDGPU_HSA: u8 = 64;
/// `kernel_code_properties` bit 10: ENABLE_WAVEFRONT_SIZE32 (must be clear: wave64).
const KCP_WAVE32: u16 = 1 << 10;
/// `kernel_code_properties` bit 11: USES_DYNAMIC_STACK.
const KCP_DYNAMIC_STACK: u16 = 1 << 11;
/// `ELFABIVERSION_AMDGPU_HSA_V4` (code object v4 is the lowest supported).
const ELFABIVERSION_AMDGPU_HSA_V4: u8 = 2;
const ARCH: Arch = Arch::Gfx942;

/// One typed instruction and the dword offset of its first dword in the kernel's code.
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    pub dword_offset: usize,
    pub inst: peacemaker_cdna::Inst,
}

/// A kernel's ELF-side image (code, descriptor, metadata) and its typed instructions.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub image: crate::elf::KernelImage,
    pub instructions: Vec<Instruction>,
}

/// A lifted gfx942 code object: the retained envelope plus the typed kernels (entry order).
#[derive(Clone, Debug, PartialEq)]
pub struct Lifted {
    pub envelope: peacemaker_ir::envelope::Envelope,
    pub kernels: Vec<Kernel>,
}

/// Lifts a raw gfx942 ELF. Fails closed; on success every kernel is typed, structurally
/// valid and re-emitting the module reproduces `input` byte for byte.
pub fn lift_object(input: &[u8]) -> Result<Lifted, String> {
    if input.starts_with(MAGIC) || input.starts_with(COMPRESSED_MAGIC) {
        return Err("input is a clang offload bundle; lift_object takes a raw gfx942 ELF".into());
    }
    let (envelope, images) = Envelope::read(input).map_err(|e| format!("envelope: {e}"))?;
    validate_envelope(&envelope)?;
    if images.is_empty() {
        return Err("code object has no kernels".into());
    }
    let mut kernels = Vec::with_capacity(images.len());
    for image in images {
        validate_image(&image)?;
        let instructions = decode_kernel(&image)?;
        validate_cfg(&image.name, &instructions, image.code.len() / 4)?;
        kernels.push(Kernel {
            image,
            instructions,
        });
    }
    let lifted = Lifted { envelope, kernels };
    let out = checked(&lifted)?;
    if out != input {
        let at = out
            .iter()
            .zip(input)
            .position(|(a, b)| a != b)
            .unwrap_or(out.len().min(input.len()));
        return Err(format!(
            "re-emitted ELF differs from the input (first difference at byte {at}, {} vs {} bytes)",
            out.len(),
            input.len()
        ));
    }
    Ok(lifted)
}

/// Target, code-object ABI and helper-extent validation shared by `lift_object` and `checked`.
fn validate_envelope(envelope: &Envelope) -> Result<(), String> {
    let header = &envelope.header;
    if header.e_flags & EF_AMDGPU_MACH != EF_AMDGPU_MACH_AMDGCN_GFX942 {
        return Err(format!(
            "e_flags {:#x}: machine {:#x} is not gfx942 ({EF_AMDGPU_MACH_AMDGCN_GFX942:#x})",
            header.e_flags,
            header.e_flags & EF_AMDGPU_MACH
        ));
    }
    if header.os_abi != ELFOSABI_AMDGPU_HSA {
        return Err(format!("EI_OSABI {} is not AMDGPU_HSA", header.os_abi));
    }
    if header.abi_version < ELFABIVERSION_AMDGPU_HSA_V4 {
        return Err(format!(
            "EI_ABIVERSION {} is below code object v4",
            header.abi_version
        ));
    }
    // Helper extents are executable ranges outside kernel symbols; the typed lift does not
    // cover them, so they must not be carried as opaque bytes.
    let helpers = peacemaker_ir::helpers::helper_extents(envelope)?;
    if !helpers.is_empty() {
        return Err(format!(
            "{} helper extent(s) outside the kernels are unsupported",
            helpers.len()
        ));
    }
    Ok(())
}

/// Rebuilds every instruction's words from its typed fields, checks each kernel again
/// (decode of the rebuilt words must give the same instructions, bounds/CFG/termination,
/// descriptor and metadata) and serialises the module through the shared envelope codec.
pub fn checked(lifted: &Lifted) -> Result<Vec<u8>, String> {
    validate_envelope(&lifted.envelope)?;
    let mut codes = Vec::with_capacity(lifted.kernels.len());
    for kernel in &lifted.kernels {
        let name = &kernel.image.name;
        validate_image(&kernel.image)?;
        let mut words = Vec::with_capacity(kernel.image.code.len() / 4);
        for (index, item) in kernel.instructions.iter().enumerate() {
            if item.dword_offset != words.len() {
                return Err(format!(
                    "{name}: instruction {index} claims dword offset {} but follows {} dwords",
                    item.dword_offset,
                    words.len()
                ));
            }
            item.inst
                .encode(ARCH, &mut words)
                .map_err(|e| format!("{name}: instruction {index} ({:?}): {e}", item.inst))?;
        }
        let redecoded = peacemaker_cdna::decode_all(ARCH, &words)
            .map_err(|e| format!("{name}: rebuilt words do not decode: {e}"))?;
        if redecoded.len() != kernel.instructions.len()
            || redecoded
                .iter()
                .zip(&kernel.instructions)
                .any(|((at, inst), item)| *at != item.dword_offset || *inst != item.inst)
        {
            return Err(format!(
                "{name}: rebuilt words do not decode to the typed instructions"
            ));
        }
        validate_cfg(name, &kernel.instructions, words.len())?;
        let mut code = Vec::with_capacity(words.len() * 4);
        for w in &words {
            code.extend_from_slice(&w.to_le_bytes());
        }
        codes.push(code);
    }
    let parts: Vec<KernelParts<'_>> = lifted
        .kernels
        .iter()
        .zip(&codes)
        .map(|(k, code)| KernelParts {
            code,
            descriptor: &k.image.descriptor,
            metadata: &k.image.metadata,
        })
        .collect();
    lifted
        .envelope
        .write(&parts)
        .map_err(|e| format!("envelope write: {e}"))
}

fn decode_kernel(image: &KernelImage) -> Result<Vec<Instruction>, String> {
    let name = &image.name;
    if image.code.is_empty() || image.code.len() % 4 != 0 {
        return Err(format!(
            "{name}: code is {} bytes, not a non-empty whole number of dwords",
            image.code.len()
        ));
    }
    let words: Vec<u32> = image
        .code
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let decoded = peacemaker_cdna::decode_all(ARCH, &words).map_err(|e| format!("{name}: {e}"))?;
    let mut out = Vec::with_capacity(decoded.len());
    for (at, inst) in decoded {
        let re = inst
            .words(ARCH)
            .map_err(|e| format!("{name}: dword {at}: re-encode of {inst:?}: {e}"))?;
        if words.get(at..at + re.len()) != Some(re.as_slice()) {
            return Err(format!(
                "{name}: dword {at}: {inst:?} re-encodes to {re:08x?}, input has {:08x?}",
                words.get(at..(at + re.len()).min(words.len()))
            ));
        }
        out.push(Instruction {
            dword_offset: at,
            inst,
        });
    }
    Ok(out)
}

/// Target, wave and resource consistency of one kernel's descriptor and metadata.
fn validate_image(image: &KernelImage) -> Result<(), String> {
    let name = &image.name;
    let kd = &image.descriptor;
    let meta = &image.metadata.parsed;
    let fail = |why: String| Err(format!("{name}: {why}"));
    if kd.kernel_code_properties.0 & KCP_WAVE32 != 0 {
        return fail("descriptor selects wave32; gfx942 kernels are wave64".into());
    }
    if meta.wavefront_size != 64 {
        return fail(format!(
            ".wavefront_size {} but the descriptor selects wave64",
            meta.wavefront_size
        ));
    }
    if kd.kernel_code_properties.0 & KCP_DYNAMIC_STACK != 0 || meta.uses_dynamic_stack {
        return fail("dynamic stack is unsupported".into());
    }
    if meta.symbol != format!("{name}.kd") {
        return fail(format!(".symbol {} is not {name}.kd", meta.symbol));
    }
    if meta.name != *name {
        return fail(format!(
            ".name {} differs from the kernel symbol",
            meta.name
        ));
    }
    if meta.private_segment_fixed_size != kd.private_segment_fixed_size {
        return fail(format!(
            ".private_segment_fixed_size {} but the descriptor allocates {}",
            meta.private_segment_fixed_size, kd.private_segment_fixed_size
        ));
    }
    if meta.group_segment_fixed_size != kd.group_segment_fixed_size {
        return fail(format!(
            ".group_segment_fixed_size {} but the descriptor allocates {}",
            meta.group_segment_fixed_size, kd.group_segment_fixed_size
        ));
    }
    if meta.kernarg_segment_size != kd.kernarg_size {
        return fail(format!(
            ".kernarg_segment_size {} but the descriptor says {}",
            meta.kernarg_segment_size, kd.kernarg_size
        ));
    }
    // gfx90a+/gfx942 wave64: VGPR granule is 8 registers (AGPRs share the allocation);
    // rsrc3[5:0] ACCUM_OFFSET is in units of 4 starting at 4 and cannot pass the allocation.
    let granules = (kd.compute_pgm_rsrc1.0 & 0x3f) + 1;
    let allocated = granules * 8;
    if meta.vgpr_count == 0 || meta.vgpr_count > allocated || meta.vgpr_count + 8 <= allocated {
        return fail(format!(
            ".vgpr_count {} does not match the descriptor allocation of {allocated} VGPRs",
            meta.vgpr_count
        ));
    }
    let accum_offset = ((kd.compute_pgm_rsrc3.0 & 0x3f) + 1) * 4;
    if accum_offset > allocated {
        return fail(format!(
            "ACCUM_OFFSET {accum_offset} exceeds the {allocated} allocated VGPRs"
        ));
    }
    if meta.max_flat_workgroup_size == 0 {
        return fail(".max_flat_workgroup_size is 0".into());
    }
    Ok(())
}

/// Successor sets of the typed stream: branch targets on instruction boundaries inside the
/// code, no fall-through off the end, no indirect jump, a reachable `s_endpgm`.
fn validate_cfg(name: &str, insts: &[Instruction], dwords: usize) -> Result<(), String> {
    let index: HashMap<usize, usize> = insts
        .iter()
        .enumerate()
        .map(|(i, it)| (it.dword_offset, i))
        .collect();
    if insts.is_empty() {
        return Err(format!("{name}: no instructions"));
    }
    let mut seen = vec![false; insts.len()];
    let mut stack = vec![0usize];
    let mut ends = false;
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut seen[i], true) {
            continue;
        }
        let it = &insts[i];
        let next = i + 1;
        let next_dword = it.dword_offset
            + it.inst
                .words(ARCH)
                .map_err(|e| format!("{name}: dword {}: {e}", it.dword_offset))?
                .len();
        let mut fall = true;
        match &it.inst {
            Inst::Sopp(p) if p.op == SoppOp::Endpgm => {
                ends = true;
                fall = false;
            }
            Inst::Sop1(s) if s.op == Sop1Op::SetpcB64 => {
                return Err(format!(
                    "{name}: dword {}: s_setpc_b64 (indirect jump) is unsupported",
                    it.dword_offset
                ));
            }
            other => {
                if let Some(off) = other.branch_offset() {
                    let target = next_dword as i64 + i64::from(off);
                    let at = usize::try_from(target)
                        .ok()
                        .and_then(|t| index.get(&t).copied())
                        .ok_or_else(|| {
                            format!(
                                "{name}: dword {}: branch target dword {target} is not an instruction boundary inside the {dwords}-dword code",
                                it.dword_offset
                            )
                        })?;
                    stack.push(at);
                    fall = !matches!(other, Inst::Sopp(p) if p.op == SoppOp::Branch);
                }
            }
        }
        if fall {
            if next >= insts.len() {
                return Err(format!(
                    "{name}: dword {}: execution can fall off the end of the code",
                    it.dword_offset
                ));
            }
            stack.push(next);
        }
    }
    if !ends {
        return Err(format!("{name}: no s_endpgm is reachable from the entry"));
    }
    Ok(())
}
