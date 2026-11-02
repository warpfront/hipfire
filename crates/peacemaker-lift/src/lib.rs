// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ELF, bundle, descriptor and metadata lifter for typed AMDGPU programs.
//!
//! [`lift_object`] turns a code object (plain ELF or HIP offload bundle) into a
//! `Lifted<Program>`: every kernel's `.text` range is decoded by the table-driven codec,
//! re-encoded from typed fields (never from `prov.bytes`) and compared word for word, the
//! CFG is built from encoded branch offsets, clause/delay windows are checked, and the
//! descriptor/metadata are validated. The module is then re-emitted through [`emit`] and
//! must equal the input byte for byte; that comparison is the proof, not an assumption.
//! [`lift_text`] lifts an authoritative `.s` through the object the pinned assembler made
//! from it: the bytes are the proof, the text contributes labels and line provenance. The
//! lifter never spawns tools; the process drivers live in `peacemaker-front`.
pub mod bundle;
pub mod elf;
pub mod hazards_gfx11;
pub mod kd;
pub mod layout;
pub mod metadata;
pub mod raw;
pub mod rewrite;
pub mod text;

pub use raw::lift_raw;

use peacemaker_ir::cfg::{BlockId, Body};
use peacemaker_ir::codec::{gfx11, gfx12};
use peacemaker_ir::envelope::{Bundle, KernelSlot, Source};
use peacemaker_ir::inst::{
    Abi, Arch, Frontend, Kernel, KernelOrigin, Program, Setting, SymbolId, Target, Wave,
};
use peacemaker_ir::operand::Operand;
use peacemaker_ir::passes::{
    cfg::{build_blocks, CfgError},
    windows::{check_windows_with, WindowError},
};
use peacemaker_ir::provenance::{ObjectSha, Provenance, Source as Origin, TextSha};
use peacemaker_ir::state::{LiftReport, Lifted, RoundTripProof, StreamProof};
use sha2::{Digest, Sha256};

use crate::bundle::{BundleCodec, SourceCodec, COMPRESSED_MAGIC, HIP_DEVICE_TRIPLE_PREFIX};
use crate::elf::KernelImage;
use crate::text::{parse_line, parse_source, SourceKind};

/// What the caller states about the input. The lifter never guesses the producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    pub frontend: Frontend,
}

/// The lift rule an input violated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rule {
    /// Not an ELF/bundle the envelope codec reproduces (see the reason for the codec error).
    Envelope,
    /// Target outside the supported set, ABI below code object v4, or no opcode table yet.
    UnsupportedArch,
    /// Words the target table does not declare: unknown opcode, unknown bits,
    /// a don't-care outside the benign list, or an instruction crossing the symbol end.
    Decode,
    /// Typed fields do not re-encode to the input words.
    StreamIdentity,
    /// Branch target outside the kernel or not at an instruction start.
    BranchTarget,
    /// A reachable path without `s_endpgm`, or a malformed branch.
    ControlFlow,
    /// `s_clause` window or `s_delay_alu` reach crossing a leader, or a malformed clause.
    Window,
    Descriptor,
    Metadata,
    /// `Program::validate` rejected the assembled program.
    Program,
    /// The re-emitted module differs from the input.
    ModuleIdentity,
    /// `lift_text`: the source disagrees with the object it was assembled into.
    Text,
    /// `lift_raw`: the declaration (spec, hashes, extent, frontend, resources, relocations,
    /// sidecar) is inconsistent with itself or with the binary.
    Spec,
    /// `lift_raw`: the code reads a location the declared entry facts do not define.
    Entry,
}

#[derive(Debug, thiserror::Error)]
pub enum LiftError {
    /// `offset` is the code-object virtual address of the offending instruction or
    /// descriptor for kernel-level rules, and 0 for module-level ones.
    #[error("{rule:?} at {offset:#x} ({}): {reason}", .kernel.as_deref().unwrap_or("module"))]
    Rejected {
        kernel: Option<String>,
        offset: u64,
        rule: Rule,
        reason: String,
    },
}

fn module_error(rule: Rule, reason: impl ToString) -> LiftError {
    LiftError::Rejected {
        kernel: None,
        offset: 0,
        rule,
        reason: reason.to_string(),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

// `EF_AMDGPU_MACH_*` (low byte of e_flags) and the code object v4+ feature fields.
const EF_MACH_MASK: u32 = 0xff;
const EF_XNACK_MASK: u32 = 0x300;
const EF_SRAMECC_MASK: u32 = 0xc00;
const ELFOSABI_AMDGPU_HSA: u8 = 64;
/// `e_ident[EI_ABIVERSION]` of code object v4 (v5 = 3, v6 = 4).
const ELFABIVERSION_AMDGPU_HSA_V4: u8 = 2;
const E_FLAGS_OFFSET: usize = 48;

fn arch_of_mach(mach: u32) -> Option<Arch> {
    Some(match mach {
        0x033 => Arch::Gfx1010,
        0x036 => Arch::Gfx1030,
        0x041 => Arch::Gfx1100,
        0x04a => Arch::Gfx1151,
        0x04e => Arch::Gfx1201,
        _ => return None,
    })
}

fn arch_of_name(name: &str) -> Option<Arch> {
    Some(match name {
        "gfx1010" => Arch::Gfx1010,
        "gfx1030" => Arch::Gfx1030,
        "gfx1100" => Arch::Gfx1100,
        "gfx1151" => Arch::Gfx1151,
        "gfx1201" => Arch::Gfx1201,
        _ => return None,
    })
}

/// The architecture the input declares: the HIP device triple of a bundle, or the ELF
/// `EF_AMDGPU_MACH`. It only selects the bundle entry; `target_of` re-checks the ELF.
fn input_arch(bytes: &[u8]) -> Result<Arch, LiftError> {
    if Bundle::is_bundle(bytes) || bytes.starts_with(COMPRESSED_MAGIC) {
        let (bundle, _) = Bundle::read(bytes).map_err(|e| module_error(Rule::Envelope, e))?;
        let triples: Vec<&str> = bundle
            .entries
            .iter()
            .filter_map(|entry| entry.triple.strip_prefix(HIP_DEVICE_TRIPLE_PREFIX))
            .collect();
        let [triple] = triples[..] else {
            return Err(module_error(
                Rule::Envelope,
                format!("bundle has {} HIP device entries, not one", triples.len()),
            ));
        };
        let name = triple.split(':').next().unwrap_or(triple);
        return arch_of_name(name).ok_or_else(|| {
            module_error(
                Rule::UnsupportedArch,
                format!("bundle device target {name}"),
            )
        });
    }
    let flags = bytes
        .get(E_FLAGS_OFFSET..E_FLAGS_OFFSET + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
        .ok_or_else(|| module_error(Rule::Envelope, "input is shorter than an ELF64 header"))?;
    arch_of_mach(flags & EF_MACH_MASK).ok_or_else(|| {
        module_error(
            Rule::UnsupportedArch,
            format!("EF_AMDGPU_MACH {:#x}", flags & EF_MACH_MASK),
        )
    })
}

/// Feature setting from a code object v4+ two-bit field. `unsupported` (the target has
/// no such mode) is `Any`: the code runs the same under either setting.
fn setting(bits: u32) -> Setting {
    match bits {
        0b10 => Setting::Off,
        0b11 => Setting::On,
        _ => Setting::Any,
    }
}

fn target_of(source: &Source, arch: Arch) -> Result<Target, LiftError> {
    let header = &source.elf.header;
    if header.os_abi != ELFOSABI_AMDGPU_HSA {
        return Err(module_error(
            Rule::UnsupportedArch,
            format!("EI_OSABI {} is not AMDGPU_HSA", header.os_abi),
        ));
    }
    if header.abi_version < ELFABIVERSION_AMDGPU_HSA_V4 {
        return Err(module_error(
            Rule::UnsupportedArch,
            format!(
                "EI_ABIVERSION {} is below code object v4",
                header.abi_version
            ),
        ));
    }
    let declared = arch_of_mach(header.e_flags & EF_MACH_MASK);
    if declared != Some(arch) {
        return Err(module_error(
            Rule::Envelope,
            format!(
                "ELF EF_AMDGPU_MACH {:#x} disagrees with {arch:?}",
                header.e_flags & EF_MACH_MASK
            ),
        ));
    }
    Ok(Target {
        arch,
        xnack: setting((header.e_flags & EF_XNACK_MASK) >> 8),
        sramecc: setting((header.e_flags & EF_SRAMECC_MASK) >> 10),
        abi_version: header.abi_version - ELFABIVERSION_AMDGPU_HSA_V4 + 4,
    })
}

/// The device ELF bytes of the input (the bundle's HIP payload, or the input itself).
fn device_elf<'a>(bytes: &'a [u8], source: &Source, arch: Arch) -> Result<&'a [u8], LiftError> {
    let Some(bundle) = &source.bundle else {
        return Ok(bytes);
    };
    let (_, payloads) = Bundle::read(bytes).map_err(|e| module_error(Rule::Envelope, e))?;
    let index = bundle
        .device_entry(arch)
        .map_err(|e| module_error(Rule::Envelope, e))?;
    Ok(payloads[index])
}

/// Lift a code object (ELF or uncompressed HIP bundle) with a byte-exact proof.
pub fn lift_object(bytes: &[u8], options: Options) -> Result<Lifted<Program>, LiftError> {
    let arch = input_arch(bytes)?;
    let (source, images) =
        Source::read(bytes, arch).map_err(|e| module_error(Rule::Envelope, e))?;
    let target = target_of(&source, arch)?;
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(module_error(
            Rule::UnsupportedArch,
            format!("{arch:?} has no opcode table"),
        ));
    }
    let object_sha = sha256(device_elf(bytes, &source, arch)?);
    if images.len() != source.elf.kernels.len() {
        return Err(module_error(
            Rule::Envelope,
            "kernel images disagree with envelope slots",
        ));
    }
    let mut kernels = Vec::with_capacity(images.len());
    let mut streams = Vec::with_capacity(images.len());
    for (image, slot) in images.iter().zip(&source.elf.kernels) {
        let kernel = lift_kernel(image, slot, arch, options.frontend, object_sha)?;
        let code = emit::bytes(&kernel, arch)
            .map_err(|e| kernel_error(image, image.entry_va, Rule::StreamIdentity, e))?;
        if code != image.code {
            let at = code
                .iter()
                .zip(&image.code)
                .position(|(a, b)| a != b)
                .unwrap_or(code.len().min(image.code.len()));
            return Err(kernel_error(
                image,
                image.entry_va + at as u64,
                Rule::StreamIdentity,
                "label-lowered stream differs from the input",
            ));
        }
        streams.push(StreamProof {
            entry: slot.entry_va,
            size: slot.size,
            stream_sha256: sha256(&code),
        });
        kernels.push(kernel);
    }
    let program = Program {
        target,
        kernels,
        source: Some(source),
    };
    program
        .validate()
        .map_err(|e| module_error(Rule::Program, e))?;
    let emitted = emit::module(&program).map_err(|e| module_error(Rule::ModuleIdentity, e))?;
    if emitted != bytes {
        let at = emitted
            .iter()
            .zip(bytes)
            .position(|(a, b)| a != b)
            .unwrap_or(emitted.len().min(bytes.len()));
        return Err(LiftError::Rejected {
            kernel: None,
            offset: at as u64,
            rule: Rule::ModuleIdentity,
            reason: "re-emitted module differs from the input".into(),
        });
    }
    Ok(Lifted {
        program,
        proof: RoundTripProof {
            input_sha256: sha256(bytes),
            tools: Vec::new(),
            streams,
            sidecar_sha256: None,
        },
        report: LiftReport::default(),
    })
}

fn rejected(kernel: &str, offset: u64, rule: Rule, reason: impl ToString) -> LiftError {
    LiftError::Rejected {
        kernel: Some(kernel.to_owned()),
        offset,
        rule,
        reason: reason.to_string(),
    }
}

fn kernel_error(image: &KernelImage, offset: u64, rule: Rule, reason: impl ToString) -> LiftError {
    rejected(&image.name, offset, rule, reason)
}

fn lift_kernel(
    image: &KernelImage,
    slot: &KernelSlot,
    arch: Arch,
    frontend: Frontend,
    object_sha: [u8; 32],
) -> Result<Kernel, LiftError> {
    if slot.name != image.name
        || slot.entry_va != image.entry_va
        || slot.size != image.code.len() as u64
    {
        return Err(kernel_error(
            image,
            image.entry_va,
            Rule::Envelope,
            "kernel image disagrees with its envelope slot",
        ));
    }
    if image.code.len() % 4 != 0 {
        return Err(kernel_error(
            image,
            image.entry_va,
            Rule::Decode,
            format!("symbol size {} is not a dword multiple", image.code.len()),
        ));
    }
    let body = decode_body(
        &image.name,
        &image.code,
        image.entry_va,
        arch,
        object_sha,
        frontend,
    )?;
    kd::validate(&image.descriptor, arch)
        .map_err(|e| kernel_error(image, slot.kd_va, Rule::Descriptor, e))?;
    metadata::check(&image.metadata.parsed, &image.name, &image.descriptor)
        .map_err(|e| kernel_error(image, slot.kd_va, Rule::Metadata, e))?;
    let wave = if image.descriptor.kernel_code_properties.wave32() {
        Wave::Wave32
    } else {
        Wave::Wave64
    };
    Ok(Kernel {
        symbol: SymbolId(image.name.clone()),
        wave,
        abi: Abi::Hsa {
            descriptor: image.descriptor.clone(),
            metadata: image.metadata.clone(),
        },
        body,
        origin: KernelOrigin::Frontend {
            kind: frontend,
            object_sha256: object_sha,
            entry_va: slot.entry_va,
            size: slot.size,
        },
    })
}

/// Decode one kernel stream (`code` at `entry_va`) into a typed body: every instruction is
/// decoded by the target codec, re-encoded from its typed fields and compared word for word,
/// the CFG is built from the encoded branch offsets and the clause/delay windows are checked.
/// Shared by the HSA and the raw lifters; `prov.bytes` is recorded but never read back.
fn decode_body(
    name: &str,
    code: &[u8],
    entry_va: u64,
    arch: Arch,
    object_sha: [u8; 32],
    frontend: Frontend,
) -> Result<Body, LiftError> {
    let words: Vec<u32> = code
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("dword")))
        .collect();
    let mut body = Body::default();
    let mut vas = Vec::new();
    let mut at = 0;
    while at < words.len() {
        let va = entry_va + 4 * at as u64;
        let (mut inst, n) = match arch {
            Arch::Gfx1100 | Arch::Gfx1151 => gfx11::decode(arch, &words[at..]),
            _ => gfx12::decode(&words[at..]),
        }
        .map_err(|e| rejected(name, va, Rule::Decode, e))?;
        let original = &words[at..at + n];
        let encoded = gfx12::encode_for(arch, &inst)
            .map_err(|e| rejected(name, va, Rule::StreamIdentity, e))?;
        if encoded.as_slice() != original {
            return Err(rejected(
                name,
                va,
                Rule::StreamIdentity,
                format!("re-encoded {encoded:08x?} != input {original:08x?}"),
            ));
        }
        let mut kept = [0u32; 3];
        kept[..n].copy_from_slice(original);
        let pc = u32::try_from(va)
            .map_err(|_| rejected(name, va, Rule::Envelope, "VA exceeds 32 bits"))?;
        inst.prov = Provenance {
            source: Origin::Object(ObjectSha(object_sha)),
            pc: Some(pc),
            bytes: Some(kept),
            line: None,
            edit: None,
        };
        vas.push(va);
        let id = body.insts.insert(inst);
        body.layout.push(id);
        at += n;
    }
    let va_at = |index: usize| vas.get(index).copied().unwrap_or(entry_va);
    build_blocks(&mut body, arch).map_err(|e| {
        let (index, rule) = match e {
            CfgError::TargetOutsideKernel { index, .. }
            | CfgError::TargetMidInstruction { index, .. } => (index, Rule::BranchTarget),
            CfgError::MissingOffset { index } | CfgError::UnexpectedLabel { index } => {
                (index, Rule::ControlFlow)
            }
            CfgError::MissingEndPgm => (vas.len().saturating_sub(1), Rule::ControlFlow),
            CfgError::Empty
            | CfgError::AlreadyBuilt
            | CfgError::DanglingInst { .. }
            | CfgError::BlocksNotBuilt => (0, Rule::ControlFlow),
        };
        rejected(name, va_at(index), rule, e)
    })?;
    // hipcc legally emits `s_delay_alu` hints whose consumer lies past a conditional branch; PM-authored code does not.
    check_windows_with(&body, frontend == Frontend::Hipcc).map_err(|e| {
        let index = match e {
            WindowError::CrossesLeader { index, .. }
            | WindowError::ClauseLengthMissing { index }
            | WindowError::NonMemoryMember { index, .. }
            | WindowError::MixedClass { index, .. }
            | WindowError::WaitInsideClause { index, .. }
            | WindowError::EndPgmInsideClause { index }
            | WindowError::HintMissing { index }
            | WindowError::DelayCrossesLeader { index } => index,
            WindowError::BlocksNotBuilt => 0,
        };
        rejected(name, va_at(index), Rule::Window, e)
    })?;
    Ok(body)
}

/// Lift an authoritative `.s` through `object`, the code object the pinned assembler
/// produced from it (`peacemaker-front` drives `llvm-mc`/`ld.lld`). The byte proof is
/// `lift_object(object)`'s. The source must agree with it: every kernel symbol is a label,
/// the instruction lines from it up to the next kernel label (or the end) are exactly the
/// kernel's instructions, each branch names the label of its target block, every other
/// line parses to the decoded instruction (opcode, form, operands, modifiers, literal),
/// and every label inside a kernel starts a block. Instructions then carry
/// `Source::Text` provenance with their `.s` line; pc and words stay the object's.
pub fn lift_text(
    source: &str,
    object: &[u8],
    options: Options,
) -> Result<Lifted<Program>, LiftError> {
    let mut lifted = lift_object(object, options)?;
    let file = parse_source(source).map_err(|e| module_error(Rule::Text, e))?;
    let text_sha = TextSha(sha256(source.as_bytes()));
    let arch = lifted.program.target.arch;
    // (1-based line number, text) of every instruction line, indexed by instruction ordinal.
    let lines: Vec<(usize, &str)> = file
        .lines
        .iter()
        .filter_map(|line| match &line.kind {
            SourceKind::Inst(text) => Some((line.line_no, text.as_str())),
            _ => None,
        })
        .collect();
    let starts = lifted
        .program
        .kernels
        .iter()
        .enumerate()
        .map(|(k, kernel)| {
            file.labels
                .iter()
                .find(|(name, _)| *name == kernel.symbol.0)
                .map(|(_, at)| (*at, k))
                .ok_or_else(|| {
                    module_error(
                        Rule::Text,
                        format!("kernel {} is not a label in the source", kernel.symbol.0),
                    )
                })
        })
        .collect::<Result<Vec<(usize, usize)>, _>>()?;
    let mut bounds: Vec<usize> = starts.iter().map(|(at, _)| *at).collect();
    bounds.push(lines.len());
    bounds.sort_unstable();
    for &(first, k) in &starts {
        let end = bounds[bounds.partition_point(|&b| b <= first)];
        let kernel = &mut lifted.program.kernels[k];
        let name = kernel.symbol.0.clone();
        let fail = |offset: u64, reason: String| LiftError::Rejected {
            kernel: Some(name.clone()),
            offset,
            rule: Rule::Text,
            reason,
        };
        let entry = match kernel.origin {
            KernelOrigin::Frontend { entry_va, .. } => entry_va,
            KernelOrigin::Authored { .. } => 0,
        };
        let body = &mut kernel.body;
        if end - first != body.layout.len() {
            return Err(fail(
                entry,
                format!(
                    "{} source instruction lines, {} decoded instructions",
                    end - first,
                    body.layout.len()
                ),
            ));
        }
        let block_at = |ordinal: usize| {
            body.blocks
                .iter()
                .find(|b| b.range.0 == ordinal)
                .map(|b| b.id)
        };
        let labels: Vec<(&str, usize)> = file
            .labels
            .iter()
            .filter(|(_, at)| (first..end).contains(at))
            .map(|(name, at)| (name.as_str(), at - first))
            .collect();
        for (label, ordinal) in &labels {
            if block_at(*ordinal).is_none() {
                return Err(fail(
                    entry,
                    format!("label {label} at instruction {ordinal} does not start a block"),
                ));
            }
        }
        let starts_of: Vec<Option<BlockId>> = labels
            .iter()
            .map(|(_, ordinal)| block_at(*ordinal))
            .collect();
        for ordinal in 0..body.layout.len() {
            let (line_no, text) = lines[first + ordinal];
            let inst = body
                .insts
                .get_mut(body.layout[ordinal])
                .expect("layout references live instructions");
            let va = inst.prov.pc.map_or(entry, u64::from);
            let label_target = inst.operands.iter().find_map(|op| match op {
                Operand::Label(b) => Some(*b),
                _ => None,
            });
            match label_target {
                Some(target) => {
                    let mut parts = text.split_whitespace();
                    let (mnemonic, operand) =
                        (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                    if Some(mnemonic) != inst.op.name(arch) {
                        return Err(fail(
                            va,
                            format!("line {line_no}: {text:?} is not {:?}", inst.op.name(arch)),
                        ));
                    }
                    let named = labels
                        .iter()
                        .zip(&starts_of)
                        .find(|((label, _), _)| *label == operand)
                        .and_then(|(_, block)| *block);
                    if named != Some(target) {
                        return Err(fail(
                            va,
                            format!(
                                "line {line_no}: {text:?} does not name the label of block {}",
                                target.0
                            ),
                        ));
                    }
                }
                None => {
                    let parsed = parse_line(text, arch)
                        .map_err(|e| fail(va, format!("line {line_no}: {text:?}: {e}")))?;
                    if (
                        parsed.op,
                        parsed.form,
                        &parsed.operands,
                        &parsed.mods,
                        parsed.literal,
                    ) != (inst.op, inst.form, &inst.operands, &inst.mods, inst.literal)
                    {
                        return Err(fail(
                            va,
                            format!(
                                "line {line_no}: {text:?} disagrees with the object's instruction"
                            ),
                        ));
                    }
                }
            }
            inst.prov.source = Origin::Text(text_sha);
            inst.prov.line = Some(line_no as u32);
        }
    }
    Ok(lifted)
}

/// Emission from typed fields: branch labels are lowered against the current layout and
/// every instruction is re-encoded by the codec; `prov.bytes` is never read.
pub mod emit {
    use peacemaker_ir::codec::gfx12;
    use peacemaker_ir::edit::{lower_labels, EditError};
    use peacemaker_ir::inst::{Abi, Arch, Inst, Kernel, Program};
    use peacemaker_ir::state::Lifted;

    use crate::bundle::{SourceCodec, SourceError};
    use crate::elf::{EnvelopeCodec, KernelParts};
    use crate::text::{kernel_lines, PrintError};

    #[derive(Debug, thiserror::Error)]
    pub enum EmitError {
        #[error("{kernel}: {error}")]
        Lower {
            kernel: String,
            #[source]
            error: EditError,
        },
        #[error("{kernel}: layout index {index}: {reason}")]
        Encode {
            kernel: String,
            index: usize,
            reason: String,
        },
        #[error("kernel {0} has a raw ABI and cannot be packaged without an adapter")]
        RawAbi(String),
        #[error("program has no source envelope to emit into")]
        NoEnvelope,
        #[error(transparent)]
        Source(#[from] SourceError),
        #[error(transparent)]
        Print(#[from] PrintError),
        #[error("the emitted module does not lift back: {0}")]
        Relift(String),
        #[error("the emitted module lifts back to a different {what} for kernel {kernel}")]
        Mismatch { kernel: String, what: String },
    }

    /// The kernel's instructions in layout order with every `Label` lowered to its SOPP
    /// offset (signed dwords from the next PC) under the current layout.
    pub fn insts(kernel: &Kernel, arch: Arch) -> Result<Vec<Inst>, EmitError> {
        lower_labels(&kernel.body, arch).map_err(|error| EmitError::Lower {
            kernel: kernel.symbol.0.clone(),
            error,
        })
    }

    /// Encoded words of the kernel stream.
    pub fn words(kernel: &Kernel, arch: Arch) -> Result<Vec<u32>, EmitError> {
        let mut out = Vec::with_capacity(kernel.body.layout.len() * 2);
        for (index, inst) in insts(kernel, arch)?.iter().enumerate() {
            let encoded = gfx12::encode_for(arch, inst).map_err(|e| EmitError::Encode {
                kernel: kernel.symbol.0.clone(),
                index,
                reason: e.to_string(),
            })?;
            out.extend_from_slice(&encoded);
        }
        Ok(out)
    }

    /// Little-endian bytes of the kernel stream.
    pub fn bytes(kernel: &Kernel, arch: Arch) -> Result<Vec<u8>, EmitError> {
        Ok(words(kernel, arch)?
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect())
    }

    /// Canonical (pinned `llvm-objdump`) text, one line per instruction; the input of
    /// assembler parity (`mc(text(k)) == bytes(k)` where the syntax is lossless).
    pub fn text(kernel: &Kernel, arch: Arch) -> Result<Vec<String>, EmitError> {
        Ok(kernel_lines(&insts(kernel, arch)?, arch)?)
    }

    fn parts<'a>(
        program: &'a Program,
        codes: &'a [Vec<u8>],
    ) -> Result<Vec<KernelParts<'a>>, EmitError> {
        program
            .kernels
            .iter()
            .zip(codes)
            .map(|(kernel, code)| match &kernel.abi {
                Abi::Hsa {
                    descriptor,
                    metadata,
                } => Ok(KernelParts {
                    code,
                    descriptor,
                    metadata,
                }),
                Abi::Raw { .. } => Err(EmitError::RawAbi(kernel.symbol.0.clone())),
            })
            .collect()
    }

    /// The module: the retained envelope with every kernel's re-encoded stream and
    /// re-serialised descriptor and metadata spliced into its slot (re-laid out when the
    /// code, the metadata note or a kernel name no longer fits, `crate::layout`).
    pub fn module(program: &Program) -> Result<Vec<u8>, EmitError> {
        let source = program.source.as_ref().ok_or(EmitError::NoEnvelope)?;
        let codes = program
            .kernels
            .iter()
            .map(|kernel| bytes(kernel, program.target.arch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(source.write(&parts(program, &codes)?)?)
    }

    /// [`module`], then re-read: the bytes must lift (`lift_object`) to the same kernels
    /// in the same order — symbol, instruction stream, metadata, and the descriptor the
    /// layout derived (entry offset and `INST_PREF_SIZE` follow the new layout).
    pub fn checked(
        program: &Program,
        options: crate::Options,
    ) -> Result<(Vec<u8>, Lifted<Program>), EmitError> {
        let source = program.source.as_ref().ok_or(EmitError::NoEnvelope)?;
        let codes = program
            .kernels
            .iter()
            .map(|kernel| bytes(kernel, program.target.arch))
            .collect::<Result<Vec<_>, _>>()?;
        let parts = parts(program, &codes)?;
        let layout = source.elf.layout(&parts).map_err(SourceError::from)?;
        let out = source.write(&parts)?;
        let lifted =
            crate::lift_object(&out, options).map_err(|e| EmitError::Relift(e.to_string()))?;
        if lifted.program.kernels.len() != program.kernels.len() {
            return Err(EmitError::Mismatch {
                kernel: "module".into(),
                what: "kernel count".into(),
            });
        }
        for ((kernel, back), descriptor) in program
            .kernels
            .iter()
            .zip(&lifted.program.kernels)
            .zip(&layout.descriptors)
        {
            let differ = |what: &str| EmitError::Mismatch {
                kernel: kernel.symbol.0.clone(),
                what: what.into(),
            };
            if kernel.symbol != back.symbol {
                return Err(differ("symbol"));
            }
            if words(kernel, program.target.arch)? != words(back, program.target.arch)? {
                return Err(differ("instruction stream"));
            }
            let (
                Abi::Hsa { metadata, .. },
                Abi::Hsa {
                    descriptor: read,
                    metadata: read_meta,
                },
            ) = (&kernel.abi, &back.abi)
            else {
                return Err(differ("ABI"));
            };
            if metadata.parsed != read_meta.parsed {
                return Err(differ("metadata"));
            }
            if read != descriptor {
                return Err(differ("descriptor"));
            }
        }
        Ok((out, lifted))
    }
}
