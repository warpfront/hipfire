// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native code-object emission: the builder's `.s` (one kernel or a product
//! module) → a code object laid out exactly as `llvm-mc` + `ld.lld -shared`
//! lay it out, and its HIP offload bundle — no ROCm tool in the path. The
//! only byte difference from the ROCm oracle is the `.comment` stamp
//! (`hipfire peacemaker native-emit <version>` instead of the linker's
//! identification string) and the file offsets that follow it.
//!
//! Each instruction line is lowered to a `peacemaker_ir` instruction by
//! `peacemaker_lift::text::parse_line` and encoded by the gfx11/gfx12
//! codec; branch labels resolve to word offsets. The directives the builder
//! writes (`emit::assembly`, the module combiners, `profile`) are the
//! accepted set; any other directive is an error.
//!
//! The one `.text` directive pair beyond the kernels is the module's end guard,
//! `emit::CODE_END_TAIL` (`.p2alignl 7, 3214868480` then `.fill 96, 4, 3214868480`, LLVM's
//! gfx10+ AMDHSA `s_code_end` padding): accepted once, after the last kernel, nothing may follow it
//! in `.text`, and it is optional in the source (the typed [`link_encoded`] paths never add it).
//!
//! gfx942 (CDNA3, wave64) has no text front end here: its kernels are typed
//! (`crate::cdna`), encoded by `peacemaker_cdna`, and linked through
//! [`link_encoded`] / [`link_encoded_versioned`] as code object v5 or v6 with
//! the same descriptor, metadata and ELF writers (the `.amdhsa_*` directive
//! semantics are those of `llvm-mc -mcpu=gfx942`; see `descriptor`).
mod descriptor;
mod elf;
mod metadata;

use crate::Arch;
use peacemaker_ir::inst::Arch as IrArch;
use std::collections::HashMap;

/// Provenance of the native writer, recorded as its tool identity.
pub const VERSION: &str = concat!(
    "peacemaker-isa ",
    env!("CARGO_PKG_VERSION"),
    " native code object v6 writer (ld.lld 24.0.0 layout)"
);
/// `clang-offload-bundler`'s host entry spelling; ROCm `hipcc` (and the
/// committed `.hxaco` bundles) use the trailing-hyphen form instead.
pub const DEFAULT_HOST_TARGET: &str = "host-x86_64-unknown-linux-gnu";

/// The RDNA codec architecture; CDNA kernels are typed (`crate::cdna`) and never take the text path.
fn ir_arch(arch: Arch) -> Result<IrArch, String> {
    match arch {
        Arch::Gfx1100 => Ok(IrArch::Gfx1100),
        Arch::Gfx1151 => Ok(IrArch::Gfx1151),
        Arch::Gfx1201 => Ok(IrArch::Gfx1201),
        Arch::Gfx942 => Err(
            "gfx942 code is emitted by the typed cdna builder, not the RDNA text assembler".into(),
        ),
    }
}

/// Encode one instruction line (no label operands) onto `out`; returns its dword count.
fn encode_into(line: &str, arch: Arch, out: &mut Vec<u32>) -> Result<usize, String> {
    let inst = peacemaker_lift::text::parse_line(line, ir_arch(arch)?)
        .map_err(|e| format!("{line}: {e}"))?;
    let words = if arch.gfx12() {
        peacemaker_ir::codec::gfx12::encode(&inst)
    } else {
        peacemaker_ir::codec::gfx11::encode(ir_arch(arch)?, &inst)
    }
    .map_err(|e| format!("{line}: {e}"))?;
    out.extend_from_slice(&words);
    Ok(words.len())
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    Text,
    Rodata,
}

struct Kernel<'a> {
    symbol: &'a str,
    /// Encoded words, with `(word index, mnemonic, label)` branches to patch.
    words: Vec<u32>,
    branches: Vec<(usize, &'a str, &'a str)>,
    size: Option<(&'a str, &'a str)>,
    descriptor: Option<Vec<(&'a str, &'a str)>>,
}

/// Progress through the module's end-of-text guard (`emit::CODE_END_TAIL`): `.p2alignl 7, 3214868480`
/// then `.fill 96, 4, 3214868480`, once, after the last kernel. Nothing may follow it in `.text`.
#[derive(Clone, Copy, PartialEq)]
enum CodeEnd {
    None,
    Aligned,
    Filled,
}

/// Strip a `;` comment.
fn code(line: &str) -> &str {
    line.split_once(';').map_or(line, |(c, _)| c).trim()
}

/// Assemble and link builder source into a code object for `arch`.
pub fn assemble(source: &str, arch: Arch) -> Result<Vec<u8>, String> {
    let mut section = Section::None;
    let mut target = false;
    let mut version = false;
    let mut kernels: Vec<Kernel<'_>> = Vec::new();
    // Kernel symbol attributes: (.protected, .globl, .type @function).
    let mut attributes: HashMap<&str, [bool; 3]> = HashMap::new();
    let mut pending_align: Option<u32> = None;
    // Label → (kernel, word index).
    let mut labels: HashMap<&str, (usize, usize)> = HashMap::new();
    let mut code_end = CodeEnd::None;
    let mut metadata: Option<String> = None;
    let mut lines = source.lines();
    while let Some(raw) = lines.next() {
        let line = code(raw);
        if line.is_empty() {
            continue;
        }
        let (head, rest) = line
            .split_once(char::is_whitespace)
            .map_or((line, ""), |(h, r)| (h, r.trim()));
        match head {
            ".amdgcn_target" => {
                if rest != format!("\"amdgcn-amd-amdhsa--{}\"", arch.name()) {
                    return Err(format!("target {rest} is not {}", arch.name()));
                }
                target = true;
            }
            ".amdhsa_code_object_version" => {
                if rest != "6" {
                    return Err(format!(
                        "code object version {rest}: the native writer emits version 6"
                    ));
                }
                version = true;
            }
            ".text" if rest.is_empty() => section = Section::Text,
            ".section" if rest == ".rodata,\"a\",@progbits" => section = Section::Rodata,
            ".protected" | ".globl" => {
                attributes.entry(rest).or_default()[usize::from(head == ".globl")] = true
            }
            ".type" => {
                let symbol = rest
                    .strip_suffix(",@function")
                    .ok_or_else(|| format!("unsupported {line}"))?;
                attributes.entry(symbol).or_default()[2] = true;
            }
            ".p2alignl" => {
                if section != Section::Text
                    || kernels.is_empty()
                    || pending_align.is_some()
                    || code_end != CodeEnd::None
                    || rest != "7, 3214868480"
                {
                    return Err(format!("unsupported {line}: the only accepted form is the single end-of-text guard `.p2alignl 7, 3214868480` after the last kernel"));
                }
                code_end = CodeEnd::Aligned;
            }
            ".fill" => {
                if section != Section::Text
                    || code_end != CodeEnd::Aligned
                    || rest != "96, 4, 3214868480"
                {
                    return Err(format!("unsupported {line}: the only accepted form is `.fill 96, 4, 3214868480` directly after the end-of-text `.p2alignl`"));
                }
                code_end = CodeEnd::Filled;
            }
            ".p2align" if code_end != CodeEnd::None => {
                return Err(format!(
                    "{line}: no kernel may follow the end-of-text guard"
                ))
            }
            ".p2align" => {
                let expected = match section {
                    Section::Text => "8",
                    Section::Rodata => "6",
                    Section::None => "",
                };
                if rest != expected {
                    return Err(format!("unsupported {line} in this section"));
                }
                pending_align = Some(rest.parse().unwrap());
            }
            ".size" => {
                let (symbol, expr) = rest
                    .split_once(", ")
                    .ok_or_else(|| format!("unsupported {line}"))?;
                let (end, start) = expr
                    .split_once('-')
                    .ok_or_else(|| format!("unsupported {line}"))?;
                let kernel = kernels
                    .iter_mut()
                    .find(|k| k.symbol == symbol)
                    .ok_or_else(|| format!("{line}: unknown symbol"))?;
                if kernel.size.replace((end, start)).is_some() {
                    return Err(format!("duplicate {line}"));
                }
            }
            ".amdhsa_kernel" => {
                if section != Section::Rodata || pending_align.take() != Some(6) {
                    return Err(format!(
                        "{line}: descriptor must follow .p2align 6 in .rodata"
                    ));
                }
                let mut directives = Vec::new();
                loop {
                    let l = code(lines.next().ok_or("unterminated .amdhsa_kernel")?);
                    if l == ".end_amdhsa_kernel" {
                        break;
                    }
                    if l.is_empty() {
                        continue;
                    }
                    let (name, value) = l
                        .split_once(char::is_whitespace)
                        .ok_or_else(|| format!("bad directive {l}"))?;
                    directives.push((name, value.trim()));
                }
                let kernel = kernels
                    .iter_mut()
                    .find(|k| k.symbol == rest)
                    .ok_or_else(|| format!("descriptor for undefined kernel {rest}"))?;
                if kernel.descriptor.replace(directives).is_some() {
                    return Err(format!("duplicate descriptor for {rest}"));
                }
            }
            ".amdgpu_metadata" => {
                if metadata.is_some() {
                    return Err("duplicate .amdgpu_metadata".into());
                }
                if lines.next().map(str::trim_end) != Some("---") {
                    return Err(".amdgpu_metadata must open with ---".into());
                }
                let mut yaml = String::new();
                loop {
                    let l = lines.next().ok_or("unterminated .amdgpu_metadata")?;
                    if l.trim_end() == "..." {
                        break;
                    }
                    yaml.push_str(l);
                    yaml.push('\n');
                }
                if lines.next().map(str::trim) != Some(".end_amdgpu_metadata") {
                    return Err("metadata must close with .end_amdgpu_metadata".into());
                }
                metadata = Some(yaml);
            }
            _ if line.ends_with(':') && !line.contains(char::is_whitespace) => {
                let name = &line[..line.len() - 1];
                if section != Section::Text {
                    return Err(format!("label {name} outside .text"));
                }
                if code_end != CodeEnd::None {
                    return Err(format!("label {name} follows the end-of-text guard"));
                }
                if !name.starts_with(".L") {
                    if attributes.get(name) != Some(&[true; 3]) {
                        return Err(format!(
                            "kernel {name} must be .protected, .globl and .type @function"
                        ));
                    }
                    if pending_align.take() != Some(8) {
                        return Err(format!("kernel {name} must follow .p2align 8"));
                    }
                    if kernels.iter().any(|k| k.symbol == name) {
                        return Err(format!("duplicate kernel {name}"));
                    }
                    kernels.push(Kernel {
                        symbol: name,
                        words: Vec::new(),
                        branches: Vec::new(),
                        size: None,
                        descriptor: None,
                    });
                }
                let k = kernels
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| format!("label {name} before any kernel"))?;
                if labels.insert(name, (k, kernels[k].words.len())).is_some() {
                    return Err(format!("duplicate label {name}"));
                }
            }
            _ if head.starts_with('.') => return Err(format!("unsupported directive {line}")),
            _ => {
                if section != Section::Text || pending_align.is_some() || code_end != CodeEnd::None
                {
                    return Err(format!("instruction outside a kernel: {line}"));
                }
                let kernel = kernels
                    .last_mut()
                    .ok_or_else(|| format!("instruction before any kernel: {line}"))?;
                if (head == "s_branch" || head.starts_with("s_cbranch_")) && rest.starts_with(".L")
                {
                    kernel.branches.push((kernel.words.len(), head, rest));
                    kernel.words.push(0);
                } else {
                    encode_into(line, arch, &mut kernel.words)?;
                }
            }
        }
    }
    if !target || !version {
        return Err("missing .amdgcn_target or .amdhsa_code_object_version".into());
    }
    if code_end == CodeEnd::Aligned {
        return Err("the end-of-text `.p2alignl` is not followed by its `.fill`".into());
    }
    let metadata = metadata::msgpack(&metadata.ok_or("missing .amdgpu_metadata")?)?;
    let mut linked = Vec::with_capacity(kernels.len());
    let mut branch = Vec::with_capacity(1);
    for (k, kernel) in kernels.iter_mut().enumerate() {
        for &(at, mnemonic, label) in &kernel.branches {
            let &(target_kernel, target) = labels
                .get(label)
                .ok_or_else(|| format!("{}: undefined label {label}", kernel.symbol))?;
            if target_kernel != k {
                return Err(format!(
                    "{}: branch to {label} leaves the kernel",
                    kernel.symbol
                ));
            }
            let offset = target as i64 - (at as i64 + 1);
            if i16::try_from(offset).is_err() {
                return Err(format!(
                    "{}: branch to {label} is out of range",
                    kernel.symbol
                ));
            }
            branch.clear();
            if encode_into(&format!("{mnemonic} {offset}"), arch, &mut branch)? != 1 {
                return Err(format!("{mnemonic}: branch is not one dword"));
            }
            kernel.words[at] = branch[0];
        }
        let (end, start) = kernel
            .size
            .ok_or_else(|| format!("{}: missing .size", kernel.symbol))?;
        if start != kernel.symbol || labels.get(end) != Some(&(k, kernel.words.len())) {
            return Err(format!(
                "{}: .size must span the kernel to its end label",
                kernel.symbol
            ));
        }
        let code_bytes = 4 * kernel.words.len() as u64;
        let directives = kernel
            .descriptor
            .as_ref()
            .ok_or_else(|| format!("{}: missing .amdhsa_kernel", kernel.symbol))?;
        let descriptor = descriptor::encode(arch, kernel.symbol, directives, end, code_bytes)?;
        linked.push(elf::Kernel {
            symbol: kernel.symbol.to_owned(),
            code: kernel.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            descriptor,
        });
    }
    elf::link(arch, 6, &linked, &metadata, code_end == CodeEnd::Filled)
}

/// One kernel whose code is already encoded: the typed encoder's words (not text) plus
/// the `.amdhsa_*` directive set of its descriptor. On gfx11/gfx12 `.amdhsa_inst_pref_size`
/// is derived from the code extent here, exactly as `assemble` derives it, and must not be
/// listed; gfx942 has no such field and rejects it.
pub struct EncodedKernel<'a> {
    pub symbol: &'a str,
    pub words: &'a [u32],
    /// `(".amdhsa_<field>", value)`, each field once.
    pub directives: &'a [(String, String)],
}

/// Link already-encoded kernels and their `.amdgpu_metadata` YAML body (the lines between
/// `---` and `...`) into the same v6 code object `assemble` lays out. This is the descriptor,
/// metadata and ELF writer `assemble` uses, minus the text front end: no instruction is
/// parsed or re-encoded here. Equivalent to [`link_encoded_versioned`] at version 6.
pub fn link_encoded(
    kernels: &[EncodedKernel<'_>],
    metadata_yaml: &str,
    arch: Arch,
) -> Result<Vec<u8>, String> {
    link_encoded_versioned(kernels, metadata_yaml, arch, 6)
}

/// [`link_encoded`] at `code_object_version` 5 or 6 (`EI_ABIVERSION` 3 or 4; the descriptor,
/// metadata and layout are identical). Version 5 is gfx942-only: the RDNA kernels are all v6.
/// The caller's directives and `.amdgpu_metadata` carry the version-specific text of the `.s`
/// the ROCm oracle assembles (`.amdhsa_code_object_version`, and for a v5 `hipcc` kernel
/// `.amdhsa_reserve_xnack_mask 1`, which the descriptor accepts and which changes no bit:
/// gfx942's target ID is `xnack any`); `amdhsa.version` is `[1, 2]` for both.
pub fn link_encoded_versioned(
    kernels: &[EncodedKernel<'_>],
    metadata_yaml: &str,
    arch: Arch,
    code_object_version: u32,
) -> Result<Vec<u8>, String> {
    if kernels.is_empty() {
        return Err("no kernel to link".into());
    }
    let metadata = metadata::msgpack(metadata_yaml)?;
    let mut linked = Vec::with_capacity(kernels.len());
    for (index, kernel) in kernels.iter().enumerate() {
        if kernels[..index].iter().any(|k| k.symbol == kernel.symbol) {
            return Err(format!("duplicate kernel {}", kernel.symbol));
        }
        if kernel.words.is_empty() {
            return Err(format!("{}: empty kernel", kernel.symbol));
        }
        let end = format!(".L{}_end", kernel.symbol);
        let mut directives: Vec<(&str, &str)> = kernel
            .directives
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect();
        let inst_pref;
        if !arch.wave64() {
            let mask = if arch.gfx12() { 4080 } else { 1008 };
            inst_pref = format!("((instprefsize({end}-{})<<4)&{mask})>>4", kernel.symbol);
            if kernel
                .directives
                .iter()
                .any(|(name, _)| name == ".amdhsa_inst_pref_size")
            {
                return Err(format!(
                    "{}: .amdhsa_inst_pref_size is derived from the code extent",
                    kernel.symbol
                ));
            }
            directives.push((".amdhsa_inst_pref_size", inst_pref.as_str()));
        }
        let code_bytes = 4 * kernel.words.len() as u64;
        let descriptor = descriptor::encode(arch, kernel.symbol, &directives, &end, code_bytes)?;
        linked.push(elf::Kernel {
            symbol: kernel.symbol.to_owned(),
            code: kernel.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            descriptor,
        });
    }
    elf::link(arch, code_object_version, &linked, &metadata, false)
}

/// Uncompressed `__CLANG_OFFLOAD_BUNDLE__` of an empty host entry and the
/// HIP device code object, each entry at a 4096-byte boundary
/// (`clang-offload-bundler -type=o -bundle-align=4096`).
pub fn bundle(elf: &[u8], arch: Arch, host_target: &str) -> Vec<u8> {
    const ALIGN: u64 = 4096;
    let triples = [
        host_target.to_owned(),
        format!("hipv4-amdgcn-amd-amdhsa--{}", arch.name()),
    ];
    let payloads: [&[u8]; 2] = [&[], elf];
    let mut out = b"__CLANG_OFFLOAD_BUNDLE__".to_vec();
    out.extend((triples.len() as u64).to_le_bytes());
    let mut end = out.len() as u64 + triples.iter().map(|t| 24 + t.len() as u64).sum::<u64>();
    let mut offsets = Vec::new();
    for (triple, payload) in triples.iter().zip(payloads) {
        let offset = end.next_multiple_of(ALIGN);
        offsets.push(offset);
        end = offset + payload.len() as u64;
        out.extend(offset.to_le_bytes());
        out.extend((payload.len() as u64).to_le_bytes());
        out.extend((triple.len() as u64).to_le_bytes());
        out.extend(triple.as_bytes());
    }
    for (offset, payload) in offsets.into_iter().zip(payloads) {
        out.resize(offset as usize, 0);
        out.extend(payload);
    }
    out
}
