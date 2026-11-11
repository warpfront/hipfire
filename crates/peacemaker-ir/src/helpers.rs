// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Out-of-line device helpers of a lifted object: the executable `STT_FUNC` extents that are not metadata kernels (for
//! example LLVM's `llv_tail`, reached from a kernel through `s_getpc_b64`/`s_swappc_b64`).
//!
//! [`helper_extents`] discovers them from the ELF symbol tables; [`load_helpers`] typed-decodes each one from the lifted
//! envelope's `.text` bytes (only kernel holes are zeroed there) with the same codec selection and re-encode identity
//! rule as the kernel lift. No CFG, hazard or semantic proof is built for a helper: it is a verbatim instruction
//! stream at its original virtual address, which is all the emulator's byte-PC execution (`emu`) needs.

use crate::{
    codec::{gfx11, gfx12},
    envelope::{Envelope, SectionData, Symbol},
    inst::{Arch, KernelOrigin, Program},
    provenance::{ObjectSha, Provenance, Source as Origin},
    Inst,
};
use std::collections::BTreeSet;

pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_NOBITS: u32 = 8;
pub const SHT_DYNSYM: u32 = 11;
pub const SHF_EXECINSTR: u64 = 4;
pub const STT_FUNC: u8 = 2;
pub const SHN_UNDEF: u16 = 0;
pub const SHN_LORESERVE: u16 = 0xff00;

/// A defined, nonempty, executable `STT_FUNC` extent that is not a metadata kernel (an out-of-line device helper such
/// as LLVM's `llv_build_p`), with every symbol naming it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HelperExtent {
    /// Virtual address of the first byte.
    pub start: u64,
    pub size: u64,
    pub names: BTreeSet<String>,
    /// Index of the executable section holding the extent.
    pub section: usize,
}

/// The NUL-terminated UTF-8 name of `symbol`, resolved through the string table linked from symbol-table section
/// `table`.
pub fn helper_symbol_name(env: &Envelope, table: usize, symbol: &Symbol) -> Result<String, String> {
    let link = env.sections[table].header.sh_link as usize;
    let strings = env
        .sections
        .get(link)
        .filter(|s| s.header.sh_type == SHT_STRTAB)
        .ok_or_else(|| format!("symbol table section {table} links to no string table ({link})"))?;
    let SectionData::Bytes(bytes) = &strings.data else {
        return Err(format!("string table section {link} holds no bytes"));
    };
    let tail = bytes.get(symbol.st_name as usize..).ok_or_else(|| {
        format!(
            "symbol name offset {} is outside string table {link}",
            symbol.st_name
        )
    })?;
    let end = tail.iter().position(|&b| b == 0).ok_or_else(|| {
        format!(
            "symbol name at offset {} is not NUL-terminated",
            symbol.st_name
        )
    })?;
    std::str::from_utf8(&tail[..end])
        .map(str::to_owned)
        .map_err(|_| format!("symbol name at offset {} is not UTF-8", symbol.st_name))
}

/// Every defined `STT_FUNC` extent (dynsym and symtab) that is not exactly a metadata kernel extent, sorted by start.
/// Aliases (same extent, any name) are merged; partial overlap, a zero-sized body, a non-executable or out-of-range
/// extent, or a size/address that is not a multiple of 4 is an error (a defined function whose body cannot be
/// established is never silently skipped).
pub fn helper_extents(env: &Envelope) -> Result<Vec<HelperExtent>, String> {
    let mut bodies: Vec<HelperExtent> = Vec::new();
    for (table, sec) in env.sections.iter().enumerate() {
        if !matches!(sec.header.sh_type, SHT_SYMTAB | SHT_DYNSYM) {
            continue;
        }
        let SectionData::Symbols(symbols) = &sec.data else {
            return Err(format!(
                "symbol table section {table} is not parsed as symbols"
            ));
        };
        for s in symbols {
            if s.kind() != STT_FUNC || s.st_shndx == SHN_UNDEF {
                continue;
            }
            let name = helper_symbol_name(env, table, s)?;
            let bad = |why: String| format!("function symbol `{name}`: {why}");
            if s.st_shndx >= SHN_LORESERVE {
                return Err(bad(format!(
                    "defined in reserved section index {:#x}",
                    s.st_shndx
                )));
            }
            if s.st_size == 0 {
                return Err(bad("defined with size 0; its body extent is unknown".into()));
            }
            let section = usize::from(s.st_shndx);
            let h = &env
                .sections
                .get(section)
                .ok_or_else(|| bad(format!("section index {section} does not exist")))?
                .header;
            if h.sh_flags & SHF_EXECINSTR == 0 || h.sh_type == SHT_NOBITS {
                return Err(bad(format!("section {section} is not executable code")));
            }
            let (start, size) = (s.st_value, s.st_size);
            let end = start
                .checked_add(size)
                .ok_or_else(|| bad("extent overflows".into()))?;
            let sec_end = h
                .sh_addr
                .checked_add(h.sh_size)
                .ok_or_else(|| bad(format!("section {section} extent overflows")))?;
            if start < h.sh_addr || end > sec_end {
                return Err(bad(format!(
                    "extent [{start:#x}, {end:#x}) is outside section {section} [{:#x}, {sec_end:#x})",
                    h.sh_addr
                )));
            }
            if start % 4 != 0 || size % 4 != 0 {
                return Err(bad(format!(
                    "extent [{start:#x}, +{size:#x}) is not a multiple of 4"
                )));
            }
            if env
                .kernels
                .iter()
                .any(|k| k.entry_va == start && k.size == size)
            {
                continue;
            }
            if let Some(k) = env
                .kernels
                .iter()
                .find(|k| k.entry_va < end && start < k.entry_va.saturating_add(k.size))
            {
                return Err(bad(format!(
                    "extent [{start:#x}, {end:#x}) partially overlaps kernel `{}`",
                    k.name
                )));
            }
            if let Some(b) = bodies
                .iter_mut()
                .find(|b| b.start == start && b.size == size)
            {
                b.names.insert(name);
                continue;
            }
            if let Some(b) = bodies
                .iter()
                .find(|b| b.start < end && start < b.start + b.size)
            {
                return Err(bad(format!(
                    "extent [{start:#x}, {end:#x}) overlaps helper `{}` [{:#x}, {:#x}) inconsistently",
                    b.names.iter().next().map_or("?", String::as_str),
                    b.start,
                    b.start + b.size
                )));
            }
            bodies.push(HelperExtent {
                start,
                size,
                names: BTreeSet::from([name]),
                section,
            });
        }
    }
    bodies.sort_by_key(|b| b.start);
    Ok(bodies)
}

/// One typed-decoded helper: every instruction re-encodes to the object's words, carries its virtual address in
/// `prov.pc` and its words in `prov.bytes`. Branch operands stay as decoded (`Operand::Imm(ImmField::Sopp(simm16))`,
/// relative to the next instruction); there is no CFG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Helper {
    /// Every symbol naming the extent (aliases included).
    pub names: BTreeSet<String>,
    pub entry_va: u64,
    /// Extent size in bytes.
    pub size: u64,
    pub insts: Vec<Inst>,
}

/// Decode every helper extent of `program.source` (the lifted envelope), sorted by entry. A helper instruction must
/// decode, re-encode to exactly the object's words, and end inside its extent; a virtual address above `u32::MAX`, a
/// program without a lifted source, or one with no frontend-lifted kernel to carry the object digest is an error.
pub fn load_helpers(program: &Program) -> Result<Vec<Helper>, String> {
    let env = &program
        .source
        .as_ref()
        .ok_or("helpers are loaded from the lifted object, but the program has no source")?
        .elf;
    let object_sha = program
        .kernels
        .iter()
        .find_map(|k| match &k.origin {
            KernelOrigin::Frontend { object_sha256, .. } => Some(*object_sha256),
            KernelOrigin::Authored { .. } => None,
        })
        .ok_or("helper provenance needs the object digest of a frontend-lifted kernel")?;
    let arch = program.target.arch;
    let mut out = Vec::new();
    for extent in helper_extents(env)? {
        let name = extent
            .names
            .iter()
            .next()
            .expect("helper has a name")
            .clone();
        let fail = |va: u64, why: String| format!("helper `{name}` at {va:#x}: {why}");
        let section = &env.sections[extent.section];
        let SectionData::Bytes(bytes) = &section.data else {
            return Err(fail(
                extent.start,
                format!("section {} holds no bytes", extent.section),
            ));
        };
        let code = usize::try_from(extent.start - section.header.sh_addr)
            .ok()
            .and_then(|off| {
                let len = usize::try_from(extent.size).ok()?;
                bytes.get(off..off.checked_add(len)?)
            })
            .ok_or_else(|| {
                fail(
                    extent.start,
                    format!("extent is outside the bytes of section {}", extent.section),
                )
            })?;
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("dword")))
            .collect();
        let mut insts = Vec::new();
        let mut at = 0;
        while at < words.len() {
            let va = extent.start + 4 * at as u64;
            let (mut inst, n) = match arch {
                Arch::Gfx1100 | Arch::Gfx1151 => gfx11::decode(arch, &words[at..]),
                _ => gfx12::decode_for(arch, &words[at..]),
            }
            .map_err(|e| fail(va, format!("decode: {e}")))?;
            if n == 0 || n > 3 || at + n > words.len() {
                return Err(fail(
                    va,
                    format!("instruction of {n} dword(s) leaves the helper extent"),
                ));
            }
            let original = &words[at..at + n];
            let encoded =
                gfx12::encode_for(arch, &inst).map_err(|e| fail(va, format!("re-encode: {e}")))?;
            if encoded.as_slice() != original {
                return Err(fail(
                    va,
                    format!("re-encoded {encoded:08x?} != input {original:08x?}"),
                ));
            }
            let mut kept = [0u32; 3];
            kept[..n].copy_from_slice(original);
            inst.prov = Provenance {
                source: Origin::Object(ObjectSha(object_sha)),
                pc: Some(u32::try_from(va).map_err(|_| fail(va, "VA exceeds 32 bits".into()))?),
                bytes: Some(kept),
                line: None,
                edit: None,
            };
            insts.push(inst);
            at += n;
        }
        out.push(Helper {
            names: extent.names,
            entry_va: extent.start,
            size: extent.size,
            insts,
        });
    }
    Ok(out)
}
