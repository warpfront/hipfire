// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Code-object loading: a linked AMDGPU ELF64 little-endian object (HSA code
//! object v5/v6; the shared objects `ld.lld -shared` emits, or an
//! `ET_EXEC`) → [`KernelImage`] for one kernel symbol (`name` and
//! `name.kd`), checking the ELF machine (`EM_AMDGPU`, `e_flags` MACH of the
//! [`crate::Arch`] variant that selects the image's architecture).
//!
//! Layout used (System V gABI ELF64; LLVM `AMDGPUUsage` "Code Object V5/V6
//! Metadata" and "Kernel Descriptor"): the kernel code is the `STT_FUNC`
//! symbol `symbol` in an executable section, the descriptor is the 64-byte
//! `STT_OBJECT` symbol `symbol.kd` in a data section, and the descriptor's
//! `kernel_code_entry_byte_offset` (signed, relative to the descriptor's own
//! address) must land exactly on the function symbol. Symbols are looked up
//! in `.symtab`, then `.dynsym`. An unlinked relocatable (`ET_REL`) is
//! rejected: its entry offset is a relocation not yet applied.
use super::{fault, unsupported, KernelImage, SimResult};
use crate::kd::{KernelDescriptor, KD_BYTES};
use crate::Arch;

const EM_AMDGPU: u16 = 224;
const ELFOSABI_AMDGPU_HSA: u8 = 64;
const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const ET_REL: u16 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_NOBITS: u32 = 8;
const SHT_DYNSYM: u32 = 11;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const SHN_LORESERVE: u16 = 0xff00;
const EHDR_BYTES: usize = 64;
const SHDR_BYTES: usize = 64;
const SYM_BYTES: usize = 24;

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// `elf[off..off + len]`, or a fault naming `what` when out of the file.
fn slice<'a>(elf: &'a [u8], off: u64, len: u64, what: &str) -> SimResult<&'a [u8]> {
    off.checked_add(len)
        .filter(|&end| end <= elf.len() as u64)
        .map(|end| &elf[off as usize..end as usize])
        .map_or_else(
            || {
                fault(format!(
                    "code object truncated: {what} at {off:#x}+{len:#x} exceeds the {} byte file",
                    elf.len()
                ))
            },
            Ok,
        )
}

#[derive(Clone, Copy)]
struct Section {
    ty: u32,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
}

struct Sym {
    addr: u64,
    size: u64,
    shndx: u16,
}

fn sections(elf: &[u8]) -> SimResult<Vec<Section>> {
    let shoff = u64_at(elf, 40);
    let shentsize = usize::from(u16_at(elf, 58));
    let shnum = usize::from(u16_at(elf, 60));
    if shoff == 0 || shnum == 0 {
        return fault("code object has no section headers");
    }
    if shentsize < SHDR_BYTES {
        return fault(format!(
            "section header entry size {shentsize} is below {SHDR_BYTES}"
        ));
    }
    let table = slice(
        elf,
        shoff,
        (shentsize * shnum) as u64,
        "section header table",
    )?;
    Ok((0..shnum)
        .map(|i| {
            let h = &table[i * shentsize..i * shentsize + SHDR_BYTES];
            Section {
                ty: u32_at(h, 4),
                addr: u64_at(h, 16),
                offset: u64_at(h, 24),
                size: u64_at(h, 32),
                link: u32_at(h, 40),
            }
        })
        .collect())
}

/// The `want`-typed symbol named `name` in `.symtab`, falling back to `.dynsym`.
fn find_symbol(elf: &[u8], secs: &[Section], name: &str, want: u8) -> SimResult<Option<Sym>> {
    for table_ty in [SHT_SYMTAB, SHT_DYNSYM] {
        for table in secs.iter().filter(|s| s.ty == table_ty) {
            let strtab = secs.get(table.link as usize).ok_or_else(|| {
                super::SimError::Fault(format!(
                    "symbol table links to missing string table section {}",
                    table.link
                ))
            })?;
            let strings = slice(elf, strtab.offset, strtab.size, "string table")?;
            let syms = slice(elf, table.offset, table.size, "symbol table")?;
            for s in syms.chunks_exact(SYM_BYTES) {
                if s[4] & 0xf != want {
                    continue;
                }
                let start = u32_at(s, 0) as usize;
                let Some(tail) = strings.get(start..) else {
                    return fault(format!(
                        "symbol name offset {start:#x} is outside the string table"
                    ));
                };
                let end = tail.iter().position(|&c| c == 0).ok_or_else(|| {
                    super::SimError::Fault("unterminated symbol name in the string table".into())
                })?;
                if &tail[..end] == name.as_bytes() {
                    return Ok(Some(Sym {
                        addr: u64_at(s, 8),
                        size: u64_at(s, 16),
                        shndx: u16_at(s, 6),
                    }));
                }
            }
        }
    }
    Ok(None)
}

/// The file bytes of `sym` (its `size` bytes at virtual address `addr`),
/// taken from the section the symbol is defined in.
fn symbol_bytes<'a>(elf: &'a [u8], secs: &[Section], sym: &Sym, what: &str) -> SimResult<&'a [u8]> {
    if sym.shndx == 0 || sym.shndx >= SHN_LORESERVE {
        return fault(format!(
            "{what} is undefined or absolute (section index {:#x})",
            sym.shndx
        ));
    }
    let sec = secs.get(usize::from(sym.shndx)).ok_or_else(|| {
        super::SimError::Fault(format!(
            "{what} is defined in missing section {}",
            sym.shndx
        ))
    })?;
    if sec.ty == SHT_NOBITS {
        return fault(format!("{what} lives in a NOBITS section"));
    }
    let rel = sym
        .addr
        .checked_sub(sec.addr)
        .filter(|&r| r.checked_add(sym.size).is_some_and(|e| e <= sec.size));
    let Some(rel) = rel else {
        return fault(format!(
            "{what} at {:#x}+{:#x} lies outside its section {:#x}+{:#x}",
            sym.addr, sym.size, sec.addr, sec.size
        ));
    };
    slice(elf, sec.offset + rel, sym.size, what)
}

/// Load kernel `symbol` from `elf`: its code from the entry to the symbol's
/// end, and its 64-byte descriptor.
pub fn load_code_object(elf: &[u8], symbol: &str) -> SimResult<KernelImage> {
    if elf.len() < EHDR_BYTES {
        return fault(format!(
            "code object truncated: {} bytes is shorter than the ELF64 header",
            elf.len()
        ));
    }
    if &elf[0..4] != b"\x7fELF" {
        return fault("not an ELF file (bad magic)");
    }
    if elf[4] != 2 {
        return unsupported("ELF class is not ELF64");
    }
    if elf[5] != 1 {
        return unsupported("ELF data encoding is not little-endian");
    }
    if elf[7] != ELFOSABI_AMDGPU_HSA {
        return unsupported(format!(
            "ELF OS/ABI {} is not AMDGPU_HSA ({ELFOSABI_AMDGPU_HSA})",
            elf[7]
        ));
    }
    let e_type = u16_at(elf, 16);
    let machine = u16_at(elf, 18);
    if machine != EM_AMDGPU {
        return unsupported(format!(
            "ELF machine {machine} is not EM_AMDGPU ({EM_AMDGPU})"
        ));
    }
    let mach = u32_at(elf, 48) & 0xff;
    let Some(arch) = Arch::ALL
        .iter()
        .copied()
        .find(|a| a.caps().elf_mach == mach)
    else {
        return unsupported(format!(
            "EF_AMDGPU_MACH {mach:#x} matches no supported CDNA architecture"
        ));
    };
    match e_type {
        ET_DYN | ET_EXEC => {}
        ET_REL => return unsupported("relocatable object: link it (ld.lld -shared) so the descriptor's entry offset is resolved"),
        t => return unsupported(format!("ELF type {t} is neither ET_DYN nor ET_EXEC")),
    }

    let secs = sections(elf)?;
    let kd_name = format!("{symbol}.kd");
    let func = find_symbol(elf, &secs, symbol, STT_FUNC)?.ok_or_else(|| {
        super::SimError::Fault(format!("kernel function symbol `{symbol}` not found"))
    })?;
    let kd_sym = find_symbol(elf, &secs, &kd_name, STT_OBJECT)?.ok_or_else(|| {
        super::SimError::Fault(format!("kernel descriptor symbol `{kd_name}` not found"))
    })?;
    if kd_sym.size != KD_BYTES as u64 {
        return fault(format!(
            "`{kd_name}` is {} bytes, not {KD_BYTES}",
            kd_sym.size
        ));
    }
    if func.size == 0 {
        return fault(format!("kernel function `{symbol}` has size 0"));
    }
    if func.size % 4 != 0 {
        return fault(format!(
            "kernel function `{symbol}` size {} is not a whole number of dwords",
            func.size
        ));
    }

    let kd = KernelDescriptor::from_bytes(symbol_bytes(elf, &secs, &kd_sym, &kd_name)?)?;
    let entry = i128::from(kd_sym.addr) + i128::from(kd.kernel_code_entry_byte_offset);
    if entry != i128::from(func.addr) {
        return fault(format!("`{kd_name}` kernel_code_entry_byte_offset {} lands on {entry:#x}, not on `{symbol}` at {:#x}", kd.kernel_code_entry_byte_offset, func.addr));
    }
    let code = symbol_bytes(elf, &secs, &func, symbol)?;
    let words = code
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(KernelImage {
        arch,
        name: symbol.to_string(),
        words,
        kd,
    })
}
