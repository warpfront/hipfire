// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure-data ELF and HIP bundle envelope. Kernel code, descriptors, and metadata maps are
//! holes owned by `Program::kernels`; the lifter parses and re-emits this structure.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileHeader {
    pub ident_version: u8,
    pub os_abi: u8,
    pub abi_version: u8,
    pub ident_padding: [u8; 7],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_shentsize: u16,
    pub e_shstrndx: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramHeader {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionHeader {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section {
    pub header: SectionHeader,
    pub data: SectionData,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SectionData {
    NoBits,
    /// Uninterpreted section payload, with kernel code and descriptor holes zeroed.
    Bytes(Vec<u8>),
    Symbols(Vec<Symbol>),
    Dynamic(Vec<Dyn>),
    Notes(Vec<Note>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Symbol {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}
impl Symbol {
    pub fn kind(&self) -> u8 {
        self.st_info & 0xf
    }
    pub fn binding(&self) -> u8 {
        self.st_info >> 4
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dyn {
    pub d_tag: u64,
    pub d_val: u64,
}

/// Metadata document skeleton: each non-kernel key and value is its own MessagePack bytes,
/// rather than a copy of the input note or a dependence on the MessagePack codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetadataDoc {
    pub entries: Vec<DocEntry>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DocEntry {
    Kernels(usize),
    Raw { key: Vec<u8>, value: Vec<u8> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Note {
    pub name: Vec<u8>,
    pub n_type: u32,
    pub desc: NoteDesc,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NoteDesc {
    Bytes(Vec<u8>),
    AmdgpuMetadata(MetadataDoc),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gap {
    pub offset: u64,
    pub fill: Fill,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Fill {
    Zero(u64),
    Bytes(Vec<u8>),
}
impl Fill {
    pub fn len(&self) -> u64 {
        match self {
            Self::Zero(n) => *n,
            Self::Bytes(bytes) => bytes.len() as u64,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSlot {
    pub name: String,
    pub entry_va: u64,
    pub size: u64,
    pub kd_va: u64,
    pub metadata_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Envelope {
    pub header: FileHeader,
    pub segments: Vec<ProgramHeader>,
    pub sections: Vec<Section>,
    pub gaps: Vec<Gap>,
    /// Kernel holes in entry-address order; the writer fills these from `Program::kernels`.
    pub kernels: Vec<KernelSlot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleEntry {
    pub triple: String,
    pub offset: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bundle {
    pub entries: Vec<BundleEntry>,
    pub gaps: Vec<Gap>,
}

/// A lifted ELF may come from a HIP bundle; authored/raw programs have no source.
/// Nesting the bundle prevents representing a bundle without a device ELF.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Source {
    pub elf: Envelope,
    pub bundle: Option<Bundle>,
}
