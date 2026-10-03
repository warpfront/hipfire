//! C2: ELF section, symbol, note and envelope read/write.
//!
//! An `Envelope` is every byte of an AMDGPU code object except what its kernels own:
//! each kernel's instruction range, its 64-byte descriptor and its metadata map. Those are
//! holes. `read` returns them as `KernelImage`s (code bytes, `KernelDescriptor`,
//! `HsaKernelMetadata`) and `write` takes them back as `KernelParts`, re-serialising the
//! descriptor and metadata and splicing the supplied code, so module emission is the
//! retained envelope plus re-encoded kernel parts and never a copy of the input file.
//!
//! Everything else is typed and re-serialised: the ELF header, program headers, section
//! headers, symbol tables, `.dynamic` entries and notes. Section payloads the envelope does
//! not interpret (string tables, hashes, `.comment`, `s_code_end` padding in `.text`, …)
//! and inter-section padding (`Gap`) are retained as bytes. `read` rejects any input whose
//! typed parts would not re-serialise to the original bytes.

use std::borrow::Cow;
use std::collections::HashMap;

use object::elf::{self as E, Dyn64, FileHeader64, Ident, NoteHeader32, ProgramHeader64, SectionHeader64, Sym64};
use object::endian::{LittleEndian as LE, U16, U32, U64};
use object::pod;
use object::read::elf::{FileHeader as _, NoteIterator};
use peacemaker_ir::descriptor::{DescriptorError, KernelDescriptor};
use peacemaker_ir::envelope::{
    Dyn, Envelope, Fill, FileHeader, Gap, KernelSlot, MetadataDoc, Note, NoteDesc, ProgramHeader, Section, SectionData,
    SectionHeader, Symbol,
};
use peacemaker_ir::metadata::HsaKernelMetadata;

use crate::kd::{DescriptorCodec, KD_SIZE};
use crate::layout::{self, Layout};
use crate::metadata::{MetadataDocCodec, MetadataError};

/// `NT_AMDGPU_METADATA`: the code object V3+ msgpack metadata note.
pub const NT_AMDGPU_METADATA: u32 = 32;
/// Owner name of AMDGPU notes, including its terminating NUL (`n_namesz` = 7).
pub const AMDGPU_NOTE_NAME: &[u8] = b"AMDGPU\0";
const EHDR_SIZE: u64 = std::mem::size_of::<FileHeader64<LE>>() as u64;
const PHDR_SIZE: u64 = std::mem::size_of::<ProgramHeader64<LE>>() as u64;
const SHDR_SIZE: u64 = std::mem::size_of::<SectionHeader64<LE>>() as u64;

#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error("not a supported little-endian ELF64 AMDGPU code object: {0}")]
    Format(String),
    #[error("{what} [{start:#x}, {end:#x}) lies outside the {len}-byte file")]
    Truncated { what: String, start: u64, end: u64, len: u64 },
    #[error("{first} and {second} overlap at file offset {offset:#x}")]
    Overlap { first: String, second: String, offset: u64 },
    #[error("section {index} ({name}): {reason}")]
    Section { index: usize, name: String, reason: String },
    #[error("metadata note: {0}")]
    Metadata(#[from] MetadataError),
    #[error("kernel {kernel}: {reason}")]
    Kernel { kernel: String, reason: String },
    #[error("descriptor of kernel {kernel}: {error}")]
    Descriptor { kernel: String, #[source] error: DescriptorError },
    #[error("write needs one KernelParts per kernel slot: {expected} slots, {got} parts")]
    PartCount { expected: usize, got: usize },
    #[error("re-layout: {0}")]
    Layout(String),
}

fn has_file_bytes(h: &SectionHeader) -> bool { h.sh_type != E::SHT_NOBITS && h.sh_size != 0 }

fn contains_va(h: &SectionHeader, va: u64, len: u64) -> bool {
    h.sh_flags & u64::from(E::SHF_ALLOC) != 0
        && h.sh_type != E::SHT_NOBITS
        && va >= h.sh_addr
        && va.checked_add(len).is_some_and(|end| end <= h.sh_addr + h.sh_size)
}

/// What `read` hands the lifter for one kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelImage {
    pub name: String,
    pub entry_va: u64,
    pub code: Vec<u8>,
    pub descriptor: KernelDescriptor,
    pub metadata: HsaKernelMetadata,
}

impl KernelImage {
    pub fn parts(&self) -> KernelParts<'_> {
        KernelParts { code: &self.code, descriptor: &self.descriptor, metadata: &self.metadata }
    }
}

/// What `write` needs for one kernel slot.
#[derive(Clone, Copy, Debug)]
pub struct KernelParts<'a> {
    pub code: &'a [u8],
    pub descriptor: &'a KernelDescriptor,
    pub metadata: &'a HsaKernelMetadata,
}

/// Envelope read/write (`Envelope::read(..)` with this trait in scope).
pub trait EnvelopeCodec: Sized {
    /// Parses a code object into its envelope and kernel images.
    fn read(bytes: &[u8]) -> Result<(Self, Vec<KernelImage>), ElfError>;
    /// Emits the module: the envelope with each kernel slot filled from `parts` (same order
    /// as `kernels`): code spliced, descriptor and metadata map re-serialised. Parts that
    /// no longer fit the input layout (code or metadata note size, kernel renames) are
    /// written through [`EnvelopeCodec::layout`].
    fn write(&self, parts: &[KernelParts<'_>]) -> Result<Vec<u8>, ElfError>;
    /// The layout `write` emits `parts` into: the envelope itself when they fit, else the
    /// ld.lld re-layout (`crate::layout`) with each kernel's layout-derived descriptor.
    fn layout(&self, parts: &[KernelParts<'_>]) -> Result<Layout, ElfError>;
    /// Section name from the section header string table.
    fn section_name(&self, index: usize) -> Option<&str>;
    /// Symbol name from the string table linked by symbol table section `table`.
    fn symbol_name(&self, table: usize, symbol: &Symbol) -> Option<&str>;
    /// File offset of `[va, va + len)` inside one allocated section with file bytes.
    fn file_offset(&self, va: u64, len: u64) -> Option<usize>;
}

impl EnvelopeCodec for Envelope {
    fn read(bytes: &[u8]) -> Result<(Envelope, Vec<KernelImage>), ElfError> {
        let format = |e: object::read::Error| ElfError::Format(e.to_string());
        let fh = FileHeader64::<LE>::parse(bytes).map_err(format)?;
        let ident = fh.e_ident;
        if ident.class != E::ELFCLASS64 || ident.data != E::ELFDATA2LSB {
            return Err(ElfError::Format("not ELFCLASS64 / ELFDATA2LSB".into()));
        }
        let header = FileHeader {
            ident_version: ident.version,
            os_abi: ident.os_abi,
            abi_version: ident.abi_version,
            ident_padding: ident.padding,
            e_type: fh.e_type.get(LE),
            e_machine: fh.e_machine.get(LE),
            e_version: fh.e_version.get(LE),
            e_entry: fh.e_entry.get(LE),
            e_phoff: fh.e_phoff.get(LE),
            e_shoff: fh.e_shoff.get(LE),
            e_flags: fh.e_flags.get(LE),
            e_ehsize: fh.e_ehsize.get(LE),
            e_phentsize: fh.e_phentsize.get(LE),
            e_shentsize: fh.e_shentsize.get(LE),
            e_shstrndx: fh.e_shstrndx.get(LE),
        };
        if header.e_machine != E::EM_AMDGPU {
            return Err(ElfError::Format(format!("e_machine {} is not EM_AMDGPU", header.e_machine)));
        }
        if u64::from(header.e_ehsize) != EHDR_SIZE {
            return Err(ElfError::Format(format!("e_ehsize {} is not {EHDR_SIZE}", header.e_ehsize)));
        }
        let (phnum, shnum) = (fh.e_phnum.get(LE), fh.e_shnum.get(LE));
        if phnum == E::PN_XNUM || (shnum == 0 && header.e_shoff != 0) || header.e_shstrndx == E::SHN_XINDEX {
            return Err(ElfError::Format("extended program/section header numbering".into()));
        }
        let phdrs = fh.program_headers(LE, bytes).map_err(format)?;
        let shdrs = fh.section_headers(LE, bytes).map_err(format)?;
        if phdrs.len() != usize::from(phnum) || shdrs.len() != usize::from(shnum) {
            return Err(ElfError::Format("header table offset is 0 while its count is not".into()));
        }
        let segments = phdrs.iter().map(|p| ProgramHeader {
            p_type: p.p_type.get(LE),
            p_flags: p.p_flags.get(LE),
            p_offset: p.p_offset.get(LE),
            p_vaddr: p.p_vaddr.get(LE),
            p_paddr: p.p_paddr.get(LE),
            p_filesz: p.p_filesz.get(LE),
            p_memsz: p.p_memsz.get(LE),
            p_align: p.p_align.get(LE),
        }).collect::<Vec<_>>();
        let headers = shdrs.iter().map(|s| SectionHeader {
            sh_name: s.sh_name.get(LE),
            sh_type: s.sh_type.get(LE),
            sh_flags: s.sh_flags.get(LE),
            sh_addr: s.sh_addr.get(LE),
            sh_offset: s.sh_offset.get(LE),
            sh_size: s.sh_size.get(LE),
            sh_link: s.sh_link.get(LE),
            sh_info: s.sh_info.get(LE),
            sh_addralign: s.sh_addralign.get(LE),
            sh_entsize: s.sh_entsize.get(LE),
        }).collect::<Vec<_>>();

        let file_len = bytes.len() as u64;
        let mut regions = vec![Region::new("ELF header", 0, EHDR_SIZE)];
        if !segments.is_empty() {
            regions.push(Region::new("program headers", header.e_phoff, segments.len() as u64 * PHDR_SIZE));
        }
        if !headers.is_empty() {
            regions.push(Region::new("section headers", header.e_shoff, headers.len() as u64 * SHDR_SIZE));
        }
        let shstrtab = headers.get(usize::from(header.e_shstrndx)).filter(|h| has_file_bytes(h))
            .and_then(|h| bytes.get(h.sh_offset as usize..(h.sh_offset + h.sh_size) as usize));
        let name_of = |h: &SectionHeader| shstrtab.and_then(|t| cstr(t, h.sh_name)).unwrap_or("?").to_owned();
        let mut sections = Vec::with_capacity(headers.len());
        let mut metadata = None;
        for (index, h) in headers.iter().enumerate() {
            let data = if h.sh_type == E::SHT_NOBITS {
                SectionData::NoBits
            } else {
                let end = h.sh_offset.checked_add(h.sh_size).filter(|&end| end <= file_len)
                    .ok_or_else(|| ElfError::Truncated { what: format!("section {index} ({})", name_of(h)), start: h.sh_offset, end: h.sh_offset.saturating_add(h.sh_size), len: file_len })?;
                let raw = &bytes[h.sh_offset as usize..end as usize];
                if h.sh_size != 0 {
                    regions.push(Region::new(format!("section {index} ({})", name_of(h)), h.sh_offset, h.sh_size));
                }
                let malformed = |reason: String| ElfError::Section { index, name: name_of(h), reason };
                match h.sh_type {
                    E::SHT_SYMTAB | E::SHT_DYNSYM => {
                        if h.sh_entsize != std::mem::size_of::<Sym64<LE>>() as u64 {
                            return Err(malformed(format!("symbol entry size {}", h.sh_entsize)));
                        }
                        let syms = pod::slice_from_all_bytes::<Sym64<LE>>(raw).map_err(|()| malformed("size is not a whole number of symbols".into()))?;
                        SectionData::Symbols(syms.iter().map(|s| Symbol {
                            st_name: s.st_name.get(LE),
                            st_info: s.st_info,
                            st_other: s.st_other,
                            st_shndx: s.st_shndx.get(LE),
                            st_value: s.st_value.get(LE),
                            st_size: s.st_size.get(LE),
                        }).collect())
                    }
                    E::SHT_DYNAMIC => {
                        if h.sh_entsize != std::mem::size_of::<Dyn64<LE>>() as u64 {
                            return Err(malformed(format!("dynamic entry size {}", h.sh_entsize)));
                        }
                        let dyns = pod::slice_from_all_bytes::<Dyn64<LE>>(raw).map_err(|()| malformed("size is not a whole number of entries".into()))?;
                        SectionData::Dynamic(dyns.iter().map(|d| Dyn { d_tag: d.d_tag.get(LE), d_val: d.d_val.get(LE) }).collect())
                    }
                    E::SHT_NOTE => {
                        let mut notes = Vec::new();
                        let mut iter = NoteIterator::<FileHeader64<LE>>::new(LE, h.sh_addralign, raw).map_err(|e| malformed(e.to_string()))?;
                        while let Some(note) = iter.next().map_err(|e| malformed(e.to_string()))? {
                            notes.push(Note { name: note.name_bytes().to_vec(), n_type: note.n_type(LE), desc: NoteDesc::Bytes(note.desc().to_vec()) });
                        }
                        if write_notes(&notes, h.sh_addralign, &[])? != raw {
                            return Err(malformed("note padding is not zero or the last note is unpadded".into()));
                        }
                        for note in &mut notes {
                            if note.name != AMDGPU_NOTE_NAME || note.n_type != NT_AMDGPU_METADATA {
                                continue;
                            }
                            if metadata.is_some() {
                                return Err(malformed("more than one NT_AMDGPU_METADATA note".into()));
                            }
                            let NoteDesc::Bytes(desc) = &note.desc else { unreachable!("notes are parsed as bytes") };
                            let (doc, kernels) = MetadataDoc::parse(desc)?;
                            note.desc = NoteDesc::AmdgpuMetadata(doc);
                            metadata = Some(kernels);
                        }
                        SectionData::Notes(notes)
                    }
                    _ => SectionData::Bytes(raw.to_vec()),
                }
            };
            sections.push(Section { header: *h, data });
        }
        let gaps = find_gaps(bytes, regions).map_err(ElfError::from)?;
        let mut envelope = Envelope { header, segments, sections, gaps, kernels: Vec::new() };
        let images = cut_kernels(&mut envelope, metadata)?;
        Ok((envelope, images))
    }

    fn write(&self, parts: &[KernelParts<'_>]) -> Result<Vec<u8>, ElfError> {
        let metadata = check_parts(self, parts)?;
        if layout::fits(self, parts, &metadata)? {
            return write_fixed(self, parts, &metadata);
        }
        let laid = layout::relayout(self, parts, &metadata)?;
        let parts: Vec<KernelParts<'_>> = parts.iter().zip(&laid.descriptors)
            .map(|(part, descriptor)| KernelParts { descriptor, ..*part }).collect();
        write_fixed(&laid.envelope, &parts, &metadata)
    }

    fn layout(&self, parts: &[KernelParts<'_>]) -> Result<Layout, ElfError> {
        let metadata = check_parts(self, parts)?;
        if layout::fits(self, parts, &metadata)? {
            return Ok(Layout { envelope: self.clone(), descriptors: parts.iter().map(|p| p.descriptor.clone()).collect() });
        }
        layout::relayout(self, parts, &metadata)
    }

    fn section_name(&self, index: usize) -> Option<&str> {
        let table = self.sections.get(usize::from(self.header.e_shstrndx))?;
        let SectionData::Bytes(strings) = &table.data else { return None };
        cstr(strings, self.sections.get(index)?.header.sh_name)
    }

    fn symbol_name(&self, table: usize, symbol: &Symbol) -> Option<&str> {
        let link = self.sections.get(table)?.header.sh_link as usize;
        let SectionData::Bytes(strings) = &self.sections.get(link)?.data else { return None };
        cstr(strings, symbol.st_name)
    }

    fn file_offset(&self, va: u64, len: u64) -> Option<usize> {
        section_at(self, va, len).map(|i| {
            let h = &self.sections[i].header;
            (h.sh_offset + (va - h.sh_addr)) as usize
        })
    }

}

/// Checks `parts` against the slots (count, `.symbol`, descriptor entry offset under the
/// current layout) and returns the metadata maps in note order.
fn check_parts<'a>(env: &Envelope, parts: &[KernelParts<'a>]) -> Result<Vec<&'a HsaKernelMetadata>, ElfError> {
    if parts.len() != env.kernels.len() {
        return Err(ElfError::PartCount { expected: env.kernels.len(), got: parts.len() });
    }
    let mut metadata: Vec<Option<&HsaKernelMetadata>> = vec![None; env.kernels.len()];
    for (slot, part) in env.kernels.iter().zip(parts) {
        let kernel_error = |reason: String| ElfError::Kernel { kernel: slot.name.clone(), reason };
        if part.metadata.parsed.symbol != format!("{}.kd", slot.name) {
            return Err(kernel_error(format!("metadata .symbol {} does not name this kernel's descriptor", part.metadata.parsed.symbol)));
        }
        if slot.kd_va.checked_add_signed(part.descriptor.kernel_code_entry_byte_offset) != Some(slot.entry_va) {
            return Err(kernel_error("descriptor entry offset does not reach the kernel entry".into()));
        }
        *metadata.get_mut(slot.metadata_index).ok_or_else(|| kernel_error("metadata index out of range".into()))? = Some(part.metadata);
    }
    metadata.into_iter().collect::<Option<Vec<_>>>()
        .ok_or_else(|| ElfError::Format("kernel slots do not cover the metadata array".into()))
}

/// Writes `parts` into `env`'s own layout; every part must fit its slot exactly.
fn write_fixed(env: &Envelope, parts: &[KernelParts<'_>], metadata: &[&HsaKernelMetadata]) -> Result<Vec<u8>, ElfError> {
    for (slot, part) in env.kernels.iter().zip(parts) {
        if part.code.len() as u64 != slot.size {
            return Err(ElfError::Layout(format!("code of kernel {} is {} bytes, its slot {}", slot.name, part.code.len(), slot.size)));
        }
        if slot.kd_va.checked_add_signed(part.descriptor.kernel_code_entry_byte_offset) != Some(slot.entry_va) {
            return Err(ElfError::Kernel { kernel: slot.name.clone(), reason: "descriptor entry offset does not reach the kernel entry".into() });
        }
    }
    let mut chunks = vec![Chunk::bytes("ELF header", 0, header_bytes(env)?)];
    if !env.segments.is_empty() {
        let mut table = Vec::with_capacity(env.segments.len() * PHDR_SIZE as usize);
        for p in &env.segments {
            table.extend_from_slice(pod::bytes_of(&ProgramHeader64::<LE> {
                p_type: U32::new(LE, p.p_type),
                p_flags: U32::new(LE, p.p_flags),
                p_offset: U64::new(LE, p.p_offset),
                p_vaddr: U64::new(LE, p.p_vaddr),
                p_paddr: U64::new(LE, p.p_paddr),
                p_filesz: U64::new(LE, p.p_filesz),
                p_memsz: U64::new(LE, p.p_memsz),
                p_align: U64::new(LE, p.p_align),
            }));
        }
        chunks.push(Chunk::bytes("program headers", env.header.e_phoff, table));
    }
    if !env.sections.is_empty() {
        let mut table = Vec::with_capacity(env.sections.len() * SHDR_SIZE as usize);
        for s in &env.sections {
            let h = &s.header;
            table.extend_from_slice(pod::bytes_of(&SectionHeader64::<LE> {
                sh_name: U32::new(LE, h.sh_name),
                sh_type: U32::new(LE, h.sh_type),
                sh_flags: U64::new(LE, h.sh_flags),
                sh_addr: U64::new(LE, h.sh_addr),
                sh_offset: U64::new(LE, h.sh_offset),
                sh_size: U64::new(LE, h.sh_size),
                sh_link: U32::new(LE, h.sh_link),
                sh_info: U32::new(LE, h.sh_info),
                sh_addralign: U64::new(LE, h.sh_addralign),
                sh_entsize: U64::new(LE, h.sh_entsize),
            }));
        }
        chunks.push(Chunk::bytes("section headers", env.header.e_shoff, table));
    }
    for (index, section) in env.sections.iter().enumerate() {
        let h = &section.header;
        let data: Cow<'_, [u8]> = match &section.data {
            SectionData::NoBits => continue,
            SectionData::Bytes(b) => Cow::Borrowed(b),
            SectionData::Symbols(syms) => Cow::Owned(syms.iter().flat_map(|s| pod::bytes_of(&Sym64::<LE> {
                st_name: U32::new(LE, s.st_name),
                st_info: s.st_info,
                st_other: s.st_other,
                st_shndx: U16::new(LE, s.st_shndx),
                st_value: U64::new(LE, s.st_value),
                st_size: U64::new(LE, s.st_size),
            }).to_vec()).collect()),
            SectionData::Dynamic(dyns) => Cow::Owned(dyns.iter().flat_map(|d| pod::bytes_of(&Dyn64::<LE> {
                d_tag: U64::new(LE, d.d_tag),
                d_val: U64::new(LE, d.d_val),
            }).to_vec()).collect()),
            SectionData::Notes(notes) => Cow::Owned(write_notes(notes, h.sh_addralign, metadata)?),
        };
        if data.len() as u64 != h.sh_size {
            let name = env.section_name(index).unwrap_or("?");
            return Err(ElfError::Layout(format!("section {index} ({name}) holds {} bytes, its header {}", data.len(), h.sh_size)));
        }
        if !data.is_empty() {
            chunks.push(Chunk { what: format!("section {index}"), offset: h.sh_offset, body: Body::Bytes(data) });
        }
    }
    chunks.extend(env.gaps.iter().map(Chunk::gap));
    let mut out = assemble(chunks).map_err(ElfError::from)?;
    for (slot, part) in env.kernels.iter().zip(parts) {
        let hole = |what: &str| ElfError::Kernel { kernel: slot.name.clone(), reason: format!("{what} hole no longer maps to an allocated section") };
        let code_at = env.file_offset(slot.entry_va, slot.size).ok_or_else(|| hole("code"))?;
        out[code_at..code_at + part.code.len()].copy_from_slice(part.code);
        let kd_at = env.file_offset(slot.kd_va, KD_SIZE as u64).ok_or_else(|| hole("descriptor"))?;
        out[kd_at..kd_at + KD_SIZE].copy_from_slice(&part.descriptor.to_bytes());
    }
    Ok(out)
}

fn section_at(env: &Envelope, va: u64, len: u64) -> Option<usize> {
    env.sections.iter().position(|s| contains_va(&s.header, va, len))
}

fn header_bytes(env: &Envelope) -> Result<Vec<u8>, ElfError> {
    let h = &env.header;
    let count = |n: usize, what: &str| u16::try_from(n).ok().filter(|&n| n != E::PN_XNUM)
        .ok_or_else(|| ElfError::Format(format!("{n} {what} need extended numbering")));
    let fh = FileHeader64::<LE> {
        e_ident: Ident {
            magic: E::ELFMAG,
            class: E::ELFCLASS64,
            data: E::ELFDATA2LSB,
            version: h.ident_version,
            os_abi: h.os_abi,
            abi_version: h.abi_version,
            padding: h.ident_padding,
        },
        e_type: U16::new(LE, h.e_type),
        e_machine: U16::new(LE, h.e_machine),
        e_version: U32::new(LE, h.e_version),
        e_entry: U64::new(LE, h.e_entry),
        e_phoff: U64::new(LE, h.e_phoff),
        e_shoff: U64::new(LE, h.e_shoff),
        e_flags: U32::new(LE, h.e_flags),
        e_ehsize: U16::new(LE, h.e_ehsize),
        e_phentsize: U16::new(LE, h.e_phentsize),
        e_phnum: U16::new(LE, count(env.segments.len(), "program headers")?),
        e_shentsize: U16::new(LE, h.e_shentsize),
        e_shnum: U16::new(LE, count(env.sections.len(), "sections")?),
        e_shstrndx: U16::new(LE, h.e_shstrndx),
    };
    Ok(pod::bytes_of(&fh).to_vec())
}

/// Finds kernels (`STT_FUNC` + `<name>.kd` `STT_OBJECT`, core.md §5.2), pairs them with
/// their metadata maps, and cuts code and descriptor bytes out of the envelope.
fn cut_kernels(env: &mut Envelope, metadata: Option<Vec<HsaKernelMetadata>>) -> Result<Vec<KernelImage>, ElfError> {
    let table = env.sections.iter().position(|s| s.header.sh_type == E::SHT_SYMTAB)
        .or_else(|| env.sections.iter().position(|s| s.header.sh_type == E::SHT_DYNSYM));
    let mut found = Vec::new();
    if let Some(table) = table {
        let SectionData::Symbols(symbols) = &env.sections[table].data else { unreachable!("symbol tables parse as symbols") };
        let named = symbols.iter().map(|s| {
            env.symbol_name(table, s).map(|n| (n, s)).ok_or_else(|| ElfError::Section {
                index: table,
                name: env.section_name(table).unwrap_or("?").into(),
                reason: format!("symbol name offset {} does not resolve to a UTF-8 string", s.st_name),
            })
        }).collect::<Result<Vec<_>, _>>()?;
        let mut descriptors: HashMap<&str, &Symbol> = HashMap::new();
        for &(name, s) in &named {
            if s.kind() == E::STT_OBJECT && name.ends_with(".kd") && descriptors.insert(name, s).is_some() {
                return Err(ElfError::Kernel { kernel: name.into(), reason: "descriptor symbol is defined twice".into() });
            }
        }
        for &(name, s) in &named {
            if s.kind() != E::STT_FUNC {
                continue;
            }
            if let Some(kd) = descriptors.remove(format!("{name}.kd").as_str()) {
                found.push((name.to_owned(), *s, *kd));
            } else if found.iter().any(|(n, _, _)| n == name) {
                return Err(ElfError::Kernel { kernel: name.into(), reason: "kernel symbol is defined twice".into() });
            }
        }
        if let Some(orphan) = descriptors.keys().next() {
            return Err(ElfError::Kernel { kernel: (*orphan).into(), reason: "descriptor symbol has no STT_FUNC kernel".into() });
        }
    }
    found.sort_by_key(|(_, f, _)| f.st_value);

    let mut metadata: Vec<Option<HsaKernelMetadata>> = metadata.unwrap_or_default().into_iter().map(Some).collect();
    let mut images = Vec::with_capacity(found.len());
    for (name, func, kd) in found {
        let fail = |reason: String| ElfError::Kernel { kernel: name.clone(), reason };
        if func.st_size == 0 {
            return Err(fail("kernel symbol size is 0".into()));
        }
        if kd.st_size != KD_SIZE as u64 {
            return Err(fail(format!("descriptor symbol size {} is not {KD_SIZE}", kd.st_size)));
        }
        let kd_section = section_at(env, kd.st_value, KD_SIZE as u64)
            .ok_or_else(|| fail(format!("descriptor at {:#x} is not inside an allocated section", kd.st_value)))?;
        let code_section = section_at(env, func.st_value, func.st_size)
            .filter(|&i| env.sections[i].header.sh_flags & u64::from(E::SHF_EXECINSTR) != 0)
            .ok_or_else(|| fail(format!("code [{:#x}, +{:#x}) is not inside an executable section", func.st_value, func.st_size)))?;
        if let Some(prev) = env.kernels.last() {
            if prev.entry_va + prev.size > func.st_value {
                return Err(fail(format!("code overlaps kernel {}", prev.name)));
            }
        }
        if env.kernels.iter().any(|k| k.kd_va < kd.st_value + KD_SIZE as u64 && kd.st_value < k.kd_va + KD_SIZE as u64) {
            return Err(fail("descriptor overlaps another kernel's descriptor".into()));
        }
        let kd_bytes = bytes_mut(env, kd_section, kd.st_value, KD_SIZE as u64).ok_or_else(|| fail("descriptor section holds no bytes".into()))?;
        let descriptor = KernelDescriptor::from_bytes(kd_bytes).map_err(|error| ElfError::Descriptor { kernel: name.clone(), error })?;
        if kd.st_value.checked_add_signed(descriptor.kernel_code_entry_byte_offset) != Some(func.st_value) {
            return Err(fail(format!("descriptor entry offset {} does not reach the symbol at {:#x}", descriptor.kernel_code_entry_byte_offset, func.st_value)));
        }
        if func.st_value % 256 != 0 {
            return Err(fail(format!("entry {:#x} is not 256-byte aligned", func.st_value)));
        }
        let symbol = format!("{name}.kd");
        let mut matches = metadata.iter().enumerate().filter(|(_, m)| m.as_ref().is_some_and(|m| m.parsed.symbol == symbol)).map(|(i, _)| i);
        let metadata_index = matches.next().ok_or_else(|| fail(format!("no metadata map has .symbol {symbol}")))?;
        if matches.next().is_some() {
            return Err(fail(format!("several metadata maps have .symbol {symbol}")));
        }
        let meta = metadata[metadata_index].take().expect("matched metadata is present");
        kd_bytes.fill(0);
        let code_bytes = bytes_mut(env, code_section, func.st_value, func.st_size).ok_or_else(|| fail("code section holds no bytes".into()))?;
        let code = code_bytes.to_vec();
        code_bytes.fill(0);
        env.kernels.push(KernelSlot { name: name.clone(), entry_va: func.st_value, size: func.st_size, kd_va: kd.st_value, metadata_index });
        images.push(KernelImage { name, entry_va: func.st_value, code, descriptor, metadata: meta });
    }
    if let Some(extra) = metadata.into_iter().flatten().next() {
        return Err(ElfError::Kernel { kernel: extra.parsed.name, reason: format!("metadata map for {} has no kernel symbol pair", extra.parsed.symbol) });
    }
    Ok(images)
}

fn bytes_mut(env: &mut Envelope, section: usize, va: u64, len: u64) -> Option<&mut [u8]> {
    let h = env.sections[section].header;
    let SectionData::Bytes(data) = &mut env.sections[section].data else { return None };
    let start = (va - h.sh_addr) as usize;
    data.get_mut(start..start + len as usize)
}

pub(crate) fn write_notes(notes: &[Note], align: u64, metadata: &[&HsaKernelMetadata]) -> Result<Vec<u8>, ElfError> {
    let align = if align <= 4 { 4 } else { align as usize };
    let pad = |out: &mut Vec<u8>| out.resize(out.len().next_multiple_of(align), 0);
    let mut out = Vec::new();
    for note in notes {
        let desc: Cow<'_, [u8]> = match &note.desc {
            NoteDesc::Bytes(b) => Cow::Borrowed(b),
            NoteDesc::AmdgpuMetadata(doc) => Cow::Owned(doc.write(metadata)?),
        };
        let size = |n: usize| u32::try_from(n).map_err(|_| ElfError::Format(format!("note field of {n} bytes")));
        out.extend_from_slice(pod::bytes_of(&NoteHeader32::<LE> {
            n_namesz: U32::new(LE, size(note.name.len())?),
            n_descsz: U32::new(LE, size(desc.len())?),
            n_type: U32::new(LE, note.n_type),
        }));
        out.extend_from_slice(&note.name);
        pad(&mut out);
        out.extend_from_slice(&desc);
        pad(&mut out);
    }
    Ok(out)
}

pub(crate) fn cstr(table: &[u8], offset: u32) -> Option<&str> {
    let tail = table.get(offset as usize..)?;
    std::str::from_utf8(&tail[..tail.iter().position(|&b| b == 0)?]).ok()
}

/// Two modelled byte ranges claim the same file offset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Overlap {
    pub first: String,
    pub second: String,
    pub offset: u64,
}

impl From<Overlap> for ElfError {
    fn from(o: Overlap) -> Self { ElfError::Overlap { first: o.first, second: o.second, offset: o.offset } }
}

pub(crate) struct Region {
    what: String,
    start: u64,
    end: u64,
}

impl Region {
    pub(crate) fn new(what: impl Into<String>, start: u64, len: u64) -> Self {
        Region { what: what.into(), start, end: start.saturating_add(len) }
    }
}

/// Every byte of `bytes` not covered by a non-empty region, as gaps. Regions must lie
/// inside `bytes` and must not overlap.
pub(crate) fn find_gaps(bytes: &[u8], mut regions: Vec<Region>) -> Result<Vec<Gap>, Overlap> {
    regions.retain(|r| r.end > r.start);
    regions.sort_by_key(|r| (r.start, r.end));
    let mut gaps = Vec::new();
    let mut cursor = 0u64;
    let gap = |from: u64, to: u64, gaps: &mut Vec<Gap>| {
        let run = &bytes[from as usize..to as usize];
        let fill = if run.iter().all(|&b| b == 0) { Fill::Zero(to - from) } else { Fill::Bytes(run.to_vec()) };
        gaps.push(Gap { offset: from, fill });
    };
    for (i, r) in regions.iter().enumerate() {
        if r.start < cursor {
            return Err(Overlap { first: regions[i - 1].what.clone(), second: r.what.clone(), offset: r.start });
        }
        if r.start > cursor {
            gap(cursor, r.start, &mut gaps);
        }
        cursor = r.end;
    }
    if cursor < bytes.len() as u64 {
        gap(cursor, bytes.len() as u64, &mut gaps);
    }
    Ok(gaps)
}

pub(crate) enum Body<'a> {
    Bytes(Cow<'a, [u8]>),
    Zero(u64),
}

pub(crate) struct Chunk<'a> {
    pub what: String,
    pub offset: u64,
    pub body: Body<'a>,
}

impl<'a> Chunk<'a> {
    pub(crate) fn bytes(what: impl Into<String>, offset: u64, bytes: impl Into<Cow<'a, [u8]>>) -> Self {
        Chunk { what: what.into(), offset, body: Body::Bytes(bytes.into()) }
    }
    pub(crate) fn gap(gap: &'a Gap) -> Self {
        let body = match &gap.fill { Fill::Zero(n) => Body::Zero(*n), Fill::Bytes(b) => Body::Bytes(Cow::Borrowed(b)) };
        Chunk { what: format!("gap at {:#x}", gap.offset), offset: gap.offset, body }
    }
    fn len(&self) -> u64 { match &self.body { Body::Bytes(b) => b.len() as u64, Body::Zero(n) => *n } }
}

/// Lays chunks out at their offsets; the image ends at the last chunk's end.
pub(crate) fn assemble(mut chunks: Vec<Chunk<'_>>) -> Result<Vec<u8>, Overlap> {
    chunks.retain(|c| c.len() != 0);
    chunks.sort_by_key(|c| c.offset);
    for pair in chunks.windows(2) {
        if pair[0].offset + pair[0].len() > pair[1].offset {
            return Err(Overlap { first: pair[0].what.clone(), second: pair[1].what.clone(), offset: pair[1].offset });
        }
    }
    let len = chunks.last().map_or(0, |c| c.offset + c.len()) as usize;
    let mut out = vec![0u8; len];
    for c in &chunks {
        if let Body::Bytes(b) = &c.body {
            out[c.offset as usize..c.offset as usize + b.len()].copy_from_slice(b);
        }
    }
    Ok(out)
}

/// Pinned inputs for C2's gates: the committed KT48 fixture (core.md §8, `provenance.json`)
/// and the committed F2 builder bundle.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::path::Path;

    pub const KT48_SELECTED: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201";

    fn load(relative: &str, len: usize) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
        assert_eq!(bytes.len(), len, "{} is not the pinned fixture", path.display());
        bytes
    }

    /// SHA-256 f876e2ce520ba8d97358b34c903a48dafc5cff2e4cc552d7ffc775361cbcb6bc.
    pub fn kt48_co() -> Vec<u8> { load("tests/fixtures/kt48/hipcc.co", 42_208) }

    /// The F2 builder bundle, SHA-256 1efb0b8b16b510407446dae8dc4ec137955de27d818b424daeb84383c8191fa7.
    pub fn f2_hxaco() -> Vec<u8> { load("../../kernels/gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco", 361_976) }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{f2_hxaco, kt48_co, KT48_SELECTED};
    use super::*;
    use crate::bundle::SourceCodec;
    use peacemaker_ir::envelope::Source;
    use peacemaker_ir::inst::Arch;

    fn parts(images: &[KernelImage]) -> Vec<KernelParts<'_>> { images.iter().map(KernelImage::parts).collect() }

    fn emit(envelope: &Envelope, images: &[KernelImage]) -> Vec<u8> { envelope.write(&parts(images)).unwrap() }

    /// Input → `Source` + kernel images → emitted bytes (bundle and plain-ELF paths alike).
    fn source_round_trip(input: &[u8]) -> (Vec<u8>, Source, Vec<KernelImage>) {
        let (source, images) = Source::read(input, Arch::Gfx1201).unwrap();
        (source.write(&parts(&images)).unwrap(), source, images)
    }

    fn section(envelope: &Envelope, name: &str) -> usize {
        (0..envelope.sections.len()).find(|&i| envelope.section_name(i) == Some(name)).unwrap()
    }

    #[test]
    fn kt48_module_identity() {
        let co = kt48_co();
        let (out, source, images) = source_round_trip(&co);
        assert_eq!(out, co);
        assert!(source.bundle.is_none());
        let envelope = source.elf;

        assert_eq!((envelope.segments.len(), envelope.sections.len(), images.len()), (9, 18, 6));
        let text = &envelope.sections[section(&envelope, ".text")].header;
        assert_eq!((text.sh_offset, text.sh_addr, text.sh_size), (0x2400, 0x3400, 30_208));
        let slot = envelope.kernels.iter().find(|k| k.name == KT48_SELECTED).unwrap();
        assert_eq!((slot.entry_va, slot.size, slot.kd_va), (0x7f00, 10_604, 0x2240));
        // Every byte outside the modelled regions is zero padding.
        assert!(envelope.gaps.iter().all(|g| matches!(g.fill, Fill::Zero(_))), "{:?}", envelope.gaps);
    }

    #[test]
    fn f2_bundle_module_identity() {
        let hxaco = f2_hxaco();
        let (out, source, images) = source_round_trip(&hxaco);
        assert_eq!(out, hxaco);
        assert!(source.bundle.is_some());
        assert_eq!(images.len(), 12);
        assert!(source.elf.gaps.iter().all(|g| matches!(g.fill, Fill::Zero(_))), "{:?}", source.elf.gaps);
    }

    /// The envelope holds no kernel-owned bytes: each kernel part lands at its own place and
    /// descriptor/metadata are re-serialised from their typed fields, not copied.
    #[test]
    fn kernel_parts_are_filled_holes() {
        let co = kt48_co();
        let (envelope, mut images) = Envelope::read(&co).unwrap();
        let image = images.iter_mut().find(|k| k.name == KT48_SELECTED).unwrap();
        image.code[100] ^= 0x01;
        image.descriptor.kernarg_size = 332;
        image.metadata.parsed.vgpr_count = 240;
        let out = emit(&envelope, &images);

        let diffs: Vec<usize> = (0..co.len()).filter(|&i| out[i] != co[i]).collect();
        let note = &envelope.sections[section(&envelope, ".note")].header;
        assert_eq!(diffs.len(), 3, "{diffs:x?}");
        assert!((note.sh_offset..note.sh_offset + note.sh_size).contains(&(diffs[0] as u64)), ".vgpr_count in the metadata note");
        assert_eq!(diffs[1], 0x2240 + 8, "kernarg_size LSB in the descriptor");
        assert_eq!(diffs[2], 0x2400 + (0x7f00 - 0x3400) + 100, "code byte 100 of the selected kernel");

        let (_, again) = Envelope::read(&out).unwrap();
        let edited = again.iter().find(|k| k.name == KT48_SELECTED).unwrap();
        assert_eq!((edited.descriptor.kernarg_size, edited.metadata.parsed.vgpr_count), (332, 240));
    }

    #[test]
    fn write_rejects_parts_it_cannot_place() {
        let (envelope, images) = Envelope::read(&kt48_co()).unwrap();
        let parts: Vec<_> = images.iter().map(KernelImage::parts).collect();
        assert!(matches!(envelope.write(&parts[1..]), Err(ElfError::PartCount { expected: 6, got: 5 })));

        let mut short = parts.clone();
        short[0].code = &images[0].code[1..];
        assert!(matches!(envelope.write(&short), Err(ElfError::Layout(reason)) if reason.contains("dword")));

        let mut swapped = parts.clone();
        swapped.swap(0, 1);
        assert!(matches!(envelope.write(&swapped), Err(ElfError::Kernel { .. })));
    }

    /// The ld.lld rules applied to an unedited module's own sizes re-derive every layout
    /// byte of it: KT48 (hipcc: `.eh_frame`, `.bss`, `s_code_end` tail) and the F2 builder
    /// bundle (no `.eh_frame`, no tail).
    #[test]
    fn relayout_of_an_unedited_module_reproduces_it() {
        for (input, bundled) in [(kt48_co(), false), (f2_hxaco(), true)] {
            let (source, images) = Source::read(&input, Arch::Gfx1201).unwrap();
            let env = &source.elf;
            let parts = parts(&images);
            let metadata = check_parts(env, &parts).unwrap();
            let laid = layout::relayout(env, &parts, &metadata).unwrap_or_else(|e| panic!("bundled={bundled}: {e}"));
            assert_eq!(laid.envelope.header, env.header, "bundled={bundled}");
            assert_eq!(laid.envelope.segments, env.segments, "bundled={bundled}");
            for (i, (a, b)) in laid.envelope.sections.iter().zip(&env.sections).enumerate() {
                assert_eq!(a.header, b.header, "bundled={bundled} section {i} header");
                assert!(a.data == b.data, "bundled={bundled} section {i} data");
            }
            assert_eq!(laid.envelope.kernels, env.kernels, "bundled={bundled}");
            let descriptors: Vec<_> = images.iter().map(|k| k.descriptor.clone()).collect();
            assert_eq!(laid.descriptors, descriptors);
            let laid_parts: Vec<_> = parts.iter().zip(&laid.descriptors).map(|(p, d)| KernelParts { descriptor: d, ..*p }).collect();
            let elf = write_fixed(&laid.envelope, &laid_parts, &metadata).unwrap();
            let device: Vec<u8> = if bundled {
                let (_, payloads) = <peacemaker_ir::envelope::Bundle as crate::bundle::BundleCodec>::read(&input).unwrap();
                payloads.into_iter().find(|p| !p.is_empty()).unwrap().to_vec()
            } else { input.clone() };
            assert_eq!(elf, device, "bundled={bundled}");
        }
    }

    /// A grown metadata note moves every later allocated section: the module is re-laid
    /// out, re-reads, and carries the same kernels with the grown map.
    #[test]
    fn grown_metadata_note_relays_the_module() {
        let co = kt48_co();
        let (envelope, images) = Envelope::read(&co).unwrap();
        let mut grown = images[0].metadata.clone();
        grown.parsed.args.push(peacemaker_ir::metadata::Kernarg { name: "extra".into(), size: 8, offset: 328, value_kind: "by_value".into(), address_space: None });
        let mut parts = parts(&images);
        parts[0].metadata = &grown;
        let out = envelope.write(&parts).unwrap();
        let (again, relifted) = Envelope::read(&out).unwrap();
        assert_eq!(relifted.len(), images.len());
        for (a, b) in relifted.iter().zip(&images) {
            assert_eq!((&a.name, &a.code), (&b.name, &b.code));
            assert_eq!(a.metadata.parsed, if a.name == images[0].name { grown.parsed.clone() } else { b.metadata.parsed.clone() });
            let mut expected = b.descriptor.clone();
            expected.kernel_code_entry_byte_offset = a.descriptor.kernel_code_entry_byte_offset;
            assert_eq!(a.descriptor, expected, "only the entry offset is layout-derived here");
        }
        let note = |env: &Envelope| env.sections[section(env, ".note")].header.sh_size;
        assert!(note(&again) > note(&envelope));
        let header = |env: &Envelope, name: &str| env.sections[section(env, name)].header;
        assert!(header(&again, ".dynsym").sh_addr > header(&envelope, ".dynsym").sh_addr, "sections after the note move");
        assert_eq!(header(&again, ".text").sh_size, header(&envelope, ".text").sh_size);
        assert_eq!(header(&again, ".text").sh_addr % 256, 0);
    }

    #[test]
    fn read_rejects_what_it_cannot_reproduce() {
        let co = kt48_co();
        let mut entry = co.clone();
        entry[0x2240 + 16] ^= 0x40;
        assert!(matches!(Envelope::read(&entry), Err(ElfError::Kernel { reason, .. }) if reason.contains("entry offset")));

        let mut reserved = co.clone();
        reserved[0x2240 + 30] = 1;
        assert!(matches!(Envelope::read(&reserved), Err(ElfError::Descriptor { error: DescriptorError::Reserved { offset: 30 }, .. })));

        // Padding after the 7-byte "AMDGPU\0" note name.
        let mut padding = co.clone();
        padding[0x238 + 12 + 7] = 1;
        assert!(matches!(Envelope::read(&padding), Err(ElfError::Section { reason, .. }) if reason.contains("padding")));
    }
}
