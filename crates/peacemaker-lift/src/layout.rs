// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! M3: re-layout of ld.lld-linked gfx1201 code objects (code object v6).
//!
//! When a kernel's code, the metadata note or a kernel's symbol name changes size, the
//! envelope can no longer keep its input layout. [`relayout`] re-derives every
//! layout-dependent byte the way `ld.lld` laid out the input and returns a new
//! [`Envelope`] (plus the layout-derived descriptor fields) whose fixed-layout write is the
//! re-laid-out module. The rules:
//!
//! 1. Sections keep their index order, which is their file order; allocated sections come
//!    first. The first one starts at `align(ELF header + program headers, sh_addralign)`.
//! 2. A section that starts a `PT_LOAD` gets `addr = align(dot, page) + off % page`, then its
//!    own alignment, and `off = align(off, page, skew addr)` (lld's "no file padding" rule);
//!    later sections of the same `PT_LOAD` get `addr = align(dot, sh_addralign)` and
//!    `off = first.off + (addr - first.addr)`; `SHT_NOBITS` advances no file offset.
//! 3. `.relro_padding` pads its `PT_GNU_RELRO` to the next page: `align(dot, page) - dot`.
//! 4. Non-allocated sections follow at `align(off, sh_addralign)`; the section header table
//!    at `align(off, 8)`.
//! 5. `.text`: kernels in address order, each at `align(previous end, 256)`, padded with
//!    `s_nop 0`; after the last one either nothing (builder) or `s_code_end` up to the
//!    128-byte instruction cache line plus three more lines (LLVM's `EmitCodeEnd`, GFX11+).
//! 6. String tables are `"\0"` followed by their names in input order; a renamed kernel's
//!    two names are replaced in place.
//! 7. `.dynsym` is the null symbol, the undefined symbols, then the defined ones sorted by
//!    `(gnu_hash % nbuckets, .dynstr offset)`; `.gnu.hash` (lld: `nbuckets = max(n/4, 1)`,
//!    Bloom words `NextPowerOf2(12n/64)`, shift 26) and SysV `.hash` are regenerated.
//! 8. `.eh_frame` FDEs (pc-relative `sdata4` begin, `udata4` range) cover their kernel.
//! 9. `.dynamic` `DT_HASH`/`DT_GNU_HASH`/`DT_STRTAB`/`DT_SYMTAB` hold section addresses and
//!    `DT_STRSZ` the `.dynstr` size.
//! 10. Kernel symbols get their kernel's new entry/size and `.kd` address; every other
//!     symbol moves with its section. Program headers keep their first/last section and the
//!     head/tail distance to them.
//! 11. The descriptor's entry offset is `entry - kd`; `INST_PREF_SIZE` (rsrc3 11:4) becomes
//!     `min(ceil(size / 128), 255; 63 on gfx11, whose prefetch field is 6 bits)` when the input descriptor followed that rule (hipcc and
//!     the builder both do); a resized kernel whose input did not follow it is refused.
//!
//! Every rule is first applied to the input's own sizes and must reproduce the input
//! exactly (addresses, offsets, segments, `.text` padding, string and hash tables, FDEs);
//! an input these rules do not describe is refused, never approximated.

use std::collections::HashMap;

use object::elf as E;
use peacemaker_ir::descriptor::{KernelDescriptor, Rsrc3};
use peacemaker_ir::envelope::{Dyn, Envelope, Fill, SectionData, SectionHeader, Symbol};
use peacemaker_ir::metadata::HsaKernelMetadata;

use crate::elf::{write_notes, ElfError, EnvelopeCodec, KernelParts};
use crate::kd::KD_SIZE;

const PAGE: u64 = 0x1000;
const KERNEL_ALIGN: u64 = 256;
const S_NOP_0: u32 = 0xbf80_0000;
const S_CODE_END: u32 = 0xbf9f_0000;
/// GFX11+ instruction cache line (`EmitCodeEnd` and `INST_PREF_SIZE` granule).
const CACHE_LINE: u64 = 128;
const CODE_END_EXTRA: u64 = 3 * CACHE_LINE;
const INST_PREF_SHIFT: u32 = 4;
const INST_PREF_MAX: u32 = 0xff;
/// gfx11 rsrc3 prefetch is 6 bits [9:4] (bits [11:10] must be zero; AMDGPUUsage
/// compute_pgm_rsrc3-gfx10-gfx11), so hipcc caps it at 63.
const INST_PREF_MAX_GFX11: u32 = 0x3f;
const GNU_HASH_SHIFT2: u32 = 26;
const EF_AMDGPU_MACH_GFX1201: u32 = 0x4e;
/// gfx1100 / gfx1151 (hipcc and the builder link them with the same ld.lld rules; the
/// input-reproduction check below is the proof, not the machine number).
const EF_AMDGPU_MACH_GFX1100: u32 = 0x41;
const EF_AMDGPU_MACH_GFX1151: u32 = 0x4a;
/// `e_ident[EI_ABIVERSION]` of code object v6.
const ABI_VERSION_V6: u8 = 4;
const SHN_LORESERVE: u16 = 0xff00;

fn fail(reason: impl Into<String>) -> ElfError {
    ElfError::Layout(reason.into())
}

fn align(x: u64, a: u64) -> u64 {
    if a <= 1 {
        x
    } else {
        x.next_multiple_of(a)
    }
}

/// Smallest `y >= x` with `y % a == skew % a`.
fn align_skew(x: u64, a: u64, skew: u64) -> u64 {
    let skew = skew % a;
    let base = x - x % a + skew;
    if base >= x {
        base
    } else {
        base + a
    }
}

/// `INST_PREF_SIZE` for a kernel of `size` bytes, saturating at the field's `max`.
pub(crate) fn inst_pref_size(size: u64, max: u32) -> u32 {
    size.div_ceil(CACHE_LINE).min(u64::from(max)) as u32
}

fn inst_pref_field(rsrc3: Rsrc3, max: u32) -> u32 {
    (rsrc3.0 >> INST_PREF_SHIFT) & max
}

/// The re-laid-out module: an envelope whose fixed-layout write is the module, and each
/// kernel's descriptor with its layout-derived fields (entry offset, `INST_PREF_SIZE`).
#[derive(Clone, Debug)]
pub struct Layout {
    pub envelope: Envelope,
    pub descriptors: Vec<KernelDescriptor>,
}

/// Whether `parts` fit the envelope's current layout: same code sizes, same note size and
/// every kernel symbol still named after its slot.
pub(crate) fn fits(
    env: &Envelope,
    parts: &[KernelParts<'_>],
    metadata: &[&HsaKernelMetadata],
) -> Result<bool, ElfError> {
    if env
        .kernels
        .iter()
        .zip(parts)
        .any(|(slot, part)| part.code.len() as u64 != slot.size)
    {
        return Ok(false);
    }
    for section in &env.sections {
        if let SectionData::Notes(notes) = &section.data {
            if write_notes(notes, section.header.sh_addralign, metadata)?.len() as u64
                != section.header.sh_size
            {
                return Ok(false);
            }
        }
    }
    let symbols = KernelSymbols::find(env)?;
    Ok(env
        .kernels
        .iter()
        .enumerate()
        .all(|(k, slot)| symbols.old_names[k] == slot.name))
}

/// Which symbol (by table section and index) belongs to which kernel slot.
struct KernelSymbols {
    /// `(table, index)` → `(slot, is_descriptor)`.
    roles: HashMap<(usize, usize), (usize, bool)>,
    /// The code symbol's current name per slot.
    old_names: Vec<String>,
}

impl KernelSymbols {
    fn find(env: &Envelope) -> Result<Self, ElfError> {
        let mut roles = HashMap::new();
        let mut old_names: Vec<Option<String>> = vec![None; env.kernels.len()];
        for (t, section) in env.sections.iter().enumerate() {
            let SectionData::Symbols(symbols) = &section.data else {
                continue;
            };
            let mut seen = vec![[0usize; 2]; env.kernels.len()];
            for (i, s) in symbols.iter().enumerate() {
                let hit = env.kernels.iter().enumerate().find_map(|(k, slot)| {
                    if s.kind() == E::STT_FUNC && s.st_value == slot.entry_va {
                        Some((k, false))
                    } else if s.kind() == E::STT_OBJECT
                        && s.st_value == slot.kd_va
                        && s.st_size == KD_SIZE as u64
                    {
                        Some((k, true))
                    } else {
                        None
                    }
                });
                let Some((k, kd)) = hit else { continue };
                let name = env
                    .symbol_name(t, s)
                    .ok_or_else(|| fail(format!("symbol {i} of section {t} has no name")))?;
                let expected = old_names[k].clone();
                let stem = if kd {
                    name.strip_suffix(".kd")
                } else {
                    Some(name)
                };
                let Some(stem) = stem else {
                    return Err(fail(format!(
                        "descriptor symbol {name} lacks the .kd suffix"
                    )));
                };
                if expected.as_deref().is_some_and(|e| e != stem) {
                    return Err(fail(format!(
                        "kernel {k} symbols disagree on its name ({stem} vs {})",
                        expected.unwrap_or_default()
                    )));
                }
                old_names[k] = Some(stem.to_owned());
                seen[k][usize::from(kd)] += 1;
                roles.insert((t, i), (k, kd));
            }
            if let Some(k) = seen.iter().position(|c| *c != [1, 1]) {
                return Err(fail(format!(
                    "symbol table {t} does not name kernel {} exactly once with one .kd",
                    env.kernels[k].name
                )));
            }
        }
        let old_names = old_names
            .into_iter()
            .enumerate()
            .map(|(k, n)| {
                n.ok_or_else(|| fail(format!("kernel {} has no symbol", env.kernels[k].name)))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { roles, old_names })
    }
}

/// Section roles the rules need.
struct Roles {
    text: usize,
    kd: usize,
    dynsym: Option<usize>,
    gnu_hash: Option<usize>,
    hash: Option<usize>,
    dynamic: Option<usize>,
    eh_frame: Option<usize>,
    relro_padding: Option<usize>,
}

fn roles(env: &Envelope) -> Result<Roles, ElfError> {
    let section_of = |va: u64, len: u64| {
        env.sections.iter().position(|s| {
            let h = &s.header;
            h.sh_flags & u64::from(E::SHF_ALLOC) != 0
                && h.sh_type != E::SHT_NOBITS
                && va >= h.sh_addr
                && va + len <= h.sh_addr + h.sh_size
        })
    };
    let first = env.kernels.first().ok_or_else(|| fail("no kernels"))?;
    let text = section_of(first.entry_va, first.size)
        .ok_or_else(|| fail("kernel code is not in a section"))?;
    let kd = section_of(first.kd_va, KD_SIZE as u64)
        .ok_or_else(|| fail("descriptor is not in a section"))?;
    for slot in &env.kernels {
        if section_of(slot.entry_va, slot.size) != Some(text)
            || section_of(slot.kd_va, KD_SIZE as u64) != Some(kd)
        {
            return Err(fail("kernels span several code or descriptor sections"));
        }
    }
    let unique = |pred: &dyn Fn(usize, &SectionHeader) -> bool,
                  what: &str|
     -> Result<Option<usize>, ElfError> {
        let hits: Vec<usize> = env
            .sections
            .iter()
            .enumerate()
            .filter(|(i, s)| pred(*i, &s.header))
            .map(|(i, _)| i)
            .collect();
        match hits[..] {
            [] => Ok(None),
            [i] => Ok(Some(i)),
            _ => Err(fail(format!("several {what} sections"))),
        }
    };
    if env
        .sections
        .iter()
        .any(|s| matches!(s.header.sh_type, E::SHT_REL | E::SHT_RELA))
    {
        return Err(fail("relocation sections are not modelled"));
    }
    Ok(Roles {
        text,
        kd,
        dynsym: unique(&|_, h| h.sh_type == E::SHT_DYNSYM, ".dynsym")?,
        gnu_hash: unique(&|_, h| h.sh_type == E::SHT_GNU_HASH, ".gnu.hash")?,
        hash: unique(&|_, h| h.sh_type == E::SHT_HASH, ".hash")?,
        dynamic: unique(&|_, h| h.sh_type == E::SHT_DYNAMIC, ".dynamic")?,
        eh_frame: unique(
            &|i, _| env.section_name(i) == Some(".eh_frame"),
            ".eh_frame",
        )?,
        relro_padding: unique(
            &|i, h| h.sh_type == E::SHT_NOBITS && env.section_name(i) == Some(".relro_padding"),
            ".relro_padding",
        )?,
    })
}

// ---------------------------------------------------------------------------
// .text
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tail {
    None,
    CodeEnd,
}

fn words_are(bytes: &[u8], word: u32) -> bool {
    bytes.len() % 4 == 0
        && bytes
            .chunks_exact(4)
            .all(|w| u32::from_le_bytes(w.try_into().expect("dword")) == word)
}

fn fill(out: &mut Vec<u8>, bytes: u64, word: u32) {
    for _ in 0..bytes / 4 {
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// The `.text` payload for kernels of `sizes` (holes zeroed) and each kernel's offset.
fn text_bytes(prefix: &[u8], sizes: &[u64], tail: Tail) -> (Vec<u8>, Vec<u64>) {
    let mut out = prefix.to_vec();
    let mut starts = Vec::with_capacity(sizes.len());
    for (k, &size) in sizes.iter().enumerate() {
        if k > 0 {
            let at = align(out.len() as u64, KERNEL_ALIGN);
            let pad = at - out.len() as u64;
            fill(&mut out, pad, S_NOP_0);
        }
        starts.push(out.len() as u64);
        out.resize(out.len() + size as usize, 0);
    }
    if tail == Tail::CodeEnd {
        let pad = align(out.len() as u64, CACHE_LINE) - out.len() as u64 + CODE_END_EXTRA;
        fill(&mut out, pad, S_CODE_END);
    }
    (out, starts)
}

/// Reads the input `.text` as prefix + kernels + padding + tail and checks that the rules
/// reproduce it.
fn text_model(env: &Envelope, text: usize) -> Result<(Vec<u8>, Tail), ElfError> {
    let h = &env.sections[text].header;
    let SectionData::Bytes(bytes) = &env.sections[text].data else {
        return Err(fail(".text holds no bytes"));
    };
    let rel = |va: u64| (va - h.sh_addr) as usize;
    let first = &env.kernels[0];
    let prefix = bytes[..rel(first.entry_va)].to_vec();
    let last = env.kernels.last().expect("kernels");
    let end = rel(last.entry_va + last.size);
    let tail = if end == bytes.len() {
        Tail::None
    } else if words_are(&bytes[end..], S_CODE_END) {
        Tail::CodeEnd
    } else {
        return Err(fail(
            ".text after the last kernel is neither empty nor s_code_end padding",
        ));
    };
    let sizes: Vec<u64> = env.kernels.iter().map(|k| k.size).collect();
    let (model, starts) = text_bytes(&prefix, &sizes, tail);
    let placed = env
        .kernels
        .iter()
        .zip(&starts)
        .all(|(slot, &at)| rel(slot.entry_va) as u64 == at);
    if !placed || model != *bytes {
        return Err(fail(".text is not kernels at 256-byte alignment padded with s_nop 0 (and an s_code_end tail)"));
    }
    Ok((prefix, tail))
}

// ---------------------------------------------------------------------------
// String tables
// ---------------------------------------------------------------------------

/// A string table as `"\0"` + NUL-terminated names; `index` maps a start offset to a name.
struct StrTab {
    names: Vec<String>,
    index: HashMap<u32, usize>,
}

impl StrTab {
    fn parse(bytes: &[u8]) -> Result<Self, ElfError> {
        if bytes.first() != Some(&0) || bytes.last() != Some(&0) {
            return Err(fail("string table does not start and end with NUL"));
        }
        let mut names = Vec::new();
        let mut index = HashMap::new();
        let mut at = 1usize;
        while at < bytes.len() {
            let len = bytes[at..]
                .iter()
                .position(|&b| b == 0)
                .expect("ends with NUL");
            let name = std::str::from_utf8(&bytes[at..at + len])
                .map_err(|_| fail("string table name is not UTF-8"))?;
            index.insert(at as u32, names.len());
            names.push(name.to_owned());
            at += len + 1;
        }
        Ok(Self { names, index })
    }

    fn bytes(names: &[String]) -> (Vec<u8>, Vec<u32>) {
        let mut out = vec![0u8];
        let mut offsets = Vec::with_capacity(names.len());
        for n in names {
            offsets.push(out.len() as u32);
            out.extend_from_slice(n.as_bytes());
            out.push(0);
        }
        (out, offsets)
    }
}

// ---------------------------------------------------------------------------
// Hash tables (lld)
// ---------------------------------------------------------------------------

fn gnu_hash(name: &[u8]) -> u32 {
    name.iter().fold(5381u32, |h, &c| {
        h.wrapping_mul(33).wrapping_add(u32::from(c))
    })
}

fn sysv_hash(name: &[u8]) -> u32 {
    let mut h = 0u32;
    for &c in name {
        h = (h << 4).wrapping_add(u32::from(c));
        let g = h & 0xf000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

fn gnu_nbuckets(defined: usize) -> u32 {
    (defined / 4).max(1) as u32
}

/// lld's `.dynsym` order: the null symbol, undefined symbols in input order, then the
/// defined ones sorted by `(gnu_hash % nbuckets, name offset)`. Returns the new order as
/// input indices.
fn dynsym_order(symbols: &[Symbol], names: &[&str]) -> Vec<usize> {
    let (undefined, defined): (Vec<usize>, Vec<usize>) =
        (1..symbols.len()).partition(|&i| symbols[i].st_shndx == E::SHN_UNDEF);
    let buckets = gnu_nbuckets(defined.len());
    let mut sorted = defined;
    sorted.sort_by_key(|&i| (gnu_hash(names[i].as_bytes()) % buckets, symbols[i].st_name));
    std::iter::once(0).chain(undefined).chain(sorted).collect()
}

fn gnu_hash_bytes(symbols: &[Symbol], names: &[&str]) -> Vec<u8> {
    let first = symbols
        .iter()
        .position(|s| s.st_shndx != E::SHN_UNDEF)
        .filter(|&i| i > 0)
        .unwrap_or(symbols.len());
    let hashed: Vec<(u32, usize)> = (first..symbols.len())
        .map(|i| (gnu_hash(names[i].as_bytes()), i))
        .collect();
    let buckets = gnu_nbuckets(hashed.len());
    let mask_words = if hashed.is_empty() {
        1
    } else {
        ((hashed.len() * 12 / 64) as u64 + 1).next_power_of_two() as usize
    };
    let mut out = Vec::new();
    for v in [buckets, first as u32, mask_words as u32, GNU_HASH_SHIFT2] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut bloom = vec![0u64; mask_words];
    for &(h, _) in &hashed {
        let word = (h as usize / 64) & (mask_words - 1);
        bloom[word] |= 1u64 << (h % 64);
        bloom[word] |= 1u64 << ((h >> GNU_HASH_SHIFT2) % 64);
    }
    for w in bloom {
        out.extend_from_slice(&w.to_le_bytes());
    }
    let mut bucket_words = vec![0u32; buckets as usize];
    let mut chain = Vec::with_capacity(hashed.len());
    for (n, &(h, i)) in hashed.iter().enumerate() {
        let b = h % buckets;
        let last = hashed.get(n + 1).is_none_or(|&(h2, _)| h2 % buckets != b);
        chain.push(if last { h | 1 } else { h & !1 });
        if n == 0 || hashed[n - 1].0 % buckets != b {
            bucket_words[b as usize] = i as u32;
        }
    }
    for w in bucket_words.into_iter().chain(chain) {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn sysv_hash_bytes(names: &[&str]) -> Vec<u8> {
    let n = names.len();
    let mut buckets = vec![0u32; n];
    let mut chains = vec![0u32; n];
    for (i, name) in names.iter().enumerate().skip(1) {
        let b = (sysv_hash(name.as_bytes()) as usize) % n;
        chains[i] = buckets[b];
        buckets[b] = i as u32;
    }
    let mut out = Vec::with_capacity(4 * (2 + 2 * n));
    for w in [n as u32, n as u32]
        .into_iter()
        .chain(buckets)
        .chain(chains)
    {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// .eh_frame
// ---------------------------------------------------------------------------

const DW_EH_PE_PCREL_SDATA4: u8 = 0x1b;

/// FDEs as `(record offset, pc_begin field offset)`; every CIE must be `zR` with
/// pc-relative `sdata4` addresses.
fn fdes(bytes: &[u8]) -> Result<Vec<usize>, ElfError> {
    let u32_at = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("four")))
            .ok_or_else(|| fail(".eh_frame is truncated"))
    };
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let len = u32_at(at)? as usize;
        if len == 0 {
            if at + 4 != bytes.len() {
                return Err(fail(".eh_frame has bytes after its terminator"));
            }
            break;
        }
        if len == 0xffff_ffff {
            return Err(fail("64-bit .eh_frame records are not modelled"));
        }
        let end = at + 4 + len;
        if end > bytes.len() {
            return Err(fail(".eh_frame record runs past the section"));
        }
        if u32_at(at + 4)? == 0 {
            let aug_at = at + 9;
            let aug_len = bytes[aug_at..end]
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(|| fail("CIE augmentation is unterminated"))?;
            if &bytes[aug_at..aug_at + aug_len] != b"zR" {
                return Err(fail("only zR CIEs are modelled"));
            }
            let mut p = aug_at + aug_len + 1;
            let mut uleb = || -> Result<(), ElfError> {
                while *bytes.get(p).ok_or_else(|| fail("CIE is truncated"))? & 0x80 != 0 {
                    p += 1;
                }
                p += 1;
                Ok(())
            };
            uleb()?;
            uleb()?;
            uleb()?;
            uleb()?;
            if bytes.get(p) != Some(&DW_EH_PE_PCREL_SDATA4) {
                return Err(fail("CIE address encoding is not pcrel|sdata4"));
            }
        } else {
            out.push(at);
        }
        at = end;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Address assignment
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
struct Placement {
    addr: Vec<u64>,
    off: Vec<u64>,
    size: Vec<u64>,
    shoff: u64,
}

/// `PT_LOAD` index of every allocated section (sections are placed in index order).
fn load_of(env: &Envelope) -> Result<Vec<Option<usize>>, ElfError> {
    let loads: Vec<(usize, u64, u64)> = env
        .segments
        .iter()
        .enumerate()
        .filter(|(_, p)| p.p_type == E::PT_LOAD)
        .map(|(i, p)| (i, p.p_vaddr, p.p_vaddr + p.p_memsz))
        .collect();
    env.sections
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let h = &s.header;
            if i == 0 || h.sh_flags & u64::from(E::SHF_ALLOC) == 0 {
                return Ok(None);
            }
            let end = h.sh_addr + h.sh_size;
            loads
                .iter()
                .find(|(_, lo, hi)| {
                    h.sh_addr >= *lo && end <= *hi && (h.sh_size > 0 || h.sh_addr < *hi)
                })
                .map(|(l, ..)| Some(*l))
                .ok_or_else(|| fail(format!("allocated section {i} is in no PT_LOAD")))
        })
        .collect()
}

fn place(
    env: &Envelope,
    loads: &[Option<usize>],
    sizes: &[u64],
    relro_padding: Option<usize>,
) -> Result<Placement, ElfError> {
    let n = env.sections.len();
    let mut p = Placement {
        addr: vec![0; n],
        off: vec![0; n],
        size: sizes.to_vec(),
        shoff: 0,
    };
    let head = env.header.e_phoff + env.segments.len() as u64 * u64::from(env.header.e_phentsize);
    let (mut dot, mut off) = (head, head);
    let mut current: Option<(usize, usize)> = None; // (PT_LOAD, first section)
    for i in 1..n {
        let h = &env.sections[i].header;
        let a = h.sh_addralign.max(1);
        let nobits = h.sh_type == E::SHT_NOBITS;
        let Some(load) = loads[i] else {
            if h.sh_flags & u64::from(E::SHF_ALLOC) != 0 {
                return Err(fail(format!("allocated section {i} is in no PT_LOAD")));
            }
            p.off[i] = align(off, a);
            if !nobits {
                off = p.off[i] + p.size[i];
            }
            continue;
        };
        match current {
            // Same PT_LOAD: address order, file offset congruent with the first section.
            Some((l, first)) if l == load => {
                p.addr[i] = align(dot, a);
                p.off[i] = if nobits {
                    off
                } else {
                    p.off[first] + (p.addr[i] - p.addr[first])
                };
            }
            // A later PT_LOAD: lld's "no file padding" start.
            Some(_) => {
                let addr = align(align(dot, PAGE) + off % PAGE, a);
                p.addr[i] = addr;
                p.off[i] = align_skew(off, PAGE, addr);
                current = Some((load, i));
            }
            // The first PT_LOAD maps the headers at VA 0: address == offset.
            None => {
                p.addr[i] = align(dot, a);
                p.off[i] = p.addr[i];
                current = Some((load, i));
            }
        }
        if Some(i) == relro_padding {
            p.size[i] = align(p.addr[i], PAGE) - p.addr[i];
        }
        dot = p.addr[i] + p.size[i];
        if !nobits {
            off = p.off[i] + p.size[i];
        }
    }
    p.shoff = align(off, 8);
    Ok(p)
}

/// A program header as its first/last covered section plus fixed distances.
struct SegmentModel {
    first: usize,
    last: usize,
    head_off: u64,
    head_va: u64,
    file_tail: Option<(usize, u64)>,
    mem_tail: u64,
}

fn segment_models(env: &Envelope) -> Result<Vec<Option<SegmentModel>>, ElfError> {
    env.segments
        .iter()
        .map(|p| {
            if matches!(p.p_type, E::PT_PHDR | E::PT_GNU_STACK)
                || p.p_memsz == 0 && p.p_filesz == 0 && p.p_type != E::PT_LOAD
            {
                return Ok(None);
            }
            let covered: Vec<usize> = env
                .sections
                .iter()
                .enumerate()
                .filter(|(i, s)| {
                    let h = &s.header;
                    *i > 0
                        && h.sh_flags & u64::from(E::SHF_ALLOC) != 0
                        && h.sh_addr >= p.p_vaddr
                        && h.sh_addr + h.sh_size <= p.p_vaddr + p.p_memsz
                        && (h.sh_size > 0 || h.sh_addr < p.p_vaddr + p.p_memsz)
                })
                .map(|(i, _)| i)
                .collect();
            let (&first, &last) = (
                covered.first().ok_or_else(|| {
                    fail(format!("segment type {:#x} covers no section", p.p_type))
                })?,
                covered.last().expect("non-empty"),
            );
            let fh = &env.sections[first].header;
            let file_last = covered
                .iter()
                .rev()
                .copied()
                .find(|&i| env.sections[i].header.sh_type != E::SHT_NOBITS);
            let file_tail = match file_last {
                Some(i) => {
                    let h = &env.sections[i].header;
                    Some((
                        i,
                        (p.p_offset + p.p_filesz)
                            .checked_sub(h.sh_offset + h.sh_size)
                            .ok_or_else(|| fail("segment file size ends inside a section"))?,
                    ))
                }
                None if p.p_filesz == 0 => None,
                None => return Err(fail("segment has file bytes but only NOBITS sections")),
            };
            let lh = &env.sections[last].header;
            Ok(Some(SegmentModel {
                first,
                last,
                head_off: fh
                    .sh_offset
                    .checked_sub(p.p_offset)
                    .ok_or_else(|| fail("segment starts after its first section"))?,
                head_va: fh
                    .sh_addr
                    .checked_sub(p.p_vaddr)
                    .ok_or_else(|| fail("segment starts after its first section"))?,
                file_tail,
                mem_tail: (p.p_vaddr + p.p_memsz)
                    .checked_sub(lh.sh_addr + lh.sh_size)
                    .ok_or_else(|| fail("segment memory ends inside a section"))?,
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The re-layout
// ---------------------------------------------------------------------------

pub(crate) fn relayout(
    env: &Envelope,
    parts: &[KernelParts<'_>],
    metadata: &[&HsaKernelMetadata],
) -> Result<Layout, ElfError> {
    if env.header.abi_version != ABI_VERSION_V6
        || !matches!(
            env.header.e_flags & 0xff,
            EF_AMDGPU_MACH_GFX1201 | EF_AMDGPU_MACH_GFX1100 | EF_AMDGPU_MACH_GFX1151
        )
    {
        return Err(fail(
            "re-layout is modelled for code object v6 gfx1100, gfx1151 and gfx1201 only",
        ));
    }
    if env.gaps.iter().any(|g| !matches!(g.fill, Fill::Zero(_))) {
        return Err(fail("inter-section padding is not all zero"));
    }
    let r = roles(env)?;
    let symbols = KernelSymbols::find(env)?;
    let loads = load_of(env)?;
    let n = env.sections.len();
    let old_sizes: Vec<u64> = env.sections.iter().map(|s| s.header.sh_size).collect();

    // The rules must reproduce the input before they are trusted with new sizes.
    let input = place(env, &loads, &old_sizes, r.relro_padding)?;
    let file_end = env
        .gaps
        .iter()
        .map(|g| g.offset + g.fill.len())
        .chain([input.shoff + n as u64 * u64::from(env.header.e_shentsize)])
        .max()
        .unwrap_or(0);
    for i in 1..n {
        let h = &env.sections[i].header;
        if (input.addr[i], input.off[i], input.size[i]) != (h.sh_addr, h.sh_offset, h.sh_size) {
            return Err(fail(format!(
                "the lld layout rules do not reproduce section {i} ({})",
                env.section_name(i).unwrap_or("?")
            )));
        }
    }
    if input.shoff != env.header.e_shoff
        || file_end != input.shoff + n as u64 * u64::from(env.header.e_shentsize)
    {
        return Err(fail(
            "the lld layout rules do not reproduce the section header table offset or the file end",
        ));
    }
    let segments = segment_models(env)?;
    let (prefix, tail) = text_model(env, r.text)?;

    // Sizes and contents that do not depend on addresses.
    let mut out = env.clone();
    let mut sizes = old_sizes.clone();
    let code_sizes: Vec<u64> = parts.iter().map(|p| p.code.len() as u64).collect();
    if let Some(k) = code_sizes.iter().position(|s| s % 4 != 0 || *s == 0) {
        return Err(fail(format!(
            "code of kernel {} is {} bytes, not a positive dword multiple",
            env.kernels[k].name, code_sizes[k]
        )));
    }
    let (text, starts) = text_bytes(&prefix, &code_sizes, tail);
    sizes[r.text] = text.len() as u64;
    out.sections[r.text].data = SectionData::Bytes(text);
    for (i, section) in env.sections.iter().enumerate() {
        if let SectionData::Notes(notes) = &section.data {
            sizes[i] = write_notes(notes, section.header.sh_addralign, metadata)?.len() as u64;
        }
    }
    let renamed: Vec<bool> = env
        .kernels
        .iter()
        .zip(&symbols.old_names)
        .map(|(slot, old)| slot.name != *old)
        .collect();
    // New name offsets per string table: old offset -> new offset.
    let mut name_maps: HashMap<usize, HashMap<u32, u32>> = HashMap::new();
    if renamed.iter().any(|&r| r) {
        for (t, section) in env.sections.iter().enumerate() {
            let SectionData::Symbols(syms) = &section.data else {
                continue;
            };
            let link = section.header.sh_link as usize;
            if name_maps.contains_key(&link) {
                return Err(fail("two symbol tables share a string table"));
            }
            let SectionData::Bytes(strings) = &env.sections[link].data else {
                return Err(fail("symbol string table holds no bytes"));
            };
            let table = StrTab::parse(strings)?;
            if StrTab::bytes(&table.names).0 != *strings {
                return Err(fail("string table is not NUL-separated names"));
            }
            let mut names = table.names.clone();
            let mut owner: HashMap<usize, (usize, bool)> = HashMap::new();
            for (i, s) in syms.iter().enumerate() {
                if s.st_name == 0 {
                    continue;
                }
                let at = *table.index.get(&s.st_name).ok_or_else(|| {
                    fail(format!(
                        "symbol {i} of table {t} names the middle of a string"
                    ))
                })?;
                match (symbols.roles.get(&(t, i)), owner.get(&at)) {
                    (Some(&role), None) => {
                        owner.insert(at, role);
                    }
                    (Some(&role), Some(&prev)) if role != prev => {
                        return Err(fail("two kernel symbols share a name string"))
                    }
                    _ => {}
                }
            }
            for (at, (k, kd)) in &owner {
                if renamed[*k] {
                    names[*at] = if *kd {
                        format!("{}.kd", env.kernels[*k].name)
                    } else {
                        env.kernels[*k].name.clone()
                    };
                }
            }
            let referenced_by_other = syms.iter().enumerate().any(|(i, s)| {
                s.st_name != 0
                    && !symbols.roles.contains_key(&(t, i))
                    && table
                        .index
                        .get(&s.st_name)
                        .is_some_and(|at| owner.get(at).is_some_and(|(k, _)| renamed[*k]))
            });
            if referenced_by_other {
                return Err(fail(
                    "a renamed kernel's name string is shared with another symbol",
                ));
            }
            let (bytes, offsets) = StrTab::bytes(&names);
            let old_offsets = StrTab::bytes(&table.names).1;
            name_maps.insert(
                link,
                old_offsets
                    .into_iter()
                    .zip(offsets)
                    .chain([(0, 0)])
                    .collect(),
            );
            sizes[link] = bytes.len() as u64;
            out.sections[link].data = SectionData::Bytes(bytes);
        }
        if let Some(dynamic) = r.dynamic {
            let SectionData::Dynamic(dyns) = &env.sections[dynamic].data else {
                return Err(fail(".dynamic holds no entries"));
            };
            if dyns.iter().any(|d| {
                matches!(
                    d.d_tag as u32,
                    E::DT_NEEDED | E::DT_SONAME | E::DT_RPATH | E::DT_RUNPATH
                )
            }) {
                return Err(fail(
                    ".dynamic names .dynstr strings; renaming would move them",
                ));
            }
        }
    }

    // Addresses.
    let placed = place(env, &loads, &sizes, r.relro_padding)?;
    let va_delta = |i: usize| placed.addr[i].wrapping_sub(env.sections[i].header.sh_addr);
    let text_addr = placed.addr[r.text];
    let new_entry: Vec<u64> = starts.iter().map(|s| text_addr + s).collect();
    let new_kd: Vec<u64> = env
        .kernels
        .iter()
        .map(|k| k.kd_va.wrapping_add(va_delta(r.kd)))
        .collect();

    // Symbols.
    let first_entry = env.kernels[0].entry_va;
    for (t, section) in env.sections.iter().enumerate() {
        let SectionData::Symbols(syms) = &section.data else {
            continue;
        };
        let names = name_maps.get(&(section.header.sh_link as usize));
        let mut moved = Vec::with_capacity(syms.len());
        for (i, s) in syms.iter().enumerate() {
            let mut s = *s;
            if let Some(map) = names {
                s.st_name = *map
                    .get(&s.st_name)
                    .ok_or_else(|| fail("symbol name outside its string table"))?;
            }
            match symbols.roles.get(&(t, i)) {
                Some(&(k, false)) => {
                    s.st_value = new_entry[k];
                    s.st_size = code_sizes[k];
                }
                Some(&(k, true)) => s.st_value = new_kd[k],
                None if s.st_shndx != E::SHN_UNDEF && s.st_shndx < SHN_LORESERVE => {
                    let sec = usize::from(s.st_shndx);
                    if sec == r.text && s.st_value >= first_entry {
                        return Err(fail(format!(
                            "symbol {i} of table {t} lies among the kernels in .text"
                        )));
                    }
                    if sec >= n {
                        return Err(fail(format!("symbol {i} of table {t} names section {sec}")));
                    }
                    s.st_value = s.st_value.wrapping_add(va_delta(sec));
                }
                None => {}
            }
            moved.push(s);
        }
        out.sections[t].data = SectionData::Symbols(moved);
    }
    // `.dynsym` order and hash tables follow the (possibly renamed) names.
    if let Some(dynsym) = r.dynsym.filter(|_| !name_maps.is_empty()) {
        let link = env.sections[dynsym].header.sh_link as usize;
        let name_list = |data: &SectionData,
                         strings: &SectionData|
         -> Result<Vec<String>, ElfError> {
            let (SectionData::Symbols(syms), SectionData::Bytes(strings)) = (data, strings) else {
                return Err(fail(".dynsym has no string table"));
            };
            syms.iter()
                .map(|s| {
                    crate::elf::cstr(strings, s.st_name)
                        .map(str::to_owned)
                        .ok_or_else(|| fail(".dynsym name is not a string"))
                })
                .collect()
        };
        let SectionData::Symbols(old_syms) = &env.sections[dynsym].data else {
            unreachable!("dynsym parses as symbols")
        };
        let old_names = name_list(&env.sections[dynsym].data, &env.sections[link].data)?;
        let old_refs: Vec<&str> = old_names.iter().map(String::as_str).collect();
        let check = |i: Option<usize>, bytes: Vec<u8>, what: &str| -> Result<(), ElfError> {
            match i.map(|i| &env.sections[i].data) {
                Some(SectionData::Bytes(b)) if *b == bytes => Ok(()),
                None => Ok(()),
                _ => Err(fail(format!(
                    "the lld {what} rule does not reproduce the input"
                ))),
            }
        };
        if dynsym_order(old_syms, &old_refs) != (0..old_syms.len()).collect::<Vec<_>>() {
            return Err(fail("the lld .dynsym order does not reproduce the input"));
        }
        check(r.gnu_hash, gnu_hash_bytes(old_syms, &old_refs), ".gnu.hash")?;
        check(r.hash, sysv_hash_bytes(&old_refs), ".hash")?;
        let SectionData::Symbols(new_syms) = &out.sections[dynsym].data else {
            unreachable!("set above")
        };
        let new_names = name_list(&out.sections[dynsym].data, &out.sections[link].data)?;
        let new_refs: Vec<&str> = new_names.iter().map(String::as_str).collect();
        let order = dynsym_order(new_syms, &new_refs);
        let sorted: Vec<Symbol> = order.iter().map(|&i| new_syms[i]).collect();
        let sorted_names: Vec<&str> = order.iter().map(|&i| new_refs[i]).collect();
        if let Some(g) = r.gnu_hash {
            out.sections[g].data = SectionData::Bytes(gnu_hash_bytes(&sorted, &sorted_names));
        }
        if let Some(h) = r.hash {
            out.sections[h].data = SectionData::Bytes(sysv_hash_bytes(&sorted_names));
        }
        out.sections[dynsym].data = SectionData::Symbols(sorted);
    }
    for i in [r.gnu_hash, r.hash].into_iter().flatten() {
        let SectionData::Bytes(b) = &out.sections[i].data else {
            return Err(fail("hash section holds no bytes"));
        };
        if b.len() as u64 != sizes[i] {
            return Err(fail("hash table changed size"));
        }
    }

    // .dynamic
    if let Some(dynamic) = r.dynamic {
        let SectionData::Dynamic(dyns) = &env.sections[dynamic].data else {
            return Err(fail(".dynamic holds no entries"));
        };
        let dynstr = r.dynsym.map(|d| env.sections[d].header.sh_link as usize);
        let mut moved = Vec::with_capacity(dyns.len());
        for d in dyns {
            let target = match d.d_tag as u32 {
                E::DT_HASH => r.hash,
                E::DT_GNU_HASH => r.gnu_hash,
                E::DT_STRTAB => dynstr,
                E::DT_SYMTAB => r.dynsym,
                E::DT_STRSZ => {
                    let s = dynstr.ok_or_else(|| fail("DT_STRSZ without .dynstr"))?;
                    if d.d_val != env.sections[s].header.sh_size {
                        return Err(fail("DT_STRSZ is not the .dynstr size"));
                    }
                    moved.push(Dyn {
                        d_tag: d.d_tag,
                        d_val: sizes[s],
                    });
                    continue;
                }
                E::DT_NULL | E::DT_SYMENT | E::DT_FLAGS | E::DT_FLAGS_1 => {
                    moved.push(*d);
                    continue;
                }
                other => return Err(fail(format!("dynamic tag {other:#x} is not modelled"))),
            };
            let s = target.ok_or_else(|| {
                fail(format!(
                    "dynamic tag {:#x} names a missing section",
                    d.d_tag
                ))
            })?;
            if d.d_val != env.sections[s].header.sh_addr {
                return Err(fail(format!(
                    "dynamic tag {:#x} is not its section's address",
                    d.d_tag
                )));
            }
            moved.push(Dyn {
                d_tag: d.d_tag,
                d_val: placed.addr[s],
            });
        }
        out.sections[dynamic].data = SectionData::Dynamic(moved);
    }

    // .eh_frame
    if let Some(eh) = r.eh_frame {
        let SectionData::Bytes(bytes) = &env.sections[eh].data else {
            return Err(fail(".eh_frame holds no bytes"));
        };
        let old_base = env.sections[eh].header.sh_addr;
        let new_base = placed.addr[eh];
        let mut patched = bytes.clone();
        for at in fdes(bytes)? {
            let field = at + 8;
            let begin = i32::from_le_bytes(bytes[field..field + 4].try_into().expect("four"));
            let range = u32::from_le_bytes(bytes[field + 4..field + 8].try_into().expect("four"));
            let target = (old_base + field as u64).wrapping_add_signed(i64::from(begin));
            let k = env
                .kernels
                .iter()
                .position(|slot| slot.entry_va == target && slot.size == u64::from(range))
                .ok_or_else(|| fail(format!("FDE at {at:#x} does not cover exactly one kernel")))?;
            let begin = i64::try_from(new_entry[k]).expect("VA fits")
                - i64::try_from(new_base + field as u64).expect("VA fits");
            let begin =
                i32::try_from(begin).map_err(|_| fail("FDE pc_begin does not fit sdata4"))?;
            let range = u32::try_from(code_sizes[k])
                .map_err(|_| fail("FDE pc_range does not fit udata4"))?;
            patched[field..field + 4].copy_from_slice(&begin.to_le_bytes());
            patched[field + 4..field + 8].copy_from_slice(&range.to_le_bytes());
        }
        out.sections[eh].data = SectionData::Bytes(patched);
    }

    // Section headers, program headers, ELF header.
    for i in 1..n {
        let h = &mut out.sections[i].header;
        if h.sh_flags & u64::from(E::SHF_ALLOC) != 0 {
            h.sh_addr = placed.addr[i];
        }
        h.sh_offset = placed.off[i];
        h.sh_size = placed.size[i];
    }
    for (p, model) in out.segments.iter_mut().zip(&segments) {
        let Some(m) = model else { continue };
        let first = &out.sections[m.first].header;
        let last = &out.sections[m.last].header;
        p.p_offset = first.sh_offset - m.head_off;
        p.p_vaddr = first.sh_addr - m.head_va;
        p.p_paddr = p.p_vaddr;
        p.p_filesz = m.file_tail.map_or(0, |(i, tail)| {
            let h = &out.sections[i].header;
            h.sh_offset + h.sh_size + tail - p.p_offset
        });
        p.p_memsz = last.sh_addr + last.sh_size + m.mem_tail - p.p_vaddr;
    }
    out.header.e_shoff = placed.shoff;
    out.gaps.clear();

    // Kernel slots and layout-derived descriptor fields.
    let mut descriptors = Vec::with_capacity(parts.len());
    for (k, (slot, part)) in out.kernels.iter_mut().zip(parts).enumerate() {
        let mut d = part.descriptor.clone();
        let old_size = env.kernels[k].size;
        if code_sizes[k] != old_size {
            let pref_max = if env.header.e_flags & 0xff == EF_AMDGPU_MACH_GFX1201 {
                INST_PREF_MAX
            } else {
                INST_PREF_MAX_GFX11
            };
            if inst_pref_field(d.compute_pgm_rsrc3, pref_max) != inst_pref_size(old_size, pref_max)
            {
                return Err(fail(format!("kernel {} changed size but its INST_PREF_SIZE {} does not follow min(ceil(size/128), {pref_max}) = {}",
                    slot.name, inst_pref_field(d.compute_pgm_rsrc3, pref_max), inst_pref_size(old_size, pref_max))));
            }
            d.compute_pgm_rsrc3 = Rsrc3(
                d.compute_pgm_rsrc3.0 & !(pref_max << INST_PREF_SHIFT)
                    | inst_pref_size(code_sizes[k], pref_max) << INST_PREF_SHIFT,
            );
        }
        d.kernel_code_entry_byte_offset = i64::try_from(new_entry[k]).expect("VA fits")
            - i64::try_from(new_kd[k]).expect("VA fits");
        slot.entry_va = new_entry[k];
        slot.size = code_sizes[k];
        slot.kd_va = new_kd[k];
        descriptors.push(d);
    }
    Ok(Layout {
        envelope: out,
        descriptors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lld_hashes_match_known_values() {
        // ELF gABI example and the GNU hash of the empty string.
        assert_eq!(gnu_hash(b""), 5381);
        assert_eq!(gnu_hash(b"printf"), 0x156b_2bb8);
        assert_eq!(sysv_hash(b"printf"), 0x0779_05a6);
    }

    #[test]
    fn skewed_alignment_keeps_the_page_offset() {
        assert_eq!(align_skew(0x24c4, PAGE, 0x3500), 0x2500);
        assert_eq!(align_skew(0xa200, PAGE, 0xc200), 0xa200);
        assert_eq!(align_skew(0x2501, PAGE, 0x3500), 0x3500);
    }
}
