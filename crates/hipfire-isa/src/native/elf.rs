//! The code object `ld.lld -shared` links from one builder `.o`, written directly.
//!
//! Layout (lld's rules for an AMDGPU HSA input of one `.text`, one `.rodata`
//! of descriptors and one metadata note; `-z relro`, 4 KiB pages):
//!
//! 1. ELF header, 8 program headers (PHDR, LOAD r, LOAD rx, LOAD rw, DYNAMIC,
//!    GNU_RELRO, GNU_STACK, NOTE), then sections in this order: `.note`,
//!    `.dynsym`, `.gnu.hash`, `.hash`, `.dynstr`, `.rodata` (first, read-only
//!    segment), `.text` (executable segment), `.dynamic`, `.relro_padding`
//!    (read-write segment), then `.comment`, `.symtab`, `.shstrtab`, `.strtab`
//!    and the section header table at the next 8-byte boundary.
//! 2. A section starting a segment gets `addr = align(dot, page) + dot % page`
//!    aligned to its own alignment and the smallest file offset congruent to
//!    it modulo the page; later sections follow at their own alignment.
//! 3. `.relro_padding` (NOBITS) runs the read-write segment to the page end.
//! 4. Symbols: every kernel `sym` (FUNC, `.text`) then `sym.kd` (OBJECT,
//!    `.rodata`), all GLOBAL PROTECTED, in source order; `.symtab` leads with
//!    the local hidden `_DYNAMIC`, whose name `.strtab` stores last. `.dynsym`
//!    holds the globals stably sorted by GNU-hash bucket (`max(n / 4, 1)`
//!    buckets); its Bloom filter has `NextPowerOf2(12 n / 64)` words, shift 26;
//!    the SysV `.hash` has one bucket per `.dynsym` entry.
//! 5. `.comment` is this writer's own stamp, `hipfire peacemaker native-emit <crate version>`
//!    (NUL-terminated; the section is mergeable strings). Its length feeds
//!    `.symtab`'s offset, so every later offset follows the stamp.
use crate::Arch;
use super::descriptor::{ENTRY_OFFSET, KD_SIZE};

const PAGE: u64 = 0x1000;
const EHDR: u64 = 64;
const PHDR: u64 = 56;
const PHNUM: u64 = 8;
const SHDR: u64 = 64;
const SYM: u64 = 24;
const DYN: u64 = 16;
/// `.dynamic`: SYMTAB, SYMENT, STRTAB, STRSZ, GNU_HASH, HASH, NULL.
const DYNAMIC_ENTRIES: u64 = 7;
const KERNEL_ALIGN: u64 = 256;
const S_NOP_0: u32 = 0xbf80_0000;
/// `.comment`: the native writer's identity. It is deliberately not the ROCm linker's
/// `Linker: AMD LLD ...` string, which would claim a tool that never ran.
pub(crate) const NATIVE_COMMENT: &str = concat!("hipfire peacemaker native-emit ", env!("CARGO_PKG_VERSION"));
const SHSTRTAB: &[u8] = b"\0.note\0.dynsym\0.gnu.hash\0.hash\0.dynstr\0.rodata\0.text\0.dynamic\0.relro_padding\0.comment\0.symtab\0.shstrtab\0.strtab\0";
const NT_AMDGPU_METADATA: u32 = 32;
const GNU_HASH_SHIFT2: u32 = 26;

/// One kernel to link: its symbol, code, and descriptor (entry offset zero).
pub(crate) struct Kernel { pub symbol: String, pub code: Vec<u8>, pub descriptor: [u8; KD_SIZE] }

fn align(x: u64, a: u64) -> u64 { if a <= 1 { x } else { x.next_multiple_of(a) } }
/// Smallest `y >= x` with `y % a == skew % a`.
fn align_skew(x: u64, a: u64, skew: u64) -> u64 {
    let base = x - x % a + skew % a;
    if base >= x { base } else { base + a }
}
/// The address of a section that starts a new segment (rule 2).
fn segment_start(dot: u64, section_align: u64) -> u64 { align(align(dot, PAGE) + dot % PAGE, section_align) }

fn gnu_hash(name: &str) -> u32 { name.bytes().fold(5381u32, |h, c| h.wrapping_mul(33).wrapping_add(u32::from(c))) }
fn sysv_hash(name: &str) -> u32 {
    name.bytes().fold(0u32, |h, c| {
        let h = (h << 4).wrapping_add(u32::from(c));
        let g = h & 0xf000_0000;
        (if g != 0 { h ^ (g >> 24) } else { h }) & !g
    })
}

fn e_flags(arch: Arch) -> u32 {
    // EF_AMDGPU_MACH_AMDGCN_*; gfx11/gfx12 have neither XNACK nor SRAMECC.
    match arch { Arch::Gfx1100 => 0x41, Arch::Gfx1151 => 0x4a, Arch::Gfx1201 => 0x4e }
}

#[derive(Default)]
struct Out(Vec<u8>);
impl Out {
    fn u8(&mut self, v: u8) { self.0.push(v) }
    fn u16(&mut self, v: u16) { self.0.extend(v.to_le_bytes()) }
    fn u32(&mut self, v: u32) { self.0.extend(v.to_le_bytes()) }
    fn u64(&mut self, v: u64) { self.0.extend(v.to_le_bytes()) }
    fn pad_to(&mut self, offset: u64) { assert!(self.0.len() as u64 <= offset, "layout overlap"); self.0.resize(offset as usize, 0) }
    fn bytes(&mut self, at: u64, b: &[u8]) { self.pad_to(at); self.0.extend(b) }
}

struct Section { name: u32, kind: u32, flags: u64, addr: u64, offset: u64, size: u64, link: u32, info: u32, align: u64, entsize: u64 }

fn shstr(name: &str) -> u32 {
    let needle = format!("\0{name}\0");
    SHSTRTAB.windows(needle.len()).position(|w| w == needle.as_bytes()).expect("section name") as u32 + 1
}

/// Link `kernels` and the metadata blob into the code object `ld.lld -shared` writes.
pub(crate) fn link(arch: Arch, kernels: &[Kernel], metadata: &[u8]) -> Result<Vec<u8>, String> {
    if kernels.is_empty() { return Err("a code object needs at least one kernel".into()) }
    // .text: each kernel at the next 256-byte boundary, padded with `s_nop 0`.
    let mut text = Vec::new();
    let mut entries = Vec::with_capacity(kernels.len());
    for k in kernels {
        if k.code.len() % 4 != 0 { return Err(format!("{}: code is not whole dwords", k.symbol)) }
        while text.len() as u64 % KERNEL_ALIGN != 0 { text.extend(S_NOP_0.to_le_bytes()) }
        entries.push(text.len() as u64);
        text.extend(&k.code);
    }
    // .note: one NT_AMDGPU_METADATA, owner "AMDGPU".
    let mut note = Out::default();
    note.u32(7); note.u32(metadata.len() as u32); note.u32(NT_AMDGPU_METADATA);
    note.0.extend(b"AMDGPU\0\0");
    note.0.extend(metadata);
    note.pad_to(align(note.0.len() as u64, 4));
    // Symbols in source order: (name, is_descriptor, kernel index).
    let names: Vec<(String, bool, usize)> = kernels.iter().enumerate()
        .flat_map(|(i, k)| [(k.symbol.clone(), false, i), (format!("{}.kd", k.symbol), true, i)]).collect();
    let mut dynstr = vec![0u8];
    let mut name_offset = Vec::with_capacity(names.len());
    for (name, _, _) in &names { name_offset.push(dynstr.len() as u32); dynstr.extend(name.as_bytes()); dynstr.push(0) }
    let mut strtab = dynstr.clone();
    let dynamic_name = strtab.len() as u32;
    strtab.extend(b"_DYNAMIC\0");
    // .dynsym order: globals stably sorted by GNU-hash bucket.
    let buckets = (names.len() / 4).max(1) as u32;
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by_key(|&i| gnu_hash(&names[i].0) % buckets);
    let bloom_words = ((names.len() as u64 * 12) / 64 + 1).next_power_of_two();
    let dynsym_size = (1 + names.len() as u64) * SYM;
    let gnu_hash_size = 16 + 8 * bloom_words + 4 * u64::from(buckets) + 4 * names.len() as u64;
    let hash_size = 4 * (2 + 2 * (1 + names.len() as u64));
    let rodata_size = KD_SIZE as u64 * kernels.len() as u64;

    // Addresses and offsets (first segment: address == offset).
    let note_off = align(EHDR + PHNUM * PHDR, 4);
    let dynsym_off = align(note_off + note.0.len() as u64, 8);
    let gnu_hash_off = align(dynsym_off + dynsym_size, 8);
    let hash_off = align(gnu_hash_off + gnu_hash_size, 4);
    let dynstr_off = hash_off + hash_size;
    let rodata_off = align(dynstr_off + dynstr.len() as u64, KD_SIZE as u64);
    let ro_end = rodata_off + rodata_size;
    let text_addr = segment_start(ro_end, KERNEL_ALIGN);
    let text_off = align_skew(ro_end, PAGE, text_addr);
    let dynamic_size = DYNAMIC_ENTRIES * DYN;
    let dynamic_addr = segment_start(text_addr + text.len() as u64, 8);
    let dynamic_off = align_skew(text_off + text.len() as u64, PAGE, dynamic_addr);
    let relro_addr = dynamic_addr + dynamic_size;
    let relro_size = align(relro_addr, PAGE) - relro_addr;
    let comment_off = dynamic_off + dynamic_size;
    let comment_size = NATIVE_COMMENT.len() as u64 + 1;
    let symtab_off = align(comment_off + comment_size, 8);
    let symtab_size = (2 + names.len() as u64) * SYM;
    let shstrtab_off = symtab_off + symtab_size;
    let strtab_off = shstrtab_off + SHSTRTAB.len() as u64;
    let shoff = align(strtab_off + strtab.len() as u64, 8);

    let kd_addr = |i: usize| rodata_off + (i * KD_SIZE) as u64;
    let symbol_entry = |out: &mut Out, i: usize| {
        let (_, kd, k) = names[i];
        out.u32(name_offset[i]);
        out.u8(0x10 | if kd { 1 } else { 2 }); // GLOBAL, OBJECT / FUNC
        out.u8(3); // STV_PROTECTED
        out.u16(if kd { 6 } else { 7 });
        out.u64(if kd { kd_addr(k) } else { text_addr + entries[k] });
        out.u64(if kd { KD_SIZE as u64 } else { kernels[k].code.len() as u64 });
    };

    let mut out = Out::default();
    // ELF header.
    out.0.extend([0x7f, b'E', b'L', b'F', 2, 1, 1, 0x40, 4, 0, 0, 0, 0, 0, 0, 0]);
    out.u16(3); out.u16(224); out.u32(1); out.u64(0); out.u64(EHDR); out.u64(shoff);
    out.u32(e_flags(arch)); out.u16(EHDR as u16); out.u16(PHDR as u16); out.u16(PHNUM as u16);
    out.u16(SHDR as u16); out.u16(14); out.u16(12);
    // Program headers: (type, flags, offset, addr, filesz, memsz, align).
    let phdrs = [
        (6, 4, EHDR, EHDR, PHNUM * PHDR, PHNUM * PHDR, 8),
        (1, 4, 0, 0, ro_end, ro_end, PAGE),
        (1, 5, text_off, text_addr, text.len() as u64, text.len() as u64, PAGE),
        (1, 6, dynamic_off, dynamic_addr, dynamic_size, dynamic_size + relro_size, PAGE),
        (2, 6, dynamic_off, dynamic_addr, dynamic_size, dynamic_size, 8),
        (0x6474_e552, 4, dynamic_off, dynamic_addr, dynamic_size, dynamic_size + relro_size, 1),
        (0x6474_e551, 6, 0, 0, 0, 0, 0),
        (4, 4, note_off, note_off, note.0.len() as u64, note.0.len() as u64, 4),
    ];
    for (kind, flags, offset, addr, filesz, memsz, palign) in phdrs {
        out.u32(kind); out.u32(flags); out.u64(offset); out.u64(addr); out.u64(addr);
        out.u64(filesz); out.u64(memsz); out.u64(palign);
    }
    out.bytes(note_off, &note.0);
    // .dynsym
    out.pad_to(dynsym_off);
    out.0.extend([0u8; SYM as usize]);
    for &i in &order { symbol_entry(&mut out, i) }
    // .gnu.hash
    out.pad_to(gnu_hash_off);
    out.u32(buckets); out.u32(1); out.u32(bloom_words as u32); out.u32(GNU_HASH_SHIFT2);
    let mut bloom = vec![0u64; bloom_words as usize];
    for &i in &order {
        let h = gnu_hash(&names[i].0);
        let word = ((h / 64) as u64 & (bloom_words - 1)) as usize;
        bloom[word] |= (1u64 << (h % 64)) | (1u64 << ((h >> GNU_HASH_SHIFT2) % 64));
    }
    for w in bloom { out.u64(w) }
    let mut first = vec![0u32; buckets as usize];
    for (position, &i) in order.iter().enumerate().rev() { first[(gnu_hash(&names[i].0) % buckets) as usize] = position as u32 + 1 }
    for b in first { out.u32(b) }
    for (position, &i) in order.iter().enumerate() {
        let h = gnu_hash(&names[i].0);
        let last = order.get(position + 1).is_none_or(|&j| gnu_hash(&names[j].0) % buckets != h % buckets);
        out.u32(if last { h | 1 } else { h & !1 });
    }
    // .hash (SysV): one bucket per .dynsym entry, chains by insertion.
    out.pad_to(hash_off);
    let nsyms = 1 + names.len();
    let mut sysv_buckets = vec![0u32; nsyms];
    let mut chains = vec![0u32; nsyms];
    for (position, &i) in order.iter().enumerate() {
        let index = position + 1;
        let b = (sysv_hash(&names[i].0) % nsyms as u32) as usize;
        chains[index] = sysv_buckets[b];
        sysv_buckets[b] = index as u32;
    }
    out.u32(nsyms as u32); out.u32(nsyms as u32);
    for v in sysv_buckets.into_iter().chain(chains) { out.u32(v) }
    out.bytes(dynstr_off, &dynstr);
    // .rodata: descriptors with their code entry offsets.
    out.pad_to(rodata_off);
    for (i, k) in kernels.iter().enumerate() {
        let mut kd = k.descriptor;
        let entry = (text_addr + entries[i]) as i64 - kd_addr(i) as i64;
        kd[ENTRY_OFFSET..ENTRY_OFFSET + 8].copy_from_slice(&entry.to_le_bytes());
        out.0.extend(kd);
    }
    out.bytes(text_off, &text);
    // .dynamic
    out.pad_to(dynamic_off);
    for (tag, value) in [(6u64, dynsym_off), (11, SYM), (5, dynstr_off), (10, dynstr.len() as u64), (0x6fff_fef5, gnu_hash_off), (4, hash_off), (0, 0)] {
        out.u64(tag); out.u64(value);
    }
    let mut comment = NATIVE_COMMENT.as_bytes().to_vec();
    comment.push(0);
    out.bytes(comment_off, &comment);
    // .symtab: null, local _DYNAMIC, then the globals in source order.
    out.pad_to(symtab_off);
    out.0.extend([0u8; SYM as usize]);
    out.u32(dynamic_name); out.u8(0); out.u8(2); out.u16(8); out.u64(dynamic_addr); out.u64(0);
    for i in 0..names.len() { symbol_entry(&mut out, i) }
    out.bytes(shstrtab_off, SHSTRTAB);
    out.bytes(strtab_off, &strtab);
    out.pad_to(shoff);
    let alloc = 2; let write = 1; let exec = 4; let merge = 0x10; let strings = 0x20;
    let sections = [
        Section { name: 0, kind: 0, flags: 0, addr: 0, offset: 0, size: 0, link: 0, info: 0, align: 0, entsize: 0 },
        Section { name: shstr(".note"), kind: 7, flags: alloc, addr: note_off, offset: note_off, size: note.0.len() as u64, link: 0, info: 0, align: 4, entsize: 0 },
        Section { name: shstr(".dynsym"), kind: 11, flags: alloc, addr: dynsym_off, offset: dynsym_off, size: dynsym_size, link: 5, info: 1, align: 8, entsize: SYM },
        Section { name: shstr(".gnu.hash"), kind: 0x6fff_fff6, flags: alloc, addr: gnu_hash_off, offset: gnu_hash_off, size: gnu_hash_size, link: 2, info: 0, align: 8, entsize: 0 },
        Section { name: shstr(".hash"), kind: 5, flags: alloc, addr: hash_off, offset: hash_off, size: hash_size, link: 2, info: 0, align: 4, entsize: 4 },
        Section { name: shstr(".dynstr"), kind: 3, flags: alloc, addr: dynstr_off, offset: dynstr_off, size: dynstr.len() as u64, link: 0, info: 0, align: 1, entsize: 0 },
        Section { name: shstr(".rodata"), kind: 1, flags: alloc, addr: rodata_off, offset: rodata_off, size: rodata_size, link: 0, info: 0, align: KD_SIZE as u64, entsize: 0 },
        Section { name: shstr(".text"), kind: 1, flags: alloc | exec, addr: text_addr, offset: text_off, size: text.len() as u64, link: 0, info: 0, align: KERNEL_ALIGN, entsize: 0 },
        Section { name: shstr(".dynamic"), kind: 6, flags: alloc | write, addr: dynamic_addr, offset: dynamic_off, size: dynamic_size, link: 5, info: 0, align: 8, entsize: DYN },
        Section { name: shstr(".relro_padding"), kind: 8, flags: alloc | write, addr: relro_addr, offset: comment_off, size: relro_size, link: 0, info: 0, align: 1, entsize: 0 },
        Section { name: shstr(".comment"), kind: 1, flags: merge | strings, addr: 0, offset: comment_off, size: comment_size, link: 0, info: 0, align: 1, entsize: 1 },
        Section { name: shstr(".symtab"), kind: 2, flags: 0, addr: 0, offset: symtab_off, size: symtab_size, link: 13, info: 2, align: 8, entsize: SYM },
        Section { name: shstr(".shstrtab"), kind: 3, flags: 0, addr: 0, offset: shstrtab_off, size: SHSTRTAB.len() as u64, link: 0, info: 0, align: 1, entsize: 0 },
        Section { name: shstr(".strtab"), kind: 3, flags: 0, addr: 0, offset: strtab_off, size: strtab.len() as u64, link: 0, info: 0, align: 1, entsize: 0 },
    ];
    for s in sections {
        out.u32(s.name); out.u32(s.kind); out.u64(s.flags); out.u64(s.addr); out.u64(s.offset);
        out.u64(s.size); out.u32(s.link); out.u32(s.info); out.u64(s.align); out.u64(s.entsize);
    }
    Ok(out.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hashes_match_the_elf_definitions() {
        assert_eq!(gnu_hash(""), 5381);
        assert_eq!(gnu_hash("printf"), 0x156b_2bb8);
        assert_eq!(sysv_hash("printf"), 0x0779_05a6);
    }
    #[test]
    fn segments_start_on_the_page_after_their_predecessor_with_its_offset() {
        assert_eq!(segment_start(0xc80, 256), 0x1d00);
        assert_eq!(align_skew(0xc80, PAGE, 0x1d00), 0xd00);
        assert_eq!(segment_start(0x6f04, 8), 0x7f08);
    }
}
