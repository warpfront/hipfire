//! Shared native-vs-ROCm-oracle comparison for the test crates that check the
//! native code-object writer (`native_identity`, `hipfire-rip`'s `qsa`),
//! included by path: `#[path = ".../support/native_compare.rs"] mod native_compare;`.
#![allow(dead_code)]

/// The section holding byte `at` of an ELF, for a mismatch report.
pub fn section_at(elf: &[u8], at: usize) -> String {
    let u16_ = |o: usize| u16::from_le_bytes([elf[o], elf[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shnum, shstrndx) = (u64_(0x28), u16_(0x3c), u16_(0x3e));
    if at < 64 { return "ELF header".into() }
    if at >= shoff { return format!("section header {}", (at - shoff) / 64) }
    let names = u64_(shoff + 64 * shstrndx + 24);
    for i in 0..shnum {
        let sh = shoff + 64 * i;
        let (off, size) = (u64_(sh + 24), u64_(sh + 32));
        if (off..off + size).contains(&at) && u32_(sh + 4) != 8 {
            let name = &elf[names + u32_(sh)..];
            return format!("{} +{:#x}", String::from_utf8_lossy(&name[..name.iter().position(|&b| b == 0).unwrap()]), at - off);
        }
    }
    "padding/program headers".into()
}

pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// Byte-for-byte whole-file comparison of a native code object and bundle
/// against the oracle's (the old default; unused now that the `.comment`
/// stamp differs on purpose). First differing byte, if any.
pub fn strict_whole_file_difference(native_elf: &[u8], oracle_elf: &[u8], native_bundle: &[u8], oracle_bundle: &[u8]) -> Option<String> {
    match (first_difference(native_elf, oracle_elf), first_difference(native_bundle, oracle_bundle)) {
        (None, None) => None,
        (Some(at), _) => Some(format!("code object differs at {at:#x} ({})", section_at(oracle_elf, at))),
        (None, Some(at)) => Some(format!("bundle differs at {at:#x}")),
    }
}

/// The `.comment` the native writer must stamp: the only ELF content excluded
/// from the comparison against the oracle (its content is checked against this).
fn native_comment() -> Vec<u8> { format!("hipfire peacemaker native-emit {}\0", env!("CARGO_PKG_VERSION")).into_bytes() }

const SHT_NOBITS: u64 = 8;
const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
const BUNDLE_ALIGN: u64 = 4096;
const DEVICE_TRIPLE_PREFIX: &[u8] = b"hipv4-amdgcn-amd-amdhsa--";

fn bytes_at<'a>(b: &'a [u8], offset: u64, len: u64, what: &str) -> Result<&'a [u8], String> {
    offset.checked_add(len).filter(|&end| end <= b.len() as u64).map(|end| &b[offset as usize..end as usize])
        .ok_or_else(|| format!("{what}: {offset:#x}+{len:#x} lies outside the {:#x}-byte file", b.len()))
}
fn le(b: &[u8], at: u64, width: u64, what: &str) -> Result<u64, String> {
    Ok(bytes_at(b, at, width, what)?.iter().rev().fold(0u64, |v, &x| (v << 8) | u64::from(x)))
}
fn align_up(x: u64, a: u64) -> u64 { x.next_multiple_of(a.max(1)) }
/// Every byte of `from..to` is zero (inter-section / inter-entry padding).
fn require_zero(b: &[u8], from: u64, to: u64, what: &str) -> Result<(), String> {
    let pad = bytes_at(b, from, to.saturating_sub(from), what)?;
    match pad.iter().position(|&x| x != 0) {
        None => Ok(()),
        Some(i) => Err(format!("{what}: padding byte at {:#x} is {:#x}, not zero", from as usize + i, pad[i])),
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Ehdr { ident: Vec<u8>, kind: u64, machine: u64, version: u64, entry: u64, phoff: u64, shoff: u64, flags: u64, ehsize: u64, phentsize: u64, phnum: u64, shentsize: u64, shnum: u64, shstrndx: u64 }
#[derive(Clone, Debug, PartialEq)]
struct Phdr { kind: u64, flags: u64, offset: u64, vaddr: u64, paddr: u64, filesz: u64, memsz: u64, align: u64 }
#[derive(Clone, Debug, PartialEq)]
struct Shdr { name: u64, kind: u64, flags: u64, addr: u64, offset: u64, size: u64, link: u64, info: u64, align: u64, entsize: u64 }

/// A parsed little-endian ELF64: header, program headers, section headers and
/// section names.
struct Elf<'a> { bytes: &'a [u8], ehdr: Ehdr, phdrs: Vec<Phdr>, shdrs: Vec<Shdr>, names: Vec<String> }

impl<'a> Elf<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        let h = |at: u64, w: u64| le(bytes, at, w, "ELF header");
        let ident = bytes_at(bytes, 0, 16, "e_ident")?.to_vec();
        if ident[..4] != *b"\x7fELF" || ident[4] != 2 || ident[5] != 1 { return Err("not a little-endian ELF64".into()) }
        let ehdr = Ehdr {
            ident, kind: h(16, 2)?, machine: h(18, 2)?, version: h(20, 4)?, entry: h(24, 8)?, phoff: h(32, 8)?, shoff: h(40, 8)?,
            flags: h(48, 4)?, ehsize: h(52, 2)?, phentsize: h(54, 2)?, phnum: h(56, 2)?, shentsize: h(58, 2)?, shnum: h(60, 2)?, shstrndx: h(62, 2)?,
        };
        if ehdr.ehsize != 64 || ehdr.phentsize != 56 || ehdr.shentsize != 64 { return Err(format!("unexpected ELF header/entry sizes {ehdr:?}")) }
        let phdrs = (0..ehdr.phnum).map(|i| -> Result<Phdr, String> {
            let at = ehdr.phoff.saturating_add(i * ehdr.phentsize);
            let p = |o: u64, w: u64| le(bytes, at.saturating_add(o), w, "program header");
            Ok(Phdr { kind: p(0, 4)?, flags: p(4, 4)?, offset: p(8, 8)?, vaddr: p(16, 8)?, paddr: p(24, 8)?, filesz: p(32, 8)?, memsz: p(40, 8)?, align: p(48, 8)? })
        }).collect::<Result<Vec<_>, _>>()?;
        let shdrs = (0..ehdr.shnum).map(|i| -> Result<Shdr, String> {
            let at = ehdr.shoff.saturating_add(i * ehdr.shentsize);
            let s = |o: u64, w: u64| le(bytes, at.saturating_add(o), w, "section header");
            Ok(Shdr { name: s(0, 4)?, kind: s(4, 4)?, flags: s(8, 8)?, addr: s(16, 8)?, offset: s(24, 8)?, size: s(32, 8)?, link: s(40, 4)?, info: s(44, 4)?, align: s(48, 8)?, entsize: s(56, 8)? })
        }).collect::<Result<Vec<_>, _>>()?;
        let names_header = shdrs.get(ehdr.shstrndx as usize).ok_or("e_shstrndx outside the section table")?;
        let table = bytes_at(bytes, names_header.offset, names_header.size, "section name table")?;
        let names = shdrs.iter().map(|s| -> Result<String, String> {
            let name = table.get(s.name as usize..).ok_or("section name outside .shstrtab")?;
            let end = name.iter().position(|&b| b == 0).ok_or("unterminated section name")?;
            Ok(String::from_utf8_lossy(&name[..end]).into_owned())
        }).collect::<Result<Vec<_>, String>>()?;
        Ok(Elf { bytes, ehdr, phdrs, shdrs, names })
    }
    /// The file bytes of section `i` (none for NOBITS).
    fn section(&self, i: usize) -> Result<&'a [u8], String> {
        let s = &self.shdrs[i];
        if s.kind == SHT_NOBITS { return Ok(&[]) }
        bytes_at(self.bytes, s.offset, s.size, &format!("section {i} ({})", self.names[i]))
    }
    fn comment_index(&self) -> Result<usize, String> {
        let mut found = self.names.iter().enumerate().filter(|(_, n)| *n == ".comment").map(|(i, _)| i);
        match (found.next(), found.next()) {
            (Some(i), None) => Ok(i),
            _ => Err("expected exactly one .comment section".into()),
        }
    }
}

/// The layout rules that make every offset after `.comment` derived from its
/// size, checked independently for each file: `.comment` and everything before
/// it sits where it sits in the other file; each later section starts at the
/// next multiple of its alignment after the previous section's end, with zero
/// padding between; `e_shoff` is the next 8-byte boundary after the last
/// section; the section header table ends the file.
fn check_layout(e: &Elf, c: usize, who: &str) -> Result<(), String> {
    let comment = &e.shdrs[c];
    if comment.kind == SHT_NOBITS { return Err(format!("{who}: .comment is NOBITS")) }
    for (i, s) in e.shdrs.iter().enumerate().take(c) {
        let end = if s.kind == SHT_NOBITS { s.offset } else { s.offset + s.size };
        if end > comment.offset { return Err(format!("{who}: section {i} ({}) ends at {end:#x}, past .comment at {:#x}", e.names[i], comment.offset)) }
    }
    let mut end = comment.offset + comment.size;
    for i in c + 1..e.shdrs.len() {
        let s = &e.shdrs[i];
        if s.kind == SHT_NOBITS { return Err(format!("{who}: NOBITS section {i} ({}) after .comment", e.names[i])) }
        let want = align_up(end, s.align);
        if s.offset != want { return Err(format!("{who}: section {i} ({}) at {:#x}, layout rule gives {want:#x} (previous end {end:#x}, align {})", e.names[i], s.offset, s.align)) }
        require_zero(e.bytes, end, s.offset, &format!("{who}: before section {i} ({})", e.names[i]))?;
        end = s.offset + s.size;
    }
    let want = align_up(end, 8);
    if e.ehdr.shoff != want { return Err(format!("{who}: e_shoff {:#x}, layout rule gives {want:#x}", e.ehdr.shoff)) }
    require_zero(e.bytes, end, e.ehdr.shoff, &format!("{who}: before the section header table"))?;
    let total = e.ehdr.shoff + e.ehdr.shnum * e.ehdr.shentsize;
    if e.bytes.len() as u64 != total { return Err(format!("{who}: file is {:#x} bytes, header table ends at {total:#x}", e.bytes.len())) }
    Ok(())
}

/// Compare a native code object with the oracle's. Normalization (the ONLY
/// differences tolerated):
///  1. `.comment` section CONTENT: the oracle's is only checked to be
///     non-empty and NUL-terminated; the native one must be exactly
///     `native_comment()` (`hipfire peacemaker native-emit <version>\0`).
///  2. `.comment` `sh_size` (and nothing else of its header).
///  3. Layout fields derived from that size: `e_shoff` and the `sh_offset`
///     of every section after `.comment`; each is not ignored but validated
///     in both files against the layout rule of `check_layout` (aligned
///     offsets chained from the end of `.comment`, zero padding, header table
///     last). Their section sizes, contents and padding stay exact.
/// Everything else is exact: the ELF header (all fields and `e_ident`) minus
/// `e_shoff`, every program header field, every section header field minus
/// the above, section names, all bytes before `.comment` (program headers,
/// section contents, padding) at identical offsets, and the contents of every
/// section after `.comment`.
pub fn compare_code_objects(native: &[u8], oracle: &[u8]) -> Result<(), String> {
    let n = Elf::parse(native).map_err(|e| format!("native: {e}"))?;
    let o = Elf::parse(oracle).map_err(|e| format!("oracle: {e}"))?;
    let mut expect = o.ehdr.clone();
    expect.shoff = n.ehdr.shoff; // derived: validated by check_layout
    if n.ehdr != expect { return Err(format!("ELF header differs: native {:?} oracle {:?}", n.ehdr, o.ehdr)) }
    if n.phdrs != o.phdrs {
        let i = n.phdrs.iter().zip(&o.phdrs).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!("program header {i} differs: native {:?} oracle {:?}", n.phdrs.get(i), o.phdrs.get(i)));
    }
    if n.names != o.names { return Err(format!("section names differ: native {:?} oracle {:?}", n.names, o.names)) }
    let c = n.comment_index()?;
    check_layout(&n, c, "native")?;
    check_layout(&o, c, "oracle")?;
    for i in 0..n.shdrs.len() {
        let mut expect = o.shdrs[i].clone();
        if i == c { expect.size = n.shdrs[i].size }
        if i > c { expect.offset = n.shdrs[i].offset } // derived: validated by check_layout
        if n.shdrs[i] != expect {
            return Err(format!("section header {i} ({}) differs: native {:?} oracle {:?}", n.names[i], n.shdrs[i], o.shdrs[i]));
        }
    }
    if n.section(c)? != native_comment() { return Err(format!(".comment is {:?}, not the native-emit stamp", String::from_utf8_lossy(n.section(c)?))) }
    if o.section(c)?.last() != Some(&0) { return Err("oracle .comment is empty or not NUL-terminated".into()) }
    // ELF header bytes (minus e_shoff), then everything up to .comment, which
    // sits at the same offset in both files.
    let cut = n.shdrs[c].offset as usize;
    if cut < 64 { return Err(format!(".comment at {cut:#x} inside the ELF header")) }
    let header = |b: &[u8]| { let mut h = b[..64].to_vec(); h[0x28..0x30].fill(0); h };
    if header(native) != header(oracle) { return Err("ELF header bytes differ".into()) }
    if let Some(at) = first_difference(&native[64..cut], &oracle[64..cut]) {
        return Err(format!("bytes before .comment differ at {:#x} ({})", 64 + at, section_at(oracle, 64 + at)));
    }
    for i in c + 1..n.shdrs.len() {
        if let Some(at) = first_difference(n.section(i)?, o.section(i)?) {
            return Err(format!("section {i} ({}) differs at +{at:#x}", n.names[i]));
        }
    }
    Ok(())
}

struct Entry { offset: u64, size: u64, size_at: usize, triple: Vec<u8> }
struct Bundle<'a> { bytes: &'a [u8], entries: Vec<Entry>, header_end: u64 }

impl<'a> Bundle<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes_at(bytes, 0, 24, "bundle magic")? != BUNDLE_MAGIC { return Err("not a __CLANG_OFFLOAD_BUNDLE__".into()) }
        let count = le(bytes, 24, 8, "bundle entry count")?;
        let mut at = 32u64;
        let mut entries = Vec::new();
        for _ in 0..count {
            let (offset, size, len) = (le(bytes, at, 8, "bundle entry")?, le(bytes, at + 8, 8, "bundle entry")?, le(bytes, at + 16, 8, "bundle entry")?);
            entries.push(Entry { offset, size, size_at: at as usize + 8, triple: bytes_at(bytes, at + 24, len, "bundle triple")?.to_vec() });
            at += 24 + len;
        }
        Ok(Bundle { bytes, entries, header_end: at })
    }
    fn payload(&self, i: usize) -> Result<&'a [u8], String> {
        bytes_at(self.bytes, self.entries[i].offset, self.entries[i].size, &format!("bundle entry {i}"))
    }
    /// The device entry (the single `hipv4-amdgcn-amd-amdhsa--<arch>` one).
    fn device_index(&self) -> Result<usize, String> {
        let mut found = self.entries.iter().enumerate().filter(|(_, e)| e.triple.starts_with(DEVICE_TRIPLE_PREFIX)).map(|(i, _)| i);
        match (found.next(), found.next()) {
            (Some(i), None) => Ok(i),
            _ => Err("expected exactly one hipv4-amdgcn device entry".into()),
        }
    }
    /// Each entry starts at the next `BUNDLE_ALIGN` boundary after the header
    /// or the previous payload, zero padded; the last payload ends the file;
    /// the device payload is `elf`.
    fn check_layout(&self, d: usize, elf: &[u8], who: &str) -> Result<(), String> {
        let mut end = self.header_end;
        for (i, e) in self.entries.iter().enumerate() {
            let want = align_up(end, BUNDLE_ALIGN);
            if e.offset != want { return Err(format!("{who}: bundle entry {i} at {:#x}, layout rule gives {want:#x}", e.offset)) }
            require_zero(self.bytes, end, e.offset, &format!("{who}: before bundle entry {i}"))?;
            end = e.offset + e.size;
        }
        if self.bytes.len() as u64 != end { return Err(format!("{who}: bundle is {:#x} bytes, last payload ends at {end:#x}", self.bytes.len())) }
        if self.payload(d)? != elf { return Err(format!("{who}: bundle device payload is not the code object")) }
        Ok(())
    }
}

/// Compare a native bundle with the oracle's, given each one's code object.
/// Normalization: the device entry's size field and the offsets of entries
/// after it follow the code object's length, so they are validated against
/// the layout rule (each entry at the next 4096 boundary, zero padding, no
/// trailing bytes, device payload == that file's own code object) instead of
/// compared. Everything else is exact: magic, entry count, every triple, the
/// header bytes, the sizes and payloads of the non-device entries, and the
/// offsets of every entry up to the device one. The code objects themselves
/// are compared by `compare_code_objects`.
pub fn compare_bundles(native: &[u8], native_elf: &[u8], oracle: &[u8], oracle_elf: &[u8]) -> Result<(), String> {
    let n = Bundle::parse(native).map_err(|e| format!("native: {e}"))?;
    let o = Bundle::parse(oracle).map_err(|e| format!("oracle: {e}"))?;
    if n.entries.len() != o.entries.len() { return Err(format!("entry count: native {} oracle {}", n.entries.len(), o.entries.len())) }
    let d = n.device_index().map_err(|e| format!("native: {e}"))?;
    if o.device_index().map_err(|e| format!("oracle: {e}"))? != d { return Err("device entry position differs".into()) }
    n.check_layout(d, native_elf, "native")?;
    o.check_layout(d, oracle_elf, "oracle")?;
    for (i, (a, b)) in n.entries.iter().zip(&o.entries).enumerate() {
        if a.triple != b.triple { return Err(format!("bundle entry {i}: triple native {:?} oracle {:?}", String::from_utf8_lossy(&a.triple), String::from_utf8_lossy(&b.triple))) }
        if i != d && a.size != b.size { return Err(format!("bundle entry {i}: size native {:#x} oracle {:#x}", a.size, b.size)) }
        if i <= d && a.offset != b.offset { return Err(format!("bundle entry {i}: offset native {:#x} oracle {:#x}", a.offset, b.offset)) }
        if i != d && n.payload(i)? != o.payload(i)? { return Err(format!("bundle entry {i}: payload differs")) }
    }
    if n.header_end != o.header_end { return Err("bundle header length differs".into()) }
    let header = |b: &Bundle| {
        let mut h = b.bytes[..b.header_end as usize].to_vec();
        let at = b.entries[d].size_at;
        h[at..at + 8].fill(0); // device size: validated by check_layout
        for e in &b.entries[d + 1..] { h[e.size_at - 8..e.size_at].fill(0) } // later offsets: validated by check_layout
        h
    };
    if header(&n) != header(&o) { return Err("bundle header bytes differ".into()) }
    Ok(())
}

/// Default native-vs-oracle verdict for one code object and its bundle: `None`
/// when `compare_code_objects` and `compare_bundles` both accept, else the
/// first mismatch. The whole-file comparison is `strict_whole_file_difference`.
pub fn compare_native_to_oracle(native_elf: &[u8], native_bundle: &[u8], oracle_elf: &[u8], oracle_bundle: &[u8]) -> Option<String> {
    match (compare_code_objects(native_elf, oracle_elf), compare_bundles(native_bundle, native_elf, oracle_bundle, oracle_elf)) {
        (Ok(()), Ok(())) => None,
        (Err(why), _) => Some(format!("code object differs: {why}")),
        (Ok(()), Err(why)) => Some(format!("bundle differs: {why}")),
    }
}
