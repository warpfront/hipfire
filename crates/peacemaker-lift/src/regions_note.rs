//! Reader for the `.note.pm_regions` PM region-table metadata section.
//!
//! The section is `SHT_PROGBITS`, `sh_flags == 0` (non-alloc), with **no ELF note framing**.
//! Its payload is a concatenation of kernel records (no padding) running to the end of the
//! section. All integers are little-endian; every string is a `u32` byte length followed by
//! UTF-8 bytes. A kernel with no tagged regions has no record, and an object with none has no
//! section at all.
//!
//! Kernel record:
//!
//! | field | encoding |
//! |---|---|
//! | magic | `b"PMRG"` |
//! | version | `u32` = 1 |
//! | kernel symbol | string (the kernel code symbol, not `.kd`) |
//! | count | `u32` rows |
//! | dispatcher | eight `u16`: `lo, hi, tmpl, ka_off, idx, kernargs_pair_low, wg_x, wg_yz` |
//! | rows | `count` rows, strictly ascending by explicit `template_id` |
//!
//! Row:
//!
//! | field | encoding |
//! |---|---|
//! | `template_id` | `u32` |
//! | `entry_pc` | `u64`, byte offset relative to the kernel code symbol |
//! | `exit_pc` | `u64`, exclusive, same base |
//! | prefix | string |
//! | `wg_x`, `wg_yz` | `u16`, `u16` |
//! | `block` | `u32` |
//! | `dynamic_lds` | `u32` |
//! | digest | 32 bytes (SHA-256) |
//! | source symbol | string |
//!
//! A native PM row has an empty prefix, `wg_x == wg_yz == u16::MAX`, `block == dynamic_lds == 0`,
//! a zero digest and an empty source symbol.
//!
//! Ranges must be non-empty, `entry_pc`/`exit_pc` dword (4-byte) aligned, and rows of one kernel
//! must not overlap. Anything malformed, truncated, with trailing bytes, an unsupported version,
//! or duplicate kernel symbols / template ids is rejected.

use object::read::elf::{ElfFile64, SectionHeader};
use object::{Endianness, Object, ObjectSection, ObjectSymbol};

/// Section name carrying the region tables.
pub const SECTION_NAME: &str = ".note.pm_regions";
/// Record magic.
pub const MAGIC: &[u8; 4] = b"PMRG";
/// The only supported record version.
pub const VERSION: u32 = 1;

/// Dispatcher register/offset values of one kernel (eight `u16`, in wire order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatcherRegs {
    pub lo: u16,
    pub hi: u16,
    pub tmpl: u16,
    pub ka_off: u16,
    pub idx: u16,
    pub kernargs_pair_low: u16,
    pub wg_x: u16,
    pub wg_yz: u16,
}

/// One region row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub template_id: u32,
    /// Byte offset of the first instruction, relative to the kernel code symbol.
    pub entry_pc: u64,
    /// Exclusive end byte offset, relative to the kernel code symbol.
    pub exit_pc: u64,
    pub prefix: String,
    pub wg_x: u16,
    pub wg_yz: u16,
    pub block: u32,
    pub dynamic_lds: u32,
    pub digest: [u8; 32],
    pub source_symbol: String,
}

impl Region {
    /// True for a native PM row (empty prefix, `u16::MAX` workgroup, zero block/LDS/digest/symbol).
    pub fn is_native(&self) -> bool {
        self.prefix.is_empty()
            && self.wg_x == u16::MAX
            && self.wg_yz == u16::MAX
            && self.block == 0
            && self.dynamic_lds == 0
            && self.digest == [0; 32]
            && self.source_symbol.is_empty()
    }
}

/// All regions of one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRegions {
    pub kernel: String,
    pub dispatcher: DispatcherRegs,
    /// Ascending by `template_id`.
    pub regions: Vec<Region>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let rest = self.buf.len() - self.pos;
        if n > rest {
            return Err(format!(
                "truncated {what} at byte {}: need {n}, have {rest}",
                self.pos
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u16(&mut self, what: &str) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn string(&mut self, what: &str) -> Result<String, String> {
        let n = self.u32(what)? as usize;
        let bytes = self.take(n, what)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| format!("{what} is not valid UTF-8"))
    }
}

/// Smallest possible encoded row: id + entry + exit + prefix len + wg + block + lds + digest + sym len.
const MIN_ROW_BYTES: usize = 4 + 8 + 8 + 4 + 2 + 2 + 4 + 4 + 32 + 4;

/// Parses a raw section payload (zero or more kernel records to the end of `payload`).
pub fn parse_regions(payload: &[u8]) -> Result<Vec<KernelRegions>, String> {
    let mut c = Cursor {
        buf: payload,
        pos: 0,
    };
    let mut out: Vec<KernelRegions> = Vec::new();
    while c.pos < payload.len() {
        if c.take(4, "magic")? != MAGIC {
            return Err(format!("bad magic at byte {}", c.pos - 4));
        }
        let version = c.u32("version")?;
        if version != VERSION {
            return Err(format!("unsupported version {version}"));
        }
        let kernel = c.string("kernel symbol")?;
        if out.iter().any(|k| k.kernel == kernel) {
            return Err(format!("duplicate kernel symbol {kernel:?}"));
        }
        let count = c.u32("row count")? as usize;
        let dispatcher = DispatcherRegs {
            lo: c.u16("dispatcher")?,
            hi: c.u16("dispatcher")?,
            tmpl: c.u16("dispatcher")?,
            ka_off: c.u16("dispatcher")?,
            idx: c.u16("dispatcher")?,
            kernargs_pair_low: c.u16("dispatcher")?,
            wg_x: c.u16("dispatcher")?,
            wg_yz: c.u16("dispatcher")?,
        };
        let remaining = payload.len() - c.pos;
        if count > remaining / MIN_ROW_BYTES {
            return Err(format!(
                "row count {count} for {kernel:?} exceeds remaining {remaining} bytes"
            ));
        }
        let mut regions: Vec<Region> = Vec::with_capacity(count);
        for _ in 0..count {
            let template_id = c.u32("template_id")?;
            let entry_pc = c.u64("entry_pc")?;
            let exit_pc = c.u64("exit_pc")?;
            let prefix = c.string("prefix")?;
            let wg_x = c.u16("wg_x")?;
            let wg_yz = c.u16("wg_yz")?;
            let block = c.u32("block")?;
            let dynamic_lds = c.u32("dynamic_lds")?;
            let digest: [u8; 32] = c.take(32, "digest")?.try_into().unwrap();
            let source_symbol = c.string("source symbol")?;
            if let Some(prev) = regions.last() {
                if template_id == prev.template_id {
                    return Err(format!("duplicate template id {template_id} in {kernel:?}"));
                }
                if template_id < prev.template_id {
                    return Err(format!(
                        "template ids not ascending in {kernel:?}: {} then {template_id}",
                        prev.template_id
                    ));
                }
            }
            if entry_pc >= exit_pc {
                return Err(format!(
                    "template {template_id} in {kernel:?}: empty or inverted range {entry_pc:#x}..{exit_pc:#x}"
                ));
            }
            if entry_pc % 4 != 0 || exit_pc % 4 != 0 {
                return Err(format!(
                    "template {template_id} in {kernel:?}: range {entry_pc:#x}..{exit_pc:#x} not dword aligned"
                ));
            }
            regions.push(Region {
                template_id,
                entry_pc,
                exit_pc,
                prefix,
                wg_x,
                wg_yz,
                block,
                dynamic_lds,
                digest,
                source_symbol,
            });
        }
        let mut spans: Vec<(u64, u64, u32)> = regions
            .iter()
            .map(|r| (r.entry_pc, r.exit_pc, r.template_id))
            .collect();
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(format!(
                    "templates {} and {} overlap in {kernel:?}",
                    w[0].2, w[1].2
                ));
            }
        }
        out.push(KernelRegions {
            kernel,
            dispatcher,
            regions,
        });
    }
    Ok(out)
}

/// Reads `.note.pm_regions` from an ELF64 object and validates every range against the size of
/// its kernel code symbol. An absent section yields an empty vector.
pub fn read_regions(elf: &[u8]) -> Result<Vec<KernelRegions>, String> {
    let file = ElfFile64::<Endianness>::parse(elf).map_err(|e| format!("ELF parse: {e}"))?;
    let endian = file.endian();
    let mut found = None;
    for section in file.sections() {
        if section.name().map_err(|e| format!("section name: {e}"))? != SECTION_NAME {
            continue;
        }
        if found.is_some() {
            return Err(format!("duplicate {SECTION_NAME} sections"));
        }
        let h = section.elf_section_header();
        if h.sh_type(endian) != object::elf::SHT_PROGBITS {
            return Err(format!("{SECTION_NAME} is not SHT_PROGBITS"));
        }
        if h.sh_flags(endian) != 0 {
            return Err(format!(
                "{SECTION_NAME} has non-zero sh_flags (must be non-alloc)"
            ));
        }
        found = Some(section);
    }
    let Some(section) = found else {
        return Ok(Vec::new());
    };
    let data = section
        .data()
        .map_err(|e| format!("{SECTION_NAME} data: {e}"))?;
    let kernels = parse_regions(data)?;
    for k in &kernels {
        let mut size = None;
        for sym in file.symbols() {
            if sym.name().map_err(|e| format!("symbol name: {e}"))? != k.kernel {
                continue;
            }
            if sym.is_undefined() {
                continue;
            }
            if sym.kind() != object::SymbolKind::Text {
                return Err(format!(
                    "kernel symbol {:?} is not a code (Text) symbol",
                    k.kernel
                ));
            }
            let idx = sym
                .section_index()
                .ok_or_else(|| format!("kernel symbol {:?} has no defining section", k.kernel))?;
            let sec = file
                .section_by_index(idx)
                .map_err(|e| format!("kernel symbol {:?} section: {e}", k.kernel))?;
            let flags: u64 = sec.elf_section_header().sh_flags(endian);
            if flags & u64::from(object::elf::SHF_EXECINSTR) == 0 {
                return Err(format!(
                    "kernel symbol {:?} is not in an executable section",
                    k.kernel
                ));
            }
            let end = sym.address().checked_add(sym.size());
            let sec_end = sec.address().checked_add(sec.size());
            match (end, sec_end) {
                (Some(e), Some(se)) if sym.address() >= sec.address() && e <= se => {}
                _ => {
                    return Err(format!(
                        "kernel symbol {:?} [{:#x}+{:#x}] lies outside its section",
                        k.kernel,
                        sym.address(),
                        sym.size()
                    ))
                }
            }
            if size.is_some() {
                return Err(format!(
                    "kernel symbol {:?} is defined more than once",
                    k.kernel
                ));
            }
            size = Some(sym.size());
        }
        let size = size.ok_or_else(|| format!("kernel symbol {:?} not found", k.kernel))?;
        for r in &k.regions {
            if r.exit_pc > size {
                return Err(format!(
                    "template {} in {:?}: exit_pc {:#x} exceeds kernel code size {size:#x}",
                    r.template_id, k.kernel, r.exit_pc
                ));
            }
        }
    }
    Ok(kernels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::write::{Object as WObject, Symbol, SymbolSection};
    use object::{Architecture, BinaryFormat, SectionKind, SymbolFlags, SymbolKind, SymbolScope};

    fn put_str(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u32).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }

    fn row(v: &mut Vec<u8>, r: &Region) {
        v.extend_from_slice(&r.template_id.to_le_bytes());
        v.extend_from_slice(&r.entry_pc.to_le_bytes());
        v.extend_from_slice(&r.exit_pc.to_le_bytes());
        put_str(v, &r.prefix);
        v.extend_from_slice(&r.wg_x.to_le_bytes());
        v.extend_from_slice(&r.wg_yz.to_le_bytes());
        v.extend_from_slice(&r.block.to_le_bytes());
        v.extend_from_slice(&r.dynamic_lds.to_le_bytes());
        v.extend_from_slice(&r.digest);
        put_str(v, &r.source_symbol);
    }

    fn disp() -> DispatcherRegs {
        DispatcherRegs {
            lo: 1,
            hi: 2,
            tmpl: 3,
            ka_off: 4,
            idx: 5,
            kernargs_pair_low: 6,
            wg_x: 7,
            wg_yz: 8,
        }
    }

    fn encode_with(version: u32, k: &KernelRegions) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&version.to_le_bytes());
        put_str(&mut v, &k.kernel);
        v.extend_from_slice(&(k.regions.len() as u32).to_le_bytes());
        let d = k.dispatcher;
        for x in [
            d.lo,
            d.hi,
            d.tmpl,
            d.ka_off,
            d.idx,
            d.kernargs_pair_low,
            d.wg_x,
            d.wg_yz,
        ] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for r in &k.regions {
            row(&mut v, r);
        }
        v
    }

    fn encode(k: &KernelRegions) -> Vec<u8> {
        encode_with(VERSION, k)
    }

    fn native(id: u32, entry: u64, exit: u64) -> Region {
        Region {
            template_id: id,
            entry_pc: entry,
            exit_pc: exit,
            prefix: String::new(),
            wg_x: u16::MAX,
            wg_yz: u16::MAX,
            block: 0,
            dynamic_lds: 0,
            digest: [0; 32],
            source_symbol: String::new(),
        }
    }

    fn lifted(id: u32, entry: u64, exit: u64) -> Region {
        Region {
            template_id: id,
            entry_pc: entry,
            exit_pc: exit,
            prefix: "pfx".into(),
            wg_x: 64,
            wg_yz: 1,
            block: 0x0102_0304,
            dynamic_lds: 0x1000,
            digest: core::array::from_fn(|i| i as u8),
            source_symbol: "src_k".into(),
        }
    }

    fn kernel(name: &str, regions: Vec<Region>) -> KernelRegions {
        KernelRegions {
            kernel: name.into(),
            dispatcher: disp(),
            regions,
        }
    }

    fn object_with(sections: &[(&[u8], SectionKind, &[u8])], syms: &[(&str, u64)]) -> Vec<u8> {
        let mut o = WObject::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let text = o.add_section(vec![], b".text".to_vec(), SectionKind::Text);
        o.append_section_data(text, &[0u8; 256], 4);
        for (name, size) in syms {
            o.add_symbol(Symbol {
                name: name.as_bytes().to_vec(),
                value: 0,
                size: *size,
                kind: SymbolKind::Text,
                scope: SymbolScope::Dynamic,
                weak: false,
                section: SymbolSection::Section(text),
                flags: SymbolFlags::None,
            });
        }
        for (name, kind, data) in sections {
            let s = o.add_section(vec![], name.to_vec(), *kind);
            o.append_section_data(s, data, 1);
        }
        o.write().unwrap()
    }

    fn note_object(payload: &[u8], syms: &[(&str, u64)]) -> Vec<u8> {
        object_with(
            &[(SECTION_NAME.as_bytes(), SectionKind::Metadata, payload)],
            syms,
        )
    }

    #[test]
    fn golden_native_and_lifted() {
        let k = kernel("kern", vec![native(0, 0, 8), lifted(5, 16, 64)]);
        let mut g = Vec::new();
        g.extend_from_slice(b"PMRG");
        g.extend_from_slice(&[1, 0, 0, 0]);
        g.extend_from_slice(&[4, 0, 0, 0]);
        g.extend_from_slice(b"kern");
        g.extend_from_slice(&[2, 0, 0, 0]);
        for x in 1u16..=8 {
            g.extend_from_slice(&[x as u8, 0]);
        }
        // native row
        g.extend_from_slice(&[0; 4]);
        g.extend_from_slice(&[0; 8]);
        g.extend_from_slice(&[8, 0, 0, 0, 0, 0, 0, 0]);
        g.extend_from_slice(&[0; 4]);
        g.extend_from_slice(&[0xff; 4]);
        g.extend_from_slice(&[0; 8]);
        g.extend_from_slice(&[0; 32]);
        g.extend_from_slice(&[0; 4]);
        // lifted row
        g.extend_from_slice(&[5, 0, 0, 0]);
        g.extend_from_slice(&[16, 0, 0, 0, 0, 0, 0, 0]);
        g.extend_from_slice(&[64, 0, 0, 0, 0, 0, 0, 0]);
        g.extend_from_slice(&[3, 0, 0, 0]);
        g.extend_from_slice(b"pfx");
        g.extend_from_slice(&[64, 0, 1, 0]);
        g.extend_from_slice(&[4, 3, 2, 1]);
        g.extend_from_slice(&[0, 0x10, 0, 0]);
        g.extend((0u8..32).collect::<Vec<_>>());
        g.extend_from_slice(&[5, 0, 0, 0]);
        g.extend_from_slice(b"src_k");
        assert_eq!(encode(&k), g);
        let parsed = parse_regions(&g).unwrap();
        assert_eq!(parsed, vec![k]);
        assert!(parsed[0].regions[0].is_native());
        assert!(!parsed[0].regions[1].is_native());
    }

    #[test]
    fn empty_payload_is_empty() {
        assert_eq!(parse_regions(&[]).unwrap(), vec![]);
    }

    #[test]
    fn every_truncation_fails() {
        let mut p = encode(&kernel("kern", vec![native(0, 0, 8), lifted(5, 16, 64)]));
        p.extend(encode(&kernel("other", vec![native(1, 4, 8)])));
        let first = encode(&kernel("kern", vec![native(0, 0, 8), lifted(5, 16, 64)])).len();
        for n in 1..p.len() {
            if n == first {
                assert!(parse_regions(&p[..n]).is_ok());
            } else {
                assert!(parse_regions(&p[..n]).is_err(), "prefix {n} accepted");
            }
        }
        assert_eq!(parse_regions(&p).unwrap().len(), 2);
    }

    #[test]
    fn trailing_bytes_fail() {
        let mut p = encode(&kernel("kern", vec![native(0, 0, 8)]));
        p.push(0);
        assert!(parse_regions(&p).is_err());
    }

    #[test]
    fn bad_magic_and_version() {
        let k = kernel("kern", vec![native(0, 0, 8)]);
        let mut p = encode(&k);
        p[0] = b'X';
        assert!(parse_regions(&p).unwrap_err().contains("magic"));
        let err = parse_regions(&encode_with(2, &k)).unwrap_err();
        assert!(err.contains("unsupported version 2"), "{err}");
    }

    #[test]
    fn huge_count_rejected_without_alloc() {
        let mut p = encode(&kernel("kern", vec![]));
        let off = 4 + 4 + 4 + 4;
        p[off..off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_regions(&p).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn duplicate_kernel_and_template() {
        let mut p = encode(&kernel("kern", vec![native(0, 0, 8)]));
        p.extend(encode(&kernel("kern", vec![native(0, 0, 8)])));
        assert!(parse_regions(&p).unwrap_err().contains("duplicate kernel"));
        let k = kernel("kern", vec![native(3, 0, 8), native(3, 8, 16)]);
        assert!(parse_regions(&encode(&k))
            .unwrap_err()
            .contains("duplicate template"));
    }

    #[test]
    fn non_ascending_templates() {
        let k = kernel("kern", vec![native(4, 0, 8), native(3, 8, 16)]);
        assert!(parse_regions(&encode(&k))
            .unwrap_err()
            .contains("ascending"));
    }

    #[test]
    fn bad_ranges() {
        for (e, x, what) in [
            (8, 8, "empty"),
            (16, 8, "empty"),
            (2, 8, "aligned"),
            (0, 10, "aligned"),
        ] {
            let k = kernel("kern", vec![native(0, e, x)]);
            let err = parse_regions(&encode(&k)).unwrap_err();
            assert!(err.contains(what), "{e}..{x}: {err}");
        }
    }

    #[test]
    fn overlap_fails_adjacent_ok() {
        let k = kernel("kern", vec![native(0, 0, 16), native(1, 8, 24)]);
        assert!(parse_regions(&encode(&k)).unwrap_err().contains("overlap"));
        let k = kernel("kern", vec![native(0, 16, 32), native(1, 0, 16)]);
        assert!(parse_regions(&encode(&k)).is_ok());
    }

    #[test]
    fn invalid_utf8_fails() {
        let mut p = encode(&kernel("kern", vec![]));
        p[12] = 0xff;
        assert!(parse_regions(&p).unwrap_err().contains("UTF-8"));
    }

    #[test]
    fn absent_section_is_empty() {
        let obj = object_with(&[], &[("kern", 64)]);
        assert_eq!(read_regions(&obj).unwrap(), vec![]);
    }

    #[test]
    fn read_object_ok_and_exact_fit() {
        let k = kernel("kern", vec![native(0, 0, 8), lifted(5, 16, 64)]);
        let obj = note_object(&encode(&k), &[("kern", 64)]);
        assert_eq!(read_regions(&obj).unwrap(), vec![k]);
    }

    #[test]
    fn read_object_out_of_range() {
        let k = kernel("kern", vec![native(0, 0, 68)]);
        let obj = note_object(&encode(&k), &[("kern", 64)]);
        assert!(read_regions(&obj).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn read_object_missing_or_duplicate_symbol() {
        let p = encode(&kernel("kern", vec![native(0, 0, 8)]));
        let err = read_regions(&note_object(&p, &[("other", 64)])).unwrap_err();
        assert!(err.contains("not found"), "{err}");
        let err = read_regions(&note_object(&p, &[("kern", 64), ("kern", 64)])).unwrap_err();
        assert!(err.contains("more than once"), "{err}");
    }

    #[test]
    fn read_object_bad_sections() {
        let p = encode(&kernel("kern", vec![native(0, 0, 8)]));
        let n = SECTION_NAME.as_bytes();
        let obj = object_with(
            &[
                (n, SectionKind::Metadata, &p),
                (n, SectionKind::Metadata, &p),
            ],
            &[("kern", 64)],
        );
        assert!(read_regions(&obj).unwrap_err().contains("duplicate"));
        let obj = object_with(&[(n, SectionKind::ReadOnlyData, &p)], &[("kern", 64)]);
        assert!(read_regions(&obj).unwrap_err().contains("sh_flags"));
    }

    #[test]
    fn read_object_malformed_payload_and_non_elf() {
        let obj = note_object(b"PMRG", &[("kern", 64)]);
        assert!(read_regions(&obj).is_err());
        assert!(read_regions(b"not an elf").is_err());
    }
}
