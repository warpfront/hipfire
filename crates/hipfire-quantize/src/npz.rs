// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal read-only NPZ reader (no dependencies beyond `std`).
//!
//! Scope is exactly what the SubQ4 calibration records need: an NPZ is a ZIP
//! archive of `.npy` members, written by `numpy.savez` (method 0, "stored").
//! The reader
//!
//! * locates the End-Of-Central-Directory record (ZIP64 EOCD when the 32-bit
//!   fields are saturated) and walks the central directory, honouring the ZIP64
//!   extended-information extra field (numpy writes ZIP64 local headers),
//! * accepts only compression method 0 and checks every member's CRC-32,
//! * parses `.npy` v1/v2/v3 headers (`descr`, `fortran_order`, `shape`) and
//!   decodes `|u1`, `<f4`, `<i8` and `<U{n}` (UTF-32LE) arrays and scalars.
//!
//! Anything else is a typed [`NpzError`]; nothing is silently converted.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// Failure reading an NPZ archive or one of its `.npy` members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NpzError {
    /// Reading the file failed.
    Io(String),
    /// No End-Of-Central-Directory record: not a ZIP archive.
    NotZip,
    /// The archive ends inside a structure.
    Truncated(&'static str),
    /// A ZIP structure carries the wrong signature.
    BadSignature { what: &'static str, found: u32 },
    /// A ZIP64 field required by a saturated 32-bit field is missing or short.
    Zip64Missing(&'static str),
    /// Multi-disk archives are not supported.
    MultiDisk,
    /// The member is encrypted.
    Encrypted(String),
    /// Compression method other than 0 (stored).
    UnsupportedCompression { entry: String, method: u16 },
    /// Central directory and local header disagree, or sizes are inconsistent.
    Inconsistent { entry: String, why: &'static str },
    /// A member's span lies outside the archive.
    OutOfBounds { entry: String },
    /// The same member name appears twice.
    DuplicateEntry(String),
    /// CRC-32 of the stored bytes differs from the central directory.
    CrcMismatch {
        entry: String,
        expected: u32,
        got: u32,
    },
    /// The `.npy` payload is malformed.
    BadNpy { entry: String, why: String },
    /// `fortran_order: True` (column-major) is rejected.
    FortranOrder { entry: String },
    /// A dtype outside `|u1`, `<f4`, `<i8`, `<U{n}`.
    UnsupportedDescr { entry: String, descr: String },
    /// A requested member is absent.
    MissingEntry(String),
    /// A member has a different dtype than requested.
    WrongDtype {
        entry: String,
        expected: &'static str,
        found: &'static str,
    },
    /// A member has a different shape than requested.
    ShapeMismatch {
        entry: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "npz: i/o error: {e}"),
            Self::NotZip => write!(f, "npz: not a ZIP archive (no end-of-central-directory)"),
            Self::Truncated(w) => write!(f, "npz: archive truncated inside {w}"),
            Self::BadSignature { what, found } => {
                write!(f, "npz: bad {what} signature {found:#010x}")
            }
            Self::Zip64Missing(w) => write!(f, "npz: ZIP64 data missing for {w}"),
            Self::MultiDisk => write!(f, "npz: multi-disk archives are not supported"),
            Self::Encrypted(e) => write!(f, "npz: member `{e}` is encrypted"),
            Self::UnsupportedCompression { entry, method } => write!(
                f,
                "npz: member `{entry}` uses compression method {method}; only 0 (stored) is accepted"
            ),
            Self::Inconsistent { entry, why } => write!(f, "npz: member `{entry}`: {why}"),
            Self::OutOfBounds { entry } => {
                write!(f, "npz: member `{entry}` lies outside the archive")
            }
            Self::DuplicateEntry(e) => write!(f, "npz: duplicate member `{e}`"),
            Self::CrcMismatch {
                entry,
                expected,
                got,
            } => write!(
                f,
                "npz: member `{entry}` CRC-32 {got:#010x} != recorded {expected:#010x}"
            ),
            Self::BadNpy { entry, why } => write!(f, "npz: member `{entry}`: bad .npy: {why}"),
            Self::FortranOrder { entry } => write!(
                f,
                "npz: member `{entry}` is fortran_order (column-major); rejected"
            ),
            Self::UnsupportedDescr { entry, descr } => write!(
                f,
                "npz: member `{entry}` has unsupported dtype `{descr}` (accepted: |u1, <f4, <i8, <U<n>)"
            ),
            Self::MissingEntry(e) => write!(f, "npz: missing member `{e}`"),
            Self::WrongDtype {
                entry,
                expected,
                found,
            } => write!(f, "npz: member `{entry}` is {found}, expected {expected}"),
            Self::ShapeMismatch {
                entry,
                expected,
                found,
            } => write!(
                f,
                "npz: member `{entry}` has shape {found:?}, expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for NpzError {}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320), table driven.
// ---------------------------------------------------------------------------

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

/// CRC-32 (zlib/ZIP flavour) of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC32_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

// ---------------------------------------------------------------------------
// ZIP container
// ---------------------------------------------------------------------------

const SIG_EOCD: u32 = 0x0605_4b50;
const SIG_EOCD64: u32 = 0x0606_4b50;
const SIG_EOCD64_LOC: u32 = 0x0706_4b50;
const SIG_CDIR: u32 = 0x0201_4b50;
const SIG_LOCAL: u32 = 0x0403_4b50;

fn rd_u16(b: &[u8], at: usize, what: &'static str) -> Result<u16, NpzError> {
    b.get(at..at.wrapping_add(2))
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or(NpzError::Truncated(what))
}

fn rd_u32(b: &[u8], at: usize, what: &'static str) -> Result<u32, NpzError> {
    b.get(at..at.wrapping_add(4))
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(NpzError::Truncated(what))
}

fn rd_u64(b: &[u8], at: usize, what: &'static str) -> Result<u64, NpzError> {
    b.get(at..at.wrapping_add(8))
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or(NpzError::Truncated(what))
}

fn to_usize(v: u64, what: &'static str) -> Result<usize, NpzError> {
    usize::try_from(v).map_err(|_| NpzError::Truncated(what))
}

struct CentralEntry {
    name: String,
    method: u16,
    flags: u16,
    crc: u32,
    csize: u64,
    usize_: u64,
    local_offset: u64,
}

struct Eocd {
    entries: u64,
    cd_size: u64,
    cd_offset: u64,
}

fn find_eocd(bytes: &[u8]) -> Result<Eocd, NpzError> {
    // The EOCD is 22 bytes plus a comment of up to 65535 bytes.
    if bytes.len() < 22 {
        return Err(NpzError::NotZip);
    }
    let lowest = bytes.len().saturating_sub(22 + 65535);
    let mut pos = bytes.len() - 22;
    let at = loop {
        if rd_u32(bytes, pos, "end of central directory")? == SIG_EOCD {
            // The comment length must reach exactly to the end of the file.
            let comment = usize::from(rd_u16(bytes, pos + 20, "end of central directory")?);
            if pos + 22 + comment == bytes.len() {
                break pos;
            }
        }
        if pos == lowest {
            return Err(NpzError::NotZip);
        }
        pos -= 1;
    };
    let disk = rd_u16(bytes, at + 4, "eocd")?;
    let cd_disk = rd_u16(bytes, at + 6, "eocd")?;
    let entries_disk = rd_u16(bytes, at + 8, "eocd")?;
    let entries = rd_u16(bytes, at + 10, "eocd")?;
    let cd_size = rd_u32(bytes, at + 12, "eocd")?;
    let cd_offset = rd_u32(bytes, at + 16, "eocd")?;
    let saturated = disk == 0xFFFF
        || cd_disk == 0xFFFF
        || entries_disk == 0xFFFF
        || entries == 0xFFFF
        || cd_size == 0xFFFF_FFFF
        || cd_offset == 0xFFFF_FFFF;
    if !saturated {
        if disk != 0 || cd_disk != 0 || entries_disk != entries {
            return Err(NpzError::MultiDisk);
        }
        return Ok(Eocd {
            entries: u64::from(entries),
            cd_size: u64::from(cd_size),
            cd_offset: u64::from(cd_offset),
        });
    }
    // ZIP64: the locator sits immediately before the EOCD.
    let loc = at
        .checked_sub(20)
        .ok_or(NpzError::Zip64Missing("locator"))?;
    let sig = rd_u32(bytes, loc, "zip64 locator")?;
    if sig != SIG_EOCD64_LOC {
        return Err(NpzError::Zip64Missing("locator"));
    }
    let loc_disk = rd_u32(bytes, loc + 4, "zip64 locator")?;
    let eocd64_off = to_usize(
        rd_u64(bytes, loc + 8, "zip64 locator")?,
        "zip64 eocd offset",
    )?;
    let total_disks = rd_u32(bytes, loc + 16, "zip64 locator")?;
    if loc_disk != 0 || total_disks > 1 {
        return Err(NpzError::MultiDisk);
    }
    let sig = rd_u32(bytes, eocd64_off, "zip64 eocd")?;
    if sig != SIG_EOCD64 {
        return Err(NpzError::BadSignature {
            what: "zip64 end of central directory",
            found: sig,
        });
    }
    let disk = rd_u32(bytes, eocd64_off + 16, "zip64 eocd")?;
    let cd_disk = rd_u32(bytes, eocd64_off + 20, "zip64 eocd")?;
    let entries_disk = rd_u64(bytes, eocd64_off + 24, "zip64 eocd")?;
    let entries = rd_u64(bytes, eocd64_off + 32, "zip64 eocd")?;
    let cd_size = rd_u64(bytes, eocd64_off + 40, "zip64 eocd")?;
    let cd_offset = rd_u64(bytes, eocd64_off + 48, "zip64 eocd")?;
    if disk != 0 || cd_disk != 0 || entries_disk != entries {
        return Err(NpzError::MultiDisk);
    }
    Ok(Eocd {
        entries,
        cd_size,
        cd_offset,
    })
}

fn parse_central(bytes: &[u8], eocd: &Eocd) -> Result<Vec<CentralEntry>, NpzError> {
    let start = to_usize(eocd.cd_offset, "central directory offset")?;
    let size = to_usize(eocd.cd_size, "central directory size")?;
    let end = start
        .checked_add(size)
        .filter(|&e| e <= bytes.len())
        .ok_or(NpzError::Truncated("central directory"))?;
    let cd = &bytes[start..end];
    let mut out = Vec::new();
    let mut pos = 0usize;
    for _ in 0..eocd.entries {
        let sig = rd_u32(cd, pos, "central directory header")?;
        if sig != SIG_CDIR {
            return Err(NpzError::BadSignature {
                what: "central directory header",
                found: sig,
            });
        }
        let flags = rd_u16(cd, pos + 8, "central directory header")?;
        let method = rd_u16(cd, pos + 10, "central directory header")?;
        let crc = rd_u32(cd, pos + 16, "central directory header")?;
        let mut csize = u64::from(rd_u32(cd, pos + 20, "central directory header")?);
        let mut usize_ = u64::from(rd_u32(cd, pos + 24, "central directory header")?);
        let name_len = usize::from(rd_u16(cd, pos + 28, "central directory header")?);
        let extra_len = usize::from(rd_u16(cd, pos + 30, "central directory header")?);
        let comment_len = usize::from(rd_u16(cd, pos + 32, "central directory header")?);
        let mut disk_start = u32::from(rd_u16(cd, pos + 34, "central directory header")?);
        let mut local_offset = u64::from(rd_u32(cd, pos + 42, "central directory header")?);
        let name_at = pos + 46;
        let name_bytes = cd
            .get(name_at..name_at + name_len)
            .ok_or(NpzError::Truncated("central directory name"))?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| NpzError::Inconsistent {
            entry: String::from_utf8_lossy(name_bytes).into_owned(),
            why: "member name is not UTF-8",
        })?;
        let extra = cd
            .get(name_at + name_len..name_at + name_len + extra_len)
            .ok_or(NpzError::Truncated("central directory extra"))?;
        // ZIP64 extended information (id 0x0001): present fields appear in the
        // fixed order usize, csize, local offset, disk start — but only for the
        // 32-bit fields that are saturated.
        let mut ex = 0usize;
        while ex + 4 <= extra.len() {
            let id = u16::from_le_bytes([extra[ex], extra[ex + 1]]);
            let len = usize::from(u16::from_le_bytes([extra[ex + 2], extra[ex + 3]]));
            let body = extra
                .get(ex + 4..ex + 4 + len)
                .ok_or(NpzError::Truncated("central directory extra field"))?;
            if id == 0x0001 {
                let mut p = 0usize;
                let take64 = |p: &mut usize| -> Result<u64, NpzError> {
                    let v = rd_u64(body, *p, "zip64 extra field")
                        .map_err(|_| NpzError::Zip64Missing("extended information"))?;
                    *p += 8;
                    Ok(v)
                };
                if usize_ == 0xFFFF_FFFF {
                    usize_ = take64(&mut p)?;
                }
                if csize == 0xFFFF_FFFF {
                    csize = take64(&mut p)?;
                }
                if local_offset == 0xFFFF_FFFF {
                    local_offset = take64(&mut p)?;
                }
                if disk_start == 0xFFFF {
                    disk_start = rd_u32(body, p, "zip64 extra field")
                        .map_err(|_| NpzError::Zip64Missing("extended information"))?;
                }
            }
            ex += 4 + len;
        }
        if disk_start != 0 {
            return Err(NpzError::MultiDisk);
        }
        if usize_ == 0xFFFF_FFFF || csize == 0xFFFF_FFFF || local_offset == 0xFFFF_FFFF {
            return Err(NpzError::Zip64Missing("saturated size or offset"));
        }
        out.push(CentralEntry {
            name,
            method,
            flags,
            crc,
            csize,
            usize_,
            local_offset,
        });
        pos = name_at + name_len + extra_len + comment_len;
    }
    Ok(out)
}

/// CRC-verified stored bytes of one member.
fn member_bytes<'a>(bytes: &'a [u8], e: &CentralEntry) -> Result<&'a [u8], NpzError> {
    if e.flags & 0x1 != 0 {
        return Err(NpzError::Encrypted(e.name.clone()));
    }
    if e.method != 0 {
        return Err(NpzError::UnsupportedCompression {
            entry: e.name.clone(),
            method: e.method,
        });
    }
    if e.csize != e.usize_ {
        return Err(NpzError::Inconsistent {
            entry: e.name.clone(),
            why: "stored member has compressed size != uncompressed size",
        });
    }
    let lo = to_usize(e.local_offset, "local header offset")?;
    let sig = rd_u32(bytes, lo, "local header")?;
    if sig != SIG_LOCAL {
        return Err(NpzError::BadSignature {
            what: "local header",
            found: sig,
        });
    }
    let name_len = usize::from(rd_u16(bytes, lo + 26, "local header")?);
    let extra_len = usize::from(rd_u16(bytes, lo + 28, "local header")?);
    let name = bytes
        .get(lo + 30..lo + 30 + name_len)
        .ok_or(NpzError::Truncated("local header name"))?;
    if name != e.name.as_bytes() {
        return Err(NpzError::Inconsistent {
            entry: e.name.clone(),
            why: "local header name differs from central directory",
        });
    }
    let data = lo + 30 + name_len + extra_len;
    let len = to_usize(e.csize, "member size")?;
    let end = data
        .checked_add(len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| NpzError::OutOfBounds {
            entry: e.name.clone(),
        })?;
    let payload = &bytes[data..end];
    let got = crc32(payload);
    if got != e.crc {
        return Err(NpzError::CrcMismatch {
            entry: e.name.clone(),
            expected: e.crc,
            got,
        });
    }
    Ok(payload)
}

// ---------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------

/// Decoded array payload.
#[derive(Clone, Debug, PartialEq)]
pub enum NpyData {
    U8(Vec<u8>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// `<U{n}` elements, trailing NULs stripped.
    Str(Vec<String>),
}

impl NpyData {
    fn dtype_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "|u1",
            Self::F32(_) => "<f4",
            Self::I64(_) => "<i8",
            Self::Str(_) => "<U",
        }
    }
}

/// One `.npy` array: C-order shape plus decoded elements.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

struct Header {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

fn bad(entry: &str, why: impl Into<String>) -> NpzError {
    NpzError::BadNpy {
        entry: entry.to_string(),
        why: why.into(),
    }
}

/// Tiny parser for the Python dict literal numpy writes:
/// `{'descr': '<f4', 'fortran_order': False, 'shape': (4, 2), }`.
fn parse_header_dict(entry: &str, text: &str) -> Result<Header, NpzError> {
    let b = text.as_bytes();
    let mut i = 0usize;
    let skip_ws = |i: &mut usize| {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    let quoted = |i: &mut usize| -> Result<String, NpzError> {
        let q = *b.get(*i).ok_or_else(|| bad(entry, "unterminated header"))?;
        if q != b'\'' && q != b'"' {
            return Err(bad(entry, "expected a quoted string in header"));
        }
        *i += 1;
        let start = *i;
        while *i < b.len() && b[*i] != q {
            if b[*i] == b'\\' {
                return Err(bad(
                    entry,
                    "escape sequences in header strings are unsupported",
                ));
            }
            *i += 1;
        }
        if *i >= b.len() {
            return Err(bad(entry, "unterminated string in header"));
        }
        let s = text[start..*i].to_string();
        *i += 1;
        Ok(s)
    };
    skip_ws(&mut i);
    if b.get(i) != Some(&b'{') {
        return Err(bad(entry, "header is not a dict"));
    }
    i += 1;
    let mut descr = None;
    let mut fortran = None;
    let mut shape = None;
    loop {
        skip_ws(&mut i);
        match b.get(i) {
            Some(b'}') => {
                i += 1;
                break;
            }
            Some(_) => {}
            None => return Err(bad(entry, "unterminated header dict")),
        }
        let key = quoted(&mut i)?;
        skip_ws(&mut i);
        if b.get(i) != Some(&b':') {
            return Err(bad(entry, "expected ':' in header"));
        }
        i += 1;
        skip_ws(&mut i);
        match key.as_str() {
            "descr" => {
                if descr.replace(quoted(&mut i)?).is_some() {
                    return Err(bad(entry, "duplicate header key `descr`"));
                }
            }
            "fortran_order" => {
                let v = if b[i..].starts_with(b"True") {
                    i += 4;
                    true
                } else if b[i..].starts_with(b"False") {
                    i += 5;
                    false
                } else {
                    return Err(bad(entry, "fortran_order is not True/False"));
                };
                if fortran.replace(v).is_some() {
                    return Err(bad(entry, "duplicate header key `fortran_order`"));
                }
            }
            "shape" => {
                if b.get(i) != Some(&b'(') {
                    return Err(bad(entry, "shape is not a tuple"));
                }
                i += 1;
                let mut dims = Vec::new();
                loop {
                    skip_ws(&mut i);
                    match b.get(i) {
                        Some(b')') => {
                            i += 1;
                            break;
                        }
                        Some(b',') => {
                            i += 1;
                        }
                        Some(c) if c.is_ascii_digit() => {
                            let start = i;
                            while i < b.len() && b[i].is_ascii_digit() {
                                i += 1;
                            }
                            let d: usize = text[start..i]
                                .parse()
                                .map_err(|_| bad(entry, "shape dimension overflows"))?;
                            dims.push(d);
                        }
                        _ => return Err(bad(entry, "malformed shape tuple")),
                    }
                }
                if shape.replace(dims).is_some() {
                    return Err(bad(entry, "duplicate header key `shape`"));
                }
            }
            other => return Err(bad(entry, format!("unknown header key `{other}`"))),
        }
        skip_ws(&mut i);
        match b.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => {}
            _ => return Err(bad(entry, "expected ',' or '}' in header")),
        }
    }
    skip_ws(&mut i);
    if i != b.len() {
        return Err(bad(entry, "trailing bytes after header dict"));
    }
    Ok(Header {
        descr: descr.ok_or_else(|| bad(entry, "header lacks `descr`"))?,
        fortran_order: fortran.ok_or_else(|| bad(entry, "header lacks `fortran_order`"))?,
        shape: shape.ok_or_else(|| bad(entry, "header lacks `shape`"))?,
    })
}

fn parse_npy(entry: &str, bytes: &[u8]) -> Result<NpyArray, NpzError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad(entry, "missing \\x93NUMPY magic"));
    }
    let (major, minor) = (bytes[6], bytes[7]);
    let (hlen, hstart) = match major {
        1 => (
            usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
            10usize,
        ),
        2 | 3 => {
            let raw = bytes
                .get(8..12)
                .ok_or_else(|| bad(entry, "truncated v2/v3 header length"))?;
            (
                u32::from_le_bytes(raw.try_into().unwrap()) as usize,
                12usize,
            )
        }
        v => return Err(bad(entry, format!("unsupported .npy version {v}.{minor}"))),
    };
    let hend = hstart
        .checked_add(hlen)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| bad(entry, "header runs past the member"))?;
    let raw = &bytes[hstart..hend];
    let text = if major == 3 {
        std::str::from_utf8(raw)
            .map_err(|_| bad(entry, "v3 header is not UTF-8"))?
            .to_string()
    } else {
        // v1/v2 headers are Latin-1; numpy writes ASCII.
        raw.iter().map(|&c| c as char).collect::<String>()
    };
    let header = parse_header_dict(entry, &text)?;
    if header.fortran_order {
        return Err(NpzError::FortranOrder {
            entry: entry.to_string(),
        });
    }
    let payload = &bytes[hend..];
    let count = header
        .shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| bad(entry, "shape element count overflows"))?;
    let descr = header.descr.as_str();
    let size_check = |item: usize| -> Result<(), NpzError> {
        let want = count
            .checked_mul(item)
            .ok_or_else(|| bad(entry, "payload size overflows"))?;
        if payload.len() != want {
            return Err(bad(
                entry,
                format!(
                    "payload is {} bytes, shape {:?} x {item} B needs {want}",
                    payload.len(),
                    header.shape
                ),
            ));
        }
        Ok(())
    };
    let data = match descr {
        "|u1" => {
            size_check(1)?;
            NpyData::U8(payload.to_vec())
        }
        "<f4" => {
            size_check(4)?;
            NpyData::F32(
                payload
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        "<i8" => {
            size_check(8)?;
            NpyData::I64(
                payload
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }
        d if d.starts_with("<U") => {
            let n: usize = d[2..].parse().ok().filter(|&n| n > 0).ok_or_else(|| {
                NpzError::UnsupportedDescr {
                    entry: entry.to_string(),
                    descr: d.to_string(),
                }
            })?;
            let item = n
                .checked_mul(4)
                .ok_or_else(|| bad(entry, "<U width overflows"))?;
            size_check(item)?;
            let mut strings = Vec::with_capacity(count);
            for elem in payload.chunks_exact(item) {
                let mut s = String::new();
                for (i, c) in elem.chunks_exact(4).enumerate() {
                    let cp = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    if cp == 0 {
                        // numpy NUL-pads; nothing but NULs may follow.
                        if elem[(i + 1) * 4..].iter().any(|&b| b != 0) {
                            return Err(bad(entry, "non-NUL data after an embedded NUL"));
                        }
                        break;
                    }
                    s.push(
                        char::from_u32(cp)
                            .ok_or_else(|| bad(entry, format!("invalid code point {cp:#x}")))?,
                    );
                }
                strings.push(s);
            }
            NpyData::Str(strings)
        }
        other => {
            return Err(NpzError::UnsupportedDescr {
                entry: entry.to_string(),
                descr: other.to_string(),
            })
        }
    };
    Ok(NpyArray {
        shape: header.shape,
        data,
    })
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// A parsed, CRC-verified NPZ archive. Member names have the `.npy` suffix
/// stripped (`gds.npy` is addressed as `gds`).
#[derive(Debug)]
pub struct NpzArchive {
    arrays: BTreeMap<String, NpyArray>,
}

impl NpzArchive {
    /// Read and fully verify an NPZ file.
    pub fn open(path: &Path) -> Result<Self, NpzError> {
        let bytes =
            std::fs::read(path).map_err(|e| NpzError::Io(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    /// Parse and fully verify an in-memory NPZ (every member is CRC-checked and
    /// decoded eagerly).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NpzError> {
        let eocd = find_eocd(bytes)?;
        let entries = parse_central(bytes, &eocd)?;
        let mut arrays = BTreeMap::new();
        for e in &entries {
            let payload = member_bytes(bytes, e)?;
            let key = e.name.strip_suffix(".npy").unwrap_or(&e.name).to_string();
            let array = parse_npy(&e.name, payload)?;
            if arrays.insert(key.clone(), array).is_some() {
                return Err(NpzError::DuplicateEntry(key));
            }
        }
        Ok(Self { arrays })
    }

    /// Member names (without `.npy`), sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.arrays.keys().map(String::as_str)
    }

    /// Test helper: take the decoded arrays (to mutate and re-serialise).
    #[cfg(test)]
    pub(crate) fn into_arrays(self) -> BTreeMap<String, NpyArray> {
        self.arrays
    }

    pub fn get(&self, name: &str) -> Result<&NpyArray, NpzError> {
        self.arrays
            .get(name)
            .ok_or_else(|| NpzError::MissingEntry(name.to_string()))
    }

    fn typed<'a, T>(
        &'a self,
        name: &str,
        expected_shape: &[usize],
        dtype: &'static str,
        pick: impl Fn(&'a NpyData) -> Option<&'a [T]>,
    ) -> Result<&'a [T], NpzError> {
        let a = self.get(name)?;
        let found = a.data.dtype_name();
        let slice = pick(&a.data).ok_or_else(|| NpzError::WrongDtype {
            entry: name.to_string(),
            expected: dtype,
            found,
        })?;
        if a.shape != expected_shape {
            return Err(NpzError::ShapeMismatch {
                entry: name.to_string(),
                expected: expected_shape.to_vec(),
                found: a.shape.clone(),
            });
        }
        Ok(slice)
    }

    /// `|u1` array of exactly `shape`.
    pub fn u8_array(&self, name: &str, shape: &[usize]) -> Result<&[u8], NpzError> {
        self.typed(name, shape, "|u1", |d| match d {
            NpyData::U8(v) => Some(v.as_slice()),
            _ => None,
        })
    }

    /// `<f4` array of exactly `shape`.
    pub fn f32_array(&self, name: &str, shape: &[usize]) -> Result<&[f32], NpzError> {
        self.typed(name, shape, "<f4", |d| match d {
            NpyData::F32(v) => Some(v.as_slice()),
            _ => None,
        })
    }

    /// `<i8` 0-d scalar.
    pub fn i64_scalar(&self, name: &str) -> Result<i64, NpzError> {
        let v = self.typed(name, &[], "<i8", |d| match d {
            NpyData::I64(v) => Some(v.as_slice()),
            _ => None,
        })?;
        Ok(v[0])
    }

    /// `<U{n}` 0-d scalar.
    pub fn str_scalar(&self, name: &str) -> Result<&str, NpzError> {
        let v = self.typed(name, &[], "<U", |d| match d {
            NpyData::Str(v) => Some(v.as_slice()),
            _ => None,
        })?;
        Ok(v[0].as_str())
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;

    // --- in-test ZIP/.npy builders ------------------------------------------

    pub(crate) fn npy(descr: &str, fortran: bool, shape: &str, payload: &[u8]) -> Vec<u8> {
        let mut dict = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        // Pad so (10 + header) % 64 == 0, newline-terminated (numpy v1.0).
        while (10 + dict.len() + 1) % 64 != 0 {
            dict.push(' ');
        }
        dict.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    pub(crate) struct Member {
        pub(crate) name: String,
        pub(crate) data: Vec<u8>,
        pub(crate) method: u16,
        pub(crate) crc_xor: u32,
    }

    pub(crate) fn member(name: &str, data: Vec<u8>) -> Member {
        Member {
            name: name.to_string(),
            data,
            method: 0,
            crc_xor: 0,
        }
    }

    /// Plain (non-ZIP64) ZIP writer; `method` is only a label.
    pub(crate) fn build_zip(members: &[Member]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for m in members {
            let off = out.len() as u32;
            let crc = crc32(&m.data) ^ m.crc_xor;
            let mut local = Vec::new();
            local.extend_from_slice(&SIG_LOCAL.to_le_bytes());
            local.extend_from_slice(&20u16.to_le_bytes());
            local.extend_from_slice(&0u16.to_le_bytes());
            local.extend_from_slice(&m.method.to_le_bytes());
            local.extend_from_slice(&0u32.to_le_bytes());
            local.extend_from_slice(&crc.to_le_bytes());
            local.extend_from_slice(&(m.data.len() as u32).to_le_bytes());
            local.extend_from_slice(&(m.data.len() as u32).to_le_bytes());
            local.extend_from_slice(&(m.name.len() as u16).to_le_bytes());
            local.extend_from_slice(&0u16.to_le_bytes());
            local.extend_from_slice(m.name.as_bytes());
            out.extend_from_slice(&local);
            out.extend_from_slice(&m.data);
            central.extend_from_slice(&SIG_CDIR.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&m.method.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&(m.data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(m.data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(m.name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0u8; 8]); // extra, comment, disk, int attr
            central.extend_from_slice(&0u32.to_le_bytes()); // external attr
            central.extend_from_slice(&off.to_le_bytes());
            central.extend_from_slice(m.name.as_bytes());
        }
        let cd_off = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&SIG_EOCD.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    pub(crate) fn f4_member(name: &str, vals: &[f32], shape: &str) -> Member {
        let payload: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        member(name, npy("<f4", false, shape, &payload))
    }

    /// `.npy` v1 bytes for a decoded array (the inverse of `parse_npy`).
    pub(crate) fn npy_bytes(a: &NpyArray) -> Vec<u8> {
        let shape = match a.shape.as_slice() {
            [] => "()".to_string(),
            [d] => format!("({d},)"),
            dims => format!(
                "({})",
                dims.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        match &a.data {
            NpyData::U8(v) => npy("|u1", false, &shape, v),
            NpyData::F32(v) => npy(
                "<f4",
                false,
                &shape,
                &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            ),
            NpyData::I64(v) => npy(
                "<i8",
                false,
                &shape,
                &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            ),
            NpyData::Str(v) => {
                let width = v
                    .iter()
                    .map(|s| s.chars().count())
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let mut payload = Vec::new();
                for s in v {
                    let mut n = 0;
                    for c in s.chars() {
                        payload.extend_from_slice(&(c as u32).to_le_bytes());
                        n += 1;
                    }
                    payload.extend(std::iter::repeat(0u8).take((width - n) * 4));
                }
                npy(&format!("<U{width}"), false, &shape, &payload)
            }
        }
    }

    /// Stored NPZ with ZIP-correct CRCs from decoded arrays (member names get
    /// `.npy` appended).
    pub(crate) fn build_npz(arrays: &std::collections::BTreeMap<String, NpyArray>) -> Vec<u8> {
        let members: Vec<Member> = arrays
            .iter()
            .map(|(k, a)| member(&format!("{k}.npy"), npy_bytes(a)))
            .collect();
        build_zip(&members)
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    const W3: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w3.npz");
    const W2: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w2.npz");

    // --- tests -------------------------------------------------------------

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn reads_both_numpy_zip64_fixtures() {
        for (bytes, bits) in [(W3, 3u32), (W2, 2u32)] {
            let z = NpzArchive::from_bytes(bytes).unwrap();
            let names: Vec<&str> = z.names().collect();
            assert_eq!(
                names,
                ["bits", "dc", "dds", "gc", "gds", "grid", "scale_rule", "sd"]
            );
            assert_eq!(z.i64_scalar("bits").unwrap(), i64::from(bits));
            assert_eq!(z.str_scalar("grid").unwrap(), "uniform");
            assert_eq!(
                z.str_scalar("scale_rule").unwrap(),
                "mqi4-canonical-v1-rne-tree128"
            );
            assert_eq!(z.get("gds").unwrap().shape, [4, 2]);
            assert_eq!(z.get("gc").unwrap().shape, [4, 256]);
            assert_eq!(z.get("dds").unwrap().shape, [8, 1]);
            assert_eq!(z.get("dc").unwrap().shape, [8, 128]);
            assert_eq!(z.get("sd").unwrap().shape, [128]);
            assert_eq!(z.get("bits").unwrap().shape, Vec::<usize>::new());
            let gc = z.u8_array("gc", &[4, 256]).unwrap();
            assert_eq!(gc[0], 3);
            assert!(gc.iter().all(|&c| u32::from(c) < (1u32 << bits)));
            let sd = z.f32_array("sd", &[128]).unwrap();
            assert!(sd.iter().all(|v| v.to_bits() == 1.0f32.to_bits()));
            let gds = z.f32_array("gds", &[4, 2]).unwrap();
            assert!(gds.iter().all(|v| v.is_finite() && *v >= 0.0));
            // Row 1 group 1 is the zero-scale group.
            assert_eq!(gds[3], 0.0);
        }
        let w3 = NpzArchive::from_bytes(W3).unwrap();
        assert!((w3.f32_array("gds", &[4, 2]).unwrap()[0] - 0.01192474).abs() < 1e-8);
        assert_eq!(w3.u8_array("gc", &[4, 256]).unwrap()[..4], [3, 4, 6, 3]);
        assert_eq!(w3.u8_array("dc", &[8, 128]).unwrap()[..4], [7, 3, 0, 4]);
    }

    #[test]
    fn typed_accessors_reject_wrong_dtype_shape_and_missing() {
        let z = NpzArchive::from_bytes(W3).unwrap();
        assert!(matches!(
            z.f32_array("gc", &[4, 256]),
            Err(NpzError::WrongDtype { .. })
        ));
        assert!(matches!(
            z.u8_array("gc", &[4, 255]),
            Err(NpzError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            z.i64_scalar("gc"),
            Err(NpzError::WrongDtype { .. })
        ));
        assert!(matches!(
            z.str_scalar("bits"),
            Err(NpzError::WrongDtype { .. })
        ));
        assert!(matches!(z.get("nope"), Err(NpzError::MissingEntry(_))));
    }

    #[test]
    fn corrupted_member_bytes_fail_the_crc() {
        let mut bytes = W3.to_vec();
        // Offset 400 lies inside the first members' stored data, not a header.
        bytes[400] ^= 0xFF;
        assert!(matches!(
            NpzArchive::from_bytes(&bytes),
            Err(NpzError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn built_zip_round_trips_and_wrong_crc_is_rejected() {
        let good = build_zip(&[f4_member("a.npy", &[1.0, 2.0], "(2,)")]);
        let z = NpzArchive::from_bytes(&good).unwrap();
        assert_eq!(z.f32_array("a", &[2]).unwrap(), [1.0, 2.0]);
        let mut m = f4_member("a.npy", &[1.0, 2.0], "(2,)");
        m.crc_xor = 1;
        assert!(matches!(
            NpzArchive::from_bytes(&build_zip(&[m])),
            Err(NpzError::CrcMismatch { entry, .. }) if entry == "a.npy"
        ));
    }

    #[test]
    fn deflated_members_are_rejected() {
        let mut m = f4_member("a.npy", &[1.0], "(1,)");
        m.method = 8;
        assert!(matches!(
            NpzArchive::from_bytes(&build_zip(&[m])),
            Err(NpzError::UnsupportedCompression { method: 8, .. })
        ));
    }

    #[test]
    fn fortran_order_is_rejected() {
        let payload: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let z = build_zip(&[member("a.npy", npy("<f4", true, "(2,)", &payload))]);
        assert!(matches!(
            NpzArchive::from_bytes(&z),
            Err(NpzError::FortranOrder { .. })
        ));
    }

    #[test]
    fn unknown_descr_is_rejected() {
        for descr in ["<f8", "<f2", ">f4", "<i4", "|b1", "<U", "<U0", "<c8", "|S4"] {
            let z = build_zip(&[member("a.npy", npy(descr, false, "(1,)", &[0u8; 8]))]);
            assert!(
                matches!(
                    NpzArchive::from_bytes(&z),
                    Err(NpzError::UnsupportedDescr { .. })
                ),
                "{descr} must be rejected"
            );
        }
    }

    #[test]
    fn payload_length_must_match_shape() {
        let z = build_zip(&[member("a.npy", npy("<f4", false, "(3,)", &[0u8; 8]))]);
        assert!(matches!(
            NpzArchive::from_bytes(&z),
            Err(NpzError::BadNpy { .. })
        ));
    }

    #[test]
    fn non_zip_and_truncated_inputs_are_rejected() {
        assert_eq!(
            NpzArchive::from_bytes(b"not a zip at all").unwrap_err(),
            NpzError::NotZip
        );
        assert!(NpzArchive::from_bytes(&W3[..W3.len() - 5]).is_err());
        assert!(NpzArchive::from_bytes(&[]).is_err());
    }

    #[test]
    fn duplicate_members_are_rejected() {
        let z = build_zip(&[
            f4_member("a.npy", &[1.0], "(1,)"),
            f4_member("a.npy", &[2.0], "(1,)"),
        ]);
        assert!(matches!(
            NpzArchive::from_bytes(&z),
            Err(NpzError::DuplicateEntry(_))
        ));
    }

    #[test]
    fn utf32_strings_and_i8_scalars_decode() {
        let mut s = Vec::new();
        for c in "héllo".chars() {
            s.extend_from_slice(&(c as u32).to_le_bytes());
        }
        s.extend_from_slice(&[0u8; 12]); // padded to <U8
        let z = build_zip(&[
            member("s.npy", npy("<U8", false, "()", &s)),
            member("i.npy", npy("<i8", false, "()", &(-7i64).to_le_bytes())),
            member("u.npy", npy("|u1", false, "(2, 2)", &[1, 2, 3, 4])),
        ]);
        let a = NpzArchive::from_bytes(&z).unwrap();
        assert_eq!(a.str_scalar("s").unwrap(), "héllo");
        assert_eq!(a.i64_scalar("i").unwrap(), -7);
        assert_eq!(a.u8_array("u", &[2, 2]).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn npy_v2_and_v3_headers_parse() {
        let payload = 1.5f32.to_le_bytes();
        for major in [2u8, 3u8] {
            let mut dict =
                String::from("{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }");
            while (12 + dict.len() + 1) % 64 != 0 {
                dict.push(' ');
            }
            dict.push('\n');
            let mut v = vec![0x93, b'N', b'U', b'M', b'P', b'Y', major, 0];
            v.extend_from_slice(&(dict.len() as u32).to_le_bytes());
            v.extend_from_slice(dict.as_bytes());
            v.extend_from_slice(&payload);
            let a = NpzArchive::from_bytes(&build_zip(&[member("a.npy", v)])).unwrap();
            assert_eq!(a.f32_array("a", &[1]).unwrap(), [1.5]);
        }
    }
}
