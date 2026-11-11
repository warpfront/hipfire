// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Code-object inspection for `hip_loop`: split a clang offload bundle (what
//! `hipcc --genco` writes) into the device ELF, read the per-kernel `.text`
//! bytes from the symbol table and the per-kernel resource metadata from the
//! AMDGPU `amdhsa.kernels` msgpack note. No ROCm tools, no device.
//!
//! Included with `#[path = "hip_loop/elf.rs"] mod elf;`.
#![allow(dead_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

type Result<T, E = String> = std::result::Result<T, E>;

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn u16le(b: &[u8], o: usize) -> Result<u16> {
    b.get(o..o + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| format!("truncated at {o}"))
}
fn u32le(b: &[u8], o: usize) -> Result<u32> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| format!("truncated at {o}"))
}
fn u64le(b: &[u8], o: usize) -> Result<u64> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| format!("truncated at {o}"))
}

// ---------------------------------------------------------------------------
// Offload bundle

const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";

/// The device ELF for `arch` inside `image`: the entry of a clang offload
/// bundle whose id names `arch` (`hipv4-amdgcn-amd-amdhsa--gfx1100`), or
/// `image` itself when it already is an ELF.
pub fn device_elf(image: &[u8], arch: &str) -> Result<Vec<u8>> {
    if image.starts_with(b"\x7fELF") {
        return Ok(image.to_vec());
    }
    if !image.starts_with(BUNDLE_MAGIC) {
        return Err("code object is neither an ELF nor a clang offload bundle".into());
    }
    let n = u64le(image, 24)? as usize;
    let mut o = 32usize;
    for _ in 0..n {
        let off = u64le(image, o)? as usize;
        let size = u64le(image, o + 8)? as usize;
        let idlen = u64le(image, o + 16)? as usize;
        let id = image
            .get(o + 24..o + 24 + idlen)
            .ok_or("bundle entry id truncated")?;
        let id = String::from_utf8_lossy(id);
        o += 24 + idlen;
        if id.contains("amdgcn") && id.ends_with(arch) {
            return image
                .get(off..off + size)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "bundle entry out of range".into());
        }
    }
    Err(format!("no amdgcn {arch} entry in the offload bundle"))
}

// ---------------------------------------------------------------------------
// ELF64

struct Section {
    name: String,
    ty: u32,
    addr: u64,
    offset: usize,
    size: usize,
    link: u32,
}

fn sections(elf: &[u8]) -> Result<Vec<Section>> {
    if !elf.starts_with(b"\x7fELF") || elf.get(4) != Some(&2) || elf.get(5) != Some(&1) {
        return Err("not a little-endian ELF64".into());
    }
    let shoff = u64le(elf, 0x28)? as usize;
    let shentsize = u16le(elf, 0x3A)? as usize;
    let shnum = u16le(elf, 0x3C)? as usize;
    let shstrndx = u16le(elf, 0x3E)? as usize;
    if shentsize < 64 || shnum == 0 {
        return Err("ELF has no section headers".into());
    }
    let hdr = |i: usize| -> Result<(u32, u32, u64, usize, usize, u32)> {
        let b = shoff + i * shentsize;
        Ok((
            u32le(elf, b)?,
            u32le(elf, b + 4)?,
            u64le(elf, b + 16)?,
            u64le(elf, b + 24)? as usize,
            u64le(elf, b + 32)? as usize,
            u32le(elf, b + 40)?,
        ))
    };
    let (_, _, _, stroff, strsize, _) = hdr(shstrndx)?;
    let strtab = elf
        .get(stroff..stroff + strsize)
        .ok_or("section name table out of range")?;
    let cstr = |t: &[u8], o: usize| -> String {
        let s = t.get(o..).unwrap_or(&[]);
        let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
        String::from_utf8_lossy(&s[..end]).into_owned()
    };
    (0..shnum)
        .map(|i| {
            let (name, ty, addr, offset, size, link) = hdr(i)?;
            Ok(Section {
                name: cstr(strtab, name as usize),
                ty,
                addr,
                offset,
                size,
                link,
            })
        })
        .collect()
}

/// A function symbol of the device ELF.
#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub section: usize,
    pub value: u64,
    pub size: usize,
}

fn functions(elf: &[u8], secs: &[Section]) -> Result<Vec<Func>> {
    // SHT_SYMTAB = 2, SHT_DYNSYM = 11.
    let symtab = secs
        .iter()
        .find(|s| s.ty == 2)
        .or_else(|| secs.iter().find(|s| s.ty == 11))
        .ok_or("ELF has no symbol table")?;
    let strs = secs
        .get(symtab.link as usize)
        .ok_or("symbol string table missing")?;
    let strtab = elf
        .get(strs.offset..strs.offset + strs.size)
        .ok_or("symbol string table out of range")?;
    let mut out = Vec::new();
    for i in 0..symtab.size / 24 {
        let b = symtab.offset + i * 24;
        let name = u32le(elf, b)? as usize;
        let info = *elf.get(b + 4).ok_or("truncated symbol")?;
        let shndx = u16le(elf, b + 6)? as usize;
        let value = u64le(elf, b + 8)?;
        let size = u64le(elf, b + 16)? as usize;
        // STT_FUNC = 2.
        if info & 0xf != 2 || shndx == 0 || shndx >= secs.len() {
            continue;
        }
        let s = strtab.get(name..).ok_or("symbol name out of range")?;
        let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
        out.push(Func {
            name: String::from_utf8_lossy(&s[..end]).into_owned(),
            section: shndx,
            value,
            size,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// msgpack (the subset the AMDGPU metadata uses)

fn mp_value(b: &[u8], o: &mut usize) -> Result<Value> {
    let tag = *b.get(*o).ok_or("msgpack truncated")?;
    *o += 1;
    let take = |o: &mut usize, n: usize| -> Result<&[u8]> {
        let s = b.get(*o..*o + n).ok_or("msgpack truncated")?;
        *o += n;
        Ok(s)
    };
    let be = |s: &[u8]| s.iter().fold(0u64, |a, &x| (a << 8) | u64::from(x));
    let sx = |v: u64, bits: u32| ((v << (64 - bits)) as i64) >> (64 - bits);
    Ok(match tag {
        0x00..=0x7f => json!(tag),
        0xe0..=0xff => json!(tag as i8),
        0xc0 => Value::Null,
        0xc2 => json!(false),
        0xc3 => json!(true),
        0xcc => json!(be(take(o, 1)?)),
        0xcd => json!(be(take(o, 2)?)),
        0xce => json!(be(take(o, 4)?)),
        0xcf => json!(be(take(o, 8)?)),
        0xd0 => json!(sx(be(take(o, 1)?), 8)),
        0xd1 => json!(sx(be(take(o, 2)?), 16)),
        0xd2 => json!(sx(be(take(o, 4)?), 32)),
        0xd3 => json!(sx(be(take(o, 8)?), 64)),
        0xca => json!(f32::from_bits(be(take(o, 4)?) as u32)),
        0xcb => json!(f64::from_bits(be(take(o, 8)?))),
        0xa0..=0xbf | 0xd9 | 0xda | 0xdb | 0xc4 | 0xc5 | 0xc6 => {
            let n = match tag {
                0xa0..=0xbf => usize::from(tag & 0x1f),
                0xd9 | 0xc4 => be(take(o, 1)?) as usize,
                0xda | 0xc5 => be(take(o, 2)?) as usize,
                _ => be(take(o, 4)?) as usize,
            };
            let s = take(o, n)?;
            json!(String::from_utf8_lossy(s))
        }
        0x90..=0x9f | 0xdc | 0xdd => {
            let n = match tag {
                0x90..=0x9f => usize::from(tag & 0xf),
                0xdc => be(take(o, 2)?) as usize,
                _ => be(take(o, 4)?) as usize,
            };
            let mut v = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                v.push(mp_value(b, o)?);
            }
            Value::Array(v)
        }
        0x80..=0x8f | 0xde | 0xdf => {
            let n = match tag {
                0x80..=0x8f => usize::from(tag & 0xf),
                0xde => be(take(o, 2)?) as usize,
                _ => be(take(o, 4)?) as usize,
            };
            let mut m = serde_json::Map::new();
            for _ in 0..n {
                let k = mp_value(b, o)?;
                let k = k
                    .as_str()
                    .ok_or("msgpack map key is not a string")?
                    .to_owned();
                m.insert(k, mp_value(b, o)?);
            }
            Value::Object(m)
        }
        t => return Err(format!("unsupported msgpack tag {t:#x}")),
    })
}

/// The AMDGPU metadata document (`NT_AMDGPU_METADATA`, note type 32, name
/// `AMDGPU`) of the device ELF.
pub fn amdgpu_metadata(elf: &[u8]) -> Result<Value> {
    let secs = sections(elf)?;
    for s in secs.iter().filter(|s| s.ty == 7) {
        let mut o = s.offset;
        let end = s.offset + s.size;
        while o + 12 <= end {
            let namesz = u32le(elf, o)? as usize;
            let descsz = u32le(elf, o + 4)? as usize;
            let ty = u32le(elf, o + 8)?;
            let name_at = o + 12;
            let desc_at = name_at + namesz.div_ceil(4) * 4;
            let next = desc_at + descsz.div_ceil(4) * 4;
            let name = elf.get(name_at..name_at + namesz).unwrap_or(&[]);
            if ty == 32 && name.starts_with(b"AMDGPU") {
                let desc = elf
                    .get(desc_at..desc_at + descsz)
                    .ok_or("AMDGPU metadata note out of range")?;
                return mp_value(desc, &mut 0);
            }
            o = next;
        }
    }
    Err("no AMDGPU metadata note".into())
}

// ---------------------------------------------------------------------------
// Per-kernel record

/// Resources and code identity of one kernel symbol.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelInfo {
    pub symbol: String,
    /// sha256 of the symbol's `.text` bytes (not the whole file).
    pub code_sha256: String,
    pub code_bytes: usize,
    pub vgprs: u64,
    pub sgprs: u64,
    pub lds_bytes: u64,
    pub scratch_bytes: u64,
    pub vgpr_spill: u64,
    pub sgpr_spill: u64,
    pub wavefront_size: u64,
    pub max_flat_workgroup_size: u64,
    pub kernarg_bytes: u64,
    /// Explicit (non-hidden) kernel arguments: `(offset, size, "pointer"|"value")`.
    pub explicit_args: Vec<(u64, u64, String)>,
}

impl KernelInfo {
    /// Why `self` cannot be launched with `base`'s kernarg blob, if it cannot:
    /// the explicit argument layouts (offset, size, pointer-ness) must be equal.
    /// Hidden arguments are filled by the runtime and may differ. When either
    /// object carries no argument metadata the kernarg segment sizes must match.
    pub fn kernarg_mismatch(&self, base: &KernelInfo) -> Option<String> {
        if self.explicit_args.is_empty() || base.explicit_args.is_empty() {
            return (self.kernarg_bytes != base.kernarg_bytes).then(|| {
                format!(
                    "kernarg segment is {} B, the incumbent's is {} B (no argument metadata to compare)",
                    self.kernarg_bytes, base.kernarg_bytes
                )
            });
        }
        (self.explicit_args != base.explicit_args).then(|| {
            format!(
                "explicit kernel arguments {:?} differ from the incumbent's {:?}",
                self.explicit_args, base.explicit_args
            )
        })
    }

    /// Waves per SIMD the VGPR budget allows on gfx11/gfx12 (1536 VGPR per SIMD lane in wave32
    /// mode, allocation granule 8, 16 wave slots; a wave64 holds each VGPR for both halves of
    /// the SIMD32, so it costs twice the wave32 allocation). LDS and workgroup size can only
    /// lower it further.
    pub fn occupancy_waves_per_simd(&self) -> u64 {
        let alloc = self.vgprs.max(1).div_ceil(8) * 8;
        let per_wave = if self.wavefront_size == 64 {
            alloc * 2
        } else {
            alloc
        };
        (1536 / per_wave).min(16)
    }

    pub fn spill(&self) -> u64 {
        self.vgpr_spill + self.sgpr_spill
    }

    pub fn json(&self) -> Value {
        json!({
            "symbol": self.symbol,
            "code_sha256": self.code_sha256,
            "vgprs": self.vgprs,
            "vgprs_allocated": self.vgprs.max(1).div_ceil(8) * 8,
            "sgprs": self.sgprs,
            "lds_bytes": self.lds_bytes,
            "scratch_bytes": self.scratch_bytes,
            "spill": self.spill(),
            "vgpr_spill": self.vgpr_spill,
            "sgpr_spill": self.sgpr_spill,
            "code_bytes": self.code_bytes,
            "occupancy_waves_per_simd": self.occupancy_waves_per_simd(),
            "occupancy_basis": if self.wavefront_size == 64 {
                "vgpr (wave64: 2 x the wave32 allocation; 1536 VGPR/SIMD, granule 8, 16 slots)"
            } else {
                "vgpr (gfx11 wave32: 1536 VGPR/SIMD, granule 8, 16 slots)"
            },
            "wavefront_size": self.wavefront_size,
            "max_flat_workgroup_size": self.max_flat_workgroup_size,
            "kernarg_bytes": self.kernarg_bytes,
            "explicit_args": self.explicit_args.iter()
                .map(|(o, s, k)| json!({"offset": o, "size": s, "class": k}))
                .collect::<Vec<_>>(),
        })
    }
}

/// Every kernel of the object (bundle or ELF) for `arch`, by symbol.
pub fn kernels(image: &[u8], arch: &str) -> Result<BTreeMap<String, KernelInfo>> {
    let elf = device_elf(image, arch)?;
    let secs = sections(&elf)?;
    let funcs = functions(&elf, &secs)?;
    let meta = amdgpu_metadata(&elf)?;
    let empty = Vec::new();
    let mk = meta["amdhsa.kernels"].as_array().unwrap_or(&empty);
    let mut out = BTreeMap::new();
    for k in mk {
        let Some(name) = k[".name"].as_str() else {
            continue;
        };
        let f = funcs
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| format!("kernel {name} has no function symbol"))?;
        let s = &secs[f.section];
        let start = (f.value - s.addr) as usize + s.offset;
        let text = elf
            .get(start..start + f.size)
            .ok_or_else(|| format!("kernel {name}: text out of range"))?;
        let n = |key: &str| k[key].as_u64().unwrap_or(0);
        out.insert(
            name.to_owned(),
            KernelInfo {
                symbol: name.to_owned(),
                code_sha256: sha256_hex(text),
                code_bytes: f.size,
                vgprs: n(".vgpr_count"),
                sgprs: n(".sgpr_count"),
                lds_bytes: n(".group_segment_fixed_size"),
                scratch_bytes: n(".private_segment_fixed_size"),
                vgpr_spill: n(".vgpr_spill_count"),
                sgpr_spill: n(".sgpr_spill_count"),
                wavefront_size: n(".wavefront_size"),
                max_flat_workgroup_size: n(".max_flat_workgroup_size"),
                kernarg_bytes: n(".kernarg_segment_size"),
                explicit_args: k[".args"]
                    .as_array()
                    .unwrap_or(&empty)
                    .iter()
                    .filter_map(|a| {
                        let kind = a[".value_kind"].as_str()?;
                        if kind.starts_with("hidden_") {
                            return None;
                        }
                        // global_buffer / dynamic_shared_pointer / by_value are all one
                        // launch-ABI slot; only offset, size and pointer-ness matter.
                        let class = if kind == "by_value" {
                            "value"
                        } else {
                            "pointer"
                        };
                        Some((
                            a[".offset"].as_u64()?,
                            a[".size"].as_u64()?,
                            class.to_owned(),
                        ))
                    })
                    .collect(),
            },
        );
    }
    if out.is_empty() {
        return Err("no kernels in the AMDGPU metadata".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msgpack_scalars_and_containers() {
        // {"a": [1, -2, 300, "hi"], "b": true}
        let b = [
            0x82, 0xa1, b'a', 0x94, 0x01, 0xfe, 0xcd, 0x01, 0x2c, 0xa2, b'h', b'i', 0xa1, b'b',
            0xc3,
        ];
        let v = mp_value(&b, &mut 0).unwrap();
        assert_eq!(v, json!({"a": [1, -2, 300, "hi"], "b": true}));
    }

    #[test]
    fn bundle_selects_the_arch_entry() {
        let mut img = BUNDLE_MAGIC.to_vec();
        img.extend_from_slice(&2u64.to_le_bytes());
        let ids = [
            "host-x86_64-unknown-linux-gnu-",
            "hipv4-amdgcn-amd-amdhsa--gfx1100",
        ];
        let payloads: [&[u8]; 2] = [b"", b"\x7fELFxx"];
        let table: usize = 32 + ids.iter().map(|i| 24 + i.len()).sum::<usize>();
        let mut off = table;
        for (id, p) in ids.iter().zip(payloads) {
            img.extend_from_slice(&(off as u64).to_le_bytes());
            img.extend_from_slice(&(p.len() as u64).to_le_bytes());
            img.extend_from_slice(&(id.len() as u64).to_le_bytes());
            img.extend_from_slice(id.as_bytes());
            off += p.len();
        }
        for p in payloads {
            img.extend_from_slice(p);
        }
        assert_eq!(device_elf(&img, "gfx1100").unwrap(), b"\x7fELFxx");
        assert!(device_elf(&img, "gfx1201").is_err());
    }

    #[test]
    fn occupancy_follows_vgprs() {
        let mut k = KernelInfo {
            symbol: "k".into(),
            code_sha256: String::new(),
            code_bytes: 0,
            vgprs: 96,
            sgprs: 0,
            lds_bytes: 0,
            scratch_bytes: 0,
            vgpr_spill: 0,
            sgpr_spill: 0,
            wavefront_size: 32,
            max_flat_workgroup_size: 256,
            kernarg_bytes: 0,
            explicit_args: Vec::new(),
        };
        assert_eq!(k.occupancy_waves_per_simd(), 16);
        k.vgprs = 128;
        assert_eq!(k.occupancy_waves_per_simd(), 12);
        k.vgprs = 256;
        assert_eq!(k.occupancy_waves_per_simd(), 6);
    }
}
