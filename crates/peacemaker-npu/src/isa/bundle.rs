// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Allocation-free AIE2P VLIW packing for every generated composite format.
//!
//! Payloads are encoded slot bits, not whole standalone instructions. `pack_slots`
//! chooses the shortest exact slot-set format; `Bundle` retains an explicit format
//! and architectural don't-care bits for byte-identical re-encoding. Long occupies
//! the ALU/MV alternative: illegal combinations have no format and are rejected.

use super::gen::{Encoding, Format, Slot, FORMATS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    UnknownFormat,
    DuplicateSlot(Slot),
    MissingSlot(Slot),
    UnexpectedSlot(Slot),
    PayloadOverflow(Slot),
    InvalidDontcare,
    MissingOperand(&'static str),
    OperandOverflow(&'static str),
    UnexpectedOperand,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub struct PackedBundle {
    pub bytes: [u8; 16],
    pub len: u8,
}
impl PackedBundle {
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}
impl AsRef<[u8]> for PackedBundle {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bundle {
    pub format: &'static Format,
    pub slots: [Option<u64>; 8],
    pub dontcare: u128,
}
impl Bundle {
    pub fn new(format: &'static Format) -> Self {
        Self {
            format,
            slots: [None; 8],
            dontcare: 0,
        }
    }
    pub fn set(&mut self, slot: Slot, payload: u64) -> Result<(), Error> {
        if !self.format.slots.iter().any(|f| f.slot == slot) {
            return Err(Error::UnexpectedSlot(slot));
        }
        if payload >> slot.width() != 0 {
            return Err(Error::PayloadOverflow(slot));
        }
        if self.slots[slot as usize].is_some() {
            return Err(Error::DuplicateSlot(slot));
        }
        self.slots[slot as usize] = Some(payload);
        Ok(())
    }
    pub fn pack(&self) -> Result<PackedBundle, Error> {
        pack(self)
    }
}

pub fn pack(bundle: &Bundle) -> Result<PackedBundle, Error> {
    if bundle.dontcare & !bundle.format.dontcare_mask != 0 {
        return Err(Error::InvalidDontcare);
    }
    let mut bits = bundle.format.value | bundle.dontcare;
    let mut used = 0u8;
    for field in bundle.format.slots {
        let payload = bundle.slots[field.slot as usize].ok_or(Error::MissingSlot(field.slot))?;
        if payload >> field.width != 0 {
            return Err(Error::PayloadOverflow(field.slot));
        }
        bits |= (payload as u128) << field.lsb;
        used |= 1 << field.slot as usize;
    }
    for (index, payload) in bundle.slots.iter().enumerate() {
        if payload.is_some() && used & (1 << index) == 0 {
            let slot = [
                Slot::Ldb,
                Slot::Alu,
                Slot::Lng,
                Slot::Lda,
                Slot::Mv,
                Slot::St,
                Slot::Vec,
                Slot::Nop16,
            ][index];
            return Err(Error::UnexpectedSlot(slot));
        }
    }
    Ok(PackedBundle {
        bytes: bits.to_le_bytes(),
        len: bundle.format.bits / 8,
    })
}

pub fn pack_slots(slots: &[(Slot, u64)]) -> Result<PackedBundle, Error> {
    let mut wanted = 0u8;
    for &(slot, _) in slots {
        if wanted & (1 << slot as usize) != 0 {
            return Err(Error::DuplicateSlot(slot));
        }
        wanted |= 1 << slot as usize;
    }
    let format = FORMATS
        .iter()
        .filter(|f| {
            f.slots
                .iter()
                .fold(0u8, |mask, field| mask | 1 << field.slot as usize)
                == wanted
        })
        .min_by_key(|f| f.bits)
        .ok_or(Error::UnknownFormat)?;
    let mut bundle = Bundle::new(format);
    for &(slot, value) in slots {
        bundle.set(slot, value)?;
    }
    bundle.pack()
}

/// Length tag is self-delimiting in the low bits of the first byte.
pub const fn bundle_len(first: u8) -> usize {
    if first & 1 != 0 {
        16
    } else if first & 3 == 0 && first & 7 == 4 {
        6
    } else {
        match first & 15 {
            0 => 2,
            8 => 4,
            2 => 8,
            10 => 10,
            6 => 12,
            14 => 14,
            _ => 0,
        }
    }
}

/// Decode just one bundle, accepting trailing bytes for stream iteration.
pub fn unpack(bytes: &[u8]) -> Result<Bundle, Error> {
    let first = *bytes.first().ok_or(Error::Truncated)?;
    let len = bundle_len(first);
    if len == 0 {
        return Err(Error::UnknownFormat);
    }
    if bytes.len() < len {
        return Err(Error::Truncated);
    }
    let mut raw = [0u8; 16];
    raw[..len].copy_from_slice(&bytes[..len]);
    let bits = u128::from_le_bytes(raw);
    let format = FORMATS
        .iter()
        .find(|f| f.bits as usize == len * 8 && bits & f.mask == f.value)
        .ok_or(Error::UnknownFormat)?;
    let mut bundle = Bundle::new(format);
    bundle.dontcare = bits & format.dontcare_mask;
    for field in format.slots {
        bundle.slots[field.slot as usize] =
            Some(((bits >> field.lsb) & ((1u128 << field.width) - 1)) as u64);
    }
    Ok(bundle)
}

/// Encode raw operand bit values using the generated (possibly split) fields.
/// Register and immediate transforms are described by `Encoding::operands`.
/// Preserve instruction don't-care bits explicitly when re-encoding a decode.
pub fn encode_slot(
    encoding: &Encoding,
    operands: &[(&str, u64)],
    dontcare: u64,
) -> Result<u64, Error> {
    if dontcare & !encoding.dontcare_mask != 0 {
        return Err(Error::InvalidDontcare);
    }
    for (index, &(name, value)) in operands.iter().enumerate() {
        if operands[..index].iter().any(|&(n, _)| n == name) {
            return Err(Error::UnexpectedOperand);
        }
        let width = encoding
            .fields
            .iter()
            .filter(|f| f.name == name)
            .map(|f| f.source_lsb + f.width)
            .max()
            .ok_or(Error::UnexpectedOperand)?;
        if value >> width != 0 {
            let field = encoding.fields.iter().find(|f| f.name == name).unwrap();
            return Err(Error::OperandOverflow(field.name));
        }
    }
    let mut bits = encoding.value | dontcare;
    for field in encoding.fields {
        let value = operands
            .iter()
            .find(|&&(name, _)| name == field.name)
            .ok_or(Error::MissingOperand(field.name))?
            .1;
        bits |= ((value >> field.source_lsb) & ((1u64 << field.width) - 1)) << field.lsb;
    }
    Ok(bits)
}

/// Fixed opcode bits plus equality constraints from repeated operand slices.
/// A mask-only match would incorrectly decode arbitrary OR as the MOV_OR alias.
pub fn matches_encoding(encoding: &Encoding, payload: u64) -> bool {
    if payload >> encoding.slot.width() != 0 || payload & encoding.mask != encoding.value {
        return false;
    }
    for (index, field) in encoding.fields.iter().enumerate() {
        let value = ((payload >> field.lsb) & ((1u64 << field.width) - 1)) << field.source_lsb;
        let mask = ((1u64 << field.width) - 1) << field.source_lsb;
        for other in &encoding.fields[..index] {
            if field.name != other.name {
                continue;
            }
            let other_value =
                ((payload >> other.lsb) & ((1u64 << other.width) - 1)) << other.source_lsb;
            let other_mask = ((1u64 << other.width) - 1) << other.source_lsb;
            if (value ^ other_value) & mask & other_mask != 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::gen::ENCODINGS;
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn every_composite_format_boundaries() {
        for format in FORMATS {
            for maximum in [false, true] {
                let mut bundle = Bundle::new(format);
                bundle.dontcare = if maximum { format.dontcare_mask } else { 0 };
                for field in format.slots {
                    bundle
                        .set(
                            field.slot,
                            if maximum {
                                (1u64 << field.width) - 1
                            } else {
                                0
                            },
                        )
                        .unwrap();
                }
                let packed = bundle.pack().unwrap();
                let decoded = unpack(packed.as_slice()).unwrap();
                assert_eq!(decoded.format.name, format.name);
                assert_eq!(decoded.slots, bundle.slots);
                assert_eq!(decoded.dontcare, bundle.dontcare);
                assert_eq!(decoded.pack().unwrap().as_slice(), packed.as_slice());
                assert_eq!(
                    unpack(&packed.as_slice()[..packed.len as usize - 1]).unwrap_err(),
                    Error::Truncated
                );
            }
        }
    }

    #[test]
    fn rejects_illegal_slot_sets_and_values() {
        assert!(matches!(
            pack_slots(&[(Slot::Lng, 0), (Slot::Alu, 0)]),
            Err(Error::UnknownFormat)
        ));
        assert_eq!(
            pack_slots(&[(Slot::Alu, 0), (Slot::Alu, 0)]).unwrap_err(),
            Error::DuplicateSlot(Slot::Alu)
        );
        assert_eq!(
            pack_slots(&[(Slot::Ldb, 1 << 17)]).unwrap_err(),
            Error::PayloadOverflow(Slot::Ldb)
        );
        let mut bundle = Bundle::new(FORMATS.iter().find(|f| f.name == "I32_ALU").unwrap());
        assert_eq!(bundle.pack().unwrap_err(), Error::MissingSlot(Slot::Alu));
        bundle.set(Slot::Alu, 0).unwrap();
        bundle.dontcare = 1;
        assert_eq!(bundle.pack().unwrap_err(), Error::InvalidDontcare);
    }

    #[test]
    fn split_immediate_and_repeated_register_fields() {
        let mov = ENCODINGS.iter().find(|e| e.name == "MOVXM").unwrap();
        let bits = encode_slot(mov, &[("dst", 5), ("i", 0xdeadbeef)], 0).unwrap();
        assert_eq!(
            bits,
            ((0xdeadbeefu64 >> 12) << 22) | (5 << 15) | ((0xdeadbeefu64 & 0xfff) << 3) | 1
        );
        let mov_or = ENCODINGS.iter().find(|e| e.name == "MOV_OR").unwrap();
        let bits = encode_slot(mov_or, &[("s0", 17), ("d0", 3)], 0).unwrap();
        assert_eq!(bits, (17 << 15) | (3 << 10) | (17 << 5) | 0b01011);
        assert!(matches_encoding(mov_or, bits));
        assert!(!matches_encoding(mov_or, 43)); // OR with unequal source registers, not MOV_OR.
        assert!(matches!(
            encode_slot(mov, &[("dst", 128), ("i", 0)], 0),
            Err(Error::OperandOverflow("dst"))
        ));
    }

    fn collect_elfs(root: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect_elfs(&path, out);
            } else if path.extension().is_some_and(|e| e == "elf") {
                out.push(path);
            }
        }
    }
    fn u16_at(data: &[u8], offset: usize) -> usize {
        u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize
    }
    fn u32_at(data: &[u8], offset: usize) -> usize {
        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize
    }

    /// Every executable byte of every core ELF is decoded into format and slot
    /// operand fields and re-encoded. No raw-byte pass-through or unknown slots.
    #[test]
    fn vendor_corpus_byte_identical() {
        let Some(root) = crate::vendor_corpus().map(|r| r.join("cache")) else {
            return;
        };
        let root = root.as_path();
        let mut paths = Vec::new();
        collect_elfs(root, &mut paths);
        paths.sort();
        assert!(!paths.is_empty(), "vendor corpus has no core ELFs");
        let mut total = 0usize;
        let mut matched = 0usize;
        let mut code_bytes = 0usize;
        let mut format_counts = [0usize; 8];
        let mut unknown_slots = 0usize;
        for path in &paths {
            let elf = std::fs::read(path).unwrap();
            assert_eq!(&elf[..6], b"\x7fELF\x01\x01", "{}", path.display());
            let table = u32_at(&elf, 32);
            let stride = u16_at(&elf, 46);
            for index in 0..u16_at(&elf, 48) {
                let section = table + index * stride;
                if u32_at(&elf, section + 4) != 1 || u32_at(&elf, section + 8) & 4 == 0 {
                    continue;
                }
                let offset = u32_at(&elf, section + 16);
                let size = u32_at(&elf, section + 20);
                let text = &elf[offset..offset + size];
                code_bytes += size;
                let mut pc = 0;
                while pc < text.len() {
                    total += 1;
                    let mut bundle = unpack(&text[pc..]).unwrap_or_else(|e| {
                        panic!("{} section {index} pc {pc:#x}: {e}", path.display())
                    });
                    let len = bundle.format.bits as usize / 8;
                    format_counts[len / 2 - 1] += 1;
                    for field in bundle.format.slots {
                        let payload = bundle.slots[field.slot as usize].unwrap();
                        let Some(encoding) = ENCODINGS
                            .iter()
                            .find(|e| e.slot == field.slot && matches_encoding(e, payload))
                        else {
                            unknown_slots += 1;
                            if unknown_slots < 10 {
                                eprintln!(
                                    "unknown instruction {} pc {pc:#x} {:?} {payload:#x}",
                                    path.display(),
                                    field.slot
                                );
                            }
                            continue;
                        };
                        let mut operands = Vec::<(&str, u64)>::new();
                        for f in encoding.fields {
                            let value =
                                ((payload >> f.lsb) & ((1u64 << f.width) - 1)) << f.source_lsb;
                            if let Some((_, bits)) = operands.iter_mut().find(|(n, _)| *n == f.name)
                            {
                                *bits |= value;
                            } else {
                                operands.push((f.name, value));
                            }
                        }
                        let reencoded =
                            encode_slot(encoding, &operands, payload & encoding.dontcare_mask)
                                .unwrap();
                        assert_eq!(
                            reencoded,
                            payload,
                            "{} pc {pc:#x} {}",
                            path.display(),
                            encoding.name
                        );
                        bundle.slots[field.slot as usize] = Some(reencoded);
                    }
                    if bundle.pack().unwrap().as_slice() == &text[pc..pc + len] {
                        matched += 1;
                    }
                    pc += len;
                }
            }
        }
        eprintln!("vendor re-encode: {matched}/{total} bundles; {} ELFs; {code_bytes} executable bytes; formats 16..128 step16={format_counts:?}; unknown slots={unknown_slots}", paths.len());
        assert_eq!(unknown_slots, 0, "instruction table coverage incomplete");
        assert_eq!(matched, total);
    }
}
