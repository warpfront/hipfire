// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Immediate and printer semantics derived from llvm-aie AIEBaseDisassembler.h.
//! Lossless AIE2P decoding and LLVM-compatible assembly rendering.
//!
//! `decode` consumes one bundle at the supplied PC. Operand kinds, access direction,
//! and itinerary identifiers remain available for schedulers and CPU simulators.
//! `encode` reconstructs slot operands and preserves reserved/don't-care bits.
//! The offline CLI accepts ELF32 cores: `cargo run -p peacemaker-npu --bin peacemaker-npu-objdump -- CORE.elf`.
//! Oracle gate: `AIE2P_OBJDUMP=/path/to/llvm-objdump cargo test -p peacemaker-npu vendor_decode_oracle_roundtrip`.
use super::{bundle, gen};
use std::fmt;

#[derive(Debug, Clone)]
pub struct DecodedOperand {
    pub name: &'static str,
    pub kind: &'static str,
    /// Architectural immediate, or encoded register index.
    pub value: i64,
    pub output: bool,
}
#[derive(Debug)]
pub struct DecodedInst {
    pub encoding: &'static gen::Encoding,
    pub raw: u64,
    pub operands: Vec<DecodedOperand>,
}
pub struct DecodedBundle {
    pub pc: u64,
    pub len: usize,
    pub raw: bundle::Bundle,
    pub instructions: Vec<DecodedInst>,
}
#[derive(Debug)]
pub struct DecodeError(pub String);
impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DecodeError {}

fn transform(op: &gen::Operand) -> (u8, i64, bool, bool) {
    let mut args = op
        .decoder
        .split_once('<')
        .map(|(_, s)| s.trim_end_matches('>').split(','));
    let width = args
        .as_mut()
        .and_then(|a| a.next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(op.width);
    if op.decoder.starts_with("decodeSImmOperandXStep") {
        let mut args = args.unwrap();
        let step = args.next().unwrap().trim().parse().unwrap();
        let negative = matches!(args.next().unwrap().trim(), "1" | "true");
        let unsigned = matches!(args.next().unwrap().trim(), "1" | "true");
        (width, step, negative, !unsigned && !negative)
    } else {
        (width, 1, false, op.decoder.starts_with("decodeSImm"))
    }
}
fn immediate(op: &gen::Operand, raw: u64) -> i64 {
    let (width, step, negative, signed) = transform(op);
    let v = if signed && width > 0 {
        ((raw << (64 - width)) as i64) >> (64 - width)
    } else {
        raw as i64
    };
    let v = if negative { v - (1i64 << width) } else { v };
    v * step
}

pub fn decode(bytes: &[u8], pc: u64) -> Result<DecodedBundle, DecodeError> {
    let raw = bundle::unpack(bytes).map_err(|e| DecodeError(format!("{pc:#x}: {e:?}")))?;
    let mut instructions = Vec::new();
    for field in raw.format.slots {
        let bits =
            raw.slots[field.slot as usize].ok_or_else(|| DecodeError("missing slot".into()))?;
        let encoding = gen::ENCODINGS
            .iter()
            .filter(|e| e.slot == field.slot && bundle::matches_encoding(e, bits))
            .filter(|e| {
                e.operands.iter().all(|op| {
                    let value = operand_value(e, op, bits);
                    op.registers.is_empty() || resolve_register(op, value).is_some()
                })
            })
            .max_by_key(|e| specificity(e))
            .ok_or_else(|| {
                DecodeError(format!("{pc:#x}: unknown {:?} slot {bits:#x}", field.slot))
            })?;
        let operands = encoding
            .operands
            .iter()
            .map(|op| {
                let value = operand_value(encoding, op, bits);
                DecodedOperand {
                    name: op.name,
                    kind: op.kind,
                    value: if op.registers.is_empty() {
                        immediate(op, value)
                    } else {
                        value as i64
                    },
                    output: op.output,
                }
            })
            .collect();
        instructions.push(DecodedInst {
            encoding,
            raw: bits,
            operands,
        });
    }
    Ok(DecodedBundle {
        pc,
        len: raw.format.bits as usize / 8,
        raw,
        instructions,
    })
}
fn specificity(e: &gen::Encoding) -> u32 {
    let mut count = e.mask.count_ones();
    for (index, field) in e.fields.iter().enumerate() {
        let mask = ((1u64 << field.width) - 1) << field.source_lsb;
        let prior = e.fields[..index]
            .iter()
            .filter(|f| f.name == field.name)
            .fold(0, |m, f| m | (((1u64 << f.width) - 1) << f.source_lsb));
        count += (mask & prior).count_ones();
    }
    count
}
fn operand_bits(e: &gen::Encoding, name: &str, bits: u64) -> u64 {
    e.fields.iter().filter(|f| f.name == name).fold(0, |v, f| {
        v | (((bits >> f.lsb) & ((1u64 << f.width) - 1)) << f.source_lsb)
    })
}
fn operand_value(e: &gen::Encoding, op: &gen::Operand, bits: u64) -> u64 {
    if e.fields.iter().any(|f| f.name == op.name) {
        return operand_bits(e, op.name, bits);
    }
    if op.registers.len() == 1 {
        return op.registers[0].value;
    }
    for constraint in e.constraints.split(',') {
        if let Some((a, b)) = constraint.split_once('=') {
            let a = a.trim().trim_start_matches('$');
            let b = b.trim().trim_start_matches('$');
            if a == op.name {
                return operand_bits(e, b, bits);
            }
            if b == op.name {
                return operand_bits(e, a, bits);
            }
        }
    }
    0
}
impl DecodedInst {
    pub fn assembly(&self) -> String {
        let mut text = String::with_capacity(self.encoding.asm.len());
        self.write_assembly(&mut text);
        text
    }
    fn write_assembly(&self, text: &mut String) {
        use fmt::Write;
        let mut rest = self.encoding.asm;
        while let Some(index) = rest.find('$') {
            text.push_str(&rest[..index]);
            rest = &rest[index + 1..];
            let braced = rest.starts_with('{');
            if braced {
                rest = &rest[1..];
            }
            let end = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            let name = &rest[..end];
            rest = &rest[end..];
            if braced {
                rest = rest.strip_prefix('}').expect("generated operand brace");
            }
            let op = self
                .operands
                .iter()
                .find(|op| op.name == name)
                .expect("generated assembly operand");
            let descriptor = self
                .encoding
                .operands
                .iter()
                .find(|p| p.name == op.name)
                .unwrap();
            if let Some(reg) = resolve_register(descriptor, op.value as u64) {
                text.push_str(reg.asm);
            } else {
                let v = op.value;
                if descriptor.printer.starts_with("printImmOffset<") {
                    let offset = descriptor
                        .printer
                        .trim_start_matches("printImmOffset<")
                        .trim_end_matches('>')
                        .trim_start_matches("/*offset=*/")
                        .parse::<i64>()
                        .unwrap();
                    write!(text, "{}", v + offset).unwrap();
                } else if v < 0 {
                    write!(text, "-0x{:x}", v.unsigned_abs()).unwrap();
                } else {
                    write!(text, "0x{v:x}").unwrap();
                }
            }
        }
        text.push_str(rest);
    }
}
impl DecodedBundle {
    /// Preserve all reserved and don't-care bits, not just canonical operand encodings.
    pub fn encode(&self) -> Result<Vec<u8>, DecodeError> {
        let mut raw = bundle::Bundle {
            format: self.raw.format,
            slots: self.raw.slots,
            dontcare: self.raw.dontcare,
        };
        for inst in &self.instructions {
            let e = inst.encoding;
            let mut bits = e.value | (inst.raw & e.dontcare_mask);
            for field in e.fields {
                let descriptor = e
                    .operands
                    .iter()
                    .find(|op| op.name == field.name)
                    .ok_or_else(|| {
                        DecodeError(format!("{}: missing field operand {}", e.name, field.name))
                    })?;
                let operand = inst
                    .operands
                    .iter()
                    .find(|op| op.name == field.name)
                    .ok_or_else(|| {
                        DecodeError(format!(
                            "{}: missing decoded operand {}",
                            e.name, field.name
                        ))
                    })?;
                let value = if descriptor.registers.is_empty() {
                    let (_, step, _, _) = transform(descriptor);
                    (operand.value / step) as u64
                } else {
                    operand.value as u64
                };
                bits |= ((value >> field.source_lsb) & ((1u64 << field.width) - 1)) << field.lsb;
            }
            raw.slots[e.slot as usize] = Some(bits);
        }
        bundle::pack(&raw)
            .map(|p| p.as_slice().to_vec())
            .map_err(|e| DecodeError(format!("{e:?}")))
    }
    pub fn assembly(&self) -> String {
        let mut text = String::new();
        for (index, inst) in self.instructions.iter().enumerate() {
            if index > 0 {
                text.push_str("; ");
            }
            inst.write_assembly(&mut text);
        }
        text
    }
}

pub struct ExecutableSection<'a> {
    pub name: &'a str,
    pub address: u64,
    pub bytes: &'a [u8],
}
/// Read executable PROGBITS sections of little-endian ELF32 AIE core images.
pub fn executable_sections(bytes: &[u8]) -> Result<Vec<ExecutableSection<'_>>, DecodeError> {
    let err = || DecodeError("invalid or unsupported ELF32 little-endian core image".into());
    let slice = |start: usize, len: usize| {
        bytes
            .get(start..start.checked_add(len).ok_or_else(err)?)
            .ok_or_else(err)
    };
    if slice(0, 6)? != b"\x7fELF\x01\x01" {
        return Err(err());
    }
    let u16at = |p| -> Result<usize, DecodeError> {
        Ok(u16::from_le_bytes(slice(p, 2)?.try_into().unwrap()) as usize)
    };
    let u32at = |p| -> Result<usize, DecodeError> {
        Ok(u32::from_le_bytes(slice(p, 4)?.try_into().unwrap()) as usize)
    };
    let base = u32at(32)?;
    let stride = u16at(46)?;
    let count = u16at(48)?;
    let strings = u16at(50)?;
    if stride < 40 || strings >= count {
        return Err(err());
    }
    slice(base, stride.checked_mul(count).ok_or_else(err)?)?;
    let shstr = base + strings * stride;
    let names = slice(u32at(shstr + 16)?, u32at(shstr + 20)?)?;
    let mut sections = Vec::new();
    for index in 0..count {
        let sh = base + index * stride;
        if u32at(sh + 4)? != 1 || u32at(sh + 8)? & 4 == 0 {
            continue;
        }
        let name = names.get(u32at(sh)?..).ok_or_else(err)?;
        let end = name.iter().position(|b| *b == 0).ok_or_else(err)?;
        sections.push(ExecutableSection {
            name: std::str::from_utf8(&name[..end]).map_err(|_| err())?,
            address: u32at(sh + 12)? as u64,
            bytes: slice(u32at(sh + 16)?, u32at(sh + 20)?)?,
        });
    }
    Ok(sections)
}

/// Decode an architectural register index. Invalid encodings return None.
pub fn reg_name(kind: &str, value: u64) -> Option<&'static str> {
    gen::ENCODINGS
        .iter()
        .flat_map(|e| e.operands.iter())
        .find(|o| o.kind == kind && o.width > 0)
        .or_else(|| {
            gen::ENCODINGS
                .iter()
                .flat_map(|e| e.operands.iter())
                .find(|o| o.kind == kind)
        })
        .and_then(|o| resolve_register(o, value))
        .map(|r| r.asm)
}
/// Register maps retain full HW encodings; operands consume only their low width bits.
pub fn resolve_register(op: &gen::Operand, value: u64) -> Option<&'static gen::RegisterEncoding> {
    let width = if op.width > 0 || op.registers.len() == 1 {
        op.width
    } else {
        gen::ENCODINGS
            .iter()
            .flat_map(|e| e.operands.iter())
            .find(|o| o.kind == op.kind && o.width > 0)
            .map_or(0, |o| o.width)
    };
    let mask = if width == 0 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    op.registers.iter().find(|r| r.value & mask == value)
}
