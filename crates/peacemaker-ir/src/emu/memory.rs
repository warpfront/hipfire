// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::{Memory, Result, State, WriteSet};
/// One LDS byte range touched by an instruction (one entry per active lane access): the happens-before race
/// detector's input. Atomics are read-modify-write and carry `atomic`.
#[derive(Clone, Copy, Debug)]
pub(super) struct LdsTouch {
    pub lo: usize,
    pub len: usize,
    pub write: bool,
    pub atomic: bool,
}
use crate::{
    operand::{ImmField, Operand, VmemToken},
    Inst,
};
/// Access size from the data-size token (`b8 u8 i8 b16 u16 i16 b32 u32 f32 b64 b96 b128 b256`); `d16`, `hi`,
/// `rtn`, `2addr` and the address-space prefix are modifiers.
fn width(name: &str) -> Result<usize> {
    for token in name.split('_') {
        match token {
            "b512" => return Ok(64),
            "b256" => return Ok(32),
            "b128" => return Ok(16),
            "b96" => return Ok(12),
            "b64" => return Ok(8),
            "b32" | "u32" | "i32" | "f32" => return Ok(4),
            "b16" | "u16" | "i16" => return Ok(2),
            "b8" | "u8" | "i8" => return Ok(1),
            _ => (),
        }
    }
    Err(format!("unknown memory width {name}"))
}
/// Width in bits of a signed (`i8`/`i16`) load field; unsigned and untyped loads zero-extend.
fn sign_bits(name: &str) -> Option<u32> {
    name.split('_').find_map(|token| match token {
        "i8" => Some(8),
        "i16" => Some(16),
        _ => None,
    })
}
#[derive(Clone, Copy)]
enum Atomic {
    AddU32,
    OrB32,
    AddF32,
}
fn ds_atomic(name: &str) -> Option<(Atomic, bool)> {
    Some(match name {
        "ds_add_u32" => (Atomic::AddU32, false),
        "ds_add_rtn_u32" => (Atomic::AddU32, true),
        "ds_or_b32" => (Atomic::OrB32, false),
        "ds_add_f32" => (Atomic::AddF32, false),
        _ => return None,
    })
}
/// LDS read-modify-write: `tmp = MEM[addr]; MEM[addr] op= DATA; RETURN_DATA = tmp`. The ISA does not state how lanes
/// of one wave that target the same dword are ordered, so a returning or floating-point atomic with two active lanes
/// on one dword is an error; integer add/or without a return are order independent and apply lane by lane.
/// `ds_add_f32` is host IEEE binary32 (the ISA: "handles NAN/INF/denorm"; NaN results/operands carry no stated
/// payload rule and are an error). Addresses must be dword aligned.
fn ds_atomic_execute(
    kind: Atomic,
    returns: bool,
    name: &str,
    i: &Inst,
    s: &mut State,
    lds: &mut [u8],
    mut defined: Option<&mut WriteSet>,
    mut log: Option<&mut Vec<LdsTouch>>,
) -> Result<()> {
    let ops = &i.operands;
    let (addr_op, data_op) = if returns { (1, 2) } else { (0, 1) };
    if ops.len() <= data_op {
        return Err(format!("{name}: missing operand"));
    }
    let offset: u64 = ops
        .iter()
        .map(|op| {
            if let Operand::Imm(ImmField::DsOffset(n)) = op {
                u64::from(*n)
            } else {
                0
            }
        })
        .sum();
    let mut seen = std::collections::HashSet::new();
    for lane in 0..s.lanes {
        if s.exec & (1u64 << lane) == 0 {
            continue;
        }
        // LDS addresses are 32-bit (VGPR + 16-bit offset, carry out discarded).
        let addr = u64::from((s.read(&ops[addr_op], lane, 0)? as u32).wrapping_add(offset as u32));
        if addr & 3 != 0 {
            return Err(format!("{name}: unaligned LDS atomic address {addr:#x}"));
        }
        if (returns || matches!(kind, Atomic::AddF32)) && !seen.insert(addr) {
            return Err(format!(
                "{name}: two active lanes target LDS dword {addr:#x}; the ISA does not order them"
            ));
        }
        let slot_at = addr as usize;
        if let Some(w) = defined.as_deref() {
            if let Some(b) = w.first_unwritten(slot_at, 4) {
                return Err(format!(
                    "{name}: LDS atomic reads byte {b:#x} never written in this workgroup"
                ));
            }
        }
        let slot = lds
            .get_mut(slot_at..slot_at + 4)
            .ok_or_else(|| format!("LDS atomic out of bounds {addr:#x}"))?;
        let old = u32::from_le_bytes((&*slot).try_into().unwrap());
        let data = s.read(&ops[data_op], lane, 0)?;
        let new = match kind {
            Atomic::AddU32 => old.wrapping_add(data),
            Atomic::OrB32 => old | data,
            Atomic::AddF32 => {
                let (x, y) = (f32::from_bits(old), f32::from_bits(data));
                let r = x + y;
                if x.is_nan() || y.is_nan() || r.is_nan() {
                    return Err(format!(
                        "{name}: NaN operand or result: payload is not specified"
                    ));
                }
                r.to_bits()
            }
        };
        slot.copy_from_slice(&new.to_le_bytes());
        if let Some(w) = defined.as_deref_mut() {
            w.mark(slot_at, 4);
        }
        if let Some(l) = log.as_deref_mut() {
            l.push(LdsTouch {
                lo: slot_at,
                len: 4,
                write: true,
                atomic: true,
            });
        }
        if returns {
            s.put(&ops[0], lane, 0, old)?;
        }
    }
    Ok(())
}
/// Validates the MUBUF operand shape and returns whether OFFEN supplies a vector offset.
/// IDXEN is not representable in the IR: the decoder narrows the VADDR register to one
/// dword for IDXEN xor OFFEN, so a one-dword VADDR without OFFEN, or a wider VADDR
/// with OFFEN (index+offset pair), can only be an index-addressed access and is refused.
fn buffer_shape(ops: &[Operand]) -> Result<bool> {
    if ops.len() < 4 {
        return Err("buffer instruction requires data, address, SRD and soffset operands".into());
    }
    for op in &ops[4..] {
        if !matches!(
            op,
            Operand::Vmem(VmemToken::Offen)
                | Operand::Imm(ImmField::VmemOffset(_))
                | Operand::Scope(_)
                | Operand::CacheTh(_)
        ) {
            return Err(format!("unsupported buffer operand {op}"));
        }
    }
    let offen = ops[4..]
        .iter()
        .any(|o| matches!(o, Operand::Vmem(VmemToken::Offen)));
    match (&ops[1], offen) {
        (Operand::Reg(r), true) if r.kind == crate::reg::Kind::V && r.len == 1 => Ok(true),
        (Operand::Reg(_), true) => {
            Err("buffer OFFEN requires one 32-bit VGPR; IDXEN addressing unsupported".into())
        }
        (Operand::Vmem(VmemToken::Off), false) => Ok(false),
        (Operand::Reg(r), false) if r.kind == crate::reg::Kind::V && r.len > 1 => Ok(false),
        (Operand::Reg(_), false) => Err("buffer IDXEN addressing unsupported".into()),
        _ => Err("unsupported buffer address operand".into()),
    }
}
/// RDNA3/RDNA4 V# word3 (descriptor bits 127:96): DST_SEL[11:0] and FORMAT[17:12] are inert for
/// raw accesses (FORMAT==0 is the unbound-resource encoding and is refused), bits 25:24 are compression
/// flags accepted only because captured SRDs (0x31004000) set bit 24; the docs do not state that they are
/// raw-inert. OOB_SELECT is [29:28]. Everything else changes addressing and is refused.
fn raw_srd_word3(word3: u32) -> Result<()> {
    if word3 >> 30 != 0 {
        return Err("buffer SRD TYPE must be 0".into());
    }
    if word3 >> 28 & 3 != 3 {
        return Err("only raw buffer OOB mode 3 is supported".into());
    }
    // §9.6 Unbound Resources: FORMAT==0 (INVALID) without ADD_TID makes loads return zero and drops stores.
    // That rule is not modeled, so refuse rather than perform a normal memory access.
    if word3 >> 12 & 0x3f == 0 {
        return Err("buffer SRD FORMAT 0 is an unbound resource; unmodeled".into());
    }
    if word3 & (1 << 23) != 0 {
        return Err("buffer SRD ADD_TID_ENABLE unsupported".into());
    }
    if word3 & (3 << 21) != 0 {
        return Err("buffer SRD INDEX_STRIDE/swizzle unsupported".into());
    }
    if word3 & 0x0c0c_0000 != 0 {
        return Err("buffer SRD stride-scale/compression-access bits unsupported".into());
    }
    if word3 & (1 << 20) != 0 {
        return Err("buffer SRD reserved bit 20 set".into());
    }
    Ok(())
}
/// `global_atomic_add_u32 [vdst,] vaddr, vdata, saddr|off`: per active lane `tmp = MEM[addr]; MEM[addr] = tmp + DATA.u32
/// (wrapping); RETURN_DATA = tmp` (RDNA3 ISA `GLOBAL_ATOMIC_ADD_U32`). `glc` selects the returning form, which decodes
/// with a leading VDST. The ISA does not say how lanes of one wave that hit the same dword are ordered: the integer
/// adds commute, so the non-returning form applies them lane by lane, but a returning form with two active lanes on one
/// dword (each would observe a different intermediate value) is an error. Memory is written with `Memory::write` and the
/// returned dword lands through `State::put`, like the generic loads and stores.
fn global_atomic_add_u32(i: &Inst, s: &mut State, mem: &mut Memory) -> Result<()> {
    let regs: Vec<&Operand> = i
        .operands
        .iter()
        .filter(|o| !matches!(o, Operand::Imm(_)))
        .collect();
    let ret = match regs.len() {
        4 => true,
        3 => false,
        _ => return Err("global_atomic_add_u32: unexpected operand shape".into()),
    };
    let (vaddr, vdata, saddr) = (
        regs[usize::from(ret)],
        regs[1 + usize::from(ret)],
        regs[2 + usize::from(ret)],
    );
    let mut offset = 0i64;
    for op in &i.operands {
        if let Operand::Imm(ImmField::VmemOffset(n)) = op {
            offset += i64::from(*n);
        }
    }
    // SADDR form: a 32-bit VGPR byte offset added to the SGPR-pair base; `off`: a 64-bit VGPR address.
    let saddr_pair =
        matches!(saddr, Operand::Reg(r) if r.kind == crate::reg::Kind::S && r.len == 2);
    let mut lanes: Vec<(usize, u64)> = Vec::new();
    for lane in 0..s.lanes {
        if s.exec & (1u64 << lane) == 0 {
            continue;
        }
        let addr = if saddr_pair {
            s.read64(saddr, 0)?
                .wrapping_add(u64::from(s.read(vaddr, lane, 0)?))
        } else {
            s.read64(vaddr, lane)?
        }
        .wrapping_add_signed(offset);
        if ret && lanes.iter().any(|(_, a)| *a == addr) {
            return Err("global_atomic_add_u32: two active lanes return from one dword: the order is not specified".into());
        }
        lanes.push((lane, addr));
    }
    for (lane, addr) in lanes {
        let old = u32::from_le_bytes(mem.read(addr, 4)?.try_into().unwrap());
        mem.write(
            addr,
            &old.wrapping_add(s.read(vdata, lane, 0)?).to_le_bytes(),
        )?;
        if ret {
            s.put(regs[0], lane, 0, old)?;
        }
    }
    Ok(())
}
/// `scratch_{load,store}_*`: per-lane private memory (RDNA3 ISA ch. 11, "Scratch"). The lane's byte offset is
/// `SGPR (when SADDR names one) + VGPR (when ADDR names one: SVE) + OFFSET`; every lane owns
/// `private_segment_fixed_size` bytes (`State::scratch`) that only it reads or writes, so there is no cross-lane or
/// cross-wave visibility to model, and the hardware's swizzling of scratch into global memory is not observable by the
/// lane. An access outside the lane's private segment is an error. Size tokens, sign extension and `d16` merging follow
/// the generic global/flat path.
fn scratch_access(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let n = width(name)?;
    let load = name.contains("load");
    let regs: Vec<&Operand> = i
        .operands
        .iter()
        .filter(|o| !matches!(o, Operand::Imm(_)))
        .collect();
    if regs.len() != 3 {
        return Err(format!("{name}: unexpected operand shape"));
    }
    let (data, vaddr, saddr) = if load {
        (regs[0], regs[1], regs[2])
    } else {
        (regs[1], regs[0], regs[2])
    };
    let mut offset = 0i64;
    for op in &i.operands {
        if let Operand::Imm(ImmField::VmemOffset(k)) = op {
            offset += i64::from(*k);
        }
    }
    for lane in 0..s.lanes {
        if s.exec & (1u64 << lane) == 0 {
            continue;
        }
        let mut addr = offset;
        if matches!(saddr, Operand::Reg(r) if r.kind == crate::reg::Kind::S) {
            addr += i64::from(s.read(saddr, 0, 0)?);
        }
        if matches!(vaddr, Operand::Reg(r) if r.kind == crate::reg::Kind::V) {
            addr += i64::from(s.read(vaddr, lane, 0)?);
        }
        let size = s.scratch.get(lane).map_or(0, Vec::len);
        let at = usize::try_from(addr)
            .ok()
            .filter(|a| a.checked_add(n).is_some_and(|end| end <= size))
            .ok_or_else(|| {
                format!("{name}: private offset {addr:#x}+{n} is outside lane {lane}'s {size}-byte private segment")
            })?;
        if load {
            let mut bytes = [0u8; 16];
            bytes[..n].copy_from_slice(&s.scratch[lane][at..at + n]);
            if let Some(bits) = sign_bits(name) {
                let raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                bytes[0..4].copy_from_slice(
                    &((((raw << (32 - bits)) as i32) >> (32 - bits)) as u32).to_le_bytes(),
                );
            }
            for w in 0..n.div_ceil(4) {
                let mut val = u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap());
                if name.contains("d16_hi") {
                    val = (s.read(data, lane, w)? & 0xffff) | (val << 16);
                } else if name.contains("d16") {
                    val = (s.read(data, lane, w)? & 0xffff_0000) | (val & 0xffff);
                }
                s.put(data, lane, w, val)?;
            }
        } else {
            let mut bytes = [0u8; 16];
            for w in 0..n.div_ceil(4) {
                let mut val = s.read(data, lane, w)?;
                if name.contains("d16_hi") {
                    val >>= 16;
                }
                bytes[w * 4..w * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
            s.scratch[lane][at..at + n].copy_from_slice(&bytes[..n]);
        }
    }
    Ok(())
}
pub(super) fn execute(
    name: &str,
    i: &Inst,
    s: &mut State,
    mem: &mut Memory,
    lds: &mut [u8],
) -> Result<()> {
    execute_tracked(name, i, s, mem, lds, None, None)
}
/// `defined`: when present, the per-workgroup set of LDS bytes written so far; a read of any other byte is an error.
pub(super) fn execute_tracked(
    name: &str,
    i: &Inst,
    s: &mut State,
    mem: &mut Memory,
    lds: &mut [u8],
    mut defined: Option<&mut WriteSet>,
    mut log: Option<&mut Vec<LdsTouch>>,
) -> Result<()> {
    if i.mods.ds.gds {
        return Err("GDS not modeled".into());
    }
    let ops = &i.operands;
    // `s_prefetch_data sbase, ioffset, soffset, len`: scalar L2 prefetch of (len + 1) * 128 bytes (len = the 5-bit imm
    // in SDATA[4:0], word0 bits 10:6, kept as the honored `smem_len` field; LLVM: "prefetch (len+1)*128 bytes") at
    // SGPR-pair base + SGPR soffset (u32, when not `null`) + the sign-extended 24-bit ioffset. No architectural effect
    // and no counter (the table row has none): a no-op that is FAIL-CLOSED on the address. A prefetched range that
    // does not lie inside ONE mapped (guarded) region is reported as an unmapped access: an unmapped prefetch is
    // unsafe until proven otherwise. `s_prefetch_data_pc_rel` and `s_buffer_prefetch_data` have no table row (their
    // base is the PC / a buffer SRD): the decoder refuses them.
    if name == "s_prefetch_data" {
        let mut addr = s.read64(ops.first().ok_or("s_prefetch_data: missing base")?, 0)?;
        for op in &ops[1..] {
            addr = match op {
                Operand::Imm(ImmField::SmemOffset(k) | ImmField::SmemDisplacement(k)) => {
                    addr.wrapping_add_signed(i64::from(*k))
                }
                Operand::Reg(_) => addr.wrapping_add(u64::from(s.read(op, 0, 0)?)),
                _ => return Err(format!("s_prefetch_data: unexpected operand {op}")),
            };
        }
        let lines = match &i.fields {
            crate::inst::FormFields::Bits { honored, .. } => honored
                .iter()
                .find(|f| f.name == "smem_len")
                .map_or(0, |f| (f.value >> 6) as usize),
            _ => 0,
        } + 1;
        let len = lines * 128;
        return mem
            .locate(addr, len)
            .map(|_| ())
            .map_err(|_| format!("s_prefetch_data: prefetched range {addr:#x}+{len} is not inside a mapped buffer (unmapped memory access)"));
    }
    let n = width(name)?;
    if name.starts_with("s_load_") {
        let mut off = 0u64;
        for op in &ops[2..] {
            off = off.wrapping_add(u64::from(s.read(op, 0, 0)?));
        }
        let addr = s.read64(&ops[1], 0)?.wrapping_add(off);
        for w in 0..n / 4 {
            let bytes = mem.read(addr + 4 * w as u64, 4)?;
            s.put(&ops[0], 0, w, u32::from_le_bytes(bytes.try_into().unwrap()))?;
        }
        return Ok(());
    }
    if name == "ds_swizzle_b32" {
        let pat = ops
            .iter()
            .find_map(|o| {
                if let Operand::Imm(ImmField::DsOffset(n)) = o {
                    Some(*n)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let mut values = [0; 64];
        for (lane, val) in values.iter_mut().enumerate() {
            // BitMode: and/or/xor masks act on the lane's index inside its 32-lane group (the upper half of a wave64 is
            // its own group); QDMode permutes within each quad.
            let src = if pat & 0x8000 != 0 {
                (lane & !3) | usize::from((pat >> (2 * (lane & 3))) & 3)
            } else {
                (lane & !31)
                    | (((lane & usize::from(pat & 31)) | usize::from((pat >> 5) & 31))
                        ^ usize::from((pat >> 10) & 31))
            };
            *val = if src < s.lanes && s.exec & (1u64 << src) != 0 {
                s.read(&ops[1], src, 0)?
            } else {
                0
            };
        }
        for (lane, val) in values.into_iter().enumerate() {
            if s.exec & (1u64 << lane) != 0 {
                s.put(&ops[0], lane, 0, val)?;
            }
        }
        return Ok(());
    }
    // Backward permute on the LDS crossbar: lane i reads DATA0 of lane ((ADDR[i]+OFFSET)/4) mod 32
    // (high bits ignored); a disabled source lane reads 0. No LDS memory is touched.
    if name == "ds_bpermute_b32" {
        let offset = ops
            .iter()
            .find_map(|o| {
                if let Operand::Imm(ImmField::DsOffset(n)) = o {
                    Some(u32::from(*n))
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let mut values = [0; 64];
        for (lane, val) in values.iter_mut().enumerate() {
            if s.exec & (1u64 << lane) == 0 {
                continue;
            }
            let src = (s.read(&ops[1], lane, 0)?.wrapping_add(offset) / 4) as usize % s.lanes;
            *val = if s.exec & (1u64 << src) != 0 {
                s.read(&ops[2], src, 0)?
            } else {
                0
            };
        }
        for (lane, val) in values.into_iter().enumerate() {
            if s.exec & (1u64 << lane) != 0 {
                s.put(&ops[0], lane, 0, val)?;
            }
        }
        return Ok(());
    }
    if name.starts_with("scratch_") {
        return scratch_access(name, i, s);
    }
    if name == "global_atomic_add_u32" {
        return global_atomic_add_u32(i, s, mem);
    }
    if let Some((kind, returns)) = ds_atomic(name) {
        return ds_atomic_execute(kind, returns, name, i, s, lds, defined, log);
    }
    let load = name.contains("load");
    let ds = name.starts_with("ds_");
    let buffer = name.starts_with("buffer_");
    let two = name.contains("2addr");
    for lane in 0..s.lanes {
        if s.exec & (1u64 << lane) == 0 {
            continue;
        }
        let mut offset = 0i64;
        let mut off0 = 0usize;
        let mut off1 = 0usize;
        for op in ops {
            match op {
                Operand::Imm(ImmField::VmemOffset(n)) => offset += i64::from(*n),
                Operand::Imm(ImmField::DsOffset(n)) => offset += i64::from(*n),
                Operand::Imm(ImmField::DsOffset0(n)) => off0 = usize::from(*n),
                Operand::Imm(ImmField::DsOffset1(n)) => off1 = usize::from(*n),
                _ => (),
            }
        }
        let mut addr;
        let mut raw_bounds = None;
        if ds {
            // LDS addresses are 32-bit (VGPR + 16-bit offset, carry out discarded): hipcc folds a negative base plus
            // a large constant into the offset field.
            addr = u64::from(
                s.read(&ops[usize::from(load)], lane, 0)?
                    .wrapping_add(offset as u32),
            );
        } else if buffer {
            let offen = buffer_shape(ops)?;
            let Operand::Reg(srd) = ops[2] else {
                return Err("buffer SRD missing".into());
            };
            if srd.kind != crate::reg::Kind::S || srd.len != 4 {
                return Err("buffer SRD requires four scalar registers".into());
            }
            let r = usize::from(srd.base);
            let word1 = s.s[r + 1];
            let word3 = s.s[r + 3];
            if word1 >> 16 != 0 {
                return Err("strided or swizzled buffer descriptor unsupported".into());
            }
            raw_srd_word3(word3)?;
            // Raw byte-addressed SRDs: base[47:0], NUM_RECORDS in bytes.
            let base = u64::from(s.s[r]) | (u64::from(word1 & 0xffff) << 32);
            let vaddr = if offen {
                u64::from(s.read(&ops[1], lane, 0)?)
            } else {
                0
            };
            let soff = u64::from(s.read(&ops[3], lane, 0)?);
            let relative = vaddr.wrapping_add(soff).wrapping_add_signed(offset);
            raw_bounds = Some((relative, u64::from(s.s[r + 2])));
            addr = base.wrapping_add(relative);
        } else if name.starts_with("global_") || name.starts_with("flat_") {
            let a = if load { 1 } else { 0 };
            // SADDR form: a 32-bit VGPR byte offset added to the SGPR-pair base (the RDNA ISA rule); otherwise a 64-bit VGPR address.
            let saddr = matches!(ops.get(2),Some(Operand::Reg(r)) if r.kind==crate::reg::Kind::S&&r.len==2)
                && matches!(&ops[a],Operand::Reg(v) if v.kind==crate::reg::Kind::V&&v.len==1);
            if saddr {
                addr = s
                    .read64(&ops[2], lane)?
                    .wrapping_add(u64::from(s.read(&ops[a], lane, 0)?))
                    .wrapping_add_signed(offset);
            } else {
                addr = s.read64(&ops[a], lane)?.wrapping_add_signed(offset);
                if let Some(Operand::Reg(r)) = ops.get(2) {
                    if r.kind == crate::reg::Kind::S {
                        addr = addr.wrapping_add(s.read64(&ops[2], lane)?);
                    }
                }
            }
        } else {
            return Err(format!("memory opcode {name} unsupported"));
        }
        // Generic (flat) addressing: an address in the shared aperture is an LDS access at its low 32 bits, with the same
        // bounds, defined-byte and race logging as `ds_*`; any other address is global (an unmapped one is an error).
        let ds = if name.starts_with("flat_") && (addr >> 32) as u32 == super::SHARED_APERTURE_HI {
            addr &= 0xffff_ffff;
            true
        } else {
            ds
        };
        let data = if load {
            &ops[0]
        } else if buffer {
            &ops[0]
        } else {
            &ops[1]
        };
        let count = if two { 2 } else { 1 };
        // `*_stride64`: the two element offsets scale by 64 elements instead of 1 (OFFSETn * 64 * size).
        let stride = if name.contains("stride64") { 64 } else { 1 };
        for part in 0..count {
            let cur =
                if two {
                    if ds {
                        u64::from((addr as u32).wrapping_add(
                            (if part == 0 { off0 } else { off1 } * stride * n) as u32,
                        ))
                    } else {
                        addr + (if part == 0 { off0 } else { off1 } * stride * n) as u64
                    }
                } else {
                    addr
                };
            // A 2addr load returns both elements packed in one VDST; a 2addr store reads DATA0 / DATA1 from their own
            // (not necessarily adjacent) operands.
            let (data, data_word) = if two && !load {
                (&ops[1 + part], 0)
            } else {
                (data, part * n.div_ceil(4))
            };
            if load {
                let mut bytes = [0u8; 16];
                for offset in (0..n).step_by(4) {
                    let size = (n - offset).min(4);
                    if raw_bounds.is_some_and(|(relative, limit)| {
                        relative
                            .checked_add((offset + size) as u64)
                            .is_none_or(|end| end > limit)
                    }) {
                        continue;
                    }
                    let src = if ds {
                        let lo = cur as usize + offset;
                        let bytes = lds
                            .get(lo..lo + size)
                            .ok_or_else(|| format!("LDS read out of bounds {cur:#x}+{n}"))?;
                        if let Some(b) =
                            defined.as_deref().and_then(|w| w.first_unwritten(lo, size))
                        {
                            return Err(format!(
                                "LDS read of byte {b:#x} never written in this workgroup (access {cur:#x}+{n})"
                            ));
                        }
                        if let Some(l) = log.as_deref_mut() {
                            l.push(LdsTouch {
                                lo,
                                len: size,
                                write: false,
                                atomic: false,
                            });
                        }
                        bytes
                    } else {
                        mem.read(cur + offset as u64, size)?
                    };
                    bytes[offset..offset + size].copy_from_slice(src);
                }
                if let Some(bits) = sign_bits(name) {
                    // flat_load_i8 / *_i16: sign-extend the loaded field (to 16 bits for d16 forms, merged below).
                    let raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                    bytes[0..4].copy_from_slice(
                        &((((raw << (32 - bits)) as i32) >> (32 - bits)) as u32).to_le_bytes(),
                    );
                }
                for w in 0..n.div_ceil(4) {
                    let mut val = u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap());
                    if name.contains("d16_hi") {
                        val = (s.read(data, lane, data_word + w)? & 0xffff) | (val << 16);
                    } else if name.contains("d16") {
                        val = (s.read(data, lane, data_word + w)? & 0xffff0000) | (val & 0xffff);
                    }
                    s.put(data, lane, data_word + w, val)?;
                }
            } else {
                let mut bytes = [0u8; 16];
                for w in 0..n.div_ceil(4) {
                    let mut val = s.read(data, lane, data_word + w)?;
                    if name.contains("d16_hi") {
                        val >>= 16;
                    }
                    bytes[w * 4..w * 4 + 4].copy_from_slice(&val.to_le_bytes());
                }
                for offset in (0..n).step_by(4) {
                    let size = (n - offset).min(4);
                    if raw_bounds.is_some_and(|(relative, limit)| {
                        relative
                            .checked_add((offset + size) as u64)
                            .is_none_or(|end| end > limit)
                    }) {
                        continue;
                    }
                    if ds {
                        let lo = cur as usize + offset;
                        lds.get_mut(lo..lo + size)
                            .ok_or_else(|| format!("LDS write out of bounds {cur:#x}+{n}"))?
                            .copy_from_slice(&bytes[offset..offset + size]);
                        if let Some(w) = defined.as_deref_mut() {
                            w.mark(lo, size);
                        }
                        if let Some(l) = log.as_deref_mut() {
                            l.push(LdsTouch {
                                lo,
                                len: size,
                                write: true,
                                atomic: false,
                            });
                        }
                    } else {
                        mem.write(cur + offset as u64, &bytes[offset..offset + size])?;
                    }
                }
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        reg::{Kind, RegRef},
        Arch,
    };
    fn state() -> State {
        State {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes: 32,
            nan_loose: false,
            exec: 1,
            vcc: 0,
            scc: false,
            m0: 0,
            pc: 0,
            ended: false,
            barrier: None,
            signals: 0,
            scratch: Vec::new(),
        }
    }
    fn reg(kind: Kind, base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind, base, len })
    }
    fn insn(name: &str, operands: Vec<Operand>) -> Inst {
        let row = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            row.form,
            Default::default(),
            operands.into_iter().collect(),
            Default::default(),
            None,
            Default::default(),
        )
        .unwrap()
    }
    fn setup(word3: u32) -> (State, Memory) {
        let mut st = state();
        st.s[4] = 0x1000;
        st.s[6] = 8;
        st.s[7] = word3;
        st.v[2][0] = 4;
        let mut mem = Memory::default();
        mem.map("buf".into(), 0x1000, (0..8u8).collect()).unwrap();
        (st, mem)
    }
    fn load(st: &mut State, mem: &mut Memory, vaddr: Operand, extra: Vec<Operand>) -> Result<()> {
        let mut ops = vec![
            reg(Kind::V, 0, 1),
            vaddr,
            reg(Kind::S, 4, 4),
            Operand::Special(crate::operand::Special::Null),
        ];
        ops.extend(extra);
        execute(
            "buffer_load_b32",
            &insn("buffer_load_b32", ops),
            st,
            mem,
            &mut [],
        )
    }
    fn offen() -> Operand {
        Operand::Vmem(VmemToken::Offen)
    }
    /// `s_prefetch_data s[22:23], 0x0, null, 0` (F404C00B F8000000, as hipcc emits it): one 128-byte line at s[22:23] + offset.
    /// In range it changes nothing (not even strict-read bookkeeping); a line that is not wholly inside one mapped region
    /// is an unmapped-access error.
    #[test]
    fn prefetch_data_is_a_noop_in_range_and_refused_out_of_range() {
        let (inst, n) = crate::codec::gfx12::decode(&[0xF404_C00B, 0xF800_0000]).unwrap();
        assert_eq!(n, 2);
        let name = crate::isa::lookup(Arch::Gfx1201, inst.op, inst.form)
            .unwrap()
            .name;
        assert_eq!(name, "s_prefetch_data");
        assert_eq!(crate::emu::semantic_class(name), Some("memory"));
        let run = |base: u64, size: usize| {
            let mut st = state();
            st.s[22] = base as u32;
            st.s[23] = (base >> 32) as u32;
            let mut mem = Memory::default();
            mem.map("buf".into(), 0x4000, vec![7u8; size]).unwrap();
            mem.set_strict_read("buf").unwrap();
            let before = st.s;
            let r = execute(name, &inst, &mut st, &mut mem, &mut []);
            assert_eq!(st.s, before, "no architectural effect");
            assert_eq!(mem.regions[0].written.count(), 0);
            r
        };
        assert!(run(0x4000, 128).is_ok(), "exact fit");
        assert!(run(0x4000 + 64, 256).is_ok());
        let e = run(0x4000 + 1, 128).unwrap_err();
        assert!(
            e.contains("unmapped memory access") && e.contains("s_prefetch_data"),
            "{e}"
        );
        assert!(run(0x4000 - 8, 256).is_err(), "before the region");
        assert!(run(0x9000, 256).is_err(), "nowhere");
    }
    /// `s_prefetch_data s[2:3], 0x0, s5, 3` (F404C0C1 0A000000: 4 lines = 512 B at s[2:3] + s5) and
    /// `s_prefetch_data s[2:3], 0x10, null, 31` (F404C7C1 F8000010: 32 lines = 4096 B at s[2:3] + 0x10).
    #[test]
    fn prefetch_data_length_field_and_sgpr_soffset() {
        let dec = |w: [u32; 2]| crate::codec::gfx12::decode(&w).unwrap().0;
        let a = dec([0xF404_C0C1, 0x0A00_0000]);
        let b = dec([0xF404_C7C1, 0xF800_0010]);
        // the typed fields survive an encode round trip
        for i in [&a, &b] {
            assert_eq!(
                crate::codec::gfx12::encode_for(Arch::Gfx1201, i)
                    .unwrap()
                    .as_slice()
                    .len(),
                2
            );
        }
        let run = |inst: &Inst, base: u64, s5: u32, size: usize| {
            let mut st = state();
            (st.s[2], st.s[3], st.s[5]) = (base as u32, (base >> 32) as u32, s5);
            let mut mem = Memory::default();
            mem.map("buf".into(), 0x4000, vec![0u8; size]).unwrap();
            execute("s_prefetch_data", inst, &mut st, &mut mem, &mut [])
        };
        assert!(run(&a, 0x4000, 64, 64 + 512).is_ok());
        assert!(
            run(&a, 0x4000, 64, 64 + 511).unwrap_err().contains("+512"),
            "one byte short"
        );
        assert!(run(&b, 0x4000, 0, 0x10 + 4096).is_ok());
        assert!(run(&b, 0x4000, 0, 0x10 + 4095)
            .unwrap_err()
            .contains("+4096"));
    }
    // Captured raw SRDs (0x31004000) and offset-only/offen forms keep exact byte-bounds semantics.
    #[test]
    fn raw_forms_and_captured_srd_word3_are_accepted() {
        let (mut st, mut mem) = setup(0x3100_4000);
        load(&mut st, &mut mem, reg(Kind::V, 2, 1), vec![offen()]).unwrap();
        assert_eq!(st.v[0][0], u32::from_le_bytes([4, 5, 6, 7]));
        load(&mut st, &mut mem, Operand::Vmem(VmemToken::Off), vec![]).unwrap();
        assert_eq!(st.v[0][0], u32::from_le_bytes([0, 1, 2, 3]));
        load(&mut st, &mut mem, reg(Kind::V, 2, 2), vec![]).unwrap();
        assert_eq!(st.v[0][0], u32::from_le_bytes([0, 1, 2, 3]));
        st.v[2][0] = 8; // offset 8 == NUM_RECORDS: out of bounds reads zero
        load(&mut st, &mut mem, reg(Kind::V, 2, 1), vec![offen()]).unwrap();
        assert_eq!(st.v[0][0], 0);
    }
    #[test]
    fn idxen_shaped_addresses_are_hard_errors() {
        for (vaddr, extra) in [
            (reg(Kind::V, 2, 1), vec![]),
            (reg(Kind::V, 2, 2), vec![offen()]),
        ] {
            let (mut st, mut mem) = setup(0x3100_4000);
            st.v[0][0] = 0xdead_beef;
            let e = load(&mut st, &mut mem, vaddr, extra).unwrap_err();
            assert!(e.contains("IDXEN"), "{e}");
            assert_eq!(st.v[0][0], 0xdead_beef);
        }
    }
    #[test]
    fn unsupported_srd_word3_and_extra_operands_are_hard_errors() {
        for (word3, needle) in [
            (0x3100_4000 | 1 << 30, "TYPE"),
            (0x3100_4000 | 1 << 31, "TYPE"),
            (0x3100_4000 | 1 << 23, "ADD_TID"),
            (0x3100_4000 | 1 << 21, "INDEX_STRIDE"),
            (0x3100_4000 | 3 << 21, "INDEX_STRIDE"),
            (0x3100_4000 | 1 << 18, "stride-scale"),
            (0x3100_4000 | 1 << 26, "compression-access"),
            (0x3100_4000 | 1 << 20, "reserved"),
            (0x2100_4000, "OOB"),
            (0x3000_0000, "FORMAT 0"),
            (0x3100_0fff, "FORMAT 0"),
        ] {
            let (mut st, mut mem) = setup(word3);
            st.v[0][0] = 0xdead_beef;
            let e = load(&mut st, &mut mem, reg(Kind::V, 2, 1), vec![offen()]).unwrap_err();
            assert!(e.contains(needle), "{word3:#x}: {e}");
            assert_eq!(st.v[0][0], 0xdead_beef);
        }
        let (mut st, mut mem) = setup(0x3100_4000);
        st.s[5] = 1 << 16; // SRD stride field
        assert!(load(&mut st, &mut mem, reg(Kind::V, 2, 1), vec![offen()]).is_err());
        let (mut st, mut mem) = setup(0x3100_4000);
        let e = load(
            &mut st,
            &mut mem,
            reg(Kind::V, 2, 1),
            vec![offen(), Operand::Literal(0)],
        )
        .unwrap_err();
        assert!(e.contains("unsupported buffer operand"), "{e}");
    }
    // ds_bpermute_b32: lane i reads DATA0 of lane (ADDR[i]+OFFSET)/4 mod 32; disabled sources read 0,
    // disabled destinations keep their value, and LDS memory is untouched.
    #[test]
    fn bpermute_reads_addressed_lane_with_exec_and_offset() {
        let mut st = state();
        st.exec = 0xffff_fffe; // lane 0 disabled
        for lane in 0..32 {
            st.v[5][lane] = 100 + lane as u32;
            st.v[158][lane] = (((lane + 8) % 32) * 4) as u32;
            st.v[4][lane] = 0xdead_beef;
        }
        st.v[158][1] = 128 + 4 * 24 + 3; // high bits and low two bits are ignored: lane 24
        st.v[158][2] = 0; // disabled source lane 0 reads zero
        let mut lds = [0x5au8; 16];
        execute(
            "ds_bpermute_b32",
            &insn(
                "ds_bpermute_b32",
                vec![reg(Kind::V, 4, 1), reg(Kind::V, 158, 1), reg(Kind::V, 5, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(st.v[4][0], 0xdead_beef);
        assert_eq!(st.v[4][1], 124);
        assert_eq!(st.v[4][2], 0);
        for lane in 3..32 {
            let src = (lane + 8) % 32;
            assert_eq!(
                st.v[4][lane],
                if src == 0 { 0 } else { 100 + src as u32 },
                "lane {lane}"
            );
        }
        assert_eq!(lds, [0x5a; 16]);
        // shfl-down 16 idiom with a DS offset: (lane*4 + 64)/4 mod 32 = lane+16 mod 32.
        st.exec = 0xffff_ffff;
        for lane in 0..32 {
            st.v[158][lane] = (lane * 4) as u32;
        }
        execute(
            "ds_bpermute_b32",
            &insn(
                "ds_bpermute_b32",
                vec![
                    reg(Kind::V, 4, 1),
                    reg(Kind::V, 158, 1),
                    reg(Kind::V, 5, 1),
                    Operand::Imm(ImmField::DsOffset(64)),
                ],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        for lane in 0..32 {
            assert_eq!(
                st.v[4][lane],
                100 + ((lane + 16) % 32) as u32,
                "lane {lane}"
            );
        }
    }
    fn insn_on(arch: Arch, name: &str, operands: Vec<Operand>) -> Inst {
        let table = if arch == Arch::Gfx1201 {
            crate::isa::gfx12()
        } else {
            crate::isa::gfx1100()
        };
        let row = table
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("{name} missing from {arch:?}"));
        Inst::from_parts(
            arch,
            row.op,
            row.form,
            Default::default(),
            operands.into_iter().collect(),
            Default::default(),
            None,
            Default::default(),
        )
        .unwrap()
    }
    fn flat_state(bytes: &[u8]) -> (State, Memory) {
        let mut st = state();
        st.v[2][0] = 0x1000;
        st.v[3][0] = 0;
        let mut mem = Memory::default();
        mem.map("m".into(), 0x1000, bytes.to_vec()).unwrap();
        (st, mem)
    }
    // Loads: i8 sign-extends the byte; d16 forms merge into the other half; b96 fills three registers.
    #[test]
    fn flat_loads_sign_extend_merge_halves_and_load_b96() {
        let bytes: Vec<u8> = (0..16).map(|n| 0x80 + n).collect();
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            let (mut st, mut mem) = flat_state(&bytes);
            let run = |st: &mut State, mem: &mut Memory, name: &str, dst: Operand| {
                execute(
                    name,
                    &insn_on(arch, name, vec![dst, reg(Kind::V, 2, 2)]),
                    st,
                    mem,
                    &mut [],
                )
            };
            if arch == Arch::Gfx1201 {
                run(&mut st, &mut mem, "flat_load_i8", reg(Kind::V, 0, 1)).unwrap();
                assert_eq!(st.v[0][0], 0xffff_ff80);
                run(&mut st, &mut mem, "flat_load_b96", reg(Kind::V, 4, 3)).unwrap();
                assert_eq!(
                    [st.v[4][0], st.v[5][0], st.v[6][0]],
                    [0x8382_8180, 0x8786_8584, 0x8b8a_8988]
                );
                assert_eq!(st.v[7][0], 0);
            }
            st.v[0][0] = 0xabcd_1234;
            run(&mut st, &mut mem, "flat_load_d16_u8", reg(Kind::V, 0, 1)).unwrap();
            assert_eq!(st.v[0][0], 0xabcd_0080, "{arch:?}");
            st.v[0][0] = 0xabcd_1234;
            run(
                &mut st,
                &mut mem,
                "flat_load_d16_hi_b16",
                reg(Kind::V, 0, 1),
            )
            .unwrap();
            assert_eq!(st.v[0][0], 0x8180_1234, "{arch:?}");
            // Low-half merge: loaded 0x8180 replaces the low bits, high half preserved.
            st.v[0][0] = 0xabcd_1234;
            st.v[0][1] = 0xdead_beef; // lane 1 inactive under default exec mask
            st.vcc = 0xcafe_1234;
            st.scc = true;
            st.m0 = 0x1357_9bdf;
            run(&mut st, &mut mem, "flat_load_d16_b16", reg(Kind::V, 0, 1)).unwrap();
            assert_eq!(st.v[0][0], 0xabcd_8180, "{arch:?}");
            assert_eq!(st.v[0][1], 0xdead_beef, "{arch:?}");
            assert_eq!((st.v[2][0], st.v[3][0]), (0x1000, 0), "{arch:?}");
            assert_eq!(
                (st.vcc, st.scc, st.m0, st.exec),
                (0xcafe_1234, true, 0x1357_9bdf, 1)
            );
            // Full dword loads replace both halves through the same generic flat handler.
            st.v[0][0] = 0xabcd_1234;
            run(&mut st, &mut mem, "flat_load_b32", reg(Kind::V, 0, 1)).unwrap();
            assert_eq!(st.v[0][0], 0x8382_8180, "{arch:?}");
            assert_eq!(st.v[0][1], 0xdead_beef, "{arch:?}");
            assert_eq!((st.v[2][0], st.v[3][0]), (0x1000, 0), "{arch:?}");
            assert_eq!(
                (st.vcc, st.scc, st.m0, st.exec),
                (0xcafe_1234, true, 0x1357_9bdf, 1)
            );
        }
    }
    // K1-C1's SADDR global loads: `global_load_b128` / `_b64` read 4 / 2 dwords at `saddr + voffset` (+ imm offset) per lane.
    #[test]
    fn global_saddr_loads_b64_and_b128_fill_their_registers() {
        let bytes: Vec<u8> = (0..32).map(|n| 0x40 + n).collect();
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            let mut st = state();
            st.s[8] = 0x1000;
            st.s[9] = 0;
            st.v[2][0] = 4;
            let mut mem = Memory::default();
            mem.map("m".into(), 0x1000, bytes.clone()).unwrap();
            let run = |st: &mut State, mem: &mut Memory, name: &str, dst: Operand, imm: u16| {
                let mut ops = vec![dst, reg(Kind::V, 2, 1), reg(Kind::S, 8, 2)];
                if imm != 0 {
                    ops.push(Operand::Imm(ImmField::VmemOffset(imm.into())));
                }
                execute(name, &insn_on(arch, name, ops), st, mem, &mut [])
            };
            run(&mut st, &mut mem, "global_load_b128", reg(Kind::V, 4, 4), 0).unwrap();
            assert_eq!(
                [st.v[4][0], st.v[5][0], st.v[6][0], st.v[7][0]],
                [0x4746_4544, 0x4b4a_4948, 0x4f4e_4d4c, 0x5352_5150],
                "{arch:?}"
            );
            run(&mut st, &mut mem, "global_load_b64", reg(Kind::V, 8, 2), 8).unwrap();
            assert_eq!(
                [st.v[8][0], st.v[9][0], st.v[10][0]],
                [0x4f4e_4d4c, 0x5352_5150, 0],
                "{arch:?}"
            );
            // An unmapped read is a hard error, not a silent zero.
            st.v[2][0] = 0x1000;
            assert!(
                run(&mut st, &mut mem, "global_load_b128", reg(Kind::V, 4, 4), 0).is_err(),
                "{arch:?}"
            );
        }
    }
    // Stores: b8/b16 write exactly their size, d16_hi stores bits [31:16], b96/b128 write whole dwords.
    #[test]
    fn flat_stores_write_exact_sizes_and_the_high_half() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            let (mut st, mut mem) = flat_state(&[0x11; 16]);
            st.v[1][0] = 0xdead_beef;
            let run = |st: &mut State, mem: &mut Memory, name: &str, src: Operand| {
                execute(
                    name,
                    &insn_on(arch, name, vec![reg(Kind::V, 2, 2), src]),
                    st,
                    mem,
                    &mut [],
                )
            };
            run(&mut st, &mut mem, "flat_store_b8", reg(Kind::V, 1, 1)).unwrap();
            assert_eq!(mem.read(0x1000, 4).unwrap(), [0xef, 0x11, 0x11, 0x11]);
            st.v[4][0] = 1;
            st.v[5][0] = 2;
            st.v[6][0] = 3;
            st.v[7][0] = 4;
            if arch == Arch::Gfx1201 {
                st.v[2][0] = 0x1004;
                run(
                    &mut st,
                    &mut mem,
                    "flat_store_d16_hi_b16",
                    reg(Kind::V, 1, 1),
                )
                .unwrap();
                assert_eq!(mem.read(0x1004, 4).unwrap(), [0xad, 0xde, 0x11, 0x11]);
                run(&mut st, &mut mem, "flat_store_b16", reg(Kind::V, 1, 1)).unwrap();
                assert_eq!(mem.read(0x1004, 4).unwrap(), [0xef, 0xbe, 0x11, 0x11]);
                st.v[2][0] = 0x1000;
                run(&mut st, &mut mem, "flat_store_b96", reg(Kind::V, 4, 3)).unwrap();
                assert_eq!(
                    mem.read(0x1000, 12).unwrap(),
                    [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
                );
                assert_eq!(mem.read(0x100c, 4).unwrap(), [0x11; 4]);
            } else {
                st.v[2][0] = 0x1008;
                run(&mut st, &mut mem, "flat_store_b32", reg(Kind::V, 1, 1)).unwrap();
                assert_eq!(mem.read(0x1008, 4).unwrap(), 0xdead_beefu32.to_le_bytes());
                st.v[2][0] = 0x1000;
                run(&mut st, &mut mem, "flat_store_b128", reg(Kind::V, 4, 4)).unwrap();
                assert_eq!(
                    mem.read(0x1000, 16).unwrap(),
                    [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
                );
            }
        }
    }
    #[test]
    fn ds_u16_load_and_b96_store() {
        let mut lds = [0u8; 64];
        let mut st = state();
        st.exec = 0b11;
        st.v[1][0] = 6;
        st.v[1][1] = 20;
        st.v[2][0] = 1;
        st.v[3][0] = 2;
        st.v[4][0] = 3;
        st.v[2][1] = 4;
        st.v[3][1] = 5;
        st.v[4][1] = 6;
        execute(
            "ds_store_b96",
            &insn_on(
                Arch::Gfx1100,
                "ds_store_b96",
                vec![reg(Kind::V, 1, 1), reg(Kind::V, 2, 3)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(&lds[6..18], &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(&lds[20..32], &[4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]);
        assert_eq!(lds[18], 0);
        // ds_load_u16 zero-extends the 16-bit word at byte 24 (the dword 5).
        st.v[0][0] = 0xffff_ffff;
        st.v[1][0] = 24;
        execute(
            "ds_load_u16",
            &insn_on(
                Arch::Gfx1100,
                "ds_load_u16",
                vec![reg(Kind::V, 0, 1), reg(Kind::V, 1, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(st.v[0][0], 5);
    }
    // LDS atomics: integer add/or without return are lane-order independent; returns and float adds refuse
    // same-dword collisions; ds_add_f32 is IEEE.
    #[test]
    fn ds_atomics_apply_read_modify_write_and_refuse_unordered_collisions() {
        let mut lds = [0u8; 32];
        lds[0..4].copy_from_slice(&10u32.to_le_bytes());
        lds[4..8].copy_from_slice(&0xf0u32.to_le_bytes());
        let mut st = state();
        st.exec = 0b111;
        st.v[1] = [0; 64];
        st.v[2][0] = 1;
        st.v[2][1] = 2;
        st.v[2][2] = 4;
        // Three lanes add into dword 0: 10 + 1 + 2 + 4.
        execute(
            "ds_add_u32",
            &insn_on(
                Arch::Gfx1100,
                "ds_add_u32",
                vec![reg(Kind::V, 1, 1), reg(Kind::V, 2, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(u32::from_le_bytes(lds[0..4].try_into().unwrap()), 17);
        st.v[1] = [4; 64];
        st.v[2] = [0x0f; 64];
        execute(
            "ds_or_b32",
            &insn_on(
                Arch::Gfx1100,
                "ds_or_b32",
                vec![reg(Kind::V, 1, 1), reg(Kind::V, 2, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(u32::from_le_bytes(lds[4..8].try_into().unwrap()), 0xff);
        // Returning add: distinct dwords return the old value; the offset field adds to the address.
        let mut st = state();
        st.exec = 0b11;
        st.v[1][0] = 0;
        st.v[1][1] = 4;
        st.v[2][0] = 100;
        st.v[2][1] = 200;
        st.v[0] = [0x77; 64];
        execute(
            "ds_add_rtn_u32",
            &insn_on(
                Arch::Gfx1100,
                "ds_add_rtn_u32",
                vec![reg(Kind::V, 0, 1), reg(Kind::V, 1, 1), reg(Kind::V, 2, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!((st.v[0][0], st.v[0][1], st.v[0][2]), (17, 0xff, 0x77));
        assert_eq!(u32::from_le_bytes(lds[0..4].try_into().unwrap()), 117);
        assert_eq!(
            u32::from_le_bytes(lds[4..8].try_into().unwrap()),
            0xff + 200
        );
        st.v[1][1] = 0;
        let e = execute(
            "ds_add_rtn_u32",
            &insn_on(
                Arch::Gfx1100,
                "ds_add_rtn_u32",
                vec![reg(Kind::V, 0, 1), reg(Kind::V, 1, 1), reg(Kind::V, 2, 1)],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap_err();
        assert!(e.contains("does not order"), "{e}");
        // ds_add_f32.
        let mut lds = [0u8; 16];
        lds[8..12].copy_from_slice(&1.5f32.to_bits().to_le_bytes());
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 8;
        st.v[2][0] = 2.25f32.to_bits();
        let add = |st: &mut State, lds: &mut [u8]| {
            execute(
                "ds_add_f32",
                &insn_on(
                    Arch::Gfx1100,
                    "ds_add_f32",
                    vec![reg(Kind::V, 1, 1), reg(Kind::V, 2, 1)],
                ),
                st,
                &mut Memory::default(),
                lds,
            )
        };
        add(&mut st, &mut lds).unwrap();
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(lds[8..12].try_into().unwrap())),
            3.75
        );
        // A denormal addend is IEEE-exact (the ISA lists denormals as handled).
        st.v[2][0] = 1;
        lds[8..12].copy_from_slice(&1u32.to_le_bytes());
        add(&mut st, &mut lds).unwrap();
        assert_eq!(u32::from_le_bytes(lds[8..12].try_into().unwrap()), 2);
        st.v[2][0] = f32::NAN.to_bits();
        assert!(add(&mut st, &mut lds).is_err());
        st.v[2][0] = 1f32.to_bits();
        st.v[1][0] = 9;
        assert!(add(&mut st, &mut lds).unwrap_err().contains("unaligned"));
        st.v[1][0] = 8;
        st.exec = 0b11;
        st.v[1][1] = 8;
        assert!(add(&mut st, &mut lds)
            .unwrap_err()
            .contains("does not order"));
        st.exec = 1;
        st.v[1][0] = 16;
        assert!(add(&mut st, &mut lds).is_err());
    }
    // ds_load_b64 rides the generic ds_ load path (width b64 -> 8 bytes): per-lane ADDR plus the
    // DsOffset immediate, two dwords into the VDST pair, inactive lanes skipped.
    #[test]
    fn ds_load_b64_loads_two_dwords_and_skips_inactive_lanes() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            let mut lds = [0u8; 64];
            for (i, b) in lds.iter_mut().enumerate() {
                *b = i as u8;
            }
            let mut st = state();
            st.exec = 0b01; // lane 1 inactive
            st.v[2][0] = 4;
            st.v[2][1] = 16;
            st.v[0] = [0xdead_beef; 64];
            st.v[1] = [0xdead_beef; 64];
            execute(
                "ds_load_b64",
                &insn_on(
                    arch,
                    "ds_load_b64",
                    vec![
                        reg(Kind::V, 0, 2),
                        reg(Kind::V, 2, 1),
                        Operand::Imm(ImmField::DsOffset(4)),
                    ],
                ),
                &mut st,
                &mut Memory::default(),
                &mut lds,
            )
            .unwrap();
            // Lane 0: effective address 4 + 4 = 8; bytes 08..0f fill both halves.
            assert_eq!(st.v[0][0], 0x0b0a_0908, "{arch:?}");
            assert_eq!(st.v[1][0], 0x0f0e_0d0c, "{arch:?}");
            // Lane 1 inactive: destination pair untouched.
            assert_eq!(st.v[0][1], 0xdead_beef, "{arch:?}");
            assert_eq!(st.v[1][1], 0xdead_beef, "{arch:?}");
        }
    }
    // ds_{load,store}_2addr_stride64: element offsets are multiplied by 64 elements; the plain 2addr forms by 1.
    #[test]
    fn ds_2addr_stride64_scales_both_offsets_by_64_elements() {
        let mut lds = vec![0u8; 2048];
        for (name, size, off0, off1) in [
            ("ds_store_2addr_stride64_b32", 4usize, 1u8, 3u8),
            ("ds_store_2addr_stride64_b64", 8, 1, 3),
        ] {
            lds.fill(0);
            let mut st = state();
            st.v[2][0] = 8; // ADDR
                            // DATA0 = v4.. (1, 2), DATA1 = v8.. (3, 4): the two data operands are not adjacent.
            for w in 0..2 {
                st.v[4 + w][0] = 0x1111_1111 * (w as u32 + 1);
                st.v[8 + w][0] = 0x1111_1111 * (w as u32 + 3);
            }
            let words = size / 4;
            execute(
                name,
                &insn(
                    name,
                    vec![
                        reg(Kind::V, 2, 1),
                        reg(Kind::V, 4, words as u8),
                        reg(Kind::V, 8, words as u8),
                        Operand::Imm(ImmField::DsOffset0(off0)),
                        Operand::Imm(ImmField::DsOffset1(off1)),
                    ],
                ),
                &mut st,
                &mut Memory::default(),
                &mut lds,
            )
            .unwrap();
            let at = |byte: usize, w: u32| {
                assert_eq!(
                    u32::from_le_bytes(lds[byte..byte + 4].try_into().unwrap()),
                    0x1111_1111 * w,
                    "{name} byte {byte}"
                );
            };
            let (a0, a1) = (
                8 + usize::from(off0) * 64 * size,
                8 + usize::from(off1) * 64 * size,
            );
            at(a0, 1);
            at(a1, 3);
            if words == 2 {
                at(a0 + 4, 2);
                at(a1 + 4, 4);
            }
            let nonzero = lds.iter().filter(|&&b| b != 0).count();
            assert_eq!(
                nonzero,
                2 * size,
                "{name}: only the two elements are written"
            );
        }
        // Load: dwords come back from the same two scaled addresses.
        let mut st = state();
        st.v[2][0] = 8;
        lds.fill(0);
        lds[8 + 512..8 + 512 + 8].copy_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        lds[8 + 3 * 512..8 + 3 * 512 + 8].copy_from_slice(&[3, 0, 0, 0, 4, 0, 0, 0]);
        execute(
            "ds_load_2addr_stride64_b64",
            &insn(
                "ds_load_2addr_stride64_b64",
                vec![
                    reg(Kind::V, 4, 4),
                    reg(Kind::V, 2, 1),
                    Operand::Imm(ImmField::DsOffset0(1)),
                    Operand::Imm(ImmField::DsOffset1(3)),
                ],
            ),
            &mut st,
            &mut Memory::default(),
            &mut lds,
        )
        .unwrap();
        assert_eq!(
            [st.v[4][0], st.v[5][0], st.v[6][0], st.v[7][0]],
            [1, 2, 3, 4]
        );
    }

    // global_store_{b16,b64,b128} with the SGPR-base form `vOFF, v[data], s[base:base+1] offset:N`: exactly the access
    // size is written at base + vOFF + offset, nothing else, and an unmapped store is a hard error.
    #[test]
    fn global_saddr_stores_write_exactly_their_size() {
        let mut st = state();
        st.s[8] = 0x1000;
        st.s[9] = 0;
        st.v[2][0] = 4;
        for (n, w) in [0x0403_0201u32, 0x0807_0605, 0x0c0b_0a09, 0x100f_0e0d]
            .into_iter()
            .enumerate()
        {
            st.v[4 + n][0] = w;
        }
        let mut mem = Memory::default();
        mem.map("m".into(), 0x1000, vec![0xee; 64]).unwrap();
        let run = |st: &mut State, mem: &mut Memory, name: &str, src: Operand, imm: u16| {
            let mut ops = vec![reg(Kind::V, 2, 1), src, reg(Kind::S, 8, 2)];
            if imm != 0 {
                ops.push(Operand::Imm(ImmField::VmemOffset(imm.into())));
            }
            execute(name, &insn(name, ops), st, mem, &mut [])
        };
        run(
            &mut st,
            &mut mem,
            "global_store_b128",
            reg(Kind::V, 4, 4),
            16,
        )
        .unwrap();
        assert_eq!(
            mem.read(0x1000 + 20, 16).unwrap(),
            (1..=16u8).collect::<Vec<_>>()
        );
        run(&mut st, &mut mem, "global_store_b64", reg(Kind::V, 4, 2), 0).unwrap();
        assert_eq!(mem.read(0x1004, 8).unwrap(), (1..=8u8).collect::<Vec<_>>());
        run(
            &mut st,
            &mut mem,
            "global_store_b16",
            reg(Kind::V, 7, 1),
            40,
        )
        .unwrap();
        assert_eq!(mem.read(0x1000 + 44, 4).unwrap(), [0x0d, 0x0e, 0xee, 0xee]);
        let written: usize = mem.regions[0].written.count();
        assert_eq!(
            written,
            16 + 8 + 2,
            "only the stored bytes are marked written"
        );
        st.v[2][0] = 0x2000;
        assert!(run(
            &mut st,
            &mut mem,
            "global_store_b128",
            reg(Kind::V, 4, 4),
            0
        )
        .is_err());
    }
}
