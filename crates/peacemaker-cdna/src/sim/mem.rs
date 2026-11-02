// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Deferred SMEM, LDS, GLOBAL and untyped buffer accesses.
//! Addresses, EXEC and operands are captured at issue; memory is sampled or
//! modified only at retirement (see the adversarial model in `sim/mod.rs`).
//! VM retires in issue order. Positive LGKM waits retire DS in issue order,
//! never SMEM; an unsatisfiable positive wait is a `SimError::Wait`. A zero
//! LGKM wait retires all DS before SMEM, then SMEM in reverse issue order.
//! Hardware can return SMEM out of order; only S_WAITCNT 0 is legitimate
//! (CDNA3 ISA §4.4 p.20, §8.3 p.62). SMEM weighs 1 for a dword, 2 otherwise.
//! Overflow forces the oldest operation on that counter to retire, modelling
//! an issue stall, even for SMEM. This deterministic scheduling and ascending
//! lane order for conflicting atomics are [UNVERIFIED: confirm on MI300X].
use super::{
    fault, unsupported, Counter, Flow, Memory, Pending, Retire, SimError, SimResult, Ty, Wave,
};
use crate::enc::{
    ds::{Ds, DsOp},
    flat::{Flat, FlatOp, FlatSeg},
    mubuf::{Mubuf, MubufOp},
    smem::{Smem, SmemOffset, SmemOp},
};
use crate::inst::{Inst, RegSpan};
use crate::operand::Src;
use crate::WAVE;

fn count(w: &Wave, counter: Counter) -> u16 {
    w.pending
        .iter()
        .filter(|p| p.counter == counter)
        .map(|p| u16::from(p.weight))
        .sum()
}

fn retire_at(w: &mut Wave, index: usize, lds: &mut [u8], mem: &mut Memory) -> SimResult<()> {
    let p = w
        .pending
        .remove(index)
        .ok_or_else(|| SimError::Wait("missing pending operation".into()))?;
    // Remove its busy spans before writing the result through checked accessors.
    (p.retire)(w, lds, mem)
}

/// Apply the documented counter policy, without decoding waitcnt fields here.
pub fn wait(
    w: &mut Wave,
    lds: &mut [u8],
    mem: &mut Memory,
    vm: u8,
    lgkm: u8,
    exp: u8,
) -> SimResult<()> {
    for (counter, threshold) in [
        (Counter::Vm, vm),
        (Counter::Lgkm, lgkm),
        (Counter::Exp, exp),
    ] {
        while count(w, counter) > u16::from(threshold) {
            let index = w
                .pending
                .iter()
                .position(|p| p.counter == counter && !p.out_of_order);
            let index = match index {
                Some(i) => i,
                None if threshold == 0 => w.pending.iter().rposition(|p| p.counter == counter)
                    .ok_or_else(|| SimError::Wait("counter without pending operation".into()))?,
                None => return Err(SimError::Wait("positive lgkmcnt wait requires SMEM retirement: only S_WAITCNT 0 is legitimate (CDNA3 ISA §4.4 p.20)".into())),
            };
            retire_at(w, index, lds, mem)?;
        }
    }
    Ok(())
}

/// `s_endpgm` drains all counters under the same ordering policy.
pub fn drain(w: &mut Wave, lds: &mut [u8], mem: &mut Memory) -> SimResult<()> {
    wait(w, lds, mem, 0, 0, 0)
}

fn reg_at(src: Src, dword: usize) -> SimResult<Src> {
    let n = u8::try_from(dword).map_err(|_| SimError::Fault("register width overflow".into()))?;
    let add = |r: u8| {
        r.checked_add(n)
            .ok_or_else(|| SimError::Fault("register span overflow".into()))
    };
    Ok(match src {
        Src::Sgpr(r) => Src::Sgpr(add(r)?),
        Src::Vgpr(r) => Src::Vgpr(add(r)?),
        Src::Agpr(r) => Src::Agpr(add(r)?),
        Src::Ttmp(r) => Src::Ttmp(add(r)?),
        _ if dword == 0 => src,
        _ => Src::from_code(src.code() + u16::from(n), None).map_err(SimError::Fault)?,
    })
}

fn destination(
    w: &Wave,
    dst: Src,
    dwords: u8,
    scalar: bool,
    busy: &mut Vec<RegSpan>,
) -> SimResult<()> {
    let span = RegSpan::of(dst, dwords)
        .ok_or_else(|| SimError::Fault(format!("invalid load destination {dst:?}")))?;
    w.check_ready(span, "write")?;
    for j in 0..usize::from(dwords) {
        let reg = reg_at(dst, j)?;
        if scalar {
            w.read_s(reg, Ty::B32)?;
        } else {
            match reg {
                Src::Vgpr(n) => w.check_vgpr(crate::inst::RegFile::Vgpr, n, 1)?,
                Src::Agpr(n) => w.check_vgpr(crate::inst::RegFile::Agpr, n, 1)?,
                _ => return fault(format!("invalid vector destination {reg:?}")),
            }
        }
    }
    busy.push(span);
    Ok(())
}

#[derive(Clone, Copy)]
enum Atomic {
    Add,
    MaxU,
    MaxI,
    AddFloat { denorm: u8 },
    MinFloat { denorm: u8 },
    MaxFloat { denorm: u8 },
}

enum Action {
    Load {
        dst: Src,
        scalar: bool,
        signed: bool,
    },
    Store([u8; 16]),
    Atomic {
        op: Atomic,
        operand: u32,
        dst: Option<Src>,
    },
}
struct Access {
    lane: usize,
    addr: u64,
    size: usize,
    valid: bool,
    action: Action,
}

fn store_data(w: &Wave, src: Src, lane: usize, size: usize) -> SimResult<[u8; 16]> {
    let mut data = [0; 16];
    for j in 0..size.div_ceil(4) {
        let bytes = w.read_v(reg_at(src, j)?, lane, Ty::B32)?.to_le_bytes();
        let n = (size - j * 4).min(4);
        data[j * 4..j * 4 + n].copy_from_slice(&bytes[..n]);
    }
    Ok(data)
}

fn lds_range(addr: u64, size: usize, len: usize) -> SimResult<std::ops::Range<usize>> {
    let start =
        usize::try_from(addr).map_err(|_| SimError::Fault("LDS address overflow".into()))?;
    let end = start
        .checked_add(size)
        .filter(|&end| end <= len)
        .ok_or_else(|| SimError::Fault(format!("out-of-range LDS access {addr:#x} size {size}")))?;
    Ok(start..end)
}

/// Atomic floating rules: nearest-even regardless of MODE rounding; LDS uses
/// MODE denorm controls, L2 ADD_F32 flushes input/output denorms. NaN priority,
/// signed-zero selection and min/max ordering follow CDNA3 ISA §9.2 p.77-78.
fn flush(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0 {
        bits & 0x8000_0000
    } else {
        bits
    }
}
fn nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}
fn snan(bits: u32) -> bool {
    nan(bits) && bits & 0x0040_0000 == 0
}
fn float_add(a: u32, b: u32, denorm: u8) -> u32 {
    let a = if denorm & 1 == 0 { flush(a) } else { a };
    let b = if denorm & 1 == 0 { flush(b) } else { b };
    let result = if nan(a) {
        a | 0x0040_0000
    } else if nan(b) {
        b | 0x0040_0000
    } else if a & 0x7fff_ffff == 0x7f80_0000 && b == (a ^ 0x8000_0000) {
        0x7fc0_0000
    } else {
        (f32::from_bits(a) + f32::from_bits(b)).to_bits()
    };
    if denorm & 2 == 0 {
        flush(result)
    } else {
        result
    }
}
fn float_select(a: u32, b: u32, denorm: u8, max: bool) -> u32 {
    if snan(a) {
        return a | 0x0040_0000;
    }
    if snan(b) {
        return b | 0x0040_0000;
    }
    if nan(a) {
        return if nan(b) { a } else { b };
    }
    if nan(b) {
        return a;
    }
    let ca = if denorm & 1 == 0 { flush(a) } else { a };
    let cb = if denorm & 1 == 0 { flush(b) } else { b };
    let fa = f32::from_bits(ca);
    let fb = f32::from_bits(cb);
    if fa == 0.0 && fb == 0.0 && ca != cb {
        if (ca & 0x8000_0000 == 0) == max {
            a
        } else {
            b
        }
    } else if if max { fb > fa } else { fb < fa } {
        b
    } else {
        a
    }
}
fn atomic_result(op: Atomic, old: u32, operand: u32) -> u32 {
    match op {
        Atomic::Add => old.wrapping_add(operand),
        Atomic::MaxU => old.max(operand),
        Atomic::MaxI => (old as i32).max(operand as i32) as u32,
        Atomic::AddFloat { denorm } => float_add(old, operand, denorm),
        Atomic::MinFloat { denorm } => float_select(old, operand, denorm, false),
        Atomic::MaxFloat { denorm } => float_select(old, operand, denorm, true),
    }
}

fn accesses_retire(accesses: Vec<Access>, local: bool) -> Retire {
    Box::new(move |w, lds, mem| {
        for a in accesses {
            match a.action {
                Action::Load {
                    dst,
                    scalar,
                    signed,
                } => {
                    let mut bytes = [0u8; 16];
                    if a.valid {
                        let source = if local {
                            &lds[lds_range(a.addr, a.size, lds.len())?]
                        } else {
                            mem.read(a.addr, a.size)?
                        };
                        bytes[..a.size].copy_from_slice(source);
                    }
                    for j in 0..a.size.div_ceil(4) {
                        let value = if a.size == 1 && signed {
                            i32::from(bytes[0] as i8) as u32
                        } else if a.size == 2 && signed {
                            i32::from(i16::from_le_bytes([bytes[0], bytes[1]])) as u32
                        } else {
                            u32::from_le_bytes(bytes[j * 4..j * 4 + 4].try_into().unwrap())
                        };
                        let reg = reg_at(dst, j)?;
                        if scalar {
                            w.write_s(reg, value)?;
                        } else {
                            w.write_v(reg, a.lane, value)?;
                        }
                    }
                }
                Action::Store(bytes) if a.valid => {
                    if local {
                        let range = lds_range(a.addr, a.size, lds.len())?;
                        lds[range].copy_from_slice(&bytes[..a.size]);
                    } else {
                        mem.write(a.addr, &bytes[..a.size])?;
                    }
                }
                Action::Atomic { op, operand, dst } if a.valid => {
                    let bytes = if local {
                        &lds[lds_range(a.addr, 4, lds.len())?]
                    } else {
                        mem.read(a.addr, 4)?
                    };
                    let old = u32::from_le_bytes(bytes.try_into().unwrap());
                    let new = atomic_result(op, old, operand).to_le_bytes();
                    if local {
                        let range = lds_range(a.addr, 4, lds.len())?;
                        lds[range].copy_from_slice(&new);
                    } else {
                        mem.write(a.addr, &new)?;
                    }
                    if let Some(dst) = dst {
                        w.write_v(dst, a.lane, old)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    })
}

fn required(src: Option<Src>, what: &str) -> SimResult<Src> {
    src.ok_or_else(|| SimError::Fault(format!("missing {what} operand")))
}

struct Prepared {
    counter: Counter,
    weight: u8,
    out_of_order: bool,
    busy: Vec<RegSpan>,
    retire: Retire,
}
fn prepared(
    counter: Counter,
    weight: u8,
    out_of_order: bool,
    busy: Vec<RegSpan>,
    accesses: Vec<Access>,
    local: bool,
) -> Prepared {
    Prepared {
        counter,
        weight,
        out_of_order,
        busy,
        retire: accesses_retire(accesses, local),
    }
}

/// LDS element sizes and double-address strides: CDNA3 ISA §11.3 p.84-86,
/// instruction pseudocode §12.12 p.409-444 (reads/writes/atomics).
/// Bounds are the workgroup LDS allocation, not M0: LLVM's gfx9+
/// `ldsRequiresM0Init()` is false. [UNVERIFIED: confirm on MI300X] Legacy
/// "M0 must be initialized" wording in §11.3.1 p.86 conflicts with gfx9+
/// compiler behaviour; this emulator does not apply that legacy M0 bound.
fn ds(w: &Wave, i: &Ds) -> SimResult<Prepared> {
    use DsOp::*;
    if i.gds {
        return unsupported(format!("{} GDS", i.op.name()));
    }
    let mut busy = Vec::new();
    if matches!(i.op, SwizzleB32 | PermuteB32 | BpermuteB32) {
        let dst = required(i.vdst, "DS destination")?;
        destination(w, dst, 1, false, &mut busy)?;
        let data_src = required(
            if i.op == SwizzleB32 { i.addr } else { i.data0 },
            "DS cross-lane data",
        )?;
        let mut data = [0; WAVE];
        let mut addr = [0; WAVE];
        for lane in 0..WAVE {
            if w.lane_active(lane) {
                data[lane] = w.read_v(data_src, lane, Ty::B32)?;
                if i.op != SwizzleB32 {
                    addr[lane] =
                        w.read_v(required(i.addr, "DS cross-lane address")?, lane, Ty::B32)?;
                }
            }
        }
        let exec = w.exec;
        let op = i.op;
        let offset = i.offset();
        return Ok(Prepared {
            counter: Counter::Lgkm,
            weight: 1,
            out_of_order: false,
            busy,
            retire: Box::new(move |w, _, _| {
                let result = match op {
                    SwizzleB32 => super::xlane::ds_swizzle(offset, &data, exec)?,
                    PermuteB32 => super::xlane::ds_permute(&addr, offset, &data, exec),
                    BpermuteB32 => super::xlane::ds_bpermute(&addr, offset, &data, exec),
                    _ => return unsupported(op.name()),
                };
                for (lane, value) in result.into_iter().enumerate() {
                    if exec >> lane & 1 != 0 {
                        w.write_v(dst, lane, value)?;
                    }
                }
                Ok(())
            }),
        });
    }
    let (size, load, signed, two, st64, atomic) = match i.op {
        ReadB32 => (4, true, false, false, false, None),
        ReadB64 => (8, true, false, false, false, None),
        ReadB96 => (12, true, false, false, false, None),
        ReadB128 => (16, true, false, false, false, None),
        Read2B32 | Read2st64B32 => (4, true, false, true, i.op == Read2st64B32, None),
        Read2B64 | Read2st64B64 => (8, true, false, true, i.op == Read2st64B64, None),
        ReadU8 | ReadI8 => (1, true, i.op == ReadI8, false, false, None),
        ReadU16 | ReadI16 => (2, true, i.op == ReadI16, false, false, None),
        WriteB8 => (1, false, false, false, false, None),
        WriteB16 => (2, false, false, false, false, None),
        WriteB32 => (4, false, false, false, false, None),
        WriteB64 => (8, false, false, false, false, None),
        WriteB96 => (12, false, false, false, false, None),
        WriteB128 => (16, false, false, false, false, None),
        Write2B32 | Write2st64B32 => (4, false, false, true, i.op == Write2st64B32, None),
        Write2B64 | Write2st64B64 => (8, false, false, true, i.op == Write2st64B64, None),
        AddU32 | AddRtnU32 => (4, false, false, false, false, Some(Atomic::Add)),
        MaxU32 => (4, false, false, false, false, Some(Atomic::MaxU)),
        MaxI32 => (4, false, false, false, false, Some(Atomic::MaxI)),
        AddF32 => (
            4,
            false,
            false,
            false,
            false,
            Some(Atomic::AddFloat {
                denorm: w.mode.denorm32,
            }),
        ),
        MinF32 => (
            4,
            false,
            false,
            false,
            false,
            Some(Atomic::MinFloat {
                denorm: w.mode.denorm32,
            }),
        ),
        MaxF32 => (
            4,
            false,
            false,
            false,
            false,
            Some(Atomic::MaxFloat {
                denorm: w.mode.denorm32,
            }),
        ),
        Nop => return Ok(prepared(Counter::Lgkm, 1, false, busy, Vec::new(), true)),
        SwizzleB32 | PermuteB32 | BpermuteB32 => return unsupported(i.op.name()),
    };
    let dst = if load || i.op == AddRtnU32 {
        Some(required(i.vdst, "DS destination")?)
    } else {
        None
    };
    if let Some(dst) = dst {
        destination(
            w,
            dst,
            (size / 4).max(1) as u8 * if two { 2 } else { 1 },
            false,
            &mut busy,
        )?;
    }
    let mut accesses = Vec::with_capacity(w.exec.count_ones() as usize * if two { 2 } else { 1 });
    for lane in 0..WAVE {
        if w.lane_active(lane) {
            let addr = u64::from(w.read_v(required(i.addr, "DS address")?, lane, Ty::B32)?);
            for part in 0..if two { 2 } else { 1 } {
                let offset = if two {
                    u64::from(if part == 0 { i.offset0 } else { i.offset1 })
                        * size as u64
                        * if st64 { 64 } else { 1 }
                } else {
                    u64::from(i.offset())
                };
                let action = if load {
                    Action::Load {
                        dst: reg_at(dst.unwrap(), part * size / 4)?,
                        scalar: false,
                        signed,
                    }
                } else if let Some(op) = atomic {
                    Action::Atomic {
                        op,
                        operand: w.read_v(required(i.data0, "DS atomic data")?, lane, Ty::B32)?,
                        dst,
                    }
                } else {
                    Action::Store(store_data(
                        w,
                        required(if part == 0 { i.data0 } else { i.data1 }, "DS store data")?,
                        lane,
                        size,
                    )?)
                };
                accesses.push(Access {
                    lane,
                    addr: addr + offset,
                    size,
                    valid: true,
                    action,
                });
            }
        }
    }
    Ok(prepared(Counter::Lgkm, 1, false, busy, accesses, true))
}

/// GLOBAL addresses and signed offsets: CDNA3 ISA §10.1 p.79-81;
/// opcode semantics §12.15 p.495-504. L2 float ADD flushes denorms (§9.2 p.77).
fn global(w: &Wave, i: &Flat) -> SimResult<Prepared> {
    use FlatOp::*;
    if i.seg != FlatSeg::Global {
        return unsupported(format!("{:?} {:?} segment", i.op, i.seg));
    }
    let (size, load, signed, atomic): (usize, bool, bool, Option<Atomic>) = match i.op {
        LoadUbyte | LoadSbyte => (1, true, i.op == LoadSbyte, None),
        LoadUshort | LoadSshort => (2, true, i.op == LoadSshort, None),
        LoadDword => (4, true, false, None),
        LoadDwordx2 => (8, true, false, None),
        LoadDwordx3 => (12, true, false, None),
        LoadDwordx4 => (16, true, false, None),
        StoreByte => (1, false, false, None),
        StoreShort => (2, false, false, None),
        StoreDword => (4, false, false, None),
        StoreDwordx2 => (8, false, false, None),
        StoreDwordx3 => (12, false, false, None),
        StoreDwordx4 => (16, false, false, None),
        AtomicAdd => (4, false, false, Some(Atomic::Add)),
        AtomicAddF32 => (4, false, false, Some(Atomic::AddFloat { denorm: 0 })),
    };
    let mut busy = Vec::new();
    let dst = if load || (atomic.is_some() && i.sc0) {
        Some(required(i.vdst, "GLOBAL destination")?)
    } else {
        None
    };
    if let Some(dst) = dst {
        destination(w, dst, size.div_ceil(4) as u8, false, &mut busy)?;
    }
    let base = i.saddr.map(|s| w.read_s64(s, Ty::B64)).transpose()?;
    let mut accesses = Vec::with_capacity(w.exec.count_ones() as usize);
    for lane in 0..WAVE {
        if w.lane_active(lane) {
            let addr = match base {
                Some(base) => base.wrapping_add(u64::from(w.read_v(
                    required(i.addr, "GLOBAL address")?,
                    lane,
                    Ty::B32,
                )?)),
                None => w.read_v64(required(i.addr, "GLOBAL address pair")?, lane, Ty::B64)?,
            }
            .wrapping_add_signed(i64::from(i.offset));
            let action = if load {
                Action::Load {
                    dst: dst.unwrap(),
                    scalar: false,
                    signed,
                }
            } else if let Some(op) = atomic {
                Action::Atomic {
                    op,
                    operand: w.read_v(required(i.data, "GLOBAL atomic data")?, lane, Ty::B32)?,
                    dst,
                }
            } else {
                Action::Store(store_data(
                    w,
                    required(i.data, "GLOBAL store data")?,
                    lane,
                    size,
                )?)
            };
            accesses.push(Access {
                lane,
                addr,
                size,
                valid: true,
                action,
            });
        }
    }
    Ok(prepared(Counter::Vm, 1, false, busy, accesses, false))
}

struct Buffer {
    base: u64,
    stride: u32,
    records: u32,
}
fn buffer(w: &Wave, src: Src, scalar: bool) -> SimResult<Buffer> {
    let mut words = [0; 4];
    for (j, word) in words.iter_mut().enumerate() {
        *word = w.read_s(reg_at(src, j)?, Ty::B32)?;
    }
    if !scalar && (words[1] & 0x8000_0000 != 0 || words[3] & (1 << 23) != 0) {
        return unsupported("MUBUF V# swizzle/add_tid descriptor");
    }
    if !scalar && words[3] & ((3 << 30) | (3 << 19)) != 0 {
        return unsupported("MUBUF V# non-buffer type/user VM descriptor");
    }
    Ok(Buffer {
        base: u64::from(words[0]) | (u64::from(words[1] & 0xffff) << 32),
        stride: (words[1] >> 16) & 0x3fff,
        records: words[2],
    })
}

/// Linear V# addressing: CDNA3 ISA §9.1.5 Tables 44-45 p.67-69, Table 47
/// p.71. Range checks exclude the SGPR offset, and x2/x3/x4 check each dword
/// independently. Out-of-range loads return zero; stores are discarded.
/// The guide omits range checking for idxen with zero stride; that form is
/// Unsupported, as are swizzled, add_tid and user-VM descriptors.
fn mubuf(w: &Wave, i: &Mubuf) -> SimResult<Prepared> {
    use MubufOp::*;
    let (size, load, signed): (usize, bool, bool) = match i.op {
        LoadUbyte | LoadSbyte => (1, true, i.op == LoadSbyte),
        LoadUshort | LoadSshort => (2, true, i.op == LoadSshort),
        LoadDword => (4, true, false),
        LoadDwordx2 => (8, true, false),
        LoadDwordx3 => (12, true, false),
        LoadDwordx4 => (16, true, false),
        StoreByte => (1, false, false),
        StoreShort => (2, false, false),
        StoreDword => (4, false, false),
        StoreDwordx2 => (8, false, false),
        StoreDwordx3 => (12, false, false),
        StoreDwordx4 => (16, false, false),
    };
    let resource = buffer(w, i.srsrc, false)?;
    if i.idxen && resource.stride == 0 {
        return unsupported(format!(
            "{} idxen with zero V# stride: omitted range-check form (CDNA3 ISA §9.1.5.1 p.68)",
            i.op.name()
        ));
    }
    let soffset = u64::from(w.read_s(i.soffset, Ty::B32)?);
    let mut busy = Vec::new();
    if load {
        destination(w, i.vdata, size.div_ceil(4) as u8, false, &mut busy)?;
    }
    let mut accesses = Vec::with_capacity(w.exec.count_ones() as usize * size.div_ceil(4));
    for lane in 0..WAVE {
        if w.lane_active(lane) {
            let index = if i.idxen {
                u64::from(w.read_v(required(i.vaddr, "MUBUF index")?, lane, Ty::B32)?)
            } else {
                0
            };
            let offset = u64::from(i.offset)
                + if i.offen {
                    u64::from(w.read_v(
                        reg_at(required(i.vaddr, "MUBUF offset")?, usize::from(i.idxen))?,
                        lane,
                        Ty::B32,
                    )?)
                } else {
                    0
                };
            for part in 0..size.div_ceil(4) {
                let component = offset + part as u64 * 4;
                let valid = if i.idxen && resource.stride != 0 {
                    index < u64::from(resource.records)
                } else {
                    component < u64::from(resource.records)
                };
                let addr = resource
                    .base
                    .wrapping_add(soffset)
                    .wrapping_add(index * u64::from(resource.stride))
                    .wrapping_add(component);
                let reg = reg_at(i.vdata, part)?;
                let n = size.min(4);
                let action = if load {
                    Action::Load {
                        dst: reg,
                        scalar: false,
                        signed,
                    }
                } else {
                    Action::Store(store_data(w, reg, lane, n)?)
                };
                accesses.push(Access {
                    lane,
                    addr,
                    size: n,
                    valid,
                    action,
                });
            }
        }
    }
    Ok(prepared(Counter::Vm, 1, false, busy, accesses, false))
}

/// Scalar addresses are dword aligned (base and offset low bits ignored),
/// buffer stride affects only bounds: CDNA3 ISA §8.2.1.1 p.60-61, §8.4 p.62.
/// Raw descriptors use num_records bytes and leave clamped SGPR components
/// unchanged (§8.4 p.62). Structured SMEM is Unsupported: the guide's
/// `m_size` formula does not specify unambiguous stride/record units.
fn smem(w: &Wave, i: &Smem) -> SimResult<Prepared> {
    use SmemOp::*;
    let dwords = match i.op {
        LoadDword | BufferLoadDword => 1,
        LoadDwordx2 | BufferLoadDwordx2 => 2,
        LoadDwordx4 | BufferLoadDwordx4 => 4,
        LoadDwordx8 | BufferLoadDwordx8 => 8,
        LoadDwordx16 | BufferLoadDwordx16 => 16,
        Memtime | Memrealtime | DcacheInv => return unsupported(i.op.name()),
    };
    let dst = required(i.sdata, "SMEM destination")?;
    let mut busy = Vec::new();
    destination(w, dst, dwords, true, &mut busy)?;
    let base_src = required(i.sbase, "SMEM base")?;
    let (base, limit) = if i.op.buffer() {
        let resource = buffer(w, base_src, true)?;
        if resource.stride != 0 {
            return unsupported(format!(
                "{} structured V# stride: ambiguous SMEM m_size units (CDNA3 ISA §8.2.1.1 p.61)",
                i.op.name()
            ));
        }
        (resource.base, Some(u64::from(resource.records)))
    } else {
        (w.read_s64(base_src, Ty::B64)?, None)
    };
    let offset = match i.offset {
        SmemOffset::Imm(n) => i64::from(n),
        SmemOffset::Sgpr(s) => i64::from(w.read_s(s, Ty::B32)?),
        SmemOffset::SgprImm(s, n) => i64::from(w.read_s(s, Ty::B32)?) + i64::from(n),
    };
    if offset < 0 {
        return fault("SMEM negative combined offset is undefined (CDNA3 ISA §8.2.1.1 p.60)");
    }
    let offset = offset as u64 & !3;
    let mut accesses = Vec::with_capacity(usize::from(dwords));
    for j in 0..usize::from(dwords) {
        let component = offset + j as u64 * 4;
        if limit.is_some_and(|n| component >= n) {
            continue;
        }
        accesses.push(Access {
            lane: 0,
            addr: (base & !3).wrapping_add(component),
            size: 4,
            valid: true,
            action: Action::Load {
                dst: reg_at(dst, j)?,
                scalar: true,
                signed: false,
            },
        });
    }
    Ok(prepared(
        Counter::Lgkm,
        if dwords == 1 { 1 } else { 2 },
        true,
        busy,
        accesses,
        false,
    ))
}

/// Snapshot register operands, stall on counter overflow, then queue the
/// deferred operation. Load destinations are busy until the op is removed.
pub fn issue(w: &mut Wave, inst: &Inst, lds: &mut [u8], mem: &mut Memory) -> SimResult<Flow> {
    let prepared = match inst {
        Inst::Smem(i) => smem(w, i)?,
        Inst::Ds(i) => ds(w, i)?,
        Inst::Flat(i) => global(w, i)?,
        Inst::Mubuf(i) => mubuf(w, i)?,
        _ => return unsupported(format!("memory issue {:?}", inst.format())),
    };
    let caps = w.arch.caps();
    let max = match prepared.counter {
        Counter::Vm => caps.vmcnt_max,
        Counter::Lgkm => caps.lgkmcnt_max,
        Counter::Exp => caps.expcnt_max,
    };
    while count(w, prepared.counter) + u16::from(prepared.weight) > u16::from(max) {
        let index = w
            .pending
            .iter()
            .position(|p| p.counter == prepared.counter)
            .ok_or_else(|| SimError::Wait("counter weight exceeds architecture maximum".into()))?;
        retire_at(w, index, lds, mem)?;
    }
    let id = w.next_id;
    w.next_id += 1;
    // Diagnostics use opcode names, not encoders' Display implementations.
    let what = match inst {
        Inst::Smem(i) => i.op.name().to_owned(),
        Inst::Ds(i) => i.op.name().to_owned(),
        Inst::Flat(i) => format!("{:?} {:?}", i.seg, i.op),
        Inst::Mubuf(i) => i.op.name().to_owned(),
        _ => return unsupported("non-memory instruction"),
    };
    w.pending.push_back(Pending {
        id,
        counter: prepared.counter,
        weight: prepared.weight,
        out_of_order: prepared.out_of_order,
        busy: prepared.busy,
        what,
        retire: prepared.retire,
    });
    Ok(Flow::Next)
}
