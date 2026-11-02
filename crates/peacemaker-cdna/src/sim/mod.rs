// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave64 workgroup emulator for PeaceMaker-emitted CDNA kernels.
//!
//! Executes a decoded code object over a simulated dispatch: per wave an SGPR
//! file, 256 VGPRs and 256 AccVGPRs × 64 lanes, EXEC/VCC/SCC/M0 and MODE; per
//! workgroup an LDS; one global [`Memory`]. Workgroups run one after another;
//! inside a workgroup waves run round-robin, each until it reaches
//! `s_barrier` or `s_endpgm`, and a barrier releases once every live wave has
//! arrived (CDNA3 ISA §4.3 p.19).
//!
//! **Wait counters are ordering, adversarially.** A memory instruction is
//! *issued* (addresses and store data are read from registers then) and
//! queued on its counter; it *retires* — load data sampled from memory and
//! written to registers, store data written to memory — only when an
//! `s_waitcnt` (or `s_endpgm`) requires it, or when the counter would exceed
//! its field maximum. Reading or writing a register an in-flight load will
//! write is a [`SimError::Wait`]. So a missing `s_waitcnt` is a deterministic
//! error or a wrong result, never a silent pass: a store not waited for before
//! `s_barrier` is invisible to the other waves, and a load not waited for
//! before a barrier observes the other waves' later stores (the LDS WAR rule
//! of `peacemaker-isa`'s ledger). VM_CNT retires in issue order including
//! mixed loads and stores; LGKM_CNT retires DS operations in order and SMEM
//! only at `lgkmcnt(0)` (CDNA3 ISA §4.4 p.19-20).
//!
//! Hazards (required wait states, CDNA3 ISA §4.5 Table 11, §7.5) are not
//! modelled: every instruction sees the architectural result of its
//! predecessors. `peacemaker-isa`'s `cdna::hazard` pass owns them.
//!
//! Module ownership: this file (scheduler, state, operand access) is the
//! crate owner's; `salu`, `valu`, `xlane`, `mfma`, `mem`, `load` are one
//! child each.
pub mod load;
pub mod mem;
pub mod mfma;
pub mod salu;
pub mod valu;
pub mod xlane;

use crate::arch::Arch;
use crate::enc::sop::{waitcnt_fields, SoppOp};
use crate::inst::{Inst, RegFile, RegSpan};
use crate::kd::KernelDescriptor;
use crate::operand::Src;
use crate::WAVE;
use std::collections::{HashMap, VecDeque};
use std::fmt;

pub use load::load_code_object;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    /// An instruction, operand form or descriptor feature the emulator does
    /// not model: never silently approximated.
    Unsupported(String),
    /// An architectural fault: unmapped or out-of-bounds memory, register
    /// beyond the descriptor's allocation, PC outside the code.
    Fault(String),
    /// A register accessed while an in-flight memory operation will write it
    /// (missing `s_waitcnt`), or an illegal counter use.
    Wait(String),
    /// Deadlock (barrier never satisfied) or the step limit.
    Hang(String),
}

impl SimError {
    /// Prefix the message with the failing instruction.
    pub fn at(self, offset: usize, inst: &Inst) -> Self {
        let tag = |m: String| format!("dword {offset} `{inst}`: {m}");
        match self {
            Self::Unsupported(m) => Self::Unsupported(tag(m)),
            Self::Fault(m) => Self::Fault(tag(m)),
            Self::Wait(m) => Self::Wait(tag(m)),
            Self::Hang(m) => Self::Hang(tag(m)),
        }
    }
}
impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Fault(m) => write!(f, "fault: {m}"),
            Self::Wait(m) => write!(f, "wait: {m}"),
            Self::Hang(m) => write!(f, "hang: {m}"),
        }
    }
}
impl std::error::Error for SimError {}
impl From<String> for SimError {
    fn from(m: String) -> Self {
        Self::Fault(m)
    }
}

pub type SimResult<T> = std::result::Result<T, SimError>;
pub fn unsupported<T>(what: impl fmt::Display) -> SimResult<T> {
    Err(SimError::Unsupported(what.to_string()))
}
pub fn fault<T>(what: impl fmt::Display) -> SimResult<T> {
    Err(SimError::Fault(what.to_string()))
}

// ----------------------------------------------------------------------------
// Global memory

#[derive(Clone, Debug)]
pub struct Region {
    pub name: String,
    pub base: u64,
    pub bytes: Vec<u8>,
    pub writable: bool,
}

/// Flat global memory: named, non-overlapping regions; any access outside
/// them faults.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub regions: Vec<Region>,
}

impl Memory {
    fn add(&mut self, name: &str, base: u64, bytes: Vec<u8>, writable: bool) -> SimResult<()> {
        let end = base
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| SimError::Fault("region address overflow".into()))?;
        if let Some(r) = self
            .regions
            .iter()
            .find(|r| base < r.base + r.bytes.len() as u64 && r.base < end)
        {
            return fault(format!("region {name} overlaps {}", r.name));
        }
        self.regions.push(Region {
            name: name.into(),
            base,
            bytes,
            writable,
        });
        Ok(())
    }
    pub fn map(&mut self, name: &str, base: u64, bytes: Vec<u8>) -> SimResult<()> {
        self.add(name, base, bytes, true)
    }
    pub fn map_readonly(&mut self, name: &str, base: u64, bytes: Vec<u8>) -> SimResult<()> {
        self.add(name, base, bytes, false)
    }
    pub fn unmap(&mut self, name: &str) -> Option<Region> {
        let i = self.regions.iter().position(|r| r.name == name)?;
        Some(self.regions.remove(i))
    }
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }
    fn locate(&self, addr: u64, n: usize) -> SimResult<(usize, usize)> {
        self.regions
            .iter()
            .enumerate()
            .find_map(|(i, r)| {
                let off = addr.checked_sub(r.base)?;
                let end = off.checked_add(n as u64)?;
                (end <= r.bytes.len() as u64).then_some((i, off as usize))
            })
            .ok_or_else(|| SimError::Fault(format!("unmapped global access {addr:#x} size {n}")))
    }
    pub fn read(&self, addr: u64, n: usize) -> SimResult<&[u8]> {
        let (i, o) = self.locate(addr, n)?;
        Ok(&self.regions[i].bytes[o..o + n])
    }
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> SimResult<()> {
        let (i, o) = self.locate(addr, bytes.len())?;
        let r = &mut self.regions[i];
        if !r.writable {
            return fault(format!("store to read-only region {} at {addr:#x}", r.name));
        }
        r.bytes[o..o + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Wait counters

/// The three CDNA3 counters (CDNA3 ISA §4.4 p.19-20).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Counter {
    Vm,
    Lgkm,
    Exp,
}

/// The deferred half of a memory instruction: samples memory / writes
/// registers, or writes memory, when the op retires.
pub type Retire = Box<dyn FnOnce(&mut Wave, &mut [u8], &mut Memory) -> SimResult<()>>;

/// One in-flight memory operation.
pub struct Pending {
    pub id: u64,
    pub counter: Counter,
    /// Retires only at count 0 (SMEM; CDNA3 ISA §4.4 p.20).
    pub out_of_order: bool,
    /// Units this op adds to its counter (SMEM: 1 for one dword, 2 for wider;
    /// everything else 1).
    pub weight: u8,
    /// Registers the op will write at retire (load destinations).
    pub busy: Vec<RegSpan>,
    /// The issuing instruction's text, for diagnostics.
    pub what: String,
    pub retire: Retire,
}

impl fmt::Debug for Pending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pending")
            .field("id", &self.id)
            .field("counter", &self.counter)
            .field("out_of_order", &self.out_of_order)
            .field("weight", &self.weight)
            .field("busy", &self.busy)
            .field("what", &self.what)
            .finish_non_exhaustive()
    }
}

// ----------------------------------------------------------------------------
// Wave state

/// MODE register fields the emulator honours (from COMPUTE_PGM_RSRC1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    /// FLOAT_ROUND_MODE_32 / _16_64 (0 = round to nearest even).
    pub round32: u8,
    pub round16_64: u8,
    /// FLOAT_DENORM_MODE_32 / _16_64 (3 = preserve input and output denormals).
    pub denorm32: u8,
    pub denorm16_64: u8,
    pub dx10_clamp: bool,
    pub ieee: bool,
    /// FP16_OVFL: F16 and F32→FP8/BF8 overflow saturates instead of Inf/NaN.
    pub fp16_ovfl: bool,
}
impl Default for Mode {
    fn default() -> Self {
        Self {
            round32: 0,
            round16_64: 0,
            denorm32: 3,
            denorm16_64: 3,
            dx10_clamp: true,
            ieee: true,
            fp16_ovfl: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveState {
    Running,
    Barrier,
    Ended,
}

/// How an operand's constant is materialized: inline constants and literals
/// depend on the operand type (CDNA3 ISA §5.2 Table 13; §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    B32,
    F32,
    B16,
    F16,
    B64,
    F64,
}

pub struct Wave {
    pub arch: Arch,
    /// Wave index inside its workgroup.
    pub id: usize,
    pub s: Vec<u32>,
    pub ttmp: [u32; 16],
    /// `v[reg][lane]`.
    pub v: Vec<[u32; WAVE]>,
    /// `a[reg][lane]`.
    pub a: Vec<[u32; WAVE]>,
    pub exec: u64,
    pub vcc: u64,
    pub scc: bool,
    pub m0: u32,
    pub flat_scratch: u64,
    pub xnack_mask: u64,
    pub mode: Mode,
    /// Index into [`Program::insts`].
    pub pc: usize,
    pub state: WaveState,
    pub pending: VecDeque<Pending>,
    pub next_id: u64,
    pub steps: u64,
    /// Registers the descriptor allocates: v0..vgpr_limit, a0..agpr_limit.
    pub vgpr_limit: u16,
    pub agpr_limit: u16,
    /// This wave's workgroup ID.
    pub workgroup: [u32; 3],
}

impl fmt::Debug for Wave {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wave")
            .field("id", &self.id)
            .field("pc", &self.pc)
            .field("state", &self.state)
            .field("exec", &format_args!("{:#018x}", self.exec))
            .field("vcc", &format_args!("{:#018x}", self.vcc))
            .field("scc", &self.scc)
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl Wave {
    /// A wave with every register set to `poison`, EXEC all ones, the full
    /// register files allocated, default MODE (for unit tests and `run`).
    pub fn new(arch: Arch, id: usize, poison: u32) -> Self {
        let caps = arch.caps();
        Self {
            arch,
            id,
            s: vec![poison; usize::from(caps.sgprs)],
            ttmp: [poison; 16],
            v: vec![[poison; WAVE]; usize::from(caps.vgprs)],
            a: vec![[poison; WAVE]; usize::from(caps.agprs)],
            exec: u64::MAX,
            vcc: u64::from(poison) | (u64::from(poison) << 32),
            scc: false,
            m0: poison,
            flat_scratch: 0,
            xnack_mask: 0,
            mode: Mode::default(),
            pc: 0,
            state: WaveState::Running,
            pending: VecDeque::new(),
            next_id: 0,
            steps: 0,
            vgpr_limit: caps.vgprs,
            agpr_limit: caps.agprs,
            workgroup: [0; 3],
        }
    }

    pub fn lane_active(&self, lane: usize) -> bool {
        self.exec >> lane & 1 != 0
    }

    /// Error if `span` is a destination of an in-flight memory operation.
    pub fn check_ready(&self, span: RegSpan, access: &str) -> SimResult<()> {
        for p in &self.pending {
            if p.busy.iter().any(|b| b.overlaps(span)) {
                return Err(SimError::Wait(format!(
                    "{access} {span:?} while `{}` (op #{}) is in flight: missing s_waitcnt",
                    p.what, p.id
                )));
            }
        }
        Ok(())
    }

    fn check_vgpr(&self, file: RegFile, n: u8, dwords: u8) -> SimResult<()> {
        let (limit, name) = if file == RegFile::Agpr {
            (self.agpr_limit, 'a')
        } else {
            (self.vgpr_limit, 'v')
        };
        if u16::from(n) + u16::from(dwords.max(1)) > limit {
            return fault(format!(
                "{name}{n} (+{dwords}) beyond the {limit} allocated by the kernel descriptor"
            ));
        }
        Ok(())
    }

    /// The 32-bit value of a scalar-readable operand (SGPR, special
    /// register, constant, literal). VGPR/AGPR operands are an error.
    pub fn read_s(&self, src: Src, ty: Ty) -> SimResult<u32> {
        if let Some(span) = crate::inst::RegSpan::of(src, 1) {
            self.check_ready(span, "read")?
        }
        Ok(match src {
            Src::Sgpr(n) => *self
                .s
                .get(usize::from(n))
                .ok_or_else(|| SimError::Fault(format!("s{n} beyond the SGPR file")))?,
            Src::Ttmp(n) => self.ttmp[usize::from(n)],
            Src::VccLo => self.vcc as u32,
            Src::VccHi => (self.vcc >> 32) as u32,
            Src::ExecLo => self.exec as u32,
            Src::ExecHi => (self.exec >> 32) as u32,
            Src::FlatScratchLo => self.flat_scratch as u32,
            Src::FlatScratchHi => (self.flat_scratch >> 32) as u32,
            Src::XnackMaskLo => self.xnack_mask as u32,
            Src::XnackMaskHi => (self.xnack_mask >> 32) as u32,
            Src::M0 => self.m0,
            Src::Scc => u32::from(self.scc),
            Src::Vccz => u32::from(self.vcc == 0),
            Src::Execz => u32::from(self.exec == 0),
            Src::Int(i) => match ty {
                Ty::B16 | Ty::F16 => u32::from(i16::from(i) as u16),
                _ => i32::from(i) as u32,
            },
            Src::Float(f) => match ty {
                Ty::B16 | Ty::F16 => u32::from(f.f16_bits()),
                _ => f.f32_bits(),
            },
            Src::Lit(v) => match ty {
                Ty::B16 | Ty::F16 => v & 0xffff,
                _ => v,
            },
            Src::SharedBase | Src::SharedLimit | Src::PrivateBase | Src::PrivateLimit => {
                return unsupported(format!("aperture operand {src:?}"))
            }
            Src::Vgpr(_) | Src::Agpr(_) => {
                return fault(format!("{src:?} is not a scalar operand"))
            }
        })
    }

    /// The 64-bit value of a scalar-readable operand pair. Inline integers
    /// sign-extend; inline floats give their F64 bits for `F64` and B64
    /// (`[UNVERIFIED]` for B64); a literal is the high dword for `F64` and
    /// zero-extended otherwise (`[UNVERIFIED]` against MI300X).
    pub fn read_s64(&self, src: Src, ty: Ty) -> SimResult<u64> {
        if let Some(span) = crate::inst::RegSpan::of(src, 2) {
            self.check_ready(span, "read")?
        }
        Ok(match src {
            Src::Sgpr(n) => {
                if n % 2 != 0 {
                    return fault(format!("64-bit operand at odd s{n}"));
                }
                let i = usize::from(n);
                if i + 1 >= self.s.len() {
                    return fault(format!("s[{n}:{}] beyond the SGPR file", n + 1));
                }
                u64::from(self.s[i]) | (u64::from(self.s[i + 1]) << 32)
            }
            Src::Ttmp(n) => {
                u64::from(self.ttmp[usize::from(n)])
                    | (u64::from(self.ttmp[usize::from(n) + 1]) << 32)
            }
            Src::VccLo => self.vcc,
            Src::ExecLo => self.exec,
            Src::FlatScratchLo => self.flat_scratch,
            Src::XnackMaskLo => self.xnack_mask,
            Src::Int(i) => i64::from(i) as u64,
            Src::Float(f) => f.f64_bits(),
            Src::Lit(v) => {
                if ty == Ty::F64 {
                    u64::from(v) << 32
                } else {
                    u64::from(v)
                }
            }
            Src::Scc | Src::Vccz | Src::Execz | Src::M0 => u64::from(self.read_s(src, Ty::B32)?),
            _ => return fault(format!("{src:?} is not a 64-bit scalar operand")),
        })
    }

    /// Write a 32-bit scalar destination.
    pub fn write_s(&mut self, dst: Src, value: u32) -> SimResult<()> {
        if let Some(span) = crate::inst::RegSpan::of(dst, 1) {
            self.check_ready(span, "write")?
        }
        let lo = |r: &mut u64| *r = (*r & !0xffff_ffff) | u64::from(value);
        let hi = |r: &mut u64| *r = (*r & 0xffff_ffff) | (u64::from(value) << 32);
        match dst {
            Src::Sgpr(n) => {
                *self
                    .s
                    .get_mut(usize::from(n))
                    .ok_or_else(|| SimError::Fault(format!("s{n} beyond the SGPR file")))? = value
            }
            Src::Ttmp(n) => self.ttmp[usize::from(n)] = value,
            Src::VccLo => lo(&mut self.vcc),
            Src::VccHi => hi(&mut self.vcc),
            Src::ExecLo => lo(&mut self.exec),
            Src::ExecHi => hi(&mut self.exec),
            Src::FlatScratchLo => lo(&mut self.flat_scratch),
            Src::FlatScratchHi => hi(&mut self.flat_scratch),
            Src::XnackMaskLo => lo(&mut self.xnack_mask),
            Src::XnackMaskHi => hi(&mut self.xnack_mask),
            Src::M0 => self.m0 = value,
            _ => return fault(format!("{dst:?} is not a scalar destination")),
        }
        Ok(())
    }

    /// Write a 64-bit scalar destination pair.
    pub fn write_s64(&mut self, dst: Src, value: u64) -> SimResult<()> {
        match dst {
            Src::Sgpr(n) if n % 2 == 0 => {
                self.write_s(Src::Sgpr(n), value as u32)?;
                self.write_s(Src::Sgpr(n + 1), (value >> 32) as u32)
            }
            Src::Ttmp(n) if n % 2 == 0 => {
                self.write_s(Src::Ttmp(n), value as u32)?;
                self.write_s(Src::Ttmp(n + 1), (value >> 32) as u32)
            }
            Src::VccLo | Src::ExecLo | Src::FlatScratchLo | Src::XnackMaskLo => {
                if let Some(span) = crate::inst::RegSpan::of(dst, 2) {
                    self.check_ready(span, "write")?
                }
                match dst {
                    Src::VccLo => self.vcc = value,
                    Src::ExecLo => self.exec = value,
                    Src::FlatScratchLo => self.flat_scratch = value,
                    _ => self.xnack_mask = value,
                }
                Ok(())
            }
            _ => fault(format!("{dst:?} is not a 64-bit scalar destination")),
        }
    }

    /// Lane `lane` of a 32-bit vector-ALU source: a VGPR/AGPR lane, or a
    /// scalar operand broadcast to every lane.
    pub fn read_v(&self, src: Src, lane: usize, ty: Ty) -> SimResult<u32> {
        match src {
            Src::Vgpr(n) => {
                self.check_vgpr(RegFile::Vgpr, n, 1)?;
                self.check_ready(RegSpan::new(RegFile::Vgpr, n, 1), "read")?;
                Ok(self.v[usize::from(n)][lane])
            }
            Src::Agpr(n) => {
                self.check_vgpr(RegFile::Agpr, n, 1)?;
                self.check_ready(RegSpan::new(RegFile::Agpr, n, 1), "read")?;
                Ok(self.a[usize::from(n)][lane])
            }
            _ => self.read_s(src, ty),
        }
    }

    /// Lane `lane` of a 64-bit vector-ALU source (VGPR pair or scalar pair /
    /// constant per [`Wave::read_s64`]).
    pub fn read_v64(&self, src: Src, lane: usize, ty: Ty) -> SimResult<u64> {
        match src {
            Src::Vgpr(n) => {
                self.check_vgpr(RegFile::Vgpr, n, 2)?;
                self.check_ready(RegSpan::new(RegFile::Vgpr, n, 2), "read")?;
                Ok(u64::from(self.v[usize::from(n)][lane])
                    | (u64::from(self.v[usize::from(n) + 1][lane]) << 32))
            }
            Src::Agpr(n) => {
                self.check_vgpr(RegFile::Agpr, n, 2)?;
                self.check_ready(RegSpan::new(RegFile::Agpr, n, 2), "read")?;
                Ok(u64::from(self.a[usize::from(n)][lane])
                    | (u64::from(self.a[usize::from(n) + 1][lane]) << 32))
            }
            _ => self.read_s64(src, ty),
        }
    }

    /// Write lane `lane` of VGPR/AGPR `dst` (the caller applies EXEC).
    pub fn write_v(&mut self, dst: Src, lane: usize, value: u32) -> SimResult<()> {
        match dst {
            Src::Vgpr(n) => {
                self.check_vgpr(RegFile::Vgpr, n, 1)?;
                self.check_ready(RegSpan::new(RegFile::Vgpr, n, 1), "write")?;
                self.v[usize::from(n)][lane] = value;
                Ok(())
            }
            Src::Agpr(n) => {
                self.check_vgpr(RegFile::Agpr, n, 1)?;
                self.check_ready(RegSpan::new(RegFile::Agpr, n, 1), "write")?;
                self.a[usize::from(n)][lane] = value;
                Ok(())
            }
            _ => fault(format!("{dst:?} is not a vector destination")),
        }
    }

    /// Write lane `lane` of a VGPR/AGPR pair.
    pub fn write_v64(&mut self, dst: Src, lane: usize, value: u64) -> SimResult<()> {
        let next = match dst {
            Src::Vgpr(n) => Src::Vgpr(n + 1),
            Src::Agpr(n) => Src::Agpr(n + 1),
            _ => return fault(format!("{dst:?} is not a vector destination")),
        };
        self.write_v(dst, lane, value as u32)?;
        self.write_v(next, lane, (value >> 32) as u32)
    }
}

// ----------------------------------------------------------------------------
// Program and dispatch

/// One decoded instruction at dword `offset` spanning `len` dwords.
#[derive(Clone, Debug)]
pub struct Decoded {
    pub offset: usize,
    pub len: usize,
    pub inst: Inst,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub arch: Arch,
    pub insts: Vec<Decoded>,
    index: HashMap<usize, usize>,
}

impl Program {
    pub fn decode(arch: Arch, words: &[u32]) -> SimResult<Self> {
        let decoded = crate::inst::decode_all(arch, words)?;
        let mut insts = Vec::with_capacity(decoded.len());
        for (i, (offset, inst)) in decoded.iter().enumerate() {
            let end = decoded.get(i + 1).map_or(words.len(), |(o, _)| *o);
            insts.push(Decoded {
                offset: *offset,
                len: end - offset,
                inst: inst.clone(),
            });
        }
        let index = insts
            .iter()
            .enumerate()
            .map(|(i, d)| (d.offset, i))
            .collect();
        Ok(Self { arch, insts, index })
    }
    /// Instruction index starting at dword `offset`.
    pub fn at(&self, offset: i64) -> SimResult<usize> {
        usize::try_from(offset)
            .ok()
            .and_then(|o| self.index.get(&o).copied())
            .ok_or_else(|| {
                SimError::Fault(format!(
                    "branch target dword {offset} is not an instruction boundary"
                ))
            })
    }
}

/// A loaded kernel: its code dwords (entry first) and descriptor.
#[derive(Clone, Debug)]
pub struct KernelImage {
    pub arch: Arch,
    pub name: String,
    pub words: Vec<u32>,
    pub kd: KernelDescriptor,
}

/// One dispatch.
#[derive(Clone, Debug)]
pub struct Launch {
    /// Workgroups per dimension.
    pub grid: [u32; 3],
    /// Work-items per workgroup per dimension.
    pub block: [u32; 3],
    /// Kernel argument bytes, mapped read-only at `kernarg_addr` for the run.
    pub kernarg: Vec<u8>,
    pub kernarg_addr: u64,
    /// LDS bytes beyond the descriptor's `group_segment_fixed_size`.
    pub dynamic_lds: u32,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Initial value of every SGPR/VGPR/AGPR not set by the dispatch ABI.
    pub poison: u32,
    /// Initial LDS byte.
    pub lds_poison: u8,
    /// Per-wave instruction limit.
    pub max_steps: u64,
    /// MFMA F32 accumulation model ([`mfma::AccumModel`]).
    pub accum: mfma::AccumModel,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            poison: 0xbaad_f00d,
            lds_poison: 0xa5,
            max_steps: 50_000_000,
            accum: mfma::AccumModel::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub workgroups: u64,
    pub waves: u64,
    pub instructions: u64,
}

/// Control-flow outcome of one instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Next,
    Branch(i16),
    Barrier,
    End,
}

/// Execute one instruction on `w` (no PC update). `s_waitcnt`, `s_barrier`
/// and `s_endpgm` are handled here; every other format goes to its module.
pub fn exec_one(
    w: &mut Wave,
    inst: &Inst,
    lds: &mut [u8],
    mem: &mut Memory,
    cfg: &Config,
) -> SimResult<Flow> {
    match inst {
        Inst::Sopp(p) if p.op == SoppOp::Waitcnt => {
            let (vm, exp, lgkm) = waitcnt_fields(w.arch, p.simm16);
            self::mem::wait(w, lds, mem, vm, lgkm, exp)?;
            Ok(Flow::Next)
        }
        Inst::Sopp(p) if p.op == SoppOp::Barrier => Ok(Flow::Barrier),
        Inst::Sopp(p) if p.op == SoppOp::Endpgm => {
            self::mem::drain(w, lds, mem)?;
            Ok(Flow::End)
        }
        Inst::Sop1(_) | Inst::Sop2(_) | Inst::Sopk(_) | Inst::Sopc(_) | Inst::Sopp(_) => {
            salu::exec(w, inst)
        }
        Inst::Smem(_) | Inst::Ds(_) | Inst::Flat(_) | Inst::Mubuf(_) => {
            self::mem::issue(w, inst, lds, mem)
        }
        Inst::Vop1(_) | Inst::Vop2(_) | Inst::Vopc(_) | Inst::Vop3(_) => valu::exec(w, inst),
        Inst::Vop3p(_) | Inst::Mai(_) => mfma::exec(w, inst, cfg),
    }
}

fn run_wave(
    w: &mut Wave,
    prog: &Program,
    lds: &mut [u8],
    mem: &mut Memory,
    cfg: &Config,
) -> SimResult<u64> {
    let mut n = 0;
    loop {
        if w.steps >= cfg.max_steps {
            return Err(SimError::Hang(format!(
                "wave {} exceeded {} instructions",
                w.id, cfg.max_steps
            )));
        }
        let d = prog.insts.get(w.pc).ok_or_else(|| {
            SimError::Fault(format!("wave {} ran past the end of the code", w.id))
        })?;
        let flow = exec_one(w, &d.inst, lds, mem, cfg).map_err(|e| e.at(d.offset, &d.inst))?;
        w.steps += 1;
        n += 1;
        match flow {
            Flow::Next => w.pc += 1,
            Flow::Branch(off) => w.pc = prog.at(d.offset as i64 + d.len as i64 + i64::from(off))?,
            Flow::Barrier => {
                w.pc += 1;
                w.state = WaveState::Barrier;
                return Ok(n);
            }
            Flow::End => {
                w.state = WaveState::Ended;
                return Ok(n);
            }
        }
    }
}

/// The initial state of wave `index` of workgroup `wg` (LLVM `AMDGPUUsage`
/// "Initial Kernel Execution State"): user SGPRs (only the kernarg segment
/// pointer is modelled), then the enabled workgroup-ID SGPRs, work-item IDs
/// in v0 (packed X[9:0] Y[19:10] Z[29:20] on gfx90a+), EXEC = the lanes that
/// hold work-items, MODE from COMPUTE_PGM_RSRC1, register allocation from the
/// granulated VGPR count and ACCUM_OFFSET.
pub fn init_wave(
    image: &KernelImage,
    launch: &Launch,
    wg: [u32; 3],
    index: usize,
    cfg: &Config,
) -> SimResult<Wave> {
    let kd = &image.kd;
    let caps = image.arch.caps();
    let mut w = Wave::new(image.arch, index, cfg.poison);
    w.workgroup = wg;
    if kd.enable_sgpr_private_segment_buffer()
        || kd.enable_sgpr_dispatch_ptr()
        || kd.enable_sgpr_queue_ptr()
        || kd.enable_sgpr_dispatch_id()
        || kd.enable_sgpr_flat_scratch_init()
        || kd.enable_sgpr_private_segment_size()
    {
        return unsupported("user SGPRs other than the kernarg segment pointer");
    }
    if kd.enable_wavefront_size32() {
        return unsupported("wave32 descriptor on a wave64 CDNA target");
    }
    if kd.kernarg_preload != 0 {
        return unsupported("kernarg preload");
    }
    if kd.private_segment_fixed_size != 0 || kd.enable_private_segment() {
        return unsupported("private (scratch) segment");
    }
    if kd.enable_workgroup_info() {
        return unsupported("workgroup info SGPR");
    }
    let mut sg = 0usize;
    if kd.enable_sgpr_kernarg_segment_ptr() {
        w.s[0] = launch.kernarg_addr as u32;
        w.s[1] = (launch.kernarg_addr >> 32) as u32;
        sg = 2;
    }
    if kd.user_sgpr_count() as usize != sg {
        return unsupported(format!(
            "user_sgpr_count {} with {sg} modelled user SGPRs",
            kd.user_sgpr_count()
        ));
    }
    for dim in 0..3 {
        if kd.enable_workgroup_id(dim) {
            w.s[sg] = wg[dim as usize];
            sg += 1;
        }
    }
    let [bx, by, bz] = launch.block;
    let items = (bx * by * bz) as usize;
    let tid_dims = kd.enable_vgpr_workitem_id();
    if !caps.packed_tid {
        return unsupported("unpacked work-item IDs");
    }
    w.exec = 0;
    for lane in 0..WAVE {
        let flat = index * WAVE + lane;
        if flat >= items {
            continue;
        }
        w.exec |= 1 << lane;
        let (x, y, z) = (
            (flat as u32) % bx,
            (flat as u32 / bx) % by,
            flat as u32 / (bx * by),
        );
        w.v[0][lane] =
            x | if tid_dims >= 1 { y << 10 } else { 0 } | if tid_dims >= 2 { z << 20 } else { 0 };
    }
    w.mode = Mode {
        round32: kd.float_round_mode_32() as u8,
        round16_64: kd.float_round_mode_16_64() as u8,
        denorm32: kd.float_denorm_mode_32() as u8,
        denorm16_64: kd.float_denorm_mode_16_64() as u8,
        dx10_clamp: kd.dx10_clamp(),
        ieee: kd.ieee_mode(),
        fp16_ovfl: kd.fp16_overflow(),
    };
    let unified = (kd.granulated_workitem_vgpr_count() as u16 + 1) * caps.vgpr_granule;
    let accum = 4 * (kd.accum_offset_field() as u16 + 1);
    if accum > unified || unified > caps.unified_vgprs {
        return fault(format!(
            "accum_offset {accum} / unified VGPRs {unified} inconsistent"
        ));
    }
    w.vgpr_limit = accum.min(caps.vgprs);
    w.agpr_limit = (unified - accum).min(caps.agprs);
    Ok(w)
}

/// Run a dispatch to completion. The kernarg buffer is mapped read-only at
/// `launch.kernarg_addr` for the duration and unmapped afterwards.
pub fn run(
    image: &KernelImage,
    launch: &Launch,
    mem: &mut Memory,
    cfg: &Config,
) -> SimResult<Stats> {
    let caps = image.arch.caps();
    let prog = Program::decode(image.arch, &image.words)?;
    let items = launch.block.iter().map(|&d| u64::from(d)).product::<u64>();
    if items == 0 || items > 1024 || launch.grid.contains(&0) {
        return fault(format!(
            "launch {launch:?} is empty or exceeds 1024 work-items"
        ));
    }
    let lds_bytes = image.kd.group_segment_fixed_size + launch.dynamic_lds;
    if lds_bytes > caps.lds_bytes {
        return fault(format!("{lds_bytes} LDS bytes exceed {}", caps.lds_bytes));
    }
    let waves = (items as usize).div_ceil(WAVE);
    mem.map_readonly("kernarg", launch.kernarg_addr, launch.kernarg.clone())?;
    let result = (|| {
        let mut stats = Stats::default();
        for z in 0..launch.grid[2] {
            for y in 0..launch.grid[1] {
                for x in 0..launch.grid[0] {
                    let mut lds = vec![cfg.lds_poison; lds_bytes as usize];
                    let mut ws = (0..waves)
                        .map(|i| init_wave(image, launch, [x, y, z], i, cfg))
                        .collect::<SimResult<Vec<_>>>()?;
                    loop {
                        let mut progressed = false;
                        for w in ws.iter_mut().filter(|w| w.state == WaveState::Running) {
                            stats.instructions += run_wave(w, &prog, &mut lds, mem, cfg)?;
                            progressed = true;
                        }
                        if ws.iter().all(|w| w.state == WaveState::Ended) {
                            break;
                        }
                        if ws.iter().all(|w| w.state != WaveState::Running) {
                            for w in ws.iter_mut().filter(|w| w.state == WaveState::Barrier) {
                                w.state = WaveState::Running
                            }
                            continue;
                        }
                        if !progressed {
                            return Err(SimError::Hang(format!(
                                "workgroup {:?}: no wave can make progress",
                                [x, y, z]
                            )));
                        }
                    }
                    stats.workgroups += 1;
                    stats.waves += waves as u64;
                }
            }
        }
        Ok(stats)
    })();
    mem.unmap("kernarg");
    result
}
