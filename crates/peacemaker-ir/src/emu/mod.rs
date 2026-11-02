// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! In-order wave32/wave64 interpreter. By default wait counters are no-ops (loads complete before the next instruction)
//! and workgroup waves run in one fixed interleaving; `Emulator::hw_model` enables the gate model (in-flight
//! memory, LDS race detection, schedule exploration; see `hw.rs`). Workgroup barriers rendezvous distinct waves.
//!
//! Wavefront size: taken from the lifted kernel (`Kernel::wave`, derived from KERNEL_CODE_PROPERTIES
//! ENABLE_WAVEFRONT_SIZE32; `State::new` refuses a disagreement). A wave owns `lanes` (32 or 64) VGPR lanes, 64-bit
//! EXEC/VCC (a wave32 keeps the upper words zero) and splits the workgroup's work-items `wave * lanes + lane`. The
//! encodings are wave-size neutral (`vcc_lo`/`exec_lo` and `s[n:n+1]` name whole lane masks in wave64), so lane-mask
//! operands (`v_cmp*` destination, `v_cndmask`/carry-in source, carry-out, `v_div_scale` flag) go through
//! `State::read_mask`/`put_mask`: one dword in wave32, an SGPR pair in wave64. DPP rows are 16 lanes in both sizes,
//! `ds_swizzle` bit/quad modes act inside 32-lane groups, `ds_bpermute` wraps at the wave size, `v_mbcnt_lo/hi` count per
//! 32-lane dword, `v_readlane`/`v_writelane` select modulo the wave size. The WMMA/SWMMAC, permlane16 and wave32-only
//! `s_*_saveexec_b32` rows refuse a wave64 wave.
//!
//! Numerics (every kernel arch): the descriptor must preserve f32/f16/f64 denormals and round to nearest even
//! (`State::new` rejects any other FLOAT_MODE: gfx1201 f32 is IEEE with denormals when the KD says so). gfx1201 shares
//! the measured gfx1151/gfx1201 tables of `alu.rs` wherever the sweeps are identical (see its module doc for the
//! per-opcode differences, selected through `Arch::Gfx1201`); a gfx1201 object never runs on the gfx1151 model.
//! `Emulator::nan_meets_nan` (lab exactness contract) relaxes only the NaN-layout refusals, never non-NaN arithmetic.
mod alu;
pub mod convert;
mod dot;
#[cfg(test)]
mod gate_ops_tests;
#[cfg(test)]
mod gfx11_decode_gap_tests;
#[cfg(test)]
mod gfx12_decode_gap_tests;
mod hw;
pub use hw::RunReport;
mod memory;
pub mod mma;
#[cfg(test)]
mod swmmac_fixtures;
pub mod trans;
#[cfg(test)]
mod wave64_tests;
mod wide;
mod wmma;
use crate::{
    inst::{Abi, Wave},
    operand::{Half, ImmField, InlineConst, Operand, Special},
    reg::Kind,
    Arch, Inst, Kernel, Program,
};

pub type Result<T, E = String> = std::result::Result<T, E>;
#[derive(Clone, Debug)]
pub struct WriteSet {
    bits: Vec<u64>,
    bytes: usize,
}
impl WriteSet {
    fn new(bytes: usize) -> Self {
        Self {
            bits: vec![0; bytes.div_ceil(64)],
            bytes,
        }
    }
    pub fn contains(&self, byte: usize) -> bool {
        byte < self.bytes && self.bits[byte / 64] & (1u64 << (byte % 64)) != 0
    }
    pub fn count(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }
    /// First byte of `[start, start+len)` that was never written, if any (bytes past the set count as unwritten).
    pub fn first_unwritten(&self, start: usize, len: usize) -> Option<usize> {
        (start..start + len).find(|&b| !self.contains(b))
    }
    fn set(&mut self, byte: usize, on: bool) {
        if on {
            self.bits[byte / 64] |= 1u64 << (byte % 64);
        } else {
            self.bits[byte / 64] &= !(1u64 << (byte % 64));
        }
    }
    fn mark(&mut self, start: usize, len: usize) {
        let mut pos = start;
        let end = start + len;
        while pos < end {
            let n = (64 - pos % 64).min(end - pos);
            let mask = if n == 64 {
                u64::MAX
            } else {
                ((1u64 << n) - 1) << (pos % 64)
            };
            self.bits[pos / 64] |= mask;
            pos += n;
        }
    }
}
#[derive(Clone, Debug)]
pub struct Region {
    pub name: String,
    pub base: u64,
    pub bytes: Vec<u8>,
    pub written: WriteSet,
    /// A load of a byte of this region that no store has written is an error (an uninitialised workspace).
    pub strict_read: bool,
}
/// One journaled global store (at most 16 bytes): the bytes and written-bits it replaced.
#[derive(Clone, Debug)]
struct JournalEntry {
    region: usize,
    off: usize,
    len: usize,
    old: [u8; 16],
    was_written: u16,
}
/// The bytes one workgroup run stored, as sorted unique `(region, offset, byte)`: two schedules of the same
/// workgroup agree iff their images are equal.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StoreImage(pub Vec<(u32, u64, u8)>);
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub regions: Vec<Region>,
    journal: Option<Vec<JournalEntry>>,
}
impl Memory {
    pub fn map(&mut self, name: String, base: u64, bytes: Vec<u8>) -> Result<()> {
        let end = base
            .checked_add(bytes.len() as u64)
            .ok_or("region address overflow")?;
        if self
            .regions
            .iter()
            .any(|r| base < r.base + r.bytes.len() as u64 && r.base < end)
        {
            return Err("overlapping captured regions".into());
        }
        self.regions.push(Region {
            name,
            base,
            written: WriteSet::new(bytes.len()),
            bytes,
            strict_read: false,
        });
        Ok(())
    }
    fn locate(&self, addr: u64, n: usize) -> Result<(usize, usize)> {
        self.regions
            .iter()
            .enumerate()
            .find_map(|(i, r)| {
                let off = addr.checked_sub(r.base)?;
                let end = off.checked_add(n as u64)?;
                (end <= r.bytes.len() as u64).then_some((i, off as usize))
            })
            .ok_or_else(|| format!("unmapped memory access {addr:#x} size {n}"))
    }
    pub fn read(&self, addr: u64, n: usize) -> Result<&[u8]> {
        let (i, o) = self.locate(addr, n)?;
        let r = &self.regions[i];
        if r.strict_read {
            if let Some(b) = r.written.first_unwritten(o, n) {
                return Err(format!(
                    "[uninit] read of byte {b:#x} (address {:#x}) of region `{}` that no store has written",
                    r.base + b as u64,
                    r.name
                ));
            }
        }
        Ok(&r.bytes[o..o + n])
    }
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<()> {
        let (i, o) = self.locate(addr, bytes.len())?;
        let r = &mut self.regions[i];
        if let Some(j) = self.journal.as_mut() {
            for (c, chunk) in bytes.chunks(16).enumerate() {
                let off = o + c * 16;
                let mut e = JournalEntry {
                    region: i,
                    off,
                    len: chunk.len(),
                    old: [0; 16],
                    was_written: 0,
                };
                e.old[..chunk.len()].copy_from_slice(&r.bytes[off..off + chunk.len()]);
                for k in 0..chunk.len() {
                    e.was_written |= u16::from(r.written.contains(off + k)) << k;
                }
                j.push(e);
            }
        }
        r.bytes[o..o + bytes.len()].copy_from_slice(bytes);
        r.written.mark(o, bytes.len());
        Ok(())
    }
    /// Marks `name` so a load of any byte not yet stored is an error.
    pub fn set_strict_read(&mut self, name: &str) -> Result<()> {
        let r = self
            .regions
            .iter_mut()
            .find(|r| r.name == name)
            .ok_or_else(|| format!("no region `{name}`"))?;
        r.strict_read = true;
        Ok(())
    }
    /// Starts recording every store so the run can be undone (`journal_undo`) and compared (`journal_image`).
    fn journal_start(&mut self) {
        self.journal = Some(Vec::new());
    }
    fn journal_stop(&mut self) {
        self.journal = None;
    }
    /// Final bytes of every location stored since `journal_start`/the last undo.
    fn journal_image(&self) -> StoreImage {
        let mut at: Vec<(u32, u64)> = self
            .journal
            .iter()
            .flatten()
            .flat_map(|e| (0..e.len).map(move |k| (e.region as u32, (e.off + k) as u64)))
            .collect();
        at.sort_unstable();
        at.dedup();
        StoreImage(
            at.into_iter()
                .map(|(r, o)| (r, o, self.regions[r as usize].bytes[o as usize]))
                .collect(),
        )
    }
    /// Restores the bytes and written-bits every journaled store replaced (newest first) and clears the journal.
    fn journal_undo(&mut self) {
        let Some(j) = self.journal.as_mut() else {
            return;
        };
        let entries = std::mem::take(j);
        for e in entries.iter().rev() {
            let r = &mut self.regions[e.region];
            r.bytes[e.off..e.off + e.len].copy_from_slice(&e.old[..e.len]);
            for k in 0..e.len {
                r.written.set(e.off + k, e.was_written >> k & 1 != 0);
            }
        }
    }
    /// Writes an image (schedule 0's result) back through the normal store path.
    fn apply_image(&mut self, image: &StoreImage) -> Result<()> {
        for &(r, o, b) in &image.0 {
            let region = &mut self.regions[r as usize];
            region.bytes[o as usize] = b;
            region.written.mark(o as usize, 1);
        }
        Ok(())
    }
    /// Maps `bytes` as its own region placed above every existing region with at least `guard` unmapped bytes before it
    /// (and, for the first region, below it), 4 KiB aligned; returns the base. The region is exactly `bytes.len()` long, so any
    /// access past either end lands in unmapped space and `locate` refuses it.
    pub fn map_guarded(&mut self, name: String, bytes: Vec<u8>, guard: u64) -> Result<u64> {
        let top = self
            .regions
            .iter()
            .map(|r| r.base + r.bytes.len() as u64)
            .max()
            .unwrap_or(0);
        let base = top
            .checked_add(guard)
            .and_then(|b| b.checked_add(0xfff))
            .ok_or("region address overflow")?
            & !0xfff;
        self.map(name, base, bytes)?;
        Ok(base)
    }
}

/// How a 32-bit literal extends into a 64-bit source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Wide {
    Unsigned,
    Signed,
    Float,
}
pub struct Launch {
    pub kernarg_ptr: u64,
    pub workgroup: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_lds: u32,
}
/// High dword of the generic (flat) address of the LDS ("shared") aperture, the value of `src_shared_base`'s upper half,
/// and of the private aperture (`src_private_base`). The hardware's aperture bases are programmed by the driver; the
/// emulator fixes distinct small values that cannot collide with a mapped global region (every captured buffer lives
/// below 4 GiB, i.e. high dword 0). A flat access whose address has this high dword goes to LDS at the low 32 bits.
pub(super) const SHARED_APERTURE_HI: u32 = 2;
pub(super) const PRIVATE_APERTURE_HI: u32 = 3;
struct State {
    s: [u32; 106],
    t: [u32; 16],
    /// 64 lane slots; only `lanes` (32 or 64, the kernel's wavefront size) are architecturally present.
    v: Box<[[u32; 64]; 256]>,
    /// Wavefront size: 32 or 64. Wave32 keeps the upper halves of `exec`/`vcc` zero.
    lanes: usize,
    exec: u64,
    vcc: u64,
    /// Lab NaN policy (`Emulator::nan_meets_nan`): unmeasured NaN operand layouts of otherwise-qualified f32 opcodes
    /// widen and propagate a NaN instead of failing closed; no payload claim.
    nan_loose: bool,
    scc: bool,
    m0: u32,
    pc: usize,
    ended: bool,
    barrier: Option<u64>,
    signals: u64,
    /// Per-lane private (scratch) memory, `private_segment_fixed_size` bytes per work-item; empty when the kernel
    /// allocates none. Lane-private and wave-private by construction: never shared, never journaled.
    scratch: Vec<Vec<u8>>,
}
impl State {
    fn new(arch: Arch, kernel: &Kernel, launch: &Launch, wave: usize) -> Result<Self> {
        let lanes = match kernel.wave {
            Wave::Wave32 => 32usize,
            Wave::Wave64 => 64usize,
        };
        let Abi::Hsa {
            descriptor: d,
            metadata,
        } = &kernel.abi
        else {
            return Err("descriptor-driven HSA ABI required".into());
        };
        // The lifter derives `kernel.wave` from KERNEL_CODE_PROPERTIES.ENABLE_WAVEFRONT_SIZE32; refuse a disagreement.
        if d.kernel_code_properties.wave32() != (lanes == 32) {
            return Err(
                "kernel wave disagrees with the descriptor ENABLE_WAVEFRONT_SIZE32 bit".into(),
            );
        }
        if metadata.parsed.uses_dynamic_stack {
            return Err("dynamic stack (private segment beyond the fixed size) unsupported".into());
        }
        let mode = d.compute_pgm_rsrc1.0;
        if (mode >> 12) & 255 != 0xf0 || mode & (1 << 26) != 0 {
            return Err("only RNE with preserved denormals and FP16_OVFL=0 is modeled".into());
        }
        if arch == Arch::Gfx1151 && mode & ((1 << 23) | (1 << 21)) != ((1 << 23) | (1 << 21)) {
            return Err("gfx1151 emulator requires IEEE_MODE and DX10_CLAMP".into());
        }
        let mut st = Self {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes,
            nan_loose: false,
            exec: 0,
            vcc: 0,
            scc: false,
            m0: 0,
            pc: 0,
            ended: false,
            barrier: None,
            signals: 0,
            scratch: vec![vec![0; d.private_segment_fixed_size as usize]; lanes],
        };
        let props = d.kernel_code_properties.0;
        let mut sg = 0;
        for (bit, n) in [(0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1)] {
            if props & (1 << bit) == 0 {
                continue;
            }
            if bit != 3 {
                return Err(format!("unsupported descriptor user SGPR property {bit}"));
            }
            st.s[sg] = launch.kernarg_ptr as u32;
            st.s[sg + 1] = (launch.kernarg_ptr >> 32) as u32;
            sg += n;
        }
        let rs = d.compute_pgm_rsrc2.0;
        // hipcc gfx11 kernels declare more user SGPRs than they enable (padding between the last enabled user
        // register and the workgroup ids, left 0); the ids follow the DECLARED count. gfx1201 ids are in ttmps.
        let declared = ((rs >> 1) & 31) as usize;
        if sg > declared || (sg != declared && arch == Arch::Gfx1201) {
            return Err("descriptor user SGPR count disagrees with enabled properties".into());
        }
        sg = declared;
        if arch == Arch::Gfx1201 {
            st.t[9] = launch.workgroup[0];
            st.t[7] = if rs & (1 << 8) != 0 {
                launch.workgroup[1] & 0xffff
            } else {
                0
            } | if rs & (1 << 9) != 0 {
                (launch.workgroup[2] & 0xffff) << 16
            } else {
                0
            };
            st.t[8] = ((wave as u32 & 31) << 25)
                | if rs & ((1 << 8) | (1 << 9)) != 0 {
                    1 << 30
                } else {
                    0
                };
        } else {
            for (axis, bit) in [7, 8, 9].into_iter().enumerate() {
                if rs & (1 << bit) != 0 {
                    st.s[sg] = launch.workgroup[axis];
                    sg += 1;
                }
            }
            if rs & (1 << 10) != 0 {
                st.s[sg] = launch.block[0] * launch.block[1] * launch.block[2];
            }
        }
        let preload = d.kernarg_preload.0;
        if preload != 0 {
            return Err("kernarg preload ABI unsupported".into());
        }
        let workitem_mode = (rs >> 11) & 3;
        if workitem_mode == 3 {
            return Err("undefined ENABLE_VGPR_WORKITEM_ID=3".into());
        }
        let threads = launch.block[0]
            .checked_mul(launch.block[1])
            .and_then(|n| n.checked_mul(launch.block[2]))
            .ok_or("block size overflow")? as usize;
        for lane in 0..lanes {
            let id = wave * lanes + lane;
            if id >= threads {
                continue;
            }
            st.exec |= 1 << lane;
            let x = id as u32 % launch.block[0];
            let y = id as u32 / launch.block[0] % launch.block[1];
            let z = id as u32 / (launch.block[0] * launch.block[1]);
            st.v[0][lane] = x
                | (if workitem_mode > 0 { y << 10 } else { 0 })
                | (if workitem_mode > 1 { z << 20 } else { 0 });
        }
        Ok(st)
    }
    fn read(&self, op: &Operand, lane: usize, word: usize) -> Result<u32> {
        Ok(match op {
            Operand::Reg(r) => {
                let n = usize::from(r.base) + word;
                if word >= usize::from(r.len) {
                    return Err("register component exceeds operand width".into());
                }
                match r.kind {
                    Kind::S => self.s[n],
                    Kind::V => self.v[n][lane],
                    Kind::Ttmp => self.t[n],
                }
            }
            Operand::Half(r, h) => {
                let bits = self.read(&Operand::Reg(*r), lane, 0)?;
                (bits >> if *h == Half::Hi { 16 } else { 0 }) & 0xffff
            }
            Operand::Special(s) => match s {
                Special::Exec | Special::ExecLo => self.exec as u32,
                Special::Vcc | Special::VccLo => self.vcc as u32,
                Special::ExecHi => (self.exec >> 32) as u32,
                Special::VccHi => (self.vcc >> 32) as u32,
                Special::Null => 0,
                Special::Scc => u32::from(self.scc),
                Special::M0 => self.m0,
                Special::Ttmp(n) => self.t[usize::from(*n)],
                _ => return Err(format!("special source {s:?} unsupported")),
            },
            Operand::Inline(InlineConst::Integer(n)) => *n as i32 as u32,
            Operand::Inline(InlineConst::FloatBits(n)) | Operand::Literal(n) => *n,
            Operand::Inline(InlineConst::InvTwoPi) => 0x3e22f983,
            Operand::Imm(i) => match i {
                ImmField::Sopp(n) | ImmField::Sopk(n) => *n as i32 as u32,
                ImmField::Unsigned(n) => *n,
                ImmField::DsOffset(n) => u32::from(*n),
                ImmField::DsOffset0(n) | ImmField::DsOffset1(n) => u32::from(*n),
                ImmField::VmemOffset(n)
                | ImmField::SmemOffset(n)
                | ImmField::SmemDisplacement(n) => *n as u32,
            },
            _ => return Err(format!("not a value operand {op:?}")),
        })
    }
    fn read64(&self, op: &Operand, lane: usize) -> Result<u64> {
        Ok(u64::from(self.read(op, lane, 0)?) | u64::from(self.read(op, lane, 1)?) << 32)
    }
    /// 64-bit source of a 64-bit vector/scalar operation. `Reg` is a register pair, a wave32 `vcc`/`exec` pair has a
    /// zero upper half, inline integers sign-extend, and a 32-bit literal is zero-extended (`Unsigned`), sign-extended
    /// (`Signed`) or the most-significant half of a double (`Float`), as the RDNA ISA literal rules state.
    fn read_wide(&self, op: &Operand, lane: usize, kind: Wide) -> Result<u64> {
        Ok(match op {
            Operand::Reg(r) if r.len >= 2 => self.read64(op, lane)?,
            Operand::Reg(_) => {
                return Err(format!("64-bit operand needs a register pair, got {op}"))
            }
            Operand::Special(Special::SrcSharedBase) => u64::from(SHARED_APERTURE_HI) << 32,
            Operand::Special(Special::SrcPrivateBase) => u64::from(PRIVATE_APERTURE_HI) << 32,
            Operand::Special(Special::Vcc | Special::VccLo) => self.vcc & self.lane_mask(),
            Operand::Special(Special::Exec | Special::ExecLo) => self.exec & self.lane_mask(),
            Operand::Special(Special::Null) => 0,
            Operand::Inline(InlineConst::Integer(n)) => i64::from(*n) as u64,
            Operand::Inline(InlineConst::FloatBits(b)) if kind == Wide::Float => {
                f64::from(f32::from_bits(*b)).to_bits()
            }
            Operand::Inline(InlineConst::InvTwoPi) if kind == Wide::Float => 0x3fc4_5f30_6dc9_c882,
            Operand::Literal(n) => match kind {
                Wide::Unsigned => u64::from(*n),
                Wide::Signed => i64::from(*n as i32) as u64,
                Wide::Float => u64::from(*n) << 32,
            },
            _ => return Err(format!("unsupported 64-bit {kind:?} operand {op}")),
        })
    }
    fn put_wide(&mut self, op: &Operand, lane: usize, value: u64) -> Result<()> {
        match op {
            Operand::Reg(r) if r.len >= 2 => {
                self.put(op, lane, 0, value as u32)?;
                self.put(op, lane, 1, (value >> 32) as u32)
            }
            Operand::Special(Special::Exec | Special::ExecLo) => {
                self.exec = value & self.lane_mask();
                Ok(())
            }
            Operand::Special(Special::Vcc | Special::VccLo) => {
                self.vcc = value & self.lane_mask();
                Ok(())
            }
            Operand::Special(Special::Null) => Ok(()),
            _ => Err(format!(
                "64-bit destination needs a register pair, got {op}"
            )),
        }
    }
    /// Bits of `exec`/`vcc` that exist in this wavefront.
    fn lane_mask(&self) -> u64 {
        if self.lanes == 64 {
            u64::MAX
        } else {
            0xffff_ffff
        }
    }
    /// A lane-mask operand (`v_cmp` destination, `v_cndmask`/carry-in source, carry-out): one SGPR/`vcc_lo`/`exec_lo`
    /// in wave32, an SGPR pair or the full `vcc`/`exec` in wave64. The encoding names `vcc_lo`/`exec_lo` and `s_n`
    /// for both wave sizes, so the kernel's wave size (not the operand spelling) decides the width.
    fn read_mask(&self, op: &Operand) -> Result<u64> {
        let wide = self.lanes == 64;
        Ok(match op {
            Operand::Special(Special::Vcc | Special::VccLo | Special::VccHi) => {
                self.vcc & self.lane_mask()
            }
            Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi) => {
                self.exec & self.lane_mask()
            }
            Operand::Special(Special::Null) => 0,
            Operand::Reg(r) if r.kind != Kind::V => {
                if wide && r.len < 2 {
                    return Err(format!("wave64 lane mask needs a register pair, got {op}"));
                }
                let lo = u64::from(self.read(op, 0, 0)?);
                if wide {
                    lo | u64::from(self.read(op, 0, 1)?) << 32
                } else {
                    lo
                }
            }
            Operand::Inline(InlineConst::Integer(n)) => {
                if wide {
                    i64::from(*n) as u64
                } else {
                    u64::from(*n as i32 as u32)
                }
            }
            Operand::Literal(n) => u64::from(*n),
            _ => return Err(format!("not a lane-mask operand {op}")),
        })
    }
    fn put_mask(&mut self, op: &Operand, value: u64) -> Result<()> {
        let wide = self.lanes == 64;
        let value = value & self.lane_mask();
        match op {
            Operand::Special(Special::Vcc | Special::VccLo) => self.vcc = value,
            Operand::Special(Special::Exec | Special::ExecLo) => self.exec = value,
            Operand::Special(Special::Null) => (),
            Operand::Reg(r) if r.kind != Kind::V => {
                if wide && r.len < 2 {
                    return Err(format!("wave64 lane mask needs a register pair, got {op}"));
                }
                self.put(op, 0, 0, value as u32)?;
                if wide {
                    self.put(op, 0, 1, (value >> 32) as u32)?;
                }
            }
            _ => return Err(format!("not a lane-mask destination {op}")),
        }
        Ok(())
    }
    fn put(&mut self, op: &Operand, lane: usize, word: usize, value: u32) -> Result<()> {
        match op {
            Operand::Reg(r) => {
                let n = usize::from(r.base) + word;
                match r.kind {
                    Kind::S => self.s[n] = value,
                    Kind::V => self.v[n][lane] = value,
                    Kind::Ttmp => self.t[n] = value,
                }
            }
            Operand::Half(r, h) => {
                let old = self.read(&Operand::Reg(*r), lane, 0)?;
                let shift = if *h == Half::Hi { 16 } else { 0 };
                self.put(
                    &Operand::Reg(*r),
                    lane,
                    0,
                    (old & !(0xffff << shift)) | ((value & 0xffff) << shift),
                )?;
            }
            Operand::Special(s) => match s {
                Special::Exec | Special::ExecLo => {
                    self.exec = (self.exec & !0xffff_ffff) | u64::from(value)
                }
                Special::Vcc | Special::VccLo => {
                    self.vcc = (self.vcc & !0xffff_ffff) | u64::from(value)
                }
                Special::ExecHi if self.lanes == 64 => {
                    self.exec = (self.exec & 0xffff_ffff) | u64::from(value) << 32
                }
                Special::VccHi if self.lanes == 64 => {
                    self.vcc = (self.vcc & 0xffff_ffff) | u64::from(value) << 32
                }
                Special::ExecHi | Special::VccHi => (),
                Special::Scc => self.scc = value != 0,
                Special::M0 => self.m0 = value,
                Special::Null => (),
                _ => return Err(format!("special destination {s:?} unsupported")),
            },
            _ => return Err("non-register destination".into()),
        }
        Ok(())
    }
}

/// Name-keyed semantic side table, deliberately separate from encoding tables.
pub fn semantic_class(name: &str) -> Option<&'static str> {
    static TABLE: std::sync::LazyLock<std::collections::HashMap<&'static str, &'static str>> =
        std::sync::LazyLock::new(|| {
            include_str!("../../isa/semantics.tbl")
                .lines()
                .filter(|l| !l.starts_with('#'))
                .filter_map(|l| l.split_once('\t'))
                .collect()
        });
    TABLE.get(name).copied()
}
#[derive(Clone, Copy)]
enum Class {
    Wait,
    Control,
    Memory,
    Alu,
}
struct Step<'a> {
    inst: &'a Inst,
    name: &'static str,
    class: Class,
}
/// Decode/name/semantic lookup is compiled once, not repeated inside hot loops
/// or for each workgroup of a real-shape replay.
pub struct Emulator<'a> {
    kernel: &'a Kernel,
    arch: Arch,
    steps: Vec<Step<'a>>,
    nan_loose: bool,
    strict_lds: bool,
    /// The program's own target: instruction tables are looked up with this even when numerics run as gfx1151.
    decode_arch: Arch,
    hw: Option<Vec<hw::StepHw>>,
    /// Workgroups run so far (the gate model's schedule budget).
    runs: std::sync::atomic::AtomicU64,
}
impl<'a> Emulator<'a> {
    pub fn new(program: &'a Program, symbol: &str) -> Result<Self> {
        let kernel = program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == symbol)
            .ok_or("snapshot symbol absent in Program")?;
        let steps = kernel
            .body
            .layout
            .iter()
            .map(|id| {
                let inst = kernel.body.insts.get(*id).ok_or("missing instruction")?;
                let name = crate::isa::lookup(program.target.arch, inst.op, inst.form)
                    .ok_or("unknown decoded opcode")?
                    .name;
                let class = match semantic_class(name)
                    .ok_or_else(|| format!("no semantics for {name}"))?
                {
                    "wait" => Class::Wait,
                    "control" => Class::Control,
                    "memory" => Class::Memory,
                    "alu" => Class::Alu,
                    class => return Err(format!("unsupported semantics class {class} for {name}")),
                };
                Ok(Step { inst, name, class })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kernel,
            arch: program.target.arch,
            steps,
            nan_loose: false,
            strict_lds: false,
            decode_arch: program.target.arch,
            hw: None,
            runs: std::sync::atomic::AtomicU64::new(0),
        })
    }
    /// Track which LDS bytes the workgroup has written; a read of any other LDS byte (including by an LDS atomic) is an
    /// error. Off by default so existing differential tests keep their (zero-initialised LDS) behaviour.
    pub fn strict_lds(mut self) -> Self {
        self.strict_lds = true;
        self
    }
    /// The lab's exactness contract for the lane kernels: non-NaN words bit-equal, a NaN must meet a NaN (payload and
    /// sign are tallied, not claimed). With it, a NaN operand of `v_fma_mix_f32` in a layout the hardware sweeps do not
    /// cover (op_sel / half A or C source) widens exactly and takes the measured f32 fma NaN rule instead of failing
    /// closed. Whether a result is a NaN is exact under IEEE; which payload it carries is NOT claimed. Off by default.
    pub fn nan_meets_nan(mut self) -> Self {
        self.nan_loose = true;
        self
    }
    /// Execute a gfx1100 program with gfx1151's (RDNA3.5) numerics and WMMA model. The emulator has no hardware-
    /// qualified gfx1100 model (WMMA fragments, V_EXP/LOG/RCP/RSQ/SQRT), so gfx1100 programs are refused without this.
    /// Only valid where emulated arms are compared with each other or with an exact CPU reference (the pre-card safety
    /// gate), never as a claim about gfx1100 hardware ulps. Instruction decode still uses the gfx1100 tables.
    pub fn gfx1100_as_gfx1151(mut self) -> Self {
        if self.arch == Arch::Gfx1100 {
            self.arch = Arch::Gfx1151;
        }
        self
    }
    pub fn run_workgroup(&self, launch: &Launch, mem: &mut Memory) -> Result<()> {
        self.run_workgroup_counted(launch, mem).map(|_| ())
    }
    /// As `run_workgroup`, also naming the schedules run (the gate model runs several per workgroup).
    pub fn run_workgroup_report(&self, launch: &Launch, mem: &mut Memory) -> Result<RunReport> {
        if self.hw_enabled() {
            return self.run_hw(launch, mem);
        }
        Ok(RunReport {
            instructions: self.run_workgroup_counted(launch, mem)?,
            schedules: vec!["round-robin".into()],
        })
    }
    /// As `run_workgroup`, returning the number of instructions the workgroup's waves executed.
    pub fn run_workgroup_counted(&self, launch: &Launch, mem: &mut Memory) -> Result<u64> {
        if self.hw_enabled() {
            return self.run_hw(launch, mem).map(|r| r.instructions);
        }
        let kernel = self.kernel;
        let symbol = &kernel.symbol.0;
        let Abi::Hsa { descriptor: d, .. } = &kernel.abi else {
            return Err("HSA descriptor required".into());
        };
        let n = launch.block[0]
            .checked_mul(launch.block[1])
            .and_then(|n| n.checked_mul(launch.block[2]))
            .ok_or("block size overflow")? as usize;
        if n == 0 || n > 1024 {
            return Err("workgroup must contain 1..1024 threads".into());
        }
        let mut states = (0..n.div_ceil(kernel.wave.lanes()))
            .map(|w| {
                State::new(self.arch, kernel, launch, w).map(|mut s| {
                    s.nan_loose = self.nan_loose;
                    s
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut lds = vec![
            0u8;
            d.group_segment_fixed_size
                .checked_add(launch.dynamic_lds)
                .ok_or("LDS size overflow")? as usize
        ];
        let mut defined = self.strict_lds.then(|| WriteSet::new(lds.len()));
        let mut executed = 0u64;
        while states.iter().any(|s| !s.ended) {
            let mut progress = false;
            for (wave, st) in states.iter_mut().enumerate() {
                if st.ended || st.barrier.is_some() {
                    continue;
                }
                let step = self.steps.get(st.pc).ok_or("PC ran past kernel end")?;
                let inst = step.inst;
                let name = step.name;
                st.pc += 1;
                progress = true;
                executed += 1;
                if executed > 100_000_000 {
                    return Err("instruction limit exceeded (possible divergent loop)".into());
                }
                let result = match step.class {
                    Class::Wait => Ok(()),
                    Class::Control => control(name, inst, st, kernel),
                    Class::Memory => memory::execute_tracked(
                        name,
                        inst,
                        st,
                        mem,
                        &mut lds,
                        defined.as_mut(),
                        None,
                    ),
                    Class::Alu => alu::execute(self.arch, name, inst, st),
                };
                result.map_err(|e| {
                    format!(
                        "{symbol} wg={:?} inst={} {name}: {e}{}",
                        launch.workgroup,
                        st.pc.saturating_sub(1),
                        failure_bits(self.arch, wave, st, inst)
                    )
                })?;
            }
            // A gfx12 signal is an arrival, not a wait. A wave may continue
            // executing after its arrival while peers' waits already retire.
            if let Some(arrivals) = states.iter().filter(|s| !s.ended).map(|s| s.signals).min() {
                for st in &mut states {
                    if st.barrier.is_some_and(|phase| phase < arrivals) {
                        st.barrier = None;
                        progress = true;
                    }
                }
            }
            if !progress {
                return Err("workgroup barrier deadlock".into());
            }
        }
        Ok(executed)
    }
}
/// Err-path only: exact architectural bits of the failing wave state and every
/// operand (destinations included, so implicit accumulators are visible).
/// Reads are guarded so diagnostics can never mask the original error.
fn failure_bits(arch: Arch, wave: usize, st: &State, inst: &Inst) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _=write!(out," | arch={arch:?} wave={wave} exec={:#010x} vcc={:#010x} scc={} m0={:#010x} mods={:?} fields={:?}",st.exec,st.vcc,u8::from(st.scc),st.m0,inst.mods,inst.fields);
    match inst.literal {
        Some(value) => {
            let _ = write!(out, " literal={value:#010x}");
        }
        None => out.push_str(" literal=None"),
    }
    for (i, op) in inst.operands.iter().enumerate() {
        let _ = write!(out, " | op{i} {op} {op:?}:");
        if let Err(e) = op.validate() {
            let _ = write!(out, " invalid({e})");
            continue;
        }
        match op {
            Operand::Reg(r) => {
                for word in 0..usize::from(r.len) {
                    if r.kind == Kind::V {
                        let _ = write!(out, " w{word}=[");
                        for lane in 0..st.lanes {
                            if lane != 0 {
                                out.push(' ');
                            }
                            match st.read(op, lane, word) {
                                Ok(v) => {
                                    let _ = write!(out, "{v:#010x}");
                                }
                                Err(e) => {
                                    let _ = write!(out, "unreadable({e})");
                                }
                            }
                        }
                        out.push(']');
                    } else {
                        match st.read(op, 0, word) {
                            Ok(v) => {
                                let _ = write!(out, " w{word}={v:#010x}");
                            }
                            Err(e) => {
                                let _ = write!(out, " w{word}=unreadable({e})");
                            }
                        }
                    }
                }
            }
            Operand::Half(r, _) if r.kind == Kind::V => {
                out.push_str(" h=[");
                for lane in 0..st.lanes {
                    if lane != 0 {
                        out.push(' ');
                    }
                    match st.read(op, lane, 0) {
                        Ok(v) => {
                            let _ = write!(out, "{v:#06x}");
                        }
                        Err(e) => {
                            let _ = write!(out, "unreadable({e})");
                        }
                    }
                }
                out.push(']');
            }
            Operand::Special(Special::Ttmp(n)) if usize::from(*n) >= st.t.len() => {
                let _ = write!(out, " unreadable(ttmp{n} out of range)");
            }
            _ => match st.read(op, 0, 0) {
                Ok(v) => {
                    let _ = write!(out, " {v:#010x}");
                }
                Err(e) => {
                    let _ = write!(out, " unreadable({e})");
                }
            },
        }
    }
    out
}
fn control(name: &str, i: &Inst, s: &mut State, k: &Kernel) -> Result<()> {
    match name {
        "s_endpgm" => s.ended = true,
        "s_barrier" => {
            s.barrier = Some(s.signals);
            s.signals += 1;
        }
        "s_barrier_signal" => {
            if s.read(&i.operands[0], 0, 0)? != u32::MAX {
                return Err("named barrier signal is not modeled".into());
            }
            s.signals += 1;
        }
        "s_barrier_wait" => {
            // SOPP is a 16-bit field: decoded 0xffff is represented as signed -1.
            let id = match &i.operands[0] {
                Operand::Imm(ImmField::Sopp(value)) => *value as u16,
                op => u16::try_from(s.read(op, 0, 0)?)
                    .map_err(|_| "barrier ID exceeds its 16-bit field")?,
            };
            if id != u16::MAX {
                return Err("named barrier wait is not modeled".into());
            }
            if s.signals == 0 {
                return Err("barrier wait without signal".into());
            }
            s.barrier = Some(s.signals - 1);
        }
        "s_branch" | "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_execz"
        | "s_cbranch_execnz" | "s_cbranch_vccz" | "s_cbranch_vccnz" => {
            let take = match name {
                "s_branch" => true,
                "s_cbranch_scc0" => !s.scc,
                "s_cbranch_scc1" => s.scc,
                "s_cbranch_execz" => s.exec == 0,
                "s_cbranch_execnz" => s.exec != 0,
                "s_cbranch_vccz" => s.vcc == 0,
                _ => s.vcc != 0,
            };
            if take {
                let Operand::Label(id) = i.operands[0] else {
                    return Err("branch missing CFG label".into());
                };
                s.pc = k
                    .body
                    .blocks
                    .iter()
                    .find(|b| b.id == id)
                    .ok_or("branch block missing")?
                    .range
                    .0;
            }
        }
        _ => return Err(format!("control opcode {name} unsupported")),
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> State {
        State {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes: 32,
            nan_loose: false,
            exec: 0xffff_ffff,
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
        Operand::Reg(crate::reg::RegRef { kind, base, len })
    }
    fn v(base: u16) -> Operand {
        reg(Kind::V, base, 1)
    }
    fn s(base: u16) -> Operand {
        reg(Kind::S, base, 1)
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
    fn kernel(body: crate::Body) -> Kernel {
        use crate::{
            descriptor::*,
            inst::{KernelOrigin, SymbolId},
            metadata::{HsaKernelMetadata, KernelMeta},
        };
        Kernel {
            symbol: SymbolId("exchange".into()),
            wave: Wave::Wave32,
            body,
            abi: Abi::Hsa {
                descriptor: KernelDescriptor {
                    group_segment_fixed_size: 256,
                    private_segment_fixed_size: 0,
                    kernarg_size: 0,
                    kernel_code_entry_byte_offset: 0,
                    compute_pgm_rsrc1: Rsrc1(0xaf0000),
                    compute_pgm_rsrc2: Rsrc2(0x180),
                    compute_pgm_rsrc3: Rsrc3(0),
                    kernel_code_properties: KernelCodeProperties(0x400),
                    kernarg_preload: KernargPreload(0),
                    reserved: [0; 28],
                },
                metadata: HsaKernelMetadata {
                    raw_msgpack: vec![],
                    parsed: KernelMeta::default(),
                },
            },
            origin: KernelOrigin::Authored {
                builder_crate: "test".into(),
                version: "1".into(),
                git: "test".into(),
            },
        }
    }
    #[test]
    fn split_barrier_rendezvous_waits_for_delayed_wave() {
        use crate::{
            cfg::{Block, BlockId, Cond, Terminator},
            inst::{Setting, Target},
        };
        let mut body = crate::Body::default();
        let mut add = |name: &str, ops: Vec<Operand>| {
            let id = body.insts.insert(insn(name, ops));
            body.layout.push(id);
        };
        add("s_mov_b32", vec![s(4), Operand::Literal(0x10000)]);
        add("s_mov_b32", vec![s(5), Operand::Literal(0)]);
        add("s_mov_b32", vec![s(6), Operand::Literal(256)]);
        add("s_mov_b32", vec![s(7), Operand::Literal(0x3100_4000)]);
        add(
            "v_lshlrev_b32_e32",
            vec![v(1), Operand::Inline(InlineConst::Integer(2)), v(0)],
        );
        add("v_readfirstlane_b32", vec![s(0), v(0)]);
        add(
            "s_cmp_lt_u32",
            vec![s(0), Operand::Inline(InlineConst::Integer(32))],
        );
        add("s_cbranch_scc0", vec![Operand::Label(BlockId(3))]);
        add(
            "s_mov_b32",
            vec![s(8), Operand::Inline(InlineConst::Integer(16))],
        );
        add(
            "s_sub_co_i32",
            vec![s(8), s(8), Operand::Inline(InlineConst::Integer(1))],
        );
        add(
            "s_cmp_lg_u32",
            vec![s(8), Operand::Inline(InlineConst::Integer(0))],
        );
        add("s_cbranch_scc1", vec![Operand::Label(BlockId(2))]);
        add("ds_store_b32", vec![v(1), v(0)]);
        add("s_barrier_signal", vec![Operand::Imm(ImmField::Sopp(-1))]);
        add("s_barrier_wait", vec![Operand::Imm(ImmField::Sopp(-1))]);
        add("v_xor_b32_e32", vec![v(2), v(1), Operand::Literal(128)]);
        add("ds_load_b32", vec![v(3), v(2)]);
        add(
            "buffer_store_b32",
            vec![
                v(3),
                v(1),
                reg(Kind::S, 4, 4),
                Operand::Special(Special::Null),
                Operand::Vmem(crate::operand::VmemToken::Offen),
            ],
        );
        add("s_endpgm", vec![]);
        for (id, range, term, preds, succs) in [
            (
                0,
                (0, 8),
                Terminator::Branch {
                    cond: Cond::Scc0,
                    taken: BlockId(3),
                    fallthrough: BlockId(1),
                },
                vec![],
                vec![3, 1],
            ),
            (1, (8, 9), Terminator::FallThrough, vec![0], vec![2]),
            (
                2,
                (9, 12),
                Terminator::Branch {
                    cond: Cond::Scc1,
                    taken: BlockId(2),
                    fallthrough: BlockId(3),
                },
                vec![1, 2],
                vec![2, 3],
            ),
            (3, (12, 19), Terminator::EndPgm, vec![0, 2], vec![]),
        ] {
            body.blocks.push(Block {
                id: BlockId(id),
                range,
                term,
                preds: preds.into_iter().map(BlockId).collect(),
                succs: succs.into_iter().map(BlockId).collect(),
            });
        }
        let p = Program {
            target: Target {
                arch: Arch::Gfx1201,
                xnack: Setting::Any,
                sramecc: Setting::Any,
                abi_version: 4,
            },
            kernels: vec![kernel(body)],
            source: None,
        };
        let mut mem = Memory::default();
        mem.map("out".into(), 0x10000, vec![0xcd; 256]).unwrap();
        Emulator::new(&p, "exchange")
            .unwrap()
            .run_workgroup(
                &Launch {
                    kernarg_ptr: 0,
                    workgroup: [0, 0, 0],
                    block: [64, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap();
        let expected = (0..64u32)
            .flat_map(|n| (n ^ 32).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(mem.regions[0].bytes, expected);
        assert_eq!(mem.regions[0].written.count(), 256);
    }
    #[test]
    fn descriptor_selects_packed_ids_and_arch_workgroup_registers() {
        let mut k = kernel(crate::Body::default());
        let Abi::Hsa { descriptor: d, .. } = &mut k.abi else {
            unreachable!()
        };
        d.compute_pgm_rsrc2.0 = 0x180 | (1 << 11);
        let launch = Launch {
            kernarg_ptr: 0,
            workgroup: [7, 9, 0],
            block: [8, 8, 2],
            dynamic_lds: 0,
        };
        let a = State::new(Arch::Gfx1201, &k, &launch, 1).unwrap();
        assert_eq!(a.t[9], 7);
        assert_eq!(a.t[7], 9);
        assert_eq!(a.t[8], (1 << 30) | (1 << 25));
        assert_eq!(a.v[0][0], 4 << 10);
        let b = State::new(Arch::Gfx1151, &k, &launch, 3).unwrap();
        assert_eq!((b.s[0], b.s[1]), (7, 9));
        assert_eq!(b.v[0][0], 4 << 10);
        let Abi::Hsa { descriptor: d, .. } = &mut k.abi else {
            unreachable!()
        };
        d.compute_pgm_rsrc2.0 = 0x180 | (2 << 11);
        let c = State::new(Arch::Gfx1151, &k, &launch, 3).unwrap();
        assert_eq!(c.v[0][0], (1 << 20) | (4 << 10));
    }
    // RDNA4 ISA scalar unsigned carry differs from signed overflow.
    #[test]
    fn scalar_carry_propagates_and_mask_updates_scc() {
        let mut st = state();
        st.s[1] = u32::MAX;
        st.s[2] = 1;
        alu::execute(
            Arch::Gfx1201,
            "s_add_co_u32",
            &insn("s_add_co_u32", vec![s(3), s(1), s(2)]),
            &mut st,
        )
        .unwrap();
        assert_eq!(st.s[3], 0);
        assert!(st.scc);
        alu::execute(
            Arch::Gfx1201,
            "s_add_co_ci_u32",
            &insn("s_add_co_ci_u32", vec![s(4), s(3), s(3)]),
            &mut st,
        )
        .unwrap();
        assert_eq!(st.s[4], 1);
        assert!(!st.scc);
        alu::execute(
            Arch::Gfx1201,
            "s_and_b32",
            &insn(
                "s_and_b32",
                vec![
                    Operand::Special(Special::ExecLo),
                    Operand::Special(Special::ExecLo),
                    s(4),
                ],
            ),
            &mut st,
        )
        .unwrap();
        assert_eq!(st.exec, 1);
        assert!(st.scc);
    }
    // A raw b64 access straddling NUM_RECORDS bounds is checked per DWORD.
    #[test]
    fn raw_buffer_zeroes_oob_dword_and_drops_store() {
        let mut st = state();
        st.exec = 1;
        st.s[4] = 0x1000;
        st.s[6] = 4;
        st.s[7] = 0x3100_4000;
        let args = vec![
            reg(Kind::V, 0, 2),
            v(2),
            reg(Kind::S, 4, 4),
            Operand::Special(Special::Null),
            Operand::Vmem(crate::operand::VmemToken::Offen),
        ];
        let mut mem = Memory::default();
        mem.map("input".into(), 0x1000, 0x12345678u32.to_le_bytes().to_vec())
            .unwrap();
        memory::execute(
            "buffer_load_b64",
            &insn("buffer_load_b64", args),
            &mut st,
            &mut mem,
            &mut [],
        )
        .unwrap();
        assert_eq!(st.v[0][0], 0x12345678);
        assert_eq!(st.v[1][0], 0);
        st.v[2][0] = 4;
        st.v[0][0] = 0xdeadbeef;
        let args = vec![
            v(0),
            v(2),
            reg(Kind::S, 4, 4),
            Operand::Special(Special::Null),
            Operand::Vmem(crate::operand::VmemToken::Offen),
        ];
        memory::execute(
            "buffer_store_b32",
            &insn("buffer_store_b32", args),
            &mut st,
            &mut mem,
            &mut [],
        )
        .unwrap();
        assert_eq!(mem.regions[0].bytes, 0x12345678u32.to_le_bytes());
        assert_eq!(mem.regions[0].written.count(), 0);
    }
    // Permlane reads the old source before any aliasing destination writes.
    #[test]
    fn permlane_swaps_halves_with_inactive_source_policy() {
        let mut st = state();
        for lane in 0..32 {
            st.v[0][lane] = lane as u32;
        }
        let i = insn(
            "v_permlanex16_b32",
            vec![
                v(0),
                v(0),
                Operand::Literal(0x76543210),
                Operand::Literal(0xfedcba98),
            ],
        );
        alu::execute(Arch::Gfx1201, "v_permlanex16_b32", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 16);
        assert_eq!(st.v[0][16], 0);
        st.exec = 1;
        st.v[0][0] = 123;
        let mut i = i;
        i.mods.op_sel = 0;
        alu::execute(Arch::Gfx1201, "v_permlanex16_b32", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 123);
        i.mods.op_sel = 2;
        alu::execute(Arch::Gfx1201, "v_permlanex16_b32", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
    }
    #[test]
    fn half_destination_preserves_high_bits_and_exec() {
        let mut st = state();
        st.exec = 1;
        st.v[0][0] = 0xdeadbeef;
        st.v[0][1] = 0xfeedface;
        st.v[1][0] = 1f32.to_bits();
        let dest = Operand::Half(
            crate::reg::RegRef {
                kind: Kind::V,
                base: 0,
                len: 1,
            },
            Half::Lo,
        );
        alu::execute(
            Arch::Gfx1201,
            "v_cvt_f16_f32_e32",
            &insn("v_cvt_f16_f32_e32", vec![dest, v(1)]),
            &mut st,
        )
        .unwrap();
        assert_eq!(st.v[0][0], 0xdead3c00);
        assert_eq!(st.v[0][1], 0xfeedface);
    }
    #[test]
    fn mapped_access_tracks_identical_stores_and_rejects_crossing() {
        let mut m = Memory::default();
        m.map("out".into(), 0x1000, vec![1, 2, 3, 4]).unwrap();
        m.write(0x1001, &[2, 9]).unwrap();
        assert_eq!(m.read(0x1000, 4).unwrap(), [1, 2, 9, 4]);
        assert_eq!(
            (0..4)
                .map(|n| m.regions[0].written.contains(n))
                .collect::<Vec<_>>(),
            [false, true, true, false]
        );
        assert!(m.read(0x1003, 2).is_err());
        assert!(m.map("alias".into(), 0x1002, vec![0; 4]).is_err());
    }
    /// One-block straight-line program ending in `s_endpgm`, symbol "exchange", 256 bytes of static LDS.
    fn straight(ops: Vec<(&str, Vec<Operand>)>) -> Program {
        use crate::{
            cfg::{Block, BlockId, Terminator},
            inst::{Setting, Target},
        };
        let mut body = crate::Body::default();
        let mut n = 0;
        for (name, o) in ops {
            let id = body.insts.insert(insn(name, o));
            body.layout.push(id);
            n += 1;
        }
        let id = body.insts.insert(insn("s_endpgm", vec![]));
        body.layout.push(id);
        body.blocks.push(Block {
            id: BlockId(0),
            range: (0, n + 1),
            term: Terminator::EndPgm,
            preds: Default::default(),
            succs: Default::default(),
        });
        Program {
            target: Target {
                arch: Arch::Gfx1201,
                xnack: Setting::Any,
                sramecc: Setting::Any,
                abi_version: 4,
            },
            kernels: vec![kernel(body)],
            source: None,
        }
    }
    fn run_straight(p: &Program, strict: bool) -> Result<u64> {
        let emu = Emulator::new(p, "exchange").unwrap();
        let emu = if strict { emu.strict_lds() } else { emu };
        emu.run_workgroup_counted(
            &Launch {
                kernarg_ptr: 0,
                workgroup: [0, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut Memory::default(),
        )
    }
    #[test]
    fn strict_lds_rejects_read_before_write_only_when_enabled() {
        let p = straight(vec![("ds_load_b32", vec![v(3), v(1)])]);
        assert_eq!(run_straight(&p, false).unwrap(), 2);
        let e = run_straight(&p, true).unwrap_err();
        assert!(e.contains("never written in this workgroup"), "{e}");
        assert!(e.contains("ds_load_b32"), "{e}");
        assert!(e.contains("inst=0"), "{e}");
    }
    #[test]
    fn strict_lds_accepts_read_after_write_and_rejects_partial_cover() {
        // v1 == 0 in every lane: all lanes store dword 0, then load it back.
        let ok = straight(vec![
            ("ds_store_b32", vec![v(1), v(0)]),
            ("ds_load_b32", vec![v(3), v(1)]),
        ]);
        assert_eq!(run_straight(&ok, true).unwrap(), 3);
        // A b64 load over a dword that was only half written reads an unwritten byte 4.
        let partial = straight(vec![
            ("ds_store_b32", vec![v(1), v(0)]),
            ("ds_load_b64", vec![reg(Kind::V, 4, 2), v(1)]),
        ]);
        assert_eq!(run_straight(&partial, false).unwrap(), 3);
        let e = run_straight(&partial, true).unwrap_err();
        assert!(e.contains("byte 0x4 never written"), "{e}");
    }
    #[test]
    fn strict_lds_still_reports_out_of_bounds() {
        // LDS is 256 bytes; offset 0x100 is past the end.
        let mut o = vec![v(3), v(1)];
        o.push(Operand::Imm(ImmField::DsOffset(0x100)));
        let p = straight(vec![("ds_load_b32", o)]);
        for strict in [false, true] {
            let e = run_straight(&p, strict).unwrap_err();
            assert!(e.contains("LDS read out of bounds"), "{e}");
        }
    }
    #[test]
    fn map_guarded_leaves_unmapped_gaps_around_exact_size_regions() {
        let mut m = Memory::default();
        let a = m.map_guarded("a".into(), vec![1; 64], 0x10000).unwrap();
        let b = m.map_guarded("b".into(), vec![2; 4096], 0x10000).unwrap();
        assert!(a >= 0x10000 && a % 0x1000 == 0 && b % 0x1000 == 0);
        assert!(b >= a + 64 + 0x10000);
        assert!(m.read(a, 64).is_ok() && m.read(b, 4096).is_ok());
        // one byte past the end, one byte before the start, and the whole gap are unmapped
        assert!(m.read(a + 64, 1).is_err());
        assert!(m.read(a - 1, 1).is_err());
        assert!(m.read(b - 1, 1).is_err());
        assert!(m.read(b + 4096, 4).is_err());
        assert!(m.read(a + 60, 8).is_err());
        let w = WriteSet::new(10);
        assert_eq!(w.first_unwritten(2, 3), Some(2));
    }
    // ---- gate model (hw.rs): in-flight memory, LDS races, schedules ----
    fn winsn(name: &str, waits: &[(crate::wait::Counter, u8)]) -> Inst {
        let row = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
        let mut w = crate::wait::WaitImm::default();
        for (c, n) in waits {
            w.per_counter[*c as usize] = Some(*n);
        }
        Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            row.form,
            Default::default(),
            [Operand::Imm(ImmField::Sopp(0))].into_iter().collect(),
            crate::operand::Modifiers {
                wait: Some(w),
                ..Default::default()
            },
            None,
            Default::default(),
        )
        .unwrap()
    }
    fn prog(insts: Vec<Inst>) -> Program {
        use crate::{
            cfg::{Block, BlockId, Terminator},
            inst::{Setting, Target},
        };
        let mut body = crate::Body::default();
        let n = insts.len();
        for i in insts {
            let id = body.insts.insert(i);
            body.layout.push(id);
        }
        let id = body.insts.insert(insn("s_endpgm", vec![]));
        body.layout.push(id);
        body.blocks.push(Block {
            id: BlockId(0),
            range: (0, n + 1),
            term: Terminator::EndPgm,
            preds: Default::default(),
            succs: Default::default(),
        });
        Program {
            target: Target {
                arch: Arch::Gfx1201,
                xnack: Setting::Any,
                sramecc: Setting::Any,
                abi_version: 4,
            },
            kernels: vec![kernel(body)],
            source: None,
        }
    }
    /// `s4..s7` = a raw SRD over the mapped "buf" region (base 0x10000, 256 bytes), `v1` = 4 * tid.
    fn srd_prologue() -> Vec<Inst> {
        vec![
            insn("s_mov_b32", vec![s(4), Operand::Literal(0x10000)]),
            insn("s_mov_b32", vec![s(5), Operand::Literal(0)]),
            insn("s_mov_b32", vec![s(6), Operand::Literal(256)]),
            insn("s_mov_b32", vec![s(7), Operand::Literal(0x3100_4000)]),
            insn(
                "v_lshlrev_b32_e32",
                vec![v(1), Operand::Inline(InlineConst::Integer(2)), v(0)],
            ),
        ]
    }
    fn buf_op(name: &str, data: u16, addr: u16) -> Inst {
        insn(
            name,
            vec![
                v(data),
                v(addr),
                reg(Kind::S, 4, 4),
                Operand::Special(Special::Null),
                Operand::Vmem(crate::operand::VmemToken::Offen),
            ],
        )
    }
    fn run_gate(p: &Program, block: u32, hw: bool) -> (Result<RunReport>, Memory) {
        let mut mem = Memory::default();
        mem.map("buf".into(), 0x10000, (0..=255u8).collect())
            .unwrap();
        let emu = Emulator::new(p, "exchange").unwrap();
        let emu = if hw { emu.hw_model() } else { emu };
        let r = emu.run_workgroup_report(
            &Launch {
                kernarg_ptr: 0,
                workgroup: [0, 0, 0],
                block: [block, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        );
        (r, mem)
    }
    #[test]
    fn in_flight_load_destination_is_only_readable_after_its_wait() {
        use crate::wait::Counter;
        let mut body = srd_prologue();
        body.push(buf_op("buffer_load_b32", 3, 1));
        body.push(insn("v_add_f32_e32", vec![v(4), v(3), v(3)]));
        let p = prog(body.clone());
        assert!(
            run_gate(&p, 32, false).0.is_ok(),
            "the legacy model retires loads at issue"
        );
        let e = run_gate(&p, 32, true).0.unwrap_err();
        assert!(
            e.contains("[in-flight]") && e.contains("reads v3") && e.contains("buffer_load_b32"),
            "{e}"
        );
        // wait first: fine
        let mut ok = srd_prologue();
        ok.push(buf_op("buffer_load_b32", 3, 1));
        ok.push(winsn("s_wait_loadcnt", &[(Counter::Load, 0)]));
        ok.push(insn("v_add_f32_e32", vec![v(4), v(3), v(3)]));
        assert!(run_gate(&prog(ok), 32, true).0.is_ok());
    }
    #[test]
    fn partial_wait_retires_only_the_older_load() {
        use crate::wait::Counter;
        let mk = |tail: Vec<Inst>| {
            let mut b = srd_prologue();
            b.push(buf_op("buffer_load_b32", 3, 1));
            b.push(buf_op("buffer_load_b32", 5, 1));
            b.push(winsn("s_wait_loadcnt", &[(Counter::Load, 1)]));
            b.extend(tail);
            prog(b)
        };
        let older = mk(vec![insn("v_add_f32_e32", vec![v(4), v(3), v(3)])]);
        assert!(run_gate(&older, 32, true).0.is_ok());
        let younger = mk(vec![insn("v_add_f32_e32", vec![v(4), v(5), v(5)])]);
        let e = run_gate(&younger, 32, true).0.unwrap_err();
        assert!(e.contains("reads v5") && e.contains("loadcnt"), "{e}");
        // writing the in-flight destination (WAW against an ALU write) is an error too
        let waw = mk(vec![insn("v_mov_b32_e32", vec![v(5), v(0)])]);
        assert!(run_gate(&waw, 32, true)
            .0
            .unwrap_err()
            .contains("writes v5"));
    }
    #[test]
    fn gfx12_store_data_is_locked_until_dscnt_retires() {
        use crate::wait::Counter;
        let ds = |wait: bool| {
            let mut b = srd_prologue();
            b.push(insn("ds_store_b32", vec![v(1), v(0)]));
            if wait {
                b.push(winsn("s_wait_dscnt", &[(Counter::Ds, 0)]));
            }
            b.push(insn("v_mov_b32_e32", vec![v(0), v(1)]));
            prog(b)
        };
        assert!(
            run_gate(&ds(false), 32, false).0.is_ok(),
            "the legacy model never overwrites late"
        );
        let e = run_gate(&ds(false), 32, true).0.unwrap_err();
        assert!(
            e.contains("[in-flight]") && e.contains("store source") && e.contains("ds_store_b32"),
            "{e}"
        );
        assert!(run_gate(&ds(true), 32, true).0.is_ok());
    }
    /// Same pattern as `gfx12_store_data_is_locked_until_dscnt_retires`, but the object comes from stock hipcc: LLVM
    /// leaves `ds_store_b128 v39, v[17:20]` ... `v_mov_b32 v17, 0` unwaited in shipped, card-proven gfx1201 objects
    /// (gated_delta_net_q8_fast, gdn_chunk_scan, ...), so the gate model reads store data at issue for them; the
    /// strict variant (and PM-builder objects, the test above) still flag it.
    #[test]
    fn hipcc_objects_read_ds_store_data_at_issue_on_gfx1201() {
        let hipcc = |mut p: Program| {
            p.kernels[0].origin = crate::inst::KernelOrigin::Frontend {
                kind: crate::inst::Frontend::Hipcc,
                object_sha256: [0; 32],
                entry_va: 0,
                size: 0,
            };
            p
        };
        let ds = |wait: bool| {
            let mut b = srd_prologue();
            b.push(insn("ds_store_b32", vec![v(1), v(0)]));
            if wait {
                b.push(winsn("s_wait_dscnt", &[(crate::wait::Counter::Ds, 0)]));
            }
            b.push(insn("v_mov_b32_e32", vec![v(0), v(1)]));
            prog(b)
        };
        let run = |p: &Program, strict: bool| {
            let mut mem = Memory::default();
            mem.map("buf".into(), 0x10000, (0..=255u8).collect())
                .unwrap();
            let emu = Emulator::new(p, "exchange").unwrap();
            let emu = if strict {
                emu.hw_model_strict_store_sources()
            } else {
                emu.hw_model()
            };
            emu.run_workgroup_report(
                &Launch {
                    kernarg_ptr: 0,
                    workgroup: [0, 0, 0],
                    block: [32, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
        };
        assert!(
            run(&hipcc(ds(false)), false).is_ok(),
            "hipcc: no store-source lock"
        );
        assert!(
            run(&ds(false), false).unwrap_err().contains("store source"),
            "builder object keeps it"
        );
        assert!(
            run(&hipcc(ds(false)), true)
                .unwrap_err()
                .contains("store source"),
            "strict variant keeps it"
        );
        assert!(run(&hipcc(ds(true)), true).is_ok());
    }
    /// gfx12 combined waits (`s_wait_storecnt_dscnt`, `s_wait_loadcnt_dscnt`; simm16 = {cnt[13:8], dscnt[5:0]}) retire
    /// both counters in the gate model.
    #[test]
    fn combined_storecnt_dscnt_wait_retires_ds_and_stores() {
        use crate::wait::Counter;
        for name in ["s_wait_storecnt_dscnt", "s_wait_loadcnt_dscnt"] {
            let counter = if name.contains("store") {
                Counter::Store
            } else {
                Counter::Load
            };
            let mk = |wait: bool| {
                let mut b = srd_prologue();
                b.push(insn("ds_store_b32", vec![v(1), v(1)]));
                b.push(insn("ds_load_b32", vec![v(3), v(1)]));
                if wait {
                    b.push(winsn(name, &[(counter, 0), (Counter::Ds, 0)]));
                }
                b.push(insn("v_add_f32_e32", vec![v(4), v(3), v(3)]));
                prog(b)
            };
            let e = run_gate(&mk(false), 32, true).0.unwrap_err();
            assert!(
                e.contains("[in-flight]") && e.contains("reads v3"),
                "{name}: {e}"
            );
            assert!(run_gate(&mk(true), 32, true).0.is_ok(), "{name}");
        }
    }
    /// Every thread stores `tid` at `4*tid`, then (optionally after a wait and a barrier) loads the slot of the thread 32 away.
    fn exchange(wait_before_barrier: bool, barrier: bool) -> Program {
        use crate::wait::Counter;
        let mut b = srd_prologue();
        b.push(insn("ds_store_b32", vec![v(1), v(0)]));
        if wait_before_barrier {
            b.push(winsn("s_wait_dscnt", &[(Counter::Ds, 0)]));
        }
        if barrier {
            b.push(insn(
                "s_barrier_signal",
                vec![Operand::Imm(ImmField::Sopp(-1))],
            ));
            b.push(insn(
                "s_barrier_wait",
                vec![Operand::Imm(ImmField::Sopp(-1))],
            ));
        }
        b.push(insn(
            "v_xor_b32_e32",
            vec![v(2), v(1), Operand::Literal(128)],
        ));
        b.push(insn("ds_load_b32", vec![v(3), v(2)]));
        b.push(winsn("s_wait_dscnt", &[(Counter::Ds, 0)]));
        b.push(buf_op("buffer_store_b32", 3, 1));
        prog(b)
    }
    #[test]
    fn lds_exchange_needs_a_waited_barrier() {
        let racy = exchange(true, false);
        assert!(
            run_gate(&racy, 64, false).0.is_ok(),
            "the legacy model cannot see the race"
        );
        let e = run_gate(&racy, 64, true).0.unwrap_err();
        assert!(
            e.contains("[race]")
                && e.contains("read-after-write")
                && e.contains("same barrier epoch"),
            "{e}"
        );
        // barrier without waiting for the store: the store is still in flight at the barrier
        let e = run_gate(&exchange(false, true), 64, true).0.unwrap_err();
        assert!(e.contains("[race]") && e.contains("still in flight"), "{e}");
        // wait + barrier: ordered, and every schedule agrees
        let (r, mem) = run_gate(&exchange(true, true), 64, true);
        let rep = r.unwrap();
        // round-robin, run-ahead of each of the two waves, two seeded-random orders
        assert_eq!(rep.schedules.len(), 5, "{:?}", rep.schedules);
        let expect: Vec<u8> = (0..64u32).flat_map(|n| (n ^ 32).to_le_bytes()).collect();
        assert_eq!(&mem.regions[0].bytes[..256], &expect[..]);
        assert_eq!(mem.regions[0].written.count(), 256);
    }
    #[test]
    fn schedules_must_agree_on_global_stores() {
        // Both waves store their lane 31 value at byte 0: the last writer decides, which a different schedule flips.
        let mut b = srd_prologue();
        b.push(insn(
            "v_mov_b32_e32",
            vec![v(5), Operand::Inline(InlineConst::Integer(0))],
        ));
        b.push(buf_op("buffer_store_b32", 0, 5));
        let p = prog(b);
        assert!(run_gate(&p, 64, false).0.is_ok());
        let e = run_gate(&p, 64, true).0.unwrap_err();
        assert!(
            e.contains("[schedule]") && e.contains("stores differ"),
            "{e}"
        );
        // one wave has one schedule and passes
        assert_eq!(run_gate(&p, 32, true).0.unwrap().schedules, ["round-robin"]);
    }
    #[test]
    fn unwritten_bytes_of_a_strict_region_are_an_uninit_read() {
        let mut b = srd_prologue();
        b.push(buf_op("buffer_load_b32", 3, 1));
        let p = prog(b);
        let mut mem = Memory::default();
        mem.map("buf".into(), 0x10000, vec![0; 256]).unwrap();
        mem.set_strict_read("buf").unwrap();
        let e = Emulator::new(&p, "exchange")
            .unwrap()
            .hw_model()
            .run_workgroup(
                &Launch {
                    kernarg_ptr: 0,
                    workgroup: [0, 0, 0],
                    block: [32, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_err();
        assert!(e.contains("[uninit]"), "{e}");
    }
}
