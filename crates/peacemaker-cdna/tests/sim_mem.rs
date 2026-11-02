// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `sim::mem` acceptance tests: SMEM, DS, GLOBAL and raw MUBUF semantics and
//! the wait-counter discipline (CDNA3 ISA §4.4 p.19-20, §8-9, §11).
//!
//! Every instruction is built explicitly from its format struct (no encoder,
//! decoder, printer or footprint function is used), and `sim::mem::{issue,
//! wait, drain}` are driven directly; only the barrier tests go through
//! `sim::exec_one`.
//!
//! Counter policy under test: VM_CNT retires in issue order (loads, stores
//! and atomics mixed); LGKM_CNT partial waits retire DS operations only (in
//! issue order) and leave every SMEM operation pending until the threshold is
//! zero, where DS retires first and SMEM then retires in reverse issue
//! order; a partial LGKM wait that DS cannot satisfy is `SimError::Wait`;
//! counter overflow retires the oldest operation of that counter (any kind)
//! until the new one fits; SMEM weighs 1 (one dword) or 2 (wider).
//!
//! The MUBUF bounds rules follow the parent contract: raw buffers check each
//! component's start offset (`inst_offset + vgpr_offset`, soffset excluded)
//! against `num_records` bytes, structured (`idxen`) buffers check the index
//! against `num_records`, out-of-range loads return zero and stores/atomics
//! are dropped; descriptor swizzle and `add_tid` are unsupported.
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::enc::mubuf::{Mubuf, MubufOp};
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::enc::sop::{Sopp, SoppOp};
use peacemaker_cdna::inst::{Inst, RegFile, RegSpan};
use peacemaker_cdna::sim::{
    exec_one, mem as simmem, Config, Counter, Flow, Memory, SimError, SimResult, Ty, Wave,
};
use peacemaker_cdna::{Arch, Src, WAVE};

// ----------------------------------------------------------------------------
// Fixtures

/// Base of the main global test region (above 4 GiB: exercises 64-bit math).
const G: u64 = 0x1_0000_0000;
const SENT: u32 = 0x5151_5151;
const LDS_BYTES: usize = 65536;
/// A typical gfx9 raw-buffer descriptor dword 3 (no swizzle, no add_tid).
const W3: u32 = 0x0002_7000;

/// Deterministic byte pattern with plenty of bytes above 0x7f.
fn pat(i: usize) -> u8 {
    i.wrapping_mul(37).wrapping_add(11) as u8
}
fn pat_u16(i: usize) -> u16 {
    u16::from_le_bytes([pat(i), pat(i + 1)])
}
fn pat_u32(i: usize) -> u32 {
    u32::from_le_bytes([pat(i), pat(i + 1), pat(i + 2), pat(i + 3)])
}
/// The k-th data dword of `lane`.
fn val(lane: usize, k: usize) -> u32 {
    0x9e37_79b9u32.wrapping_mul((lane * 8 + k + 1) as u32)
}
/// The first `n` little-endian bytes of `lane`'s data dwords.
fn st_bytes(lane: usize, n: usize) -> Vec<u8> {
    (0..4)
        .flat_map(|k| val(lane, k).to_le_bytes())
        .take(n)
        .collect()
}

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn s(n: u8) -> Src {
    Src::Sgpr(n)
}

fn is_wait<T>(r: &SimResult<T>) -> bool {
    matches!(r, Err(SimError::Wait(_)))
}
fn is_fault<T>(r: &SimResult<T>) -> bool {
    matches!(r, Err(SimError::Fault(_)))
}
fn is_unsupported<T>(r: &SimResult<T>) -> bool {
    matches!(r, Err(SimError::Unsupported(_)))
}

struct Rig {
    w: Wave,
    lds: Vec<u8>,
    m: Memory,
}

impl Rig {
    fn new() -> Self {
        Self::with_lds(LDS_BYTES)
    }
    /// LDS holds [`pat`] bytes.
    fn with_lds(n: usize) -> Self {
        Self {
            w: Wave::new(Arch::Gfx942, 0, 0),
            lds: (0..n).map(pat).collect(),
            m: Memory::default(),
        }
    }
    /// A rig with the main global region `"g"` at [`G`] holding [`pat`] bytes.
    fn global(len: usize) -> Self {
        let mut r = Self::new();
        r.map_pat("g", G, len);
        r
    }
    fn map_pat(&mut self, name: &str, base: u64, len: usize) {
        self.m.map(name, base, (0..len).map(pat).collect()).unwrap();
    }
    fn map_fill(&mut self, name: &str, base: u64, len: usize, b: u8) {
        self.m.map(name, base, vec![b; len]).unwrap();
    }
    fn bytes(&self, name: &str) -> &[u8] {
        &self.m.region(name).unwrap().bytes
    }

    fn issue(&mut self, i: &Inst) -> SimResult<Flow> {
        simmem::issue(&mut self.w, i, &mut self.lds, &mut self.m)
    }
    fn ok(&mut self, i: &Inst) {
        assert_eq!(self.issue(i), Ok(Flow::Next), "{i:?}");
    }
    fn wait(&mut self, vm: u8, lgkm: u8) -> SimResult<()> {
        simmem::wait(&mut self.w, &mut self.lds, &mut self.m, vm, lgkm, 7)
    }
    fn drain(&mut self) -> SimResult<()> {
        simmem::drain(&mut self.w, &mut self.lds, &mut self.m)
    }
    /// Issue and drain; the first error from either step.
    fn try_run(&mut self, i: &Inst) -> SimResult<()> {
        self.issue(i)?;
        self.drain()
    }
    /// Issue, drain, expect success.
    fn run(&mut self, i: &Inst) {
        self.ok(i);
        self.drain().unwrap();
    }

    fn set_v(&mut self, reg: usize, f: impl Fn(usize) -> u32) {
        for l in 0..WAVE {
            self.w.v[reg][l] = f(l)
        }
    }
    fn set_v64(&mut self, reg: usize, f: impl Fn(usize) -> u64) {
        for l in 0..WAVE {
            let x = f(l);
            self.w.v[reg][l] = x as u32;
            self.w.v[reg + 1][l] = (x >> 32) as u32
        }
    }
    fn fill_v(&mut self, reg: usize, n: usize, x: u32) {
        for k in 0..n {
            self.set_v(reg + k, |_| x)
        }
    }
    fn set_s64(&mut self, reg: usize, x: u64) {
        self.w.s[reg] = x as u32;
        self.w.s[reg + 1] = (x >> 32) as u32
    }
    /// A buffer resource in `s[sreg..sreg+4]`.
    fn set_desc(&mut self, sreg: usize, base: u64, stride: u32, nrec: u32, w3: u32) {
        self.w.s[sreg] = base as u32;
        self.w.s[sreg + 1] = ((base >> 32) as u32 & 0xffff) | (stride << 16);
        self.w.s[sreg + 2] = nrec;
        self.w.s[sreg + 3] = w3;
    }

    fn busy_v(&self, n: usize) -> bool {
        self.w
            .check_ready(RegSpan::new(RegFile::Vgpr, n as u8, 1), "probe")
            .is_err()
    }
    fn busy_s(&self, n: usize) -> bool {
        self.w
            .check_ready(RegSpan::new(RegFile::Sgpr, n as u8, 1), "probe")
            .is_err()
    }
    /// Units outstanding on `c` (sum of pending weights).
    fn units(&self, c: Counter) -> u32 {
        self.w
            .pending
            .iter()
            .filter(|p| p.counter == c)
            .map(|p| u32::from(p.weight))
            .sum()
    }
    fn word(&self, addr: u64) -> u32 {
        let b = self.m.read(addr, 4).unwrap();
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn put(&mut self, addr: u64, x: u32) {
        self.m.write(addr, &x.to_le_bytes()).unwrap()
    }
    fn lds_word(&self, a: usize) -> u32 {
        u32::from_le_bytes([
            self.lds[a],
            self.lds[a + 1],
            self.lds[a + 2],
            self.lds[a + 3],
        ])
    }
    fn lds_put(&mut self, a: usize, x: u32) {
        self.lds[a..a + 4].copy_from_slice(&x.to_le_bytes())
    }
}

// ----------------------------------------------------------------------------
// Explicit instruction constructors

#[allow(clippy::too_many_arguments)]
fn ds(
    op: DsOp,
    vdst: Option<u8>,
    addr: u8,
    d0: Option<u8>,
    d1: Option<u8>,
    off0: u8,
    off1: u8,
) -> Inst {
    Inst::Ds(Ds {
        op,
        vdst: vdst.map(v),
        addr: Some(v(addr)),
        data0: d0.map(v),
        data1: d1.map(v),
        offset0: off0,
        offset1: off1,
        gds: false,
    })
}
/// Single-address DS with the 16-bit byte `offset` (`offset1:offset0`).
fn ds_off(op: DsOp, vdst: Option<u8>, addr: u8, d0: Option<u8>, offset: u16) -> Inst {
    ds(op, vdst, addr, d0, None, offset as u8, (offset >> 8) as u8)
}

#[allow(clippy::too_many_arguments)]
fn flat(
    seg: FlatSeg,
    op: FlatOp,
    vdst: Option<u8>,
    addr: Option<u8>,
    data: Option<u8>,
    saddr: Option<u8>,
    offset: i16,
    sc0: bool,
) -> Inst {
    Inst::Flat(Flat {
        op,
        seg,
        vdst: vdst.map(v),
        addr: addr.map(v),
        data: data.map(v),
        saddr: saddr.map(s),
        offset,
        sc0,
        sc1: false,
        nt: false,
    })
}
/// `global_*` with a 64-bit VGPR address pair (`off`).
fn gl(op: FlatOp, vdst: Option<u8>, addr: u8, data: Option<u8>, offset: i16, sc0: bool) -> Inst {
    flat(
        FlatSeg::Global,
        op,
        vdst,
        Some(addr),
        data,
        None,
        offset,
        sc0,
    )
}
/// `global_*` with an SGPR-pair base `saddr` and a 32-bit VGPR offset.
fn gls(
    op: FlatOp,
    vdst: Option<u8>,
    voff: u8,
    data: Option<u8>,
    saddr: u8,
    offset: i16,
    sc0: bool,
) -> Inst {
    flat(
        FlatSeg::Global,
        op,
        vdst,
        Some(voff),
        data,
        Some(saddr),
        offset,
        sc0,
    )
}

#[allow(clippy::too_many_arguments)]
fn mb(
    op: MubufOp,
    vdata: u8,
    vaddr: Option<u8>,
    srsrc: u8,
    soffset: Src,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> Inst {
    Inst::Mubuf(Mubuf {
        op,
        vdata: v(vdata),
        vaddr: vaddr.map(v),
        srsrc: s(srsrc),
        soffset,
        offset,
        offen,
        idxen,
        sc0: false,
        sc1: false,
        nt: false,
    })
}

fn sm(op: SmemOp, sdata: Option<u8>, sbase: Option<u8>, offset: SmemOffset) -> Inst {
    Inst::Smem(Smem {
        op,
        sdata: sdata.map(s),
        sbase: sbase.map(s),
        offset,
        glc: false,
    })
}

fn barrier() -> Inst {
    Inst::Sopp(Sopp {
        op: SoppOp::Barrier,
        simm16: 0,
    })
}

// ----------------------------------------------------------------------------
// Expected load data

#[derive(Clone, Copy)]
enum Ld {
    U8,
    I8,
    U16,
    I16,
    Dw(usize),
}

/// The destination dwords of a load of kind `k` from byte `at` of `b`.
fn ld_expect(k: Ld, at: usize, b: &dyn Fn(usize) -> u8) -> Vec<u32> {
    let w = |o: usize| u32::from_le_bytes([b(o), b(o + 1), b(o + 2), b(o + 3)]);
    match k {
        Ld::U8 => vec![u32::from(b(at))],
        Ld::I8 => vec![i32::from(b(at) as i8) as u32],
        Ld::U16 => vec![u32::from(u16::from_le_bytes([b(at), b(at + 1)]))],
        Ld::I16 => vec![i32::from(i16::from_le_bytes([b(at), b(at + 1)])) as u32],
        Ld::Dw(n) => (0..n).map(|i| w(at + 4 * i)).collect(),
    }
}

/// Lane `lane` of `dst..dst+total` is `want` followed by [`SENT`] filler.
fn check_regs(r: &Rig, dst: usize, lane: usize, want: &[u32], total: usize, what: &str) {
    for (k, x) in want.iter().enumerate() {
        assert_eq!(r.w.v[dst + k][lane], *x, "{what}: lane {lane} dword {k}")
    }
    for k in want.len()..total {
        assert_eq!(
            r.w.v[dst + k][lane],
            SENT,
            "{what}: lane {lane} dword {k} must be untouched"
        )
    }
}

const DS_LOADS: [(DsOp, Ld); 8] = [
    (DsOp::ReadU8, Ld::U8),
    (DsOp::ReadI8, Ld::I8),
    (DsOp::ReadU16, Ld::U16),
    (DsOp::ReadI16, Ld::I16),
    (DsOp::ReadB32, Ld::Dw(1)),
    (DsOp::ReadB64, Ld::Dw(2)),
    (DsOp::ReadB96, Ld::Dw(3)),
    (DsOp::ReadB128, Ld::Dw(4)),
];
const FLAT_LOADS: [(FlatOp, Ld); 8] = [
    (FlatOp::LoadUbyte, Ld::U8),
    (FlatOp::LoadSbyte, Ld::I8),
    (FlatOp::LoadUshort, Ld::U16),
    (FlatOp::LoadSshort, Ld::I16),
    (FlatOp::LoadDword, Ld::Dw(1)),
    (FlatOp::LoadDwordx2, Ld::Dw(2)),
    (FlatOp::LoadDwordx3, Ld::Dw(3)),
    (FlatOp::LoadDwordx4, Ld::Dw(4)),
];
const MUBUF_LOADS: [(MubufOp, Ld); 8] = [
    (MubufOp::LoadUbyte, Ld::U8),
    (MubufOp::LoadSbyte, Ld::I8),
    (MubufOp::LoadUshort, Ld::U16),
    (MubufOp::LoadSshort, Ld::I16),
    (MubufOp::LoadDword, Ld::Dw(1)),
    (MubufOp::LoadDwordx2, Ld::Dw(2)),
    (MubufOp::LoadDwordx3, Ld::Dw(3)),
    (MubufOp::LoadDwordx4, Ld::Dw(4)),
];
const FLAT_STORES: [(FlatOp, usize); 6] = [
    (FlatOp::StoreByte, 1),
    (FlatOp::StoreShort, 2),
    (FlatOp::StoreDword, 4),
    (FlatOp::StoreDwordx2, 8),
    (FlatOp::StoreDwordx3, 12),
    (FlatOp::StoreDwordx4, 16),
];
const MUBUF_STORES: [(MubufOp, usize); 6] = [
    (MubufOp::StoreByte, 1),
    (MubufOp::StoreShort, 2),
    (MubufOp::StoreDword, 4),
    (MubufOp::StoreDwordx2, 8),
    (MubufOp::StoreDwordx3, 12),
    (MubufOp::StoreDwordx4, 16),
];
const DS_STORES: [(DsOp, usize); 6] = [
    (DsOp::WriteB8, 1),
    (DsOp::WriteB16, 2),
    (DsOp::WriteB32, 4),
    (DsOp::WriteB64, 8),
    (DsOp::WriteB96, 12),
    (DsOp::WriteB128, 16),
];

// ============================================================================
// DS

#[test]
fn ds_read_every_width_and_extension() {
    for (op, kind) in DS_LOADS {
        let mut r = Rig::new();
        r.set_v(1, |l| (l * 16) as u32);
        r.fill_v(10, 6, SENT);
        r.ok(&ds_off(op, Some(10), 1, None, 0x0104)); // offset1:offset0 = 1:4 = 260 bytes
        assert!(
            r.busy_v(10),
            "{op:?}: destination must be busy until retire"
        );
        assert_eq!(r.units(Counter::Lgkm), 1, "{op:?}");
        r.drain().unwrap();
        let b = |i: usize| r.lds[i];
        for l in 0..WAVE {
            let want = ld_expect(kind, l * 16 + 260, &b);
            check_regs(&r, 10, l, &want, 6, &format!("{op:?}"));
        }
    }
}

#[test]
fn ds_signed_and_unsigned_extension_explicit_bytes() {
    let mut r = Rig::new();
    r.lds[0x300] = 0x80;
    r.lds[0x301] = 0x7f;
    r.lds[0x302] = 0x01;
    r.lds[0x303] = 0x80; // u16 0x8001
    r.lds[0x304] = 0xff;
    r.lds[0x305] = 0x7f; // u16 0x7fff
    r.set_v(1, |_| 0x300);
    let cases: [(DsOp, u16, u32); 8] = [
        (DsOp::ReadU8, 0x300, 0x0000_0080),
        (DsOp::ReadI8, 0x300, 0xffff_ff80),
        (DsOp::ReadI8, 0x301, 0x0000_007f),
        (DsOp::ReadU16, 0x302, 0x0000_8001),
        (DsOp::ReadI16, 0x302, 0xffff_8001),
        (DsOp::ReadI16, 0x304, 0x0000_7fff),
        (DsOp::ReadU16, 0x304, 0x0000_7fff),
        (DsOp::ReadU8, 0x303, 0x0000_0080),
    ];
    for (op, off, want) in cases {
        r.fill_v(10, 1, SENT);
        r.run(&ds_off(op, Some(10), 1, None, off - 0x300));
        for l in 0..WAVE {
            assert_eq!(r.w.v[10][l], want, "{op:?} @{off:#x}")
        }
    }
}

#[test]
fn ds_write_every_width_with_snapshot() {
    for (op, n) in DS_STORES {
        let mut r = Rig::new();
        r.lds.fill(0xEE);
        r.set_v(1, |l| (l * 16) as u32);
        for k in 0..4 {
            r.set_v(4 + k, |l| val(l, k))
        }
        r.ok(&ds_off(op, None, 1, Some(4), 0x0104));
        assert!(
            r.lds.iter().all(|&b| b == 0xEE),
            "{op:?}: store must not be visible before retire"
        );
        // The address and data are snapshotted at issue.
        r.fill_v(1, 1, 0xFFFF_0000);
        r.fill_v(4, 4, 0);
        r.drain().unwrap();
        let mut want = vec![0xEEu8; LDS_BYTES];
        for l in 0..WAVE {
            want[l * 16 + 260..l * 16 + 260 + n].copy_from_slice(&st_bytes(l, n))
        }
        assert!(r.lds == want, "{op:?}: LDS image differs");
        assert!(r.w.pending.is_empty());
    }
}

#[test]
fn ds_two_address_write2_and_st64() {
    // (op, byte multiplier of one offset unit, dwords per element, lane stride, offset0, offset1)
    let cases: [(DsOp, usize, usize, usize, u8, u8); 5] = [
        (DsOp::Write2B32, 4, 1, 64, 1, 5),
        (DsOp::Write2B64, 8, 2, 64, 1, 3),
        (DsOp::Write2st64B32, 256, 1, 4, 1, 2),
        (DsOp::Write2st64B32, 256, 1, 4, 0, 255),
        (DsOp::Write2st64B64, 512, 2, 8, 1, 2),
    ];
    for (op, mult, nd, ls, o0, o1) in cases {
        let mut r = Rig::new();
        r.lds.fill(0xEE);
        r.set_v(1, |l| (l * ls) as u32);
        for k in 0..2 * nd {
            r.set_v(4 + k, |l| val(l, k))
        }
        r.ok(&ds(op, None, 1, Some(4), Some(4 + nd as u8), o0, o1));
        assert!(
            r.lds.iter().all(|&b| b == 0xEE),
            "{op:?}: not visible before retire"
        );
        r.drain().unwrap();
        let mut want = vec![0xEEu8; LDS_BYTES];
        for l in 0..WAVE {
            for (e, off) in [(0usize, o0), (1, o1)] {
                let a = l * ls + usize::from(off) * mult;
                for k in 0..nd {
                    want[a + 4 * k..a + 4 * k + 4]
                        .copy_from_slice(&val(l, e * nd + k).to_le_bytes())
                }
            }
        }
        assert!(r.lds == want, "{op:?} off {o0},{o1}: LDS image differs");
    }
}

#[test]
fn ds_two_address_read2_and_st64() {
    // (op, multiplier, dwords per element, lane stride, offset0, offset1)
    let cases: [(DsOp, usize, usize, usize, u8, u8); 5] = [
        (DsOp::Read2B32, 4, 1, 64, 1, 5),
        (DsOp::Read2B64, 8, 2, 64, 1, 3),
        (DsOp::Read2st64B32, 256, 1, 4, 1, 2),
        (DsOp::Read2st64B32, 256, 1, 4, 0, 255),
        (DsOp::Read2st64B64, 512, 2, 8, 1, 2),
    ];
    for (op, mult, nd, ls, o0, o1) in cases {
        let mut r = Rig::new();
        r.set_v(1, |l| (l * ls) as u32);
        r.fill_v(10, 6, SENT);
        r.ok(&ds(op, Some(10), 1, None, None, o0, o1));
        assert!(
            r.busy_v(10) && r.busy_v(10 + 2 * nd - 1),
            "{op:?}: whole destination tuple busy"
        );
        r.drain().unwrap();
        let b = |i: usize| r.lds[i];
        for l in 0..WAVE {
            let mut want = ld_expect(Ld::Dw(nd), l * ls + usize::from(o0) * mult, &b);
            want.extend(ld_expect(Ld::Dw(nd), l * ls + usize::from(o1) * mult, &b));
            check_regs(&r, 10, l, &want, 6, &format!("{op:?}"));
        }
    }
}

#[test]
fn ds_masked_exec_and_snapshot() {
    let mask = 0xA5A5_5A5A_0F0F_F0F0u64;
    // Store: inactive lanes carry an out-of-range address and must not fault.
    let mut r = Rig::new();
    r.lds.fill(0xEE);
    r.set_v(1, |l| {
        if mask >> l & 1 == 1 {
            (l * 4) as u32
        } else {
            0xFFFF_FFF0
        }
    });
    r.set_v(4, |l| val(l, 0));
    r.w.exec = mask;
    r.ok(&ds_off(DsOp::WriteB32, None, 1, Some(4), 0));
    r.w.exec = u64::MAX; // the lane mask is the one at issue
    r.drain().unwrap();
    for l in 0..WAVE {
        let want = if mask >> l & 1 == 1 {
            val(l, 0)
        } else {
            0xEEEE_EEEE
        };
        assert_eq!(r.lds_word(l * 4), want, "write lane {l}");
    }
    // Load: inactive lanes keep their destination.
    let mut r = Rig::new();
    r.set_v(1, |l| {
        if mask >> l & 1 == 1 {
            (l * 4) as u32
        } else {
            0xFFFF_FFF0
        }
    });
    r.fill_v(10, 1, SENT);
    r.w.exec = mask;
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 1, None, 0));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    for l in 0..WAVE {
        let want = if mask >> l & 1 == 1 {
            pat_u32(l * 4)
        } else {
            SENT
        };
        assert_eq!(r.w.v[10][l], want, "read lane {l}");
    }
    // EXEC = 0: nothing is accessed, nothing faults.
    let mut r = Rig::with_lds(64);
    r.set_v(1, |_| 0xFFFF_FFF0);
    r.w.exec = 0;
    r.try_run(&ds_off(DsOp::ReadB32, Some(10), 1, None, 0))
        .unwrap();
    r.try_run(&ds_off(DsOp::WriteB32, None, 1, Some(4), 0))
        .unwrap();
}

#[test]
fn ds_load_samples_lds_at_retire() {
    let mut r = Rig::new();
    r.set_v(1, |_| 0x40);
    r.fill_v(10, 1, SENT);
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 1, None, 0));
    r.lds_put(0x40, 0xC0DE_0001); // later write, before the wait
    r.fill_v(1, 1, 0x80); // address snapshotted
    r.wait(63, 0).unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], 0xC0DE_0001)
    }
}

#[test]
fn ds_integer_and_f32_atomics_distinct_addresses() {
    fn fadd(a: u32, b: u32) -> u32 {
        (f32::from_bits(a) + f32::from_bits(b)).to_bits()
    }
    fn fmax(a: u32, b: u32) -> u32 {
        f32::from_bits(a).max(f32::from_bits(b)).to_bits()
    }
    fn fmin(a: u32, b: u32) -> u32 {
        f32::from_bits(a).min(f32::from_bits(b)).to_bits()
    }
    fn imax(a: u32, b: u32) -> u32 {
        (a as i32).max(b as i32) as u32
    }
    let int_init = |l: usize| val(l, 0);
    let int_data = |l: usize| val(l, 1);
    let f_init = |l: usize| ((l as f32 - 20.0) * 0.5).to_bits();
    let f_data = |l: usize| (l as f32 * 0.25 - 3.0).to_bits();
    type Model = fn(u32, u32) -> u32;
    let cases: [(DsOp, Model, bool); 6] = [
        (DsOp::AddU32, u32::wrapping_add, false),
        (DsOp::MaxU32, u32::max, false),
        (DsOp::MaxI32, imax, false),
        (DsOp::AddF32, fadd, true),
        (DsOp::MaxF32, fmax, true),
        (DsOp::MinF32, fmin, true),
    ];
    for (op, model, float) in cases {
        let (init, data): (&dyn Fn(usize) -> u32, &dyn Fn(usize) -> u32) = if float {
            (
                &f_init as &dyn Fn(usize) -> u32,
                &f_data as &dyn Fn(usize) -> u32,
            )
        } else {
            (
                &int_init as &dyn Fn(usize) -> u32,
                &int_data as &dyn Fn(usize) -> u32,
            )
        };
        let mut r = Rig::new();
        for l in 0..WAVE {
            r.lds_put(0x200 + l * 4, init(l))
        }
        r.set_v(1, |l| (l * 4) as u32);
        r.set_v(4, data);
        r.ok(&ds_off(op, None, 1, Some(4), 0x200));
        for l in 0..WAVE {
            assert_eq!(
                r.lds_word(0x200 + l * 4),
                init(l),
                "{op:?}: not visible before retire"
            )
        }
        r.fill_v(4, 1, 0); // data snapshot
        r.drain().unwrap();
        for l in 0..WAVE {
            assert_eq!(
                r.lds_word(0x200 + l * 4),
                model(init(l), data(l)),
                "{op:?} lane {l}"
            )
        }
    }
}

#[test]
fn ds_add_rtn_u32_returns_old_value() {
    // Distinct addresses.
    let mut r = Rig::new();
    for l in 0..WAVE {
        r.lds_put(0x200 + l * 4, val(l, 0))
    }
    r.set_v(1, |l| (l * 4) as u32);
    r.set_v(4, |l| val(l, 1));
    r.fill_v(10, 1, SENT);
    r.ok(&ds_off(DsOp::AddRtnU32, Some(10), 1, Some(4), 0x200));
    assert!(r.busy_v(10));
    for l in 0..WAVE {
        assert_eq!(
            r.lds_word(0x200 + l * 4),
            val(l, 0),
            "not visible before retire"
        )
    }
    r.drain().unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], val(l, 0), "returned old value, lane {l}");
        assert_eq!(
            r.lds_word(0x200 + l * 4),
            val(l, 0).wrapping_add(val(l, 1)),
            "lane {l}"
        );
    }
    // One address, masked lanes: lane order, each lane returns the running value.
    let mask = 0xF0F0_3C3C_00FF_FF00u64;
    let mut r = Rig::new();
    r.lds_put(0x80, 7);
    r.set_v(1, |_| 0x80);
    r.set_v(4, |l| l as u32 + 1);
    r.fill_v(10, 1, SENT);
    r.w.exec = mask;
    r.ok(&ds_off(DsOp::AddRtnU32, Some(10), 1, Some(4), 0));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    let mut acc = 7u32;
    for l in 0..WAVE {
        if mask >> l & 1 == 1 {
            assert_eq!(
                r.w.v[10][l], acc,
                "lane {l} returns the value before its own add"
            );
            acc += l as u32 + 1;
        } else {
            assert_eq!(r.w.v[10][l], SENT, "inactive lane {l}");
        }
    }
    assert_eq!(r.lds_word(0x80), acc);
}

#[test]
fn ds_add_u32_same_address_accumulates() {
    let mut r = Rig::new();
    r.lds_put(0x100, 1000);
    r.set_v(1, |_| 0x100);
    r.set_v(4, |l| l as u32 + 1);
    r.run(&ds_off(DsOp::AddU32, None, 1, Some(4), 0));
    assert_eq!(r.lds_word(0x100), 1000 + (1..=64).sum::<u32>());
}

#[test]
fn ds_out_of_range_lds_faults() {
    let cases: [(&str, Inst, u32); 8] = [
        (
            "read at end",
            ds_off(DsOp::ReadB32, Some(10), 1, None, 0),
            256,
        ),
        (
            "read straddling end",
            ds_off(DsOp::ReadB32, Some(10), 1, None, 0),
            253,
        ),
        (
            "read b128 straddling end",
            ds_off(DsOp::ReadB128, Some(10), 1, None, 0),
            244,
        ),
        (
            "write at end",
            ds_off(DsOp::WriteB32, None, 1, Some(4), 0),
            256,
        ),
        (
            "write b64 straddling end",
            ds_off(DsOp::WriteB64, None, 1, Some(4), 0),
            252,
        ),
        (
            "atomic rtn at end",
            ds_off(DsOp::AddRtnU32, Some(10), 1, Some(4), 0),
            256,
        ),
        (
            "read2 second element past end",
            ds(DsOp::Read2B32, Some(10), 1, None, None, 0, 255),
            0,
        ),
        (
            "write2st64 past end",
            ds(DsOp::Write2st64B32, None, 1, Some(4), Some(5), 0, 2),
            0,
        ),
    ];
    for (name, inst, addr) in cases {
        let mut r = Rig::with_lds(256);
        r.set_v(1, |_| addr);
        let res = r.try_run(&inst);
        assert!(is_fault(&res), "{name}: {res:?}");
    }
    // An in-range offset that carries the address out of range.
    let mut r = Rig::with_lds(256);
    r.set_v(1, |_| 200);
    let res = r.try_run(&ds_off(DsOp::ReadB32, Some(10), 1, None, 0x0100));
    assert!(is_fault(&res), "{res:?}");
}

#[test]
fn ds_gds_is_unsupported() {
    let mut r = Rig::new();
    r.set_v(1, |l| (l * 4) as u32);
    for (op, vdst, d0) in [
        (DsOp::ReadB32, Some(10u8), None),
        (DsOp::WriteB32, None, Some(4u8)),
        (DsOp::AddRtnU32, Some(10), Some(4)),
    ] {
        let inst = Inst::Ds(Ds {
            op,
            vdst: vdst.map(v),
            addr: Some(v(1)),
            data0: d0.map(v),
            data1: None,
            offset0: 0,
            offset1: 0,
            gds: true,
        });
        let res = r.try_run(&inst);
        assert!(is_unsupported(&res), "{op:?}: {res:?}");
    }
}

// ============================================================================
// GLOBAL

#[test]
fn global_load_every_width_and_extension() {
    for (op, kind) in FLAT_LOADS {
        let mut r = Rig::global(0x4000);
        r.set_v64(2, |l| G + 0x100 + l as u64 * 16);
        r.fill_v(10, 6, SENT);
        r.ok(&gl(op, Some(10), 2, None, 4, false));
        assert_eq!(r.w.pending.len(), 1);
        let p = &r.w.pending[0];
        assert_eq!(
            (p.counter, p.weight, p.out_of_order),
            (Counter::Vm, 1, false),
            "{op:?}"
        );
        assert!(r.busy_v(10), "{op:?}");
        r.drain().unwrap();
        let b = |i: usize| pat(i);
        for l in 0..WAVE {
            let want = ld_expect(kind, 0x100 + l * 16 + 4, &b);
            check_regs(&r, 10, l, &want, 6, &format!("{op:?}"));
        }
    }
}

#[test]
fn global_load_sign_extension_explicit_bytes() {
    let mut r = Rig::global(0x100);
    r.m.write(G + 0x10, &[0x80, 0x7f, 0x01, 0x80, 0xff, 0xff])
        .unwrap();
    let cases: [(FlatOp, i16, u32); 6] = [
        (FlatOp::LoadUbyte, 0, 0x80),
        (FlatOp::LoadSbyte, 0, 0xffff_ff80),
        (FlatOp::LoadSbyte, 1, 0x7f),
        (FlatOp::LoadUshort, 2, 0x8001),
        (FlatOp::LoadSshort, 2, 0xffff_8001),
        (FlatOp::LoadSshort, 4, 0xffff_ffff),
    ];
    for (op, off, want) in cases {
        r.set_v64(2, |_| G + 0x10);
        r.fill_v(10, 1, SENT);
        r.run(&gl(op, Some(10), 2, None, off, false));
        for l in 0..WAVE {
            assert_eq!(r.w.v[10][l], want, "{op:?} +{off}")
        }
    }
}

#[test]
fn global_store_every_width_with_snapshot() {
    for (op, n) in FLAT_STORES {
        let mut r = Rig::global(0x4000);
        r.m.unmap("g");
        r.map_fill("g", G, 0x4000, 0xEE);
        r.set_v64(2, |l| G + 0x100 + l as u64 * 16);
        for k in 0..4 {
            r.set_v(6 + k, |l| val(l, k))
        }
        r.ok(&gl(op, None, 2, Some(6), 4, false));
        let p = &r.w.pending[0];
        assert_eq!(
            (p.counter, p.weight, p.out_of_order),
            (Counter::Vm, 1, false),
            "{op:?}"
        );
        assert!(p.busy.is_empty(), "stores have no destination");
        assert!(
            r.bytes("g").iter().all(|&b| b == 0xEE),
            "{op:?}: not visible before retire"
        );
        r.fill_v(2, 2, 0); // address and data snapshotted
        r.fill_v(6, 4, 0);
        r.drain().unwrap();
        let mut want = vec![0xEEu8; 0x4000];
        for l in 0..WAVE {
            let a = 0x100 + l * 16 + 4;
            want[a..a + n].copy_from_slice(&st_bytes(l, n))
        }
        assert!(
            r.bytes("g") == want.as_slice(),
            "{op:?}: memory image differs"
        );
    }
}

#[test]
fn global_load_samples_memory_at_retire() {
    let mut r = Rig::global(0x100);
    r.set_v64(2, |_| G + 0x20);
    r.fill_v(10, 1, SENT);
    r.ok(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    r.put(G + 0x20, 0xFEED_0001); // a store by someone else after issue, before the wait
    r.fill_v(2, 2, 0);
    r.wait(0, 15).unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], 0xFEED_0001)
    }
}

#[test]
fn global_signed_offsets_and_64bit_address() {
    for off in [-4096i16, -8, 0, 4, 4095] {
        let mut r = Rig::global(0x4000);
        r.set_v64(2, |l| G + 0x2000 + l as u64 * 4);
        r.fill_v(10, 1, SENT);
        r.run(&gl(FlatOp::LoadDword, Some(10), 2, None, off, false));
        for l in 0..WAVE {
            let at = (0x2000 + l as i64 * 4 + i64::from(off)) as usize;
            assert_eq!(r.w.v[10][l], pat_u32(at), "offset {off} lane {l}");
        }
    }
    // Carry out of the low address dword, and a negative offset borrowing from the high one.
    let mut r = Rig::new();
    r.map_pat("x", 0xFFFF_F000, 0x2000);
    r.set_v64(2, |_| 0xFFFF_FFFC);
    r.fill_v(10, 1, SENT);
    r.run(&gl(FlatOp::LoadDword, Some(10), 2, None, 8, false)); // 0x1_0000_0004
    assert_eq!(r.w.v[10][0], pat_u32(0x1004));
    r.set_v64(2, |_| 0x1_0000_0004);
    r.run(&gl(FlatOp::LoadDword, Some(10), 2, None, -8, false)); // 0xFFFF_FFFC
    assert_eq!(r.w.v[10][0], pat_u32(0xFFC));
    // Store with a negative offset.
    let mut r = Rig::new();
    r.map_fill("x", 0xFFFF_F000, 0x2000, 0);
    r.set_v64(2, |_| 0x1_0000_0004);
    r.fill_v(6, 1, 0xA1B2_C3D4);
    r.run(&gl(FlatOp::StoreDword, None, 2, Some(6), -8, false));
    assert_eq!(r.word(0xFFFF_FFFC), 0xA1B2_C3D4);
}

#[test]
fn global_saddr_form() {
    // saddr base + zero-extended 32-bit VGPR offset + signed immediate.
    for off in [-4i16, 0, 4] {
        let mut r = Rig::global(0x4000);
        r.set_s64(4, G + 0x80);
        r.set_v(2, |l| l as u32 * 16);
        r.fill_v(10, 1, SENT);
        r.ok(&gls(FlatOp::LoadDword, Some(10), 2, None, 4, off, false));
        r.set_s64(4, 0); // base snapshotted at issue
        r.drain().unwrap();
        for l in 0..WAVE {
            let at = (0x80 + l as i64 * 16 + i64::from(off)) as usize;
            assert_eq!(r.w.v[10][l], pat_u32(at), "saddr offset {off} lane {l}");
        }
    }
    // The VGPR offset is unsigned: 0xFFFF_FFF0 must not be sign-extended.
    let mut r = Rig::new();
    r.map_pat("hi", 0x2_0000_0000, 0x100);
    r.set_s64(4, 0x1_0000_0010);
    r.set_v(2, |_| 0xFFFF_FFF0);
    r.fill_v(10, 1, SENT);
    r.w.exec = 1;
    r.run(&gls(FlatOp::LoadDword, Some(10), 2, None, 4, 0, false));
    assert_eq!(r.w.v[10][0], pat_u32(0));
    // Stores through saddr.
    let mut r = Rig::new();
    r.map_fill("g", G, 0x1000, 0xEE);
    r.set_s64(4, G + 0x40);
    r.set_v(2, |l| l as u32 * 8);
    r.set_v(6, |l| val(l, 0));
    r.set_v(7, |l| val(l, 1));
    r.run(&gls(FlatOp::StoreDwordx2, None, 2, Some(6), 4, 8, false));
    for l in 0..WAVE {
        assert_eq!(r.word(G + 0x40 + 8 + l as u64 * 8), val(l, 0));
        assert_eq!(r.word(G + 0x40 + 8 + l as u64 * 8 + 4), val(l, 1));
    }
}

#[test]
fn global_atomic_add_with_and_without_return() {
    // Without return: sc0 clear, no destination.
    let mut r = Rig::global(0x1000);
    for l in 0..WAVE {
        r.put(G + l as u64 * 4, 100 + l as u32)
    }
    r.set_v64(2, |l| G + l as u64 * 4);
    r.set_v(6, |l| val(l, 0));
    r.ok(&gl(FlatOp::AtomicAdd, None, 2, Some(6), 0, false));
    assert!(r.w.pending[0].busy.is_empty());
    for l in 0..WAVE {
        assert_eq!(
            r.word(G + l as u64 * 4),
            100 + l as u32,
            "not visible before retire"
        )
    }
    r.fill_v(2, 2, 0);
    r.fill_v(6, 1, 0); // snapshots
    r.drain().unwrap();
    for l in 0..WAVE {
        assert_eq!(
            r.word(G + l as u64 * 4),
            (100 + l as u32).wrapping_add(val(l, 0)),
            "lane {l}"
        )
    }

    // With return (sc0): the pre-op value lands in vdst at retire.
    let mut r = Rig::global(0x1000);
    for l in 0..WAVE {
        r.put(G + l as u64 * 4, 100 + l as u32)
    }
    r.set_v64(2, |l| G + l as u64 * 4);
    r.set_v(6, |l| val(l, 0));
    r.fill_v(10, 1, SENT);
    r.ok(&gl(FlatOp::AtomicAdd, Some(10), 2, Some(6), 0, true));
    assert!(r.busy_v(10));
    r.drain().unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], 100 + l as u32, "old value, lane {l}");
        assert_eq!(
            r.word(G + l as u64 * 4),
            (100 + l as u32).wrapping_add(val(l, 0))
        );
    }

    // One address, masked lanes: lane order.
    let mask = 0x0123_4567_89AB_CDEFu64;
    let mut r = Rig::global(0x1000);
    r.put(G + 0x40, 7);
    r.set_v64(2, |_| G + 0x40);
    r.set_v(6, |l| l as u32 + 1);
    r.fill_v(10, 1, SENT);
    r.w.exec = mask;
    r.ok(&gl(FlatOp::AtomicAdd, Some(10), 2, Some(6), 0, true));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    let mut acc = 7u32;
    for l in 0..WAVE {
        if mask >> l & 1 == 1 {
            assert_eq!(r.w.v[10][l], acc, "lane {l}");
            acc += l as u32 + 1
        } else {
            assert_eq!(r.w.v[10][l], SENT, "inactive lane {l}")
        }
    }
    assert_eq!(r.word(G + 0x40), acc);
}

#[test]
fn global_atomic_add_f32_with_and_without_return() {
    let init = |l: usize| (l as f32 * 1.5 - 10.0).to_bits();
    let data = |l: usize| (l as f32 * 0.25 + 0.5).to_bits();
    let add = |a: u32, b: u32| (f32::from_bits(a) + f32::from_bits(b)).to_bits();
    for ret in [false, true] {
        let mut r = Rig::global(0x1000);
        for l in 0..WAVE {
            r.put(G + l as u64 * 4, init(l))
        }
        r.set_v64(2, |l| G + l as u64 * 4);
        r.set_v(6, data);
        r.fill_v(10, 1, SENT);
        r.ok(&gl(
            FlatOp::AtomicAddF32,
            ret.then_some(10),
            2,
            Some(6),
            0,
            ret,
        ));
        for l in 0..WAVE {
            assert_eq!(
                r.word(G + l as u64 * 4),
                init(l),
                "not visible before retire"
            )
        }
        r.drain().unwrap();
        for l in 0..WAVE {
            assert_eq!(
                r.word(G + l as u64 * 4),
                add(init(l), data(l)),
                "ret={ret} lane {l}"
            );
            assert_eq!(
                r.w.v[10][l],
                if ret { init(l) } else { SENT },
                "ret={ret} lane {l}"
            );
        }
    }
    // One address: running f32 sum in lane order (all partial sums exact).
    let mut r = Rig::global(0x1000);
    r.put(G + 0x40, 1.0f32.to_bits());
    r.set_v64(2, |_| G + 0x40);
    r.set_v(6, |l| (0.5 * (l as f32 + 1.0)).to_bits());
    r.fill_v(10, 1, SENT);
    r.run(&gl(FlatOp::AtomicAddF32, Some(10), 2, Some(6), 0, true));
    let mut acc = 1.0f32;
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], acc.to_bits(), "lane {l}");
        acc += 0.5 * (l as f32 + 1.0)
    }
    assert_eq!(r.word(G + 0x40), acc.to_bits());
    // saddr form with return and a signed offset.
    let mut r = Rig::global(0x1000);
    for l in 0..WAVE {
        r.put(G + 0x100 + l as u64 * 4, init(l))
    }
    r.set_s64(4, G + 0x108);
    r.set_v(2, |l| l as u32 * 4);
    r.set_v(6, data);
    r.run(&gls(
        FlatOp::AtomicAddF32,
        Some(10),
        2,
        Some(6),
        4,
        -8,
        true,
    ));
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], init(l), "saddr lane {l}")
    }
}

#[test]
fn global_masked_exec_and_inactive_unmapped_addresses() {
    let mask = 0x0F0F_00FF_F0F0_3C3Cu64;
    let addr = |l: usize| {
        if mask >> l & 1 == 1 {
            G + 0x100 + l as u64 * 4
        } else {
            0
        }
    };
    // Load.
    let mut r = Rig::global(0x4000);
    r.set_v64(2, addr);
    r.fill_v(10, 1, SENT);
    r.w.exec = mask;
    r.ok(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    r.w.exec = u64::MAX; // the issue-time mask governs the retire
    r.drain().unwrap();
    for l in 0..WAVE {
        assert_eq!(
            r.w.v[10][l],
            if mask >> l & 1 == 1 {
                pat_u32(0x100 + l * 4)
            } else {
                SENT
            },
            "lane {l}"
        )
    }
    // Store.
    let mut r = Rig::new();
    r.map_fill("g", G, 0x4000, 0xEE);
    r.set_v64(2, addr);
    r.set_v(6, |l| val(l, 0));
    r.w.exec = mask;
    r.ok(&gl(FlatOp::StoreDword, None, 2, Some(6), 0, false));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    for l in 0..WAVE {
        let want = if mask >> l & 1 == 1 {
            val(l, 0)
        } else {
            0xEEEE_EEEE
        };
        assert_eq!(r.word(G + 0x100 + l as u64 * 4), want, "lane {l}");
    }
    // Atomic with return.
    let mut r = Rig::global(0x4000);
    r.set_v64(2, addr);
    r.set_v(6, |_| 1);
    r.fill_v(10, 1, SENT);
    r.w.exec = mask;
    r.ok(&gl(FlatOp::AtomicAdd, Some(10), 2, Some(6), 0, true));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    for l in 0..WAVE {
        let a = 0x100 + l * 4;
        if mask >> l & 1 == 1 {
            assert_eq!(r.w.v[10][l], pat_u32(a));
            assert_eq!(r.word(G + a as u64), pat_u32(a).wrapping_add(1));
        } else {
            assert_eq!(r.w.v[10][l], SENT);
        }
    }
    // EXEC = 0: nothing touches memory.
    let mut r = Rig::new();
    r.set_v64(2, |_| 0xdead_0000_0000);
    r.w.exec = 0;
    r.try_run(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false))
        .unwrap();
    r.try_run(&gl(FlatOp::StoreDword, None, 2, Some(6), 0, false))
        .unwrap();
}

#[test]
fn global_unmapped_and_readonly_access_faults() {
    let mut r = Rig::global(0x100);
    r.w.exec = 1;
    r.set_v64(2, |_| 0xdead_0000_0000);
    let res = r.try_run(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    assert!(is_fault(&res), "{res:?}");

    let mut r = Rig::global(0x100);
    r.w.exec = 1;
    r.set_v64(2, |_| 0xdead_0000_0000);
    let res = r.try_run(&gl(FlatOp::StoreDword, None, 2, Some(6), 0, false));
    assert!(is_fault(&res), "{res:?}");

    // One active lane is enough to fault.
    let mut r = Rig::global(0x100);
    r.w.exec = 0b11;
    r.set_v64(2, |l| if l == 0 { G } else { 0xdead_0000_0000 });
    let res = r.try_run(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    assert!(is_fault(&res), "{res:?}");

    // Straddling the end of a region.
    let mut r = Rig::global(0x100);
    r.w.exec = 1;
    r.set_v64(2, |_| G + 0xFE);
    let res = r.try_run(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    assert!(is_fault(&res), "{res:?}");

    // Atomic and store to a read-only region.
    let mut r = Rig::new();
    r.m.map_readonly("ro", G, vec![0; 0x100]).unwrap();
    r.w.exec = 1;
    r.set_v64(2, |_| G);
    let res = r.try_run(&gl(FlatOp::StoreDword, None, 2, Some(6), 0, false));
    assert!(is_fault(&res), "{res:?}");
    let res = r.try_run(&gl(FlatOp::AtomicAdd, None, 2, Some(6), 0, false));
    assert!(is_fault(&res), "{res:?}");
    r.w.pending.clear(); // the faulting operations stay queued; discard them
                         // Loads from read-only memory are fine.
    r.fill_v(10, 1, SENT);
    r.run(&gl(FlatOp::LoadDword, Some(10), 2, None, 0, false));
    assert_eq!(r.w.v[10][0], 0);
}

#[test]
fn flat_and_scratch_segments_are_unsupported() {
    let mut r = Rig::global(0x100);
    r.w.exec = 1;
    r.set_v64(2, |_| G);
    let cases = [
        flat(
            FlatSeg::Flat,
            FlatOp::LoadDword,
            Some(10),
            Some(2),
            None,
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Flat,
            FlatOp::LoadUbyte,
            Some(10),
            Some(2),
            None,
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Flat,
            FlatOp::StoreDword,
            None,
            Some(2),
            Some(6),
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Flat,
            FlatOp::StoreDwordx4,
            None,
            Some(2),
            Some(6),
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Flat,
            FlatOp::AtomicAdd,
            Some(10),
            Some(2),
            Some(6),
            None,
            0,
            true,
        ),
        flat(
            FlatSeg::Flat,
            FlatOp::AtomicAddF32,
            None,
            Some(2),
            Some(6),
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Scratch,
            FlatOp::LoadDword,
            Some(10),
            Some(2),
            None,
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Scratch,
            FlatOp::LoadDwordx4,
            Some(10),
            Some(2),
            None,
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Scratch,
            FlatOp::StoreDword,
            None,
            Some(2),
            Some(6),
            None,
            0,
            false,
        ),
        flat(
            FlatSeg::Scratch,
            FlatOp::StoreByte,
            None,
            Some(2),
            Some(6),
            None,
            0,
            false,
        ),
    ];
    for inst in &cases {
        let res = r.try_run(inst);
        assert!(is_unsupported(&res), "{inst:?}: {res:?}");
        r.w.pending.clear();
    }
}

// ============================================================================
// SMEM

fn smem_rig() -> Rig {
    let mut r = Rig::new();
    r.map_pat("g", G, 0x1000);
    r.set_s64(2, G);
    for k in 40..80 {
        r.w.s[k] = SENT
    }
    r
}

#[test]
fn smem_every_load_width() {
    for (op, n) in [
        (SmemOp::LoadDword, 1usize),
        (SmemOp::LoadDwordx2, 2),
        (SmemOp::LoadDwordx4, 4),
        (SmemOp::LoadDwordx8, 8),
        (SmemOp::LoadDwordx16, 16),
    ] {
        let mut r = smem_rig();
        r.ok(&sm(op, Some(48), Some(2), SmemOffset::Imm(0x10)));
        let p = &r.w.pending[0];
        assert_eq!(
            (p.counter, p.out_of_order, p.weight),
            (Counter::Lgkm, true, if n == 1 { 1 } else { 2 }),
            "{op:?}"
        );
        assert!(
            r.busy_s(48) && r.busy_s(48 + n - 1),
            "{op:?}: whole destination busy"
        );
        assert!(
            !r.busy_s(48 + n),
            "{op:?}: nothing beyond the destination is busy"
        );
        r.drain().unwrap();
        for k in 0..n {
            assert_eq!(r.w.s[48 + k], pat_u32(0x10 + 4 * k), "{op:?} dword {k}")
        }
        assert_eq!(r.w.s[48 + n], SENT, "{op:?}: next register untouched");
    }
}

#[test]
fn smem_every_buffer_load_width() {
    for (op, n) in [
        (SmemOp::BufferLoadDword, 1usize),
        (SmemOp::BufferLoadDwordx2, 2),
        (SmemOp::BufferLoadDwordx4, 4),
        (SmemOp::BufferLoadDwordx8, 8),
        (SmemOp::BufferLoadDwordx16, 16),
    ] {
        let mut r = smem_rig();
        r.set_desc(8, G, 0, 0x1000, W3);
        r.ok(&sm(op, Some(48), Some(8), SmemOffset::Imm(0x20)));
        let p = &r.w.pending[0];
        assert_eq!(
            (p.counter, p.out_of_order, p.weight),
            (Counter::Lgkm, true, if n == 1 { 1 } else { 2 }),
            "{op:?}"
        );
        assert!(r.busy_s(48) && r.busy_s(48 + n - 1), "{op:?}");
        r.drain().unwrap();
        for k in 0..n {
            assert_eq!(r.w.s[48 + k], pat_u32(0x20 + 4 * k), "{op:?} dword {k}")
        }
        assert_eq!(r.w.s[48 + n], SENT);
    }
}

#[test]
fn smem_offset_forms() {
    // Negative immediate.
    let mut r = smem_rig();
    r.set_s64(2, G + 0x40);
    // ISA §8.2.1.1 p.60: a negative combined offset is illegal; the emulator faults.
    let res = r.try_run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::Imm(-16),
    ));
    assert!(is_fault(&res), "{res:?}");
    // A negative immediate is fine while the SGPR total stays non-negative.
    let mut r = smem_rig();
    r.set_s64(2, G + 0x40);
    r.w.s[6] = 0x40;
    r.run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::SgprImm(s(6), -16),
    ));
    assert_eq!(r.w.s[48], pat_u32(0x70));
    // SGPR offset.
    let mut r = smem_rig();
    r.w.s[6] = 0x44;
    r.run(&sm(
        SmemOp::LoadDwordx2,
        Some(48),
        Some(2),
        SmemOffset::Sgpr(s(6)),
    ));
    assert_eq!((r.w.s[48], r.w.s[49]), (pat_u32(0x44), pat_u32(0x48)));
    // SGPR plus immediate.
    let mut r = smem_rig();
    r.w.s[6] = 0x44;
    r.run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::SgprImm(s(6), 8),
    ));
    assert_eq!(r.w.s[48], pat_u32(0x4c));
    // Buffer forms.
    let mut r = smem_rig();
    r.set_desc(8, G, 0, 0x1000, W3);
    r.w.s[6] = 0x44;
    r.run(&sm(
        SmemOp::BufferLoadDword,
        Some(48),
        Some(8),
        SmemOffset::Sgpr(s(6)),
    ));
    assert_eq!(r.w.s[48], pat_u32(0x44));
    r.run(&sm(
        SmemOp::BufferLoadDword,
        Some(49),
        Some(8),
        SmemOffset::SgprImm(s(6), 8),
    ));
    assert_eq!(r.w.s[49], pat_u32(0x4c));
    // SMEM is scalar: EXEC = 0 does not suppress it.
    let mut r = smem_rig();
    r.w.exec = 0;
    r.run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::Imm(0),
    ));
    assert_eq!(r.w.s[48], pat_u32(0));
}

#[test]
fn smem_ignores_low_two_address_and_offset_bits() {
    let mut r = smem_rig();
    r.set_s64(2, G + 3);
    r.run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::Imm(8),
    ));
    assert_eq!(r.w.s[48], pat_u32(8), "base low bits ignored");
    for (imm, want) in [(6, 4usize), (7, 4), (5, 4), (4, 4)] {
        let mut r = smem_rig();
        r.run(&sm(
            SmemOp::LoadDword,
            Some(48),
            Some(2),
            SmemOffset::Imm(imm),
        ));
        assert_eq!(r.w.s[48], pat_u32(want), "offset {imm} low bits ignored");
    }
    let mut r = smem_rig();
    r.set_s64(2, G + 2);
    r.run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::Imm(0),
    ));
    assert_eq!(r.w.s[48], pat_u32(0));
}

#[test]
fn smem_samples_memory_at_retire_and_snapshots_registers() {
    let mut r = smem_rig();
    r.w.s[6] = 0x20;
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(2),
        SmemOffset::Sgpr(s(6)),
    ));
    r.put(G + 0x20, 0xABCD_0123); // later memory write
    r.set_s64(2, 0); // base and offset registers snapshotted at issue
    r.w.s[6] = 0x30;
    r.wait(63, 0).unwrap();
    assert_eq!(r.w.s[48], 0xABCD_0123);
}

#[test]
fn smem_buffer_bounds_clamp_dwords_not_performed() {
    // Raw (stride 0): limit = num_records bytes; dwords at or past it are not performed.
    let mut r = smem_rig();
    r.set_desc(8, G, 0, 24, W3);
    r.ok(&sm(
        SmemOp::BufferLoadDwordx8,
        Some(48),
        Some(8),
        SmemOffset::Imm(8),
    ));
    r.drain().unwrap();
    // dwords at byte 8, 12, 16, 20 are inside the 24-byte limit; 24.. are clamped.
    for k in 0..4 {
        assert_eq!(r.w.s[48 + k], pat_u32(8 + 4 * k), "dword {k}")
    }
    for k in 4..8 {
        assert_eq!(r.w.s[48 + k], SENT, "clamped dword {k} must stay unchanged")
    }
    // Entirely beyond the limit: nothing is written, and the load still completes.
    let mut r = smem_rig();
    r.set_desc(8, G, 0, 24, W3);
    r.run(&sm(
        SmemOp::BufferLoadDwordx4,
        Some(48),
        Some(8),
        SmemOffset::Imm(24),
    ));
    for k in 0..4 {
        assert_eq!(r.w.s[48 + k], SENT, "dword {k}")
    }
    assert!(r.w.pending.is_empty());
    // The limit also applies to an SGPR offset.
    let mut r = smem_rig();
    r.set_desc(8, G, 0, 16, W3);
    r.w.s[6] = 12;
    r.run(&sm(
        SmemOp::BufferLoadDwordx2,
        Some(48),
        Some(8),
        SmemOffset::Sgpr(s(6)),
    ));
    assert_eq!((r.w.s[48], r.w.s[49]), (pat_u32(12), SENT));
}

#[test]
fn smem_buffer_structured_stride_is_unsupported() {
    // A non-zero descriptor stride makes the m_size formula ambiguous in the ISA guide: no guessing.
    for op in [
        SmemOp::BufferLoadDword,
        SmemOp::BufferLoadDwordx2,
        SmemOp::BufferLoadDwordx4,
        SmemOp::BufferLoadDwordx8,
        SmemOp::BufferLoadDwordx16,
    ] {
        for (stride, nrec) in [(8u32, 3u32), (4, 0x1000), (16, 1)] {
            let mut r = smem_rig();
            r.set_desc(8, G, stride, nrec, W3);
            let res = r.try_run(&sm(op, Some(48), Some(8), SmemOffset::Imm(0)));
            assert!(is_unsupported(&res), "{op:?} stride {stride}: {res:?}");
        }
    }
}

#[test]
fn smem_unmapped_memory_faults() {
    let mut r = smem_rig();
    r.set_s64(4, 0xdead_0000_0000);
    let res = r.try_run(&sm(
        SmemOp::LoadDword,
        Some(48),
        Some(4),
        SmemOffset::Imm(0),
    ));
    assert!(is_fault(&res), "{res:?}");
    // In bounds per the descriptor but not mapped.
    let mut r = smem_rig();
    r.set_desc(8, 0xdead_0000_0000, 0, 0x1000, W3);
    let res = r.try_run(&sm(
        SmemOp::BufferLoadDword,
        Some(48),
        Some(8),
        SmemOffset::Imm(0),
    ));
    assert!(is_fault(&res), "{res:?}");
}

#[test]
fn smem_time_ops_are_unsupported() {
    for op in [SmemOp::Memtime, SmemOp::Memrealtime] {
        let mut r = smem_rig();
        let res = r.try_run(&sm(op, Some(48), None, SmemOffset::Imm(0)));
        assert!(is_unsupported(&res), "{op:?}: {res:?}");
    }
}

// ============================================================================
// MUBUF (raw)

/// A rig with the pattern region at [`G`] and a descriptor in `s[4:7]`,
/// `soffset` register `s8` = 0.
fn mb_rig(stride: u32, nrec: u32) -> Rig {
    let mut r = Rig::global(0x1000);
    r.set_desc(4, G, stride, nrec, W3);
    r.w.s[8] = 0;
    r.fill_v(10, 6, SENT);
    r
}

#[test]
fn mubuf_load_every_width_and_extension() {
    for (op, kind) in MUBUF_LOADS {
        let mut r = mb_rig(0, 0x1000);
        r.w.s[8] = 0x100;
        r.set_v(2, |l| l as u32 * 16);
        r.ok(&mb(op, 10, Some(2), 4, s(8), 4, true, false));
        let p = &r.w.pending[0];
        assert_eq!(
            (p.counter, p.weight, p.out_of_order),
            (Counter::Vm, 1, false),
            "{op:?}"
        );
        assert!(r.busy_v(10), "{op:?}");
        r.drain().unwrap();
        let b = |i: usize| pat(i);
        for l in 0..WAVE {
            let want = ld_expect(kind, 0x100 + l * 16 + 4, &b);
            check_regs(&r, 10, l, &want, 6, &format!("{op:?}"));
        }
    }
}

#[test]
fn mubuf_store_every_width_with_snapshot() {
    for (op, n) in MUBUF_STORES {
        let mut r = Rig::new();
        r.map_fill("g", G, 0x1000, 0xEE);
        r.set_desc(4, G, 0, 0x1000, W3);
        r.w.s[8] = 0x100;
        r.set_v(2, |l| l as u32 * 16);
        for k in 0..4 {
            r.set_v(6 + k, |l| val(l, k))
        }
        r.ok(&mb(op, 6, Some(2), 4, s(8), 4, true, false));
        assert!(
            r.bytes("g").iter().all(|&b| b == 0xEE),
            "{op:?}: not visible before retire"
        );
        r.fill_v(2, 1, 0);
        r.fill_v(6, 4, 0);
        r.set_desc(4, 0, 0, 0, 0); // descriptor snapshotted
        r.w.s[8] = 0;
        r.drain().unwrap();
        let mut want = vec![0xEEu8; 0x1000];
        for l in 0..WAVE {
            let a = 0x100 + l * 16 + 4;
            want[a..a + n].copy_from_slice(&st_bytes(l, n))
        }
        assert!(
            r.bytes("g") == want.as_slice(),
            "{op:?}: memory image differs"
        );
    }
}

#[test]
fn mubuf_load_samples_memory_at_retire() {
    let mut r = mb_rig(0, 0x400);
    r.set_v(2, |_| 0x20);
    r.ok(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    r.put(G + 0x20, 0x7777_0001);
    r.wait(0, 15).unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], 0x7777_0001)
    }
}

#[test]
fn mubuf_descriptor_base_soffset_and_inline_soffset() {
    // Base from the descriptor (48-bit), soffset from an SGPR, immediate offset, no vaddr.
    let mut r = mb_rig(0, 0x400);
    r.w.s[8] = 0x20;
    r.run(&mb(MubufOp::LoadDword, 10, None, 4, s(8), 4, false, false));
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], pat_u32(0x24), "lane {l}")
    }
    // Inline-constant soffset.
    let mut r = mb_rig(0, 0x400);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        None,
        4,
        Src::Int(16),
        4,
        false,
        false,
    ));
    assert_eq!(r.w.v[10][0], pat_u32(20));
    // Descriptor at a different SGPR group with a high base-address half.
    let mut r = Rig::new();
    r.map_pat("hi", 0x1234_8000_0000, 0x100);
    r.set_desc(12, 0x1234_8000_0000, 0, 0x100, W3);
    r.w.s[8] = 0;
    r.run(&mb(MubufOp::LoadDword, 10, None, 12, s(8), 8, false, false));
    assert_eq!(r.w.v[10][0], pat_u32(8));
    // A raw buffer ignores the descriptor stride.
    let mut r = mb_rig(12, 0x400);
    r.set_v(2, |l| l as u32 * 16);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        4,
        true,
        false,
    ));
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], pat_u32(l * 16 + 4), "lane {l}")
    }
}

#[test]
fn mubuf_raw_bounds_loads_return_zero_out_of_range() {
    let word = |r: &Rig, k: usize, l: usize| r.w.v[10 + k][l];
    // Per lane: dword at voffset = 4*lane against 16 bytes.
    let mut r = mb_rig(0, 16);
    r.set_v(2, |l| l as u32 * 4);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    for l in 0..WAVE {
        assert_eq!(
            word(&r, 0, l),
            if l * 4 < 16 { pat_u32(l * 4) } else { 0 },
            "lane {l}"
        )
    }
    // The immediate offset counts towards the check.
    let mut r = mb_rig(0, 16);
    r.set_v(2, |l| l as u32 * 4);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        8,
        true,
        false,
    ));
    for l in 0..WAVE {
        assert_eq!(
            word(&r, 0, l),
            if l * 4 + 8 < 16 {
                pat_u32(l * 4 + 8)
            } else {
                0
            },
            "lane {l}"
        )
    }
    // Without offen only the immediate offset counts.
    let mut r = mb_rig(0, 16);
    r.run(&mb(MubufOp::LoadDword, 10, None, 4, s(8), 12, false, false));
    assert_eq!(word(&r, 0, 0), pat_u32(12));
    let mut r = mb_rig(0, 16);
    r.run(&mb(MubufOp::LoadDword, 10, None, 4, s(8), 16, false, false));
    assert_eq!(word(&r, 0, 0), 0);
    // dwordx4 at 8 of 16 bytes: per-component checks, tail returns zero (not SENT).
    let mut r = mb_rig(0, 16);
    r.set_v(2, |_| 8);
    r.run(&mb(
        MubufOp::LoadDwordx4,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    for l in 0..WAVE {
        assert_eq!(
            [
                word(&r, 0, l),
                word(&r, 1, l),
                word(&r, 2, l),
                word(&r, 3, l)
            ],
            [pat_u32(8), pat_u32(12), 0, 0],
            "lane {l}"
        )
    }
    assert_eq!(word(&r, 4, 0), SENT, "register beyond the tuple untouched");
    // Byte and short extensions at the edge.
    for (op, off, want) in [
        (MubufOp::LoadUbyte, 15u32, u32::from(pat(15))),
        (MubufOp::LoadUbyte, 16, 0),
        (MubufOp::LoadSbyte, 16, 0),
        (MubufOp::LoadUshort, 14, u32::from(pat_u16(14))),
        (MubufOp::LoadUshort, 16, 0),
        (MubufOp::LoadSshort, 16, 0),
    ] {
        let mut r = mb_rig(0, 16);
        r.set_v(2, move |_| off);
        r.run(&mb(op, 10, Some(2), 4, s(8), 0, true, false));
        assert_eq!(word(&r, 0, 0), want, "{op:?} @{off}");
    }
}

#[test]
fn mubuf_raw_bounds_check_component_start_not_whole_width() {
    let word = |r: &Rig, k: usize| r.w.v[10 + k][0];
    // num_records = 14: the dword at 12 starts in range and is read whole.
    let mut r = mb_rig(0, 14);
    r.set_v(2, |_| 12);
    r.run(&mb(
        MubufOp::LoadDwordx4,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!(
        [word(&r, 0), word(&r, 1), word(&r, 2), word(&r, 3)],
        [pat_u32(12), 0, 0, 0]
    );
    let mut r = mb_rig(0, 14);
    r.set_v(2, |_| 8);
    r.run(&mb(
        MubufOp::LoadDwordx2,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!([word(&r, 0), word(&r, 1)], [pat_u32(8), pat_u32(12)]);
    let mut r = mb_rig(0, 14);
    r.set_v(2, |_| 12);
    r.run(&mb(
        MubufOp::LoadDwordx2,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!([word(&r, 0), word(&r, 1)], [pat_u32(12), 0]);
    // A dword starting at 16 is out of range.
    let mut r = mb_rig(0, 14);
    r.set_v(2, |_| 16);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!(word(&r, 0), 0);
}

#[test]
fn mubuf_raw_bounds_exclude_soffset() {
    let mut r = mb_rig(0, 16);
    r.w.s[8] = 32;
    r.set_v(2, |_| 0);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        12,
        true,
        false,
    ));
    assert_eq!(
        r.w.v[10][0],
        pat_u32(44),
        "soffset 32 + offset 12 is in range of 16 records"
    );
    let mut r = mb_rig(0, 16);
    r.w.s[8] = 32;
    r.set_v(2, |_| 16);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!(
        r.w.v[10][0], 0,
        "voffset 16 is out of range whatever soffset is"
    );
}

#[test]
fn mubuf_raw_bounds_stores_dropped_out_of_range() {
    let fresh = |nrec: u32| {
        let mut r = Rig::new();
        r.map_fill("g", G, 0x100, 0xEE);
        r.set_desc(4, G, 0, nrec, W3);
        r.w.s[8] = 0;
        r.w.exec = 1;
        for k in 0..4 {
            r.set_v(6 + k, |l| val(l, k))
        }
        r
    };
    // dwordx4 at 8 of 16 bytes: components 8 and 12 land, 16 and 20 are dropped.
    let mut r = fresh(16);
    r.set_v(2, |_| 8);
    r.run(&mb(
        MubufOp::StoreDwordx4,
        6,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    let b = r.bytes("g");
    assert_eq!(&b[8..12], &val(0, 0).to_le_bytes());
    assert_eq!(&b[12..16], &val(0, 1).to_le_bytes());
    assert!(
        b[16..24].iter().all(|&x| x == 0xEE),
        "components past num_records must be dropped"
    );
    assert!(b[..8].iter().all(|&x| x == 0xEE));
    // A dword store starting at 16 is dropped entirely.
    let mut r = fresh(16);
    r.set_v(2, |_| 16);
    r.run(&mb(
        MubufOp::StoreDword,
        6,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert!(r.bytes("g").iter().all(|&x| x == 0xEE));
    // Byte store at 15 lands, at 16 does not.
    let mut r = fresh(16);
    r.set_v(2, |_| 15);
    r.run(&mb(MubufOp::StoreByte, 6, Some(2), 4, s(8), 0, true, false));
    assert_eq!(r.bytes("g")[15], val(0, 0).to_le_bytes()[0]);
    let mut r = fresh(16);
    r.set_v(2, |_| 16);
    r.run(&mb(MubufOp::StoreByte, 6, Some(2), 4, s(8), 0, true, false));
    assert!(r.bytes("g").iter().all(|&x| x == 0xEE));
    // num_records = 14: the dword at 12 starts in range and is stored whole.
    let mut r = fresh(14);
    r.set_v(2, |_| 12);
    r.run(&mb(
        MubufOp::StoreDword,
        6,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert_eq!(&r.bytes("g")[12..16], &val(0, 0).to_le_bytes());
}

#[test]
fn mubuf_structured_idxen_index_bounds_and_stride() {
    // idxen only: address = base + index * stride + offset; index < num_records.
    let mut r = mb_rig(16, 4);
    r.set_v(2, |l| l as u32);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        8,
        false,
        true,
    ));
    for l in 0..WAVE {
        assert_eq!(
            r.w.v[10][l],
            if l < 4 { pat_u32(l * 16 + 8) } else { 0 },
            "lane {l}"
        )
    }
    // A different stride.
    let mut r = mb_rig(24, 8);
    r.set_v(2, |l| l as u32);
    r.run(&mb(
        MubufOp::LoadDwordx2,
        10,
        Some(2),
        4,
        s(8),
        4,
        false,
        true,
    ));
    for l in 0..WAVE {
        let want = if l < 8 {
            [pat_u32(l * 24 + 4), pat_u32(l * 24 + 8)]
        } else {
            [0, 0]
        };
        assert_eq!([r.w.v[10][l], r.w.v[11][l]], want, "lane {l}");
    }
    // idxen + offen: the 64-bit vaddr is {index, offset}, index in the first VGPR.
    // Only the index is bounds-checked, even with a large offset.
    let mut r = mb_rig(16, 4);
    r.set_v(2, |l| (l % 8) as u32);
    r.set_v(3, |l| 0x40 + 4 * (l % 3) as u32);
    r.run(&mb(MubufOp::LoadDword, 10, Some(2), 4, s(8), 4, true, true));
    for l in 0..WAVE {
        let (idx, off) = (l % 8, 0x40 + 4 * (l % 3));
        assert_eq!(
            r.w.v[10][l],
            if idx < 4 {
                pat_u32(idx * 16 + off + 4)
            } else {
                0
            },
            "lane {l}"
        );
    }
    // soffset is added after the structured offset and is not bounds-checked.
    let mut r = mb_rig(16, 4);
    r.w.s[8] = 0x100;
    r.set_v(2, |l| (l % 8) as u32);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        false,
        true,
    ));
    for l in 0..WAVE {
        assert_eq!(
            r.w.v[10][l],
            if l % 8 < 4 {
                pat_u32(0x100 + (l % 8) * 16)
            } else {
                0
            },
            "lane {l}"
        )
    }
}

#[test]
fn mubuf_structured_stores_and_order_of_idxen_offen() {
    let mut r = Rig::new();
    r.map_fill("g", G, 0x400, 0xEE);
    r.set_desc(4, G, 16, 4, W3);
    r.w.s[8] = 0;
    r.w.exec = 0xFF; // eight lanes with distinct indices: no two lanes store to one address
    r.set_v(2, |l| l as u32); // index
    r.set_v(3, |l| 4 * (l % 2) as u32); // offset
    r.set_v(6, |l| val(l, 0));
    r.run(&mb(MubufOp::StoreDword, 6, Some(2), 4, s(8), 8, true, true));
    let mut want = vec![0xEEu8; 0x400];
    for l in 0..8usize {
        let (idx, off) = (l % 8, 4 * (l % 2));
        if idx < 4 {
            let a = idx * 16 + off + 8;
            want[a..a + 4].copy_from_slice(&val(l, 0).to_le_bytes())
        }
    }
    assert!(
        r.bytes("g") == want.as_slice(),
        "structured store image differs"
    );
    // Swapped roles would address index*16 from v3: prove v2 is the index by a one-lane case.
    let mut r = Rig::new();
    r.map_pat("g", G, 0x400);
    r.set_desc(4, G, 32, 100, W3);
    r.w.s[8] = 0;
    r.w.exec = 1;
    r.set_v(2, |_| 3);
    r.set_v(3, |_| 5 * 4);
    r.fill_v(10, 1, SENT);
    r.run(&mb(MubufOp::LoadDword, 10, Some(2), 4, s(8), 0, true, true));
    assert_eq!(r.w.v[10][0], pat_u32(3 * 32 + 20));
}

#[test]
fn mubuf_masked_exec() {
    let mask = 0x00FF_0F0F_F0F0_AAAAu64;
    let mut r = mb_rig(0, 0x400);
    r.set_v(2, |l| {
        if mask >> l & 1 == 1 {
            l as u32 * 4
        } else {
            0x8000_0000
        }
    });
    r.w.exec = mask;
    r.ok(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    for l in 0..WAVE {
        let want = if mask >> l & 1 == 1 {
            pat_u32(l * 4)
        } else {
            SENT
        };
        assert_eq!(r.w.v[10][l], want, "load lane {l}");
    }
    let mut r = Rig::new();
    r.map_fill("g", G, 0x400, 0xEE);
    r.set_desc(4, G, 0, 0x400, W3);
    r.w.s[8] = 0;
    r.set_v(2, |l| l as u32 * 4);
    r.set_v(6, |l| val(l, 0));
    r.w.exec = mask;
    r.ok(&mb(
        MubufOp::StoreDword,
        6,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    r.w.exec = u64::MAX;
    r.drain().unwrap();
    for l in 0..WAVE {
        let want = if mask >> l & 1 == 1 {
            val(l, 0)
        } else {
            0xEEEE_EEEE
        };
        assert_eq!(r.word(G + l as u64 * 4), want, "store lane {l}");
    }
    let mut r = mb_rig(0, 0x400);
    r.w.exec = 0;
    r.set_v(2, |_| 0x8000_0000);
    r.try_run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ))
    .unwrap();
}

#[test]
fn mubuf_unmapped_and_unsupported_descriptor_features() {
    // In bounds per the descriptor but unmapped.
    let mut r = mb_rig(0, 0x400);
    r.set_desc(4, 0x7000_0000_0000, 0, 0x400, W3);
    r.w.exec = 1;
    let res = r.try_run(&mb(MubufOp::LoadDword, 10, None, 4, s(8), 0, false, false));
    assert!(is_fault(&res), "{res:?}");
    let mut r = mb_rig(0, 0x400);
    r.set_desc(4, 0x7000_0000_0000, 0, 0x400, W3);
    r.w.exec = 1;
    let res = r.try_run(&mb(MubufOp::StoreDword, 6, None, 4, s(8), 0, false, false));
    assert!(is_fault(&res), "{res:?}");
    // In range of the descriptor, past the end of the mapped region.
    let mut r = mb_rig(0, 0x2000);
    r.w.exec = 1;
    r.set_v(2, |_| 0x1000);
    let res = r.try_run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        true,
        false,
    ));
    assert!(is_fault(&res), "{res:?}");

    // Descriptor swizzle (dword1 bit 31) and add_tid (dword3 bit 23).
    for (name, tweak) in [("swizzle", 1usize), ("add_tid", 3usize)] {
        for (form, offen, idxen) in [
            ("raw", true, false),
            ("structured", false, true),
            ("both", true, true),
        ] {
            let mut r = mb_rig(16, 0x400);
            r.set_v(2, |_| 0);
            r.set_v(3, |_| 0);
            r.w.s[4 + tweak] |= if tweak == 1 { 1 << 31 } else { 1 << 23 };
            let res = r.try_run(&mb(
                MubufOp::LoadDword,
                10,
                Some(2),
                4,
                s(8),
                0,
                offen,
                idxen,
            ));
            assert!(is_unsupported(&res), "{name}/{form} load: {res:?}");
            let mut r = mb_rig(16, 0x400);
            r.set_v(2, |_| 0);
            r.set_v(3, |_| 0);
            r.w.s[4 + tweak] |= if tweak == 1 { 1 << 31 } else { 1 << 23 };
            let res = r.try_run(&mb(
                MubufOp::StoreDword,
                6,
                Some(2),
                4,
                s(8),
                0,
                offen,
                idxen,
            ));
            assert!(is_unsupported(&res), "{name}/{form} store: {res:?}");
        }
    }
}

// ============================================================================
// Counter proofs

fn gstore(r: &mut Rig, addr: u64, data: u32) {
    r.set_v64(2, |_| addr);
    r.fill_v(4, 1, data);
    r.ok(&gl(FlatOp::StoreDword, None, 2, Some(4), 0, false));
}
fn gload(r: &mut Rig, dst: u8, addr: u64) {
    r.set_v64(2, |_| addr);
    r.ok(&gl(FlatOp::LoadDword, Some(dst), 2, None, 0, false));
}

#[test]
fn counter_a_busy_registers_reject_access_with_wait() {
    // GLOBAL load.
    let mut r = Rig::global(0x100);
    r.w.exec = 1;
    r.set_v64(2, |_| G);
    r.set_v64(14, |_| G);
    r.ok(&gl(FlatOp::LoadDwordx2, Some(10), 2, None, 0, false));
    for n in [10u8, 11] {
        let rd = r.w.read_v(v(n), 0, Ty::B32);
        assert!(is_wait(&rd), "read v{n}: {rd:?}");
        let wr = r.w.write_v(v(n), 0, 1);
        assert!(is_wait(&wr), "write v{n}: {wr:?}");
    }
    r.w.write_v(v(12), 0, 1).unwrap();
    assert_eq!(r.w.read_v(v(12), 0, Ty::B32), Ok(1));
    // Instructions that read a busy register fail, and leave nothing queued.
    let before = r.w.pending.len();
    let res = r.issue(&gl(FlatOp::StoreDword, None, 2, Some(11), 0, false)); // busy data
    assert!(is_wait(&res), "store with busy data: {res:?}");
    let res = r.issue(&gl(FlatOp::LoadDword, Some(20), 10, None, 0, false)); // busy address
    assert!(is_wait(&res), "load with busy address: {res:?}");
    r.set_desc(4, G, 0, 0x100, W3);
    let res = r.issue(&mb(MubufOp::StoreDword, 10, None, 4, s(8), 0, false, false)); // busy data
    assert!(is_wait(&res), "mubuf store with busy data: {res:?}");
    assert_eq!(
        r.w.pending.len(),
        before,
        "a rejected issue must not queue an operation"
    );
    r.wait(0, 15).unwrap();
    assert!(r.w.read_v(v(10), 0, Ty::B32).is_ok());
    assert!(r.w.write_v(v(11), 0, 5).is_ok());

    // DS read (b64): both destination dwords.
    let mut r = Rig::new();
    r.set_v(1, |l| l as u32 * 8);
    r.ok(&ds_off(DsOp::ReadB64, Some(10), 1, None, 0));
    for n in [10u8, 11] {
        assert!(is_wait(&r.w.read_v(v(n), 3, Ty::B32)), "ds read v{n}");
        assert!(is_wait(&r.w.write_v(v(n), 3, 0)), "ds write v{n}");
    }
    let res = r.issue(&ds_off(DsOp::WriteB32, None, 1, Some(11), 0));
    assert!(is_wait(&res), "ds write of busy data: {res:?}");
    r.wait(63, 0).unwrap();
    assert!(r.w.read_v(v(11), 3, Ty::B32).is_ok());

    // SMEM load.
    let mut r = smem_rig();
    r.ok(&sm(
        SmemOp::LoadDwordx4,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    ));
    for n in [20u8, 23] {
        assert!(is_wait(&r.w.read_s(s(n), Ty::B32)), "smem read s{n}");
        assert!(is_wait(&r.w.write_s(s(n), 0)), "smem write s{n}");
    }
    assert!(r.w.read_s(s(24), Ty::B32).is_ok());
    // A load whose base is busy.
    let res = r.issue(&sm(
        SmemOp::LoadDword,
        Some(30),
        Some(20),
        SmemOffset::Imm(0),
    ));
    assert!(is_wait(&res), "smem with busy base: {res:?}");
    r.wait(63, 0).unwrap();
    assert!(r.w.read_s(s(20), Ty::B32).is_ok());
}

#[test]
fn counter_b_vmcnt_retires_exactly_the_oldest_mixed_ops() {
    let mut r = Rig::new();
    r.map_fill("g", G, 0x1000, 0);
    r.w.exec = 1;
    let a = |i: u64| G + 0x100 * i;
    r.put(a(1), 0x1111_1111);
    r.put(a(3), 0x3333_3333);
    gstore(&mut r, a(0), 0xAAA0); // op0 store
    gload(&mut r, 10, a(1)); //      op1 load
    gstore(&mut r, a(2), 0xAAA2); // op2 store
    gload(&mut r, 11, a(3)); //      op3 load
    gstore(&mut r, a(4), 0xAAA4); // op4 store
                                  // Unrelated LGKM work must not be touched by VM waits.
    r.set_v(6, |l| l as u32 * 4);
    r.ok(&ds_off(DsOp::ReadB32, Some(12), 6, None, 0));
    r.set_s64(2, G);
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(30),
        Some(2),
        SmemOffset::Imm(0x40),
    ));
    assert_eq!((r.units(Counter::Vm), r.units(Counter::Lgkm)), (5, 2));

    r.wait(5, 15).unwrap(); // already satisfied
    assert_eq!(r.units(Counter::Vm), 5);
    assert_eq!(r.word(a(0)), 0);

    r.wait(4, 15).unwrap(); // exactly op0
    assert_eq!(r.units(Counter::Vm), 4);
    assert_eq!(r.word(a(0)), 0xAAA0);
    assert!(r.busy_v(10) && r.busy_v(11));
    assert_eq!(r.word(a(2)), 0);

    r.wait(2, 15).unwrap(); // op1 (load) and op2 (store)
    assert_eq!(r.units(Counter::Vm), 2);
    assert!(!r.busy_v(10) && r.busy_v(11));
    assert_eq!(r.w.v[10][0], 0x1111_1111);
    assert_eq!(r.word(a(2)), 0xAAA2);
    assert_eq!(r.word(a(4)), 0, "op4 is younger and must still be pending");

    r.wait(1, 15).unwrap(); // op3 (load)
    assert!(!r.busy_v(11));
    assert_eq!(r.w.v[11][0], 0x3333_3333);
    assert_eq!(r.word(a(4)), 0);

    r.wait(0, 15).unwrap(); // op4
    assert_eq!(r.word(a(4)), 0xAAA4);
    assert_eq!(r.units(Counter::Vm), 0);
    assert_eq!(r.units(Counter::Lgkm), 2, "VM waits never retire DS/SMEM");
    assert!(r.busy_v(12) && r.busy_s(30));
    r.wait(63, 0).unwrap();
    assert!(!r.busy_v(12) && !r.busy_s(30));
}

#[test]
fn counter_b_vmcnt_retires_in_issue_order_for_atomics_and_mubuf() {
    let mut r = Rig::new();
    r.map_fill("g", G, 0x1000, 0);
    r.set_desc(4, G, 0, 0x1000, W3);
    r.w.s[8] = 0;
    r.w.exec = 1;
    r.put(G + 0x200, 10);
    r.put(G + 0x300, 0x0DD0);
    r.set_v64(2, |_| G + 0x200);
    r.fill_v(6, 1, 5);
    r.ok(&gl(FlatOp::AtomicAdd, Some(10), 2, Some(6), 0, true)); // op0: returns 10
    r.set_v(7, |_| 0x300);
    r.ok(&mb(
        MubufOp::LoadDword,
        11,
        Some(7),
        4,
        s(8),
        0,
        true,
        false,
    )); // op1
    r.fill_v(8, 1, 0xBEEF);
    r.ok(&mb(
        MubufOp::StoreDword,
        8,
        None,
        4,
        s(8),
        0x40,
        false,
        false,
    )); // op2
    assert_eq!(r.units(Counter::Vm), 3);
    r.wait(2, 15).unwrap();
    assert_eq!((r.w.v[10][0], r.word(G + 0x200)), (10, 15));
    assert!(r.busy_v(11));
    r.wait(1, 15).unwrap();
    assert_eq!(r.w.v[11][0], 0x0DD0);
    assert_eq!(r.word(G + 0x40), 0);
    r.wait(0, 15).unwrap();
    assert_eq!(r.word(G + 0x40), 0xBEEF);
}

#[test]
fn counter_c_barrier_does_not_drain_stores_two_waves_sharing_lds() {
    let cfg = Config::default();
    let mut lds = vec![0u8; 4096];
    let mut m = Memory::default();
    m.map("g", G, vec![0u8; 0x100]).unwrap();
    let mut a = Wave::new(Arch::Gfx942, 0, 0);
    let mut b = Wave::new(Arch::Gfx942, 1, 0);
    a.exec = 1;
    b.exec = 1;
    a.v[1][0] = 0x100; // LDS address
    a.v[4][0] = 0xA1A1_A1A1;
    a.v[2][0] = G as u32; // global address
    a.v[2 + 1][0] = (G >> 32) as u32;
    a.v[5][0] = 0xB2B2_B2B2;
    b.v[1][0] = 0x100;
    b.v[2][0] = G as u32;
    b.v[3][0] = (G >> 32) as u32;

    let ds_store = ds_off(DsOp::WriteB32, None, 1, Some(4), 0);
    let g_store = gl(FlatOp::StoreDword, None, 2, Some(5), 0, false);
    let ds_load = ds_off(DsOp::ReadB32, Some(10), 1, None, 0);
    let g_load = gl(FlatOp::LoadDword, Some(11), 2, None, 0, false);

    // Wave A stores to LDS and global, then reaches the barrier without waiting.
    assert_eq!(
        exec_one(&mut a, &ds_store, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut a, &g_store, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut a, &barrier(), &mut lds, &mut m, &cfg),
        Ok(Flow::Barrier)
    );
    assert_eq!(
        a.pending.len(),
        2,
        "s_barrier must not drain outstanding stores"
    );

    // Wave B, past the barrier, reads and waits: A's stores are invisible.
    assert_eq!(
        exec_one(&mut b, &barrier(), &mut lds, &mut m, &cfg),
        Ok(Flow::Barrier)
    );
    assert_eq!(
        exec_one(&mut b, &ds_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut b, &g_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    simmem::wait(&mut b, &mut lds, &mut m, 0, 0, 7).unwrap();
    assert_eq!(
        b.v[10][0], 0,
        "LDS store not yet retired: B must see the old value"
    );
    assert_eq!(
        b.v[11][0], 0,
        "global store not yet retired: B must see the old value"
    );

    // A waits; now the stores are visible.
    simmem::wait(&mut a, &mut lds, &mut m, 0, 0, 7).unwrap();
    assert!(a.pending.is_empty());
    assert_eq!(&lds[0x100..0x104], &0xA1A1_A1A1u32.to_le_bytes());
    assert_eq!(
        exec_one(&mut b, &ds_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut b, &g_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    simmem::wait(&mut b, &mut lds, &mut m, 0, 0, 7).unwrap();
    assert_eq!(b.v[10][0], 0xA1A1_A1A1);
    assert_eq!(b.v[11][0], 0xB2B2_B2B2);
}

#[test]
fn counter_d_ds_load_before_barrier_sees_late_store_war() {
    let cfg = Config::default();
    let mut lds = vec![0u8; 4096];
    let mut m = Memory::default();
    m.map("g", G, vec![0u8; 0x100]).unwrap();
    m.write(G, &0x0101_0101u32.to_le_bytes()).unwrap();
    lds[0x100..0x104].copy_from_slice(&0x0202_0202u32.to_le_bytes());
    let mut a = Wave::new(Arch::Gfx942, 0, 0);
    let mut b = Wave::new(Arch::Gfx942, 1, 0);
    for w in [&mut a, &mut b] {
        w.exec = 1;
        w.v[1][0] = 0x100;
        w.v[2][0] = G as u32;
        w.v[3][0] = (G >> 32) as u32;
    }
    a.v[4][0] = 0xFEED_0001;
    a.v[5][0] = 0xFEED_0002;
    let ds_load = ds_off(DsOp::ReadB32, Some(10), 1, None, 0);
    let g_load = gl(FlatOp::LoadDword, Some(11), 2, None, 0, false);

    // B issues its loads and does NOT wait before the barrier.
    assert_eq!(
        exec_one(&mut b, &ds_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut b, &g_load, &mut lds, &mut m, &cfg),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(&mut b, &barrier(), &mut lds, &mut m, &cfg),
        Ok(Flow::Barrier)
    );
    // After the barrier A overwrites both locations and waits for its stores.
    assert_eq!(
        exec_one(&mut a, &barrier(), &mut lds, &mut m, &cfg),
        Ok(Flow::Barrier)
    );
    assert_eq!(
        exec_one(
            &mut a,
            &ds_off(DsOp::WriteB32, None, 1, Some(4), 0),
            &mut lds,
            &mut m,
            &cfg
        ),
        Ok(Flow::Next)
    );
    assert_eq!(
        exec_one(
            &mut a,
            &gl(FlatOp::StoreDword, None, 2, Some(5), 0, false),
            &mut lds,
            &mut m,
            &cfg
        ),
        Ok(Flow::Next)
    );
    simmem::wait(&mut a, &mut lds, &mut m, 0, 0, 7).unwrap();
    // B's loads sample at retire: they observe the later stores (the WAR hazard is exposed).
    simmem::wait(&mut b, &mut lds, &mut m, 0, 0, 7).unwrap();
    assert_eq!(
        b.v[10][0], 0xFEED_0001,
        "DS load not waited before the barrier must observe the later store"
    );
    assert_eq!(b.v[11][0], 0xFEED_0002, "global load likewise");
}

/// Common setup of the LGKM tests: pattern LDS, pattern global region, base in `s[2:3]`.
fn lgkm_rig() -> Rig {
    let mut r = Rig::new();
    r.map_pat("g", G, 0x1000);
    r.set_s64(2, G);
    r.set_v(6, |l| l as u32 * 4);
    r
}

#[test]
fn counter_e_partial_lgkm_waits_retire_ds_only_and_smem_only_at_zero() {
    let mut r = lgkm_rig();
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    )); //   S_a  weight 1
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 6, None, 0)); //                    D1   weight 1
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(22),
        Some(2),
        SmemOffset::Imm(8),
    )); // S_b  weight 2
    r.ok(&ds_off(DsOp::ReadB32, Some(11), 6, None, 4)); //                    D3   weight 1
                                                        // A VM store that LGKM waits must not touch.
    r.w.exec = 1;
    gstore(&mut r, G + 0x800, 0xC0FF_EE00);
    assert_eq!(r.units(Counter::Lgkm), 5);

    r.wait(63, 4).unwrap(); // retire only the oldest DS
    assert_eq!(r.units(Counter::Lgkm), 4);
    assert!(!r.busy_v(10) && r.busy_v(11) && r.busy_s(20) && r.busy_s(22));
    assert_eq!(r.w.v[10][5], pat_u32(5 * 4));

    r.wait(63, 3).unwrap(); // the other DS; SMEM (weights 1 + 2 = 3) is within the threshold
    assert_eq!(r.units(Counter::Lgkm), 3);
    assert!(!r.busy_v(11) && r.busy_s(20) && r.busy_s(22));
    assert_eq!(r.w.v[11][5], pat_u32(5 * 4 + 4));

    // Threshold 2 would need SMEM to retire early: not achievable.
    let res = r.wait(63, 2);
    assert!(is_wait(&res), "{res:?}");
    assert!(
        r.busy_s(20) && r.busy_s(22),
        "SMEM must not retire before lgkmcnt(0)"
    );
    assert_eq!(r.units(Counter::Lgkm), 3);
    assert_eq!(
        r.word(G + 0x800),
        pat_u32(0x800),
        "lgkm waits do not retire VM operations"
    );
    assert_eq!(r.units(Counter::Vm), 1);

    r.wait(63, 0).unwrap();
    assert_eq!(r.units(Counter::Lgkm), 0);
    assert_eq!(
        (r.w.s[20], r.w.s[22], r.w.s[23]),
        (pat_u32(0), pat_u32(8), pat_u32(12))
    );
    assert_eq!(r.units(Counter::Vm), 1, "the VM store is still pending");
}

#[test]
fn counter_e_smem_only_waits_succeed_only_at_zero_or_within_threshold() {
    let mut r = lgkm_rig();
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    ));
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(22),
        Some(2),
        SmemOffset::Imm(8),
    ));
    assert_eq!(r.units(Counter::Lgkm), 3);
    r.wait(63, 15).unwrap();
    r.wait(63, 3).unwrap(); // already within the threshold: nothing retires
    assert!(r.busy_s(20) && r.busy_s(22));
    for n in [2u8, 1] {
        let res = r.wait(63, n);
        assert!(
            is_wait(&res),
            "lgkmcnt({n}) with only SMEM pending: {res:?}"
        );
        assert!(r.busy_s(20) && r.busy_s(22));
    }
    r.wait(63, 0).unwrap();
    assert!(!r.busy_s(20) && !r.busy_s(22));
    assert_eq!(
        (r.w.s[20], r.w.s[22], r.w.s[23]),
        (pat_u32(0), pat_u32(8), pat_u32(12))
    );
}

#[test]
fn counter_e_partial_wait_that_ds_cannot_satisfy_is_wait_error() {
    let mut r = lgkm_rig();
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    )); // 1
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(22),
        Some(2),
        SmemOffset::Imm(8),
    )); // 2
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 6, None, 0)); // 1: total 4
    let res = r.wait(63, 2); // DS retires to 3 > 2, SMEM may not
    assert!(is_wait(&res), "{res:?}");
    let res = r.wait(63, 0);
    assert!(res.is_ok(), "{res:?}");
    assert_eq!(r.units(Counter::Lgkm), 0);
}

#[test]
fn counter_e_zero_wait_retires_ds_first_then_smem_in_reverse_issue_order() {
    // S_old faults at retire (unmapped); S_young and the DS load are valid.
    // If lgkmcnt(0) retires DS first and then SMEM youngest-first, the DS
    // destination and S_young's registers are written before S_old faults.
    let mut r = lgkm_rig();
    r.set_s64(4, 0xdead_0000_0000);
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(20),
        Some(4),
        SmemOffset::Imm(0),
    )); //   S_old: faults
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 6, None, 0)); //                    DS
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(22),
        Some(2),
        SmemOffset::Imm(16),
    )); // S_young
    let res = r.wait(63, 0);
    assert!(is_fault(&res), "{res:?}");
    assert_eq!(
        r.w.v[10][7],
        pat_u32(7 * 4),
        "DS must retire before any SMEM"
    );
    assert_eq!(
        (r.w.s[22], r.w.s[23]),
        (pat_u32(16), pat_u32(20)),
        "the younger SMEM retires before the older one"
    );
}

#[test]
fn counter_e_drain_retires_everything() {
    let mut r = lgkm_rig();
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    ));
    r.ok(&ds_off(DsOp::ReadB32, Some(10), 6, None, 0));
    r.w.exec = 1;
    gstore(&mut r, G + 0x800, 0x5EED_0001);
    gload(&mut r, 12, G + 0x40);
    r.drain().unwrap();
    assert!(r.w.pending.is_empty());
    assert_eq!(r.word(G + 0x800), 0x5EED_0001);
    assert_eq!(r.w.v[12][0], pat_u32(0x40));
    assert_eq!(r.w.s[20], pat_u32(0));
    assert_eq!(r.w.v[10][1], pat_u32(4));
    r.drain().unwrap(); // idempotent on an empty queue
}

#[test]
fn counter_f_vm_overflow_retires_oldest_regardless_of_kind() {
    let mut r = Rig::new();
    r.map_fill("g", G, 0x4000, 0);
    r.w.exec = 1;
    r.put(G + 0x2000, 0x6666_6666);
    gload(&mut r, 10, G + 0x2000); // op0: the oldest is a load
    for i in 1..63u32 {
        gstore(&mut r, G + 4 * u64::from(i), i + 1)
    }
    assert_eq!(r.units(Counter::Vm), 63);
    assert!(r.busy_v(10));
    assert_eq!(r.word(G + 4), 0);

    gstore(&mut r, G + 4 * 63, 64); // 64th: forces op0 (the load) out
    assert_eq!(r.units(Counter::Vm), 63);
    assert!(!r.busy_v(10));
    assert_eq!(r.w.v[10][0], 0x6666_6666);
    assert_eq!(r.word(G + 4), 0, "op1 is still pending");

    gstore(&mut r, G + 4 * 64, 65); // 65th: forces op1 (a store) out
    assert_eq!(r.units(Counter::Vm), 63);
    assert_eq!(r.word(G + 4), 2);
    assert_eq!(r.word(G + 8), 0, "op2 is still pending");
    assert_eq!(r.w.pending.len(), 63);
    r.drain().unwrap();
    for i in 1..65u32 {
        assert_eq!(r.word(G + 4 * u64::from(i)), i + 1, "store {i}")
    }
}

#[test]
fn counter_f_lgkm_overflow_retires_oldest_of_either_kind_with_smem_weights() {
    // Oldest is SMEM (weight 2): 7 x2 loads + 1 DS = 15; the next DS overflows.
    let mut r = lgkm_rig();
    for i in 0..7u8 {
        r.ok(&sm(
            SmemOp::LoadDwordx2,
            Some(20 + 2 * i),
            Some(2),
            SmemOffset::Imm(8 * i32::from(i)),
        ))
    }
    r.ok(&ds_off(DsOp::ReadB32, Some(40), 6, None, 0));
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.ok(&ds_off(DsOp::ReadB32, Some(41), 6, None, 0));
    assert!(
        !r.busy_s(20) && !r.busy_s(21),
        "oldest SMEM retired by the overflow"
    );
    assert!(r.busy_s(22), "only one operation retires");
    assert_eq!((r.w.s[20], r.w.s[21]), (pat_u32(0), pat_u32(4)));
    assert!(r.busy_v(40) && r.busy_v(41));
    assert_eq!(r.units(Counter::Lgkm), 13 + 1);
    assert_eq!(r.w.pending.len(), 6 + 2);

    // Oldest is DS: DS + 7 x2 SMEM = 15; the next dword SMEM overflows and retires the DS.
    let mut r = lgkm_rig();
    r.ok(&ds_off(DsOp::ReadB32, Some(40), 6, None, 0));
    for i in 0..7u8 {
        r.ok(&sm(
            SmemOp::LoadDwordx2,
            Some(20 + 2 * i),
            Some(2),
            SmemOffset::Imm(8 * i32::from(i)),
        ))
    }
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.ok(&sm(
        SmemOp::LoadDword,
        Some(50),
        Some(2),
        SmemOffset::Imm(0),
    ));
    assert!(!r.busy_v(40), "oldest DS retired by the overflow");
    assert!(r.busy_s(20) && r.busy_s(50));
    assert_eq!(r.units(Counter::Lgkm), 14 + 1);
    assert_eq!(r.w.pending.len(), 7 + 1);

    // One retire is enough: 14 dword SMEM = 14 units, a x2 load needs 2 and retires one weight-1 op.
    let mut r = lgkm_rig();
    for i in 0..14u8 {
        r.ok(&sm(
            SmemOp::LoadDword,
            Some(40 + i),
            Some(2),
            SmemOffset::Imm(4 * i32::from(i)),
        ))
    }
    assert_eq!(r.units(Counter::Lgkm), 14);
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    ));
    assert!(!r.busy_s(40) && r.busy_s(41), "exactly the oldest retired");
    assert_eq!(r.w.s[40], pat_u32(0));
    assert_eq!(r.units(Counter::Lgkm), 13 + 2);

    // Two retires needed: 15 dword SMEM = 15 units, a x2 load pushes to 17 and retires two.
    let mut r = lgkm_rig();
    for i in 0..15u8 {
        r.ok(&sm(
            SmemOp::LoadDword,
            Some(40 + i),
            Some(2),
            SmemOffset::Imm(4 * i32::from(i)),
        ))
    }
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.ok(&sm(
        SmemOp::LoadDwordx2,
        Some(20),
        Some(2),
        SmemOffset::Imm(0),
    ));
    assert!(
        !r.busy_s(40) && !r.busy_s(41) && r.busy_s(42),
        "the two oldest retired"
    );
    assert_eq!((r.w.s[40], r.w.s[41]), (pat_u32(0), pat_u32(4)));
    assert_eq!(r.units(Counter::Lgkm), 13 + 2);
    assert_eq!(r.w.pending.len(), 14);
    r.drain().unwrap();
    assert_eq!(r.w.s[54], pat_u32(14 * 4));

    // 15 DS loads fill the counter; the 16th retires the oldest DS.
    let mut r = lgkm_rig();
    for i in 0..15u8 {
        r.ok(&ds_off(
            DsOp::ReadB32,
            Some(20 + i),
            6,
            None,
            4 * u16::from(i),
        ))
    }
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.ok(&ds_off(DsOp::ReadB32, Some(40), 6, None, 0));
    assert!(!r.busy_v(20) && r.busy_v(21) && r.busy_v(40));
    assert_eq!(r.w.v[20][3], pat_u32(3 * 4));
    assert_eq!(r.units(Counter::Lgkm), 15);
}

// ============================================================================
// DS ds_nop and cross-lane (swizzle / permute / bpermute)

fn ds_nop() -> Inst {
    Inst::Ds(Ds {
        op: DsOp::Nop,
        vdst: None,
        addr: None,
        data0: None,
        data1: None,
        offset0: 0,
        offset1: 0,
        gds: false,
    })
}
/// `ds_swizzle_b32 vdst, vsrc offset:pattern`: the source VGPR is the `addr` field.
fn ds_swizzle(vdst: u8, src: u8, pattern: u16) -> Inst {
    Inst::Ds(Ds {
        op: DsOp::SwizzleB32,
        vdst: Some(v(vdst)),
        addr: Some(v(src)),
        data0: None,
        data1: None,
        offset0: pattern as u8,
        offset1: (pattern >> 8) as u8,
        gds: false,
    })
}
/// `ds_permute_b32` / `ds_bpermute_b32 vdst, vaddr, vdata offset:offset`.
fn ds_perm(op: DsOp, vdst: u8, addr: u8, data: u8, offset: u16) -> Inst {
    ds_off(op, Some(vdst), addr, Some(data), offset)
}

/// Source lane of destination lane `i` under swizzle `pattern` (bitmask and quad-perm modes).
fn swizzle_src(pattern: u16, i: usize) -> usize {
    let p = usize::from(pattern);
    if p >> 15 & 1 == 1 {
        (i & !3) + (p >> (2 * (i & 3)) & 3)
    } else {
        ((((i & 0x1f) & (p & 0x1f)) | (p >> 5 & 0x1f)) ^ (p >> 10 & 0x1f)) | (i & 0x20)
    }
}
fn perm_lane(addr: u32, offset: u16) -> usize {
    (addr.wrapping_add(u32::from(offset)) / 4 % 64) as usize
}

#[test]
fn ds_nop_counts_on_lgkm_and_touches_nothing() {
    let mut r = Rig::new();
    r.fill_v(10, 1, SENT);
    let (v0, s0) = (r.w.v.clone(), r.w.s.clone());
    r.ok(&ds_nop());
    let p = &r.w.pending[0];
    assert_eq!(
        (p.counter, p.weight, p.out_of_order),
        (Counter::Lgkm, 1, false)
    );
    assert!(p.busy.is_empty(), "ds_nop has no destination");
    assert_eq!(r.units(Counter::Lgkm), 1);
    r.wait(63, 0).unwrap();
    assert!(r.w.pending.is_empty());
    assert!(r.w.v == v0 && r.w.s == s0, "ds_nop changes no register");
    assert!(
        r.lds.iter().enumerate().all(|(i, &b)| b == pat(i)),
        "ds_nop changes no LDS byte"
    );
    // It occupies one LGKM unit: the 16th operation forces the oldest out.
    for _ in 0..15 {
        r.ok(&ds_nop())
    }
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.ok(&ds_nop());
    assert_eq!(r.units(Counter::Lgkm), 15);
    r.drain().unwrap();
    // EXEC = 0 does not matter.
    r.w.exec = 0;
    r.try_run(&ds_nop()).unwrap();
}

#[test]
fn ds_swizzle_patterns_snapshot_and_masking() {
    // 0x80e4: quad-perm identity; 0x801b: reverse inside each quad; 0x041f: bitmask xor 1 (swap neighbours).
    for pattern in [0x80e4u16, 0x801b, 0x041f] {
        for mask in [
            u64::MAX,
            0x7777_7777_7777_7777,
            0x5555_5555_5555_5555,
            0xF0F0_3C3C_00FF_FF00,
        ] {
            let mut r = Rig::new();
            r.set_v(4, |l| val(l, 0) | 1); // never zero
            r.fill_v(10, 1, SENT);
            let data = r.w.v[4];
            r.w.exec = mask;
            r.ok(&ds_swizzle(10, 4, pattern));
            let p = &r.w.pending[0];
            assert_eq!(
                (p.counter, p.weight, p.out_of_order),
                (Counter::Lgkm, 1, false)
            );
            assert!(r.busy_v(10), "destination busy until waited");
            // Source and EXEC are snapshotted at issue.
            r.fill_v(4, 1, 0xDEAD_BEEF);
            r.w.exec = u64::MAX;
            r.wait(63, 0).unwrap();
            for l in 0..WAVE {
                let want = if mask >> l & 1 == 0 {
                    SENT // destination lanes outside the issue-time EXEC are not written
                } else {
                    let j = swizzle_src(pattern, l);
                    if mask >> j & 1 == 1 {
                        data[j]
                    } else {
                        0
                    } // an inactive source lane reads zero
                };
                assert_eq!(
                    r.w.v[10][l], want,
                    "pattern {pattern:#06x} mask {mask:#x} lane {l}"
                );
            }
            // The LDS is not touched.
            assert!(r.lds.iter().enumerate().all(|(i, &b)| b == pat(i)));
        }
    }
    // Identity is the identity for a full wave.
    let mut r = Rig::new();
    r.set_v(4, |l| val(l, 3));
    r.run(&ds_swizzle(10, 4, 0x80e4));
    for l in 0..WAVE {
        assert_eq!(r.w.v[10][l], val(l, 3), "lane {l}")
    }
}

#[test]
fn ds_permute_and_bpermute_addressing_snapshot_and_masking() {
    type Addr = fn(usize) -> u32;
    let identity: Addr = |l| l as u32 * 4;
    let reversed: Addr = |l| (63 - l) as u32 * 4;
    let pairs: Addr = |l| (l / 2) as u32 * 4; // collisions: two sources per destination
    let wrapped: Addr = |l| (l as u32 + 70) * 4; // high bits ignored
    for op in [DsOp::PermuteB32, DsOp::BpermuteB32] {
        for (name, addr_fn, offset) in [
            ("identity", identity, 0u16),
            ("reversed", reversed, 0),
            ("pairs", pairs, 0),
            ("shift", identity, 8),
            ("wrapped", wrapped, 0),
        ] {
            for mask in [u64::MAX, 0x7777_7777_7777_7777, 0xF0F0_3C3C_00FF_FF00] {
                let mut r = Rig::new();
                r.set_v(4, |l| val(l, 0) | 1);
                r.set_v(5, |l| {
                    if mask >> l & 1 == 1 {
                        addr_fn(l)
                    } else {
                        0xFFFF_FFFF
                    }
                }); // garbage in inactive lanes
                r.fill_v(10, 1, SENT);
                let (data, addr) = (r.w.v[4], r.w.v[5]);
                r.w.exec = mask;
                r.ok(&ds_perm(op, 10, 5, 4, offset));
                let p = &r.w.pending[0];
                assert_eq!(
                    (p.counter, p.weight, p.out_of_order),
                    (Counter::Lgkm, 1, false),
                    "{op:?}"
                );
                assert!(r.busy_v(10), "{op:?}: destination busy until waited");
                // Address, data and EXEC are snapshotted at issue.
                r.fill_v(4, 1, 0xDEAD_BEEF);
                r.fill_v(5, 1, 0);
                r.w.exec = u64::MAX;
                r.wait(63, 0).unwrap();
                let mut want = [0u32; WAVE];
                if op == DsOp::BpermuteB32 {
                    for l in 0..WAVE {
                        let j = perm_lane(addr[l], offset);
                        want[l] = if mask >> l & 1 == 1 && mask >> j & 1 == 1 {
                            data[j]
                        } else {
                            0
                        };
                    }
                } else {
                    for l in 0..WAVE {
                        if mask >> l & 1 == 1 {
                            want[perm_lane(addr[l], offset)] = data[l]
                        }
                    }
                    for l in 0..WAVE {
                        if mask >> l & 1 == 0 {
                            want[l] = 0
                        }
                    }
                }
                for l in 0..WAVE {
                    let w = if mask >> l & 1 == 0 { SENT } else { want[l] };
                    assert_eq!(r.w.v[10][l], w, "{op:?} {name} mask {mask:#x} lane {l}");
                }
            }
        }
    }
    // Full-wave sanity: bpermute reversed is a pure lane reversal; permute reversed likewise.
    for op in [DsOp::PermuteB32, DsOp::BpermuteB32] {
        let mut r = Rig::new();
        r.set_v(4, |l| val(l, 2));
        r.set_v(5, |l| (63 - l) as u32 * 4);
        r.run(&ds_perm(op, 10, 5, 4, 0));
        for l in 0..WAVE {
            assert_eq!(r.w.v[10][l], val(63 - l, 2), "{op:?} lane {l}")
        }
    }
}

#[test]
fn ds_cross_lane_destination_and_sources_are_busy_checked() {
    let mut r = Rig::new();
    r.set_v(4, |l| val(l, 0));
    r.set_v(5, |l| l as u32 * 4);
    r.ok(&ds_perm(DsOp::BpermuteB32, 10, 5, 4, 0));
    assert!(is_wait(&r.w.read_v(v(10), 0, Ty::B32)));
    assert!(is_wait(&r.w.write_v(v(10), 0, 0)));
    // A second cross-lane op reading the busy destination is rejected.
    let res = r.issue(&ds_swizzle(11, 10, 0x80e4));
    assert!(is_wait(&res), "{res:?}");
    r.wait(63, 0).unwrap();
    r.ok(&ds_swizzle(11, 10, 0x80e4));
    r.wait(63, 0).unwrap();
    for l in 0..WAVE {
        assert_eq!(r.w.v[11][l], val(l, 0))
    }
}

#[test]
fn mubuf_idxen_with_zero_stride_is_unsupported() {
    // ISA §9.1.5.1 p.68 defines the structured range check only for Stride != 0.
    for (form, offen) in [("idxen", false), ("idxen+offen", true)] {
        let mut r = mb_rig(0, 0x400);
        r.set_v(2, |_| 0);
        r.set_v(3, |_| 0);
        let res = r.try_run(&mb(
            MubufOp::LoadDword,
            10,
            Some(2),
            4,
            s(8),
            0,
            offen,
            true,
        ));
        assert!(is_unsupported(&res), "{form} load: {res:?}");
        let mut r = mb_rig(0, 0x400);
        r.set_v(2, |_| 0);
        r.set_v(3, |_| 0);
        let res = r.try_run(&mb(
            MubufOp::StoreDword,
            6,
            Some(2),
            4,
            s(8),
            0,
            offen,
            true,
        ));
        assert!(is_unsupported(&res), "{form} store: {res:?}");
    }
    // A non-zero stride with the same instruction is fine.
    let mut r = mb_rig(4, 0x400);
    r.set_v(2, |_| 1);
    r.run(&mb(
        MubufOp::LoadDword,
        10,
        Some(2),
        4,
        s(8),
        0,
        false,
        true,
    ));
    assert_eq!(r.w.v[10][0], pat_u32(4));
}

#[test]
fn smem_buffer_ignores_swizzle_and_add_tid_flags() {
    // Scalar buffer loads only use base, stride and num_records (ISA §8.2).
    for (word, bit) in [(1usize, 31u32), (3, 23)] {
        let mut r = smem_rig();
        r.set_desc(8, G, 0, 0x1000, W3);
        r.w.s[8 + word] |= 1 << bit;
        r.run(&sm(
            SmemOp::BufferLoadDword,
            Some(48),
            Some(8),
            SmemOffset::Imm(0x10),
        ));
        assert_eq!(
            r.w.s[48],
            pat_u32(0x10),
            "descriptor dword {word} bit {bit}"
        );
    }
}
