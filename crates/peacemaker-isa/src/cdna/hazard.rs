// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA3 software wait states: the dependencies the hardware does not
//! interlock (CDNA3 ISA §4.5 Table 11 p.20-21; §7.5 "Dependency Resolution:
//! Required Independent Instructions" Table 37 p.55-58 for MFMA/accvgpr, pass
//! counts from §7.1.2 Table 28 p.42; §7.2 p.51 for the FP8 converts writing
//! parts of one VGPR), cross-checked against LLVM `GCNHazardRecognizer.cpp`
//! (llvm-project main) for gfx940. [`resolve`] inserts the minimal `s_nop`s
//! (an `s_nop N` provides N+1 wait states; every other instruction provides
//! one) and records why; [`check`] reports what a final instruction list
//! violates. Both are sound across branches: the producer window is joined
//! at labels from the fallthrough and every branch to them (worst case = the
//! shortest distance over all paths), backward edges iterated to a fixpoint.
//!
//! Every rule is a pair (producer, consumer, wait states). Where the ISA
//! guide and LLVM differ, the larger requirement is enforced and the
//! difference is noted on the rule ([`Rule::cite`]):
//! - VALU writes SGPR/VCC → VALU reads it as carry-in: ISA 0, LLVM
//!   (`hasVDecCoExecHazard`, gfx940) 2 — 2 enforced.
//! - VALU writes VGPR → `v_readfirstlane` src0: ISA lists only `v_readlane`,
//!   LLVM both — both enforced (1).
//! - Wide buffer store with an SGPR SOFFSET → VALU overwrites its data: ISA 0,
//!   LLVM gfx940 1 — 1 enforced. LLVM applies the store-data rules to VALU
//!   writers only; the ISA's 1-wait row "write VGPRs holding writedata" is
//!   enforced for every writer.
//! - Dst-sel forwarding (OPSEL/SDWA partial write → consumer): ISA says
//!   "consumes result", LLVM any operand overlap incl. WAW — LLVM enforced.
//! - SGEMM write → XDL read SrcC overlapped (not exact): ISA 2/4/8/16 passes,
//!   LLVM 0 — ISA enforced.
//! - Exact SrcC accumulation with a different MFMA opcode of equal passes:
//!   ISA 0, LLVM (`isSameMFMA`) the overlapped value — LLVM enforced.
//! - DL op write → MFMA read: ISA 3 ("any opcode"), LLVM none — 3 enforced.
//! - MFMA write → VMEM/DS write of the same VGPRs (WAW) and XDL SrcC read →
//!   VMEM/DS write (WAR): LLVM only — enforced.
//! - DGEMM → `v_fma_f64`: LLVM only (2) — enforced.
//! - VALU writes EXEC → MFMA: ISA names `v_cmpx`, LLVM any VALU — any VALU.
//! - Table 37 rows sit on printed pages 55-58 (the DGEMM, `V_CMPX` → MFMA and
//!   SrcC-WAR rows are on p.58).
//!
//! Rows the instruction table cannot trigger are not modelled: `s_setvskip`,
//! `s_rfe`, `s_movrel*`, LDS-DMA / add-TID, export, SMFMAC and the MFMA 4x4
//! variants (the 2-pass values still follow from the generic formulas), and
//! "mixed use of VCC: alias vs SGPR#" (VCC has no SGPR alias in the operand
//! space; the 2-wait "VALU reads SGPR as constant" row covers it anyway).
//!
//! [UNVERIFIED: confirm on MI300X] every wait-state count above is taken from
//! the ISA guide / LLVM, not measured; `s_nop` is emitted with at most 8 wait
//! states each (`s_nop 7`, LLVM's `insertNoops` cap) although the ISA allows
//! `s_nop 15`.
use super::Item;
use crate::hazard::HazardProof;
use peacemaker_cdna::enc::sop::{SopkOp, Sopp, SoppOp};
use peacemaker_cdna::enc::vop::{SdwaSel, Vop1Op, VopExt};
use peacemaker_cdna::enc::vop3::{Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::enc::vop3p::{MaiOp, MfmaType, Vop3pOp};
use peacemaker_cdna::inst::{RegFile, RegSpan};
use peacemaker_cdna::{Arch, Inst, Src};
use std::collections::{BTreeMap, HashMap};

/// Longest window any rule needs (XDL 16 passes → VALU/VMEM: 19).
const WINDOW: u32 = 19;
/// Wait states per emitted `s_nop`.
const NOP_MAX: u32 = 8;
/// Passes in which `resolve` recomputes NOPs from scratch before it only
/// lets them grow (guarantees termination on loops).
const FREE_PASSES: usize = 8;
const PASS_LIMIT: usize = 10_000;

/// MFMA classes of CDNA3 ISA §7.5 p.55 (XDL = I8/F16/BF16 and, as in LLVM
/// `getMAIIsGFX940XDL`, FP8/BF8; SGEMM = F32; DGEMM = F64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Xdl,
    Sgemm,
    Dgemm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    SetregGetreg,
    SetregSetreg,
    SetregVskip,
    VccExecToVcczExecz,
    SgprLaneSelect,
    VccDivFmas,
    StoreDataWrite,
    StoreDataValu,
    StoreDataValuSoffset,
    SgprVmem,
    M0SendmsgGds,
    VgprDpp,
    ExecDpp,
    SgprConst,
    ExecConst,
    ExecRwLane,
    VgprReadlane,
    DstSel,
    Trans,
    Fp8PartialWrite,
    ValuMfma,
    ExecMfma,
    DotSameSrcAB,
    DotOther,
    MfmaSrcC {
        w: Class,
        r: Class,
        passes: u32,
        exact: bool,
        same: bool,
    },
    MfmaSrcAB {
        w: Class,
        passes: u32,
    },
    MfmaValu {
        w: Class,
        passes: u32,
    },
    MfmaMemRead {
        w: Class,
        passes: u32,
    },
    MfmaMemWaw {
        w: Class,
        passes: u32,
    },
    XdlSrcCWar {
        passes: u32,
    },
    DgemmFma64,
}

impl Rule {
    /// The table row this rule enforces.
    fn cite(self) -> String {
        const T11A: &str = "CDNA3 ISA §4.5 Table 11 p.20";
        const T11B: &str = "CDNA3 ISA §4.5 Table 11 p.21";
        match self {
            Self::SetregGetreg => format!("{T11A}: S_SETREG -> S_GETREG same reg (2)"),
            Self::SetregSetreg => format!("{T11A}: S_SETREG -> S_SETREG same reg (2)"),
            Self::SetregVskip => format!("{T11A}: S_SETREG MODE.vskip (bit 28, §3.5 Table 4 p.11) -> any vector op (2)"),
            Self::VccExecToVcczExecz => format!("{T11A}: VALU that sets VCC or EXEC -> VALU that uses EXECZ or VCCZ as a data source (5)"),
            Self::SgprLaneSelect => format!("{T11A}: VALU writes SGPR/VCC -> V_{{READ,WRITE}}LANE using that SGPR/VCC as the lane select (4)"),
            Self::VccDivFmas => format!("{T11A}: VALU writes VCC -> V_DIV_FMAS (4)"),
            Self::StoreDataWrite => format!("{T11A}: VMEM store/atomic with > 64 bits of data -> write VGPRs holding writedata (1)"),
            Self::StoreDataValu => format!("{T11A}: VMEM store/atomic with > 64 bits of data -> VALU writes VGPRs holding writedata (2)"),
            Self::StoreDataValuSoffset => "LLVM GCNHazardRecognizer checkSOFFSETWindowVALUHazardsHelper (gfx940): BUFFER_STORE > 64 bits with SGPR SOFFSET -> VALU writes its data (1; CDNA3 ISA §4.5 Table 11 p.20 says 0)".into(),
            Self::SgprVmem => format!("{T11B}: VALU writes SGPR -> VMEM reads that SGPR (5)"),
            Self::M0SendmsgGds => format!("{T11B}: SALU writes M0 -> GDS, S_SENDMSG (1)"),
            Self::VgprDpp => format!("{T11B}: VALU writes VGPR -> VALU DPP reads that VGPR (2)"),
            Self::ExecDpp => format!("{T11B}: VALU writes EXEC -> VALU DPP op (5)"),
            Self::SgprConst => format!("{T11B}: VALU writes SGPR/VCC -> VALU reads SGPR as constant (2; carry-in 0 per ISA, 2 per LLVM hasVDecCoExecHazard, 2 enforced)"),
            Self::ExecConst => format!("{T11B}: v_cmpx (any VALU EXEC write) -> VALU reads EXEC as constant (2)"),
            Self::ExecRwLane => format!("{T11B}: v_cmpx (any VALU EXEC write) -> v_readlane, v_readfirstlane, v_writelane (4)"),
            Self::VgprReadlane => format!("{T11B}: VALU writes VGPRn -> v_readlane vsrc0 reads VGPRn (1; LLVM adds v_readfirstlane)"),
            Self::DstSel => format!("{T11B}: VALU op which uses OPSEL or SDWA with changes the result's bit position -> VALU op consumes result (1; any operand overlap per LLVM getDstSelForwardingOperand)"),
            Self::Trans => format!("{T11B}: VALU Trans op (Table 12) -> non-trans VALU op consumes result (1)"),
            Self::Fp8PartialWrite => "CDNA3 ISA §7.2 p.51: CVT_*_F32 FP8/BF8 conversions writing halves/bytes of the same destination VGPR need a NOP or other instruction between them (1)".into(),
            Self::ValuMfma => "CDNA3 ISA §7.5 Table 37 p.55: non-DLops VALU write VGPR -> V_MFMA read VGPR (2)".into(),
            Self::ExecMfma => "CDNA3 ISA §7.5 Table 37 p.58: V_CMPX* (any VALU per LLVM) write EXEC -> V_MFMA* (4)".into(),
            Self::DotSameSrcAB => "CDNA3 ISA §7.5 Table 37 p.55: DLops write VGPR -> same DLops opcode reads it as SrcA/B (3)".into(),
            Self::DotOther => "CDNA3 ISA §7.5 Table 37 p.55: DLops write VGPR -> any other opcode reads/writes it (RAW + WAW) (3)".into(),
            Self::MfmaSrcC { w, r, passes, exact, same } => {
                let page = match w { Class::Xdl => 56, Class::Sgemm => 57, Class::Dgemm => 58 };
                let kind = match (exact, same) { (true, true) => "exactly same vDst, same opcode", (true, false) => "same vDst, different opcode (overlapped per LLVM isSameMFMA)", _ => "overlapped with 1st vDst" };
                format!("CDNA3 ISA §7.5 Table 37 p.{page}: {w:?} ({passes} passes) write VGPR -> {r:?} read VGPR as Source C, {kind}")
            }
            Self::MfmaSrcAB { w, passes } => format!("CDNA3 ISA §7.5 Table 37 p.{}: {w:?} ({passes} passes) write VGPR -> V_MFMA read VGPR as SrcA or SrcB", page(w)),
            Self::MfmaValu { w, passes } => format!("CDNA3 ISA §7.5 Table 37 p.{}: {w:?} ({passes} passes) write VGPR -> VALU read/write VGPR (RAW + WAW)", page(w)),
            Self::MfmaMemRead { w, passes } => format!("CDNA3 ISA §7.5 Table 37 p.{}: {w:?} ({passes} passes) write VGPR -> VM, L/GDS, FLAT read VGPR overlapped with 1st vDst", page(w)),
            Self::MfmaMemWaw { w, passes } => format!("LLVM GCNHazardRecognizer checkMAIVALUHazards (gfx940): {w:?} ({passes} passes) write VGPR -> VMEM/DS write of the same VGPR (WAW; CDNA3 ISA §7.5 Table 37 p.{} lists VALU only)", page(w)),
            Self::XdlSrcCWar { passes } => format!("CDNA3 ISA §7.5 Table 37 p.58: XDL ({passes} passes) read VGPR SrcC -> VALU (and, per LLVM, VMEM/DS) write VGPR overlapping 1st SrcC (WAR)"),
            Self::DgemmFma64 => "LLVM GCNHazardRecognizer checkMAIVALUHazards DMFMAToFMA64WaitStates: DGEMM -> v_fma_f64 (2; not in the CDNA3 ISA guide)".into(),
        }
    }
}

fn page(w: Class) -> u32 {
    match w {
        Class::Xdl => 56,
        Class::Sgemm => 57,
        Class::Dgemm => 58,
    }
}

/// An MFMA's class, passes (CDNA3 ISA §7.1.2 Table 28 p.42: cycles / 4)
/// and register operands.
#[derive(Clone, Debug)]
struct Mfma {
    op: MaiOp,
    class: Class,
    passes: u32,
    dst: Option<RegSpan>,
    ab: Vec<RegSpan>,
    c: Option<RegSpan>,
}

/// MFMA cycles, CDNA3 ISA §7.1.2 Table 28 p.42.
fn cycles(op: MaiOp) -> u32 {
    use MaiOp::*;
    match op {
        AccvgprReadB32 | AccvgprWriteB32 => 0,
        MfmaF32_16x16x16Bf16
        | MfmaF32_16x16x16F16
        | MfmaI32_16x16x32I8
        | MfmaF32_16x16x32Fp8Fp8
        | MfmaF32_16x16x32Fp8Bf8
        | MfmaF32_16x16x32Bf8Fp8
        | MfmaF32_16x16x32Bf8Bf8 => 16,
        MfmaF32_32x32x8Bf16
        | MfmaF32_32x32x8F16
        | MfmaI32_32x32x16I8
        | MfmaF32_32x32x16Fp8Fp8
        | MfmaF32_32x32x16Fp8Bf8
        | MfmaF32_32x32x16Bf8Fp8
        | MfmaF32_32x32x16Bf8Bf8 => 32,
        MfmaF32_16x16x4F32 => 32,
        MfmaF32_32x32x2F32 => 64,
        MfmaF64_16x16x4F64 => 32,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Readlane,
    Readfirstlane,
    Writelane,
}

/// What the rules need to know about one instruction.
#[derive(Clone, Debug, Default)]
struct Info {
    /// Wait states the instruction itself provides.
    ws: u32,
    /// VALU other than MFMA (accvgpr moves and DL ops included).
    valu: bool,
    mfma: Option<Mfma>,
    vmem: bool,
    ds: bool,
    salu: bool,
    dot: Option<Vop3pOp>,
    dot_src: [Option<RegSpan>; 3],
    dpp: bool,
    trans: bool,
    dst_sel: bool,
    fp8_cvt: bool,
    lane: Option<(Lane, Option<RegSpan>, Option<RegSpan>)>,
    div_fmas: bool,
    fma64: bool,
    explicit_exec: bool,
    vccz_execz: bool,
    setreg: Option<(u16, bool)>,
    getreg: Option<u16>,
    sendmsg_or_gds: bool,
    wide_store: Option<(RegSpan, bool)>,
    vdefs: Vec<RegSpan>,
    vuses: Vec<RegSpan>,
    sdefs: Vec<RegSpan>,
    suses: Vec<RegSpan>,
    exec_def: bool,
    m0_def: bool,
}

impl Info {
    fn vector(&self) -> bool {
        self.valu || self.mfma.is_some() || self.vmem || self.ds
    }
    fn any_valu(&self) -> bool {
        self.valu || self.mfma.is_some()
    }
    fn writes_vcc(&self) -> bool {
        self.sdefs.iter().any(|s| s.file == RegFile::Vcc)
    }
    /// Vector registers read, MFMA operands included.
    fn vreads(&self) -> Vec<RegSpan> {
        let mut v = self.vuses.clone();
        if let Some(m) = &self.mfma {
            v.extend(m.ab.iter().copied());
            v.extend(m.c)
        }
        v
    }
    fn vwrites(&self) -> Vec<RegSpan> {
        let mut v = self.vdefs.clone();
        if let Some(d) = self.mfma.as_ref().and_then(|m| m.dst) {
            v.push(d)
        }
        v
    }
}

fn hit(a: &[RegSpan], b: &[RegSpan]) -> bool {
    a.iter().any(|x| b.iter().any(|y| x.overlaps(*y)))
}

fn is_trans(op: Vop1Op) -> bool {
    use Vop1Op::*;
    matches!(
        op,
        ExpF32 | LogF32 | RcpF32 | RcpIflagF32 | RsqF32 | SqrtF32
    )
}

fn info(inst: &Inst) -> Info {
    let fp = inst.footprint();
    let mut i = Info {
        ws: 1,
        ..Info::default()
    };
    for d in &fp.defs {
        match d.file {
            RegFile::Vgpr | RegFile::Agpr => i.vdefs.push(*d),
            RegFile::Exec => i.exec_def = true,
            RegFile::Scc => {}
            RegFile::M0 => {
                i.m0_def = true;
                i.sdefs.push(*d)
            }
            _ => i.sdefs.push(*d),
        }
    }
    for u in &fp.uses {
        match u.file {
            RegFile::Vgpr | RegFile::Agpr => i.vuses.push(*u),
            RegFile::Exec | RegFile::Scc => {}
            _ => i.suses.push(*u),
        }
    }
    let span = |s: Option<Src>| s.and_then(|s| RegSpan::of(s, 1));
    let mut srcs: Vec<Src> = Vec::new();
    let ext = |i: &mut Info, ext: VopExt, has_dst_sel: bool| match ext {
        VopExt::Dpp(_) => i.dpp = true,
        VopExt::Sdwa(s) => {
            if has_dst_sel && s.dst_sel != SdwaSel::Dword {
                i.dst_sel = true
            }
        }
        VopExt::None => {}
    };
    match inst {
        Inst::Sopp(p) => match p.op {
            SoppOp::Nop => i.ws = u32::from(p.simm16 & 0xf) + 1,
            SoppOp::Sendmsg => i.sendmsg_or_gds = true,
            _ => {}
        },
        Inst::Sop1(_) | Inst::Sop2(_) | Inst::Sopc(_) => i.salu = true,
        Inst::Sopk(k) => {
            i.salu = true;
            let id = k.simm16 & 0x3f;
            match k.op {
                SopkOp::SetregB32 => {
                    let offset = (k.simm16 >> 6) & 0x1f;
                    let size = ((k.simm16 >> 11) & 0x1f) + 1;
                    // HW_REG_MODE = 1 (CDNA3 ISA §5.8 Table 19 p.29); VSKIP is MODE bit 28.
                    i.setreg = Some((id, id == 1 && offset <= 28 && 28 < offset + size));
                }
                SopkOp::GetregB32 => i.getreg = Some(id),
                _ => {}
            }
        }
        Inst::Smem(_) => {}
        Inst::Vop1(v) => {
            i.valu = true;
            srcs.push(v.src0);
            ext(&mut i, v.ext, true);
            i.trans = is_trans(v.op);
            if v.op == Vop1Op::ReadfirstlaneB32 {
                i.lane = Some((Lane::Readfirstlane, None, span(Some(v.src0))))
            }
        }
        Inst::Vop2(v) => {
            i.valu = true;
            srcs.extend([v.src0, v.src1]);
            ext(&mut i, v.ext, true)
        }
        Inst::Vopc(v) => {
            i.valu = true;
            srcs.extend([v.src0, v.src1]);
            ext(&mut i, v.ext, false)
        }
        Inst::Vop3(v) => {
            i.valu = true;
            srcs.extend(v.src.iter().flatten().copied());
            if v.op_sel & 0b1000 != 0 {
                i.dst_sel = true
            }
            match v.op {
                Vop3Op::V1(op) => {
                    i.trans = is_trans(op);
                    if op == Vop1Op::ReadfirstlaneB32 {
                        i.lane = Some((Lane::Readfirstlane, None, span(v.src[0])))
                    }
                }
                Vop3Op::Only(op) => match op {
                    Vop3OnlyOp::ReadlaneB32 => {
                        i.lane = Some((Lane::Readlane, span(v.src[1]), span(v.src[0])))
                    }
                    Vop3OnlyOp::WritelaneB32 => {
                        i.lane = Some((Lane::Writelane, span(v.src[1]), None))
                    }
                    Vop3OnlyOp::DivFmasF32 => i.div_fmas = true,
                    Vop3OnlyOp::FmaF64 => i.fma64 = true,
                    Vop3OnlyOp::CvtPkFp8F32 | Vop3OnlyOp::CvtPkBf8F32 => i.fp8_cvt = true,
                    Vop3OnlyOp::CvtSrFp8F32 | Vop3OnlyOp::CvtSrBf8F32 => {
                        i.fp8_cvt = true;
                        if v.op_sel & 0b0100 != 0 {
                            i.dst_sel = true
                        }
                    }
                    _ => {}
                },
                Vop3Op::C(_) | Vop3Op::V2(_) => {}
            }
        }
        Inst::Vop3p(v) => {
            use Vop3pOp::*;
            i.valu = true;
            srcs.extend(v.src.iter().flatten().copied());
            match v.op {
                Dot2F32F16 | Dot2I32I16 | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4
                | Dot8U32U4 => {
                    i.dot = Some(v.op);
                    i.dot_src = v.src.map(span);
                }
                // vdst_in tied: writes one half and keeps the other (LLVM type 2b).
                FmaMixloF16 | FmaMixhiF16 => i.dst_sel = true,
                _ => {}
            }
        }
        Inst::Mai(m) => match m.op.shape() {
            None => {
                i.valu = true;
                srcs.push(m.src_a)
            }
            Some(s) => {
                let class = match s.a {
                    MfmaType::F32 => Class::Sgemm,
                    MfmaType::F64 => Class::Dgemm,
                    _ => Class::Xdl,
                };
                let mut ab: Vec<RegSpan> = RegSpan::of(m.src_a, s.a_dwords).into_iter().collect();
                ab.extend(m.src_b.and_then(|b| RegSpan::of(b, s.b_dwords)));
                i.mfma = Some(Mfma {
                    op: m.op,
                    class,
                    passes: cycles(m.op) / 4,
                    dst: RegSpan::of(m.dst, s.cd_dwords),
                    ab,
                    c: m.src_c.and_then(|c| RegSpan::of(c, s.cd_dwords)),
                });
            }
        },
        Inst::Ds(d) => {
            i.ds = true;
            if d.gds {
                i.sendmsg_or_gds = true
            }
        }
        Inst::Flat(f) => {
            i.vmem = true;
            let (vdst, data) = f.op.widths();
            if vdst == 0 && data > 2 {
                i.wide_store = f
                    .data
                    .and_then(|d| RegSpan::of(d, data))
                    .map(|s| (s, false))
            }
        }
        Inst::Mubuf(m) => {
            i.vmem = true;
            let n = m.op.data_dwords();
            if m.op.is_store() && n > 2 {
                let sgpr_soffset = RegSpan::of(m.soffset, 1).is_some()
                    && !matches!(m.soffset, Src::Scc | Src::Vccz | Src::Execz);
                i.wide_store = RegSpan::of(m.vdata, n).map(|s| (s, sgpr_soffset));
            }
        }
    }
    i.explicit_exec = srcs.iter().any(|s| matches!(s, Src::ExecLo | Src::ExecHi));
    i.vccz_execz = srcs.iter().any(|s| matches!(s, Src::Vccz | Src::Execz));
    i
}

fn srcc(w: &Mfma, r: &Mfma, exact: bool) -> u32 {
    let same = w.op == r.op;
    let full = if w.passes == 2 { 2 } else { 0 };
    match w.class {
        Class::Xdl => {
            if exact && same {
                full
            } else {
                w.passes + 1
            }
        }
        Class::Sgemm => {
            if exact && same {
                full
            } else if r.class == Class::Xdl {
                if exact {
                    0
                } else {
                    w.passes
                }
            } else {
                w.passes
            }
        }
        Class::Dgemm => {
            if (exact && same) || r.class == Class::Xdl {
                0
            } else {
                9
            }
        }
    }
}

/// Every `(wait states, rule)` producer `p` imposes on a later consumer `c`.
fn rules(p: &Info, c: &Info, out: &mut Vec<(u32, Rule)>) {
    if let Some((id, vskip)) = p.setreg {
        if c.getreg == Some(id) {
            out.push((2, Rule::SetregGetreg))
        }
        if c.setreg.map(|s| s.0) == Some(id) {
            out.push((2, Rule::SetregSetreg))
        }
        if vskip && c.vector() {
            out.push((2, Rule::SetregVskip))
        }
    }
    if p.salu && p.m0_def && c.sendmsg_or_gds {
        out.push((1, Rule::M0SendmsgGds))
    }
    if let Some((data, sgpr_soffset)) = p.wide_store {
        if c.vwrites().iter().any(|w| w.overlaps(data)) {
            if !sgpr_soffset {
                out.push((1, Rule::StoreDataWrite));
                if c.any_valu() {
                    out.push((2, Rule::StoreDataValu))
                }
            } else if c.any_valu() {
                out.push((1, Rule::StoreDataValuSoffset));
            }
        }
    }
    if p.any_valu() {
        let vcc = p.writes_vcc();
        if c.any_valu() && c.vccz_execz && (vcc || p.exec_def) {
            out.push((5, Rule::VccExecToVcczExecz))
        }
        if let Some((kind, sel, vsrc0)) = c.lane {
            if sel.is_some_and(|s| p.sdefs.iter().any(|d| d.overlaps(s))) {
                out.push((4, Rule::SgprLaneSelect))
            }
            if p.exec_def {
                out.push((4, Rule::ExecRwLane))
            }
            if kind != Lane::Writelane
                && vsrc0.is_some_and(|s| p.vwrites().iter().any(|d| d.overlaps(s)))
            {
                out.push((1, Rule::VgprReadlane))
            }
        }
        if c.div_fmas && vcc {
            out.push((4, Rule::VccDivFmas))
        }
        if c.vmem && hit(&p.sdefs, &c.suses) {
            out.push((5, Rule::SgprVmem))
        }
        if c.any_valu() {
            if hit(&p.sdefs, &c.suses) {
                out.push((2, Rule::SgprConst))
            }
            if c.explicit_exec && p.exec_def {
                out.push((2, Rule::ExecConst))
            }
        }
        if c.dpp {
            if hit(&p.vwrites(), &c.vuses) {
                out.push((2, Rule::VgprDpp))
            }
            if p.exec_def {
                out.push((5, Rule::ExecDpp))
            }
        }
        if p.dst_sel && c.any_valu() {
            let d = p.vwrites();
            if hit(&d, &c.vreads()) || hit(&d, &c.vwrites()) {
                out.push((1, Rule::DstSel))
            }
        }
        if p.trans && c.any_valu() && !c.trans && hit(&p.vwrites(), &c.vreads()) {
            out.push((1, Rule::Trans))
        }
        if p.fp8_cvt && c.fp8_cvt && hit(&p.vdefs, &c.vdefs) {
            out.push((1, Rule::Fp8PartialWrite))
        }
    }
    if p.valu {
        if c.mfma.is_some() {
            if p.exec_def {
                out.push((4, Rule::ExecMfma))
            }
            if p.dot.is_none() && hit(&p.vdefs, &c.vreads()) {
                out.push((2, Rule::ValuMfma))
            }
        }
    }
    if let Some(op) = p.dot {
        if c.dot == Some(op) {
            if c.dot_src[..2]
                .iter()
                .flatten()
                .any(|s| p.vdefs.iter().any(|d| d.overlaps(*s)))
            {
                out.push((3, Rule::DotSameSrcAB))
            }
        } else if c.vector() && (hit(&p.vdefs, &c.vreads()) || hit(&p.vdefs, &c.vwrites())) {
            out.push((3, Rule::DotOther));
        }
    }
    if let Some(w) = &p.mfma {
        let (cls, passes) = (w.class, w.passes);
        if let Some(d) = w.dst {
            let d = [d];
            if let Some(r) = &c.mfma {
                if let Some(cc) = r.c.filter(|cc| cc.overlaps(d[0])) {
                    let exact = cc == d[0];
                    let need = srcc(w, r, exact);
                    out.push((
                        need,
                        Rule::MfmaSrcC {
                            w: cls,
                            r: r.class,
                            passes,
                            exact,
                            same: w.op == r.op,
                        },
                    ));
                }
                if hit(&d, &r.ab) {
                    out.push((
                        match cls {
                            Class::Xdl => passes + 3,
                            Class::Sgemm => passes + 2,
                            Class::Dgemm => 11,
                        },
                        Rule::MfmaSrcAB { w: cls, passes },
                    ));
                }
            } else if c.valu {
                if hit(&d, &c.vuses) || hit(&d, &c.vdefs) {
                    out.push((
                        match cls {
                            Class::Xdl => passes + 3,
                            Class::Sgemm => passes + 2,
                            Class::Dgemm => 11,
                        },
                        Rule::MfmaValu { w: cls, passes },
                    ));
                }
            } else if c.vmem || c.ds {
                if hit(&d, &c.vuses) {
                    out.push((
                        match cls {
                            Class::Xdl => passes + 3,
                            Class::Sgemm => passes + 2,
                            Class::Dgemm => 18,
                        },
                        Rule::MfmaMemRead { w: cls, passes },
                    ));
                }
                if hit(&d, &c.vdefs) {
                    out.push((
                        match cls {
                            Class::Xdl => passes + 3,
                            Class::Sgemm => passes + 2,
                            Class::Dgemm => 11,
                        },
                        Rule::MfmaMemWaw { w: cls, passes },
                    ));
                }
            }
        }
        if cls == Class::Dgemm && c.fma64 {
            out.push((2, Rule::DgemmFma64))
        }
        if cls == Class::Xdl && (c.valu || c.vmem || c.ds) {
            if let Some(cc) = w.c {
                // 2/4/8/16 passes -> 1/3/7/15 (LLVM: SMFMA4x4/GFX940_XDL4Pass/SMFMA16x16/SMFMA32x32 ReadVgprVALUWarWaitStates).
                if c.vdefs.iter().any(|d| d.overlaps(cc)) {
                    out.push((passes - 1, Rule::XdlSrcCWar { passes }))
                }
            }
        }
    }
}

/// Producer node → fewest wait states between it and the current point over
/// every path reaching it.
type State = BTreeMap<usize, u32>;

/// One unmet requirement at a consumer.
#[derive(Clone, Debug)]
struct Deficit {
    producer: usize,
    dist: u32,
    need: u32,
    rule: Rule,
}

fn deficits(infos: &[Option<Info>], state: &State, i: usize) -> Vec<Deficit> {
    let Some(c) = &infos[i] else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut reqs = Vec::new();
    for (&p, &dist) in state {
        let Some(pi) = &infos[p] else { continue };
        reqs.clear();
        rules(pi, c, &mut reqs);
        for &(need, rule) in &reqs {
            debug_assert!(need <= WINDOW, "{rule:?} needs {need} > window {WINDOW}");
            if need > dist {
                out.push(Deficit {
                    producer: p,
                    dist,
                    need,
                    rule,
                })
            }
        }
    }
    out.sort_by(|a, b| {
        (b.need - b.dist)
            .cmp(&(a.need - a.dist))
            .then(a.producer.cmp(&b.producer))
    });
    out
}

/// Forward dataflow over nodes (`None` = label: no wait state, no rule).
/// With `adjust`, `nops[i]` becomes the wait states node `i` still needs
/// given the state reaching it (inserted before it); otherwise `nops` is
/// fixed. Returns the in-state of every node at the fixpoint.
fn solve(
    infos: &[Option<Info>],
    preds: &[Vec<usize>],
    nops: &mut [u32],
    adjust: bool,
) -> Result<Vec<State>, String> {
    let n = infos.len();
    let mut out: Vec<State> = vec![State::new(); n];
    let mut ins: Vec<State> = vec![State::new(); n];
    for pass in 1..=PASS_LIMIT {
        let mut changed = false;
        for i in 0..n {
            let mut s = State::new();
            for &p in &preds[i] {
                for (&k, &d) in &out[p] {
                    s.entry(k).and_modify(|e| *e = (*e).min(d)).or_insert(d);
                }
            }
            let o = match &infos[i] {
                None => s.clone(),
                Some(info) => {
                    if adjust {
                        let need = deficits(infos, &s, i)
                            .iter()
                            .map(|d| d.need - d.dist)
                            .max()
                            .unwrap_or(0);
                        let new = if pass <= FREE_PASSES {
                            need
                        } else {
                            need.max(nops[i])
                        };
                        if new != nops[i] {
                            nops[i] = new;
                            changed = true
                        }
                    }
                    let shift = nops[i] + info.ws;
                    let mut o: State = s
                        .iter()
                        .filter_map(|(&k, &d)| (d + shift < WINDOW).then_some((k, d + shift)))
                        .collect();
                    o.insert(i, 0);
                    o
                }
            };
            if o != out[i] {
                out[i] = o;
                changed = true
            }
            ins[i] = s;
        }
        if !changed {
            return Ok(ins);
        }
    }
    Err(format!(
        "hazard dataflow did not converge in {PASS_LIMIT} passes"
    ))
}

fn preds_of(succ: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); succ.len()];
    for (i, s) in succ.iter().enumerate() {
        for &t in s {
            preds[t].push(i)
        }
    }
    preds
}

fn ends_flow(inst: &Inst) -> bool {
    matches!(inst, Inst::Sopp(p) if p.op == SoppOp::Endpgm)
}
fn indirect(inst: &Inst) -> bool {
    matches!(inst, Inst::Sop1(s) if s.op == peacemaker_cdna::enc::sop::Sop1Op::SetpcB64)
}

fn nop(waits: u32) -> Inst {
    Inst::Sopp(Sopp {
        op: SoppOp::Nop,
        simm16: (waits - 1) as u16,
    })
}

/// Insert the required wait states; `HazardProof::pc_index` indexes the
/// returned items.
pub fn resolve(arch: Arch, items: Vec<Item>) -> Result<(Vec<Item>, Vec<HazardProof>), String> {
    match arch {
        Arch::Gfx942 => {}
    }
    let n = items.len();
    let mut labels: HashMap<&str, usize> = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        if let Item::Label(l) = item {
            if labels.insert(l.as_str(), i).is_some() {
                return Err(format!("label {l} defined twice"));
            }
        }
    }
    let mut infos = Vec::with_capacity(n);
    let mut succ = vec![Vec::new(); n];
    for (i, item) in items.iter().enumerate() {
        let falls = match item {
            Item::Label(_) => {
                infos.push(None);
                true
            }
            Item::Branch { op, target } => {
                if (Sopp { op: *op, simm16: 0 }).branch_offset().is_none() {
                    return Err(format!("item {i}: {op:?} is not a branch"));
                }
                let t = *labels
                    .get(target.as_str())
                    .ok_or_else(|| format!("item {i}: branch to undefined label {target}"))?;
                succ[i].push(t);
                infos.push(Some(Info {
                    ws: 1,
                    ..Info::default()
                }));
                *op != SoppOp::Branch
            }
            Item::Inst(inst) => {
                if inst.branch_offset().is_some() {
                    return Err(format!(
                        "item {i}: branch instruction with a raw offset; use Item::Branch"
                    ));
                }
                if indirect(inst) {
                    return Err(format!("item {i}: s_setpc_b64 (indirect branch) is outside what the hazard pass can follow"));
                }
                infos.push(Some(info(inst)));
                !ends_flow(inst)
            }
        };
        if falls && i + 1 < n {
            succ[i].push(i + 1)
        }
    }
    let preds = preds_of(&succ);
    let mut nops = vec![0u32; n];
    // Sequential insertion (minimal on straight-line code), then raise any
    // count a fresh fixpoint over the final NOPs still finds short (NOPs only
    // grow here, bounded by WINDOW, so this terminates).
    solve(&infos, &preds, &mut nops, true)?;
    let ins = loop {
        let ins = solve(&infos, &preds, &mut nops, false)?;
        let mut short = false;
        for i in 0..n {
            let need = deficits(&infos, &ins[i], i)
                .iter()
                .map(|d| d.need - d.dist)
                .max()
                .unwrap_or(0);
            if need > nops[i] {
                nops[i] = need;
                short = true
            }
        }
        if !short {
            break ins;
        }
    };
    // Output positions: every NOP chunk, then the item.
    let mut at = Vec::with_capacity(n);
    let mut pos = 0usize;
    for &k in &nops {
        pos += k.div_ceil(NOP_MAX) as usize;
        at.push(pos);
        pos += 1
    }
    let mut out = Vec::with_capacity(pos);
    let mut proofs = Vec::new();
    for (i, item) in items.into_iter().enumerate() {
        if nops[i] > 0 {
            let mut seen = Vec::new();
            let mut why = Vec::new();
            for d in deficits(&infos, &ins[i], i) {
                if seen.contains(&(d.producer, d.rule)) {
                    continue;
                }
                seen.push((d.producer, d.rule));
                why.push(format!(
                    "{} [item {} -> item {}: {} of {} wait states]",
                    d.rule.cite(),
                    at[d.producer],
                    at[i],
                    d.dist,
                    d.need
                ));
            }
            let rule = why.join(" | ");
            let mut rem = nops[i];
            while rem > 0 {
                let k = rem.min(NOP_MAX);
                proofs.push(HazardProof {
                    pc_index: out.len(),
                    insn: format!("s_nop {}", k - 1),
                    rule: rule.clone(),
                });
                out.push(Item::Inst(nop(k)));
                rem -= k;
            }
        }
        out.push(item);
    }
    Ok((out, proofs))
}

/// Every violated rule in a final straight-line instruction list (branches
/// as decoded), as `"index: rule"` strings; empty when hazard-free.
pub fn check(arch: Arch, insts: &[Inst]) -> Vec<String> {
    let mut bad = Vec::new();
    let n = insts.len();
    let mut offset = Vec::with_capacity(n + 1);
    let mut at = 0usize;
    let mut start: HashMap<usize, usize> = HashMap::new();
    for (i, inst) in insts.iter().enumerate() {
        start.insert(at, i);
        offset.push(at);
        match inst.words(arch) {
            Ok(w) => at += w.len(),
            Err(e) => {
                bad.push(format!("{i}: cannot encode to size branches: {e}"));
                at += 1
            }
        }
    }
    offset.push(at);
    let infos: Vec<Option<Info>> = insts.iter().map(|i| Some(info(i))).collect();
    let mut succ = vec![Vec::new(); n];
    for (i, inst) in insts.iter().enumerate() {
        let mut falls = !ends_flow(inst);
        if indirect(inst) {
            bad.push(format!(
                "{i}: s_setpc_b64 indirect branch: successors unknown, hazards past it unchecked"
            ));
            falls = false
        }
        if let Some(off) = inst.branch_offset() {
            let target = offset[i + 1] as i64 + i64::from(off);
            match usize::try_from(target).ok().and_then(|t| start.get(&t)) {
                Some(&t) => succ[i].push(t),
                None if target == at as i64 => {}
                None => bad.push(format!("{i}: branch target dword {target} is not an instruction boundary; hazards along it unchecked")),
            }
            if matches!(inst, Inst::Sopp(p) if p.op == SoppOp::Branch) {
                falls = false
            }
        }
        if falls && i + 1 < n {
            succ[i].push(i + 1)
        }
    }
    let preds = preds_of(&succ);
    let mut nops = vec![0u32; n];
    let ins = match solve(&infos, &preds, &mut nops, false) {
        Ok(ins) => ins,
        Err(e) => {
            bad.push(format!("0: {e}"));
            return bad;
        }
    };
    for i in 0..n {
        let mut seen = Vec::new();
        for d in deficits(&infos, &ins[i], i) {
            if seen.contains(&(d.producer, d.rule)) {
                continue;
            }
            seen.push((d.producer, d.rule));
            bad.push(format!(
                "{i}: {} (after {}: {} of {} wait states)",
                d.rule.cite(),
                d.producer,
                d.dist,
                d.need
            ));
        }
    }
    bad
}
