// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA (gfx942) code-object emission acceptance: hand-built straight-line
//! kernels (`peacemaker_isa::cdna::Kernel`, instructions encoded with
//! `Inst::encode`) across the resource axes the descriptor depends on —
//! AccVGPRs or none, LDS or none, one to three workgroup-id dimensions, one
//! to three work-item-id dimensions, `fp16_ovfl` on/off, register-file
//! extremes — and both code-object versions (V5, V6), each alone and as one
//! multi-kernel module.
//!
//! * The native writer (`cdna::emit::code_object`) must equal the ROCm oracle
//!   (`llvm-mc` over `cdna::emit::assembly`, then `ld.lld -shared`, bundled
//!   with `clang-offload-bundler -bundle-align=4096`) field by field and byte
//!   by byte in everything but the `.comment` stamp: the comparison and its
//!   normalization are `native_identity`'s (`support/native_compare.rs`).
//! * `llvm-objdump` of both code objects must print, instruction for
//!   instruction, what `Inst`'s `Display` prints (the kernel's code plus the
//!   `s_nop 0` linker alignment padding between kernels of a module).
//! * `sim::load_code_object` must give back each kernel's words and a
//!   descriptor that agrees with its `Resources`, identical for the native and
//!   the oracle object.
//! * The kernel words themselves must equal `llvm-mc`'s encoding of each
//!   instruction's `Display` text, and `llvm-mc -disassemble` plus
//!   `decode_all` must read them back to the same text and instructions.
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::enc::mubuf::{Mubuf, MubufOp};
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::enc::sop::{
    Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp,
};
use peacemaker_cdna::enc::vop::{Dpp, DppCtrl, Vop1, Vop1Op, Vop2, Vop2Op, VopExt, Vopc, VopcOp};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp, Vop3p, Vop3pOp};
use peacemaker_cdna::kd::KernelDescriptor;
use peacemaker_cdna::sim::load_code_object;
use peacemaker_cdna::{decode_all, encode_all, oracle, Inst, Src};
use peacemaker_isa::cdna::emit::{self, CodeObjectVersion};
use peacemaker_isa::cdna::{KernArg, KernArgKind, Kernel, Resources};
use peacemaker_isa::native;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/native_compare.rs"]
mod native_compare;

const ARCH: peacemaker_cdna::Arch = peacemaker_cdna::Arch::Gfx942;
const ISA_ARCH: peacemaker_isa::Arch = peacemaker_isa::Arch::Gfx942;
/// The host entry spelling of the committed `.hxaco` bundles.
const HOST: &str = "host-x86_64-unknown-linux-gnu-";
const VERSIONS: [CodeObjectVersion; 2] = [CodeObjectVersion::V5, CodeObjectVersion::V6];

// ----- instruction constructors -------------------------------------------------------------

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn a(n: u8) -> Src {
    Src::Agpr(n)
}
fn s(n: u8) -> Src {
    Src::Sgpr(n)
}

fn sop1(op: Sop1Op, dst: Src, src: Src) -> Inst {
    Inst::Sop1(Sop1 {
        op,
        sdst: Some(dst),
        ssrc0: Some(src),
    })
}
fn sop2(op: Sop2Op, dst: Src, a: Src, b: Src) -> Inst {
    Inst::Sop2(Sop2 {
        op,
        sdst: dst,
        ssrc0: a,
        ssrc1: b,
    })
}
fn sopc(op: SopcOp, a: Src, b: Src) -> Inst {
    Inst::Sopc(Sopc {
        op,
        ssrc0: a,
        ssrc1: b,
    })
}
fn sopk(op: SopkOp, dst: Src, simm16: u16) -> Inst {
    Inst::Sopk(Sopk {
        op,
        sdst: dst,
        simm16,
    })
}
fn sopp(op: SoppOp, simm16: u16) -> Inst {
    Inst::Sopp(Sopp { op, simm16 })
}
fn endpgm() -> Inst {
    sopp(SoppOp::Endpgm, 0)
}
/// `s_waitcnt lgkmcnt(0)`: vmcnt and expcnt at their maxima (gfx9 field layout).
fn wait_lgkm() -> Inst {
    sopp(SoppOp::Waitcnt, 0xc07f)
}
/// `s_waitcnt vmcnt(0) lgkmcnt(0)`.
fn wait_all() -> Inst {
    sopp(SoppOp::Waitcnt, 0x0070)
}
fn s_load(op: SmemOp, sdata: Src, sbase: Src, offset: i32) -> Inst {
    Inst::Smem(Smem {
        op,
        sdata: Some(sdata),
        sbase: Some(sbase),
        offset: SmemOffset::Imm(offset),
        glc: false,
    })
}
fn vop1(op: Vop1Op, dst: Src, src: Src) -> Inst {
    Inst::Vop1(Vop1 {
        op,
        vdst: dst,
        src0: src,
        ext: VopExt::None,
    })
}
fn vop2(op: Vop2Op, dst: Src, a: Src, b: Src) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst: dst,
        src0: a,
        src1: b,
        ext: VopExt::None,
    })
}
fn vopc(op: VopcOp, a: Src, b: Src) -> Inst {
    Inst::Vopc(Vopc {
        op,
        src0: a,
        src1: b,
        ext: VopExt::None,
    })
}
fn vop3(op: Vop3Op, dst: Src, src: [Option<Src>; 3]) -> Inst {
    Inst::Vop3(Vop3 {
        op,
        dst,
        sdst: None,
        src,
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    })
}
fn vop3p(op: Vop3pOp, dst: Src, src: [Option<Src>; 3]) -> Inst {
    Inst::Vop3p(Vop3p {
        op,
        dst,
        src,
        op_sel: 0,
        op_sel_hi: 7,
        neg_lo: 0,
        neg_hi: 0,
        clamp: false,
    })
}
fn mfma(op: MaiOp, dst: Src, a: Src, b: Src, c: Src) -> Inst {
    Inst::Mai(Mai {
        op,
        dst,
        src_a: a,
        src_b: Some(b),
        src_c: Some(c),
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
fn accvgpr_write(dst: Src, src: Src) -> Inst {
    Inst::Mai(Mai {
        op: MaiOp::AccvgprWriteB32,
        dst,
        src_a: src,
        src_b: None,
        src_c: None,
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
fn accvgpr_read(dst: Src, src: Src) -> Inst {
    Inst::Mai(Mai {
        op: MaiOp::AccvgprReadB32,
        dst,
        src_a: src,
        src_b: None,
        src_c: None,
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
fn ds(
    op: DsOp,
    vdst: Option<Src>,
    addr: Src,
    data0: Option<Src>,
    offset0: u8,
    offset1: u8,
) -> Inst {
    Inst::Ds(Ds {
        op,
        vdst,
        addr: Some(addr),
        data0,
        data1: None,
        offset0,
        offset1,
        gds: false,
    })
}
fn flat(
    op: FlatOp,
    seg: FlatSeg,
    vdst: Option<Src>,
    addr: Src,
    data: Option<Src>,
    saddr: Option<Src>,
    offset: i16,
) -> Inst {
    Inst::Flat(Flat {
        op,
        seg,
        vdst,
        addr: Some(addr),
        data,
        saddr,
        offset,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
fn mubuf(op: MubufOp, vdata: Src, vaddr: Src, srsrc: Src, offset: u16) -> Inst {
    Inst::Mubuf(Mubuf {
        op,
        vdata,
        vaddr: Some(vaddr),
        srsrc,
        soffset: Src::Int(0),
        offset,
        offen: true,
        idxen: false,
        sc0: false,
        sc1: false,
        nt: false,
    })
}

// ----- the kernels --------------------------------------------------------------------------

fn kernargs3() -> Vec<KernArg> {
    vec![
        KernArg {
            name: "src".into(),
            offset: 0,
            size: 8,
            kind: KernArgKind::GlobalBuffer { readonly: true },
        },
        KernArg {
            name: "dst".into(),
            offset: 8,
            size: 8,
            kind: KernArgKind::GlobalBuffer { readonly: false },
        },
        KernArg {
            name: "n".into(),
            offset: 16,
            size: 4,
            kind: KernArgKind::ByValue,
        },
    ]
}
fn kernargs1() -> Vec<KernArg> {
    vec![KernArg {
        name: "buf".into(),
        offset: 0,
        size: 8,
        kind: KernArgKind::GlobalBuffer { readonly: false },
    }]
}

/// `next_free_vgpr` rounded up to 4 (at least 4): the first AGPR.
fn accum(next_free_vgpr: u16) -> u16 {
    next_free_vgpr.next_multiple_of(4).max(4)
}

struct Spec {
    name: &'static str,
    insts: Vec<Inst>,
    next_free_sgpr: u16,
    next_free_vgpr: u16,
    agprs: u16,
    uses_vcc: bool,
    lds: u32,
    kernargs: Vec<KernArg>,
    kernarg_size: u32,
    workgroup_id: [bool; 3],
    workitem_id: u8,
    workgroup_size: u32,
    fp16_ovfl: bool,
}

impl Spec {
    fn kernel(self) -> Kernel {
        let words = encode_all(ARCH, &self.insts).unwrap_or_else(|e| {
            panic!("{}: hand-built instructions do not encode: {e}", self.name)
        });
        Kernel {
            arch: ARCH,
            name: self.name.to_owned(),
            insts: self.insts,
            words,
            labels: Vec::new(),
            resources: Resources {
                next_free_sgpr: self.next_free_sgpr,
                next_free_vgpr: self.next_free_vgpr,
                agprs: self.agprs,
                accum_offset: accum(self.next_free_vgpr),
                uses_vcc: self.uses_vcc,
                group_segment_fixed_size: self.lds,
                kernarg_size: self.kernarg_size,
                workgroup_id: self.workgroup_id,
                workitem_id: self.workitem_id,
                workgroup_size: self.workgroup_size,
                fp16_ovfl: self.fp16_ovfl,
            },
            kernargs: self.kernargs,
            waits: Vec::new(),
            hazards: Vec::new(),
        }
    }
}

/// Scalar ALU, SOPK/SOPC/SOPP and scalar loads only: no VGPR, no AGPR, no LDS,
/// X workgroup id.
fn scalar_only() -> Kernel {
    Spec {
        name: "pm_cdna_scalar",
        insts: vec![
            s_load(SmemOp::LoadDwordx2, s(10), s(0), 0),
            s_load(SmemOp::LoadDwordx4, s(12), s(0), 0x10),
            sop1(Sop1Op::MovB32, s(0), Src::Int(5)),
            sop1(Sop1Op::MovB32, s(1), Src::Lit(0x1234_5678)),
            sop2(Sop2Op::AddU32, s(2), s(0), s(1)),
            sop2(Sop2Op::AndB64, s(4), Src::ExecLo, s(6)),
            sop2(Sop2Op::LshlB32, s(8), s(2), Src::Int(3)),
            sopc(SopcOp::CmpEqU32, s(0), Src::Int(0)),
            sopk(SopkOp::MovkI32, s(3), 0x1234),
            wait_lgkm(),
            sopp(SoppOp::Nop, 3),
            sopp(SoppOp::Setprio, 2),
            sopp(SoppOp::Sleep, 1),
            sopp(SoppOp::Barrier, 0),
            endpgm(),
        ],
        next_free_sgpr: 16,
        next_free_vgpr: 1,
        agprs: 0,
        uses_vcc: false,
        lds: 0,
        kernargs: kernargs3(),
        kernarg_size: 24,
        workgroup_id: [true, false, false],
        workitem_id: 0,
        workgroup_size: 64,
        fp16_ovfl: false,
    }
    .kernel()
}

/// VOP1/2/C, VOP3 (modifiers, VOP3B, op_sel), packed math and DPP: no AGPR, VCC, XY workgroup id,
/// `fp16_ovfl` on.
fn valu() -> Kernel {
    let add = Vop3 {
        op: Vop3Op::V2(Vop2Op::AddF32),
        dst: v(0),
        sdst: None,
        src: [Some(v(1)), Some(v(2)), None],
        abs: 0b010,
        neg: 0b001,
        clamp: true,
        omod: 1,
        op_sel: 0,
    };
    let cvt = Vop3 {
        op: Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32),
        dst: v(0),
        sdst: None,
        src: [Some(v(1)), Some(v(2)), None],
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 8,
    };
    let mad64 = Vop3 {
        op: Vop3Op::Only(Vop3OnlyOp::MadU64U32),
        dst: v(0),
        sdst: Some(s(4)),
        src: [Some(v(2)), Some(v(3)), Some(v(6))],
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    };
    let dpp = Dpp {
        ctrl: DppCtrl::QuadPerm([1, 0, 3, 2]),
        row_mask: 0xf,
        bank_mask: 0xf,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    };
    Spec {
        name: "pm_cdna_valu",
        insts: vec![
            vop1(Vop1Op::MovB32, v(0), v(1)),
            vop1(
                Vop1Op::MovB32,
                v(2),
                Src::Float(peacemaker_cdna::operand::InlineFloat::One),
            ),
            vop1(Vop1Op::MovB32, v(3), Src::Lit(0x3f80_0001)),
            vop1(Vop1Op::CvtF32Fp8, v(7), v(1)),
            Inst::Vop1(Vop1 {
                op: Vop1Op::MovB32,
                vdst: v(1),
                src0: v(2),
                ext: VopExt::Dpp(dpp),
            }),
            vop2(Vop2Op::AddF32, v(3), v(1), v(2)),
            vop2(Vop2Op::AddU32, v(4), v(1), v(2)),
            vop2(Vop2Op::AddCoU32, v(5), v(1), v(2)),
            vopc(VopcOp::CmpLtU32, v(1), v(2)),
            vop2(Vop2Op::CndmaskB32, v(6), v(1), v(2)),
            vop3(
                Vop3Op::Only(Vop3OnlyOp::FmaF32),
                v(0),
                [Some(v(1)), Some(v(2)), Some(v(3))],
            ),
            vop3(
                Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32),
                v(0),
                [Some(v(1)), Some(v(2)), None],
            ),
            Inst::Vop3(cvt),
            Inst::Vop3(add),
            vop3(
                Vop3Op::Only(Vop3OnlyOp::LshlAddU32),
                v(0),
                [Some(v(1)), Some(Src::Int(2)), Some(v(3))],
            ),
            vop3(
                Vop3Op::Only(Vop3OnlyOp::Add3U32),
                v(0),
                [Some(v(1)), Some(v(2)), Some(v(3))],
            ),
            Inst::Vop3(mad64),
            vop3p(Vop3pOp::PkAddF16, v(0), [Some(v(1)), Some(v(2)), None]),
            vop3p(
                Vop3pOp::PkFmaF32,
                v(0),
                [Some(v(2)), Some(v(4)), Some(v(6))],
            ),
            endpgm(),
        ],
        next_free_sgpr: 8,
        next_free_vgpr: 8,
        agprs: 0,
        uses_vcc: true,
        lds: 0,
        kernargs: kernargs3(),
        kernarg_size: 24,
        workgroup_id: [true, true, false],
        workitem_id: 1,
        workgroup_size: 256,
        fp16_ovfl: true,
    }
    .kernel()
}

/// MFMA over AccVGPR accumulators plus the accvgpr moves: AGPRs, XYZ workgroup id and work-item id,
/// `fp16_ovfl` off.
fn mfma_agpr() -> Kernel {
    Spec {
        name: "pm_cdna_mfma_agpr",
        insts: vec![
            accvgpr_write(a(0), v(1)),
            accvgpr_write(a(1), Src::Int(0)),
            mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, a(0), v(0), v(2), a(0)),
            mfma(MaiOp::MfmaF32_32x32x8Bf16, a(0), v(0), v(2), a(0)),
            accvgpr_read(v(1), a(0)),
            wait_all(),
            endpgm(),
        ],
        next_free_sgpr: 8,
        next_free_vgpr: 8,
        agprs: 16,
        uses_vcc: false,
        lds: 0,
        kernargs: kernargs3(),
        kernarg_size: 24,
        workgroup_id: [true, true, true],
        workitem_id: 2,
        workgroup_size: 512,
        fp16_ovfl: false,
    }
    .kernel()
}

/// MFMA accumulating in architectural VGPRs (ACC_CD clear): no AGPR, X and Z workgroup id, `fp16_ovfl` on.
fn mfma_vgpr() -> Kernel {
    Spec {
        name: "pm_cdna_mfma_vgpr",
        insts: vec![
            mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, v(4), v(0), v(2), v(4)),
            mfma(MaiOp::MfmaF32_16x16x16F16, v(4), v(0), v(2), v(4)),
            endpgm(),
        ],
        next_free_sgpr: 8,
        next_free_vgpr: 8,
        agprs: 0,
        uses_vcc: false,
        lds: 0,
        kernargs: kernargs1(),
        kernarg_size: 8,
        workgroup_id: [true, false, true],
        workitem_id: 0,
        workgroup_size: 128,
        fp16_ovfl: true,
    }
    .kernel()
}

/// DS reads and writes: 4 KiB of LDS, X workgroup id.
fn lds() -> Kernel {
    Spec {
        name: "pm_cdna_lds",
        insts: vec![
            ds(DsOp::WriteB32, None, v(0), Some(v(1)), 16, 0),
            ds(DsOp::WriteB64, None, v(0), Some(v(2)), 8, 0),
            sopp(SoppOp::Barrier, 0),
            ds(DsOp::ReadB32, Some(v(2)), v(0), None, 0, 0),
            Inst::Ds(Ds {
                op: DsOp::Read2B32,
                vdst: Some(v(2)),
                addr: Some(v(0)),
                data0: None,
                data1: None,
                offset0: 1,
                offset1: 2,
                gds: false,
            }),
            ds(DsOp::ReadB128, Some(v(4)), v(0), None, 0, 0),
            sopp(SoppOp::Waitcnt, 0xc07f),
            endpgm(),
        ],
        next_free_sgpr: 8,
        next_free_vgpr: 8,
        agprs: 0,
        uses_vcc: false,
        lds: 4096,
        kernargs: kernargs1(),
        kernarg_size: 8,
        workgroup_id: [true, false, false],
        workitem_id: 0,
        workgroup_size: 256,
        fp16_ovfl: false,
    }
    .kernel()
}

/// FLAT/GLOBAL/MUBUF/SMEM memory instructions: no AGPR, no LDS, XY workgroup id.
fn memory() -> Kernel {
    Spec {
        name: "pm_cdna_memory",
        insts: vec![
            s_load(SmemOp::LoadDwordx2, s(2), s(0), 0),
            wait_lgkm(),
            flat(
                FlatOp::LoadDwordx4,
                FlatSeg::Global,
                Some(v(0)),
                v(4),
                None,
                None,
                0,
            ),
            flat(
                FlatOp::LoadDword,
                FlatSeg::Global,
                Some(v(0)),
                v(1),
                None,
                Some(s(2)),
                16,
            ),
            flat(
                FlatOp::StoreDword,
                FlatSeg::Global,
                None,
                v(4),
                Some(v(1)),
                None,
                0,
            ),
            flat(
                FlatOp::LoadDword,
                FlatSeg::Flat,
                Some(v(0)),
                v(2),
                None,
                None,
                0,
            ),
            mubuf(MubufOp::LoadDword, v(0), v(1), s(4), 4),
            mubuf(MubufOp::StoreDwordx4, v(0), v(4), s(4), 0),
            wait_all(),
            endpgm(),
        ],
        next_free_sgpr: 16,
        next_free_vgpr: 8,
        agprs: 0,
        uses_vcc: false,
        lds: 0,
        kernargs: kernargs3(),
        kernarg_size: 24,
        workgroup_id: [true, true, false],
        workitem_id: 1,
        workgroup_size: 64,
        fp16_ovfl: false,
    }
    .kernel()
}

/// The register-file and LDS extremes: v130, a63 above a 132-register architectural file, s99, the full
/// 64 KiB of LDS, 1024 work-items, XYZ ids, `fp16_ovfl` on.
fn wide() -> Kernel {
    Spec {
        name: "pm_cdna_wide",
        insts: vec![
            sop1(Sop1Op::MovB32, s(99), Src::Int(0)),
            vop1(Vop1Op::MovB32, v(130), v(1)),
            accvgpr_write(a(63), v(130)),
            accvgpr_read(v(129), a(63)),
            mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, a(60), v(0), v(2), a(60)),
            ds(DsOp::WriteB32, None, v(0), Some(v(130)), 0, 0),
            endpgm(),
        ],
        next_free_sgpr: 100,
        next_free_vgpr: 131,
        agprs: 64,
        uses_vcc: false,
        lds: 65536,
        kernargs: kernargs3(),
        kernarg_size: 24,
        workgroup_id: [true, true, true],
        workitem_id: 2,
        workgroup_size: 1024,
        fp16_ovfl: true,
    }
    .kernel()
}

fn kernels() -> Vec<Kernel> {
    vec![
        scalar_only(),
        valu(),
        mfma_agpr(),
        mfma_vgpr(),
        lds(),
        memory(),
        wide(),
    ]
}

// ----- oracle plumbing ----------------------------------------------------------------------

fn run(cmd: &mut Command) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{cmd:?} failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// `llvm-mc -filetype=obj` + `ld.lld -shared` + `clang-offload-bundler -bundle-align=4096`: the code
/// object and the bundle, bytes.
fn oracle_build(dir: &Path, stem: &str, asm: &str) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (src, obj, elf, bundle) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
        dir.join(format!("{stem}.hxaco")),
    );
    std::fs::write(&src, asm).map_err(|e| e.to_string())?;
    run(Command::new(oracle::tool("llvm-mc"))
        .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx942", "-filetype=obj"])
        .arg(&src)
        .arg("-o")
        .arg(&obj))?;
    run(Command::new(oracle::tool("ld.lld"))
        .arg("-shared")
        .arg(&obj)
        .arg("-o")
        .arg(&elf))?;
    run(Command::new(oracle::tool("clang-offload-bundler"))
        .args(["-type=o", "-bundle-align=4096"])
        .arg(format!("-targets={HOST},hipv4-amdgcn-amd-amdhsa--gfx942"))
        .arg("-input=/dev/null")
        .arg(format!("-input={}", elf.display()))
        .arg(format!("-output={}", bundle.display())))?;
    Ok((
        std::fs::read(&elf).map_err(|e| e.to_string())?,
        std::fs::read(&bundle).map_err(|e| e.to_string())?,
    ))
}

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hipfire-cdna-emit-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn display_lines(k: &Kernel) -> Vec<String> {
    k.insts
        .iter()
        .map(|i| oracle::normalize(&i.to_string()))
        .collect()
}

/// What `llvm-objdump -d` prints for a linked object of `ks`: each kernel's instructions, kernels
/// 256-byte aligned with `s_nop 0` (the AMDGPU linker fill) between them.
fn expected_disassembly(ks: &[&Kernel]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut at = 0usize;
    for k in ks {
        let start = at.next_multiple_of(256);
        lines.extend(std::iter::repeat("s_nop 0".to_owned()).take((start - at) / 4));
        lines.extend(display_lines(k));
        at = start + 4 * k.words.len();
    }
    lines
}

/// `oracle::objdump` instruction lines (it also returns the `.amdgcn_target` directive line).
fn disassembly(elf: &[u8]) -> Result<Vec<String>, String> {
    Ok(oracle::objdump(ARCH, elf)?
        .into_iter()
        .filter(|l| !l.starts_with('.'))
        .collect())
}

fn first_mismatch<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[T]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} entries, expected {}", got.len(), want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(g, w)| g != w)
        .map(|i| format!("entry {i}: {:?}, expected {:?}", got[i], want[i]))
}

/// The descriptor fields `resources` determine, checked without reference to either writer.
fn check_descriptor(k: &Kernel, kd: &KernelDescriptor) -> Vec<String> {
    let r = &k.resources;
    let mut bad = Vec::new();
    let mut expect = |what: &str, ok: bool| {
        if !ok {
            bad.push(format!("{}: descriptor {what}", k.name))
        }
    };
    expect(
        "group_segment_fixed_size",
        kd.group_segment_fixed_size == r.group_segment_fixed_size,
    );
    expect(
        "private_segment_fixed_size is not 0",
        kd.private_segment_fixed_size == 0,
    );
    expect("kernarg_size", kd.kernarg_size == r.kernarg_size);
    for dim in 0..3 {
        expect(
            &format!("workgroup_id[{dim}]"),
            kd.enable_workgroup_id(dim as u32) == r.workgroup_id[dim],
        )
    }
    expect(
        "workitem id dimensions",
        kd.enable_vgpr_workitem_id() == u32::from(r.workitem_id),
    );
    expect("fp16_overflow", kd.fp16_overflow() == r.fp16_ovfl);
    expect(
        "accum_offset",
        u32::from(r.accum_offset) == 4 * (kd.accum_offset_field() + 1),
    );
    // The VGPR block count covers the unified file: architectural up to accum_offset, then the AGPRs.
    let total = u32::from(r.accum_offset) + u32::from(r.agprs);
    let blocks = kd.granulated_workitem_vgpr_count();
    expect(
        "granulated_workitem_vgpr_count",
        blocks * 8 < total && total <= (blocks + 1) * 8,
    );
    expect(
        "kernarg segment pointer / user SGPR count",
        kd.enable_sgpr_kernarg_segment_ptr() && kd.user_sgpr_count() == 2,
    );
    expect("wavefront size is not 64", !kd.enable_wavefront_size32());
    expect("tg_split is set", !kd.tg_split());
    bad
}

fn version_name(v: CodeObjectVersion) -> &'static str {
    match v {
        CodeObjectVersion::V5 => "V5",
        CodeObjectVersion::V6 => "V6",
    }
}
/// `e_ident[EI_ABIVERSION]`: `ELFABIVERSION_AMDGPU_HSA_V5` is 3, `_V6` is 4.
fn abi_version(v: CodeObjectVersion) -> u8 {
    match v {
        CodeObjectVersion::V5 => 3,
        CodeObjectVersion::V6 => 4,
    }
}

/// Every check of one native object `ks` against the oracle.
fn check_case(
    label: &str,
    ks: &[&Kernel],
    version: CodeObjectVersion,
    dir: &Path,
    stem: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut fail =
        |why: String| failures.push(format!("{label} ({}): {why}", version_name(version)));
    let native_elf = match emit::code_object(ks, version) {
        Ok(e) => e,
        Err(e) => {
            fail(format!("native code_object: {e}"));
            return failures;
        }
    };
    let asm = match emit::assembly(ks, version) {
        Ok(t) => t,
        Err(e) => {
            fail(format!("assembly: {e}"));
            return failures;
        }
    };
    for k in ks {
        if !asm.lines().any(|l| {
            l.trim()
                .strip_prefix(".amdhsa_kernel ")
                .is_some_and(|n| n.trim() == k.name)
        }) {
            fail(format!("assembly has no `.amdhsa_kernel {}`", k.name));
        }
    }
    let native_bundle = native::bundle(&native_elf, ISA_ARCH, HOST);
    let (oracle_elf, oracle_bundle) = match oracle_build(dir, stem, &asm) {
        Ok(o) => o,
        Err(e) => {
            fail(format!("oracle build of the emitted assembly: {e}"));
            return failures;
        }
    };
    if let Some(why) = native_compare::compare_native_to_oracle(
        &native_elf,
        &native_bundle,
        &oracle_elf,
        &oracle_bundle,
    ) {
        fail(why)
    }
    for (who, elf) in [("native", &native_elf), ("oracle", &oracle_elf)] {
        if elf.len() < 9 || elf[7] != 64 || elf[8] != abi_version(version) {
            fail(format!(
                "{who} ELF OS/ABI {:?} / ABI version {:?}, expected 64 / {}",
                elf.get(7),
                elf.get(8),
                abi_version(version)
            ));
        }
    }

    // llvm-objdump prints what Display prints.
    let want = expected_disassembly(ks);
    for (who, elf) in [("native", &native_elf), ("oracle", &oracle_elf)] {
        match disassembly(elf) {
            Ok(got) => {
                if let Some(why) = first_mismatch(&got, &want) {
                    fail(format!("{who} llvm-objdump vs Display: {why}"))
                }
            }
            Err(e) => fail(format!("{who} llvm-objdump: {e}")),
        }
    }

    // sim::load_code_object round-trips words and descriptor, native and oracle alike.
    for k in ks {
        let native_image = load_code_object(&native_elf, &k.name);
        let oracle_image = load_code_object(&oracle_elf, &k.name);
        match (native_image, oracle_image) {
            (Ok(n), Ok(o)) => {
                if n.arch != ARCH || n.name != k.name {
                    fail(format!("{}: loaded as {:?} {:?}", k.name, n.arch, n.name))
                }
                if n.words != k.words {
                    fail(format!(
                        "{}: loaded words differ from the kernel's: {:?}",
                        k.name,
                        first_mismatch(&n.words, &k.words)
                    ))
                }
                if o.words != k.words {
                    fail(format!("{}: oracle words differ from the kernel's", k.name))
                }
                if n.kd != o.kd {
                    fail(format!(
                        "{}: native descriptor {:?} != oracle descriptor {:?}",
                        k.name, n.kd, o.kd
                    ))
                }
                for why in check_descriptor(k, &n.kd) {
                    fail(why)
                }
            }
            (n, o) => fail(format!(
                "{}: load_code_object native {:?} oracle {:?}",
                k.name,
                n.err(),
                o.err()
            )),
        }
    }
    failures
}

/// `ROCm` oracle availability: only an explicit `HIPFIRE_TEST_REQUIRE_ROCM=0` skips.
fn require_rocm(tools: &[&str]) -> bool {
    if oracle::disabled() {
        eprintln!("skipping: ROCm oracle tests explicitly disabled");
        return false;
    }
    for t in tools {
        assert!(
            oracle::tool(t).is_file(),
            "required ROCm tool {} missing; set HIPFIRE_TEST_REQUIRE_ROCM=0 only on no-ROCm hosts",
            oracle::tool(t).display()
        )
    }
    true
}

// ----- tests --------------------------------------------------------------------------------

#[test]
fn corpus_spans_the_resource_axes() {
    let ks = kernels();
    let r = |f: fn(&Resources) -> bool| ks.iter().filter(|k| f(&k.resources)).count();
    assert!(
        r(|r| r.agprs > 0) >= 1 && r(|r| r.agprs == 0) >= 1,
        "AGPR and no-AGPR kernels"
    );
    assert!(
        r(|r| r.group_segment_fixed_size > 0) >= 1 && r(|r| r.group_segment_fixed_size == 0) >= 1,
        "LDS and no-LDS kernels"
    );
    for n in 1..=3 {
        assert!(
            ks.iter()
                .any(|k| k.resources.workgroup_id.iter().filter(|&&d| d).count() == n),
            "a kernel with {n} workgroup-id dimensions"
        )
    }
    assert!(
        r(|r| r.fp16_ovfl) >= 1 && r(|r| !r.fp16_ovfl) >= 1,
        "fp16_ovfl on and off"
    );
    let names: std::collections::BTreeSet<_> = ks.iter().map(|k| k.name.as_str()).collect();
    assert_eq!(
        names.len(),
        ks.len(),
        "kernel names are unique (modules link them together)"
    );
    // Every format the encoder family has appears.
    let picks: [fn(&Inst) -> bool; 15] = [
        |i| matches!(i, Inst::Sop1(_)),
        |i| matches!(i, Inst::Sop2(_)),
        |i| matches!(i, Inst::Sopk(_)),
        |i| matches!(i, Inst::Sopc(_)),
        |i| matches!(i, Inst::Sopp(_)),
        |i| matches!(i, Inst::Smem(_)),
        |i| matches!(i, Inst::Vop1(_)),
        |i| matches!(i, Inst::Vop2(_)),
        |i| matches!(i, Inst::Vopc(_)),
        |i| matches!(i, Inst::Vop3(_)),
        |i| matches!(i, Inst::Vop3p(_)),
        |i| matches!(i, Inst::Mai(_)),
        |i| matches!(i, Inst::Ds(_)),
        |i| matches!(i, Inst::Flat(_)),
        |i| matches!(i, Inst::Mubuf(_)),
    ];
    for (n, pick) in picks.iter().enumerate() {
        assert!(
            ks.iter().flat_map(|k| &k.insts).any(|i| pick(i)),
            "format #{n} of Inst is missing from the corpus"
        );
    }
    for k in &ks {
        assert_eq!(
            k.words,
            encode_all(ARCH, &k.insts).unwrap(),
            "{}: words are the instructions' encoding",
            k.name
        );
        assert!(
            matches!(
                k.insts.last(),
                Some(Inst::Sopp(Sopp {
                    op: SoppOp::Endpgm,
                    ..
                }))
            ),
            "{} ends with s_endpgm",
            k.name
        );
    }
}

/// `llvm-mc` agrees with each instruction's words and `Display`, and the decoder reads them back.
fn check_words(k: &Kernel) {
    let lines: Vec<String> = k.insts.iter().map(|i| i.to_string()).collect();
    let encoded = oracle::mc_encode(ARCH, &lines)
        .unwrap_or_else(|e| panic!("{}: llvm-mc rejects the Display text: {e}", k.name));
    // Each instruction's words are llvm-mc's bytes for its own text.
    let mut per_inst = Vec::new();
    for (i, inst) in k.insts.iter().enumerate() {
        let words = inst.words(ARCH).unwrap();
        assert_eq!(
            oracle::bytes(&words),
            encoded[i].1,
            "{}: `{}` encodes differently from llvm-mc",
            k.name,
            lines[i]
        );
        per_inst.push(oracle::bytes(&words));
    }
    assert_eq!(
        oracle::bytes(&k.words),
        per_inst.concat(),
        "{}: the kernel words are the concatenated instructions",
        k.name
    );
    // llvm's disassembler reads them back to the Display text.
    let disassembled = oracle::mc_disassemble(ARCH, &per_inst)
        .unwrap_or_else(|e| panic!("{}: llvm-mc -disassemble: {e}", k.name));
    let shown: Vec<String> = lines.iter().map(|l| oracle::normalize(l)).collect();
    assert_eq!(
        first_mismatch(&disassembled, &shown),
        None,
        "{}: llvm-mc -disassemble vs Display",
        k.name
    );
    // And our decoder reads them back to the same instructions, at the right dword offsets.
    let decoded =
        decode_all(ARCH, &k.words).unwrap_or_else(|e| panic!("{}: decode_all: {e}", k.name));
    let mut at = 0;
    for ((offset, inst), want) in decoded.iter().zip(&k.insts) {
        assert_eq!((*offset, inst), (at, want), "{}: decode_all", k.name);
        at += want.words(ARCH).unwrap().len();
    }
    assert_eq!(
        decoded.len(),
        k.insts.len(),
        "{}: decode_all instruction count",
        k.name
    );
}

/// Native vs oracle, `objdump`, descriptor and loader checks for every kernel alone, all together and
/// the last three reversed, for V5 and V6.
fn check_identity(ks: &[Kernel], tag: &str) {
    let dir = scratch_dir(tag);
    let all: Vec<&Kernel> = ks.iter().collect();
    let mut failures = Vec::new();
    for version in VERSIONS {
        for (i, k) in ks.iter().enumerate() {
            failures.extend(check_case(
                &k.name,
                &[k],
                version,
                &dir,
                &format!("k{i}_{}", version_name(version)),
            ));
        }
        failures.extend(check_case(
            "module of all kernels",
            &all,
            version,
            &dir,
            &format!("module_{}", version_name(version)),
        ));
        // A three-kernel module in reverse order: layout follows the kernel order given.
        let reversed: Vec<&Kernel> = ks.iter().rev().take(3).collect();
        failures.extend(check_case(
            "reversed module",
            &reversed,
            version,
            &dir,
            &format!("reversed_{}", version_name(version)),
        ));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "{} native-vs-oracle failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn kernel_words_equal_llvm_mc_and_decode_back() {
    if !require_rocm(&["llvm-mc"]) {
        return;
    }
    for k in kernels() {
        check_words(&k)
    }
}

#[test]
fn native_code_objects_equal_the_rocm_oracle_alone_and_as_a_module() {
    if !require_rocm(&["llvm-mc", "ld.lld", "clang-offload-bundler", "llvm-objdump"]) {
        return;
    }
    check_identity(&kernels(), "identity");
}

/// Scalar-ALU-only (SOP1/2/K/C/P) kernels over the resource axes. They exercise the whole emission
/// path (descriptor, metadata, code object, bundle, `objdump`, loader) independently of the VALU, MAI,
/// DS, FLAT, MUBUF and SMEM encoders. The resources are declared, not derived from the code, so AGPR,
/// LDS and register-file extremes are covered without instructions that use them.
fn scalar_kernels() -> Vec<Kernel> {
    let body = |extra: Vec<Inst>| {
        let mut insts = vec![
            sop1(Sop1Op::MovB32, s(0), Src::Int(5)),
            sop1(Sop1Op::MovB32, s(1), Src::Lit(0x1234_5678)),
            sop2(Sop2Op::AddU32, s(2), s(0), s(1)),
            sop2(Sop2Op::AndB64, s(4), Src::ExecLo, s(6)),
            sop2(Sop2Op::LshlB32, s(8), s(2), Src::Int(3)),
            sopc(SopcOp::CmpEqU32, s(0), Src::Int(0)),
            sopk(SopkOp::MovkI32, s(3), 0x1234),
            wait_all(),
            sopp(SoppOp::Nop, 3),
            sopp(SoppOp::Setprio, 2),
            sopp(SoppOp::Sleep, 1),
            sopp(SoppOp::Barrier, 0),
        ];
        insts.extend(extra);
        insts.push(endpgm());
        insts
    };
    let spec = |name,
                insts,
                next_free_vgpr,
                agprs,
                uses_vcc,
                lds,
                kernargs,
                kernarg_size,
                workgroup_id,
                workitem_id,
                workgroup_size,
                fp16_ovfl| {
        Spec {
            name,
            insts,
            next_free_sgpr: 16,
            next_free_vgpr,
            agprs,
            uses_vcc,
            lds,
            kernargs,
            kernarg_size,
            workgroup_id,
            workitem_id,
            workgroup_size,
            fp16_ovfl,
        }
        .kernel()
    };
    vec![
        // No VGPR use, no AGPR, no LDS, X id.
        spec(
            "pm_cdna_sop_min",
            body(vec![]),
            1,
            0,
            false,
            0,
            kernargs3(),
            24,
            [true, false, false],
            0,
            64,
            false,
        ),
        // AGPRs declared, VCC reserved (SOP1 write of vcc), XY ids, fp16_ovfl on.
        spec(
            "pm_cdna_sop_agpr",
            body(vec![sop1(Sop1Op::MovB64, Src::VccLo, Src::Int(0))]),
            8,
            16,
            true,
            0,
            kernargs3(),
            24,
            [true, true, false],
            1,
            256,
            true,
        ),
        // 4 KiB LDS declared, XYZ ids, one by-value argument.
        spec(
            "pm_cdna_sop_lds",
            body(vec![sop2(Sop2Op::OrB32, s(9), s(2), s(3))]),
            4,
            0,
            false,
            4096,
            kernargs1(),
            8,
            [true, true, true],
            2,
            512,
            false,
        ),
        // Register-file and LDS extremes: 131 architectural VGPRs, 64 AGPRs, 64 KiB LDS, 1024 work-items.
        spec(
            "pm_cdna_sop_wide",
            body(vec![sop1(Sop1Op::MovB32, s(12), Src::Int(0))]),
            131,
            64,
            false,
            65536,
            kernargs3(),
            24,
            [true, true, true],
            2,
            1024,
            true,
        ),
        // X and Z ids only, a different AGPR count.
        spec(
            "pm_cdna_sop_xz",
            body(vec![sopk(SopkOp::AddkI32, s(5), 7)]),
            33,
            4,
            false,
            1024,
            kernargs1(),
            8,
            [true, false, true],
            0,
            128,
            false,
        ),
    ]
}

#[test]
fn scalar_only_words_equal_llvm_mc_and_decode_back() {
    if !require_rocm(&["llvm-mc"]) {
        return;
    }
    for k in scalar_kernels() {
        check_words(&k)
    }
}

#[test]
fn scalar_only_native_code_objects_equal_the_rocm_oracle_alone_and_as_a_module() {
    if !require_rocm(&["llvm-mc", "ld.lld", "clang-offload-bundler", "llvm-objdump"]) {
        return;
    }
    check_identity(&scalar_kernels(), "scalar");
}

#[test]
fn code_object_versions_differ_only_where_the_abi_does() {
    let ks = kernels();
    let all: Vec<&Kernel> = ks.iter().collect();
    let v5 = emit::code_object(&all, CodeObjectVersion::V5).unwrap();
    let v6 = emit::code_object(&all, CodeObjectVersion::V6).unwrap();
    assert_eq!((v5[8], v6[8]), (3, 4), "e_ident[EI_ABIVERSION]");
    assert_ne!(v5, v6);
    let again = emit::code_object(&all, CodeObjectVersion::V6).unwrap();
    assert_eq!(v6, again, "emission is deterministic");
    for (version, elf) in [(CodeObjectVersion::V5, &v5), (CodeObjectVersion::V6, &v6)] {
        for k in &ks {
            let image = load_code_object(elf, &k.name)
                .unwrap_or_else(|e| panic!("{} {}: {e}", k.name, version_name(version)));
            assert_eq!(image.words, k.words, "{} {}", k.name, version_name(version));
            assert_eq!(check_descriptor(k, &image.kd), Vec::<String>::new());
        }
    }
    // The metadata names every kernel and states its AGPR count and wavefront size 64.
    for version in VERSIONS {
        let yaml = emit::metadata_yaml(&all, version).unwrap();
        for k in &ks {
            assert!(
                yaml.contains(&k.name),
                "{}: metadata names the kernel ({})",
                k.name,
                version_name(version)
            );
            assert!(
                yaml.contains(&format!("agpr_count: {}", k.resources.agprs)),
                "{}: metadata agpr_count ({})",
                k.name,
                version_name(version)
            );
            let total = if k.resources.agprs > 0 {
                k.resources.accum_offset + k.resources.agprs
            } else {
                k.resources.next_free_vgpr
            };
            assert!(
                yaml.contains(&format!(".vgpr_count: {total}\n")),
                "{}: metadata vgpr_count is the unified total {total} ({})",
                k.name,
                version_name(version)
            );
        }
        assert!(
            yaml.contains("wavefront_size: 64"),
            "metadata states wavefront size 64 ({})",
            version_name(version)
        );
    }
    for k in &ks {
        let directives = emit::directives(k).unwrap();
        assert!(!directives.is_empty(), "{}: directives", k.name);
    }
}
