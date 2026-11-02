// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Workstream acceptance: a tiny GEMM through the whole emulator — encoded
//! words, kernel descriptor, dispatch ABI, SMEM kernarg loads, global loads,
//! `s_waitcnt`, one MFMA, global stores — equal to a CPU reference that
//! shares nothing with the emulator.
//!
//! `C[16×16] = A[16×K] · B[K×16]` with A row-major and B stored transposed
//! (`Bt[16×K]`), so both operands use the same per-lane address: lane `l`
//! holds `A[l%16][kl..kl+KL]` and `B[kl..kl+KL][l%16]` with
//! `kl = KL·(l/16)` (CDNA3 ISA §7.1.4.1 p.46-47); output item `r` of lane
//! `l` is `C[4·(l/16)+r][l%16]` (§7.1.4.2 p.48).
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::enc::sop::{waitcnt_imm, Sopp, SoppOp};
use peacemaker_cdna::enc::vop::{Vop2, Vop2Op, VopExt};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp};
use peacemaker_cdna::kd::KernelDescriptor;
use peacemaker_cdna::sim::{self, mfma::AccumModel, Config, KernelImage, Launch, Memory};
use peacemaker_cdna::{encode_all, Arch, Inst, Src};

const ARCH: Arch = Arch::Gfx942;
const A_ADDR: u64 = 0x1_0000;
const B_ADDR: u64 = 0x2_0000;
const C_ADDR: u64 = 0x3_0000;
const KARG: u64 = 0x100;

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn s(n: u8) -> Src {
    Src::Sgpr(n)
}
fn sopp(op: SoppOp, simm16: u16) -> Inst {
    Inst::Sopp(Sopp { op, simm16 })
}
fn vop2(op: Vop2Op, d: u8, a: Src, b: u8) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst: v(d),
        src0: a,
        src1: v(b),
        ext: VopExt::None,
    })
}
fn vop3(op: Vop3OnlyOp, d: u8, srcs: [Src; 3]) -> Inst {
    Inst::Vop3(Vop3 {
        op: Vop3Op::Only(op),
        dst: v(d),
        sdst: None,
        src: [Some(srcs[0]), Some(srcs[1]), Some(srcs[2])],
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    })
}
fn load(op: FlatOp, d: u8, off: u8, base: u8) -> Inst {
    Inst::Flat(Flat {
        op,
        seg: FlatSeg::Global,
        vdst: Some(v(d)),
        addr: Some(v(off)),
        data: None,
        saddr: Some(s(base)),
        offset: 0,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
fn store(data: u8, off: u8, base: u8, offset: i16) -> Inst {
    Inst::Flat(Flat {
        op: FlatOp::StoreDword,
        seg: FlatSeg::Global,
        vdst: None,
        addr: Some(v(off)),
        data: Some(v(data)),
        saddr: Some(s(base)),
        offset,
        sc0: false,
        sc1: false,
        nt: false,
    })
}

/// One wave: load both operand fragments, one MFMA, store the 16×16 tile.
/// `kl_bytes` is the bytes per lane per operand (8 for every 8-byte-operand shape here).
fn program(op: MaiOp, k_bytes: u8, kl_bytes: u8) -> Vec<Inst> {
    let kl_shift = kl_bytes.trailing_zeros() as i8;
    let row_shift = k_bytes.trailing_zeros() as i8;
    vec![
        Inst::Smem(Smem {
            op: SmemOp::LoadDwordx4,
            sdata: Some(s(4)),
            sbase: Some(s(0)),
            offset: SmemOffset::Imm(0),
            glc: false,
        }),
        Inst::Smem(Smem {
            op: SmemOp::LoadDwordx2,
            sdata: Some(s(8)),
            sbase: Some(s(0)),
            offset: SmemOffset::Imm(16),
            glc: false,
        }),
        vop2(Vop2Op::AndB32, 1, Src::Int(15), 0), // v1 = l % 16
        vop2(Vop2Op::LshrrevB32, 2, Src::Int(4), 0), // v2 = l / 16
        vop2(Vop2Op::LshlrevB32, 3, Src::Int(row_shift), 1), // v3 = (l%16)·row bytes
        vop3(Vop3OnlyOp::LshlAddU32, 3, [v(2), Src::Int(kl_shift), v(3)]), //    + (l/16)·KL bytes
        sopp(SoppOp::Waitcnt, waitcnt_imm(ARCH, 63, 7, 0).unwrap()),
        load(FlatOp::LoadDwordx2, 4, 3, 4),           // A fragment
        load(FlatOp::LoadDwordx2, 6, 3, 6),           // B fragment (B stored transposed)
        vop2(Vop2Op::LshlrevB32, 12, Src::Int(8), 2), // v12 = (l/16)·256
        vop3(Vop3OnlyOp::LshlAddU32, 12, [v(1), Src::Int(2), v(12)]), //     + (l%16)·4
        sopp(SoppOp::Waitcnt, waitcnt_imm(ARCH, 0, 7, 15).unwrap()),
        Inst::Mai(Mai {
            op,
            dst: v(8),
            src_a: v(4),
            src_b: Some(v(6)),
            src_c: Some(Src::Int(0)),
            cbsz: 0,
            abid: 0,
            blgp: 0,
        }),
        sopp(SoppOp::Nop, 7),
        sopp(SoppOp::Nop, 7),
        sopp(SoppOp::Nop, 7),
        store(8, 12, 8, 0),
        store(9, 12, 8, 64),
        store(10, 12, 8, 128),
        store(11, 12, 8, 192),
        sopp(SoppOp::Endpgm, 0),
    ]
}

fn image(insts: &[Inst]) -> KernelImage {
    let words = encode_all(ARCH, insts).expect("encode");
    let kd = KernelDescriptor {
        kernarg_size: 24,
        // 16 unified VGPRs (granule 8 → field 1), FLOAT_DENORM_MODE_32/16_64 = 3, DX10_CLAMP, IEEE.
        compute_pgm_rsrc1: 1 | (0xf << 16) | (1 << 21) | (1 << 23),
        // USER_SGPR_COUNT = 2 (kernarg pointer), work-item ID X only.
        compute_pgm_rsrc2: 2 << 1,
        // ACCUM_OFFSET 16 → field 3.
        compute_pgm_rsrc3: 3,
        kernel_code_properties: 1 << 3,
        ..Default::default()
    };
    KernelImage {
        arch: ARCH,
        name: "tiny_gemm".into(),
        words,
        kd,
    }
}

fn run(insts: &[Inst], a: Vec<u8>, bt: Vec<u8>, accum: AccumModel) -> Vec<u32> {
    let mut mem = Memory::default();
    mem.map_readonly("A", A_ADDR, a).unwrap();
    mem.map_readonly("Bt", B_ADDR, bt).unwrap();
    mem.map("C", C_ADDR, vec![0xee; 16 * 16 * 4]).unwrap();
    let kernarg = [A_ADDR, B_ADDR, C_ADDR]
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();
    let launch = Launch {
        grid: [1, 1, 1],
        block: [64, 1, 1],
        kernarg,
        kernarg_addr: KARG,
        dynamic_lds: 0,
    };
    let cfg = Config {
        accum,
        ..Config::default()
    };
    sim::run(&image(insts), &launch, &mut mem, &cfg).unwrap_or_else(|e| panic!("{e}"));
    mem.region("C")
        .unwrap()
        .bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// FNUZ E4M3 (bias 8, CDNA3 ISA §7.2 Table 30 p.50), written out here so the
/// reference shares no code with the emulator.
fn fnuz_e4m3(code: u8) -> f64 {
    assert_ne!(code, 0x80, "NaN code");
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((code >> 3) & 15), f64::from(code & 7));
    sign * if e == 0 {
        m * 2f64.powi(-10)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 8)
    }
}

#[test]
fn fp8_gemm_16x16x32_matches_cpu() {
    if peacemaker_cdna::oracle::disabled() { /* the test needs no ROCm tool */ }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let code = |rng: &mut Rng| loop {
        let c = rng.next() as u8;
        if c != 0x80 {
            break c;
        }
    };
    for trial in 0..8 {
        let a: Vec<u8> = (0..16 * 32).map(|_| code(&mut rng)).collect();
        let bt: Vec<u8> = (0..16 * 32).map(|_| code(&mut rng)).collect();
        let c = run(
            &program(MaiOp::MfmaF32_16x16x32Fp8Fp8, 32, 8),
            a.clone(),
            bt.clone(),
            AccumModel::Exact,
        );
        for i in 0..16 {
            for j in 0..16 {
                // 32 products of FNUZ E4M3 values span 2^-20..2^16: their sum is exact in f64,
                // so one cast is the single rounding of AccumModel::Exact.
                let exact: f64 = (0..32)
                    .map(|k| fnuz_e4m3(a[i * 32 + k]) * fnuz_e4m3(bt[j * 32 + k]))
                    .sum();
                assert_eq!(
                    c[i * 16 + j],
                    (exact as f32).to_bits(),
                    "trial {trial} C[{i}][{j}]"
                );
            }
        }
    }
}

#[test]
fn i8_gemm_16x16x32_is_exact() {
    let mut rng = Rng(42);
    let a: Vec<u8> = (0..16 * 32).map(|_| rng.next() as u8).collect();
    let bt: Vec<u8> = (0..16 * 32).map(|_| rng.next() as u8).collect();
    let c = run(
        &program(MaiOp::MfmaI32_16x16x32I8, 32, 8),
        a.clone(),
        bt.clone(),
        AccumModel::default(),
    );
    for i in 0..16 {
        for j in 0..16 {
            let exact: i32 = (0..32)
                .map(|k| i32::from(a[i * 32 + k] as i8) * i32::from(bt[j * 32 + k] as i8))
                .sum();
            assert_eq!(c[i * 16 + j] as i32, exact, "C[{i}][{j}]");
        }
    }
}

#[test]
fn bf16_gemm_16x16x16_matches_cpu() {
    let mut rng = Rng(7);
    // Moderate exponents keep the 16-term sum exact in f64.
    let mut bf16 = |rng: &mut Rng| -> u16 {
        let e = 120 + (rng.next() % 16) as u16;
        ((rng.next() as u16 & 1) << 15) | (e << 7) | (rng.next() as u16 & 0x7f)
    };
    let a16: Vec<u16> = (0..16 * 16).map(|_| bf16(&mut rng)).collect();
    let b16: Vec<u16> = (0..16 * 16).map(|_| bf16(&mut rng)).collect();
    let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let c = run(
        &program(MaiOp::MfmaF32_16x16x16Bf16, 32, 8),
        bytes(&a16),
        bytes(&b16),
        AccumModel::Exact,
    );
    let f = |x: u16| f64::from(f32::from_bits(u32::from(x) << 16));
    for i in 0..16 {
        for j in 0..16 {
            let exact: f64 = (0..16)
                .map(|k| f(a16[i * 16 + k]) * f(b16[j * 16 + k]))
                .sum();
            assert_eq!(c[i * 16 + j], (exact as f32).to_bits(), "C[{i}][{j}]");
        }
    }
}
