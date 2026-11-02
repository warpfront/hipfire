// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emulator coverage table: every opcode of every table executed once on a
//! synthetic wave. An opcode is covered when its executor runs it (a fault
//! from the synthetic operands still proves the semantics exist); it is
//! uncovered only when the executor answers `SimError::Unsupported`.
//! `cargo test -p peacemaker-cdna --test sim_coverage -- --nocapture` prints
//! the table (also written to `$CARGO_TARGET_TMPDIR/sim_coverage.md`).
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::enc::mubuf::{Mubuf, MubufOp};
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::enc::sop::{
    Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp,
};
use peacemaker_cdna::enc::vop::{Vop1, Vop1Op, Vop2, Vop2Op, VopExt, Vopc, VopcOp};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3Op};
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp, Vop3p, Vop3pOp};
use peacemaker_cdna::sim::{self, Config, Memory, SimError, Wave};
use peacemaker_cdna::{Arch, Inst, Src, WAVE};

const ARCH: Arch = Arch::Gfx942;
const BASE: u64 = 0x10_0000;

fn sreg(n: u8, dwords: u8) -> Option<Src> {
    (dwords > 0).then_some(Src::Sgpr(n))
}
fn vreg(n: u8, dwords: u8) -> Option<Src> {
    (dwords > 0).then_some(Src::Vgpr(n))
}

fn wave() -> Wave {
    let mut w = Wave::new(ARCH, 0, 0);
    for r in 0..64 {
        for lane in 0..WAVE {
            w.v[r][lane] = (lane as u32) * 16;
        }
    }
    for r in 0..64 {
        for lane in 0..WAVE {
            w.a[r][lane] = 0x3f80_0000;
        }
    }
    for (i, s) in w.s.iter_mut().enumerate() {
        *s = i as u32;
    }
    // s[2:3] global base, s[4:7] raw buffer V# (base, stride 0, num_records 64 KiB).
    w.s[2] = BASE as u32;
    w.s[3] = (BASE >> 32) as u32;
    w.s[4] = BASE as u32;
    w.s[5] = (BASE >> 32) as u32;
    w.s[6] = 0x1_0000;
    w.s[7] = 0x0002_7000;
    w.vcc = 0x5555_5555_5555_5555;
    w
}

fn memory() -> Memory {
    let mut m = Memory::default();
    m.map("global", BASE, vec![0; 0x2_0000]).unwrap();
    m
}

fn cases() -> Vec<(&'static str, String, Inst)> {
    let mut out: Vec<(&'static str, String, Inst)> = Vec::new();
    for &op in Sop2Op::ALL {
        let (d, a, b) = op.widths();
        out.push((
            "SOP2",
            op.name().into(),
            Inst::Sop2(Sop2 {
                op,
                sdst: Src::Sgpr(10),
                ssrc0: Src::Sgpr(if a > 1 { 12 } else { 13 }),
                ssrc1: Src::Sgpr(if b > 1 { 14 } else { 15 }),
            }),
        ));
        let _ = d;
    }
    for &op in Sop1Op::ALL {
        let (d, a) = op.widths();
        out.push((
            "SOP1",
            op.name().into(),
            Inst::Sop1(Sop1 {
                op,
                sdst: sreg(10, d),
                ssrc0: sreg(12, a),
            }),
        ));
    }
    for &op in SopkOp::ALL {
        // hwreg(HW_REG_MODE, 0, 4): FP_ROUND, a modelled MODE field.
        let simm16 = if matches!(op, SopkOp::GetregB32 | SopkOp::SetregB32) {
            (3 << 11) | 1
        } else {
            5
        };
        out.push((
            "SOPK",
            op.name().into(),
            Inst::Sopk(Sopk {
                op,
                sdst: Src::Sgpr(10),
                simm16,
            }),
        ));
    }
    for &op in SopcOp::ALL {
        out.push((
            "SOPC",
            op.name().into(),
            Inst::Sopc(Sopc {
                op,
                ssrc0: Src::Sgpr(12),
                ssrc1: Src::Sgpr(14),
            }),
        ));
    }
    for &op in SoppOp::ALL {
        out.push(("SOPP", op.name().into(), Inst::Sopp(Sopp { op, simm16: 0 })));
    }
    for &op in SmemOp::ALL {
        let (sdata, sbase) = match op {
            SmemOp::DcacheInv => (None, None),
            SmemOp::Memtime | SmemOp::Memrealtime => (Some(Src::Sgpr(20)), None),
            _ if op.buffer() => (Some(Src::Sgpr(20)), Some(Src::Sgpr(4))),
            _ => (Some(Src::Sgpr(20)), Some(Src::Sgpr(2))),
        };
        out.push((
            "SMEM",
            op.name().into(),
            Inst::Smem(Smem {
                op,
                sdata,
                sbase,
                offset: SmemOffset::Imm(0),
                glc: false,
            }),
        ));
    }
    for &op in Vop1Op::ALL {
        let (d, a) = op.widths();
        let inst = match op {
            Vop1Op::ReadfirstlaneB32 => Vop1 {
                op,
                vdst: Src::Sgpr(10),
                src0: Src::Vgpr(12),
                ext: VopExt::None,
            },
            Vop1Op::AccvgprMovB32 => Vop1 {
                op,
                vdst: Src::Agpr(10),
                src0: Src::Agpr(12),
                ext: VopExt::None,
            },
            Vop1Op::Nop => Vop1 {
                op,
                vdst: Src::Vgpr(0),
                src0: Src::Sgpr(0),
                ext: VopExt::None,
            },
            _ => Vop1 {
                op,
                vdst: vreg(10, d).unwrap_or(Src::Vgpr(0)),
                src0: vreg(12, a).unwrap_or(Src::Sgpr(0)),
                ext: VopExt::None,
            },
        };
        out.push(("VOP1", op.name().into(), Inst::Vop1(inst)));
    }
    for &op in Vop2Op::ALL {
        out.push((
            "VOP2",
            op.name().into(),
            Inst::Vop2(Vop2 {
                op,
                vdst: Src::Vgpr(10),
                src0: Src::Vgpr(12),
                src1: Src::Vgpr(14),
                ext: VopExt::None,
            }),
        ));
    }
    for &op in VopcOp::ALL {
        out.push((
            "VOPC",
            op.name().into(),
            Inst::Vopc(Vopc {
                op,
                src0: Src::Vgpr(12),
                src1: Src::Vgpr(14),
                ext: VopExt::None,
            }),
        ));
    }
    for op in Vop3Op::all(ARCH) {
        if !matches!(op, Vop3Op::Only(_)) {
            continue;
        }
        let (d, a, b, c) = op.widths();
        use peacemaker_cdna::enc::vop3::Vop3OnlyOp::{ReadlaneB32, WritelaneB32};
        let (dst, src) = match op {
            // Lane select (and writelane data) are scalar operands.
            Vop3Op::Only(ReadlaneB32) => (
                Src::Sgpr(10),
                [Some(Src::Vgpr(12)), Some(Src::Int(3)), None],
            ),
            Vop3Op::Only(WritelaneB32) => (
                Src::Vgpr(10),
                [Some(Src::Sgpr(13)), Some(Src::Int(3)), None],
            ),
            _ => (Src::Vgpr(10), [vreg(12, a), vreg(14, b), vreg(16, c)]),
        };
        let sdst = op.is_vop3b().then_some(Src::Sgpr(20));
        let _ = d;
        out.push((
            "VOP3",
            op.name().into(),
            Inst::Vop3(Vop3 {
                op,
                dst,
                sdst,
                src,
                abs: 0,
                neg: 0,
                clamp: false,
                omod: 0,
                op_sel: 0,
            }),
        ));
    }
    for &op in Vop3pOp::ALL {
        let (_, a, b, c) = op.widths();
        out.push((
            "VOP3P",
            op.name().into(),
            Inst::Vop3p(Vop3p {
                op,
                dst: Src::Vgpr(10),
                src: [vreg(12, a), vreg(14, b), vreg(16, c)],
                op_sel: 0,
                op_sel_hi: 7,
                neg_lo: 0,
                neg_hi: 0,
                clamp: false,
            }),
        ));
    }
    for &op in MaiOp::ALL {
        let inst = match op {
            MaiOp::AccvgprReadB32 => Mai {
                op,
                dst: Src::Vgpr(10),
                src_a: Src::Agpr(12),
                src_b: None,
                src_c: None,
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
            MaiOp::AccvgprWriteB32 => Mai {
                op,
                dst: Src::Agpr(10),
                src_a: Src::Vgpr(12),
                src_b: None,
                src_c: None,
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
            _ => Mai {
                op,
                dst: Src::Agpr(32),
                src_a: Src::Vgpr(12),
                src_b: Some(Src::Vgpr(14)),
                src_c: Some(Src::Agpr(32)),
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
        };
        out.push(("MAI", op.name().into(), Inst::Mai(inst)));
    }
    for &op in DsOp::ALL {
        let (d, a, b) = op.widths();
        out.push((
            "DS",
            op.name().into(),
            Inst::Ds(Ds {
                op,
                vdst: vreg(20, d),
                addr: (op != DsOp::Nop).then_some(Src::Vgpr(12)),
                data0: vreg(14, a),
                data1: vreg(16, b),
                offset0: 0,
                offset1: 0,
                gds: false,
            }),
        ));
    }
    for &op in FlatOp::ALL {
        let (d, data) = op.widths();
        out.push((
            "GLOBAL",
            op.name(FlatSeg::Global).unwrap().into(),
            Inst::Flat(Flat {
                op,
                seg: FlatSeg::Global,
                vdst: vreg(20, d),
                addr: Some(Src::Vgpr(12)),
                data: vreg(14, data),
                saddr: Some(Src::Sgpr(2)),
                offset: 0,
                sc0: d > 0,
                sc1: false,
                nt: false,
            }),
        ));
    }
    for &op in MubufOp::ALL {
        out.push((
            "MUBUF",
            op.name().into(),
            Inst::Mubuf(Mubuf {
                op,
                vdata: Src::Vgpr(20),
                vaddr: Some(Src::Vgpr(12)),
                srsrc: Src::Sgpr(4),
                soffset: Src::Int(0),
                offset: 0,
                offen: true,
                idxen: false,
                sc0: false,
                sc1: false,
                nt: false,
            }),
        ));
    }
    out
}

#[test]
fn coverage_table() {
    let cfg = Config::default();
    let mut rows = Vec::new();
    let mut uncovered = Vec::new();
    for (format, name, inst) in cases() {
        let mut w = wave();
        let mut lds = vec![0u8; 64 * 1024];
        let mut mem = memory();
        let result = sim::exec_one(&mut w, &inst, &mut lds, &mut mem, &cfg)
            .and_then(|_| sim::mem::drain(&mut w, &mut lds, &mut mem));
        let status = match &result {
            Ok(()) => "yes".to_string(),
            Err(SimError::Unsupported(m)) => {
                uncovered.push(name.clone());
                format!("no — {m}")
            }
            Err(e) => format!("yes (synthetic operands: {e})"),
        };
        rows.push(format!("| {format} | `{name}` | {status} |"));
    }
    let table = format!(
        "| format | instruction | emulated |\n|---|---|---|\n{}\n\n{} of {} opcodes emulated.\n",
        rows.join("\n"),
        rows.len() - uncovered.len(),
        rows.len()
    );
    println!("{table}");
    let _ = std::fs::write(
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("sim_coverage.md"),
        &table,
    );
    // Opcodes deliberately left to `SimError::Unsupported` (each documented in its executor).
    // getpc/setpc: the wave has an instruction index, not a byte PC; sendmsg/trap: no
    // message or trap-handler model; memtime/memrealtime/dcache_inv: no clock or cache model.
    let mut expected_unsupported: Vec<&str> = vec![
        "s_dcache_inv",
        "s_getpc_b64",
        "s_memrealtime",
        "s_memtime",
        "s_sendmsg",
        "s_setpc_b64",
        "s_trap",
    ];
    expected_unsupported.sort_unstable();
    let mut got: Vec<&str> = uncovered.iter().map(String::as_str).collect();
    got.sort_unstable();
    assert_eq!(
        got, expected_unsupported,
        "the set of unsupported opcodes changed; update the documented list"
    );
}
