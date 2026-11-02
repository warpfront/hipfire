// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! V8 simulator features: neighbour data memory/lock views and `vst.srs.4x` int8 SRS stores.
//! Instruction slots are encoded from the `peacemaker-npu` `gen.rs` encodings (llvm-aie AIE2P TableGen).
use peacemaker_npu::sim::{
    config::{Config, TxnOp},
    decode::PreparedDecoder,
    srs_lane, Error, Neighbour, Neighbours, Step, Tile,
};
use peacemaker_npu::{
    cdo::Cmd,
    isa::{
        self, Inst, Program,
        Reg::{M, P, R},
    },
    regs,
};

const NOPS: usize = 16;
fn movxm_s(n: u64, i: u32) -> Inst {
    // OP_mMvSclDstCg encodings of s0..s3 (REGS_24 in gen.rs): 14, 46, 78, 110.
    let i = u64::from(i);
    Inst::Lng(((i >> 12) << 22) | ((14 + 32 * n) << 15) | ((i & 0xfff) << 3) | 0b001)
}
fn movx_cr_imm(cr: u64, imm: u64) -> Inst {
    Inst::Alu(0x3000 | (cr << 7) | (imm << 15))
}
fn movx_cr_r(cr: u64, r: u64) -> Inst {
    Inst::Alu(0x7000 | (cr << 7) | (r << 15))
}
const CR_SAT: u64 = 14;
const CR_RND: u64 = 10;
const CR_SRS_MODE: u64 = 12;
/// `vst.srs.4x dm{src}, s{su}, srssign{sign}, [p{ptr}], #imm` (pstm_nrm_imm); imm is in 64-byte steps.
fn srs_post(src: u64, su: u64, ptr: u64, imm: u64, sign: u64) -> Inst {
    Inst::St((ptr << 17) | (imm << 13) | 0x1802 | (sign << 3) | (src << 8) | (su << 4))
}
/// `vst.srs.4x ... [p{ptr}, #imm]` (idx_imm): no pointer update.
fn srs_idx(src: u64, su: u64, ptr: u64, imm: u64, sign: u64) -> Inst {
    Inst::St((ptr << 17) | (imm << 13) | 0x802 | (sign << 3) | (src << 8) | (su << 4))
}
/// `vst.srs.4x ... [p{ptr}], m{m}` (pstm_nrm).
fn srs_mod(src: u64, su: u64, ptr: u64, m: u64, sign: u64) -> Inst {
    Inst::St((ptr << 17) | (m << 14) | 0x1002 | (sign << 3) | (src << 8) | (su << 4))
}
fn vlda_bm(bm: u64, ptr: u64, imm: u64) -> Inst {
    Inst::Lda((ptr << 17) | (imm << 13) | (0b11 << 11) | (bm << 6) | 0b101011)
}
fn vst_bm(bm: u64, ptr: u64, imm: u64) -> Inst {
    Inst::St((ptr << 17) | (imm << 13) | (0b11 << 11) | (bm << 6) | 0b001101)
}
fn cat(p: &mut Program, ops: &[Inst]) {
    for &op in ops {
        p.push(op).nops(NOPS);
    }
}

// ---------------------------------------------------------------- SRS
/// Independent scalar reference: choose between the two integers around v/2^s from the exact
/// fraction (compared as 2*rest vs 2^s), then saturate.
fn reference(v: i32, s: u32, rnd: u32, sat: u32) -> i8 {
    let v = i128::from(v);
    let unit = 1i128 << s;
    let lo = v.div_euclid(unit);
    let rest = v.rem_euclid(unit);
    let hi = lo + 1;
    let exact = rest == 0;
    let tie = !exact && 2 * rest == unit;
    let above = 2 * rest > unit;
    let up_on_tie = match rnd {
        8 => false,
        9 => true,
        10 => v < 0,
        11 => v >= 0,
        12 => lo % 2 != 0,
        13 => lo % 2 == 0,
        _ => false,
    };
    let q = match rnd {
        0 => lo,
        1 => {
            if exact {
                lo
            } else {
                hi
            }
        }
        2 => {
            if exact || v >= 0 {
                lo
            } else {
                hi
            }
        }
        3 => {
            if exact || v < 0 {
                lo
            } else {
                hi
            }
        }
        _ => {
            if above || (tie && up_on_tie) {
                hi
            } else {
                lo
            }
        }
    };
    match sat {
        0 => q as i8,
        1 => q.clamp(-128, 127) as i8,
        3 => q.clamp(-127, 127) as i8,
        _ => unreachable!(),
    }
}
fn all_lanes(shift: u32) -> Vec<[i32; 64]> {
    // Many lane patterns so every rounding tie/sign/saturation combination lands in some dm.
    let unit = 1i64 << shift;
    let half = unit / 2;
    let mut flat: Vec<i32> = Vec::new();
    for q in [
        0i64,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        125,
        126,
        127,
        128,
        129,
        -125,
        -126,
        -127,
        -128,
        -129,
        -130,
        255,
        256,
        -256,
        -257,
        1 << 20,
        -(1 << 20),
    ] {
        for d in [
            0,
            half,
            half.saturating_sub(1),
            half + 1,
            unit - 1,
            1,
            unit / 4,
        ] {
            let v = q * unit + d;
            flat.push(v.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
        }
    }
    flat.extend([i32::MAX, i32::MIN, i32::MAX - 1, i32::MIN + 1]);
    flat.chunks(64)
        .map(|c| {
            let mut a = [0i32; 64];
            a[..c.len()].copy_from_slice(c);
            a
        })
        .collect()
}
fn srs_program(shift: u32, sat: u64, rnd: u64, sign: u64) -> Program {
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(P(0), 0x70000),
            movxm_s(0, shift),
            movx_cr_imm(CR_SAT, sat),
            movx_cr_imm(CR_RND, rnd),
            movx_cr_imm(CR_SRS_MODE, 0),
        ],
    );
    cat(
        &mut p,
        &[
            srs_post(0, 0, 0, 1, sign),
            srs_idx(1, 0, 0, 1, sign),
            isa::movxm(M(0), 128),
            srs_mod(2, 0, 0, 0, sign),
            isa::done(),
        ],
    );
    p
}
fn run_srs(shift: u32, sat: u64, rnd: u64, sign: u64, dm: &[[i32; 64]]) -> Result<Tile, Error> {
    let bytes = srs_program(shift, sat, rnd, sign).finish();
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    for (i, d) in dm.iter().enumerate().take(3) {
        tile.dm[i] = *d;
    }
    tile.run(&decoder, 10_000).map(|step| {
        assert_eq!(step, Step::Done);
        tile
    })
}
#[test]
fn srs_store_matches_scalar_reference_every_mode() {
    let mut cases = 0;
    for shift in [0u32, 5, 12, 31] {
        let groups = all_lanes(shift);
        for rnd in [0u64, 1, 2, 3, 8, 9, 10, 11, 12, 13] {
            for sat in [0u64, 1, 3] {
                // Three consecutive dm groups per run: post-immediate, idx_imm and modifier stores.
                let n = groups.len();
                for g in 0..n {
                    let dm = [groups[g], groups[(g + 1) % n], groups[(g + 2) % n]];
                    let tile = run_srs(shift, sat, rnd, 1, &dm).unwrap();
                    // dm0 -> [0,64) post-modify (p0 += 64); dm1 -> [128,192) idx_imm 64 (no update); dm2 -> [64,128) via m0 at the new p0 (p0 += 128).
                    assert_eq!(tile.p[0], 0x70000 + 64 + 128, "pointer updates");
                    for (k, dmk) in dm.iter().enumerate() {
                        for lane in 0..64 {
                            let want = reference(dmk[lane], shift, rnd as u32, sat as u32) as u8;
                            assert_eq!(
                                tile.memory[[0, 128, 64][k] + lane],
                                want,
                                "shift {shift} rnd {rnd} sat {sat} dm{k} lane {lane} value {}",
                                dmk[lane]
                            );
                            cases += 1;
                        }
                    }
                    assert!(
                        tile.memory[192..256].iter().all(|&b| b == 0),
                        "nothing written past three stores"
                    );
                }
            }
        }
    }
    assert!(cases > 50_000, "{cases}");
}
#[test]
fn srs_floor_saturate_documented_values() {
    // crSat=1, crRnd=0 (the V8 epilogue): out = clamp(floor(c / 2^shift), -128, 127).
    for shift in [0u32, 5, 12, 31] {
        let unit = 1i64 << shift;
        let mut dm = [0i32; 64];
        let probes: [i64; 8] = [
            0,
            -1,
            unit - 1,
            unit,
            -unit,
            -unit - 1,
            127 * unit + unit - 1,
            128 * unit,
        ];
        for (i, v) in probes.iter().enumerate() {
            dm[i] = (*v).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        dm[8] = i32::MAX;
        dm[9] = i32::MIN;
        let tile = run_srs(shift, 1, 0, 1, &[dm, dm, dm]).unwrap();
        for lane in 0..10 {
            let c = i64::from(dm[lane]);
            let expect = (c >> shift).clamp(-128, 127) as i8;
            assert_eq!(
                tile.memory[lane] as i8, expect,
                "shift {shift} lane {lane} c {c}"
            );
        }
    }
    // Hand-written anchors, independent of both implementations.
    let mut dm = [0i32; 64];
    dm[0] = -1;
    dm[1] = 31;
    dm[2] = 32;
    dm[3] = -33;
    dm[4] = 4095;
    dm[5] = -4096;
    dm[6] = 1 << 20;
    dm[7] = -(1 << 20);
    let tile = run_srs(5, 1, 0, 1, &[dm, dm, dm]).unwrap();
    assert_eq!(&tile.memory[..8], [0xff, 0, 1, 0xfe, 127, 0x80, 127, 0x80]);
    // Ties: 2.5 -> floor 2 / ceil 3 / pos_inf 3 / neg_inf 2 / sym_zero 2 / sym_inf 3 / even 2 / odd 3 at shift 1.
    let mut dm = [0i32; 64];
    dm[0] = 5;
    dm[1] = -5;
    dm[2] = 3;
    dm[3] = -3;
    dm[4] = 4;
    dm[5] = -1;
    let expect: [(u64, [i8; 6]); 10] = [
        (0, [2, -3, 1, -2, 2, -1]),
        (1, [3, -2, 2, -1, 2, 0]),
        (2, [2, -2, 1, -1, 2, 0]),
        (3, [3, -3, 2, -2, 2, -1]),
        (8, [2, -3, 1, -2, 2, -1]),
        (9, [3, -2, 2, -1, 2, 0]),
        (10, [2, -2, 1, -1, 2, 0]),
        (11, [3, -3, 2, -2, 2, -1]),
        (12, [2, -2, 2, -2, 2, 0]),
        (13, [3, -3, 1, -1, 2, -1]),
    ];
    for (rnd, want) in expect {
        let tile = run_srs(1, 1, rnd, 1, &[dm, dm, dm]).unwrap();
        let got: Vec<i8> = tile.memory[..6].iter().map(|&b| b as i8).collect();
        assert_eq!(got, want, "rnd {rnd}");
    }
}
#[test]
fn srs_wrap_and_symmetric_saturation() {
    let mut dm = [0i32; 64];
    dm[0] = 300;
    dm[1] = -300;
    dm[2] = -128;
    dm[3] = -129;
    dm[4] = 127;
    dm[5] = 128;
    let wrap = run_srs(0, 0, 0, 1, &[dm, dm, dm]).unwrap();
    assert_eq!(&wrap.memory[..6], [44, 0xd4, 0x80, 0x7f, 127, 0x80]); // low 8 bits
    let sym = run_srs(0, 3, 0, 1, &[dm, dm, dm]).unwrap();
    assert_eq!(&sym.memory[..6], [127, 0x81, 0x81, 0x81, 127, 127]); // [-127,127]
    let sat = run_srs(0, 1, 0, 1, &[dm, dm, dm]).unwrap();
    assert_eq!(&sat.memory[..6], [127, 0x80, 0x80, 0x80, 127, 127]);
}
#[test]
fn srs_unsourced_modes_are_unsupported() {
    let dm = [[1i32; 64]; 3];
    assert_eq!(
        run_srs(0, 1, 0, 0, &dm).err(),
        Some(Error::Unsupported),
        "srsSign0 (unsigned)"
    );
    assert_eq!(
        run_srs(0, 2, 0, 1, &dm).err(),
        Some(Error::Unsupported),
        "crSat=2"
    );
    for rnd in [4u64, 5, 6, 7, 14, 15] {
        assert_eq!(
            run_srs(0, 1, rnd, 1, &dm).err(),
            Some(Error::Unsupported),
            "crRnd={rnd}"
        );
    }
    assert_eq!(
        run_srs(32, 1, 0, 1, &dm).err(),
        Some(Error::Unsupported),
        "shift 32"
    );
    assert_eq!(srs_lane(0, 31, 0, 1), Ok(0));
    assert_eq!(srs_lane(0, 32, 0, 1), Err(Error::Unsupported));
    // Misaligned destination.
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(P(0), 0x70020),
            movxm_s(0, 0),
            srs_post(0, 0, 0, 1, 1),
            isa::done(),
        ],
    );
    let bytes = p.finish();
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    assert_eq!(tile.run(&decoder, 10_000).err(), Some(Error::Alignment));
}
#[test]
fn control_and_shift_register_writes() {
    // movx cr from an r register masks to the register width (crSat 2 bits, crRnd 4, crSRSMode 1);
    // movxm / mov imm / mov r write s0..s3.
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(R(1), 0x35),
            movx_cr_r(CR_SAT, 1),
            movx_cr_r(CR_RND, 1),
            movx_cr_r(CR_SRS_MODE, 1),
            movxm_s(1, 0x12345678),
            Inst::Mv((46 << 15) | ((0x7ffu64 & (-3i64 as u64)) << 4) | 0x5), // mov s1, #-3 (MOV_alu_mv_mv_mv_cg)
            Inst::Mv((78 << 15) | (5 << 8) | 0x7), // mov s2, r1 (MOV_alu_mv_mv_mv_scl; r1 = 5 in mMvSclSrc)
            isa::done(),
        ],
    );
    let bytes = p.finish();
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    tile.run(&decoder, 10_000).unwrap();
    assert_eq!(
        (tile.cr_sat, tile.cr_rnd, tile.cr_srs_mode),
        (0x35 & 3, 0x35 & 15, 1)
    );
    assert_eq!(tile.s[1], -3i32 as u32);
    assert_eq!(tile.s[2], 0x35);
    let mut p = Program::new();
    cat(&mut p, &[movxm_s(0, 0x12345678), isa::done()]);
    let bytes = p.finish();
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    tile.run(&decoder, 10_000).unwrap();
    assert_eq!(tile.s[0], 0x12345678);
}

// ---------------------------------------------------------------- neighbours
fn finish(p: Program) -> (Vec<u8>, PreparedDecoder) {
    let bytes = p.finish();
    let d = PreparedDecoder::new(&bytes).unwrap();
    (bytes, d)
}
/// Vertical pair: `lower` (physical row 2) has `upper` (row 3) to its North and no South/West.
fn lower_program() -> Program {
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(P(0), 0x60000),
            isa::movxm(P(1), 0x70000),
            isa::movxm(P(2), 0x60040),
            isa::movxm(P(3), 0x70100),
            isa::movxm(R(1), 0xffff_ffff),
            isa::movxm(R(2), 1),
            isa::movxm(R(5), 0xbeef),
        ],
    );
    cat(
        &mut p,
        &[
            isa::acq(32, R(1)), // upper's lock 0 through the North lock view: blocks until upper releases it
            isa::lda_idx_imm(R(3), P(0), 0x10),
            isa::st_idx_imm(R(3), P(1), 0x14),
            isa::st_idx_imm(R(5), P(0), 0x18),
            vlda_bm(0, 2, 1),
            vst_bm(0, 3, 1),
            vst_bm(0, 2, 1), // acc load from North, own store, acc store to North
            isa::rel(33, R(2)),
            isa::done(),
        ],
    );
    p
}
fn upper_program() -> Program {
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(P(0), 0x70000),
            isa::movxm(P(1), 0x40000),
            isa::movxm(R(1), 0xffff_ffff),
            isa::movxm(R(2), 1),
            isa::movxm(R(6), 0xcafe_0001),
        ],
    );
    cat(&mut p, &[isa::st_idx_imm(R(6), P(0), 0x10)]);
    p.nops(300); // lower reaches its blocking acq(32) first
    cat(
        &mut p,
        &[
            isa::rel(48, R(2)), // own lock 0
            isa::acq(3, R(1)),  // lower's lock 3 through the South lock view
            isa::lda_idx_imm(R(7), P(1), 0x08),
            isa::st_idx_imm(R(7), P(0), 0x14), // read lower memory through South
            isa::rel(4, R(2)),                 // lower's lock 4 through South
            isa::acq(49, R(1)), // own lock 1: released by lower through the North view
            isa::lda_idx_imm(R(8), P(0), 0x18),
            isa::st_idx_imm(R(8), P(0), 0x1c),
            isa::done(),
        ],
    );
    p
}
#[test]
fn neighbour_memory_and_locks_between_vertical_pair() {
    let (lbytes, ld) = finish(lower_program());
    let (ubytes, ud) = finish(upper_program());
    let mut lower = Tile::new(lbytes).unwrap();
    let mut upper = Tile::new(ubytes).unwrap();
    lower.memory[8..12].copy_from_slice(&0x5a5a_0008u32.to_le_bytes());
    lower.locks[3] = 1;
    for i in 0..64 {
        upper.memory[0x40 + i] = (i * 3 + 1) as u8;
    }
    let (mut lblocked, mut ldone, mut udone, mut ublocked) = (0, false, false, 0);
    for _ in 0..20_000 {
        if !ldone {
            let mut nb = Neighbours {
                north: Some(Neighbour {
                    memory: &mut upper.memory,
                    locks: &mut upper.locks,
                }),
                ..Default::default()
            };
            match lower.step_with(&ld, &mut nb).unwrap() {
                Step::Blocked => lblocked += 1,
                Step::Done => ldone = true,
                Step::Advanced => {}
            }
        }
        if !udone {
            let mut nb = Neighbours {
                south: Some(Neighbour {
                    memory: &mut lower.memory,
                    locks: &mut lower.locks,
                }),
                ..Default::default()
            };
            match upper.step_with(&ud, &mut nb).unwrap() {
                Step::Blocked => ublocked += 1,
                Step::Done => udone = true,
                Step::Advanced => {}
            }
        }
        if ldone && udone {
            break;
        }
    }
    assert!(ldone && udone);
    assert!(
        lblocked > 0 && ublocked > 0,
        "lock acquires through neighbour views must block ({lblocked},{ublocked})"
    );
    let word = |t: &Tile, a: usize| u32::from_le_bytes(t.memory[a..a + 4].try_into().unwrap());
    assert_eq!(
        word(&lower, 0x14),
        0xcafe_0001,
        "lower read upper memory through North"
    );
    assert_eq!(
        word(&upper, 0x18),
        0xbeef,
        "lower wrote upper memory through North"
    );
    assert_eq!(
        word(&upper, 0x1c),
        0xbeef,
        "upper observed the neighbour store after the lock handoff"
    );
    assert_eq!(
        word(&upper, 0x14),
        0x5a5a_0008,
        "upper read lower memory through South"
    );
    // Accumulator bm quarter: upper[0x40..0x80] -> lower[0x100..0x140] (own store) and -> upper[0x80..0xc0] (North store).
    let src: Vec<u8> = (0..64).map(|i| (i * 3 + 1) as u8).collect();
    assert_eq!(&lower.memory[0x100..0x140], &src[..]);
    assert_eq!(&upper.memory[0x80..0xc0], &src[..]);
    // Locks: upper's lock 0 consumed by lower's acq (-1) after upper's own rel (+1); lock 1 released once by lower.
    assert_eq!(upper.locks[0], 0);
    assert_eq!(upper.locks[1], 0, "released by lower, consumed by upper");
    assert_eq!(lower.locks[3], 0);
    assert_eq!(lower.locks[4], 1, "released by upper through South");
}
#[test]
fn missing_neighbours_and_plain_step_are_bounds_errors() {
    let probes: [(&str, Inst); 4] = [
        ("scalar load south", isa::lda_idx_imm(R(3), P(0), 0)),
        ("scalar store west", isa::st_idx_imm(R(3), P(0), 0)),
        ("vector load north", vlda_bm(0, 0, 1)),
        ("vector store south", vst_bm(0, 0, 1)),
    ];
    for (base, view) in [(0x40000u32, "south"), (0x50000, "west"), (0x60000, "north")] {
        for (name, op) in probes {
            let mut p = Program::new();
            cat(&mut p, &[isa::movxm(P(0), base), op, isa::done()]);
            let (bytes, d) = finish(p);
            let mut tile = Tile::new(bytes).unwrap();
            assert_eq!(
                tile.run(&d, 2000).err(),
                Some(Error::Bounds),
                "{name} via {view} with no neighbours"
            );
        }
    }
    // A present neighbour on one side does not make the other sides reachable.
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(P(0), 0x60000),
            isa::lda_idx_imm(R(3), P(0), 0),
            isa::done(),
        ],
    );
    let (bytes, d) = finish(p);
    let mut tile = Tile::new(bytes).unwrap();
    let (mut mem, mut locks) = (vec![0u8; 65536], [0i32; 16]);
    let mut nb = Neighbours {
        south: Some(Neighbour {
            memory: &mut mem,
            locks: &mut locks,
        }),
        ..Default::default()
    };
    assert_eq!(tile.run_with(&d, &mut nb, 2000).err(), Some(Error::Bounds));
    // Addresses outside every window.
    for addr in [0x3ffffu32, 0x80000, 0x0] {
        let mut p = Program::new();
        cat(
            &mut p,
            &[
                isa::movxm(P(0), addr),
                isa::lda_idx_imm(R(3), P(0), 0),
                isa::done(),
            ],
        );
        let (bytes, d) = finish(p);
        let mut tile = Tile::new(bytes).unwrap();
        assert_eq!(tile.run(&d, 2000).err(), Some(Error::Bounds), "{addr:#x}");
    }
    // Neighbour lock views without a neighbour; own view (48..63) always works; ids >= 64 are impossible in 6 bits.
    for id in [0u64, 15, 16, 31, 32, 47] {
        let mut p = Program::new();
        cat(
            &mut p,
            &[isa::movxm(R(2), 1), isa::rel(id, R(2)), isa::done()],
        );
        let (bytes, d) = finish(p);
        let mut tile = Tile::new(bytes).unwrap();
        assert_eq!(tile.run(&d, 2000).err(), Some(Error::Bounds), "rel id {id}");
        let mut p = Program::new();
        cat(
            &mut p,
            &[
                isa::movxm(R(2), 0xffff_ffff),
                isa::acq(id, R(2)),
                isa::done(),
            ],
        );
        let (bytes, d) = finish(p);
        let mut tile = Tile::new(bytes).unwrap();
        assert_eq!(tile.run(&d, 2000).err(), Some(Error::Bounds), "acq id {id}");
    }
    let mut p = Program::new();
    cat(
        &mut p,
        &[
            isa::movxm(R(2), 2),
            isa::rel(48, R(2)),
            isa::rel(63, R(2)),
            isa::done(),
        ],
    );
    let (bytes, d) = finish(p);
    let mut tile = Tile::new(bytes).unwrap();
    assert_eq!(tile.run(&d, 2000).unwrap(), Step::Done);
    assert_eq!((tile.locks[0], tile.locks[15]), (2, 2));
}

// ---------------------------------------------------------------- Config (array) wiring
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}
fn load(cmds: &mut Vec<Cmd>, col: u32, row: u32, p: Program, locks: &[(u32, u32)]) {
    cmds.push(Cmd::DmaWrite(
        u64::from(regs::tile(col, row, regs::core::PROGRAM_MEMORY)),
        words(&p.finish()),
    ));
    for &(id, v) in locks {
        cmds.push(Cmd::Write(
            regs::tile(col, row, regs::core::LOCK0_VALUE + 16 * id),
            v,
        ));
    }
}
#[test]
fn config_lends_north_south_and_west_neighbour_cores() {
    // Chain through three cores of a 2x2 core block: B=(0,3) -> A=(0,2) [A reads B via North] -> C=(1,2) [C reads A via West]
    // and C writes A's memory through its West view, B reads A through its South view.
    let mut b = Program::new();
    cat(
        &mut b,
        &[
            isa::movxm(P(0), 0x70000),
            isa::movxm(P(1), 0x40000),
            isa::movxm(R(1), 0xffff_ffff),
            isa::movxm(R(2), 1),
            isa::movxm(R(6), 0x1111_0001),
        ],
    );
    cat(
        &mut b,
        &[
            isa::st_idx_imm(R(6), P(0), 0x10),
            isa::rel(48, R(2)), // own lock 0: A is waiting on it through North
            isa::acq(49, R(1)), // own lock 1: A releases it through North after copying
            isa::lda_idx_imm(R(7), P(1), 0x14),
            isa::st_idx_imm(R(7), P(0), 0x18),
            isa::done(),
        ],
    ); // A's own[0x14] via South
    let mut a = Program::new();
    cat(
        &mut a,
        &[
            isa::movxm(P(0), 0x60000),
            isa::movxm(P(1), 0x70000),
            isa::movxm(R(1), 0xffff_ffff),
            isa::movxm(R(2), 1),
        ],
    );
    cat(
        &mut a,
        &[
            isa::acq(32, R(1)),
            isa::lda_idx_imm(R(3), P(0), 0x10),
            isa::st_idx_imm(R(3), P(1), 0x14),
            isa::rel(33, R(2)), // copy B[0x10] -> A[0x14], release B lock 1
            isa::rel(48, R(2)),
            isa::acq(50, R(1)),
            isa::done(),
        ],
    ); // publish own lock 0 for C, wait for C's release of own lock 2
    let mut c = Program::new();
    cat(
        &mut c,
        &[
            isa::movxm(P(0), 0x50000),
            isa::movxm(P(1), 0x70000),
            isa::movxm(R(1), 0xffff_ffff),
            isa::movxm(R(2), 1),
            isa::movxm(R(5), 0x77),
        ],
    );
    cat(
        &mut c,
        &[
            isa::acq(16, R(1)),
            isa::lda_idx_imm(R(3), P(0), 0x14),
            isa::st_idx_imm(R(3), P(1), 0x1c),
            isa::st_idx_imm(R(5), P(0), 0x18),
            isa::rel(18, R(2)),
            isa::done(),
        ],
    );
    let mut cmds = Vec::new();
    load(&mut cmds, 0, 3, b, &[]);
    load(&mut cmds, 0, 2, a, &[]);
    load(&mut cmds, 1, 2, c, &[]);
    for (col, row) in [(0, 3), (0, 2), (1, 2)] {
        cmds.push(Cmd::Write(
            regs::tile(col, row, regs::core::CORE_CONTROL),
            1,
        ));
    }
    let mut sim = Config::new();
    sim.load_cdo(&cmds).unwrap();
    let done = |col, row| {
        TxnOp::MaskPoll(
            regs::tile(col, row, regs::core::CORE_STATUS),
            1 << 20,
            1 << 20,
        )
    };
    let mem = |col, row, off, v| TxnOp::MaskPoll(regs::tile(col, row, off), u32::MAX, v);
    sim.execute(&[done(0, 3), done(0, 2), done(1, 2)], &mut [])
        .unwrap();
    sim.execute(
        &[
            mem(0, 2, 0x14, 0x1111_0001),
            mem(1, 2, 0x1c, 0x1111_0001),
            mem(0, 2, 0x18, 0x77),
            mem(0, 3, 0x18, 0x1111_0001),
        ],
        &mut [],
    )
    .unwrap();
    // C's West store changed A's memory only: C's own offset 0x18 stayed 0.
    sim.execute(&[mem(1, 2, 0x18, 0)], &mut []).unwrap();
}
