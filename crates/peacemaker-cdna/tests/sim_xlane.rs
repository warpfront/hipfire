// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `sim::xlane`: DPP source-lane selection (CDNA3 ISA §13.3.9 Table 93
//! p.544-545), ROW_MASK/BANK_MASK (p.544), SDWA select/unused (§13.3.7
//! p.543), `ds_swizzle_b32` (§12.12 p.427), `ds_permute_b32` / `ds_bpermute_b32`
//! (p.427-429). Expected values are written out from the ISA text, not
//! recomputed with the implementation's formulas.
use peacemaker_cdna::enc::vop::{Dpp, DppCtrl, SdwaSel, SdwaUnused};
use peacemaker_cdna::sim::xlane::*;
use peacemaker_cdna::sim::{SimError, Wave};
use peacemaker_cdna::{Arch, WAVE};

/// `-1` = the control defines no source lane (Table 93: "use bound_cntl").
const N: i8 = -1;

fn src(ctrl: DppCtrl, lane: usize) -> Option<usize> {
    dpp_source_lane(ctrl, lane).expect("legal control")
}

/// `row[k]` = expected source lane offset (inside the destination's row) of
/// lane `k` of every row; `-1` = no source.
fn check_row_local(ctrl: DppCtrl, row: [i8; 16]) {
    for r in 0..4 {
        for k in 0..16 {
            let want = if row[k] < 0 {
                None
            } else {
                Some(r * 16 + row[k] as usize)
            };
            assert_eq!(src(ctrl, r * 16 + k), want, "{ctrl:?} row {r} lane {k}");
        }
    }
}

/// Full 64-lane expectation, `-1` = no source.
fn check_wave(ctrl: DppCtrl, want: [i8; WAVE]) {
    for (n, w) in want.iter().enumerate() {
        let want = if *w < 0 { None } else { Some(*w as usize) };
        assert_eq!(src(ctrl, n), want, "{ctrl:?} lane {n}");
    }
}

fn dpp(ctrl: DppCtrl) -> Dpp {
    Dpp {
        ctrl,
        row_mask: 0xf,
        bank_mask: 0xf,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    }
}

/// A wave whose v1 and a1 hold `1000 + lane` / `2000 + lane`.
fn wave() -> Wave {
    let mut w = Wave::new(Arch::Gfx942, 0, 0xDEAD_BEEF);
    for l in 0..WAVE {
        w.v[1][l] = 1000 + l as u32;
        w.a[1][l] = 2000 + l as u32;
    }
    w
}

#[test]
fn quad_perm_tables() {
    // pix[n] = pix[(n & 0x3c) + sel[n % 4]]
    let cases: [([u8; 4], [usize; 4]); 4] = [
        ([2, 0, 3, 1], [2, 0, 3, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([3, 2, 1, 0], [3, 2, 1, 0]),
        ([1, 1, 3, 3], [1, 1, 3, 3]),
    ];
    for (sel, off) in cases {
        for n in 0..WAVE {
            assert_eq!(
                src(DppCtrl::QuadPerm(sel), n),
                Some(n / 4 * 4 + off[n % 4]),
                "quad_perm {sel:?} lane {n}"
            );
        }
    }
    // Spot check against the raw encoding: DPP_CTRL 0x4E = [2,3,0,1] -> lane 0 reads 2.
    assert_eq!(src(DppCtrl::QuadPerm([2, 3, 0, 1]), 0), Some(2));
    assert_eq!(src(DppCtrl::QuadPerm([2, 3, 0, 1]), 5), Some(7));
    assert_eq!(src(DppCtrl::QuadPerm([2, 3, 0, 1]), 62), Some(60));
}

#[test]
fn row_shl_tables() {
    check_row_local(
        DppCtrl::RowShl(1),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, N],
    );
    check_row_local(
        DppCtrl::RowShl(5),
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, N, N, N, N, N],
    );
    check_row_local(
        DppCtrl::RowShl(15),
        [15, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
    );
}

#[test]
fn row_shr_tables() {
    check_row_local(
        DppCtrl::RowShr(1),
        [N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    );
    check_row_local(
        DppCtrl::RowShr(4),
        [N, N, N, N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    );
    check_row_local(
        DppCtrl::RowShr(15),
        [N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, 0],
    );
}

#[test]
fn row_ror_tables() {
    check_row_local(
        DppCtrl::RowRor(1),
        [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    );
    check_row_local(
        DppCtrl::RowRor(4),
        [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    );
    check_row_local(
        DppCtrl::RowRor(15),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0],
    );
}

#[test]
fn wave_shift_rotate_tables() {
    let mut shl = [0i8; WAVE];
    let mut rol = [0i8; WAVE];
    let mut shr = [0i8; WAVE];
    let mut ror = [0i8; WAVE];
    for n in 0..WAVE as i8 {
        shl[n as usize] = if n < 63 { n + 1 } else { N };
        rol[n as usize] = if n < 63 { n + 1 } else { 0 };
        shr[n as usize] = if n > 0 { n - 1 } else { N };
        ror[n as usize] = if n > 0 { n - 1 } else { 63 };
    }
    // The tables above cross row and 32-lane boundaries on purpose.
    assert_eq!((shl[15], shl[31], shl[62], shl[63]), (16, 32, 63, N));
    assert_eq!((rol[15], rol[62], rol[63]), (16, 63, 0));
    assert_eq!((shr[0], shr[16], shr[63]), (N, 15, 62));
    assert_eq!((ror[0], ror[16], ror[63]), (63, 15, 62));
    check_wave(DppCtrl::WaveShl1, shl);
    check_wave(DppCtrl::WaveRol1, rol);
    check_wave(DppCtrl::WaveShr1, shr);
    check_wave(DppCtrl::WaveRor1, ror);
}

#[test]
fn mirror_tables() {
    // pix[n] = pix[15-(n&f)] within the row.
    check_row_local(
        DppCtrl::RowMirror,
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    );
    // pix[n] = pix[7-(n&7)] within each 8-lane half row.
    check_row_local(
        DppCtrl::RowHalfMirror,
        [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
    );
}

#[test]
fn row_bcast_tables() {
    // row_bcast:15: n>15 -> lane (n & 0x30) - 1, i.e. 15 -> row 1, 31 -> row 2, 47 -> row 3; row 0 unfed.
    let mut b15 = [N; WAVE];
    for n in 16..32 {
        b15[n] = 15;
    }
    for n in 32..48 {
        b15[n] = 31;
    }
    for n in 48..64 {
        b15[n] = 47;
    }
    check_wave(DppCtrl::RowBcast15, b15);
    // row_bcast:31: n>31 -> lane 31 for rows 2 and 3; rows 0 and 1 unfed.
    let mut b31 = [N; WAVE];
    for n in 32..64 {
        b31[n] = 31;
    }
    check_wave(DppCtrl::RowBcast31, b31);
}

#[test]
fn illegal_controls_error_and_never_panic() {
    for c in [
        DppCtrl::RowShl(0),
        DppCtrl::RowShl(16),
        DppCtrl::RowShr(0),
        DppCtrl::RowShr(16),
        DppCtrl::RowRor(0),
        DppCtrl::RowRor(16),
    ] {
        assert!(
            matches!(dpp_source_lane(c, 3), Err(SimError::Unsupported(_))),
            "{c:?}"
        );
    }
    assert!(dpp_source_lane(DppCtrl::QuadPerm([0, 1, 2, 4]), 3).is_err());
    assert!(
        dpp_source_lane(DppCtrl::QuadPerm([0, 1, 2, 4]), 0).is_ok(),
        "lane 0 only uses element 0"
    );
    assert!(dpp_source_lane(DppCtrl::RowMirror, 64).is_err());
    let mut w = wave();
    assert!(dpp_read(&w, &dpp(DppCtrl::RowMirror), 1, false, 64).is_err());
    w.vgpr_limit = 8;
    w.agpr_limit = 8;
    assert!(
        dpp_read(&w, &dpp(DppCtrl::RowMirror), 8, false, 0).is_err(),
        "beyond the allocated VGPRs"
    );
    assert!(
        dpp_read(&w, &dpp(DppCtrl::RowMirror), 8, true, 0).is_err(),
        "beyond the allocated AGPRs"
    );
    assert!(dpp_read(&w, &dpp(DppCtrl::RowMirror), 7, false, 0).is_ok());
}

#[test]
fn row_and_bank_mask_tables() {
    // ROW_MASK bit r clear disables lanes 16r..16r+15; BANK_MASK bit b clear disables lanes 4b..4b+3 of each row.
    for row_mask in 0..16u8 {
        for lane in 0..WAVE {
            let d = Dpp {
                row_mask,
                ..dpp(DppCtrl::RowMirror)
            };
            let want = match lane {
                0..=15 => row_mask & 1,
                16..=31 => row_mask & 2,
                32..=47 => row_mask & 4,
                _ => row_mask & 8,
            } != 0;
            assert_eq!(
                dpp_lane_enabled(&d, lane),
                want,
                "row_mask {row_mask:#x} lane {lane}"
            );
        }
    }
    for bank_mask in 0..16u8 {
        for lane in 0..WAVE {
            let d = Dpp {
                bank_mask,
                ..dpp(DppCtrl::RowMirror)
            };
            let want = match lane % 16 {
                0..=3 => bank_mask & 1,
                4..=7 => bank_mask & 2,
                8..=11 => bank_mask & 4,
                _ => bank_mask & 8,
            } != 0;
            assert_eq!(
                dpp_lane_enabled(&d, lane),
                want,
                "bank_mask {bank_mask:#x} lane {lane}"
            );
        }
    }
    // Spec examples: lanes[12:15, 28:31, 44:47, 60:63] disabled when bit 27 (bank bit 3) is 0.
    let d = Dpp {
        bank_mask: 0b0111,
        ..dpp(DppCtrl::RowMirror)
    };
    for l in [12, 15, 28, 31, 44, 47, 60, 63] {
        assert!(!dpp_lane_enabled(&d, l), "lane {l}");
    }
    for l in [0, 11, 16, 27, 32, 43, 48, 59] {
        assert!(dpp_lane_enabled(&d, l), "lane {l}");
    }
    assert!(!dpp_lane_enabled(&dpp(DppCtrl::RowMirror), 64));
}

#[test]
fn dpp_read_values_and_agpr() {
    let w = wave();
    // row_shl:1: lane 3 reads lane 4; lane 15 (row edge) has no source and BC=0 -> not written.
    let d = dpp(DppCtrl::RowShl(1));
    assert_eq!(dpp_read(&w, &d, 1, false, 3).unwrap(), Some(1004));
    assert_eq!(dpp_read(&w, &d, 1, false, 15).unwrap(), None);
    assert_eq!(dpp_read(&w, &d, 1, false, 16).unwrap(), Some(1017));
    // AGPR file.
    assert_eq!(dpp_read(&w, &d, 1, true, 3).unwrap(), Some(2004));
    // quad_perm [3,2,1,0]: lane 5 reads lane 6.
    assert_eq!(
        dpp_read(&w, &dpp(DppCtrl::QuadPerm([3, 2, 1, 0])), 1, false, 5).unwrap(),
        Some(1006)
    );
    // row_mirror: lane 17 reads lane 30.
    assert_eq!(
        dpp_read(&w, &dpp(DppCtrl::RowMirror), 1, false, 17).unwrap(),
        Some(1030)
    );
    // row_bcast:15, lane 20 reads lane 15; lane 40 reads lane 31; lane 60 reads lane 47.
    let b = dpp(DppCtrl::RowBcast15);
    assert_eq!(dpp_read(&w, &b, 1, false, 20).unwrap(), Some(1015));
    assert_eq!(dpp_read(&w, &b, 1, false, 40).unwrap(), Some(1031));
    assert_eq!(dpp_read(&w, &b, 1, false, 60).unwrap(), Some(1047));
    // The common scan form row_bcast:15 row_mask:0xa writes only rows 1 and 3.
    let b = Dpp {
        row_mask: 0xa,
        ..dpp(DppCtrl::RowBcast15)
    };
    assert_eq!(dpp_read(&w, &b, 1, false, 5).unwrap(), None);
    assert_eq!(dpp_read(&w, &b, 1, false, 20).unwrap(), Some(1015));
    assert_eq!(dpp_read(&w, &b, 1, false, 40).unwrap(), None);
    assert_eq!(dpp_read(&w, &b, 1, false, 60).unwrap(), Some(1047));
    // row_bcast:31 row_mask:0xc: rows 2 and 3 read lane 31.
    let b = Dpp {
        row_mask: 0xc,
        ..dpp(DppCtrl::RowBcast31)
    };
    assert_eq!(dpp_read(&w, &b, 1, false, 20).unwrap(), None);
    assert_eq!(dpp_read(&w, &b, 1, false, 33).unwrap(), Some(1031));
    assert_eq!(dpp_read(&w, &b, 1, false, 63).unwrap(), Some(1031));
}

#[test]
fn dpp_read_bound_ctrl_and_inactive_source() {
    let w = wave();
    // BOUND_CTRL = 1: no-source lanes write 0.
    let bc = Dpp {
        bound_ctrl: true,
        ..dpp(DppCtrl::RowShr(1))
    };
    assert_eq!(dpp_read(&w, &bc, 1, false, 0).unwrap(), Some(0));
    assert_eq!(dpp_read(&w, &bc, 1, false, 16).unwrap(), Some(0));
    assert_eq!(dpp_read(&w, &bc, 1, false, 1).unwrap(), Some(1000));
    let bc = Dpp {
        bound_ctrl: true,
        ..dpp(DppCtrl::WaveShl1)
    };
    assert_eq!(dpp_read(&w, &bc, 1, false, 63).unwrap(), Some(0));
    assert_eq!(dpp_read(&w, &bc, 1, false, 62).unwrap(), Some(1063));
    // Rotates always have a source.
    let r = dpp(DppCtrl::WaveRor1);
    assert_eq!(dpp_read(&w, &r, 1, false, 0).unwrap(), Some(1063));
    let r = dpp(DppCtrl::WaveRol1);
    assert_eq!(dpp_read(&w, &r, 1, false, 63).unwrap(), Some(1000));
    let r = dpp(DppCtrl::RowRor(3));
    assert_eq!(dpp_read(&w, &r, 1, false, 32).unwrap(), Some(1045));
    // row_bcast lanes the control does not feed stay unwritten even with BOUND_CTRL.
    let b = Dpp {
        bound_ctrl: true,
        ..dpp(DppCtrl::RowBcast15)
    };
    assert_eq!(dpp_read(&w, &b, 1, false, 3).unwrap(), None);

    // Inactive source lane: lane 6 disabled -> lane 5 (quad_perm [1,..]) reads it.
    let mut w = wave();
    w.exec &= !(1 << 6);
    let q = DppCtrl::QuadPerm([0, 1, 2, 3]);
    assert_eq!(dpp_read(&w, &dpp(q), 1, false, 5).unwrap(), Some(1005));
    let shl = dpp(DppCtrl::RowShl(1));
    assert_eq!(
        dpp_read(&w, &shl, 1, false, 5).unwrap(),
        None,
        "source lane 6 disabled, BC=0"
    );
    let shl_bc = Dpp {
        bound_ctrl: true,
        ..dpp(DppCtrl::RowShl(1))
    };
    assert_eq!(
        dpp_read(&w, &shl_bc, 1, false, 5).unwrap(),
        Some(0),
        "source lane 6 disabled, BC=1"
    );
    assert_eq!(dpp_read(&w, &shl_bc, 1, false, 4).unwrap(), Some(1005));
    // A disabled destination lane never writes.
    assert_eq!(dpp_read(&w, &shl_bc, 1, false, 6).unwrap(), None);
}

#[test]
fn dpp_read_masks_gate_destination_not_source() {
    // bank_mask = 0b1101 (bank 1 = lanes 4..7 of every row disabled): the destination write is
    // masked, but a lane in bank 0 may still read a source lane inside the masked bank.
    let w = wave();
    let d = Dpp {
        bank_mask: 0b1101,
        ..dpp(DppCtrl::RowShl(4))
    };
    assert_eq!(
        dpp_read(&w, &d, 1, false, 0).unwrap(),
        Some(1004),
        "source lane 4 is in the masked bank, destination 0 is not"
    );
    assert_eq!(dpp_read(&w, &d, 1, false, 5).unwrap(), None);
    assert_eq!(dpp_read(&w, &d, 1, false, 20).unwrap(), None);
    // row_mask 0b0101: rows 1 and 3 masked.
    let d = Dpp {
        row_mask: 0b0101,
        bound_ctrl: true,
        ..dpp(DppCtrl::RowMirror)
    };
    assert_eq!(dpp_read(&w, &d, 1, false, 2).unwrap(), Some(1013));
    assert_eq!(dpp_read(&w, &d, 1, false, 18).unwrap(), None);
    assert_eq!(dpp_read(&w, &d, 1, false, 34).unwrap(), Some(1045));
    assert_eq!(dpp_read(&w, &d, 1, false, 50).unwrap(), None);
    // Masked-off lanes stay unwritten even with BOUND_CTRL and no source.
    let d = Dpp {
        row_mask: 0b1110,
        bound_ctrl: true,
        ..dpp(DppCtrl::WaveShr1)
    };
    assert_eq!(dpp_read(&w, &d, 1, false, 0).unwrap(), None);
    assert_eq!(dpp_read(&w, &d, 1, false, 16).unwrap(), Some(1015));
}

#[test]
fn sdwa_src_all_selects() {
    // bytes: b0=0x85 b1=0x7F b2=0xFF b3=0x80; word0=0x7F85, word1=0x80FF.
    let raw = 0x80FF_7F85u32;
    let table: [(SdwaSel, u32, u32); 7] = [
        (SdwaSel::Byte0, 0x0000_0085, 0xFFFF_FF85),
        (SdwaSel::Byte1, 0x0000_007F, 0x0000_007F),
        (SdwaSel::Byte2, 0x0000_00FF, 0xFFFF_FFFF),
        (SdwaSel::Byte3, 0x0000_0080, 0xFFFF_FF80),
        (SdwaSel::Word0, 0x0000_7F85, 0x0000_7F85),
        (SdwaSel::Word1, 0x0000_80FF, 0xFFFF_80FF),
        (SdwaSel::Dword, raw, raw),
    ];
    for (sel, zext, sext) in table {
        assert_eq!(sdwa_src(raw, sel, false), zext, "{sel:?} zero-extended");
        assert_eq!(sdwa_src(raw, sel, true), sext, "{sel:?} sign-extended");
    }
    // Word0 with a set sign bit.
    assert_eq!(sdwa_src(0x0000_8001, SdwaSel::Word0, true), 0xFFFF_8001);
    assert_eq!(sdwa_src(0x0000_8001, SdwaSel::Word0, false), 0x0000_8001);
}

#[test]
fn sdwa_dst_all_selects_and_unused() {
    let old = 0xAABB_CCDDu32;
    let res = 0x1234_A685u32; // byte0 = 0x85 (negative), word0 = 0xA685 (negative)
                              // (sel, pad, sext, preserve)
    let table: [(SdwaSel, u32, u32, u32); 6] = [
        (SdwaSel::Byte0, 0x0000_0085, 0xFFFF_FF85, 0xAABB_CC85),
        (SdwaSel::Byte1, 0x0000_8500, 0xFFFF_8500, 0xAABB_85DD),
        (SdwaSel::Byte2, 0x0085_0000, 0xFF85_0000, 0xAA85_CCDD),
        (SdwaSel::Byte3, 0x8500_0000, 0x8500_0000, 0x85BB_CCDD),
        (SdwaSel::Word0, 0x0000_A685, 0xFFFF_A685, 0xAABB_A685),
        (SdwaSel::Word1, 0xA685_0000, 0xA685_0000, 0xA685_CCDD),
    ];
    for (sel, pad, sext, keep) in table {
        assert_eq!(sdwa_dst(old, res, sel, SdwaUnused::Pad), pad, "{sel:?} pad");
        assert_eq!(
            sdwa_dst(old, res, sel, SdwaUnused::Sext),
            sext,
            "{sel:?} sext"
        );
        assert_eq!(
            sdwa_dst(old, res, sel, SdwaUnused::Preserve),
            keep,
            "{sel:?} preserve"
        );
    }
    // Positive field: sext upper bits are zero.
    assert_eq!(
        sdwa_dst(old, 0x7F, SdwaSel::Byte0, SdwaUnused::Sext),
        0x0000_007F
    );
    assert_eq!(
        sdwa_dst(old, 0x7F00, SdwaSel::Byte1, SdwaUnused::Sext),
        0x0000_0000,
        "result byte 0 is 0x00"
    );
    assert_eq!(
        sdwa_dst(old, 0x0000_1234, SdwaSel::Word0, SdwaUnused::Sext),
        0x0000_1234
    );
    // Dword replaces everything regardless of DST_UNUSED.
    for u in [SdwaUnused::Pad, SdwaUnused::Sext, SdwaUnused::Preserve] {
        assert_eq!(sdwa_dst(old, res, SdwaSel::Dword, u), res);
    }
}

fn lanes() -> [u32; WAVE] {
    std::array::from_fn(|i| 100 + i as u32)
}

#[test]
fn swizzle_bitmask_mode() {
    let d = lanes();
    let all = u64::MAX;
    // and=0x1f, or=0, xor=0x1f -> reverse each 32-lane group.
    let out = ds_swizzle(0x7C1F, &d, all).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[(i & 0x20) | (31 - (i & 31))], "reverse lane {i}");
    }
    // xor=1 swaps neighbours.
    let out = ds_swizzle(0x041F, &d, all).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[i ^ 1], "swap lane {i}");
    }
    // and=0, or=5: every lane reads lane 5 of its own 32-lane group.
    let out = ds_swizzle(0x00A0, &d, all).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[(i & 0x20) | 5], "broadcast lane {i}");
    }
    // and=0x1c clears the low two bits, or=2: lane -> (i & 0x1c) | 2.
    let out = ds_swizzle(0x005C, &d, all).unwrap();
    assert_eq!(out[0], d[2]);
    assert_eq!(out[7], d[6]);
    assert_eq!(out[31], d[30]);
    assert_eq!(out[40], d[42], "(8 & 0x1c) | 2 in the second group");
    // Identity pattern (and=0x1f).
    assert_eq!(ds_swizzle(0x001F, &d, all).unwrap(), d);
    // The offset bits above 0x7FFF-1: xor_mask is offset[14:10] (0x7C00 = 0x1f).
    let out = ds_swizzle(0x4000 | 0x1F, &d, all).unwrap();
    assert_eq!(out[0], d[16], "xor=0x10");
    assert_eq!(out[20], d[4]);
    assert_eq!(out[32], d[48], "second group stays in the second group");
    // Reads from inactive lanes give 0: lane 5 disabled, everyone reads lane 5.
    let out = ds_swizzle(0x00A0, &d, all & !(1 << 5)).unwrap();
    assert_eq!(out[0], 0);
    assert_eq!(out[31], 0);
    assert_eq!(
        out[32], d[37],
        "other group's lane 5 is lane 37, still active"
    );
    // Partial EXEC on the source side of a swap.
    let out = ds_swizzle(0x041F, &d, 0b0110).unwrap();
    assert_eq!((out[0], out[1], out[2], out[3]), (d[1], 0, 0, d[2]));
}

#[test]
fn swizzle_quad_perm_mode() {
    let d = lanes();
    // offset[15]=1: out[i+k] = in[i + offset[2k+1:2k]]; [3,2,1,0] = 0b00_01_10_11 = 0x1B.
    let out = ds_swizzle(0x8000 | 0x1B, &d, u64::MAX).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[i / 4 * 4 + (3 - i % 4)], "lane {i}");
    }
    // [1,1,3,3] = 0b11_11_01_01 = 0xF5.
    let out = ds_swizzle(0x8000 | 0xF5, &d, u64::MAX).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[i / 4 * 4 + [1, 1, 3, 3][i % 4]], "lane {i}");
    }
    // 0xBFFF is the top of quad-perm mode (offset < 0xC000).
    let out = ds_swizzle(0xBFFF, &d, u64::MAX).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[i / 4 * 4 + 3], "lane {i}");
    }
    // Disabled source lane returns 0: lane 2 of each quad disabled, [2,2,2,2] = 0xAA.
    let exec = !0x4444_4444_4444_4444u64;
    let out = ds_swizzle(0x8000 | 0xAA, &d, exec).unwrap();
    assert!(out.iter().all(|&v| v == 0));
}

#[test]
fn swizzle_rotate_and_fft_modes() {
    let d = lanes();
    let all = u64::MAX;
    // 0xC000 | rotate<<5 | mask: rotate by +1 inside each 32-lane group, mask=0.
    let out = ds_swizzle(0xC020, &d, all).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[(i & 0x20) | ((i + 1) & 0x1f)], "rot+1 lane {i}");
    }
    // Negative rotate (offset[10]).
    let out = ds_swizzle(0xC420, &d, all).unwrap();
    for i in 0..WAVE {
        assert_eq!(out[i], d[(i & 0x20) | ((i + 31) & 0x1f)], "rot-1 lane {i}");
    }
    // rotate 4, mask 3: j = (i & 3) | (((i + 4) & ~3) & 31) | (i & 32).
    let out = ds_swizzle(0xC083, &d, all).unwrap();
    for (i, j) in [
        (0usize, 4usize),
        (1, 5),
        (3, 7),
        (4, 8),
        (30, 2),
        (31, 3),
        (32, 36),
        (63, 35),
    ] {
        assert_eq!(out[i], d[j], "rot4/mask3 lane {i}");
    }
    // FFT (>= 0xE000), mask=0: j = reverse5(i & 31) | (i & 32).
    let out = ds_swizzle(0xE000, &d, all).unwrap();
    for (i, j) in [
        (0usize, 0usize),
        (1, 16),
        (2, 8),
        (3, 24),
        (4, 4),
        (31, 31),
        (32, 32),
        (33, 48),
    ] {
        assert_eq!(out[i], d[j], "fft mask0 lane {i}");
    }
    // FFT mask=1: j = (reverse5(i) >> 1) | (i & 1) | (i & 32).
    let out = ds_swizzle(0xE001, &d, all).unwrap();
    for (i, j) in [(0usize, 0usize), (1, 9), (2, 4), (3, 13), (4, 2), (32, 32)] {
        assert_eq!(out[i], d[j], "fft mask1 lane {i}");
    }
    // Rotate/FFT boundaries: 0xBFFF is quad-perm, 0xC000 is rotate-by-0 identity, 0xDFFF still rotate.
    assert_eq!(ds_swizzle(0xC000, &d, all).unwrap(), d);
}

/// A, B, C, D in lanes 0..4 (the ISA's simplified 4-thread examples).
fn abcd() -> ([u32; WAVE], [u32; WAVE]) {
    let mut data = [0u32; WAVE];
    let mut addr = [0u32; WAVE];
    data[..4].copy_from_slice(&[0xA, 0xB, 0xC, 0xD]);
    addr[..4].copy_from_slice(&[0, 0, 12, 4]);
    (data, addr)
}

#[test]
fn bpermute_isa_examples() {
    let (data, addr) = abcd();
    // EXEC = 0xF, OFFSET = 0: VDST = { A, A, D, B }.
    let out = ds_bpermute(&addr, 0, &data, 0xF);
    assert_eq!(&out[..4], &[0xA, 0xA, 0xD, 0xB]);
    // EXEC = 0xA: VDST = { -, 0, -, B } (lane 1 reads disabled lane 0 -> 0; disabled lanes not written).
    let out = ds_bpermute(&addr, 0, &data, 0xA);
    assert_eq!((out[1], out[3]), (0, 0xB));
    assert_eq!(
        (out[0], out[2]),
        (0, 0),
        "disabled destination lanes come back as 0 for the caller to ignore"
    );
}

#[test]
fn permute_isa_examples() {
    let (data, addr) = abcd();
    // EXEC = 0xF, OFFSET = 0: VDST = { B, D, 0, C }.
    let out = ds_permute(&addr, 0, &data, 0xF);
    assert_eq!(&out[..4], &[0xB, 0xD, 0, 0xC]);
    // EXEC = 0xA: VDST = { -, D, -, 0 }.
    let out = ds_permute(&addr, 0, &data, 0xA);
    assert_eq!((out[1], out[3]), (0xD, 0));
}

#[test]
fn permute_bpermute_wave64() {
    let d = lanes();
    let rev: [u32; WAVE] = std::array::from_fn(|i| (63 - i as u32) * 4);
    let all = u64::MAX;
    let b = ds_bpermute(&rev, 0, &d, all);
    let f = ds_permute(&rev, 0, &d, all);
    for i in 0..WAVE {
        assert_eq!(b[i], d[63 - i], "bpermute lane {i}");
        assert_eq!(
            f[i],
            d[63 - i],
            "permute lane {i} (reverse is an involution)"
        );
    }
    // OFFSET adds to the byte address: +8 bytes = +2 lanes, wrapping modulo 64 lanes.
    let id: [u32; WAVE] = std::array::from_fn(|i| i as u32 * 4);
    let b = ds_bpermute(&id, 8, &d, all);
    for i in 0..WAVE {
        assert_eq!(b[i], d[(i + 2) % 64], "bpermute offset lane {i}");
    }
    let f = ds_permute(&id, 8, &d, all);
    for i in 0..WAVE {
        assert_eq!(f[(i + 2) % 64], d[i], "permute offset lane {i}");
    }
    // Byte address / 4 % 64: 256 wraps to lane 0, high bits ignored, low two bits ignored.
    let mut addr = [0u32; WAVE];
    addr[0] = 256;
    addr[1] = 0x1008;
    addr[2] = 7;
    addr[3] = 0xFFFF_FFFC;
    let b = ds_bpermute(&addr, 0, &d, all);
    assert_eq!((b[0], b[1], b[2], b[3]), (d[0], d[2], d[1], d[63]));
    // Collision: highest-numbered source wins; unselected lanes get 0.
    let f = ds_permute(&[0; WAVE], 0, &d, all);
    assert_eq!(f[0], d[63]);
    assert!(f[1..].iter().all(|&v| v == 0));
    // Same collision with partial EXEC: only lanes 0..=9 send, lane 9 wins; lanes outside EXEC are not written.
    let f = ds_permute(&[0; WAVE], 0, &d, (1 << 10) - 1);
    assert_eq!(f[0], d[9]);
    assert!(f[1..10].iter().all(|&v| v == 0));
    assert!(
        f[10..].iter().all(|&v| v == 0),
        "not written; returned as 0"
    );
}

#[test]
fn permute_bpermute_partial_exec() {
    let d = lanes();
    let id: [u32; WAVE] = std::array::from_fn(|i| i as u32 * 4);
    // Even lanes active: bpermute of the identity returns the data for even lanes; odd destinations are not written.
    let even = 0x5555_5555_5555_5555u64;
    let b = ds_bpermute(&id, 0, &d, even);
    for i in 0..WAVE {
        assert_eq!(b[i], if i % 2 == 0 { d[i] } else { 0 }, "lane {i}");
    }
    // Each active lane reads its odd neighbour (disabled) -> 0.
    let b = ds_bpermute(&id, 4, &d, even);
    assert!(b.iter().all(|&v| v == 0));
    // Each active lane reads the next even lane (+8 bytes) -> data from an active lane, except lane 62 -> lane 0 (active).
    let b = ds_bpermute(&id, 8, &d, even);
    for i in (0..WAVE).step_by(2) {
        assert_eq!(b[i], d[(i + 2) % 64], "lane {i}");
    }
    // permute: active lanes send to lane+1 (odd, disabled): nothing is written to active lanes -> all 0.
    let f = ds_permute(&id, 4, &d, even);
    assert!(f.iter().all(|&v| v == 0));
    // permute: a disabled lane's data does not propagate: lane 1 (disabled) targets lane 0, active lane 2 targets lane 0 too.
    let mut addr = [0u32; WAVE];
    addr[1] = 0;
    addr[2] = 0;
    let f = ds_permute(&addr, 0, &d, 0b101);
    assert_eq!(f[0], d[2]);
    let f = ds_permute(&addr, 0, &d, 0b011);
    assert_eq!(
        f[0], d[1],
        "lane 2 disabled: lane 1 is the highest enabled source"
    );
}
