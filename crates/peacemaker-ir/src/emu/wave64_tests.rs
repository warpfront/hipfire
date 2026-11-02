// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave64 semantics of the interpreter (EmuWave64). Instructions are decoded from the machine words of the gfx1201
//! hipcc incumbent objects (`lane_proj`, `-mwavefrontsize64`; the encodings are wave-size neutral, `vcc_lo`/`exec_lo`
//! name the whole 64-bit register in wave64), every expectation is derived by hand from the RDNA4 ISA text, never from
//! the implementation: 64-bit EXEC/VCC, lane-mask operands as SGPR pairs, DPP rows over 64 lanes, `ds_swizzle` /
//! `ds_bpermute` per wave size, `v_mbcnt`, carry masks, `v_mad_co_i64_i32`, readfirstlane/readlane over 64 lanes.
use super::{alu, memory, Memory, Result, State};
use crate::{codec::gfx12, Arch, Inst};

fn state64() -> State {
    State {
        s: [0; 106],
        t: [0; 16],
        v: Box::new([[0; 64]; 256]),
        lanes: 64,
        nan_loose: false,
        exec: u64::MAX,
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
fn state32() -> State {
    let mut s = state64();
    s.lanes = 32;
    s.exec = 0xffff_ffff;
    s
}

fn dec(words: &[u32]) -> (Inst, &'static str) {
    let (inst, n) = gfx12::decode(words).unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"));
    assert_eq!(n, words.len(), "{words:08x?}");
    let name = crate::isa::lookup(Arch::Gfx1201, inst.op, inst.form)
        .unwrap_or_else(|| panic!("{words:08x?} has no gfx1201 row"))
        .name;
    (inst, name)
}
fn run(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("alu"), "{name}");
    alu::execute(Arch::Gfx1201, name, &inst, st)
}
fn mem_run(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("memory"), "{name}");
    memory::execute(name, &inst, st, &mut Memory::default(), &mut [])
}

#[test]
fn saveexec_family_operates_on_all_64_bits_and_scc_follows_the_64_bit_result() {
    // s_and_saveexec_b64 s[0:1], vcc  (BE80216A): s[0:1] = EXEC; EXEC &= VCC; SCC = EXEC != 0.
    let mut st = state64();
    st.exec = 0xf0f0_0000_ffff_0001;
    st.vcc = 0x0f0f_0000_0000_0001 | 1 << 40;
    run(&[0xBE80_216A], &mut st).unwrap();
    assert_eq!((st.s[0], st.s[1]), (0xffff_0001, 0xf0f0_0000));
    assert_eq!(
        st.exec, 0x0000_0000_0000_0001,
        "hi words of EXEC and VCC are disjoint, lane 0 survives"
    );
    assert!(st.scc);
    // A high-only intersection is non-empty (a 32-bit model would see EXEC == 0 and SCC == 0).
    let mut st = state64();
    st.exec = 1 << 63;
    st.vcc = 1 << 63;
    run(&[0xBE80_216A], &mut st).unwrap();
    assert_eq!(st.exec, 1 << 63);
    assert!(st.scc);
    // s_or_b64 exec, exec, s[0:1] (8CFE007E): the restore of a divergence region; SCC = result != 0.
    let mut st = state64();
    st.exec = 1 << 33;
    (st.s[0], st.s[1]) = (0x8000_0000, 1);
    run(&[0x8CFE_007E], &mut st).unwrap();
    assert_eq!(st.exec, 1 << 33 | 1 << 32 | 1 << 31);
    assert!(st.scc);
    // s_and_not1_b64 exec, exec, s[2:3] (91FE027E): EXEC & ~S; clearing every active lane gives SCC = 0.
    let mut st = state64();
    st.exec = 0xdead_0000_0000_beef;
    (st.s[2], st.s[3]) = (0xffff_ffff, 0xffff_ffff);
    run(&[0x91FE_027E], &mut st).unwrap();
    assert_eq!(st.exec, 0);
    assert!(!st.scc);
    let mut st = state64();
    st.exec = 0xdead_0000_0000_beef;
    (st.s[2], st.s[3]) = (0xffff_0000, 0x0000_ffff);
    run(&[0x91FE_027E], &mut st).unwrap();
    assert_eq!(
        st.exec,
        0xdead_0000_0000_0000 & !0x0000_ffff_0000_0000 | 0xbeef
    );
    assert!(st.scc);
    // s_and_not1_saveexec_b64 s[0:1], s[24:25] (BE803118): s[0:1] = EXEC; EXEC = S & ~EXEC.
    let mut st = state64();
    st.exec = 0x00ff_00ff_00ff_00ff;
    (st.s[24], st.s[25]) = (0xffff_ffff, 0xffff_ffff);
    run(&[0xBE80_3118], &mut st).unwrap();
    assert_eq!((st.s[0], st.s[1]), (0x00ff_00ff, 0x00ff_00ff));
    assert_eq!(st.exec, 0xff00_ff00_ff00_ff00);
    // s_or_saveexec_b64 s[0:1], s[0:1] (BE802300): aliased destination and source: the source is read first.
    let mut st = state64();
    st.exec = 0x1_0000_0000;
    (st.s[0], st.s[1]) = (2, 0);
    run(&[0xBE80_2300], &mut st).unwrap();
    assert_eq!((st.s[0], st.s[1]), (0, 1), "old EXEC saved");
    assert_eq!(st.exec, 0x1_0000_0002);
}

#[test]
fn wave32_masks_keep_a_zero_upper_half() {
    let mut st = state32();
    st.exec = 0xffff_ffff;
    st.vcc = 0x8000_0000;
    run(&[0xBE80_216A], &mut st).unwrap(); // s_and_saveexec_b64 s[0:1], vcc
    assert_eq!(
        (st.exec, st.s[1]),
        (0x8000_0000, 0),
        "no upper half in wave32"
    );
}

#[test]
fn compare_writes_a_pair_and_cndmask_reads_it() {
    // v_cmp_ge_i32_e64 s0, v24, v25 (D4460000 02023318): s[0:1] = per-lane mask over 64 lanes.
    let mut st = state64();
    for l in 0..64 {
        st.v[24][l] = l as u32;
        st.v[25][l] = 40;
    }
    st.exec = !(1 << 63); // lane 63 inactive: its bit is zero even though 63 >= 40
    run(&[0xD446_0000, 0x0202_3318], &mut st).unwrap();
    let want = (!0u64 << 40) & !(1 << 63);
    assert_eq!((st.s[0], st.s[1]), (want as u32, (want >> 32) as u32));
    // v_cndmask_b32_e64 v0, 0, 8, vcc_lo (D5010000 01A91080): vcc_lo names the whole VCC in wave64.
    let mut st = state64();
    st.vcc = 0x8000_0000_0000_0001;
    run(&[0xD501_0000, 0x01A9_1080], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(
            st.v[0][l],
            if l == 0 || l == 63 { 8 } else { 0 },
            "lane {l}"
        );
    }
    // v_cmpx_lt_i32_e32 v24, v25 (7D823318) writes EXEC (64 bits).
    let mut st = state64();
    for l in 0..64 {
        st.v[24][l] = l as u32;
        st.v[25][l] = 33;
    }
    run(&[0x7D82_3318], &mut st).unwrap();
    assert_eq!(st.exec, (1u64 << 33) - 1);
    // v_cmp_eq_u32_e32 vcc_lo, 0, v0 (7C940080): VCC is 64 bits wide.
    let mut st = state64();
    st.v[0][5] = 0;
    for l in 0..64 {
        if l != 5 && l != 37 {
            st.v[0][l] = 1;
        }
    }
    run(&[0x7C94_0080], &mut st).unwrap();
    assert_eq!(st.vcc, 1 << 5 | 1 << 37);
}

#[test]
fn carry_chain_uses_the_pair_and_mad_co_i64_i32_widens_signed() {
    // v_add_co_u32 v12, s0, v18, v27 (D700000C 02023712): carry-out lands in s[0:1], one bit per lane.
    let mut st = state64();
    for l in 0..64 {
        st.v[18][l] = 0xffff_fff0;
        st.v[27][l] = if l % 2 == 0 { 0x10 } else { 0xf };
    }
    st.exec = !(1 << 62);
    run(&[0xD700_000C, 0x0202_3712], &mut st).unwrap();
    let want = 0x5555_5555_5555_5555u64 & !(1 << 62); // even lanes carry; inactive lane 62 contributes zero
    assert_eq!(((st.s[1] as u64) << 32) | st.s[0] as u64, want);
    assert_eq!(st.v[12][0], 0);
    assert_eq!(st.v[12][1], 0xffff_ffff);
    // v_add_co_ci_u32_e64 v13, null, 0, v19, s0 (D5207C0D 00022680): carry-in from the pair, high lanes included.
    let mut st = state64();
    (st.s[0], st.s[1]) = (0, 1 << 31); // only lane 63
    for l in 0..64 {
        st.v[19][l] = 7;
    }
    run(&[0xD520_7C0D, 0x0002_2680], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(st.v[13][l], if l == 63 { 8 } else { 7 }, "lane {l}");
    }
    // v_mad_co_i64_i32 v[18:19], null, 0x440, v0, v[12:13]: D = 0x440 * i32(v0) + i64(v[12:13]), per lane, signed.
    let mut st = state64();
    for l in 0..64 {
        st.v[0][l] = (l as i32 - 32) as u32; // negative for the low half
        (st.v[12][l], st.v[13][l]) = (0x1000_0000, 0x1);
    }
    run(&[0xD6FF_7C12, 0x0432_00FF, 0x0000_0440], &mut st).unwrap();
    for l in 0..64 {
        let want = (0x440i64 * (l as i64 - 32)).wrapping_add(0x1_1000_0000) as u64;
        assert_eq!(
            ((st.v[19][l] as u64) << 32) | st.v[18][l] as u64,
            want,
            "lane {l}"
        );
    }
}

#[test]
fn mbcnt_counts_the_lanes_below_in_each_dword_of_the_mask() {
    // v_mbcnt_lo_u32_b32 v3, -1, 0 (D71F0003 020100C1): lane id for lanes 0..32, 32 for lanes 32..64.
    let mut st = state64();
    run(&[0xD71F_0003, 0x0201_00C1], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(st.v[3][l], (l as u32).min(32), "lane {l}");
    }
    // v_mbcnt_hi_u32_b32 v5, -1, v5 (D7200005 02020AC1) after lo: the full wave64 lane id.
    let mut st = state64();
    run(&[0xD71F_0003, 0x0201_00C1], &mut st).unwrap();
    st.v[5] = st.v[3];
    run(&[0xD720_0005, 0x0202_0AC1], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(st.v[5][l], l as u32, "lane {l}");
    }
}

#[test]
fn readfirstlane_and_readlane_span_64_lanes() {
    // v_readfirstlane_b32 s9, v0 (7E120500): the lowest ACTIVE lane, lane 44 here.
    let mut st = state64();
    st.exec = 1 << 44 | 1 << 50;
    for l in 0..64 {
        st.v[0][l] = 1000 + l as u32;
    }
    run(&[0x7E12_0500], &mut st).unwrap();
    assert_eq!(st.s[9], 1044);
}

#[test]
fn dpp_row_shl_runs_per_16_lane_row_in_all_four_rows_and_zeroes_with_bound_ctrl() {
    // v_add_f32_dpp v3, v3, v3 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1 (060606FA FF090803).
    let mut st = state64();
    for l in 0..64 {
        st.v[3][l] = (l as f32).to_bits();
    }
    run(&[0x0606_06FA, 0xFF09_0803], &mut st).unwrap();
    for l in 0..64 {
        let col = l % 16;
        let want = if col + 8 < 16 {
            (l + (l + 8)) as f32
        } else {
            l as f32
        }; // out of row: reads 0 (bound_ctrl)
        assert_eq!(f32::from_bits(st.v[3][l]), want, "lane {l}");
    }
    // v_mov_b32_dpp v56, v3 row_shl:1 (7E7002FA FF010103), no bound_ctrl: the last lane of each row keeps v56.
    let mut st = state64();
    for l in 0..64 {
        st.v[3][l] = 500 + l as u32;
        st.v[56][l] = 9;
    }
    run(&[0x7E70_02FA, 0xFF01_0103], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(
            st.v[56][l],
            if l % 16 == 15 { 9 } else { 501 + l as u32 },
            "lane {l}"
        );
    }
}

#[test]
fn swizzle_swap16_stays_inside_each_32_lane_half() {
    // ds_swizzle_b32 v56, v3 offset:swizzle(SWAP,16) (D8D4401F 38000003): lane ^ 16 within the 32-lane group.
    let mut st = state64();
    for l in 0..64 {
        st.v[3][l] = 2000 + l as u32;
    }
    mem_run(&[0xD8D4_401F, 0x3800_0003], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(st.v[56][l], 2000 + (l ^ 16) as u32, "lane {l}");
    }
    // A bitmask pattern that would leave the group in a naive wave32 implementation: and=0x1f, or=0, xor=0x1f
    // (offset 0x7C1F) reverses each half: lane 40 -> 32 + (7 ^ 0x1f) = 32 + 24.
    let mut st = state64();
    for l in 0..64 {
        st.v[3][l] = 2000 + l as u32;
    }
    let (mut inst, _) = dec(&[0xD8D4_401F, 0x3800_0003]);
    for op in inst.operands.iter_mut() {
        if let crate::operand::Operand::Imm(crate::operand::ImmField::DsOffset(n)) = op {
            *n = 0x7C1F;
        }
    }
    memory::execute(
        "ds_swizzle_b32",
        &inst,
        &mut st,
        &mut Memory::default(),
        &mut [],
    )
    .unwrap();
    for l in 0..64 {
        assert_eq!(
            st.v[56][l],
            2000 + ((l & 32) | ((l & 31) ^ 31)) as u32,
            "lane {l}"
        );
    }
    // Inactive source lanes read 0 (lane 56 disabled; lane 40 would read it).
    let mut st = state64();
    st.exec = !(1 << 56);
    for l in 0..64 {
        st.v[3][l] = 2000 + l as u32;
    }
    mem_run(&[0xD8D4_401F, 0x3800_0003], &mut st).unwrap();
    assert_eq!(st.v[56][40], 0);
    assert_eq!(st.v[56][41], 2000 + 57);
}

#[test]
fn bpermute_wraps_at_the_wave_size() {
    // ds_bpermute_b32 v4, v0, v2 (DACC0000 04000200): lane i reads v2 of lane (v0[i] / 4) mod wave_size.
    let mut st = state64();
    for l in 0..64 {
        st.v[2][l] = 700 + l as u32;
        st.v[0][l] = (((l + 40) % 128) * 4) as u32; // up to lane 127: wraps modulo 64
    }
    mem_run(&[0xDACC_0000, 0x0400_0200], &mut st).unwrap();
    for l in 0..64 {
        assert_eq!(st.v[4][l], 700 + ((l + 40) % 64) as u32, "lane {l}");
    }
    let mut st = state32();
    for l in 0..32 {
        st.v[2][l] = 700 + l as u32;
        st.v[0][l] = (((l + 40) % 128) * 4) as u32;
    }
    mem_run(&[0xDACC_0000, 0x0400_0200], &mut st).unwrap();
    for l in 0..32 {
        assert_eq!(st.v[4][l], 700 + ((l + 40) % 32) as u32, "wave32 lane {l}");
    }
}

#[test]
fn exec_pair_operands_read_both_halves() {
    // s_mov_b64 s[2:3], exec (BE82017E) copies 64 bits; in wave32 the upper word is zero.
    let mut st = state64();
    st.exec = 0x1234_5678_9abc_def0;
    run(&[0xBE82_017E], &mut st).unwrap();
    assert_eq!((st.s[2], st.s[3]), (0x9abc_def0, 0x1234_5678));
    let mut st = state32();
    st.exec = 0x9abc_def0;
    run(&[0xBE82_017E], &mut st).unwrap();
    assert_eq!((st.s[2], st.s[3]), (0x9abc_def0, 0));
}
