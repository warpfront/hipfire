// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 decode-table gaps found by lifting the exact gfx1100 Raon decode objects (the 7900 XTX run of
//! `raon_oracle railgun-probe --phase inventory`) with `railgun-cert coverage`: four objects stopped the lifter with
//! `undefined Gfx1100 opcode/form`:
//!
//! * `v_cmp_ge_i64_e64` (VOP3 signed 64-bit compare into a lane mask): `ffn_moe_combine_f32` / `ffn_round_bf16_f32`
//!   (`0xd456001c`) and `latent_attention_decode_partial_q8` / `_merge` (`0xd4560002`);
//! * `v_ashrrev_i64` (VOP3 64-bit arithmetic shift): `latent_attention_gather_latent_f32` /
//!   `latent_attention_scatter_expanded_q8` (`0xd73e0000`);
//! * `v_cmpx_gt_u64_e32` (VOPC unsigned 64-bit compare that writes EXEC): the same gather/scatter object (`0x7db80406`).
//!
//! Every word is what `llvm-mc --arch=amdgcn --mcpu=gfx1100 --show-encoding` assembles from the text in its comment
//! (and what `llvm-objdump --mcpu=gfx1100` prints for the loaded object). Each is decoded with the production gfx11
//! decoder, re-encoded byte-identically (also rebuilt from typed fields alone) and executed through the emulator's
//! dispatch entry. Every expected value is derived by hand from the RDNA3 ISA pseudocode rules (never from the
//! implementation under test), with the edge lanes a wrong semantic fails: INT64_MIN/MAX, -1 vs 0, differences
//! confined to the high dword, signed vs unsigned order, shift counts 0/31/63/64+ and execution-mask holes.
use super::{alu, Result, State};
use crate::{
    codec::{gfx11, gfx12},
    effects::ImplicitSet,
    inst::Wave,
    provenance::Provenance,
    reg::{Kind, RegRef},
    Arch, Inst,
};

fn state() -> State {
    State {
        s: [0; 106],
        t: [0; 16],
        v: Box::new([[0; 64]; 256]),
        lanes: 32,
        nan_loose: false,
        exec: 1,
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

/// The instruction and its table name, decoded from machine words by the production gfx1100 decoder.
fn dec(words: &[u32]) -> (Inst, &'static str) {
    let (inst, n) =
        gfx11::decode(Arch::Gfx1100, words).unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"));
    assert_eq!(n, words.len(), "{words:08x?} decoded to {n} words");
    let name = crate::isa::lookup_inst(Arch::Gfx1100, &inst)
        .ok()
        .unwrap_or_else(|| panic!("{words:08x?} has no gfx1100 row"))
        .name;
    (inst, name)
}

/// Decode, check the table name, and prove both byte-identical re-encodings (decoded provenance and typed fields).
fn roundtrip(words: &[u32], want_name: &str) -> Inst {
    let (inst, name) = dec(words);
    assert_eq!(name, want_name, "{words:08x?}");
    assert_eq!(
        gfx11::encode(Arch::Gfx1100, &inst).unwrap().as_slice(),
        words,
        "{want_name}: re-encode differs"
    );
    let rebuilt = Inst::from_encoding(
        Arch::Gfx1100,
        inst.op,
        inst.form,
        inst.variant,
        inst.fields.clone(),
        inst.operands.clone(),
        inst.mods.clone(),
        inst.literal,
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(
        gfx11::encode(Arch::Gfx1100, &rebuilt).unwrap().as_slice(),
        words,
        "{want_name}: typed rebuild differs"
    );
    assert_eq!(gfx11::decode(Arch::Gfx1100, words).unwrap().0, inst);
    inst
}

/// Run an ALU instruction through the emulator's dispatch; its name must have a semantics row of class `alu`.
fn run(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("alu"), "{name}");
    alu::execute(Arch::Gfx1100, name, &inst, st)
}

fn pair64(st: &mut State, base: usize, lane: usize, x: u64) {
    st.v[base][lane] = x as u32;
    st.v[base + 1][lane] = (x >> 32) as u32;
}

fn get64(st: &State, base: usize, lane: usize) -> u64 {
    u64::from(st.v[base][lane]) | u64::from(st.v[base + 1][lane]) << 32
}

fn spair(st: &mut State, base: usize, x: u64) {
    st.s[base] = x as u32;
    st.s[base + 1] = (x >> 32) as u32;
}

fn reg(kind: Kind, base: u16, len: u8) -> RegRef {
    RegRef { kind, base, len }
}

// ---- v_ashrrev_i64 -----------------------------------------------------------------------------------------------

/// `v_ashrrev_i64 v[0:1], 31, v[0:1]` (the Raon gather word)
const ASHR_V01_31: &[u32] = &[0xd73e_0000, 0x0202_009f];
/// `v_ashrrev_i64 v[2:3], 31, v[2:3]`
const ASHR_V23_31: &[u32] = &[0xd73e_0002, 0x0202_049f];
/// `v_ashrrev_i64 v[0:1], v4, v[2:3]`
const ASHR_V01_V4: &[u32] = &[0xd73e_0000, 0x0202_0504];
/// `v_ashrrev_i64 v[0:1], 63, -1`
const ASHR_V01_63_M1: &[u32] = &[0xd73e_0000, 0x0201_82bf];

#[test]
fn v_ashrrev_i64_decodes_with_pair_operands_and_roundtrips() {
    for words in [ASHR_V01_31, ASHR_V23_31, ASHR_V01_V4, ASHR_V01_63_M1] {
        roundtrip(words, "v_ashrrev_i64");
    }
    let inst = roundtrip(ASHR_V01_31, "v_ashrrev_i64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::V, 0, 2)]);
    assert_eq!(inst.effects.uses.as_slice(), &[reg(Kind::V, 0, 2)]);
    let inst = roundtrip(ASHR_V01_V4, "v_ashrrev_i64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::V, 0, 2)]);
    assert_eq!(
        inst.effects.uses.as_slice(),
        &[reg(Kind::V, 4, 1), reg(Kind::V, 2, 2)]
    );
}

/// D.i64 = S1.i64 >> S0[5:0] (arithmetic): count 0, 31, 63, and 64+ (bits above 5:0 ignored).
#[test]
fn v_ashrrev_i64_masks_the_count_and_fills_with_the_sign() {
    const MIN: u64 = 0x8000_0000_0000_0000;
    const MAX: u64 = 0x7fff_ffff_ffff_ffff;
    // (count in v4, data, hand-computed result)
    let cases: [(u32, u64, u64); 12] = [
        (0, MIN | 1, MIN | 1),            // count 0: unchanged
        (31, MIN, 0xffff_ffff_0000_0000), // INT64_MIN >> 31 = -2^32
        (63, MIN, u64::MAX),              // sign fill: -1
        (63, MAX, 0),                     // positive: 0
        (64, MIN | 1, MIN | 1),           // 64 & 63 = 0
        (65, MIN, 0xc000_0000_0000_0000), // 65 & 63 = 1
        (127, MIN, u64::MAX),             // 127 & 63 = 63
        (0xffff_ffc0, 0x8000_0000_dead_beef, 0x8000_0000_dead_beef), // bits above 5:0 ignored
        (4, 0x0123_4567_89ab_cdef, 0x0012_3456_789a_bcde), // positive, plain shift
        (4, 0xf123_4567_89ab_cdef, 0xff12_3456_789a_bcde), // negative: top nibble filled with ones
        (32, 0x0000_0001_0000_0000, 1),   // crosses the dword boundary exactly
        (1, u64::MAX, u64::MAX),          // -1 stays -1
    ];
    let mut st = state();
    st.exec = 0xfff;
    for (lane, (count, data, _)) in cases.into_iter().enumerate() {
        st.v[4][lane] = count;
        pair64(&mut st, 2, lane, data);
    }
    pair64(&mut st, 0, 12, 0x1111_2222_3333_4444); // inactive lane
    run(ASHR_V01_V4, &mut st).unwrap();
    for (lane, (count, data, want)) in cases.into_iter().enumerate() {
        assert_eq!(
            get64(&st, 0, lane),
            want,
            "lane {lane}: ashr({data:#x}, {count})"
        );
    }
    assert_eq!(get64(&st, 0, 12), 0x1111_2222_3333_4444);
}

/// The in-place form the Raon gather kernel uses (destination aliases the source pair, inline count 31), plus the
/// inline-constant source: the inline integer -1 is sign-extended to 64 bits before the shift.
#[test]
fn v_ashrrev_i64_in_place_and_inline_constant_source() {
    let cases: [(u64, u64); 5] = [
        (0x8000_0000_0000_0000, 0xffff_ffff_0000_0000), // MIN >> 31
        (0x0000_0001_8000_0000, 3),                     // (2^32 + 2^31) >> 31 = 3
        (u64::MAX, u64::MAX),                           // -1
        (0x7fff_ffff_ffff_ffff, 0x0000_0000_ffff_ffff), // MAX >> 31 = 2^32 - 1
        (0xffff_ffff_8000_0000, 0xffff_ffff_ffff_ffff), // -2^31 >> 31 = -1
    ];
    let mut st = state();
    st.exec = 0x1f;
    for (lane, (data, _)) in cases.into_iter().enumerate() {
        pair64(&mut st, 0, lane, data);
    }
    run(ASHR_V01_31, &mut st).unwrap();
    for (lane, (data, want)) in cases.into_iter().enumerate() {
        assert_eq!(get64(&st, 0, lane), want, "lane {lane}: {data:#x} >> 31");
    }
    let mut st = state();
    st.exec = 1;
    run(ASHR_V01_63_M1, &mut st).unwrap();
    assert_eq!(get64(&st, 0, 0), u64::MAX);
}

#[test]
fn v_ashrrev_i64_refuses_float_modifiers() {
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.abs = 1;
    let mut st = state();
    assert!(alu::execute(Arch::Gfx1100, name, &inst, &mut st).is_err());
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.neg = 2;
    assert!(alu::execute(Arch::Gfx1100, name, &inst, &mut st).is_err());
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.clamp = true;
    assert!(alu::execute(Arch::Gfx1100, name, &inst, &mut st).is_err());
}

// ---- v_cmp_ge_i64_e64 --------------------------------------------------------------------------------------------

/// `v_cmp_ge_i64_e64 s28, s[12:13], s[2:3]` (the Raon `ffn_moe_combine`/`ffn_round` word)
const CMP_GE_I64_S28: &[u32] = &[0xd456_001c, 0x0200_040c];
/// `v_cmp_ge_i64_e64 s2, s[10:11], s[6:7]` (the Raon latent-attention decode word)
const CMP_GE_I64_S2: &[u32] = &[0xd456_0002, 0x0200_0c0a];
/// `v_cmp_ge_i64_e64 s0, v[0:1], v[2:3]`
const CMP_GE_I64_V: &[u32] = &[0xd456_0000, 0x0202_0500];

#[test]
fn v_cmp_ge_i64_e64_decodes_with_mask_dest_and_pair_sources() {
    for (words, dst, a, b) in [
        (CMP_GE_I64_S28, 28u16, 12u16, 2u16),
        (CMP_GE_I64_S2, 2, 10, 6),
    ] {
        let inst = roundtrip(words, "v_cmp_ge_i64_e64");
        // The table keeps the 64-bit mask slot; a wave32 consumer narrows it to one SGPR, wave64 keeps the pair.
        assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::S, dst, 2)]);
        assert_eq!(
            inst.effects.uses.as_slice(),
            &[reg(Kind::S, a, 2), reg(Kind::S, b, 2)]
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1100, Wave::Wave32, &inst, 0),
            Some(reg(Kind::S, dst, 1))
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1100, Wave::Wave64, &inst, 0),
            Some(reg(Kind::S, dst, 2))
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1100, Wave::Wave32, &inst, 1),
            Some(reg(Kind::S, a, 2))
        );
    }
    let inst = roundtrip(CMP_GE_I64_V, "v_cmp_ge_i64_e64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::S, 0, 2)]);
    assert_eq!(
        inst.effects.uses.as_slice(),
        &[reg(Kind::V, 0, 2), reg(Kind::V, 2, 2)]
    );
}

/// Signed 64-bit `>=` around INT64_MIN/MAX and for differences confined to the high dword. The scalar-pair sources
/// are uniform, so every active lane gets the same bit; inactive lanes write zero.
#[test]
fn v_cmp_ge_i64_e64_signed_edges_on_scalar_pairs() {
    const MIN: u64 = 0x8000_0000_0000_0000;
    const MAX: u64 = 0x7fff_ffff_ffff_ffff;
    // (S0 = s[12:13], S1 = s[2:3], S0 >= S1 as signed)
    let cases: [(u64, u64, bool); 10] = [
        (MIN, MIN, true),
        (MIN, MAX, false),
        (MAX, MIN, true),
        (MAX, MAX, true),
        (u64::MAX, 0, false), // -1 >= 0: false (unsigned would say true)
        (0, u64::MAX, true),  // 0 >= -1
        (0x0000_0001_0000_0000, 0x0000_0000_ffff_ffff, true), // high dword decides
        (0x0000_0000_ffff_ffff, 0x0000_0001_0000_0000, false),
        (MIN | 1, MIN, true),
        (MIN, MIN | 1, false),
    ];
    for (a, b, want) in cases {
        let mut st = state();
        st.exec = 0b1011; // lane 2 inactive
        st.s[28] = 0xffff_ffff; // stale bits must be replaced
        spair(&mut st, 12, a);
        spair(&mut st, 2, b);
        run(CMP_GE_I64_S28, &mut st).unwrap();
        assert_eq!(
            st.s[28],
            if want { 0b1011 } else { 0 },
            "{a:#x} >= {b:#x} (signed)"
        );
    }
    // The sibling word with other registers: s2 = (s[10:11] >= s[6:7]).
    let mut st = state();
    st.exec = 0xffff_ffff;
    spair(&mut st, 10, MIN);
    spair(&mut st, 6, MAX);
    run(CMP_GE_I64_S2, &mut st).unwrap();
    assert_eq!(st.s[2], 0);
    spair(&mut st, 10, MAX);
    spair(&mut st, 6, MIN);
    run(CMP_GE_I64_S2, &mut st).unwrap();
    assert_eq!(st.s[2], 0xffff_ffff);
    // Nothing else is written.
    assert_eq!((st.s[3], st.s[28]), (0, 0));
}

/// Per-lane vector operands: lanes 0..7 take the pairs below, lanes 8.. are inactive although `a >= b` would hold.
/// Signed `>=` over `(a, b)`: equal t; 1 vs 2 f; INT64_MIN vs MAX f; -1 vs 0 f; high-dword-only win t; high-dword
/// loss against a larger low dword f; equal high / smaller low f; equal INT64_MIN t  =>  0b1001_0001.
#[test]
fn v_cmp_ge_i64_e64_per_lane_vector_pairs() {
    let pairs: [(u64, u64); 8] = [
        (0, 0),
        (1, 2),
        (0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff),
        (0xffff_ffff_ffff_ffff, 0),
        (0x0000_0001_0000_0000, 0x0000_0000_ffff_ffff),
        (0x0000_0001_0000_0005, 0x0000_0002_0000_0001),
        (0x7fff_ffff_0000_0000, 0x7fff_ffff_0000_0001),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
    ];
    let mut st = state();
    st.exec = 0xff;
    st.s[0] = 0xffff_ffff;
    for (lane, (a, b)) in pairs.into_iter().enumerate() {
        pair64(&mut st, 0, lane, a);
        pair64(&mut st, 2, lane, b);
    }
    for lane in 8..32 {
        pair64(&mut st, 0, lane, 5);
        pair64(&mut st, 2, lane, 1);
    }
    run(CMP_GE_I64_V, &mut st).unwrap();
    assert_eq!(st.s[0], 0b1001_0001, "got {:#010b}", st.s[0]);
}

#[test]
fn v_cmp_ge_i64_e64_wave64_writes_the_sgpr_pair() {
    let mut st = state();
    st.lanes = 64;
    st.exec = u64::MAX;
    spair(&mut st, 12, 5);
    spair(&mut st, 2, 5);
    run(CMP_GE_I64_S28, &mut st).unwrap();
    assert_eq!((st.s[28], st.s[29]), (0xffff_ffff, 0xffff_ffff));
    st.exec = 0x8000_0000_0000_0001;
    run(CMP_GE_I64_S28, &mut st).unwrap();
    assert_eq!((st.s[28], st.s[29]), (1, 0x8000_0000));
    spair(&mut st, 2, 6);
    run(CMP_GE_I64_S28, &mut st).unwrap();
    assert_eq!((st.s[28], st.s[29]), (0, 0));
}

// ---- v_cmpx_gt_u64_e32 -------------------------------------------------------------------------------------------

/// `v_cmpx_gt_u64_e32 s[6:7], v[2:3]` (the Raon scatter word): EXEC = S0.u64 > S1.u64 per lane (S0 = s[6:7]).
const CMPX_GT_U64: &[u32] = &[0x7db8_0406];

#[test]
fn v_cmpx_gt_u64_e32_decodes_and_writes_exec() {
    let inst = roundtrip(CMPX_GT_U64, "v_cmpx_gt_u64_e32");
    assert_eq!(
        inst.effects.uses.as_slice(),
        &[reg(Kind::S, 6, 2), reg(Kind::V, 2, 2)]
    );
    assert_ne!(
        inst.effects.implicit.writes & ImplicitSet::EXEC,
        0,
        "the row must declare its EXEC write: {:?}",
        inst.effects.implicit
    );
}

/// Unsigned order with S0 uniform: the high-dword-only difference, equality, S0 = 2^63 against 2^63-1 (true
/// unsigned, false signed) and against u64::MAX (false unsigned, true signed), and the S0 = u64::MAX / 0 extremes.
#[test]
fn v_cmpx_gt_u64_e32_unsigned_edges_and_exec_holes() {
    // (S0, [S1 per lane], hand-computed EXEC bits for lanes 0..)
    let cases: [(u64, [u64; 4], u32); 4] = [
        // 2^32 > {0xffff_ffff t, 2^32 f (equal), 2^32+1 f, 0 t}
        (
            0x1_0000_0000,
            [0xffff_ffff, 0x1_0000_0000, 0x1_0000_0001, 0],
            0b1001,
        ),
        // 2^63 > {2^63-1 t (signed: f), u64::MAX f (signed: t), 2^63 f, 1 t}
        (
            0x8000_0000_0000_0000,
            [0x7fff_ffff_ffff_ffff, u64::MAX, 0x8000_0000_0000_0000, 1],
            0b1001,
        ),
        // u64::MAX > {u64::MAX-1 t, u64::MAX f, 0 t, 2^63 t}
        (
            u64::MAX,
            [u64::MAX - 1, u64::MAX, 0, 0x8000_0000_0000_0000],
            0b1101,
        ),
        // 0 > anything unsigned: always false
        (0, [0, 1, u64::MAX, 0x8000_0000_0000_0000], 0b0000),
    ];
    for (s0, s1, want) in cases {
        let mut st = state();
        st.exec = 0b1111;
        spair(&mut st, 6, s0);
        for (lane, x) in s1.into_iter().enumerate() {
            pair64(&mut st, 2, lane, x);
        }
        run(CMPX_GT_U64, &mut st).unwrap();
        assert_eq!(st.exec, u64::from(want), "{s0:#x} > {s1:x?} (unsigned)");
    }
    // A lane that is already off stays off even though its compare would be true; the result replaces EXEC.
    let mut st = state();
    st.exec = 0b0101;
    spair(&mut st, 6, u64::MAX);
    for lane in 0..4 {
        pair64(&mut st, 2, lane, 1);
    }
    run(CMPX_GT_U64, &mut st).unwrap();
    assert_eq!(st.exec, 0b0101);
}

// ---- the table rows themselves -----------------------------------------------------------------------------------

/// The three new rows agree with the llvm-mc text pinned in `gfx1100.tbl` (and are inherited, byte for byte, by the
/// gfx1151 table, which is the gfx1100 table plus deltas).
#[test]
fn new_gfx1100_rows_pin_their_llvm_text() {
    let rows: [(&str, &str, &[u32]); 3] = [
        (
            "v_cmp_ge_i64_e64",
            "v_cmp_ge_i64_e64 s28, s[12:13], s[2:3]",
            CMP_GE_I64_S28,
        ),
        (
            "v_ashrrev_i64",
            "v_ashrrev_i64 v[0:1], 31, v[0:1]",
            ASHR_V01_31,
        ),
        (
            "v_cmpx_gt_u64_e32",
            "v_cmpx_gt_u64_e32 s[6:7], v[2:3]",
            CMPX_GT_U64,
        ),
    ];
    for (name, text, words) in rows {
        let inst = roundtrip(words, name);
        let row = crate::isa::lookup_inst(Arch::Gfx1100, &inst).ok().unwrap();
        let table: Vec<u32> = row
            .encoding
            .split_whitespace()
            .map(|w| u32::from_str_radix(w, 16).unwrap())
            .collect();
        assert_eq!((row.sample, table.as_slice()), (text, words), "{name}");
        // gfx1151's table is the gfx1100 table plus deltas and encodes these three identically (llvm-mc
        // --mcpu=gfx1151 assembles the same words), so the rows are inherited, not diverged.
        let (inst151, n) = gfx11::decode(Arch::Gfx1151, words).unwrap();
        assert_eq!(n, words.len(), "{name}");
        assert_eq!(
            crate::isa::lookup_inst(Arch::Gfx1151, &inst151)
                .unwrap()
                .name,
            name
        );
        assert_eq!(
            gfx11::encode(Arch::Gfx1151, &inst151).unwrap().as_slice(),
            words,
            "{name}: gfx1151 re-encode"
        );
    }
}
