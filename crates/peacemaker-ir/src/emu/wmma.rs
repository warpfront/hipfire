//! Wave32 WMMA register-fragment gather/scatter. The numerical contract
//! (accumulation order, rounding, saturation) lives in `super::mma`; this
//! module only maps ISA fragments to logical matrices, so a mapping error
//! cannot hide inside a numerics model. All A/B/C inputs are gathered before
//! any destination write (D may alias A/B/C), and EXEC is ignored: WMMA
//! always reads and writes all 32 lanes.
use super::{mma, Result, State};
use crate::{operand::{Omod, Operand}, reg::Kind, Arch, Inst};

#[derive(Clone, Copy)]
enum Shape { F16, Iu4 { k: usize } }

pub(super) fn execute(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let shape = match (arch, name) {
        (Arch::Gfx1151 | Arch::Gfx1201, "v_wmma_f32_16x16x16_f16") => Shape::F16,
        (Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4") => Shape::Iu4 { k: 16 },
        (Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4") => Shape::Iu4 { k: 32 },
        _ => return Err(format!("{name} on {arch:?}: no source-verified fragment mapping and probed numerics; refusing to approximate")),
    };
    let m = &i.mods;
    if m.neg != 0 || m.abs != 0 || m.omod != Omod::None || m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!("{name}: VOP3 neg/abs/omod/DPP/SDWA modifiers are not part of WMMA"));
    }
    if m.op_sel != 0 || !matches!(m.op_sel_hi, 0 | 7) {
        return Err(format!("{name}: op_sel={:#x} op_sel_hi={:#x} has no modeled WMMA meaning", m.op_sel, m.op_sel_hi));
    }
    let o = &i.operands;
    if o.len() != 4 { return Err(format!("{name}: expected vdst,A,B,C operands, got {}", o.len())); }
    let gfx12 = arch == Arch::Gfx1201;
    let (a_regs, b_regs) = match shape {
        Shape::F16 => (if gfx12 { 4 } else { 8 }, if gfx12 { 4 } else { 8 }),
        Shape::Iu4 { .. } => (2, 2),
    };
    vgprs(&o[0], 8, "D")?;
    vgprs(&o[1], a_regs, "A")?;
    vgprs(&o[2], b_regs, "B")?;
    match &o[3] {
        Operand::Reg(_) => vgprs(&o[3], 8, "C")?,
        Operand::Inline(_) | Operand::Literal(_) => {}
        other => return Err(format!("{name}: unsupported C source {other:?}")),
    }
    if !gfx12 {
        // RDNA3 requires the A/B fragments replicated across lanes 0..15 and 16..31;
        // divergent halves are out of contract, so reject them before any gather.
        for (role, op, regs) in [("A", &o[1], a_regs), ("B", &o[2], b_regs)] {
            for vgpr in 0..usize::from(regs) {
                for lane in 0..16 {
                    let (lo, hi) = (s.read(op, lane, vgpr)?, s.read(op, lane + 16, vgpr)?);
                    if lo != hi {
                        return Err(format!("{name}: gfx11 {role} fragment VGPR {vgpr} lane {lane} ({lo:#010x}) differs from replica lane {} ({hi:#010x})", lane + 16));
                    }
                }
            }
        }
    }

    let mut out = [[0u32; 32]; 8];
    match shape {
        Shape::F16 => {
            if m.clamp { return Err(format!("{name}: clamp has no probed float-WMMA semantics")); }
            // NEG[1:0]/NEG_HI[1:0] negate A/B low/high halves; {NEG_HI[2],NEG[2]} = {ABS,NEG} on C.
            let flip = |operand: u8, half: usize| -> u16 {
                let bits = if half == 0 { m.neg_lo } else { m.neg_hi };
                if bits >> operand & 1 != 0 { 0x8000 } else { 0 }
            };
            let mut a = [[0u16; 16]; 16];
            let mut b = [[0u16; 16]; 16];
            for idx in 0..16 {
                for k in 0..16 {
                    let (lane, vgpr, half) = f16_loc(arch, idx, k);
                    a[idx][k] = (s.read(&o[1], lane, vgpr)? >> (16 * half)) as u16 ^ flip(0, half);
                    b[idx][k] = (s.read(&o[2], lane, vgpr)? >> (16 * half)) as u16 ^ flip(1, half);
                }
            }
            let mut c = [[0u32; 16]; 16];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    let mut v = s.read(&o[3], lane, vgpr)?;
                    if m.neg_hi & 4 != 0 { v &= 0x7fff_ffff; }
                    if m.neg_lo & 4 != 0 { v ^= 0x8000_0000; }
                    c[row][col] = v;
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    out[vgpr][lane] = mma::wmma_f32_f16(arch, &a[row], &b[col], c[row][col]);
                }
            }
        }
        Shape::Iu4 { k: depth } => {
            // NEG[0]/NEG[1] are the A/B signedness bits; NEG[2] and NEG_HI must be zero.
            if m.neg_lo & 4 != 0 || m.neg_hi != 0 {
                return Err(format!("{name}: NEG[2]/NEG_HI are undefined for integer WMMA"));
            }
            let signed = [m.neg_lo & 1 != 0, m.neg_lo & 2 != 0];
            let nibble = |word: u32, n: usize, signed: bool| -> i8 {
                let v = ((word >> (4 * n)) & 15) as i8;
                if signed { (v << 4) >> 4 } else { v }
            };
            let mut a = [[0i8; 32]; 16];
            let mut b = [[0i8; 32]; 16];
            for idx in 0..16 {
                for k in 0..depth {
                    let (lane, vgpr, n) = iu4_loc(arch, idx, k);
                    a[idx][k] = nibble(s.read(&o[1], lane, vgpr)?, n, signed[0]);
                    b[idx][k] = nibble(s.read(&o[2], lane, vgpr)?, n, signed[1]);
                }
            }
            let mut c = [[0i32; 16]; 16];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    c[row][col] = s.read(&o[3], lane, vgpr)? as i32;
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    out[vgpr][lane] = mma::wmma_i32_iu4(arch, &a[row][..depth], &b[col][..depth], c[row][col], m.clamp) as u32;
                }
            }
        }
    }
    for (vgpr, lanes) in out.iter().enumerate() {
        for (lane, &value) in lanes.iter().enumerate() { s.put(&o[0], lane, vgpr, value)?; }
    }
    Ok(())
}

fn vgprs(op: &Operand, want: u8, role: &str) -> Result<()> {
    match op {
        Operand::Reg(r) if r.kind == Kind::V && r.len == want => Ok(()),
        other => Err(format!("WMMA {role} must be v[..] of {want} registers, got {other:?}")),
    }
}

/// F16 A[r,k] / B[k,n] (`idx` = r or n) -> (lane, VGPR, half). gfx11 replicates
/// the fragment in lanes 16..31 (validated in `execute`); the low copy is read.
fn f16_loc(arch: Arch, idx: usize, k: usize) -> (usize, usize, usize) {
    if arch == Arch::Gfx1201 { (((k >> 2) & 1) * 16 + idx, (k >> 3) * 2 + ((k >> 1) & 1), k & 1) } else { (idx, k >> 1, k & 1) }
}

/// IU4 A[r,k] / B[k,n] -> (lane, VGPR, nibble).
fn iu4_loc(arch: Arch, idx: usize, k: usize) -> (usize, usize, usize) {
    if arch == Arch::Gfx1201 { (((k >> 3) & 1) * 16 + idx, k >> 4, k & 7) } else { (idx, k >> 3, k & 7) }
}

/// C/D[row,col] -> (lane, VGPR).
fn acc_loc(arch: Arch, row: usize, col: usize) -> (usize, usize) {
    if arch == Arch::Gfx1201 { ((row >> 3) * 16 + col, row & 7) } else { ((row & 1) * 16 + col, row >> 1) }
}

#[cfg(test)] mod tests {
    use super::*;
    use crate::{operand::Modifiers, FormFields};

    fn state() -> State { State { s: [0; 106], t: [0; 16], v: Box::new([[0; 32]; 256]), exec: u32::MAX, vcc: 0, scc: false, m0: 0, pc: 0, ended: false, barrier: None, signals: 0 } }
    fn reg(base: u16, len: u8) -> Operand { Operand::Reg(crate::reg::RegRef { kind: Kind::V, base, len }) }
    fn insn(arch: Arch, name: &str, ops: [Operand; 4], mods: Modifiers) -> Inst {
        let table = if arch == Arch::Gfx1201 { crate::isa::gfx12() } else { crate::isa::gfx1151() };
        let row = table.iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(arch, row.op, row.form, FormFields::default(), ops.into_iter().collect(), mods, None, Default::default()).unwrap()
    }
    fn mods(neg_lo: u8, neg_hi: u8) -> Modifiers { Modifiers { neg_lo, neg_hi, op_sel_hi: 7, ..Default::default() } }
    const D: u16 = 0; const A: u16 = 16; const B: u16 = 32; const C: u16 = 48;
    fn put_half(st: &mut State, base: u16, vgpr: usize, lane: usize, half: usize, bits: u16) { st.v[usize::from(base) + vgpr][lane] |= u32::from(bits) << (16 * half); }
    fn put_nib(st: &mut State, base: u16, vgpr: usize, lane: usize, nib: usize, v: u32) { st.v[usize::from(base) + vgpr][lane] |= (v & 15) << (4 * nib); }
    /// Every D word other than `hit` must equal C (zero here); returns the hit word.
    fn only(st: &State, hit: (usize, usize)) -> u32 {
        for vgpr in 0..8 { for lane in 0..32 { if (vgpr, lane) != hit { assert_eq!(st.v[usize::from(D) + vgpr][lane], 0, "D vgpr {vgpr} lane {lane}"); } } }
        st.v[usize::from(D) + hit.0][hit.1]
    }

    // gfx11 A[5,13]: lane 5 (+16 replica), VGPR 13/2=6, half 1. B[13,7]: lane 7, VGPR 6, half 1.
    // D[5,7]: VGPR 5/2=2, lane (5&1)*16+7=23.
    #[test] fn gfx11_f16_fragments() {
        let mut st = state();
        for lane in [5, 21] { put_half(&mut st, A, 6, lane, 1, 0x4200); }
        for lane in [7, 23] { put_half(&mut st, B, 6, lane, 1, 0x4000); }
        st.v[usize::from(C) + 2][23] = 1f32.to_bits();
        let i = insn(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 8), reg(B, 8), reg(C, 8)], mods(0, 0));
        execute(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (2, 23)), 7f32.to_bits());
    }

    // gfx12 A[5,13]: k=0b1101 -> half 1, VGPR (1<<1)|0=2, lane 16+5. B[13,7]: lane 16+7.
    // D[5,7]: lane (5>>3)*16+7=7, VGPR 5. EXEC=0 must not suppress the write.
    #[test] fn gfx12_f16_fragments_ignore_exec() {
        let mut st = state(); st.exec = 0;
        put_half(&mut st, A, 2, 21, 1, 0x4200);
        put_half(&mut st, B, 2, 23, 1, 0x4000);
        st.v[usize::from(C) + 5][7] = 1f32.to_bits();
        let i = insn(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)], mods(0, 0));
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (5, 7)), 7f32.to_bits());
        // Row 9 column 2 lives in lane 16+2, VGPR 1; A[9,2] (k=0b0010: half 0, VGPR 1, lane group 0) -> lane 9; B[2,2] -> lane 2.
        let mut st = state();
        put_half(&mut st, A, 1, 9, 0, 0x3c00);
        put_half(&mut st, B, 1, 2, 0, 0x4400);
        let i = insn(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)], mods(0, 0));
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (1, 18)), 4f32.to_bits());
    }

    // NEG[0]/NEG_HI[0] negate A's low/high half; NEG[1] negates B low; {NEG_HI[2],NEG[2]} = {ABS,NEG} on C.
    #[test] fn f16_per_half_neg_and_c_abs_neg() {
        // gfx12 A[0,1] (half 1, VGPR 0, lane 0), B[1,0] (half 1, lane 0): 2 * 3 = 6; C[0,0] = -10.
        let build = |m: Modifiers| {
            let mut st = state();
            put_half(&mut st, A, 0, 0, 1, 0x4000); put_half(&mut st, B, 0, 0, 1, 0x4200);
            st.v[usize::from(C)][0] = (-10f32).to_bits();
            let i = insn(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)], m);
            execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
            f32::from_bits(st.v[usize::from(D)][0])
        };
        assert_eq!(build(mods(0, 0)), -4.0);
        assert_eq!(build(mods(1, 0)), -4.0); // low-half NEG does not touch the high-half element
        assert_eq!(build(mods(0, 1)), -16.0); // NEG_HI[0]: A[0,1] -> -2
        assert_eq!(build(mods(0, 3)), -4.0); // A and B both negated
        assert_eq!(build(mods(4, 0)), 16.0); // NEG[2]: C -> +10
        assert_eq!(build(mods(0, 4)), 16.0); // ABS(C) = +10
        assert_eq!(build(mods(4, 4)), -4.0); // -ABS(C) = -10
    }

    // gfx11 K16 A[3,10]: lane 3, VGPR 1, nibble 2. B[10,6]: lane 6. D[3,6]: lane 16+6, VGPR 1.
    #[test] fn gfx11_iu4_fragments_and_signedness() {
        let run = |neg_lo: u8| {
            let mut st = state();
            for lane in [3, 19] { put_nib(&mut st, A, 1, lane, 2, 0xe); }
            for lane in [6, 22] { put_nib(&mut st, B, 1, lane, 2, 3); }
            st.v[usize::from(C) + 1][22] = 100;
            let i = insn(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)], mods(neg_lo, 0));
            execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", &i, &mut st).unwrap();
            only(&st, (1, 22)) as i32
        };
        assert_eq!(run(3), 94);  // 0xe signed = -2; -2*3 + 100
        assert_eq!(run(2), 142); // A unsigned 14, B signed 3
        assert_eq!(run(0), 142);
    }

    #[test] fn gfx11_divergent_replicas_are_hard_errors() {
        let f16 = |a_hi: bool, b_hi: bool, exec: u32| {
            let mut st = state(); st.exec = exec;
            for lane in [5, 21] { put_half(&mut st, A, 6, lane, 1, 0x4200); }
            for lane in [7, 23] { put_half(&mut st, B, 6, lane, 1, 0x4000); }
            if a_hi { st.v[usize::from(A) + 7][31] ^= 1; }
            if b_hi { st.v[usize::from(B)][16] ^= 0x8000; }
            let i = insn(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 8), reg(B, 8), reg(C, 8)], mods(0, 0));
            let r = execute(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", &i, &mut st);
            (r, st)
        };
        assert!(f16(false, false, u32::MAX).0.is_ok());
        assert!(f16(false, false, 0).0.is_ok());
        for exec in [u32::MAX, 0] {
            for (a, b) in [(true, false), (false, true), (true, true)] {
                let (r, st) = f16(a, b, exec);
                assert!(r.is_err(), "a={a} b={b} exec={exec:#x}");
                assert!(st.v[..8].iter().all(|row| row.iter().all(|&w| w == 0)), "D written despite error");
            }
        }
        let iu4 = |a_hi: bool, b_hi: bool, exec: u32| {
            let mut st = state(); st.exec = exec;
            for lane in [3, 19] { put_nib(&mut st, A, 1, lane, 2, 0xe); }
            for lane in [6, 22] { put_nib(&mut st, B, 1, lane, 2, 3); }
            if a_hi { put_nib(&mut st, A, 0, 31, 7, 1); }
            if b_hi { put_nib(&mut st, B, 1, 16, 0, 1); }
            let i = insn(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)], mods(3, 0));
            execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", &i, &mut st)
        };
        assert!(iu4(false, false, 0).is_ok());
        for exec in [u32::MAX, 0] {
            for (a, b) in [(true, false), (false, true)] { assert!(iu4(a, b, exec).is_err(), "a={a} b={b} exec={exec:#x}"); }
        }
    }

    // gfx12 K32 A[3,26]: k=0b11010 -> nibble 2, lane group 1 => lane 19, VGPR 1. B[26,6]: lane 22.
    // D[3,6]: lane 6, VGPR 3. A[9,5] -> lane 9, VGPR 0, nibble 5; B[5,15] -> lane 15; D[9,15]: lane 31, VGPR 1.
    #[test] fn gfx12_iu4_fragments() {
        let setup = || {
            let mut st = state();
            put_nib(&mut st, A, 1, 19, 2, 0xe); put_nib(&mut st, B, 1, 22, 2, 3);
            st
        };
        // Inline-constant C (0) accumulator: -2 * 3 = -6.
        let mut st = setup();
        let i = insn(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", [reg(D, 8), reg(A, 2), reg(B, 2), Operand::Inline(crate::operand::InlineConst::Integer(0))], mods(3, 0));
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (3, 6)) as i32, -6);
        let mut st = setup();
        st.v[usize::from(C) + 3][6] = 100;
        let i = insn(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)], mods(3, 0));
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (3, 6)) as i32, 94);
        let mut st = state();
        put_nib(&mut st, A, 0, 9, 5, 7); put_nib(&mut st, B, 0, 15, 5, 7);
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (1, 31)) as i32, 49);
    }

    #[test] fn unsupported_variants_are_hard_errors() {
        let mut st = state();
        let i = insn(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)], mods(0, 0));
        assert!(execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_bf16", &i, &mut st).is_err());
        assert!(execute(Arch::Gfx1100, "v_wmma_f32_16x16x16_f16", &i, &mut st).is_err());
        assert!(execute(Arch::Gfx1151, "v_wmma_i32_16x16x32_iu4", &i, &mut st).is_err());
        let clamped = insn(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)], Modifiers { clamp: true, ..mods(0, 0) });
        assert!(execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &clamped, &mut st).is_err());
    }
}
