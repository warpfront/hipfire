// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! G80's all-integer direct-BF16 epilogue. Shares IEF15's integer encoders,
//! dependency scheduler, row/column shuffles and highest-bit normalization.
use super::*;

/// Data-memory map and width of one BF16 core tile. `WIDE` is the 128x80 G80 tile (the pinned legacy map);
/// `NARROW` is the 128x64 tile of the N=512 K=6144 program (V8 layout-0-like map: A, B, scratch low, C 0x8000..0x10000).
/// `c` is the base of the 128 x `tn` i32 accumulator; the BF16 output is staged in its upper half `[out, c + 128*tn*4)`.
#[derive(Clone, Copy)]
pub(crate) struct Tile {
    pub tn: usize,
    pub a: [u32; 2],
    pub b: [u32; 2],
    pub c: u32,
    pub out: u32,
    pub d: u32,
}

pub(crate) const WIDE: Tile = Tile {
    tn: 80,
    a: [0xa000, 0xb200],
    b: [0xc400, 0xd980],
    c: 0,
    out: 0x5000,
    d: 0xef00,
};
pub(crate) const NARROW: Tile = Tile {
    tn: 64,
    a: [0x0000, 0x2000],
    b: [0x4000, 0x5200],
    c: 0x8000,
    out: 0xc000,
    d: 0x6400,
};

impl Tile {
    /// 8-feature blocks per tile row.
    pub(crate) const fn nb(self) -> usize {
        self.tn / 8
    }
    /// Feature sidecar lanes per input vector load (multiples of 32).
    const fn lanes(self) -> usize {
        self.tn.div_ceil(32) * 32
    }
    pub(crate) const fn a_bytes(self) -> usize {
        4096 + 512
    }
    /// K64 code bytes of one B tile, then `Sm[lanes]` u16 and `Se[lanes]` i16.
    pub(crate) const fn b_codes(self) -> usize {
        self.tn * 64
    }
    pub(crate) const fn b_bytes(self) -> usize {
        self.b_codes() + 4 * self.lanes()
    }
    pub(crate) const fn out_bytes(self) -> usize {
        128 * self.tn * 2
    }
    pub(crate) const fn c_bytes(self) -> usize {
        128 * self.tn * 4
    }
    const fn raw_s(self) -> u32 {
        self.d + 512
    }
    const fn dpat(self) -> u32 {
        self.raw_s()
    }
    const fn konst(self) -> u32 {
        self.dpat() + 1024
    }
    const fn spat(self) -> u32 {
        self.konst() + 23 * 64
    }
    const fn sepat(self) -> u32 {
        self.spat() + (self.nb() * 64) as u32
    }
    const fn cp3(self) -> u32 {
        CORE_BASE + self.konst() + 8 * 64
    }
    const fn cp2(self) -> u32 {
        CORE_BASE + self.konst() + 24 * 64
    }
}
const _: () = assert!(
    WIDE.b[1] as usize + WIDE.b_bytes() == WIDE.d as usize
        && WIDE.sepat() as usize + 640 <= 0x10000
        && WIDE.nb() == 10
        && WIDE.lanes() == 96
);
// Narrow map: A, B (both slots) and the epilogue scratch lie below C and do not overlap each other or C.
const _: () = assert!(
    NARROW.a[0] as usize + NARROW.a_bytes() <= NARROW.a[1] as usize
        && NARROW.a[1] as usize + NARROW.a_bytes() <= NARROW.b[0] as usize
        && NARROW.b[0] as usize + NARROW.b_bytes() <= NARROW.b[1] as usize
        && NARROW.b[1] as usize + NARROW.b_bytes() <= NARROW.d as usize
        && NARROW.sepat() as usize + 512 <= NARROW.c as usize
        && NARROW.out as usize + NARROW.out_bytes() == NARROW.c as usize + NARROW.c_bytes()
        && NARROW.c as usize + NARROW.c_bytes() <= 0x10000
        && NARROW.b_bytes() == 4352
);

pub(crate) const A: [u32; 2] = WIDE.a;
pub(crate) const B: [u32; 2] = WIDE.b;
pub(crate) const A_BYTES: usize = WIDE.a_bytes();
pub(crate) const B_BYTES: usize = WIDE.b_bytes();
pub(crate) const OUT: u32 = WIDE.out;
pub(crate) const OUT_BYTES: usize = WIDE.out_bytes();
const DPAT: u32 = WIDE.dpat();

fn k(l: &mut List, dst: usize, value: Konst) {
    konst(l, dst, value);
}
fn imm(l: &mut List, dst: usize, value: i32) {
    l.push(movxm_r(15, value as u32));
    l.push(bcst("VBCST_16", dst, 15));
}

pub(crate) fn initialize(a: &mut Assembler) {
    initialize_for(a, WIDE)
}

pub(crate) fn initialize_for(a: &mut Assembler, t: Tile) {
    for (reg, v) in [
        (R_UU, CONF_UU),
        (R_US, CONF_US),
        (R_US16, CONF_US16),
        (R_UU16, CONF_UU16),
        (R_NEG, CONF_NEG64),
        (R_M16, (-16i32) as u32),
    ] {
        a.mov(0, reg, v);
    }
    for mode in 12..=19 {
        a.mov(0, 20 + mode - 12, mode as u32);
    }
    a.set_cr("crSat", 0);
    a.set_cr("crUPSMode", 1);
    let mut l = List::default();
    l.push(movxm_p(7, CORE_BASE + t.konst()));
    for &(kind, original, _) in &KONST[..23] {
        let value = match kind {
            Konst::K23 => 7,
            Konst::KM149 => (-133i32) as u32,
            Konst::K55 => 63,
            Konst::K149 => 133,
            Konst::Inf32 => 0x7f80,
            Konst::Sign32 => 0x8000,
            _ => original,
        };
        l.push(movxm_r(15, value));
        l.push(bcst("VBCST_16", 9, 15));
        l.push(st(9, 7, 1, MEM_CONST));
    }
    emit_block(a, &l.finish());
}

/// Copy the last chunk's sidecars while the input locks are still held.
pub(crate) fn capture(a: &mut Assembler, slot: usize) {
    capture_for(a, slot, WIDE)
}

pub(crate) fn capture_for(a: &mut Assembler, slot: usize, t: Tile) {
    let mut l = List::default();
    l.push(movxm_p(0, CORE_BASE + t.a[slot] + 4096));
    l.push(movxm_p(1, CORE_BASE + t.b[slot] + t.b_codes() as u32));
    l.push(movxm_p(7, CORE_BASE + t.d));
    for _ in 0..8 {
        l.push(ld(8, 0, 1, false, MEM_A));
        l.push(st(8, 7, 1, MEM_PAT));
    }
    for _ in 0..t.lanes() / 16 {
        l.push(ld(8, 1, 1, true, MEM_B));
        l.push(st(8, 7, 1, MEM_PAT));
    }
    emit_block(a, &l.finish());
}

/// Zero the I32 tile, allowing one accumulating wavefront body instead of two
/// copies of the compute body in the 16-KiB instruction memory.
pub(crate) fn zero(a: &mut Assembler) {
    zero_for(a, WIDE)
}

pub(crate) fn zero_for(a: &mut Assembler, t: Tile) {
    let mut l = List::default();
    l.push(movxm_p(7, CORE_BASE + t.c));
    l.push(movxm_r(15, 0));
    l.push(bcst("VBCST_32", 11, 15));
    emit_block(a, &l.finish());
    let mut body = List::default();
    for _ in 0..16 {
        body.push(st(11, 7, 1, MEM_I));
    }
    emit_loop(a, body.finish(), (t.c_bytes() / (16 * 64)) as u32, R_LOOP);
}

fn columns(t: Tile) -> Vec<Cycle> {
    let mut l = List::default();
    let half = t.lanes() as u32 * 2;
    for (source, target) in [(t.raw_s(), t.spat()), (t.raw_s() + half, t.sepat())] {
        l.push(movxm_p(5, CORE_BASE + source));
        l.push(movxm_p(6, CORE_BASE + target));
        for group in 0..t.lanes() / 32 {
            l.push(ld(8, 5, 1, true, MEM_PAT));
            col_patterns(&mut l, 8, 0);
            for nb in 0..(t.nb() - 4 * group).min(4) {
                l.push(st(nb, 6, 1, MEM_PAT));
            }
        }
    }
    l.finish()
}

fn rows(t: Tile) -> Vec<Cycle> {
    let mut l = List::default();
    l.push(ld_at(6, 0, 0, false, MEM_PAT));
    l.push(ld_at(7, 0, 4, true, MEM_PAT));
    l.push(movxm_p(1, CORE_BASE + t.dpat() + 512));
    for (src, kind) in [(6, 0), (7, 1)] {
        row_patterns(&mut l, src, |l, mb, half, reg| {
            l.push(st_at(
                reg,
                1,
                (4 * mb + 2 * half + kind) as i64 - 8,
                MEM_PAT,
            ));
        });
    }
    l.push(padd(Slot::Lda, 0, 2)); // previous 32 tokens, -64 B
    l.finish()
}

/// One descending group of 32 outputs. Exact four-limb product, then a
/// sticky-preserving normalization to scalar SRS shifts 7/23/39/31.
fn group(t: Tile) -> Vec<Cycle> {
    use Konst::*;
    let mut l = List::default();
    let (lt, ge, eqz16, select, add, sub) = (
        "VLT_16_vaddSign1",
        "VGE_16_vaddSign0",
        "VEQZ_16",
        "VSEL_16",
        "VADD_16",
        "VSUB_16",
    );
    l.push(movx_cr("crRnd", 0));
    l.barrier();
    l.push(ld_at(0, 5, 0, false, MEM_PAT)); // Dm, four tokens repeated eight times
    l.push(ld_at(1, 5, 1, true, MEM_PAT)); // De
    l.push(ld_at(2, 1, 0, false, MEM_PAT)); // Sm, eight features repeated four times
    l.push(ld_at(3, 7, 0, true, MEM_PAT)); // Se
    l.push(padd(Slot::Lda, 5, 0)); // previous half's patterns, -128 B
    l.push(vmac(4, None, x(0), x(2), R_UU));
    l.push(alu(add, 0, 1, 3));
    imm(&mut l, 8, -300);
    l.push(alu("VMAX_LT_16_vaddSign1", 0, 0, 8));
    imm(&mut l, 8, 300);
    l.push(alu("VMIN_GE_16_vaddSign1", 0, 0, 8));
    l.push(movxm_s(0, 0));
    l.push(movxm_s(1, 16));
    l.push(srs4(4, 4, 0));
    l.push(srs4(5, 4, 1)); // Plo/Phi
                           // C's 32 I32 lanes: high half first (descending p6), then low half.
    l.push(ld(9, 6, -1, false, MEM_I));
    l.push(ld(8, 6, -1, true, MEM_I));
    l.push(movxm_r(29, 2));
    l.push(ins(
        "VSHUFFLE_vec_shuffle_x",
        &[
            ("dst", Rg(x(6))),
            ("s1", Rg(x(8))),
            ("s2", Rg(x(9))),
            ("mod", Rg(r(29))),
        ],
    ));
    l.push(movxm_r(29, 3));
    l.push(ins(
        "VSHUFFLE_vec_shuffle_x",
        &[
            ("dst", Rg(x(7))),
            ("s1", Rg(x(8))),
            ("s2", Rg(x(9))),
            ("mod", Rg(r(29))),
        ],
    ));
    // Ihi signed, Ilo unsigned; P positive. Each intermediate is exact acc64.
    l.push(vmac(0, None, x(5), x(7), R_US));
    l.push(vmac(0, Some(0), x(4), x(7), R_US16));
    l.push(vmac(0, Some(0), x(5), x(6), R_UU));
    l.push(vmac(0, Some(0), x(4), x(6), R_UU16));
    l.push(movxm_p(2, t.cp2()));
    l.push(movxm_p(3, t.cp3()));
    l.push(vneg(1, 0));
    l.push(movxm_s(2, 32));
    l.push(movxm_s(3, 48));
    k(&mut l, 3, Zero);
    l.push(srs4(1, 0, 3));
    l.push(srs4(2, 1, 3));
    l.push(cmp(lt, R_NEGMASK, 1, 3));
    l.push(sel(select, 4, 1, 2, R_NEGMASK));
    for (dst, sh) in [(5, 2), (6, 1), (7, 0)] {
        l.push(srs4(1, 0, sh));
        l.push(srs4(2, 1, sh));
        l.push(sel(select, dst, 1, 2, R_NEGMASK));
    }
    l.push(eqz(eqz16, 17, 4));
    l.push(eqz(eqz16, 18, 5));
    l.push(eqz(eqz16, 19, 6));
    l.push(sca("AND", 28, 17, 18));
    l.push(sca("AND", 29, 28, 19));
    l.push(sel(select, 1, 4, 5, 17));
    l.push(sel(select, 1, 1, 6, 28));
    l.push(sel(select, 1, 1, 7, 29));
    k(&mut l, 2, K48);
    k(&mut l, 8, K32);
    l.push(sel(select, 2, 2, 8, 17));
    k(&mut l, 8, Sixteen);
    l.push(sel(select, 2, 2, 8, 28));
    l.push(sel(select, 2, 2, 3, 29));
    for (threshold, increment, shift) in [
        (K256, Eight, 8),
        (Sixteen, Four, 4),
        (Four, Two, 2),
        (Two, One, 1),
    ] {
        k(&mut l, 8, threshold);
        l.push(cmp(ge, 30, 1, 8));
        l.push(movxm_s(2, shift));
        l.push(ups4(2, 1, 0, false));
        l.push(srs4(9, 2, 2));
        l.push(sel(select, 1, 1, 9, 30));
        k(&mut l, 8, increment);
        l.push(alu(add, 9, 2, 8));
        l.push(sel(select, 2, 2, 9, 30));
    }
    k(&mut l, 8, K23);
    l.push(alu(sub, 9, 2, 8)); // L-7
    k(&mut l, 8, KM149);
    l.push(alu(sub, 10, 8, 0)); // -133-X
    l.push(alu("VMAX_LT_16_vaddSign1", 9, 9, 10));
    k(&mut l, 8, K55);
    l.push(alu("VMIN_GE_16_vaddSign1", 9, 9, 8)); // cap at63
    k(&mut l, 8, Eight);
    l.push(cmp(lt, 17, 9, 8));
    k(&mut l, 8, K24);
    l.push(cmp(lt, 18, 9, 8));
    k(&mut l, 8, K40);
    l.push(cmp(lt, 19, 9, 8));
    imm(&mut l, 8, 56);
    l.push(cmp(lt, 28, 9, 8));
    k(&mut l, 10, K55);
    imm(&mut l, 8, 55);
    l.push(sel(select, 10, 10, 8, 28));
    k(&mut l, 8, K39);
    l.push(sel(select, 10, 10, 8, 19));
    imm(&mut l, 8, 23);
    l.push(sel(select, 10, 10, 8, 18));
    k(&mut l, 8, K7);
    l.push(sel(select, 10, 10, 8, 17));
    l.push(alu(sub, 10, 10, 9));
    // p=2^k, 0<=k<=15, exactly the IEF15 construction.
    k(&mut l, 8, Eight);
    l.push(alu("VBAND", 11, 10, 8));
    l.push(eqz(eqz16, 29, 11));
    k(&mut l, 8, Four);
    l.push(alu("VBAND", 11, 10, 8));
    l.push(eqz(eqz16, 30, 11));
    k(&mut l, 8, Sixteen);
    k(&mut l, 1, One);
    l.push(sel(select, 8, 8, 1, 30));
    k(&mut l, 11, K4096);
    k(&mut l, 1, K256);
    l.push(sel(select, 11, 11, 1, 30));
    l.push(sel(select, 11, 11, 8, 29));
    k(&mut l, 8, Two);
    l.push(alu("VBAND", 1, 10, 8));
    l.push(eqz(eqz16, 29, 1));
    k(&mut l, 8, One);
    l.push(alu("VBAND", 1, 10, 8));
    l.push(eqz(eqz16, 30, 1));
    k(&mut l, 8, Two);
    k(&mut l, 1, One);
    l.push(sel(select, 8, 8, 1, 30));
    k(&mut l, 10, Eight);
    k(&mut l, 1, Four);
    l.push(sel(select, 10, 10, 1, 30));
    l.push(sel(select, 10, 10, 8, 29));
    l.push(vmac(2, None, x(11), x(10), R_UU));
    l.push(srs4(11, 2, 0));
    // Drop16/32 low bits for the top classes, preserving their sticky bit.
    l.push(eqz(eqz16, 29, 7));
    k(&mut l, 8, One);
    l.push(alu("VBOR", 8, 6, 8));
    l.push(sel(select, 8, 8, 6, 29));
    l.push(alu("VBOR", 10, 6, 7));
    l.push(eqz(eqz16, 30, 10));
    k(&mut l, 10, One);
    l.push(alu("VBOR", 10, 5, 10));
    l.push(sel(select, 10, 10, 5, 30));
    l.push(sel(select, 10, 10, 8, 28));
    l.push(sel(select, 10, 10, 7, 19));
    l.push(sel(select, 1, 4, 5, 28));
    l.push(sel(select, 1, 1, 6, 19));
    l.push(sel(select, 2, 3, 4, 28));
    l.push(sel(select, 2, 2, 5, 19));
    l.push(sel(select, 4, 3, 4, 19));
    l.push(vmac(3, None, x(4), x(11), R_UU));
    l.push(vmac(3, Some(3), x(2), x(11), R_UU16));
    l.push(vmac(3, Some(3), x(1), x(11), R_UU16));
    l.push(vmac(3, Some(3), x(10), x(11), R_UU16));
    l.push(alu(add, 5, 9, 0));
    k(&mut l, 8, K149);
    l.push(alu(add, 5, 5, 8));
    k(&mut l, 8, K254);
    l.push(alu("VMIN_GE_16_vaddSign1", 5, 5, 8));
    l.push(alu("VMAX_LT_16_vaddSign1", 5, 5, 3));
    l.push(movxm_s(1, 7));
    l.push(ups4(4, 5, 1, true));
    l.push(srs4(5, 4, 0));
    l.barrier();
    l.push(movx_cr("crRnd", 12));
    l.barrier();
    l.push(movxm_s(2, 23));
    l.push(movxm_s(3, 39));
    l.push(srs4(1, 3, 1));
    l.push(srs4(2, 3, 2));
    l.push(srs4(6, 3, 3));
    l.push(movxm_s(2, 31));
    l.push(srs4(7, 3, 2));
    l.barrier();
    l.push(movx_cr("crRnd", 0));
    l.barrier();
    l.push(sel(select, 6, 6, 2, 18));
    l.push(sel(select, 6, 6, 1, 17));
    l.push(sel(select, 6, 7, 6, 28));
    l.push(alu(add, 9, 6, 5));
    k(&mut l, 8, Inf32);
    l.push(alu("VMIN_GE_16_vaddSign0", 9, 9, 8));
    l.push(eqz(eqz16, 29, 6));
    l.push(sel(select, 9, 9, 3, 29));
    k(&mut l, 8, Sign32);
    l.push(alu("VBOR", 8, 9, 8));
    l.push(sel(select, 9, 9, 8, R_NEGMASK));
    l.push(st(9, 4, -1, MEM_OUT));
    l.push(movxm_r(R_NEGMASK, T128_HI as u32));
    l.finish()
}

pub(crate) fn epilogue(a: &mut Assembler) {
    epilogue_for(a, WIDE)
}

pub(crate) fn epilogue_for(a: &mut Assembler, t: Tile) {
    a.set_cr("crSRSMode", 1);
    for (m, value) in [-128i32, 256, -64, -256].into_iter().enumerate() {
        a.mov(2, m, value as u32);
    }
    emit_block(a, &columns(t));
    a.mov(1, 0, CORE_BASE + t.d + 192); // tokens96..127
    a.mov(1, 6, CORE_BASE + t.c + t.c_bytes() as u32 - 64);
    a.mov(1, 4, CORE_BASE + t.out + t.out_bytes() as u32 - 64);
    a.mov(0, 5, 4);
    let outer = a.label();
    let mb = a.label();
    let nb = a.label();
    a.bind(outer);
    emit_block(a, &rows(t));
    a.mov(1, 5, CORE_BASE + t.dpat() + 14 * 64);
    a.mov(0, 13, 4);
    a.bind(mb);
    a.mov(1, 1, CORE_BASE + t.spat() + (t.nb() as u32 - 1) * 64);
    a.mov(1, 7, CORE_BASE + t.sepat() + (t.nb() as u32 - 1) * 64);
    a.mov(0, 11, t.nb() as u32);
    a.bind(nb);
    emit_loop(a, group(t), 2, R_LOOP);
    let mut advance = List::default();
    advance.push(padd(Slot::Lda, 5, 1));
    advance.push(padd(Slot::Lda, 1, 2));
    advance.push(padd(Slot::Ldb, 7, 2));
    emit_block(a, &advance.finish());
    a.add(11, -1);
    a.nop(7);
    a.branch(nb, Some((11, true)));
    a.delay();
    let mut advance = List::default();
    advance.push(padd(Slot::Lda, 5, 3));
    emit_block(a, &advance.finish());
    a.add(13, -1);
    a.nop(7);
    a.branch(mb, Some((13, true)));
    a.delay();
    a.add(5, -1);
    a.nop(7);
    a.branch(outer, Some((5, true)));
    a.delay();
    a.set_cr("crSRSMode", 0);
}

/// Issue cycles of the arithmetic group alone (not DMA/outer-loop stalls).
pub(crate) fn group_cycles() -> usize {
    group(WIDE).len()
}

/// Phase-sum issue model, excluding DMA and bank/lock stalls. Control-loop
/// padding is conservative; full-array tests report the executed simulator ticks.
pub(crate) fn wave_issue_cycles() -> u64 {
    wave_issue_cycles_for(WIDE)
}

pub(crate) fn wave_issue_cycles_for(t: Tile) -> u64 {
    let mut zero = List::default();
    for _ in 0..16 {
        zero.push(st(11, 7, 1, MEM_I));
    }
    (t.c_bytes() / 1024) as u64 * (zero.finish().len() as u64 + 8)
        + columns(t).len() as u64
        + 4 * rows(t).len() as u64
        + (4 * t.nb() as u64 * 8) * (group(t).len() as u64 + 8)
        + (t.c_bytes() / 256) as u64 * 32
        + 16 * 32
        + 4 * 32
        + 256
}

/// Standalone diagnostic: I32[32] at C=0; output BF16[32] at OUT. The
/// returned offsets are Dm, De, Sm, Se (32 lanes each, no pattern expansion).
pub(crate) fn debug_program() -> (Vec<u8>, [u32; 4]) {
    let mut a = Assembler::default();
    a.mov(0, 0, u32::MAX);
    initialize(&mut a);
    a.set_cr("crSRSMode", 1);
    a.mov(2, 0, (-128i32) as u32);
    a.mov(1, 5, CORE_BASE + WIDE.dpat());
    a.mov(1, 1, CORE_BASE + WIDE.spat());
    a.mov(1, 7, CORE_BASE + WIDE.sepat());
    a.mov(1, 6, CORE_BASE + 64);
    a.mov(1, 4, CORE_BASE + OUT);
    emit_block(&mut a, &group(WIDE));
    a.emit(&[(encoding("DONE").slot, encoding("DONE").value)]);
    (
        a.finish().finish(),
        [DPAT, DPAT + 64, WIDE.spat(), WIDE.sepat()],
    )
}
