//! The symmetric MQ4V2 x `block_i4_128` IU4 fold shared by the builder GEMMs.
//!
//! Per output and ascending K128 epoch: `C` is the exact int32
//! `v_wmma_i32_16x16x16_iu4` (gfx11, eight K16 steps) or
//! `v_wmma_i32_16x16x32_iu4` (gfx12, four K32 steps) chain, signed A (weight
//! nibbles rebiased with XOR [`REBIAS`]) and signed X, seeded with
//! [`MAGIC`]; the fold is `sum = fma(RN(d * sc), float(C), sum)` with
//! `float(C) = bits(C + magic) - 1.5*2^23` ([`MAGIC_NEG`]), exact for
//! `|C| < 2^22`.
use crate::{Arch, Builder, V, insn::Wmma, vopd::{Operand, VopdF32, VopdOp}};

/// int32 bit pattern of 12582912.0f: the WMMA chain seed.
pub const MAGIC: u32 = 0x4b40_0000;
/// `-12582912.0f`: `float(C) = bits(C + magic) - 1.5*2^23` exactly.
pub const MAGIC_NEG: u32 = 0xcb40_0000;
/// XOR turning an unsigned symmetric nibble `u` (`zp == -8*sc`) into the
/// two's-complement `u - 8`, eight nibbles per dword.
pub const REBIAS: u32 = 0x8888_8888;
/// QT44 (MQ4G256V2) group: `[f16 sc0, zp0, sc1, zp1][128 nibble bytes]`.
pub const GROUP_BYTES: u32 = 136;
/// `block_i4_128`: `[f32 d, i32 s, 64 nibble bytes]`.
pub const XBLK_BYTES: u32 = 72;

/// One WMMA step of a magic-seeded chain: the first step of an epoch reads
/// the seed registers, later steps accumulate in place.
pub fn wmma_step(b: &mut Builder, arch: Arch, dst: V<8>, a: V<2>, x: V<2>, first: bool, magic: V<8>) -> Result<(), String> {
    b.push(Wmma::iu4(arch, dst, a, x, Some(if first { magic } else { dst })))
}

/// Fold one 8-element accumulator fragment as 12 VOPD packets:
/// `t_j = d * sc_j` paired with `C_{j^1} += MAGIC_NEG`, then
/// `sum_j = fma(t_j, C_j, sum_j)` pairs. Element j's product sits at
/// `t + (j^2)`: sum and C of element j share VGPR bank j%4, so an fmac half
/// reads that bank twice and bank (j^2)%4 once instead of bank j%4 three
/// times. `c`, `sc` and `t` must share a bank alignment (`c % 4 == sc % 4`).
pub fn fold_pass(b: &mut Builder, c: u8, sum: u8, sc: u8, d: u8, t: u8) -> Result<(), String> {
    let ts = |j: u8| t + (j ^ 2);
    for j in 0..8u8 {
        let mul = VopdOp { op: VopdF32::Mul, dst: ts(j), src0: Operand::V(d), src1: sc + j };
        let cf = c + (j ^ 1);
        let add = VopdOp { op: VopdF32::Add, dst: cf, src0: Operand::Lit(MAGIC_NEG), src1: cf };
        b.vopd(mul, add)?;
    }
    for j in (0..8u8).step_by(2) {
        let x = VopdOp { op: VopdF32::Fmac, dst: sum + j, src0: Operand::V(ts(j)), src1: c + j };
        let y = VopdOp { op: VopdF32::Fmac, dst: sum + j + 1, src0: Operand::V(ts(j + 1)), src1: c + j + 1 };
        b.vopd(x, y)?;
    }
    Ok(())
}
