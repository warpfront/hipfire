// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The per-K128 fold, as VOPD packets. Per accumulator fragment `k = nb * 2 + rg` (8 rows `j`):
//!
//! ```text
//! c_j  = as_float(cacc_j) + msn          msn = -(12582912.0 + 32 * s)   (exact: Cq - 32 s)
//! t1_j = d * sc_j
//! acc_j = fma(t1_j, c_j, acc_j)
//! t2_j = d * z'_j
//! acc_j = fma(t2_j, sP, acc_j)           sP = float(s)
//! ```
//! 20 packets: 8 (`mul` + `add`), 4 (`fmac` pairs), 4 (`mul` pairs), 4 (`fmac` pairs).
//! Register banks and parities are the VOPD-legal ones (`vopd::validate_pair` rejects a bad
//! assignment at build time): the stage-1 add of packet `j` targets element `j ^ 1`, the products
//! sit at `t + (j ^ 2)`, `sc` / `z'` live in aligned planar octets so source-1 banks never
//! collide with the accumulators'.
use super::Gen;
use crate::vopd::{Operand, VopdF32, VopdOp};
use crate::Builder;

/// `sc` octet base within the fold registers (F0 start), `z'` octet right after, then the products.
pub(crate) const SC: u8 = 128;
pub(crate) const ZP: u8 = 136;
pub(crate) const T: u8 = 144;

/// Accumulator order of a K16 step's 8 WMMAs: row group 0 over the four token blocks, then row
/// group 1 (`k = nb * 2 + rg`).
pub(crate) const BUNDLE_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// `msn[nb] = -(12582912.0 + 32 * s[nb])` into `g.msn` (exact: both terms are integers below 2^24).
pub(crate) fn emit_msn(b: &mut Builder, g: &Gen, nb: usize) -> Result<(), String> {
    use crate::kernels::common::{lit, op, v};
    if nb >= 4 {
        return Err(format!("emit_msn: token block {nb} out of range (0..4)"));
    }
    let sp = g.tok + 2 * nb as u8 + 1;
    op(
        b,
        format!("v_mul_f32_e32 v{}, {}, v{sp}", g.msn, lit(0xc200_0000)),
        &[v(g.msn)],
        &[v(sp)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{0}, {1}, v{0}", g.msn, lit(0xcb40_0000)),
        &[v(g.msn)],
        &[v(g.msn)],
    )
}

/// One fragment's 20 packets: the four stages in order. `S` / `Z` / `T` octets are fixed (see the
/// constants), the accumulators `cacc[k]`, `acc[k]` come from `g`.
pub(crate) fn fragment(b: &mut Builder, g: &Gen, k: usize, sc: u8, zp: u8) -> Result<(), String> {
    for stage in 0..STAGES {
        fold_stage(b, g, k, stage, sc, zp)?;
    }
    Ok(())
}

/// The stages of a fragment's fold: `0` the eight (`mul`, `add`) packets, `1` the four `fmac`
/// pairs, `2` the four `mul` pairs, `3` the four closing `fmac` pairs.
pub(crate) const STAGES: usize = 4;

/// One stage (see [`STAGES`]) of fragment `k`'s fold: 8, 4, 4 and 4 packets.
pub(crate) fn fold_stage(
    b: &mut Builder,
    g: &Gen,
    k: usize,
    stage: usize,
    sc: u8,
    zp: u8,
) -> Result<(), String> {
    if k >= 8 || stage >= STAGES {
        return Err(format!(
            "fold_stage: fragment {k} / stage {stage} out of range (8 / {STAGES})"
        ));
    }
    let nb = k / 2;
    let (dd, sp) = (g.tok + 2 * nb as u8, g.tok + 2 * nb as u8 + 1);
    let c = g.cacc + 8 * k as u8;
    let acc = g.acc + 8 * k as u8;
    let ts = |j: u8| T + (j ^ 2);
    match stage {
        0 => {
            for j in 0..8u8 {
                let mul = VopdOp {
                    op: VopdF32::Mul,
                    dst: ts(j),
                    src0: Operand::V(dd),
                    src1: sc + j,
                };
                let cf = c + (j ^ 1);
                let add = VopdOp {
                    op: VopdF32::Add,
                    dst: cf,
                    src0: Operand::V(g.msn),
                    src1: cf,
                };
                b.vopd(mul, add)?;
            }
        }
        1 => {
            for j in (0..8u8).step_by(2) {
                b.vopd(
                    VopdOp {
                        op: VopdF32::Fmac,
                        dst: acc + j,
                        src0: Operand::V(ts(j)),
                        src1: c + j,
                    },
                    VopdOp {
                        op: VopdF32::Fmac,
                        dst: acc + j + 1,
                        src0: Operand::V(ts(j + 1)),
                        src1: c + j + 1,
                    },
                )?;
            }
        }
        2 => {
            for j in (0..8u8).step_by(2) {
                b.vopd(
                    VopdOp {
                        op: VopdF32::Mul,
                        dst: ts(j),
                        src0: Operand::V(zp + j),
                        src1: dd,
                    },
                    VopdOp {
                        op: VopdF32::Mul,
                        dst: ts(j + 1),
                        src0: Operand::V(zp + j + 1),
                        src1: dd,
                    },
                )?;
            }
        }
        _ => {
            for j in (0..8u8).step_by(2) {
                b.vopd(
                    VopdOp {
                        op: VopdF32::Fmac,
                        dst: acc + j,
                        src0: Operand::V(ts(j)),
                        src1: sp,
                    },
                    VopdOp {
                        op: VopdF32::Fmac,
                        dst: acc + j + 1,
                        src0: Operand::V(ts(j + 1)),
                        src1: sp,
                    },
                )?;
            }
        }
    }
    Ok(())
}
