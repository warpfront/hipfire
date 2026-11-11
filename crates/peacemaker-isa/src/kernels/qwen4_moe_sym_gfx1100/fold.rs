// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 one-epoch compute: scale broadcast, unpack, the exact integer
//! WMMA chain and the exact ordered fold, all through the shared
//! `iu4_fold::{wmma_step, fold_pass}` (no copy of either).
//!
//! Per output `(row r, token p)` and epoch `e` (visiting order is
//! [`epoch_order`]; ascending is the contract order):
//!
//! ```text
//! D_e = sum_{k<128} w_i4[r,128e+k] * x_i4[p,128e+k]     exact int32, eight K16 steps,
//!                                                       seeded with MAGIC 0x4b400000
//! S   = fma(RN(d_act[p,e] * sc[r,e]), float(D_e), S)    S starts at +0
//! ```
//! `|D_e| <= 8192` (MQ4), 7168 (W3), 3072 (W2): all below 2^22, so the magic
//! conversion `bits(D + magic) - 1.5*2^23` is exact.
//!
//! Row scales: lane `(hi, lr)` loaded the f16 scale of row `(lr & 14) | hi`
//! (`HOFF`). Accumulator element `j` of lane `(hi, lr)` is output row
//! `2j + hi` of token `lr`; `ds_swizzle_b32 BROADCAST,16,2j` copies lane `2j`
//! of the same 16-lane half (row `2j | hi`) to every lane of that half, so
//! `SCF[j]` is the scale of row `2j + hi` for both halves. No tree of F32
//! folds, no contraction across the scale multiply: `fold_pass` is the
//! reviewed dependency order.
use super::*;

/// f16 -> f32 scale and the eight `ds_swizzle` row broadcasts per tensor, from the
/// f16 scale dword in `src(t)` into `SCF` (through `HF`).
pub(super) fn broadcast(b: &mut Builder, g: &Gen, src: impl Fn(u8) -> u8) -> Result<(), String> {
    for t in 0..g.tn() {
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{}.l", HF + t, src(t)),
            &[v(HF + t)],
            &[v(src(t))],
        )?;
        for j in 0..8u8 {
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{})",
                SCF + 8 * t + j,
                HF + t,
                swizzle_lane(u32::from(j))
            );
            b.ds_crosslane(
                Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    Ok(())
}

/// Scale broadcast (when `bcast`; else it was issued under the previous epoch's fold), unpack
/// and the exact integer WMMA chain of the epoch whose operands are in the registers.
pub(super) fn chain(b: &mut Builder, g: &Gen, bcast: bool) -> Result<(), String> {
    let a = g.spec.arch;
    let tn = g.tn();
    if bcast {
        broadcast(b, g, |t| g.hs(t))?;
    }
    unpack::unpack_epoch(b, g)?;
    for i in 0..8u8 {
        for t in 0..tn {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(g.a(t) + 2 * i),
                V::<2>(g.x() + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    Ok(())
}

/// The ordered F32 fold of the chain's accumulators.
pub(super) fn fold(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.tn() {
        iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, g.d(), TPROD)?;
    }
    Ok(())
}
