// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::{ds_load_insn, ActScale, Builder, Instruction, Spec};
use crate::kernels::common::{op, v};
use crate::ledger::Counter;

/// Row-scale ratio staging for one row group (8 ratios, rows 8i..8i+7):
/// two b128 loads per group, alternating between the staging alias
/// v168..v175 and v[183:186] + v[188:191]. The second set holds nothing in a
/// Row K region: v184 carries the next block's ratio only from its fetch
/// before slab 0 to its publish before this fold, and the others are
/// epilogue temporaries.
const RATIO_SETS: [[u8; 8]; 2] = [
    [168, 169, 170, 171, 172, 173, 174, 175],
    [183, 184, 185, 186, 188, 189, 190, 191],
];

/// The two `ds_load_b128`s of row group `i`'s eight ratios (plane `plane` of
/// the `R` ring: slot `2 + plane`).
pub(super) fn row_group_load_insns(plane: usize, i: usize) -> Result<[Instruction; 2], String> {
    let regs = RATIO_SETS[i % 2];
    let offset = (16384 + plane * 1024 + i * 64) as u32;
    Ok([
        ds_load_insn(regs[0], 4, 180, offset)?,
        ds_load_insn(regs[4], 4, 180, offset + 16)?,
    ])
}

/// The 16 dual-mul packets of Row group `i`, scaling its four token chains by
/// the eight ratios the group's loads staged.
pub(super) fn row_group_scale(b: &mut Builder, i: usize) -> Result<(), String> {
    let regs = RATIO_SETS[i % 2];
    for e in (0..8).step_by(2) {
        let (r0, r1) = (regs[e], regs[e + 1]);
        for tt in 0..4 {
            let (a, z) = (8 * (i * 4 + tt) + e, 8 * (i * 4 + tt) + e + 1);
            op(
                b,
                format!("v_dual_mul_f32 v{a}, v{a}, v{r0} :: v_dual_mul_f32 v{z}, v{z}, v{r1}"),
                &[v(a as u8), v(z as u8)],
                &[v(a as u8), v(z as u8), v(r0), v(r1)],
            )?;
        }
    }
    Ok(())
}

/// A half's C is already complete when this runs. Published ratios for the
/// *next* half scale the accumulators; each weight ratio scales four token
/// chains. The first half omits the fold because C starts at literal zero.
pub(super) fn emit(b: &mut Builder, spec: Spec, plane: usize) -> Result<(), String> {
    if spec.act_scale == ActScale::K128 {
        return emit_k128(b, plane);
    }
    // Row: the eight ratios of row group i arrive in two b128 loads, one
    // group ahead of their 16 packets. The A-publish stores still read
    // v168..v175, so every store has drained before the first load; each
    // group then waits only for its own two loads.
    let group = |b: &mut Builder, i: usize| -> Result<(), String> {
        for insn in row_group_load_insns(plane, i)? {
            b.ds_load(2 + plane, insn)?;
        }
        Ok(())
    };
    b.wait(Counter::Ds, 0)?;
    group(b, 0)?;
    for i in 0..4 {
        if i + 1 < 4 {
            group(b, i + 1)?;
        }
        b.wait(Counter::Ds, if i + 1 < 4 { 2 } else { 0 })?;
        row_group_scale(b, i)?;
    }
    Ok(())
}

/// K128: the four per-token D ratio loads (plane `plane` of the `D` ring:
/// slot `4 + plane`).
pub(super) fn k128_token_load_insns(plane: usize) -> Result<Vec<Instruction>, String> {
    (0..4u8)
        .map(|tt| {
            ds_load_insn(
                172 + tt,
                1,
                181,
                (18432 + plane * 512 + usize::from(tt) * 64) as u32,
            )
        })
        .collect()
}

/// K128: the two row-ratio loads of pair `(i, e)` (`e` even, `0..8`) from the
/// `R` ring.
pub(super) fn k128_pair_load_insns(
    plane: usize,
    i: usize,
    e: usize,
) -> Result<[Instruction; 2], String> {
    let offset = 16384 + plane * 1024 + i * 64 + e * 4;
    Ok([
        ds_load_insn(168, 1, 180, offset as u32)?,
        ds_load_insn(169, 1, 180, (offset + 4) as u32)?,
    ])
}

/// K128: the four token chains of pair `(i, e)`: the row ratios times the
/// per-token D ratio, then the accumulators times that product.
pub(super) fn k128_pair_scale(b: &mut Builder, i: usize, e: usize) -> Result<(), String> {
    for tt in 0..4 {
        let (a, z) = (8 * (i * 4 + tt) + e, 8 * (i * 4 + tt) + e + 1);
        op(
            b,
            format!(
                "v_dual_mul_f32 v170, v168, v{} :: v_dual_mul_f32 v171, v169, v{}",
                172 + tt,
                172 + tt
            ),
            &[v(170), v(171)],
            &[v(168), v(169), v((172 + tt) as u8)],
        )?;
        op(
            b,
            format!("v_dual_mul_f32 v{a}, v{a}, v170 :: v_dual_mul_f32 v{z}, v{z}, v171"),
            &[v(a as u8), v(z as u8)],
            &[v(a as u8), v(z as u8), v(170), v(171)],
        )?;
    }
    Ok(())
}

/// K128: published ratios are read in 16 row pairs and multiplied by the
/// per-token D ratio before they scale the accumulators.
fn emit_k128(b: &mut Builder, plane: usize) -> Result<(), String> {
    for insn in k128_token_load_insns(plane)? {
        b.ds_load(4 + plane, insn)?;
    }
    for i in 0..4 {
        for e in (0..8).step_by(2) {
            for insn in k128_pair_load_insns(plane, i, e)? {
                b.ds_load(2 + plane, insn)?;
            }
            k128_pair_scale(b, i, e)?;
        }
    }
    Ok(())
}
