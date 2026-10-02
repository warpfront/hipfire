//! K1's symmetric MQ4V2 fold, with physical banks fixed before assembly.
//!
//! The eight (column-block, row-group) accumulators are contiguous V8 ranges.
//! Each pair follows the pinned K1 arithmetic DAG: `float(C)-12582912`,
//! `RN(scale*d)`, `RN(acc + scale*d*(float(C)-12582912))`.
use crate::{Builder, KernargLayout, V, kernels::iu4_fold::MAGIC, reg::{Vb, Vp}, vopd::{Src0, VopdF32, VopdF32Op}};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant { FullSet, FullAdd, GateUpSilu }

impl Variant {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::FullSet => "gemm_mq4g256v2_residual_mmq_iu4_full_set_v3",
            Self::FullAdd => "gemm_mq4g256v2_residual_mmq_iu4_full_add_v3",
            Self::GateUpSilu => "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3",
        }
    }

    /// HIP's explicit arguments occupy bytes 0..40 (0..44 for gate/up).
    /// ROCm fills the hidden fields when launching through hipModuleLaunchKernel.
    pub fn kernargs(self) -> KernargLayout {
        let gate = self == Self::GateUpSilu;
        let mut args = KernargLayout::new(if gate { 304 } else { 296 })
            .pointer(if gate { "G" } else { "A" }, 0)
            .pointer(if gate { "U" } else { "Xq" }, 8)
            .pointer(if gate { "Xq" } else { "Y" }, 16);
        if gate { args = args.pointer("H", 24); }
        let base = if gate { 32 } else { 24 };
        for (i, name) in ["M", "K", "N"].iter().enumerate() {
            args = args.hidden(name, base + 4 * i as u32, 4, "by_value");
        }
        if !gate { args = args.hidden("unused", 36, 4, "by_value"); }
        let hidden = if gate { 48 } else { 40 };
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_block_count_{name}"), hidden + 4 * i as u32, 4,
                &format!("hidden_block_count_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_group_size_{name}"), hidden + 12 + 2 * i as u32, 2,
                &format!("hidden_group_size_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_remainder_{name}"), hidden + 18 + 2 * i as u32, 2,
                &format!("hidden_remainder_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_global_offset_{name}"), hidden + 40 + 8 * i as u32, 8,
                &format!("hidden_global_offset_{name}"));
        }
        args.hidden("hidden_grid_dims", hidden + 64, 2, "hidden_grid_dims")
            .hidden("hidden_dynamic_lds_size", hidden + 120, 4, "hidden_dynamic_lds_size")
    }
}

/// The register ownership of a K128 fold. The magic operand of stage 1 is
/// the shared VOPD literal `MAGIC` (the same 12582912.0f the WMMA seed
/// holds), so the fold reads no magic register. `d` is one per token block.
/// `sc[rg]` holds the eight row scales of row group `rg` as two quads (rows
/// 0..3 and 4..7); the quads may be separate ranges so they can alias
/// fragment registers whose WMMAs have already issued. `t` (4-aligned)
/// holds the eight products with element `j` at `t + (j ^ 2)`: C and acc of
/// element `j` share VGPR bank `j % 4`, so each fmac half reads its product
/// from bank `(j + 2) % 4` instead of a third read of that bank.
#[derive(Clone, Copy)]
pub struct FoldRegisters {
    pub cacc: [V<8>; 8],
    pub acc: [V<8>; 8],
    pub sc: [[V<4>; 2]; 2],
    pub t: V<8>,
    pub d: [V<1>; 4],
}

type Pair = (V<1>, V<1>);
fn quad_pair(q: V<4>, j: u8) -> Pair { (V(q.base() + j), V(q.base() + j + 1)) }
/// Product register of element `j` (see `FoldRegisters::t`).
fn t_at(t: V<8>, j: u8) -> V<1> { V(t.base() + (j ^ 2)) }
fn t_pair(t: V<8>, j: u8) -> Pair { (t_at(t, j), t_at(t, j + 1)) }

/// `float(C) - magic`, in place: exact because |dot| < 2^22. The magic is
/// the packet's shared literal, so each half reads one VGPR (`C`): fold
/// packets issue slower the more VGPR reads land in one bank.
fn subrev<const B0: u8, const B1: u8>(b: &mut Builder, c: Pair) -> Result<(), String> where Vb<B0>: crate::reg::DistinctBanks<B1> {
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<0>::checked(c.0.base())?,
            src0: Src0::<B0>::Lit(MAGIC), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<1>::checked(c.1.base())?,
            src0: Src0::<B1>::Lit(MAGIC), src1: Vb::<B1>::checked(c.1.base())? },
    )
}
/// `t = RN(sc * d)`.
fn mul<const B0: u8, const B1: u8>(b: &mut Builder, t: Pair, sc: Pair, d: V<1>) -> Result<(), String> where Vb<B0>: crate::reg::DistinctBanks<B1> {
    b.vopd_ff_shared_src1(
        VopdF32Op { op: VopdF32::Mul, dst: Vp::<0>::checked(t.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(sc.0.base())?), src1: d },
        VopdF32Op { op: VopdF32::Mul, dst: Vp::<1>::checked(t.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(sc.1.base())?), src1: d },
    )
}
/// `acc = fma(t, float(C) - magic, acc)`; `A*` are the product banks.
fn fmac<const A0: u8, const A1: u8, const B0: u8, const B1: u8>(b: &mut Builder, a: Pair, t: Pair, c: Pair) -> Result<(), String>
    where Vb<A0>: crate::reg::DistinctBanks<A1>, Vb<B0>: crate::reg::DistinctBanks<B1> {
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<0>::checked(a.0.base())?,
            src0: Src0::V(Vb::<A0>::checked(t.0.base())?), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<1>::checked(a.1.base())?,
            src0: Src0::V(Vb::<A1>::checked(t.1.base())?), src1: Vb::<B1>::checked(c.1.base())? },
    )
}
/// X: `acc = fma(t, float(C) - magic, acc)` of one element; Y: stage 1
/// (`C2 - magic`, literal) of an element of opposite parity of another
/// accumulator. The Y half adds one VGPR read, in a bank X does not read.
fn fmac_subrev<const DX: u8, const DY: u8, const AX: u8, const AY: u8, const BX: u8, const BY: u8>(b: &mut Builder, a: V<1>, t: V<1>, c: V<1>, c2: V<1>) -> Result<(), String>
    where Vp<DX>: crate::reg::OppositeParity<DY>, Vb<AX>: crate::reg::DistinctBanks<AY>, Vb<BX>: crate::reg::DistinctBanks<BY> {
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<DX>::checked(a.base())?, src0: Src0::V(Vb::<AX>::checked(t.base())?), src1: Vb::<BX>::checked(c.base())? },
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<DY>::checked(c2.base())?, src0: Src0::<AY>::Lit(MAGIC), src1: Vb::<BY>::checked(c2.base())? },
    )
}

fn gfx12(b: &Builder) -> Result<(), String> {
    if b.spec.arch.gfx12() { Ok(()) } else { Err("the K1 shared-src1 fold is gfx12-only".into()) }
}

/// Stage 1 of accumulator `k = nb * 2 + rg`: four `subrev` packets turning
/// the WMMA result into `float(C) - magic`. Needs only the accumulator, so it
/// may issue between later WMMAs as soon as accumulator `k` is complete.
pub fn emit_fold_unbias(b: &mut Builder, regs: FoldRegisters, k: usize) -> Result<(), String> {
    gfx12(b)?;
    for j in (0..8).step_by(2) {
        let c = regs.cacc[k].pair(j)?;
        if j & 2 == 0 { subrev::<0, 1>(b, c)? } else { subrev::<2, 3>(b, c)? }
    }
    Ok(())
}

fn emit_products(b: &mut Builder, regs: FoldRegisters, k: usize) -> Result<(), String> {
    let (nb, rg) = (k / 2, k % 2);
    for j in (0..8).step_by(2) {
        let (t, sc) = (t_pair(regs.t, j), quad_pair(regs.sc[rg][usize::from(j / 4)], j % 4));
        if j & 2 == 0 { mul::<0, 1>(b, t, sc, regs.d[nb])? } else { mul::<2, 3>(b, t, sc, regs.d[nb])? }
    }
    Ok(())
}

/// Stage 2 of accumulator `k = nb * 2 + rg`: four `mul` packets forming
/// `RN(sc * d)`, then four `fmac` packets, so each product is four packets
/// old when consumed. Requires stage 1 of `k` and the scale rows of `rg`.
pub fn emit_fold_scale(b: &mut Builder, regs: FoldRegisters, k: usize) -> Result<(), String> {
    gfx12(b)?;
    emit_products(b, regs, k)?;
    for j in (0..8).step_by(2) {
        let (a, t, c) = (regs.acc[k].pair(j)?, t_pair(regs.t, j), regs.cacc[k].pair(j)?);
        if j & 2 == 0 { fmac::<2, 3, 0, 1>(b, a, t, c)? } else { fmac::<0, 1, 2, 3>(b, a, t, c)? }
    }
    Ok(())
}

/// Stage 2 of `k` carrying stage 1 of `k2`: four `mul` packets, then eight
/// packets pairing element `j`'s fmac with element `j ^ 1`'s literal subrev
/// of `k2` (same 16 ops as `k`'s four fmac and `k2`'s four subrev packets).
/// Requires stage 1 of `k`; stage 2 of `k2` must come after.
pub fn emit_fold_scale_paired(b: &mut Builder, regs: FoldRegisters, k: usize, k2: usize) -> Result<(), String> {
    gfx12(b)?;
    emit_products(b, regs, k)?;
    for j in 0..8u8 {
        let (a, t, c, c2) = (V(regs.acc[k].base() + j), t_at(regs.t, j), V(regs.cacc[k].base() + j), V(regs.cacc[k2].base() + (j ^ 1)));
        match j % 4 {
            0 => fmac_subrev::<0, 1, 2, 3, 0, 1>(b, a, t, c, c2)?,
            1 => fmac_subrev::<1, 0, 3, 2, 1, 0>(b, a, t, c, c2)?,
            2 => fmac_subrev::<0, 1, 0, 1, 2, 3>(b, a, t, c, c2)?,
            _ => fmac_subrev::<1, 0, 1, 0, 3, 2>(b, a, t, c, c2)?,
        }
    }
    Ok(())
}

/// The 48 packets of one row group, both stages per column block. Values and
/// the per-element DAG are the pinned K1 fold; only the issue order of
/// independent pairs is chosen.
pub fn emit_fold_rows(b: &mut Builder, regs: FoldRegisters, rg: usize) -> Result<(), String> {
    for nb in 0..4 {
        emit_fold_unbias(b, regs, nb * 2 + rg)?;
        emit_fold_scale(b, regs, nb * 2 + rg)?;
    }
    Ok(())
}

/// The full K128 fold: 96 packets (48 per row group). A per-256 fold is the
/// same 96 packets issued once per group, i.e. 48 per K128.
pub fn emit_fold(b: &mut Builder, regs: FoldRegisters) -> Result<(), String> {
    emit_fold_rows(b, regs, 0)?;
    emit_fold_rows(b, regs, 1)
}
