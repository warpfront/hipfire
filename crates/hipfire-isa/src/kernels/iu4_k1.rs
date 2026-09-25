//! K1's symmetric MQ4V2 fold, with physical banks fixed before assembly.
//!
//! The eight (column-block, row-group) accumulators are contiguous V8 ranges.
//! Each pair follows the pinned K1 arithmetic DAG: `float(C)-12582912`,
//! `RN(scale*d)`, `RN(acc + scale*d*(float(C)-12582912))`.
use crate::{Builder, KernargLayout, V, reg::{Vb, Vp}, vopd::{Src0, VopdF32, VopdF32Op}};

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

/// The register ownership of a K128 fold. `magic` is eight invariant vector
/// copies, not a loop-local SGPR broadcast. `d` is one per token block.
/// `sc[rg]` holds the eight row scales of row group `rg` as two quads (rows
/// 0..3 and 4..7); the quads may be separate ranges so they can alias
/// fragment registers whose WMMAs have already issued.
#[derive(Clone, Copy)]
pub struct FoldRegisters {
    pub cacc: [V<8>; 8],
    pub acc: [V<8>; 8],
    pub magic: V<8>,
    pub sc: [[V<4>; 2]; 2],
    pub t: V<8>,
    pub d: [V<1>; 4],
}

type Pair = (V<1>, V<1>);
fn quad_pair(q: V<4>, j: u8) -> Pair { (V(q.base() + j), V(q.base() + j + 1)) }

/// `float(C) - magic`, in place: exact because |dot| < 2^22.
fn subrev<const B0: u8, const B1: u8>(b: &mut Builder, c: Pair, m: Pair) -> Result<(), String> where Vb<B0>: crate::reg::DistinctBanks<B1> {
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<0>::checked(c.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(m.0.base())?), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<1>::checked(c.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(m.1.base())?), src1: Vb::<B1>::checked(c.1.base())? },
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
/// `acc = fma(t, float(C) - magic, acc)`.
fn fmac<const B0: u8, const B1: u8>(b: &mut Builder, a: Pair, t: Pair, c: Pair) -> Result<(), String> where Vb<B0>: crate::reg::DistinctBanks<B1> {
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<0>::checked(a.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(t.0.base())?), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<1>::checked(a.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(t.1.base())?), src1: Vb::<B1>::checked(c.1.base())? },
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
        let (c, m) = (regs.cacc[k].pair(j)?, regs.magic.pair(j)?);
        if j & 2 == 0 { subrev::<0, 1>(b, c, m)? } else { subrev::<2, 3>(b, c, m)? }
    }
    Ok(())
}

/// Stage 2 of accumulator `k = nb * 2 + rg`: four `mul` packets forming
/// `RN(sc * d)`, then four `fmac` packets, so each product is four packets
/// old when consumed. Requires stage 1 of `k` and the scale rows of `rg`.
pub fn emit_fold_scale(b: &mut Builder, regs: FoldRegisters, k: usize) -> Result<(), String> {
    gfx12(b)?;
    let (nb, rg) = (k / 2, k % 2);
    for j in (0..8).step_by(2) {
        let (t, sc) = (regs.t.pair(j)?, quad_pair(regs.sc[rg][usize::from(j / 4)], j % 4));
        if j & 2 == 0 { mul::<0, 1>(b, t, sc, regs.d[nb])? } else { mul::<2, 3>(b, t, sc, regs.d[nb])? }
    }
    for j in (0..8).step_by(2) {
        let (a, t, c) = (regs.acc[k].pair(j)?, regs.t.pair(j)?, regs.cacc[k].pair(j)?);
        if j & 2 == 0 { fmac::<0, 1>(b, a, t, c)? } else { fmac::<2, 3>(b, a, t, c)? }
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
