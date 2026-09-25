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

/// The register ownership of a single K128 fold. `magic` is eight invariant
/// vector copies, not a loop-local SGPR broadcast. `d` is one per token block.
#[derive(Clone, Copy)]
pub struct FoldRegisters {
    pub cacc: [V<8>; 8],
    pub acc: [V<8>; 8],
    pub magic: V<8>,
    pub sc_row: V<8>,
    pub t: V<8>,
    pub d: [V<1>; 4],
}

fn pair<const B0: u8, const B1: u8>(
    b: &mut Builder,
    c: (V<1>, V<1>),
    a: (V<1>, V<1>),
    m: (V<1>, V<1>),
    sc: (V<1>, V<1>),
    t: (V<1>, V<1>),
    d: V<1>,
) -> Result<(), String>
where
    Vb<B0>: crate::reg::DistinctBanks<B1>,
{
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<0>::checked(c.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(m.0.base())?), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Subrev, dst: Vp::<1>::checked(c.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(m.1.base())?), src1: Vb::<B1>::checked(c.1.base())? },
    )?;
    b.vopd_ff_shared_src1(
        VopdF32Op { op: VopdF32::Mul, dst: Vp::<0>::checked(t.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(sc.0.base())?), src1: d },
        VopdF32Op { op: VopdF32::Mul, dst: Vp::<1>::checked(t.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(sc.1.base())?), src1: d },
    )?;
    b.vopd_ff(
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<0>::checked(a.0.base())?,
            src0: Src0::V(Vb::<B0>::checked(t.0.base())?), src1: Vb::<B0>::checked(c.0.base())? },
        VopdF32Op { op: VopdF32::Fmac, dst: Vp::<1>::checked(a.1.base())?,
            src0: Src0::V(Vb::<B1>::checked(t.1.base())?), src1: Vb::<B1>::checked(c.1.base())? },
    )
}

/// Emit precisely 32 independent j-pairs, three legal VOPD packets per pair.
/// `sc_row` must have been loaded from the current SZ half-plane and each
/// `d[nb]` from the current DS half-plane before this call.
pub fn emit_fold(b: &mut Builder, regs: FoldRegisters) -> Result<(), String> {
    if !b.spec.arch.gfx12() {
        return Err("the K1 shared-src1 fold is gfx12-only".into());
    }
    for nb in 0..4 {
        for rg in 0..2 {
            let k = nb * 2 + rg;
            for j in (0..8).step_by(2) {
                let c = regs.cacc[k].pair(j)?;
                let a = regs.acc[k].pair(j)?;
                let m = regs.magic.pair(j)?;
                let sc = regs.sc_row.pair(j)?;
                let t = regs.t.pair(j)?;
                if j & 2 == 0 {
                    pair::<0, 1>(b, c, a, m, sc, t, regs.d[nb])?;
                } else {
                    pair::<2, 3>(b, c, a, m, sc, t, regs.d[nb])?;
                }
            }
        }
    }
    Ok(())
}
