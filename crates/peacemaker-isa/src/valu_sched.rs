// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12 VALU window scheduling and VOPD dual-issue formation (opt-in:
//! `Builder::enable_valu_schedule`).
//!
//! The Builder buffers each maximal run of plain VALU instructions (no memory
//! access, only VGPR results, no implicit VCC/EXEC/SCC/M0 operands, no
//! cross-lane DPP/permlane or partial-register writes) and releases it at the
//! next non-VALU instruction, label, wait, barrier or region boundary. The run
//! is list-scheduled over its register dependence DAG (RAW, WAR and WAW on
//! every declared operand), so only independent instructions change order:
//! results are identical. While scheduling it
//! - prefers instructions whose operands need no memory wait now (a wait the
//!   ledger would insert is postponed behind independent work),
//! - prefers the instruction whose producers issued longest ago (hides VALU
//!   dependency latency instead of stalling on `s_delay_alu`), then the
//!   longest remaining dependence chain,
//! - packs two ready (hence independent) f32 add/sub/mul/fmac instructions into
//!   one VOPD packet when the RDNA4 pairing rules hold (`vopd::validate_pair`:
//!   destination parity, src0/src1 banks, one literal) and at most one SGPR is read.
//!
//! Every scheduled instruction or packet still goes through the Builder's
//! checked `emit`: ledger waits, SGPR hazards and `s_delay_alu` hints are the
//! shared ones, computed on the final order.
use crate::insn::Instruction;
use crate::reg::Kind;
use crate::vopd::{Operand, VopdF32, VopdOp};

/// Issue slots between a VALU producer and a consumer that issues without a
/// dependency stall (the `s_delay_alu` VALU_DEP window is 4).
pub(crate) const VALU_LATENCY: usize = 4;

/// A plain VALU instruction the window may reorder.
pub(crate) fn schedulable(insn: &Instruction) -> bool {
    let m = insn.mnemonic();
    if !m.starts_with("v_") || insn.memory.is_some() || insn.text.contains('\n') {
        return false;
    }
    if insn.defs.is_empty() || insn.defs.iter().any(|d| d.kind != Kind::V) {
        return false;
    }
    // Implicit scalar operands the text parsers do not declare, cross-lane reads
    // whose disabled lanes keep the old destination, and partial-register writes.
    const TEXT: [&str; 14] = [
        "vcc",
        "exec",
        "m0",
        "scc",
        "ttmp",
        "row_",
        "quad_perm",
        "dpp",
        "op_sel",
        ".h",
        ".l",
        "bank_mask",
        "fi:",
        "sdwa",
    ];
    if TEXT.iter().any(|t| insn.text.contains(t)) {
        return false;
    }
    const MNEMONIC: [&str; 22] = [
        "v_cmp",
        "v_readlane",
        "v_readfirstlane",
        "v_writelane",
        "v_cndmask",
        "v_add_co",
        "v_sub_co",
        "v_subrev_co",
        "v_addc",
        "v_subb",
        "v_div_",
        "v_nop",
        "v_wmma",
        "v_swmmac",
        "v_swap",
        "v_movrel",
        "v_permlane",
        "v_interp",
        "v_mad_co",
        "v_mad_u64",
        "v_mad_i64",
        "v_dual_",
    ];
    if MNEMONIC.iter().any(|p| m.starts_with(p)) {
        return false;
    }
    // 16-bit results write half a VGPR; only full 32-bit results are reordered.
    const WIDE16: [&str; 5] = [
        "v_cvt_f32_f16",
        "v_cvt_f32_bf16",
        "v_fma_mix_f32",
        "v_pk_",
        "v_dot2",
    ];
    if m.contains("16") && !WIDE16.iter().any(|p| m.starts_with(p)) {
        return false;
    }
    true
}

/// The VOPD half an instruction can become: plain `v_{add,sub,subrev,mul,fmac}_f32`
/// (VOP2 encoding, no modifiers) with a VGPR src1.
pub(crate) fn vopd_half(insn: &Instruction) -> Option<VopdOp> {
    let (m, rest) = insn.text.trim().split_once(char::is_whitespace)?;
    let op = match m.strip_suffix("_e32").unwrap_or(m) {
        "v_add_f32" => VopdF32::Add,
        "v_sub_f32" => VopdF32::Sub,
        "v_subrev_f32" => VopdF32::Subrev,
        "v_mul_f32" => VopdF32::Mul,
        "v_fmac_f32" => VopdF32::Fmac,
        _ => return None,
    };
    let ops: Vec<&str> = rest.split(',').map(str::trim).collect();
    let [dst, src0, src1] = ops.as_slice() else {
        return None;
    };
    let vgpr = |t: &str| t.strip_prefix('v')?.parse::<u8>().ok();
    let src0 = if let Some(v) = vgpr(src0) {
        Operand::V(v)
    } else if let Some(s) = src0.strip_prefix('s').and_then(|n| n.parse::<u8>().ok()) {
        Operand::S(s)
    } else {
        match src0.parse::<i8>() {
            Ok(n) if (-16..=64).contains(&n) => Operand::Inline(n),
            _ => return None,
        }
    };
    Some(VopdOp {
        op,
        dst: vgpr(dst)?,
        src0,
        src1: vgpr(src1)?,
    })
}

/// The packet for `a` and `b`, when RDNA4 can dual-issue them. Operands are
/// never commuted: with two NaN inputs the result payload follows operand order.
pub(crate) fn pair(
    arch: crate::arch::Arch,
    a: &Instruction,
    b: &Instruction,
) -> Option<Instruction> {
    let (x, y) = (vopd_half(a)?, vopd_half(b)?);
    if x.dst == y.dst {
        return None;
    }
    // One constant-bus read per packet keeps the pair inside every VOPD SGPR limit.
    let sgprs = [x.src0, y.src0]
        .iter()
        .filter(|s| matches!(s, Operand::S(_)))
        .count();
    if sgprs > 1 {
        return None;
    }
    crate::vopd::packet(arch, x, y)
        .or_else(|_| crate::vopd::packet(arch, y, x))
        .ok()
}

/// The register dependence DAG of one window.
pub(crate) struct Dag {
    pub(crate) insns: Vec<Instruction>,
    /// RAW predecessors (latency-carrying) and all predecessors of each node.
    raw: Vec<Vec<usize>>,
    preds: Vec<Vec<usize>>,
    /// Longest latency-weighted path from the node to a sink.
    pub(crate) height: Vec<usize>,
}
impl Dag {
    pub(crate) fn new(insns: Vec<Instruction>) -> Self {
        let n = insns.len();
        let (mut raw, mut preds) = (vec![vec![]; n], vec![vec![]; n]);
        for j in 0..n {
            for i in 0..j {
                let (p, c) = (&insns[i], &insns[j]);
                let hit = |a: &[crate::reg::RegRef], b: &[crate::reg::RegRef]| {
                    a.iter().any(|x| b.iter().any(|y| x.overlaps(*y)))
                };
                let r = hit(&p.defs, &c.uses);
                if r || hit(&p.defs, &c.defs) || hit(&p.uses, &c.defs) {
                    preds[j].push(i);
                    if r {
                        raw[j].push(i)
                    }
                }
            }
        }
        let mut height = vec![0usize; n];
        for i in (0..n).rev() {
            let mut h = 1;
            for j in i + 1..n {
                if raw[j].contains(&i) {
                    h = h.max(height[j] + VALU_LATENCY)
                } else if preds[j].contains(&i) {
                    h = h.max(height[j] + 1)
                }
            }
            height[i] = h;
        }
        Self {
            insns,
            raw,
            preds,
            height,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.insns.len()
    }
    pub(crate) fn ready(&self, issued: &[Option<usize>], i: usize) -> bool {
        issued[i].is_none() && self.preds[i].iter().all(|p| issued[*p].is_some())
    }
    /// Issue slots this node would stall at slot `now`.
    pub(crate) fn stall(&self, issued: &[Option<usize>], i: usize, now: usize) -> usize {
        self.raw[i]
            .iter()
            .filter_map(|p| issued[*p])
            .map(|t| (t + VALU_LATENCY).saturating_sub(now))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Arch;
    use crate::reg::RegRef;
    fn v(n: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base: n,
            len: 1,
        }
    }
    fn fmac(d: u8, a: u8, b: u8) -> Instruction {
        Instruction::new(
            format!("v_fmac_f32_e32 v{d}, v{a}, v{b}"),
            vec![v(d)],
            vec![v(a), v(b), v(d)],
        )
    }
    #[test]
    fn pairs_follow_rdna4_bank_rules() {
        // Opposite dst parity, src0 banks 0/1, shared src1: legal on gfx12.
        assert!(pair(Arch::Gfx1201, &fmac(8, 0, 4), &fmac(9, 1, 4)).is_some());
        // Same dst parity.
        assert!(pair(Arch::Gfx1201, &fmac(8, 0, 4), &fmac(10, 1, 4)).is_none());
        // src0 bank collision.
        assert!(pair(Arch::Gfx1201, &fmac(8, 0, 4), &fmac(9, 4, 5)).is_none());
        // Modifiers and non-VOPD opcodes never pair.
        let e64 = Instruction::new(
            "v_fmac_f32_e64 v9, -v1, v4",
            vec![v(9)],
            vec![v(1), v(4), v(9)],
        );
        assert!(pair(Arch::Gfx1201, &fmac(8, 0, 4), &e64).is_none());
    }
    #[test]
    fn implicit_operands_are_not_reordered() {
        let cmp = Instruction::new("v_cmp_gt_f32_e32 vcc_lo, v1, v2", vec![], vec![v(1), v(2)]);
        assert!(!schedulable(&cmp));
        let dpp = Instruction::new(
            "v_add_f32_dpp v1, v2, v3 row_shr:1",
            vec![v(1)],
            vec![v(2), v(3)],
        );
        assert!(!schedulable(&dpp));
        let half = Instruction::new("v_add_f16_e32 v1, v2, v3", vec![v(1)], vec![v(2), v(3)]);
        assert!(!schedulable(&half));
        assert!(schedulable(&fmac(1, 2, 3)));
    }
    #[test]
    fn dag_orders_only_dependent_pairs() {
        let d = Dag::new(vec![fmac(8, 0, 4), fmac(8, 1, 5), fmac(9, 2, 6)]);
        let issued = vec![None; 3];
        assert!(d.ready(&issued, 0) && !d.ready(&issued, 1) && d.ready(&issued, 2));
        assert!(d.height[0] > d.height[2]);
    }
}
