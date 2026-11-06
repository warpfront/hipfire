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

/// Issue slots between a VALU producer and a consumer that issues without a dependency stall. The builder hints
/// (`s_delay_alu`) every VALU read of a result written by one of the previous four VALU instructions, so five
/// slots separate a consumer that needs none. Measured on WideNet F01 (4..8): 5 gives the fewest hints, 6..8 the
/// same, 4 about one more per hundred.
pub(crate) const VALU_LATENCY: usize = 5;

/// Pressure mode: once more than this many VGPR units are live in a window (counted from its own defs and last
/// reads), ties between equally stalled nodes go to the one that ends the most live ranges. Inert below it.
pub(crate) const PRESSURE_BUDGET: usize = 48;

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
    // `v_fma_mix_f32` names its source halves with `op_sel`/`op_sel_hi` but always writes the full 32-bit register.
    let mix = m.starts_with("v_fma_mix_f32");
    if TEXT
        .iter()
        .any(|t| insn.text.contains(t) && !(mix && t.starts_with("op_sel")))
    {
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
    latency: usize,
    /// Per node, the issue slots `-k` (the k-th newest VALU instruction before the window) whose results it reads.
    pre: Vec<Vec<usize>>,
}
impl Dag {
    /// `before`: the destinations of the VALU instructions issued before the window, newest first.
    pub(crate) fn new(
        insns: Vec<Instruction>,
        latency: usize,
        before: &[Vec<crate::reg::RegRef>],
    ) -> Self {
        let n = insns.len();
        let (mut raw, mut preds) = (vec![vec![]; n], vec![vec![]; n]);
        let hit = |a: &[crate::reg::RegRef], b: &[crate::reg::RegRef]| {
            a.iter().any(|x| b.iter().any(|y| x.overlaps(*y)))
        };
        for j in 0..n {
            for i in 0..j {
                let (p, c) = (&insns[i], &insns[j]);
                let r = hit(&p.defs, &c.uses);
                if r || hit(&p.defs, &c.defs) || hit(&p.uses, &c.defs) {
                    preds[j].push(i);
                    if r {
                        raw[j].push(i)
                    }
                }
            }
        }
        let pre: Vec<Vec<usize>> = insns
            .iter()
            .map(|c| {
                (0..before.len())
                    .filter(|&k| hit(&before[k], &c.uses))
                    .map(|k| k + 1)
                    .collect::<Vec<usize>>()
            })
            .collect();
        let mut height = vec![0usize; n];
        for i in (0..n).rev() {
            let mut h = 1;
            for j in i + 1..n {
                if raw[j].contains(&i) {
                    h = h.max(height[j] + latency)
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
            latency,
            pre,
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
        let inside = self.raw[i]
            .iter()
            .filter_map(|p| issued[*p])
            .map(|t| (t + self.latency).saturating_sub(now));
        // a producer issued k slots before the window sits at slot -k
        let outside = self.pre[i]
            .iter()
            .map(|k| self.latency.saturating_sub(now + k));
        inside.chain(outside).max().unwrap_or(0)
    }
}

/// Live VGPR units of one window, tracked from the window's own defs and last uses (the Builder has not seen what
/// follows the window): a unit read before it is defined here is live on entry, a unit defined here and never
/// read again stays live on exit, and a value dies at its last in-window read only if the window defines the
/// unit (a temporary, or overwritten later); anything else may still be read after the window.
pub(crate) struct Pressure {
    uses: Vec<Vec<u8>>,
    defs: Vec<Vec<u8>>,
    /// Remaining in-window readers / writers per unit.
    readers: Vec<u16>,
    writers: Vec<u16>,
    local: Vec<bool>,
    live: Vec<bool>,
    count: usize,
}
impl Pressure {
    fn units(regs: &[crate::reg::RegRef]) -> Vec<u8> {
        let mut out = vec![];
        for r in regs.iter().filter(|r| r.kind == Kind::V) {
            for u in u16::from(r.base)..u16::from(r.base) + u16::from(r.len) {
                if !out.contains(&(u as u8)) {
                    out.push(u as u8)
                }
            }
        }
        out
    }
    pub(crate) fn new(insns: &[Instruction]) -> Self {
        let uses: Vec<Vec<u8>> = insns.iter().map(|i| Self::units(&i.uses)).collect();
        let defs: Vec<Vec<u8>> = insns.iter().map(|i| Self::units(&i.defs)).collect();
        let (mut readers, mut writers) = (vec![0u16; 256], vec![0u16; 256]);
        let (mut live, mut local) = (vec![false; 256], vec![false; 256]);
        for k in 0..insns.len() {
            for &u in &uses[k] {
                readers[u as usize] += 1;
                if !local[u as usize] {
                    live[u as usize] = true;
                }
            }
            for &u in &defs[k] {
                writers[u as usize] += 1;
                local[u as usize] = true;
            }
        }
        let count = live.iter().filter(|l| **l).count();
        Self {
            uses,
            defs,
            readers,
            writers,
            local,
            live,
            count,
        }
    }
    pub(crate) fn live(&self) -> usize {
        self.count
    }
    fn ends(&self, i: usize, u: u8) -> bool {
        self.readers[u as usize] == 1
            && !self.defs[i].contains(&u)
            && (self.writers[u as usize] > 0 || self.local[u as usize])
    }
    /// Change of the live count if node `i` issues now: units it starts minus units whose value ends with it.
    pub(crate) fn delta(&self, i: usize) -> isize {
        let started = self.defs[i]
            .iter()
            .filter(|u| !self.live[**u as usize])
            .count() as isize;
        let ended = self.uses[i].iter().filter(|u| self.ends(i, **u)).count() as isize;
        started - ended
    }
    pub(crate) fn issue(&mut self, i: usize) {
        let ended: Vec<u8> = self.uses[i]
            .iter()
            .copied()
            .filter(|u| self.ends(i, *u))
            .collect();
        for &u in &self.uses[i] {
            self.readers[u as usize] -= 1;
        }
        for &u in &self.defs[i] {
            self.writers[u as usize] -= 1;
            if !self.live[u as usize] {
                self.live[u as usize] = true;
                self.count += 1;
            }
        }
        for u in ended {
            if self.live[u as usize] {
                self.live[u as usize] = false;
                self.count -= 1;
            }
        }
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
        // v_fma_mix_f32 names source halves with op_sel_hi but writes a whole register
        let mix = Instruction::new(
            "v_fma_mix_f32 v1, v2, v3, v4 op_sel_hi:[1,0,0]",
            vec![v(1)],
            vec![v(2), v(3), v(4)],
        );
        assert!(schedulable(&mix));
        let sel = Instruction::new(
            "v_add_f32_e64 v1, v2, v3 op_sel:[1,0]",
            vec![v(1)],
            vec![v(2), v(3)],
        );
        assert!(!schedulable(&sel));
    }
    #[test]
    fn producers_before_the_window_stall_its_first_slots() {
        // v8 was written by the newest VALU instruction before the window (slot -1).
        let d = Dag::new(vec![fmac(9, 8, 4), fmac(10, 1, 5)], 5, &[vec![v(8)]]);
        let issued = vec![None; 2];
        assert_eq!(d.stall(&issued, 0, 0), 4);
        assert_eq!(d.stall(&issued, 0, 3), 1);
        assert_eq!(d.stall(&issued, 0, 4), 0);
        assert_eq!(d.stall(&issued, 1, 0), 0);
        // an older producer (slot -3) stalls two slots less
        let d = Dag::new(vec![fmac(9, 8, 4)], 5, &[vec![], vec![], vec![v(8)]]);
        assert_eq!(d.stall(&[None], 0, 0), 2);
    }
    #[test]
    fn pressure_ends_window_local_values_at_their_last_read() {
        let mul = Instruction::new("v_mul_f32_e32 v9, v1, v2", vec![v(9)], vec![v(1), v(2)]);
        let add = Instruction::new("v_add_f32_e32 v8, v8, v9", vec![v(8)], vec![v(8), v(9)]);
        let mut p = Pressure::new(&[mul, add]);
        // v1, v2 and v8 are read before the window writes them: live on entry
        assert_eq!(p.live(), 3);
        // the product starts a range; v1/v2 may still be read after the window
        assert_eq!(p.delta(0), 1);
        p.issue(0);
        assert_eq!(p.live(), 4);
        // the accumulate ends the product (a window-local value) and redefines v8 in place
        assert_eq!(p.delta(1), -1);
        p.issue(1);
        assert_eq!(p.live(), 3);
    }
    #[test]
    fn dag_orders_only_dependent_pairs() {
        let d = Dag::new(vec![fmac(8, 0, 4), fmac(8, 1, 5), fmac(9, 2, 6)], 4, &[]);
        let issued = vec![None; 3];
        assert!(d.ready(&issued, 0) && !d.ready(&issued, 1) && d.ready(&issued, 2));
        assert!(d.height[0] > d.height[2]);
    }
}
