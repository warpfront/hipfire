// SPDX-License-Identifier: Apache-2.0
//! gfx1100 GateUp epilogue adapter: the public CPU twin plus the reviewed device `Finish::MulUp` NaN rule.
//!
//! `mq_i4_reference::grouped_prefill_epilogue_with` (public twin, untouched) finishes every GateUp output with the host
//! product `q * u` (`q` = the epilogue's divide result, `u` = the bf16-rounded Up input). On gfx1100 the final
//! `v_mul_f32_e32 dst, q, up` does not pick the host's NaN payload when its result is NaN, so the adapter keeps the twin
//! for every other output and replaces ONLY the outputs whose HOST product `q * u` is NaN, by the reviewed rule:
//!
//! 1. `q` is NaN               -> result bits `q | 0x0040_0000` (quieted `q`, sign and payload kept);
//! 2. else `u` is NaN          -> result bits `u | 0x0040_0000`;
//! 3. else (`q * u` is NaN: `0 * inf` or `inf * 0`, any signs, either order) -> the device-pinned raw F32 constant
//!    [`INVALID_MUL_BITS`];
//! 4. otherwise the public twin's output, unchanged.
//!
//! ONE public twin run with the real hooks per call. A wrapper `EpilogueIntrinsics` around the hooks captures `q`, the
//! actual quotient the hooks returned to the twin (NOT the gate), at every `divide`; the twin calls `divide` exactly once
//! per live output cell in row-major order, so the call cursor identifies the cell, and `u` is derived from that cell's
//! raw Up input per the published contract (`Bf16::RneFinitePassthrough` + the finite `clear_low`, SOL2.3): a non-finite
//! raw Up is kept as-is, a finite one is `bf16_rne_finite_passthrough_bits(raw) & 0xffff_0000` (the public helper, no
//! copy of the twin's private function). Only the NaN-result outputs are corrected afterwards; everything else is the twin's
//! own output. Production makes exactly that one call (no stub run, no extra hook query, probe key, hit or missing record);
//! only the `_audit`/`_trace` entry points add one DIRECT public `grouped_prefill_epilogue_with::<I>` call with the same
//! real hooks to compare against (audit-only extra queries). Down has no divide and is delegated
//! unchanged. Padding rows (`slot == -1`) are never visited by the twin, stay `+0`, and are never touched; the only extra
//! storage is the exception records of NaN-result outputs. This is the only MulUp model; the public reference is not
//! modified.
//!
//! Applies to `Finish::MulUp` only (gate-first `q * u`); `MulUpFirst` is not covered.

use hipfire_runtime::mq_i4_reference::{
    bf16_rne_finite_passthrough_bits, grouped_prefill_epilogue_with, DecodeKind, EpilogueIntrinsics,
};
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::ops::AddAssign;

/// Quiet bit of an f32 NaN.
pub const QUIET_BIT: u32 = 0x0040_0000;

/// Raw F32 result of the gfx1100 `v_mul_f32` on `0 * inf` / `inf * 0` (all 8 signed/ordered operand pairs observed
/// identical, exactly this constant): receipt `consumer24/rcpt/b4.mulup-pin.json`, observations
/// `artifacts/b4-nan-pair/probe/observations.json` sha256 `2a504a9ccd0b0e892235f8fe93d2c4f0eebe04db33b8f0366efa637c03df79ab`,
/// code object sha256 `3d57f5a555a1cccd35ca567ca62359918e8f5a8ec94419bd1057882f22ff7323` (j618, selfcheck PASS, 3 repeats).
pub const INVALID_MUL_BITS: u32 = 0xffc0_0000;

/// Which branch of the reviewed rule produced the replacement for one NaN-result output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MulUpRule {
    QuietQ(u32),
    QuietUp(u32),
    Invalid,
}

impl MulUpRule {
    pub fn bits(self) -> u32 {
        match self {
            MulUpRule::QuietQ(b) | MulUpRule::QuietUp(b) => b,
            MulUpRule::Invalid => INVALID_MUL_BITS,
        }
    }
}

/// The reviewed rule for one `(q, u)` pair, decided on the raw bits only (no host product): `None` when the host product
/// is not NaN (the twin's output stands). A NaN has `abs > 0x7f80_0000`; `0 * inf` / `inf * 0` (any signs, either order) is
/// `abs(q) == 0x7f80_0000 && abs(u) == 0` or `abs(q) == 0 && abs(u) == 0x7f80_0000`.
pub fn mul_up_rule(q: f32, u: f32) -> Option<MulUpRule> {
    const ABS: u32 = 0x7fff_ffff;
    const INF: u32 = 0x7f80_0000;
    let (qb, ub) = (q.to_bits(), u.to_bits());
    let (qa, ua) = (qb & ABS, ub & ABS);
    if qa > INF {
        Some(MulUpRule::QuietQ(qb | QUIET_BIT))
    } else if ua > INF {
        Some(MulUpRule::QuietUp(ub | QUIET_BIT))
    } else if (qa == INF && ua == 0) || (qa == 0 && ua == INF) {
        Some(MulUpRule::Invalid)
    } else {
        None
    }
}

/// CPU audit of adapter vs the DIRECT public twin (`grouped_prefill_epilogue_with::<I>` with the same real hooks, an extra
/// audit-only call) over GateUp outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MulUpStats {
    /// live GateUp output cells (padding excluded)
    pub cells: u64,
    /// cells whose host product is NaN because `q` is NaN
    pub q_nan: u64,
    /// cells whose host product is NaN because `u` is NaN (`q` not NaN)
    pub u_nan: u64,
    /// cells whose host product is NaN only by `0 * inf` / `inf * 0`
    pub invalid: u64,
    /// NaN-result cells whose BF16 output differs from the direct public twin's
    pub changed_nan: u64,
    /// cells whose output differs from the direct public twin's although they are NOT NaN-result cells of the rule
    /// (MUST be 0: every changed cell is a rule-applied NaN-result cell)
    pub nonnan_changed: u64,
    /// cells where the capture run's own output BEFORE correction differs from the direct public twin's (a different
    /// monomorphization can already reorder NaN payloads); informational, every such cell must be a NaN-result cell
    /// (otherwise it also counts in `nonnan_changed`)
    pub capture_vs_direct: u64,
}

impl AddAssign for MulUpStats {
    fn add_assign(&mut self, o: Self) {
        self.cells += o.cells;
        self.q_nan += o.q_nan;
        self.u_nan += o.u_nan;
        self.invalid += o.invalid;
        self.changed_nan += o.changed_nan;
        self.nonnan_changed += o.nonnan_changed;
        self.capture_vs_direct += o.capture_vs_direct;
    }
}

impl MulUpStats {
    pub fn nan_cells(&self) -> u64 {
        self.q_nan + self.u_nan + self.invalid
    }
    /// Counts accumulated since `earlier` (a snapshot of the same running total).
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            cells: self.cells - earlier.cells,
            q_nan: self.q_nan - earlier.q_nan,
            u_nan: self.u_nan - earlier.u_nan,
            invalid: self.invalid - earlier.invalid,
            changed_nan: self.changed_nan - earlier.changed_nan,
            nonnan_changed: self.nonnan_changed - earlier.nonnan_changed,
            capture_vs_direct: self.capture_vs_direct - earlier.capture_vs_direct,
        }
    }
    pub fn to_json(&self) -> Value {
        json!({"cells": self.cells, "nan_result_cells": self.nan_cells(), "q_nan": self.q_nan, "u_nan": self.u_nan, "invalid_zero_inf": self.invalid,
               "changed_nan_pairs": self.changed_nan, "nonnan_changed": self.nonnan_changed, "capture_vs_direct_pre_correction": self.capture_vs_direct})
    }
}

/// The twin's rounded Up for a raw Up F32 (published contract, SOL2.3): non-finite kept raw, finite RNE-to-BF16 with
/// the low half cleared.
fn rounded_up_bits(raw: u32) -> u32 {
    if raw & 0x7f80_0000 == 0x7f80_0000 {
        raw
    } else {
        bf16_rne_finite_passthrough_bits(raw) & 0xffff_0000
    }
}

/// One live GateUp output cell as the twin saw it: row, column, the actual quotient `q` and the rounded Up `u` (raw bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulUpCell {
    pub row: usize,
    pub col: usize,
    pub q: u32,
    pub u: u32,
}

/// Delegates to the real hooks and, at every `divide` (once per live cell, row-major), records `(output index, rule)` when
/// the host product is NaN (and, when tracing, the cell's row/col/`(q, u)`). The cursor is a (row, col) pair over the
/// borrowed `slot_ids`, skipping padding rows; nothing is allocated per call.
struct QuotientCapture<'a, I> {
    inner: &'a I,
    sums: &'a [f32],
    slots: &'a [i32],
    m: usize,
    oc: usize,
    row: Cell<usize>,
    col: Cell<usize>,
    exceptions: RefCell<Vec<(usize, MulUpRule)>>,
    trace: Option<RefCell<Vec<MulUpCell>>>,
}

impl<I: EpilogueIntrinsics> EpilogueIntrinsics for QuotientCapture<'_, I> {
    fn exp2(&self, x: f32) -> f32 {
        self.inner.exp2(x)
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        self.inner.ldexp(x, e)
    }
    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        let q = self.inner.divide(numerator, denominator);
        let (row, col) = (self.row.get(), self.col.get());
        assert!(
            row < self.slots.len(),
            "gfx1100 epilogue: divide call past the last live cell"
        );
        let u = rounded_up_bits(self.sums[row * self.m + self.oc + col].to_bits());
        if let Some(t) = &self.trace {
            t.borrow_mut().push(MulUpCell {
                row,
                col,
                q: q.to_bits(),
                u,
            });
        }
        if let Some(rule) = mul_up_rule(q, f32::from_bits(u)) {
            self.exceptions
                .borrow_mut()
                .push((row * self.oc + col, rule));
        }
        if col + 1 == self.oc {
            self.col.set(0);
            let mut next = row + 1;
            while next < self.slots.len() && self.slots[next] == -1 {
                next += 1;
            }
            self.row.set(next);
        } else {
            self.col.set(col + 1);
        }
        q
    }
}

/// Result of one adapter run. `twin` (the public twin's output with the same hooks) and `cells` are filled only when
/// auditing / tracing.
pub struct MulUpRun {
    pub out: Vec<u16>,
    pub stats: MulUpStats,
    pub twin: Option<Vec<u16>>,
    pub cells: Vec<MulUpCell>,
}

fn run<I: EpilogueIntrinsics>(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
    hooks: &I,
    audit: bool,
    trace: bool,
) -> MulUpRun {
    let mut stats = MulUpStats::default();
    let rows = slot_ids.len();
    if kind != DecodeKind::GateUp || m == 0 || m % 2 != 0 || rows.checked_mul(m) != Some(sums.len())
    {
        // Down (no divide), or arguments the public twin rejects with its own message.
        let out = grouped_prefill_epilogue_with(kind, m, sums, slot_ids, hooks);
        let twin = if audit { Some(out.clone()) } else { None };
        return MulUpRun {
            out,
            stats,
            twin,
            cells: Vec::new(),
        };
    }
    let oc = m / 2;
    let cells_total = slot_ids.iter().filter(|&&s| s != -1).count() * oc;
    let first_live = slot_ids.iter().position(|&s| s != -1).unwrap_or(rows);
    let qc = QuotientCapture {
        inner: hooks,
        sums,
        slots: slot_ids,
        m,
        oc,
        row: Cell::new(first_live),
        col: Cell::new(0),
        exceptions: RefCell::new(Vec::new()),
        trace: trace.then(|| RefCell::new(Vec::with_capacity(cells_total))),
    };
    let mut out = grouped_prefill_epilogue_with(DecodeKind::GateUp, m, sums, slot_ids, &qc);
    assert!(qc.row.get() == rows && qc.col.get() == 0, "gfx1100 epilogue: the twin did not call divide once per live cell ({cells_total} expected)");
    let exceptions = qc.exceptions.into_inner();
    let cells: Vec<MulUpCell> = qc.trace.map(RefCell::into_inner).unwrap_or_default();
    // audit only: the DIRECT public twin with the same real hooks (extra hook queries; production keeps the one capture call)
    let twin = if audit {
        Some(grouped_prefill_epilogue_with(
            DecodeKind::GateUp,
            m,
            sums,
            slot_ids,
            hooks,
        ))
    } else {
        None
    };
    let pre = if audit { Some(out.clone()) } else { None };
    let mut is_rule_cell = if audit {
        vec![false; out.len()]
    } else {
        Vec::new()
    };
    stats.cells = cells_total as u64;
    for (idx, rule) in exceptions {
        match rule {
            MulUpRule::QuietQ(_) => stats.q_nan += 1,
            MulUpRule::QuietUp(_) => stats.u_nan += 1,
            MulUpRule::Invalid => stats.invalid += 1,
        }
        let new = (bf16_rne_finite_passthrough_bits(rule.bits()) >> 16) as u16;
        let reference = twin.as_ref().map_or(out[idx], |t| t[idx]);
        if new != reference {
            stats.changed_nan += 1;
        }
        if audit {
            is_rule_cell[idx] = true;
        }
        out[idx] = new;
    }
    if let (Some(twin), Some(pre)) = (&twin, &pre) {
        stats.capture_vs_direct = pre.iter().zip(twin).filter(|(a, b)| a != b).count() as u64;
        stats.nonnan_changed = out
            .iter()
            .zip(twin)
            .enumerate()
            .filter(|(i, (a, b))| a != b && !is_rule_cell[*i])
            .count() as u64;
    }
    MulUpRun {
        out,
        stats,
        twin,
        cells,
    }
}

/// The gfx1100 `grouped_prefill_epilogue_with`: same signature and output layout, GateUp NaN-result outputs per the rule.
pub fn grouped_prefill_epilogue<I: EpilogueIntrinsics>(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
    hooks: &I,
) -> Vec<u16> {
    run(kind, m, sums, slot_ids, hooks, false, false).out
}

/// [`grouped_prefill_epilogue`] plus the CPU audit against the public twin (`nonnan_changed` MUST be 0).
pub fn grouped_prefill_epilogue_audit<I: EpilogueIntrinsics>(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
    hooks: &I,
) -> (Vec<u16>, MulUpStats) {
    let r = run(kind, m, sums, slot_ids, hooks, true, false);
    (r.out, r.stats)
}

/// Audit plus every live cell's `(q, u)` and the public twin's own output (for the CPU recheck / historical comparison).
pub fn grouped_prefill_epilogue_trace<I: EpilogueIntrinsics>(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
    hooks: &I,
) -> MulUpRun {
    run(kind, m, sums, slot_ids, hooks, true, true)
}

/// The historical (old binary, `Finish::MulUp` in function 0x224d80, instruction 0x22569c) host product: SSE `mulss` with the Up value in the
/// DESTINATION register and `q` the source (`mulss -0x4(%rax),%xmm0` after `Up -> xmm0`), frozen with inline asm so
/// that the compiler cannot commute the operands. x86 rule: one NaN operand -> that NaN quieted; both NaN -> the
/// destination (Up) quieted. Used only by the CPU recheck as the historical-expectation model.
#[cfg(target_arch = "x86_64")]
pub fn historical_mulss_up_dest_bits(q: u32, u: u32) -> u32 {
    let mut dst = f32::from_bits(u);
    // SAFETY: register-only SSE instruction (baseline on x86_64), no memory, no stack.
    unsafe {
        core::arch::asm!("mulss {d}, {s}", d = inout(xmm_reg) dst, s = in(xmm_reg) f32::from_bits(q), options(pure, nomem, nostack))
    };
    dst.to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::mq_i4_reference::PortableEpilogue;
    use std::collections::HashMap;

    /// STRICT exact-key hooks over `mulup_device_pins.json`: the device's own exp2/ldexp/divide observations for the seven
    /// gate classes of the 56-pair fixture diagnostic (+-1, +-inf, canonical NaN, +/- payload NaN, sNaN) and g = -inf.
    /// A key that is not pinned PANICS (no portable math, no model, no fallback).
    struct PinHooks {
        exp2: HashMap<u32, u32>,
        ldexp: HashMap<(u32, i32), u32>,
        divide: HashMap<(u32, u32), u32>,
    }

    impl PinHooks {
        fn load() -> Self {
            let doc: Value = serde_json::from_str(include_str!("mulup_device_pins.json"))
                .expect("mulup_device_pins.json is valid JSON");
            let u32_of = |v: &Value, k: &str| {
                v[k].as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .unwrap_or_else(|| panic!("pin field {k} is not a u32: {v}"))
            };
            let (mut exp2, mut ldexp, mut divide) =
                (HashMap::new(), HashMap::new(), HashMap::new());
            for e in doc["exp2"].as_array().expect("exp2 pins") {
                assert!(
                    exp2.insert(u32_of(e, "arg"), u32_of(e, "result")).is_none(),
                    "duplicate exp2 pin {e}"
                );
            }
            for e in doc["ldexp"].as_array().expect("ldexp pins") {
                let x = e["exponent"]
                    .as_i64()
                    .and_then(|n| i32::try_from(n).ok())
                    .expect("ldexp pin exponent is an i32");
                assert!(
                    ldexp
                        .insert((u32_of(e, "arg"), x), u32_of(e, "result"))
                        .is_none(),
                    "duplicate ldexp pin {e}"
                );
            }
            for e in doc["divide"].as_array().expect("divide pins") {
                assert!(
                    divide
                        .insert(
                            (u32_of(e, "numerator"), u32_of(e, "denominator")),
                            u32_of(e, "result")
                        )
                        .is_none(),
                    "duplicate divide pin {e}"
                );
            }
            Self {
                exp2,
                ldexp,
                divide,
            }
        }
    }

    impl EpilogueIntrinsics for PinHooks {
        fn exp2(&self, x: f32) -> f32 {
            f32::from_bits(
                *self
                    .exp2
                    .get(&x.to_bits())
                    .unwrap_or_else(|| panic!("unpinned exp2 key {:#010x}", x.to_bits())),
            )
        }
        fn ldexp(&self, x: f32, e: i32) -> f32 {
            f32::from_bits(
                *self
                    .ldexp
                    .get(&(x.to_bits(), e))
                    .unwrap_or_else(|| panic!("unpinned ldexp key ({:#010x}, {e})", x.to_bits())),
            )
        }
        fn divide(&self, n: f32, d: f32) -> f32 {
            f32::from_bits(
                *self
                    .divide
                    .get(&(n.to_bits(), d.to_bits()))
                    .unwrap_or_else(|| {
                        panic!(
                            "unpinned divide key ({:#010x}, {:#010x})",
                            n.to_bits(),
                            d.to_bits()
                        )
                    }),
            )
        }
    }

    /// TEST-ONLY CPU stub (host portable math; NOT a device observation, NOT production): used only where no device pin
    /// exists (g = 0, random finite values) and only for assertions that do not depend on the quotient's exact bits.
    struct CpuStub(PortableEpilogue);

    impl EpilogueIntrinsics for CpuStub {
        fn exp2(&self, x: f32) -> f32 {
            self.0.exp2(x)
        }
        fn ldexp(&self, x: f32, e: i32) -> f32 {
            self.0.ldexp(x, e)
        }
        fn divide(&self, n: f32, d: f32) -> f32 {
            self.0.divide(n, d)
        }
    }

    fn one_with<I: EpilogueIntrinsics>(h: &I, g: u32, u: u32) -> (u16, u16, MulUpStats) {
        let sums = [f32::from_bits(g), f32::from_bits(u)];
        let (a, st) = grouped_prefill_epilogue_audit(DecodeKind::GateUp, 2, &sums, &[0], h);
        let t = grouped_prefill_epilogue_with(DecodeKind::GateUp, 2, &sums, &[0], h);
        (a[0], t[0], st)
    }

    /// One pair through the strict device pins.
    fn one(g: u32, u: u32) -> (u16, u16, MulUpStats) {
        one_with(&PinHooks::load(), g, u)
    }

    /// One pair through the explicit test-only CPU stub.
    fn one_stub(g: u32, u: u32) -> (u16, u16, MulUpStats) {
        one_with(&CpuStub(PortableEpilogue), g, u)
    }

    /// `consumer24/k2-epilogue-nan-probe/probe.out` (device gfx1100 fixture GateUp, raw F32 gate/up -> BF16): 56 pairs.
    const PROBE_OUT: [(u32, u32, u16); 56] = [
        (0x7fc12345, 0x7fc12345, 0x7fc1),
        (0x7fc12345, 0x7f812345, 0x7fc1),
        (0x7fc12345, 0xffc54321, 0x7fc1),
        (0x7fc12345, 0x7fc00000, 0x7fc1),
        (0x7fc12345, 0x3f800000, 0x7fc1),
        (0x7fc12345, 0x7f800000, 0x7fc1),
        (0x7fc12345, 0xbf800000, 0x7fc1),
        (0x7fc12345, 0x7fc2aaaa, 0x7fc1),
        (0x7f812345, 0x7fc12345, 0x7fc1),
        (0x7f812345, 0x7f812345, 0x7fc1),
        (0x7f812345, 0xffc54321, 0x7fc1),
        (0x7f812345, 0x7fc00000, 0x7fc1),
        (0x7f812345, 0x3f800000, 0x7fc1),
        (0x7f812345, 0x7f800000, 0x7fc1),
        (0x7f812345, 0xbf800000, 0x7fc1),
        (0x7f812345, 0x7fc2aaaa, 0x7fc1),
        (0xffc54321, 0x7fc12345, 0xffc5),
        (0xffc54321, 0x7f812345, 0xffc5),
        (0xffc54321, 0xffc54321, 0xffc5),
        (0xffc54321, 0x7fc00000, 0xffc5),
        (0xffc54321, 0x3f800000, 0xffc5),
        (0xffc54321, 0x7f800000, 0xffc5),
        (0xffc54321, 0xbf800000, 0xffc5),
        (0xffc54321, 0x7fc2aaaa, 0xffc5),
        (0x7fc00000, 0x7fc12345, 0x7fc0),
        (0x7fc00000, 0x7f812345, 0x7fc0),
        (0x7fc00000, 0xffc54321, 0x7fc0),
        (0x7fc00000, 0x7fc00000, 0x7fc0),
        (0x7fc00000, 0x3f800000, 0x7fc0),
        (0x7fc00000, 0x7f800000, 0x7fc0),
        (0x7fc00000, 0xbf800000, 0x7fc0),
        (0x7fc00000, 0x7fc2aaaa, 0x7fc0),
        (0x3f800000, 0x7fc12345, 0x7fc1),
        (0x3f800000, 0x7f812345, 0x7fc1),
        (0x3f800000, 0xffc54321, 0xffc5),
        (0x3f800000, 0x7fc00000, 0x7fc0),
        (0x3f800000, 0x3f800000, 0x3f3b),
        (0x3f800000, 0x7f800000, 0x7f80),
        (0x3f800000, 0xbf800000, 0xbf3b),
        (0x3f800000, 0x7fc2aaaa, 0x7fc2),
        (0x7f800000, 0x7fc12345, 0x7fc1),
        (0x7f800000, 0x7f812345, 0x7fc1),
        (0x7f800000, 0xffc54321, 0xffc5),
        (0x7f800000, 0x7fc00000, 0x7fc0),
        (0x7f800000, 0x3f800000, 0x7f80),
        (0x7f800000, 0x7f800000, 0x7f80),
        (0x7f800000, 0xbf800000, 0xff80),
        (0x7f800000, 0x7fc2aaaa, 0x7fc2),
        (0xbf800000, 0x7fc12345, 0x7fc1),
        (0xbf800000, 0x7f812345, 0x7fc1),
        (0xbf800000, 0xffc54321, 0xffc5),
        (0xbf800000, 0x7fc00000, 0x7fc0),
        (0xbf800000, 0x3f800000, 0xbe8a),
        (0xbf800000, 0x7f800000, 0xff80),
        (0xbf800000, 0xbf800000, 0x3e8a),
        (0xbf800000, 0x7fc2aaaa, 0x7fc2),
    ];

    fn is_nan_pair(g: u32, u: u32) -> bool {
        f32::from_bits(g).is_nan() || f32::from_bits(u).is_nan()
    }

    #[test]
    fn probe_out_56_pairs() {
        assert_eq!(PROBE_OUT.len(), 56);
        for (g, u, want) in PROBE_OUT {
            let (a, twin, st) = one(g, u);
            if is_nan_pair(g, u) {
                assert_eq!(
                    a, want,
                    "g={g:#010x} u={u:#010x}: adapter {a:#06x} != device {want:#06x}"
                );
                assert_eq!(st.nan_cells(), 1, "g={g:#010x} u={u:#010x}");
            } else {
                // finite quotient path: the public twin's output stands, and for these probe pairs equals the device value
                assert_eq!(
                    a, twin,
                    "g={g:#010x} u={u:#010x}: non-NaN output must equal the public twin"
                );
                assert_eq!(
                    a, want,
                    "g={g:#010x} u={u:#010x}: finite pair {a:#06x} != device {want:#06x}"
                );
                assert_eq!(st.nan_cells(), 0);
            }
            assert_eq!(st.nonnan_changed, 0);
        }
    }

    #[test]
    fn gate_neg_inf_with_up_nan_uses_q_ffc00000() {
        // g = -inf: the pinned device quotient divide(-inf, +inf) = 0xffc00000; q is NaN, so the result is q | quiet -> 0xffc0
        for u in [0x7fc1_2345u32, 0x7f81_2345, 0xffc5_4321, 0x7fc0_0000] {
            let (a, _, st) = one(0xff80_0000, u);
            assert_eq!(a, 0xffc0, "u={u:#010x}");
            assert_eq!((st.q_nan, st.u_nan, st.invalid), (1, 0, 0));
        }
        // non-NaN Up with the same NaN q: still q
        assert_eq!(one(0xff80_0000, 0x3f80_0000).0, 0xffc0);
    }

    #[test]
    fn single_nan_branches() {
        // q finite, u NaN: quieted u, sign kept
        assert_eq!(one(0x3f80_0000, 0x7f80_0001).0, 0x7fc0);
        assert_eq!(one(0x3f80_0000, 0xff81_2345).0, 0xffc1);
        // q NaN, u finite: quieted q
        assert_eq!(one(0x7f81_2345, 0x3f80_0000).0, 0x7fc1);
        // both NaN: q wins
        assert_eq!(one(0x7fc1_2345, 0xffc5_4321).0, 0x7fc1);
    }

    #[test]
    fn invalid_zero_times_inf_is_pinned() {
        // g = 0 -> q = 0 (0/(1+1)), u = +/-inf: 0 * inf is invalid -> 0xffc00000
        for (g, u) in [
            (0x0000_0000u32, 0x7f80_0000u32),
            (0x0000_0000, 0xff80_0000),
            (0x8000_0000, 0x7f80_0000),
            (0x8000_0000, 0xff80_0000),
        ] {
            let (a, _, st) = one_stub(g, u);
            assert_eq!(a, 0xffc0, "g={g:#010x} u={u:#010x}");
            assert_eq!((st.q_nan, st.u_nan, st.invalid), (0, 0, 1));
        }
        assert_eq!(mul_up_rule(f32::INFINITY, 0.0), Some(MulUpRule::Invalid));
        assert_eq!(
            mul_up_rule(f32::NEG_INFINITY, -0.0),
            Some(MulUpRule::Invalid)
        );
        assert_eq!(mul_up_rule(-0.0, f32::INFINITY), Some(MulUpRule::Invalid));
        assert_eq!(MulUpRule::Invalid.bits(), 0xffc0_0000);
    }

    #[test]
    fn non_nan_outputs_are_the_public_twin_with_padding_and_down_untouched() {
        let h = CpuStub(PortableEpilogue);
        let vals = [
            0.25f32,
            -3.0,
            1.0e-3,
            17.5,
            -0.0,
            0.0,
            1.0e30,
            -1.0e30,
            5.0,
            -7.5,
            f32::INFINITY,
            2.0,
        ];
        let m = 6;
        let rows = 5;
        let mut sums = Vec::new();
        for r in 0..rows {
            for c in 0..m {
                sums.push(vals[(r * 5 + c * 3) % vals.len()]);
            }
        }
        let slots = [0, -1, 3, -1, 7];
        let (a, st) = grouped_prefill_epilogue_audit(DecodeKind::GateUp, m, &sums, &slots, &h);
        let t = grouped_prefill_epilogue_with(DecodeKind::GateUp, m, &sums, &slots, &h);
        assert_eq!(st.nonnan_changed, 0);
        assert_eq!(st.cells, 9);
        // padding rows stay +0, and every cell without a NaN host product equals the twin
        assert!(a[3..6].iter().chain(&a[9..12]).all(|&b| b == 0));
        let nan_cells = st.nan_cells() as usize;
        assert!(a.iter().zip(&t).filter(|(x, y)| x != y).count() <= nan_cells);
        let d = grouped_prefill_epilogue(DecodeKind::Down, m, &sums, &slots, &h);
        assert_eq!(
            d,
            grouped_prefill_epilogue_with(DecodeKind::Down, m, &sums, &slots, &h)
        );
    }

    #[test]
    fn rounded_up_is_captured_not_raw() {
        // raw up 0x7f7fffff rounds (RNE) to +inf in BF16; with q = 0 (g = 0) this is 0 * inf -> pinned, not 0 * finite
        let (a, _, st) = one_stub(0x0000_0000, 0x7f7f_ffff);
        assert_eq!(a, 0xffc0);
        assert_eq!(st.invalid, 1);
    }

    #[test]
    fn mixed_live_and_padding_cursor() {
        // rows: live(NaN up), padding, live(finite) -> exception lands in row 0 col 0 only
        let h = PinHooks::load();
        let sums = [1.0f32, f32::from_bits(0x7fc1_2345), 99.0, 99.0, 1.0, 2.0];
        let slots = [4, -1, 5];
        let (a, st) = grouped_prefill_epilogue_audit(DecodeKind::GateUp, 2, &sums, &slots, &h);
        assert_eq!((st.cells, st.u_nan), (2, 1));
        assert_eq!(a[0], 0x7fc1);
        assert_eq!(a[1], 0);
        let t = grouped_prefill_epilogue_with(DecodeKind::GateUp, 2, &sums, &slots, &h);
        assert_eq!(a[2], t[2]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn historical_sse_up_dest_semantics() {
        // both NaN: the destination (Up) payload, quieted; one NaN: that NaN quieted; none: IEEE product
        assert_eq!(
            historical_mulss_up_dest_bits(0x7fc0_0000, 0x7fc1_2345),
            0x7fc1_2345
        );
        assert_eq!(
            historical_mulss_up_dest_bits(0x7f81_2345, 0xffc5_4321),
            0xffc5_4321
        );
        assert_eq!(
            historical_mulss_up_dest_bits(0xffc0_0000, 0x7f80_0001),
            0x7fc0_0001
        );
        assert_eq!(
            historical_mulss_up_dest_bits(0xffc0_0000, 0x3f80_0000),
            0xffc0_0000
        );
        assert_eq!(
            historical_mulss_up_dest_bits(0x3f80_0000, 0x7f80_0001),
            0x7fc0_0001
        );
        assert_eq!(
            historical_mulss_up_dest_bits(0x4000_0000, 0x3f80_0000),
            0x4000_0000
        );
    }
}
