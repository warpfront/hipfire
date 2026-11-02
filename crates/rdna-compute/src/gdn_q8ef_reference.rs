// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure-CPU reference of the Qwen4 GDN `q8_ef` state precision mode (CONTRACT
//! §5-§7, `kernels/src/tensor_ops.hip` `gdn_q8ef_*` helpers and `*_q8ef`
//! entries). Test-only; no GPU, no extra crates (the `half` dev-dependency is
//! used only to cross-check the hand-written f16 conversion).
//!
//! Model: the recurrent slot is plain Q8 (i8 codes + one F32 scale per row,
//! `[head][row][key]`, `row` = value channel, `key` = key channel) and a
//! separate live F16 residual plane with the same element index. Every
//! boundary is
//!
//! ```text
//! S_in  = (f32(q) * scale) + f32(f16 ef)          (each op rounded once)
//! peak  = max |S_out|        scale = peak > 0 ? peak / 127 : 1
//! inv   = peak > 0 ? 127 / peak : 0
//! q     = clamp(rint_ties_even(S_out * inv), -128, 127)
//! ef    = f16_rne(S_out - f32(q) * scale)         (against the stored scale)
//! ```
//!
//! Route models (`route_*`): AR (a boundary per token, state round-tripped
//! through the stored slot), canonical/capture rows (a boundary per token with
//! the next row's `S_in` rebuilt in registers), persistent prefill (one boundary
//! at launch end), and the verify window (capture, then either an all-accept
//! commit that keeps the final residual, or the incumbent plain-Q8 stochastic
//! replay from the pre-window slot followed by a residual reset).

#![allow(dead_code)]

pub const HD: usize = 128;

// ───────────────────────────── scalar helpers ────────────────────────────────

/// BF16 round-to-nearest-even (the kernels' `bf16_round`).
pub fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    if x.is_nan() {
        return f32::from_bits(u | 0x0040_0000);
    }
    f32::from_bits((u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) << 16)
}

/// f32 -> f16 bits, round to nearest even, overflow to infinity, NaN stays NaN
/// (quiet). Integer arithmetic only.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        return if man == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00 | (man >> 13) as u16
        };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// f16 bits -> f32, exact.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    match exp {
        0 => {
            let magnitude = man as f32 * (1.0 / 16_777_216.0);
            if sign != 0 {
                -magnitude
            } else {
                magnitude
            }
        }
        31 => f32::from_bits(
            sign | 0x7f80_0000 | (man << 13) | if man != 0 { 0x0040_0000 } else { 0 },
        ),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

/// Round half to even (the device `rintf`), written out explicitly; non-finite
/// values pass through.
pub fn rint_ties_even(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let t = x.trunc();
    let a = (x - t).abs();
    if a < 0.5 {
        t
    } else if a > 0.5 {
        t + x.signum()
    } else if t % 2.0 != 0.0 {
        t + x.signum()
    } else {
        t
    }
}

pub fn q8_scale(peak: f32) -> f32 {
    if peak > 0.0 {
        peak / 127.0
    } else {
        1.0
    }
}

pub fn q8_inv(peak: f32) -> f32 {
    if peak > 0.0 {
        127.0 / peak
    } else {
        0.0
    }
}

/// `gdn_q8ef_code`: clamp(rint_ties_even(value * inv), -128, 127) with the
/// device `fmaxf`/`fminf` NaN rule (a NaN operand yields the other one).
pub fn ef_code(value: f32, inv: f32) -> f32 {
    rint_ties_even(value * inv).max(-128.0).min(127.0)
}

/// `gdn_q8ef_recon`: `f32(code) * scale + f32(f16 residual)`.
pub fn ef_recon(code: f32, scale: f32, residual_bits: u16) -> f32 {
    code * scale + f16_bits_to_f32(residual_bits)
}

/// Encode one 128-wide row: codes, F32 scale, F16 residual bits (against the
/// stored scale and code).
pub fn quantize_row_ef(values: &[f32]) -> (Vec<i8>, f32, Vec<u16>) {
    assert_eq!(values.len(), HD);
    let mut peak = 0.0f32;
    for &v in values {
        peak = peak.max(v.abs());
    }
    let scale = q8_scale(peak);
    let inv = q8_inv(peak);
    let mut codes = Vec::with_capacity(HD);
    let mut ef = Vec::with_capacity(HD);
    for &v in values {
        let code = ef_code(v, inv);
        codes.push(code as i8);
        ef.push(f32_to_f16_bits(v - code * scale));
    }
    (codes, scale, ef)
}

/// Decode one row: `S_in = code * scale + f16 residual`.
pub fn reconstruct_row_ef(codes: &[i8], scale: f32, ef: &[u16]) -> Vec<f32> {
    codes
        .iter()
        .zip(ef)
        .map(|(&q, &e)| ef_recon(q as f32, scale, e))
        .collect()
}

// ───────────────────────────── state containers ──────────────────────────────

/// A plain Q8 GDN state slot (what a ring slot, mirror and snapshot hold).
#[derive(Clone, Debug, PartialEq)]
pub struct Q8Slot {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Q8Slot {
    /// A reset slot: all bytes zero (code 0, scale 0.0).
    pub fn zeros(value_heads: usize) -> Self {
        Self {
            codes: vec![0; value_heads * HD * HD],
            scales: vec![0.0; value_heads * HD],
        }
    }
}

/// Live Q8 slot plus the separate live F16 residual plane.
#[derive(Clone, Debug)]
pub struct LiveState {
    pub slot: Q8Slot,
    pub ef: Vec<u16>,
}

impl LiveState {
    pub fn zeros(value_heads: usize) -> Self {
        Self {
            slot: Q8Slot::zeros(value_heads),
            ef: vec![0; value_heads * HD * HD],
        }
    }

    /// Fresh encode of a dense `[head][row][key]` F32 state (the
    /// `gdn_state_f32_to_q8ef` converter: new codes, scales and residual; no
    /// old residual is read).
    pub fn from_dense(dense: &[f32]) -> Self {
        let rows = dense.len() / HD;
        let mut slot = Q8Slot {
            codes: Vec::with_capacity(dense.len()),
            scales: Vec::with_capacity(rows),
        };
        let mut ef = Vec::with_capacity(dense.len());
        for row in 0..rows {
            let (c, s, e) = quantize_row_ef(&dense[row * HD..(row + 1) * HD]);
            slot.codes.extend(c);
            slot.scales.push(s);
            ef.extend(e);
        }
        Self { slot, ef }
    }

    /// `gdn_state_q8ef_to_f32`: the incoming dense state with the residual
    /// added once.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.slot.codes.len());
        for (row, &scale) in self.slot.scales.iter().enumerate() {
            let r = row * HD..(row + 1) * HD;
            out.extend(reconstruct_row_ef(
                &self.slot.codes[r.clone()],
                scale,
                &self.ef[r],
            ));
        }
        out
    }

    /// `gdn_q8ef_residual_zero`.
    pub fn reset_residual(&mut self) {
        self.ef.iter_mut().for_each(|e| *e = 0);
    }

    pub fn residual_is_zero(&self) -> bool {
        self.ef.iter().all(|&e| e == 0)
    }

    /// Serialized bytes (codes, scale bits LE, residual bits LE): NaN-safe
    /// equality for determinism checks.
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = slot_bytes(&self.slot);
        for e in &self.ef {
            out.extend_from_slice(&e.to_le_bytes());
        }
        out
    }
}

pub fn slot_bytes(slot: &Q8Slot) -> Vec<u8> {
    let mut out: Vec<u8> = slot.codes.iter().map(|&c| c as u8).collect();
    for s in &slot.scales {
        out.extend_from_slice(&s.to_bits().to_le_bytes());
    }
    out
}

pub fn output_bits(outputs: &[Vec<f32>]) -> Vec<u32> {
    outputs.iter().flatten().map(|v| v.to_bits()).collect()
}

// ───────────────────── plain Q8 (stochastic) for the rollback replay ─────────

/// `gdn_q8_round` (`tensor_ops.hip`): `floor(value * inv + noise)` clamped, with
/// the LCG noise seeded by head, row, key channel and frame.
fn plain_q8_round(value: f32, inv: f32, head: usize, row: usize, column: usize, frame: u32) -> i8 {
    let mut rng = (head as u32)
        .wrapping_mul(1664525)
        .wrapping_add((row as u32).wrapping_mul(1013904223))
        .wrapping_add((column as u32).wrapping_mul(214013))
        .wrapping_add(frame.wrapping_mul(2531011));
    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
    let noise = (rng >> 8) as f32 * (1.0 / 16_777_216.0);
    value.mul_add(inv, noise).floor().max(-128.0).min(127.0) as i8
}

pub fn quantize_plain(dense: &[f32], frame: u32) -> Q8Slot {
    let rows = dense.len() / HD;
    let mut slot = Q8Slot {
        codes: Vec::with_capacity(dense.len()),
        scales: Vec::with_capacity(rows),
    };
    for flat in 0..rows {
        let row = &dense[flat * HD..(flat + 1) * HD];
        let mut peak = 0.0f32;
        for &v in row {
            peak = peak.max(v.abs());
        }
        let inv = q8_inv(peak);
        let (head, r) = (flat / HD, flat % HD);
        for (column, &v) in row.iter().enumerate() {
            slot.codes
                .push(plain_q8_round(v, inv, head, r, column, frame));
        }
        slot.scales.push(q8_scale(peak));
    }
    slot
}

pub fn dequantize_plain(slot: &Q8Slot) -> Vec<f32> {
    let mut out = Vec::with_capacity(slot.codes.len());
    for (flat, &scale) in slot.scales.iter().enumerate() {
        out.extend(
            slot.codes[flat * HD..(flat + 1) * HD]
                .iter()
                .map(|&q| scale * q as f32),
        );
    }
    out
}

// ───────────────────────────── GDN recurrence ────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub key_heads: usize,
    pub value_heads: usize,
}

#[derive(Clone, Debug)]
pub struct RowInput {
    /// `key_heads * 128`
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    /// `value_heads * 128`
    pub v: Vec<f32>,
    /// `value_heads`
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
}

/// BF16 l2norm factor of one key head's q or k.
fn bf16_l2_inv(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in values {
        let b = bf16_round(x);
        sum += bf16_round(b * b);
    }
    bf16_round(1.0 / (bf16_round(sum) + bf16_round(1.0e-6)).sqrt())
}

/// Normalized q (already times `query_scale`) and k of one key head.
fn qk_norms(q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let q_inv = bf16_l2_inv(q);
    let k_inv = bf16_l2_inv(k);
    let scale = (HD as f32).sqrt().recip();
    let qn = q
        .iter()
        .map(|&x| bf16_round(bf16_round(x) * q_inv) * scale)
        .collect();
    let kn = k
        .iter()
        .map(|&x| bf16_round(bf16_round(x) * k_inv))
        .collect();
    (qn, kn)
}

/// One row of the F32 recurrence on a dense `[head][row][key]` state (the
/// per-element order of `step_f32_cpu` in the Tier C twin: decay, serial fma
/// kv chain, delta, fma update, serial fma output chain; rows are independent).
pub fn step_f32(geom: Geometry, dense: &mut [f32], r: &RowInput, out: &mut [f32]) {
    let repeat = geom.value_heads / geom.key_heads;
    for h in 0..geom.value_heads {
        let kh = h / repeat;
        let (qn, kn) = qk_norms(&r.q[kh * HD..(kh + 1) * HD], &r.k[kh * HD..(kh + 1) * HD]);
        let decay = r.gate[h].exp();
        let beta = r.beta[h];
        for vi in 0..HD {
            let base = (h * HD + vi) * HD;
            let row = &mut dense[base..base + HD];
            for x in row.iter_mut() {
                *x *= decay;
            }
            let mut kv_mem = 0.0f32;
            for kd in 0..HD {
                kv_mem = row[kd].mul_add(kn[kd], kv_mem);
            }
            let delta = (bf16_round(r.v[h * HD + vi]) - kv_mem) * beta;
            for kd in 0..HD {
                row[kd] = kn[kd].mul_add(delta, row[kd]);
            }
            let mut o = 0.0f32;
            for kd in 0..HD {
                o = row[kd].mul_add(qn[kd], o);
            }
            out[h * HD + vi] = o;
        }
    }
}

fn run_row(geom: Geometry, dense: &mut [f32], r: &RowInput) -> Vec<f32> {
    let mut out = vec![0.0f32; geom.value_heads * HD];
    step_f32(geom, dense, r, &mut out);
    out
}

/// Boundary kept in registers (capture/canon rows): quantize every row and
/// rebuild the next row's `S_in` from the new codes, scale and f16-rounded
/// residual, without a store.
fn boundary_in_registers(dense: &mut [f32]) {
    for row in dense.chunks_mut(HD) {
        let (codes, scale, ef) = quantize_row_ef(row);
        let rebuilt = reconstruct_row_ef(&codes, scale, &ef);
        row.copy_from_slice(&rebuilt);
    }
}

// ───────────────────────────── route models ──────────────────────────────────

/// AR: `gated_delta_step_norm128_q8ef` per token; the state is stored and
/// reloaded between tokens.
pub fn route_ar(live: &mut LiveState, geom: Geometry, rows: &[RowInput]) -> Vec<Vec<f32>> {
    let mut outputs = Vec::with_capacity(rows.len());
    for r in rows {
        let mut dense = live.to_dense();
        outputs.push(run_row(geom, &mut dense, r));
        *live = LiveState::from_dense(&dense);
    }
    outputs
}

/// `..._persistent256_canon_q8ef`: a boundary per token in registers, final
/// store to the live slot and the live residual.
pub fn route_canon(live: &mut LiveState, geom: Geometry, rows: &[RowInput]) -> Vec<Vec<f32>> {
    let (outputs, dense) = run_rows_with_register_boundaries(live, geom, rows);
    *live = LiveState::from_dense(&dense);
    outputs
}

fn run_rows_with_register_boundaries(
    live: &LiveState,
    geom: Geometry,
    rows: &[RowInput],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut dense = live.to_dense();
    let mut outputs = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        outputs.push(run_row(geom, &mut dense, r));
        if i + 1 < rows.len() {
            boundary_in_registers(&mut dense);
        }
    }
    (outputs, dense)
}

/// Persistent prefill (`..._persistent256_q8ef` / pipe128 / WY converters): the
/// dense state runs across every row and is quantized once at launch end.
pub fn route_persistent(live: &mut LiveState, geom: Geometry, rows: &[RowInput]) -> Vec<Vec<f32>> {
    let mut dense = live.to_dense();
    let outputs: Vec<Vec<f32>> = rows.iter().map(|r| run_row(geom, &mut dense, r)).collect();
    *live = LiveState::from_dense(&dense);
    outputs
}

pub struct CaptureResult {
    pub outputs: Vec<Vec<f32>>,
    /// The Q8 slot written to ring slot `rows - 1`.
    pub final_slot: Q8Slot,
}

/// `..._persistent256_capture_q8ef`: reads the live slot and residual, writes
/// only the final Q8 slot to the ring (returned) and the final residual to the
/// live plane; the live Q8 slot is never written.
pub fn route_capture(live: &mut LiveState, geom: Geometry, rows: &[RowInput]) -> CaptureResult {
    let (outputs, dense) = run_rows_with_register_boundaries(live, geom, rows);
    let fresh = LiveState::from_dense(&dense);
    live.ef = fresh.ef;
    CaptureResult {
        outputs,
        final_slot: fresh.slot,
    }
}

pub struct VerifyResult {
    pub outputs: Vec<Vec<f32>>,
    pub live: LiveState,
    /// True when the residual was zeroed (keep < rows).
    pub residual_reset: bool,
}

/// A verify window of `rows` rows starting at absolute position
/// `base_frame`, accepting `keep` of them. `keep == rows`: the final captured
/// slot becomes the live slot and the final residual stays. Otherwise the
/// incumbent `gated_delta_rollback_layers_q8` replay runs from the pre-window
/// plain-Q8 slot (stochastic rounding at each kept row's own position, no
/// residual involved) and the live residual is zeroed. `keep == 0` restores
/// the pre-window slot and zeroes the residual.
pub fn route_verify(
    before: &LiveState,
    geom: Geometry,
    rows: &[RowInput],
    keep: usize,
    base_frame: u32,
) -> VerifyResult {
    assert!(keep <= rows.len());
    let mut live = before.clone();
    let capture = route_capture(&mut live, geom, rows);
    if keep == rows.len() {
        live.slot = capture.final_slot;
        return VerifyResult {
            outputs: capture.outputs,
            live,
            residual_reset: false,
        };
    }
    let mut slot = before.slot.clone();
    for (i, r) in rows.iter().take(keep).enumerate() {
        let mut dense = dequantize_plain(&slot);
        run_row(geom, &mut dense, r);
        slot = quantize_plain(&dense, base_frame.wrapping_add(i as u32));
    }
    live.slot = slot;
    live.reset_residual();
    VerifyResult {
        outputs: capture.outputs,
        live,
        residual_reset: true,
    }
}

// ───────────────────────────────── tests ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0
        }
    }

    const GEOM: Geometry = Geometry {
        key_heads: 1,
        value_heads: 2,
    };

    fn random_row(rng: &mut Lcg) -> RowInput {
        let g = GEOM;
        RowInput {
            q: (0..g.key_heads * HD).map(|_| rng.unit() * 1.5).collect(),
            k: (0..g.key_heads * HD).map(|_| rng.unit() * 1.5).collect(),
            v: (0..g.value_heads * HD).map(|_| rng.unit()).collect(),
            gate: (0..g.value_heads)
                .map(|_| -0.02 - 0.28 * (rng.unit() * 0.5 + 0.5))
                .collect(),
            beta: (0..g.value_heads)
                .map(|_| 0.2 + 0.7 * (rng.unit() * 0.5 + 0.5))
                .collect(),
        }
    }

    fn random_rows(rng: &mut Lcg, n: usize) -> Vec<RowInput> {
        (0..n).map(|_| random_row(rng)).collect()
    }

    /// A non-trivial starting live state: an encoded random dense state, so the
    /// residual plane is populated.
    fn random_live(rng: &mut Lcg) -> LiveState {
        let dense: Vec<f32> = (0..GEOM.value_heads * HD * HD)
            .map(|_| rng.unit() * 0.5)
            .collect();
        LiveState::from_dense(&dense)
    }

    fn row_with(first: &[(usize, f32)]) -> Vec<f32> {
        let mut row = vec![0.0f32; HD];
        for &(i, v) in first {
            row[i] = v;
        }
        row
    }

    fn ef_f32(ef: &[u16]) -> Vec<f32> {
        ef.iter().map(|&b| f16_bits_to_f32(b)).collect()
    }

    // ── primitive conversions ──────────────────────────────────────────────

    #[test]
    fn gdn_q8ef_boundary_primitives_match_reference_implementations() {
        // f16 -> f32 is exact for every pattern; f32 -> f16 round trips them.
        for h in 0..=u16::MAX {
            let ours = f16_bits_to_f32(h);
            let theirs = half::f16::from_bits(h).to_f32();
            if theirs.is_nan() {
                assert!(ours.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(ours.to_bits(), theirs.to_bits(), "{h:#06x}");
                assert_eq!(f32_to_f16_bits(ours), h, "{h:#06x}");
            }
        }
        // Exact midpoints between adjacent finite f16 values (ties), and the
        // f32 neighbours of each midpoint, both signs.
        for h in 0u16..0x7bff {
            let mid = (f16_bits_to_f32(h) + f16_bits_to_f32(h + 1)) * 0.5;
            for x in [
                mid,
                f32::from_bits(mid.to_bits() - 1),
                f32::from_bits(mid.to_bits() + 1),
            ] {
                for s in [x, -x] {
                    assert_eq!(
                        f32_to_f16_bits(s),
                        half::f16::from_f32(s).to_bits(),
                        "x={s:e} bits={:#010x}",
                        s.to_bits()
                    );
                }
            }
        }
        // Range edges: overflow threshold 65520 (tie to inf), max finite, tiny.
        for x in [
            65504.0f32,
            65519.99,
            65520.0,
            65536.0,
            1.0e30,
            f32::INFINITY,
            0.0,
            -0.0,
            2.9802322e-8,
            2.98023e-8,
            2.9802325e-8,
            5.96046e-8,
            1.0e-30,
        ] {
            for s in [x, -x] {
                assert_eq!(
                    f32_to_f16_bits(s),
                    half::f16::from_f32(s).to_bits(),
                    "{s:e}"
                );
            }
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // rint: ties to even, both signs, vs the std implementation.
        let mut rng = Lcg(7);
        for i in 0..20_000 {
            let x = if i % 2 == 0 {
                rng.unit() * 300.0
            } else {
                (rng.next_u32() % 600) as f32 * 0.5 - 150.0
            };
            assert_eq!(
                rint_ties_even(x).to_bits(),
                x.round_ties_even().to_bits(),
                "{x}"
            );
        }
        for x in [
            0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, 127.5, -127.5, -128.5, 8388609.0,
        ] {
            assert_eq!(
                rint_ties_even(x).to_bits(),
                x.round_ties_even().to_bits(),
                "{x}"
            );
        }
        assert!(rint_ties_even(f32::NAN).is_nan());
        assert_eq!(rint_ties_even(f32::INFINITY), f32::INFINITY);
    }

    // ── frozen test: gdn_q8ef_boundary ─────────────────────────────────────

    #[test]
    fn gdn_q8ef_boundary() {
        // ±ties and exact halves. Peak 127 -> scale == 1.0, inv == 1.0, so the
        // products are exact and every half lands on a tie.
        let row = row_with(&[
            (0, 127.0),
            (1, 0.5),
            (2, 1.5),
            (3, 2.5),
            (4, -0.5),
            (5, -1.5),
            (6, -2.5),
            (7, 126.5),
            (8, -126.5),
            (9, 125.5),
        ]);
        let (codes, scale, ef) = quantize_row_ef(&row);
        assert_eq!(scale, 1.0);
        assert_eq!(&codes[..10], &[127, 0, 2, 2, 0, -2, -2, 126, -126, 126]);
        let resid = ef_f32(&ef);
        assert_eq!(
            &resid[..10],
            &[0.0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5]
        );
        assert!(codes[10..].iter().all(|&c| c == 0) && resid[10..].iter().all(|&r| r == 0.0));
        // The code/residual pair reconstructs every element exactly.
        assert_eq!(reconstruct_row_ef(&codes, scale, &ef), row);

        // The clamp: endpoints of the signed byte range, ties at the edge.
        assert_eq!(ef_code(130.0, 1.0), 127.0);
        assert_eq!(ef_code(-130.0, 1.0), -128.0);
        assert_eq!(ef_code(127.5, 1.0), 127.0); // tie -> 128 (even) -> clamp
        assert_eq!(ef_code(-127.5, 1.0), -128.0); // tie -> -128 (even)
        assert_eq!(ef_code(126.5, 1.0), 126.0);
        assert_eq!(ef_code(-128.4, 1.0), -128.0);
        // Peak endpoints: +peak -> +127, -peak -> -127 (never -128).
        for peak in [1.0f32, 0.0173, 3.3e4, 7.7e-6, 123.456] {
            let (codes, scale, ef) =
                quantize_row_ef(&row_with(&[(0, peak), (1, -peak), (5, peak * 0.31)]));
            assert_eq!((codes[0], codes[1]), (127, -127), "peak {peak}");
            assert!(codes.iter().all(|&c| c != -128));
            assert!(
                ef_f32(&ef)
                    .iter()
                    .all(|r| r.abs() <= 0.5 * scale * 1.001 + (-25f32).exp2()),
                "peak {peak}"
            );
        }

        // Zero rows: scale 1, q = 0, ef = 0 (the positive-zero row is all bits
        // zero); a negative-zero row also reconstructs to zero.
        let (codes, scale, ef) = quantize_row_ef(&vec![0.0; HD]);
        assert_eq!(scale, 1.0);
        assert!(codes.iter().all(|&c| c == 0) && ef.iter().all(|&e| e == 0));
        let (codes, scale, ef) = quantize_row_ef(&vec![-0.0; HD]);
        assert_eq!(scale, 1.0);
        assert!(codes.iter().all(|&c| c == 0));
        assert!(reconstruct_row_ef(&codes, scale, &ef)
            .iter()
            .all(|&x| x == 0.0));

        // Random rows: |ef| <= step/2 (+ f32/f16 rounding allowance), both signs
        // occur, and the reconstruction adds the residual once.
        let mut rng = Lcg(11);
        let (mut pos, mut neg) = (0usize, 0usize);
        for _ in 0..200 {
            let amp = [1.0f32, 1.0e-3, 80.0, 3.0e-5][(rng.next_u32() % 4) as usize];
            let row: Vec<f32> = (0..HD).map(|_| rng.unit() * amp).collect();
            let (codes, scale, ef) = quantize_row_ef(&row);
            let resid = ef_f32(&ef);
            let recon = reconstruct_row_ef(&codes, scale, &ef);
            for i in 0..HD {
                let bound = 0.5 * scale * (1.0 + 1.0 / 512.0) + (-24f32).exp2();
                assert!(resid[i].abs() <= bound, "ef {} > {}", resid[i], bound);
                if resid[i] > 0.0 {
                    pos += 1;
                } else if resid[i] < 0.0 {
                    neg += 1;
                }
                // Exactly the documented expression, bit for bit.
                let expect = (codes[i] as f32 * scale) + f16_bits_to_f32(ef[i]);
                assert_eq!(recon[i].to_bits(), expect.to_bits());
                // Codes + one residual reproduce the value to f16 residual
                // precision; adding the residual twice would be off by |ef|.
                let err = (recon[i] - row[i]).abs();
                assert!(
                    err <= row[i].abs() * (-22f32).exp2()
                        + resid[i].abs() * (-10f32).exp2()
                        + (-25f32).exp2(),
                    "err {err}"
                );
            }
        }
        assert!(pos > 1000 && neg > 1000, "residual signs: +{pos} -{neg}");

        // f16 subnormal residuals: scale ~1e-4, residuals ~3e-6 are below the
        // f16 normal minimum (6.1e-5); ones under half the smallest subnormal
        // (2^-25) round to zero.
        let s0 = 1.0e-4f32;
        let peak = 127.0 * s0;
        let (_, scale, _) = quantize_row_ef(&row_with(&[(0, peak)]));
        let v_sub = 10.0 * scale + 3.0e-6;
        let v_tiny_pos = 10.0 * scale + 1.0e-8;
        let v_tiny_neg = 10.0 * scale - 1.0e-8;
        let (codes, scale, ef) = quantize_row_ef(&row_with(&[
            (0, peak),
            (1, v_sub),
            (2, v_tiny_pos),
            (3, v_tiny_neg),
            (4, -v_sub),
        ]));
        for i in 1..=4 {
            assert_eq!(codes[i].abs(), 10, "element {i}");
        }
        let exact = v_sub - 10.0 * scale;
        assert!(exact.abs() < 6.1e-5 && exact.abs() > 1.0e-6);
        assert_eq!(ef[1], half::f16::from_f32(exact).to_bits());
        assert!(ef[1] & 0x7c00 == 0 && ef[1] & 0x03ff != 0, "{:#06x}", ef[1]);
        assert!((f16_bits_to_f32(ef[1]) - exact).abs() <= (-25f32).exp2());
        assert_eq!(f16_bits_to_f32(ef[2]), 0.0);
        assert_eq!(f16_bits_to_f32(ef[3]), 0.0);
        assert!(f16_bits_to_f32(ef[4]) < 0.0 && ef[4] & 0x7c00 == 0);

        // f16 overflow: a residual above 65504 (needs a row peak ~1.6e7) is NOT
        // clamped; it becomes an infinity and the reconstruction is non-finite.
        let big = 4.0e7f32;
        let (_, scale, _) = quantize_row_ef(&row_with(&[(0, big)]));
        let v_over = 100.4 * scale;
        let v_ok = 100.0 * scale + 60000.0;
        let (codes, scale, ef) =
            quantize_row_ef(&row_with(&[(0, big), (1, v_over), (2, -v_over), (3, v_ok)]));
        assert_eq!(codes[1], 100);
        assert_eq!(ef[1], 0x7c00);
        assert_eq!(ef[2], 0xfc00);
        let recon = reconstruct_row_ef(&codes, scale, &ef);
        assert_eq!(recon[1], f32::INFINITY);
        assert_eq!(recon[2], f32::NEG_INFINITY);
        let r_ok = f16_bits_to_f32(ef[3]);
        assert!(r_ok.is_finite() && (r_ok - 60000.0).abs() < 64.0, "{r_ok}");
        assert!(recon[0].is_finite() && recon[3].is_finite());

        // Non-finite rows propagate. A NaN element: its residual and
        // reconstruction are NaN (fmax ignores it for the peak, the others
        // reconstruct normally). An infinity: the scale is inf, inv 0, and the
        // whole row reconstructs non-finite.
        let mut nan_row = row_with(&[(0, 1.0), (1, -0.25), (2, 0.5)]);
        nan_row[7] = f32::NAN;
        let (codes, scale, ef) = quantize_row_ef(&nan_row);
        assert!(scale.is_finite());
        let recon = reconstruct_row_ef(&codes, scale, &ef);
        assert!(recon[7].is_nan() && f16_bits_to_f32(ef[7]).is_nan());
        assert!(recon
            .iter()
            .enumerate()
            .all(|(i, x)| i == 7 || x.is_finite()));
        let mut inf_row = row_with(&[(0, 1.0), (1, -0.25)]);
        inf_row[9] = f32::INFINITY;
        let (codes, scale, ef) = quantize_row_ef(&inf_row);
        assert_eq!(scale, f32::INFINITY);
        let recon = reconstruct_row_ef(&codes, scale, &ef);
        assert!(recon.iter().all(|x| !x.is_finite()));
        let mut ninf_row = row_with(&[(0, 1.0)]);
        ninf_row[3] = f32::NEG_INFINITY;
        let (codes, scale, ef) = quantize_row_ef(&ninf_row);
        assert!(reconstruct_row_ef(&codes, scale, &ef)
            .iter()
            .all(|x| !x.is_finite()));

        // Reconstruction adds the residual exactly once: a state with a non-zero
        // residual decodes to the zero-residual decode plus f32(ef) (one f32
        // rounding), and re-encoding a decoded state ignores the old residual
        // (fresh codes + fresh residual reproduce the decoded values).
        let live = random_live(&mut Lcg(23));
        assert!(!live.residual_is_zero());
        let with_ef = live.to_dense();
        let mut no_ef = live.clone();
        no_ef.reset_residual();
        let without_ef = no_ef.to_dense();
        for i in 0..with_ef.len() {
            let e = f16_bits_to_f32(live.ef[i]);
            assert!(
                (with_ef[i] - (without_ef[i] + e)).abs()
                    <= with_ef[i].abs() * (-23f32).exp2() * 2.0
            );
            assert_eq!(with_ef[i].to_bits(), (without_ef[i] + e).to_bits());
        }
        let reencoded = LiveState::from_dense(&with_ef);
        let again = reencoded.to_dense();
        for i in 0..with_ef.len() {
            let tol = with_ef[i].abs() * (-22f32).exp2()
                + 0.5 * reencoded.slot.scales[i / HD] * (-10f32).exp2()
                + (-25f32).exp2();
            assert!((again[i] - with_ef[i]).abs() <= tol);
        }
    }

    // ── frozen test: gdn_q8ef_route_determinism ────────────────────────────

    /// A mixed route + reset history exercised through the model.
    fn run_history(seed: u64) -> (Vec<u8>, Vec<u32>) {
        let mut rng = Lcg(seed);
        let mut live = random_live(&mut rng);
        let mut bits = Vec::new();
        let mut record = |outs: &[Vec<f32>]| bits.extend(output_bits(outs));
        // AR, a partial-accept window, AR, persistent prefill, an all-accept
        // window, a keep-0 window, then canon rows.
        let ar = random_rows(&mut rng, 3);
        record(&route_ar(&mut live, GEOM, &ar));
        let win = random_rows(&mut rng, 4);
        let v = route_verify(&live, GEOM, &win, 2, 100);
        record(&v.outputs);
        live = v.live;
        let ar = random_rows(&mut rng, 2);
        record(&route_ar(&mut live, GEOM, &ar));
        let pre = random_rows(&mut rng, 5);
        record(&route_persistent(&mut live, GEOM, &pre));
        let win = random_rows(&mut rng, 3);
        let v = route_verify(&live, GEOM, &win, 3, 200);
        record(&v.outputs);
        live = v.live;
        let win = random_rows(&mut rng, 4);
        let v = route_verify(&live, GEOM, &win, 0, 300);
        record(&v.outputs);
        live = v.live;
        let canon = random_rows(&mut rng, 3);
        record(&route_canon(&mut live, GEOM, &canon));
        (live.bytes(), bits)
    }

    #[test]
    fn gdn_q8ef_route_determinism() {
        // Same route + reset history -> identical bytes, repeated runs.
        let first = run_history(0xC0FFEE);
        for _ in 0..3 {
            assert!(
                run_history(0xC0FFEE) == first,
                "history replay not byte-identical"
            );
        }
        assert!(
            run_history(0xC0FFEF) != first,
            "different inputs must differ"
        );

        let mut rng = Lcg(5);
        let start = random_live(&mut rng);
        let rows = random_rows(&mut rng, 4);

        // Capture-with-same-boundaries == AR == canon, byte for byte, outputs
        // included: the register rebuild is the stored round trip.
        let mut ar = start.clone();
        let ar_out = route_ar(&mut ar, GEOM, &rows);
        let mut canon = start.clone();
        let canon_out = route_canon(&mut canon, GEOM, &rows);
        let all = route_verify(&start, GEOM, &rows, rows.len(), 1000);
        assert!(!ar.residual_is_zero());
        assert_eq!(ar.bytes(), canon.bytes());
        assert_eq!(ar.bytes(), all.live.bytes());
        assert_eq!(output_bits(&ar_out), output_bits(&canon_out));
        assert_eq!(output_bits(&ar_out), output_bits(&all.outputs));
        // All-accept keeps the final residual (not reset).
        assert!(!all.residual_reset && !all.live.residual_is_zero());

        // Capture alone: the live Q8 slot is untouched, the residual plane is
        // the final residual, the ring slot is the AR final slot.
        let mut cap_live = start.clone();
        let cap = route_capture(&mut cap_live, GEOM, &rows);
        assert_eq!(cap_live.slot, start.slot);
        assert_ne!(cap_live.ef, start.ef);
        assert_eq!(cap_live.ef, ar.ef);
        assert_eq!(cap.final_slot, ar.slot);

        // Every keep: k < rows zeroes the residual exactly (keep 0 restores the
        // pre-window slot), k == rows keeps it; each is deterministic.
        for keep in 0..=rows.len() {
            let a = route_verify(&start, GEOM, &rows, keep, 1000);
            let b = route_verify(&start, GEOM, &rows, keep, 1000);
            assert_eq!(a.live.bytes(), b.live.bytes(), "keep {keep}");
            assert_eq!(
                output_bits(&a.outputs),
                output_bits(&b.outputs),
                "keep {keep}"
            );
            if keep < rows.len() {
                assert!(a.residual_reset);
                assert!(
                    a.live.ef.iter().all(|&e| e == 0),
                    "keep {keep}: residual not zero"
                );
            } else {
                assert!(!a.residual_reset && !a.live.residual_is_zero());
            }
            if keep == 0 {
                assert_eq!(a.live.slot, start.slot);
            }
            // Verify outputs are the capture outputs for every keep.
            assert_eq!(output_bits(&a.outputs), output_bits(&ar_out));
        }

        // EXPECTED, REPORTED DIFFERENCE (not equality): capture + partial
        // rollback + reset is not the AR continuation. AR over the kept rows
        // leaves a non-zero residual; the rollback leaves zero and replays plain
        // stochastic Q8 from the pre-window slot.
        for keep in 1..rows.len() {
            let mut ar_k = start.clone();
            route_ar(&mut ar_k, GEOM, &rows[..keep]);
            let rb = route_verify(&start, GEOM, &rows, keep, 1000);
            assert!(!ar_k.residual_is_zero());
            assert!(rb.live.residual_is_zero());
            assert_ne!(ar_k.bytes(), rb.live.bytes(), "keep {keep}");
            let code_diff = ar_k
                .slot
                .codes
                .iter()
                .zip(&rb.live.slot.codes)
                .filter(|(a, b)| a != b)
                .count();
            let ef_nonzero = ar_k.ef.iter().filter(|&&e| e != 0).count();
            eprintln!(
                "gdn_q8ef_route_determinism: keep {keep}/{}: rollback+reset vs AR differ \
                 (codes {code_diff}/{}, AR residual non-zero {ef_nonzero}/{}, rollback residual 0)",
                rows.len(),
                ar_k.slot.codes.len(),
                ar_k.ef.len()
            );
        }

        // A single persistent boundary rounds differently from per-token AR:
        // also reported, different route/rounding order.
        let mut persistent = start.clone();
        route_persistent(&mut persistent, GEOM, &rows);
        let diff = ar
            .bytes()
            .iter()
            .zip(persistent.bytes())
            .filter(|(a, b)| **a != *b)
            .count();
        eprintln!("gdn_q8ef_route_determinism: persistent vs AR differing state bytes: {diff}");

        // After a reset the next step starts from S_in = q * scale alone.
        let rb = route_verify(&start, GEOM, &rows, 2, 1000);
        let mut next_a = rb.live.clone();
        let mut next_b = rb.live.clone();
        let step = random_rows(&mut rng, 1);
        let out_a = route_ar(&mut next_a, GEOM, &step);
        let out_b = route_ar(&mut next_b, GEOM, &step);
        assert_eq!(next_a.bytes(), next_b.bytes());
        assert_eq!(output_bits(&out_a), output_bits(&out_b));
        assert!(
            !next_a.residual_is_zero(),
            "the next uninterrupted step re-accumulates a residual"
        );
    }
}
