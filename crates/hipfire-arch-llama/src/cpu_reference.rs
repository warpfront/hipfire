// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU reference for the dense Q8 attention fixture (B2.r): the absolute anchor
//! (I10) for attention-only families.
//!
//! It shares nothing with the engine except the fixture's tensor bytes:
//!
//! * **Weight decode** is its own: `Q8F16` blocks (`f16 scale` + 32 x `i8`) and
//!   F16 norms are decoded here from the layout documented in
//!   [`crate::fixture`], not through `hipfire_runtime`'s dequantisers or the
//!   `Q8_0` GEMV kernels. The reference reads the in-memory
//!   [`HfqMemTensor`]s the writer was given, so the HFQM container read path
//!   is on the engine side of the comparison only.
//! * **Forward** is textbook pre-norm LLaMA, teacher-forced, in `f64`: RMSNorm
//!   (`x / sqrt(mean(x^2) + eps) * w`), half-split RoPE
//!   (`theta^(-2i/head_dim)`), grouped-query causal attention
//!   (`1/sqrt(head_dim)`, query head `h` reads KV head `h / (heads / kv_heads)`),
//!   SwiGLU, final norm, untied `lm_head`. The KV cache is exact `f64`; the
//!   engine's `Q8` KV cache is therefore part of what the budget measures.
//!
//! `f32 = i8 * f16` is exact (7 + 11 significant bits), so the decoded
//! weights are bit-identical to what the engine uploads or computes with, and
//! any difference between engine and reference is arithmetic, not decode.

use crate::fixture::{
    DIM, HEAD_DIM, INTERMEDIATE, N_HEADS, N_KV_HEADS, N_LAYERS, Q8_BLOCK, Q8_BLOCK_BYTES, QT_F16,
    QT_Q8F16, RMS_NORM_EPS, ROPE_THETA, VOCAB,
};
use hipfire_runtime::hfq::HfqMemTensor;

/// Decode one IEEE binary16 value, subnormals and infinities included.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f64;
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        31 if mant == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// A deliberately wrong dequantisation, to prove the budget catches one: the
/// scale of the first `blocks` blocks of `tensor` is multiplied by `scale_mul`.
#[derive(Clone, Debug)]
pub struct DequantFault {
    pub tensor: String,
    pub scale_mul: f64,
    pub blocks: usize,
}

/// Decode `n` elements of a Q8F16 payload (`n / 32` blocks of 34 bytes).
pub fn decode_q8f16(
    data: &[u8],
    n: usize,
    fault: Option<(f64, usize)>,
) -> Result<Vec<f64>, String> {
    if n % Q8_BLOCK != 0 {
        return Err(format!(
            "Q8F16 element count {n} is not a multiple of {Q8_BLOCK}"
        ));
    }
    let blocks = n / Q8_BLOCK;
    if data.len() != blocks * Q8_BLOCK_BYTES {
        return Err(format!(
            "Q8F16 payload is {} bytes, {blocks} blocks need {}",
            data.len(),
            blocks * Q8_BLOCK_BYTES
        ));
    }
    let mut out = Vec::with_capacity(n);
    for (index, block) in data.chunks_exact(Q8_BLOCK_BYTES).enumerate() {
        let mut scale = f16_to_f64(u16::from_le_bytes([block[0], block[1]]));
        if let Some((mul, first)) = fault {
            if index < first {
                scale *= mul;
            }
        }
        out.extend(block[2..].iter().map(|&q| (q as i8) as f64 * scale));
    }
    Ok(out)
}

fn decode_f16(data: &[u8], n: usize) -> Result<Vec<f64>, String> {
    if data.len() != n * 2 {
        return Err(format!(
            "F16 payload is {} bytes, {n} elements need {}",
            data.len(),
            n * 2
        ));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

struct Layer {
    attn_norm: Vec<f64>,
    wq: Vec<f64>,
    wk: Vec<f64>,
    wv: Vec<f64>,
    wo: Vec<f64>,
    ffn_norm: Vec<f64>,
    gate: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

/// Decoded fixture weights plus the forward pass.
pub struct CpuReference {
    embed: Vec<f64>,
    final_norm: Vec<f64>,
    lm_head: Vec<f64>,
    layers: Vec<Layer>,
}

fn matvec(w: &[f64], x: &[f64], rows: usize) -> Vec<f64> {
    let k = x.len();
    debug_assert_eq!(w.len(), rows * k);
    (0..rows)
        .map(|r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn rms_norm(x: &[f64], weight: &[f64]) -> Vec<f64> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Half-split RoPE of every `head_dim` head of `v` at `pos`.
fn rope(v: &mut [f64], pos: usize) {
    let half = HEAD_DIM / 2;
    for head in v.chunks_exact_mut(HEAD_DIM) {
        for i in 0..half {
            let freq = ROPE_THETA.powf(-((2 * i) as f64) / HEAD_DIM as f64);
            let (sin, cos) = (pos as f64 * freq).sin_cos();
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

impl CpuReference {
    /// Decode the fixture tensors (as written by [`crate::fixture::dense_q8_fixture`]).
    pub fn from_tensors(tensors: &[HfqMemTensor]) -> Result<Self, String> {
        Self::from_tensors_with_fault(tensors, None)
    }

    /// As [`Self::from_tensors`], with an optional planted dequantisation fault.
    pub fn from_tensors_with_fault(
        tensors: &[HfqMemTensor],
        fault: Option<&DequantFault>,
    ) -> Result<Self, String> {
        let find = |name: &str| {
            tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("fixture tensor {name} is missing"))
        };
        let linear = |name: &str, rows: usize, k: usize| -> Result<Vec<f64>, String> {
            let t = find(name)?;
            if t.quant_type != QT_Q8F16 || t.shape != [rows as u32, k as u32] {
                return Err(format!(
                    "{name}: want Q8F16 [{rows}, {k}], found qt {} {:?}",
                    t.quant_type, t.shape
                ));
            }
            let planted = fault
                .filter(|f| f.tensor == name)
                .map(|f| (f.scale_mul, f.blocks));
            decode_q8f16(&t.data, rows * k, planted).map_err(|e| format!("{name}: {e}"))
        };
        let norm = |name: &str| -> Result<Vec<f64>, String> {
            let t = find(name)?;
            if t.quant_type != QT_F16 || t.shape != [DIM as u32] {
                return Err(format!(
                    "{name}: want F16 [{DIM}], found qt {} {:?}",
                    t.quant_type, t.shape
                ));
            }
            decode_f16(&t.data, DIM).map_err(|e| format!("{name}: {e}"))
        };
        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in 0..N_LAYERS {
            let p = |suffix: &str| format!("model.layers.{l}.{suffix}");
            layers.push(Layer {
                attn_norm: norm(&p("input_layernorm.weight"))?,
                wq: linear(&p("self_attn.q_proj.weight"), q_dim, DIM)?,
                wk: linear(&p("self_attn.k_proj.weight"), kv_dim, DIM)?,
                wv: linear(&p("self_attn.v_proj.weight"), kv_dim, DIM)?,
                wo: linear(&p("self_attn.o_proj.weight"), DIM, q_dim)?,
                ffn_norm: norm(&p("post_attention_layernorm.weight"))?,
                gate: linear(&p("mlp.gate_proj.weight"), INTERMEDIATE, DIM)?,
                up: linear(&p("mlp.up_proj.weight"), INTERMEDIATE, DIM)?,
                down: linear(&p("mlp.down_proj.weight"), DIM, INTERMEDIATE)?,
            });
        }
        Ok(Self {
            embed: linear("model.embed_tokens.weight", VOCAB, DIM)?,
            final_norm: norm("model.norm.weight")?,
            lm_head: linear("lm_head.weight", VOCAB, DIM)?,
            layers,
        })
    }

    /// The decoded embedding row of `token`: what the engine's embedding lookup
    /// must produce bit for bit (when narrowed to `f32`).
    pub fn embed_row(&self, token: u32) -> &[f64] {
        &self.embed[token as usize * DIM..(token as usize + 1) * DIM]
    }

    /// Teacher-forced logits: `result[i]` is the logit row after consuming
    /// `tokens[..=i]`.
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f64>> {
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let group = N_HEADS / N_KV_HEADS;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        // keys[layer][pos], values[layer][pos], each `kv_dim` long, post-RoPE for keys.
        let mut keys: Vec<Vec<Vec<f64>>> = vec![Vec::new(); N_LAYERS];
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); N_LAYERS];
        let mut out = Vec::with_capacity(tokens.len());
        for (pos, &token) in tokens.iter().enumerate() {
            let mut x = self.embed_row(token).to_vec();
            for (l, layer) in self.layers.iter().enumerate() {
                let h = rms_norm(&x, &layer.attn_norm);
                let mut q = matvec(&layer.wq, &h, N_HEADS * HEAD_DIM);
                let mut k = matvec(&layer.wk, &h, kv_dim);
                let v = matvec(&layer.wv, &h, kv_dim);
                rope(&mut q, pos);
                rope(&mut k, pos);
                keys[l].push(k);
                values[l].push(v);
                let mut attn = vec![0.0f64; N_HEADS * HEAD_DIM];
                for head in 0..N_HEADS {
                    let kv_head = head / group;
                    let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                    let scores: Vec<f64> = keys[l]
                        .iter()
                        .map(|key| {
                            let kh = &key[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
                            qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f64>() * scale
                        })
                        .collect();
                    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    let dst = &mut attn[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                    for (w, value) in weights.iter().zip(&values[l]) {
                        let vh = &value[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
                        for (d, vv) in dst.iter_mut().zip(vh) {
                            *d += w / total * vv;
                        }
                    }
                }
                let o = matvec(&layer.wo, &attn, DIM);
                for (xi, oi) in x.iter_mut().zip(&o) {
                    *xi += oi;
                }
                let h = rms_norm(&x, &layer.ffn_norm);
                let gate = matvec(&layer.gate, &h, INTERMEDIATE);
                let up = matvec(&layer.up, &h, INTERMEDIATE);
                let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
                let down = matvec(&layer.down, &act, DIM);
                for (xi, di) in x.iter_mut().zip(&down) {
                    *xi += di;
                }
            }
            let h = rms_norm(&x, &self.final_norm);
            out.push(matvec(&self.lm_head, &h, VOCAB));
        }
        out
    }

    /// Greedy continuation: `n` tokens after `prompt`, each the argmax of the
    /// reference's own logits (first index wins ties, as the engine's argmax).
    pub fn generate_greedy(&self, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut tokens = prompt.to_vec();
        for _ in 0..n {
            let logits = self.forward(&tokens);
            tokens.push(argmax(logits.last().expect("non-empty prompt")) as u32);
        }
        tokens[prompt.len()..].to_vec()
    }
}

/// Index of the largest value; the first wins ties.
pub fn argmax<T: Copy + PartialOrd>(values: &[T]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Gap between the largest and second-largest logit.
pub fn top2_margin(logits: &[f64]) -> f64 {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[0] - sorted[1]
}

fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_total = logits.iter().map(|l| (l - peak).exp()).sum::<f64>().ln() + peak;
    logits.iter().map(|l| l - log_total).collect()
}

/// `KL(reference || candidate)` between the two logit rows' softmax
/// distributions, in nats, over the whole vocabulary.
pub fn kld(reference: &[f64], candidate: &[f64]) -> f64 {
    assert_eq!(reference.len(), candidate.len());
    let lp = log_softmax(reference);
    let lq = log_softmax(candidate);
    lp.iter()
        .zip(&lq)
        .map(|(p, q)| p.exp() * (p - q))
        .sum::<f64>()
        .max(0.0)
}

/// Logit-row comparison of one engine arm against the reference.
#[derive(Clone, Debug, PartialEq)]
pub struct LogitComparison {
    pub positions: usize,
    pub max_abs_logit_diff: f64,
    pub mean_abs_logit_diff: f64,
    pub max_kld: f64,
    pub mean_kld: f64,
    /// Positions whose engine argmax differs from the reference argmax.
    pub argmax_flips: usize,
    /// Of those, positions whose reference top-2 margin is above
    /// [`SEPARATED_MARGIN`] (a flip there is an error, not a near-tie).
    pub separated_flips: usize,
    /// Smallest reference top-2 margin among the compared positions.
    pub min_margin: f64,
    pub max_margin: f64,
}

/// A reference margin at or above this is a well-separated pick: the engine
/// must agree on it exactly. It is twice [`BUDGET`]'s `max_abs_logit_diff`, so
/// a row inside the budget cannot flip it.
pub const SEPARATED_MARGIN: f64 = 2.0 * BUDGET.max_abs_logit_diff;

/// Compare engine logit rows with reference rows, position by position.
pub fn compare_logits(reference: &[Vec<f64>], engine: &[Vec<f32>]) -> LogitComparison {
    assert_eq!(reference.len(), engine.len(), "row count");
    assert!(!reference.is_empty());
    let mut c = LogitComparison {
        positions: reference.len(),
        max_abs_logit_diff: 0.0,
        mean_abs_logit_diff: 0.0,
        max_kld: 0.0,
        mean_kld: 0.0,
        argmax_flips: 0,
        separated_flips: 0,
        min_margin: f64::INFINITY,
        max_margin: 0.0,
    };
    let mut abs_sum = 0.0;
    let mut elements = 0usize;
    for (r, e) in reference.iter().zip(engine) {
        assert_eq!(r.len(), e.len(), "vocab width");
        let e64: Vec<f64> = e.iter().map(|&v| v as f64).collect();
        for (a, b) in r.iter().zip(&e64) {
            let d = (a - b).abs();
            c.max_abs_logit_diff = c.max_abs_logit_diff.max(d);
            abs_sum += d;
            elements += 1;
        }
        let k = kld(r, &e64);
        c.max_kld = c.max_kld.max(k);
        c.mean_kld += k / reference.len() as f64;
        let margin = top2_margin(r);
        c.min_margin = c.min_margin.min(margin);
        c.max_margin = c.max_margin.max(margin);
        if argmax(r) != argmax(&e64) {
            c.argmax_flips += 1;
            if margin >= SEPARATED_MARGIN {
                c.separated_flips += 1;
            }
        }
    }
    c.mean_abs_logit_diff = abs_sum / elements as f64;
    c
}

/// The frozen budget a GPU-vs-reference row is judged against. Never loosened
/// to fit a run: a change here is a reviewed change with a new measurement.
#[derive(Clone, Copy, Debug)]
pub struct LogitBudget {
    pub max_abs_logit_diff: f64,
    pub max_kld: f64,
    pub mean_kld: f64,
}

/// Frozen on 2026-11-24, before any comparison on [`crate::fixture::PROMPTS`],
/// from a gfx1201 pilot (`B2R_PROMPT_SET=pilot`, four prompts of 6, 9, 16 and
/// 23 tokens disjoint from `PROMPTS`, 54 decode positions). Pilot worst cases:
/// max |dlogit| 0.0778, max KLD 3.36e-4, per-prompt mean KLD 1.53e-4. Each
/// bound is about twice the worst pilot value: engine noise passes, a 2%
/// scale error on the `lm_head` or 5% on a mid-layer projection does not
/// (pilot fault sweep: KLD 1.1e-3 and 2.1e-3).
/// Never loosened to fit a run: a change here needs a new measurement.
pub const BUDGET: LogitBudget = LogitBudget {
    max_abs_logit_diff: 0.16,
    max_kld: 7.0e-4,
    mean_kld: 3.0e-4,
};

impl LogitBudget {
    /// `Ok` when `c` is inside the budget, else every violated bound.
    pub fn check(&self, c: &LogitComparison) -> Result<(), String> {
        let mut broken = Vec::new();
        if c.max_abs_logit_diff > self.max_abs_logit_diff {
            broken.push(format!(
                "max |dlogit| {:.3e} > {:.3e}",
                c.max_abs_logit_diff, self.max_abs_logit_diff
            ));
        }
        if c.max_kld > self.max_kld {
            broken.push(format!("max KLD {:.3e} > {:.3e}", c.max_kld, self.max_kld));
        }
        if c.mean_kld > self.mean_kld {
            broken.push(format!(
                "mean KLD {:.3e} > {:.3e}",
                c.mean_kld, self.mean_kld
            ));
        }
        if c.separated_flips > 0 {
            broken.push(format!(
                "{} argmax flip(s) at margin >= {SEPARATED_MARGIN}",
                c.separated_flips
            ));
        }
        if broken.is_empty() {
            Ok(())
        } else {
            Err(broken.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{dense_q8_fixture, PROMPTS};

    fn reference() -> CpuReference {
        CpuReference::from_tensors(&dense_q8_fixture().tensors).expect("decode fixture")
    }

    #[test]
    fn f16_decode_matches_known_values() {
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0xc000), -2.0);
        assert_eq!(f16_to_f64(0x3800), 0.5);
        assert_eq!(f16_to_f64(0x0001), 2f64.powi(-24));
        assert_eq!(f16_to_f64(0x7bff), 65504.0);
        assert!(f16_to_f64(0x7e00).is_nan());
    }

    #[test]
    fn q8_decode_is_scale_times_quant_per_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        data.extend((0..32).map(|i| (i as i8 - 16) as u8));
        data.extend_from_slice(&0x4000u16.to_le_bytes()); // 2.0
        data.extend((0..32).map(|_| (-128i8) as u8));
        let out = decode_q8f16(&data, 64, None).unwrap();
        assert_eq!(out[0], -8.0);
        assert_eq!(out[31], 7.5);
        assert_eq!(out[32], -256.0);
        assert_eq!(out[63], -256.0);
        assert!(decode_q8f16(&data, 63, None).is_err());
        assert!(decode_q8f16(&data[..67], 64, None).is_err());
        let faulted = decode_q8f16(&data, 64, Some((4.0, 1))).unwrap();
        assert_eq!(faulted[0], -32.0);
        assert_eq!(faulted[32], -256.0, "fault stops after `blocks` blocks");
    }

    #[test]
    fn fixture_logits_are_finite_nonconstant_and_causal() {
        let r = reference();
        let prompt = PROMPTS[2];
        let full = r.forward(prompt);
        assert_eq!(full.len(), prompt.len());
        for row in &full {
            assert!(row.iter().all(|v| v.is_finite()));
            let (lo, hi) = row
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
            assert!(hi - lo > 1.0, "logit row is nearly constant: {lo}..{hi}");
        }
        assert_ne!(full[0], full[1], "positions must differ");
        // Causality: a prefix's rows do not depend on later tokens.
        let prefix = r.forward(&prompt[..7]);
        assert_eq!(&full[..7], &prefix[..]);
    }

    #[test]
    fn prompt_set_has_near_tie_and_separated_picks() {
        let r = reference();
        let mut margins = Vec::new();
        for prompt in PROMPTS {
            margins.extend(r.forward(prompt).iter().map(|row| top2_margin(row)));
        }
        eprintln!("top-2 margins over PROMPTS: {margins:.3?}");
        let near_tie = margins.iter().filter(|&&m| m < 0.15).count();
        let separated = margins.iter().filter(|&&m| m > 1.0).count();
        assert!(near_tie >= 1, "no near-tie pick in {margins:?}");
        assert!(
            separated >= 3,
            "too few well-separated picks in {margins:?}"
        );
    }

    #[test]
    fn greedy_generation_is_stable_and_extends_the_prompt() {
        let r = reference();
        let a = r.generate_greedy(PROMPTS[0], 6);
        let b = r.generate_greedy(PROMPTS[0], 6);
        assert_eq!(a, b);
        assert_eq!(a.len(), 6);
        assert!(a.iter().all(|&t| (t as usize) < VOCAB));
    }

    #[test]
    fn kld_is_zero_for_equal_rows_and_positive_otherwise() {
        let row = vec![0.5, 1.5, -2.0, 0.0];
        assert_eq!(kld(&row, &row), 0.0);
        let other = vec![0.5, 1.4, -2.0, 0.0];
        assert!(kld(&row, &other) > 0.0);
        // Shift invariance: softmax ignores a constant offset.
        let shifted: Vec<f64> = row.iter().map(|v| v + 3.0).collect();
        assert!(kld(&row, &shifted) < 1e-12);
    }

    #[test]
    fn wrong_dequant_scale_is_visible_in_the_logits() {
        let fixture = dense_q8_fixture();
        let good = CpuReference::from_tensors(&fixture.tensors).unwrap();
        let prompt = PROMPTS[1];
        let want = good.forward(prompt);
        for (tensor, mul, blocks) in [
            ("model.layers.0.self_attn.q_proj.weight", 2.0, usize::MAX),
            ("model.layers.1.mlp.down_proj.weight", 1.05, usize::MAX),
            ("model.layers.0.mlp.gate_proj.weight", 0.5, 64),
            ("lm_head.weight", 1.02, usize::MAX),
        ] {
            let bad = CpuReference::from_tensors_with_fault(
                &fixture.tensors,
                Some(&DequantFault {
                    tensor: tensor.into(),
                    scale_mul: mul,
                    blocks,
                }),
            )
            .unwrap();
            let got: Vec<Vec<f32>> = bad
                .forward(prompt)
                .iter()
                .map(|row| row.iter().map(|&v| v as f32).collect())
                .collect();
            let c = compare_logits(&want, &got);
            assert!(c.max_abs_logit_diff > 0.0, "{tensor} x{mul}: no effect");
        }
    }
}
