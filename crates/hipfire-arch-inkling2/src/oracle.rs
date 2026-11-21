// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU oracle helpers mirroring the scalar parts of llama.cpp's
//! `inkling2.cpp` graph. These are correctness references, not GPU kernels.

/// RMSNorm with no learned gain (all Inkling-2 norms are gamma-less).
pub fn rmsnorm(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * scale).collect()
}

/// Row L2 clip to `threshold`, after an optional per-channel scale.
/// Mirrors `llama_model_inkling2::graph::clip`: eps is added to the norm and
/// the factor `threshold / norm` is clamped to [0, 1].
pub fn clip(cur: &[f32], scale: Option<&[f32]>, threshold: f32) -> Vec<f32> {
    let scaled: Vec<f32> = match scale {
        Some(s) => cur.iter().zip(s.iter()).map(|(v, k)| v * k).collect(),
        None => cur.to_vec(),
    };
    let norm = scaled.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let factor = (threshold / norm).clamp(0.0, 1.0);
    scaled.iter().map(|v| v * factor).collect()
}

/// Log-scaled position multiplier applied to Q and R on full-attention layers.
/// Mirrors `tau = 1 + alpha * log(max(1, (pos+1)/floor))`.
pub fn tau(pos: usize, floor: usize, alpha: f32) -> f32 {
    let ratio = ((pos + 1) as f64 / floor as f64).max(1.0);
    1.0 + alpha * ratio.ln() as f32
}

/// Row-major matrix-vector product: `w` is `[rows, cols]`, `x` is `[cols]`.
pub fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

/// One loop-controller step from `inkling2.cpp::graph::control`:
/// `out = cur - tanh(F·h)·cur + tanh(G·h)·msg` where
/// `h = RMSNorm(cur,1e-6) ‖ RMSNorm(msg,1e-6)`.
///
/// `down_w` is `[rank, 2*n_embd]`, `gain_w` and `forget_w` are `[n_embd, rank]`.
pub fn control(
    cur: &[f32],
    msg: &[f32],
    down_w: &[f32],
    gain_w: &[f32],
    forget_w: &[f32],
    n_embd: usize,
    rank: usize,
) -> Vec<f32> {
    let cur_n = rmsnorm(cur, 1e-6);
    let msg_n = rmsnorm(msg, 1e-6);
    let mut gate = Vec::with_capacity(2 * n_embd);
    gate.extend_from_slice(&cur_n);
    gate.extend_from_slice(&msg_n);
    let g = matvec(down_w, &gate, rank, 2 * n_embd);
    let gain = matvec(gain_w, &g, n_embd, rank);
    let forget = matvec(forget_w, &g, n_embd, rank);
    (0..n_embd)
        .map(|i| {
            let f = forget[i].tanh();
            let a = gain[i].tanh();
            cur[i] - f * cur[i] + a * msg[i]
        })
        .collect()
}

/// `inkling2.cpp::graph::mix`: RMSNorm each state, score it against each
/// learned query at `1/sqrt(n_embd)`, softmax over states, then take the
/// weighted sum. `blocks` are the earlier block outputs (oldest first), `cur`
/// is appended last; each state is `[n_embd]`. Returns `[query_rows, n_embd]`.
pub fn mix(
    blocks: &[Vec<f32>],
    cur: &[f32],
    query: &[f32],
    n_embd: usize,
    query_rows: usize,
) -> Vec<f32> {
    let mut states: Vec<Vec<f32>> = blocks.iter().map(|b| rmsnorm(b, 1e-6)).collect();
    states.push(rmsnorm(cur, 1e-6));
    let ns = states.len();
    let scale = 1.0 / (n_embd as f32).sqrt();
    let mut out = vec![0.0f32; query_rows * n_embd];
    for b in 0..query_rows {
        let mut scores = Vec::with_capacity(ns);
        for s in &states {
            let dot: f32 = (0..n_embd).map(|i| s[i] * query[b * n_embd + i]).sum();
            scores.push(dot * scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (idx, e) in exps.iter().enumerate() {
            let p = e / sum;
            for i in 0..n_embd {
                out[b * n_embd + i] += states[idx][i] * p;
            }
        }
    }
    out
}

/// Static (input-independent) causal short convolution with the residual and
/// post-scale from `inkling2.cpp::graph::convolve`, for one token / many channels.
/// `cur` is `[n_seq, n_channels]`, `weight` is `[n_conv, n_channels]`, `past`
/// is `[n_conv-1, n_channels]`. Returns the output and the new past window.
pub fn convolve_static(
    cur: &[f32],
    weight: &[f32],
    past: &[f32],
    n_seq: usize,
    n_channels: usize,
    n_conv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0.0f32; (n_seq + n_conv - 1) * n_channels];
    let history = (n_conv - 1) * n_channels;
    x[..history].copy_from_slice(&past[..history]);
    x[history..].copy_from_slice(&cur[..n_seq * n_channels]);
    let mut conv = vec![0.0f32; n_seq * n_channels];
    for t in 0..n_seq {
        for ch in 0..n_channels {
            let mut acc = 0.0f32;
            for l in 0..n_conv {
                let xi = (t + n_conv - 1 - l) * n_channels + ch;
                acc += x[xi] * weight[l * n_channels + ch];
            }
            conv[t * n_channels + ch] = acc;
        }
    }
    let scale = 1.0 / ((n_conv + 1) as f32).sqrt();
    for i in 0..conv.len() {
        conv[i] = (cur[i] + conv[i]) * scale;
    }
    let new_past = x[n_seq * n_channels..].to_vec();
    (conv, new_past)
}

/// Relative-position logits for one query token, mirroring `inkling2.cpp`:
/// R is projected to `n_rel` per head, multiplied by the `[n_rel, n_extent]`
/// bias table, the per-head sink logit is subtracted, and the result is
/// gathered by distance. Returns `[n_heads, n_kv]`.
pub fn relative_bias(
    rel: &[f32],
    table: &[f32],
    sink: &[f32],
    distances: &[usize],
    n_rel: usize,
    n_heads: usize,
    n_extent: usize,
) -> Vec<f32> {
    let n_kv = distances.len();
    let mut out = vec![0.0f32; n_heads * n_kv];
    for h in 0..n_heads {
        for (k, &d) in distances.iter().enumerate() {
            assert!(d < n_extent, "relative distance {d} out of range");
            let mut v = 0.0f32;
            for r in 0..n_rel {
                v += table[r * n_extent + d] * rel[r * n_heads + h];
            }
            out[h * n_kv + k] = v - sink[h];
        }
    }
    out
}

/// Softmax attention with a zero-value sink logit, mirroring both
/// `eager_attention_forward` (concat `[scores, sink]`, softmax, drop the sink
/// column) and llama.cpp (subtract sink from the bias, pass a zero sink to
/// `build_attn`). `logits` is `[n_kv]` (= qk + relative bias, already scaled),
/// `values` is `[n_kv, n_embd_v]`. Returns `[n_embd_v]`.
pub fn attention_softmax_with_sink(
    logits: &[f32],
    sink: f32,
    values: &[f32],
    n_kv: usize,
    n_embd_v: usize,
) -> Vec<f32> {
    let max = logits.iter().copied().fold(sink, f32::max);
    let mut denom = (sink - max).exp();
    let mut probs = Vec::with_capacity(n_kv);
    for &l in logits.iter().take(n_kv) {
        let e = (l - max).exp();
        probs.push(e);
        denom += e;
    }
    let mut out = vec![0.0f32; n_embd_v];
    for (j, &p) in probs.iter().enumerate() {
        let w = p / denom;
        for i in 0..n_embd_v {
            out[i] += w * values[j * n_embd_v + i];
        }
    }
    out
}

/// `situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)`
/// from `llama-graph.cpp::LLM_FFN_SITU`. `linear_beta <= 0` leaves `up` raw.
pub fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let act = beta * (gate / beta).tanh() * sigmoid(gate);
    let up_term = if linear_beta > 0.0 {
        linear_beta * (up / linear_beta).tanh()
    } else {
        up
    };
    act * up_term
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid router with optional per-expert score-correction bias, top-k by
/// `sigmoid(logit) + bias`. The returned weights are the UNBIASED sigmoid
/// scores of the selected experts, **renormalized by their sum (eps 1e-20) and
/// scaled by `scale` (= route_scale, 8.0)** — matching llama.cpp
/// `llama-graph.cpp::build_moe_ffn` (`expert_weights_norm_eps = 1e-20`) and
/// HF `Inkling2TopKRouter`. Returns `(expert_index, weight)` sorted by
/// descending selection score.
pub fn sigmoid_topk(
    logits: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    scale: f32,
) -> Vec<(usize, f32)> {
    let probs: Vec<f32> = logits.iter().map(|&s| sigmoid(s)).collect();
    let mut selection: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p + bias.map_or(0.0, |v| v[i])))
        .collect();
    selection.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    selection.truncate(k);
    let sum: f32 = selection.iter().map(|&(i, _)| probs[i]).sum::<f32>() + 1e-20;
    selection
        .into_iter()
        .map(|(i, _)| (i, probs[i] / sum * scale))
        .collect()
}

/// Final logit softcap `cap * tanh(x / cap)`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn rmsnorm_is_gamma_less() {
        // RMS = sqrt((9 + 16)/2) = 3.5355..., not the L2 norm 5.
        let out = rmsnorm(&[3.0, 4.0], 1e-6);
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 0.8485281), "{}", out[0]);
        assert!(close(out[1], 1.1313708), "{}", out[1]);
    }

    #[test]
    fn clip_scales_down_only_when_above_threshold() {
        let out = clip(&[3.0, 4.0], None, 2.0);
        assert!(close(out[0], 1.2), "{}", out[0]);
        assert!(close(out[1], 1.6), "{}", out[1]);
        let out = clip(&[3.0, 4.0], None, 10.0);
        assert!(close(out[0], 3.0) && close(out[1], 4.0));
    }

    #[test]
    fn clip_applies_optional_channel_scale_first() {
        let out = clip(&[3.0, 4.0], Some(&[2.0, 0.5]), 10.0);
        assert!(close(out[0], 6.0), "{}", out[0]);
        assert!(close(out[1], 2.0), "{}", out[1]);
    }

    #[test]
    fn tau_clamps_floor_and_scales_log() {
        assert_eq!(tau(0, 8192, 0.1), 1.0);
        assert_eq!(tau(8191, 8192, 0.1), 1.0);
        let t = tau(16383, 8192, 0.1);
        assert!(close(t, 1.0 + 0.1 * 2.0f32.ln()), "{}", t);
    }

    #[test]
    fn matvec_is_row_major() {
        let out = matvec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0, 1.0, 1.0], 2, 3);
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn control_zero_gates_are_identity() {
        let cur = [3.0, 4.0];
        let out = control(&cur, &[9.0, 9.0], &[0.0; 4], &[0.0, 0.0], &[0.0, 0.0], 2, 1);
        assert_eq!(out, cur.to_vec());
    }

    #[test]
    fn control_gain_blends_message_and_forget_drops_current() {
        let cur = [3.0, 4.0];
        let msg = [4.0, 3.0];
        // rank 1: down gate[0] -> g[0]; gain[0] reads g[0], forget[1] reads g[0].
        let out = control(
            &cur,
            &msg,
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            2,
            1,
        );
        let g0 = rmsnorm(&cur, 1e-6)[0];
        let expected0 = cur[0] + g0.tanh() * msg[0];
        let expected1 = cur[1] - g0.tanh() * cur[1];
        assert!(close(out[0], expected0), "{} vs {}", out[0], expected0);
        assert!(close(out[1], expected1), "{} vs {}", out[1], expected1);
    }

    #[test]
    fn mix_single_state_is_its_normalized_value() {
        let out = mix(&[], &[3.0, 4.0], &[1.0, 0.0], 2, 1);
        let want = rmsnorm(&[3.0, 4.0], 1e-6);
        assert!(close(out[0], want[0]) && close(out[1], want[1]));
    }

    #[test]
    fn mix_equal_states_split_evenly() {
        let out = mix(&[vec![1.0, 1.0]], &[1.0, 1.0], &[1.0, 1.0], 2, 1);
        assert!(close(out[0], 1.0) && close(out[1], 1.0), "{out:?}");
    }

    #[test]
    fn convolve_static_uses_past_and_returns_new_window() {
        let (out, past) = convolve_static(&[1.0], &[0.5, 0.25], &[2.0], 1, 1, 2);
        assert!(
            close(out[0], (1.0 + (1.0 * 0.5 + 2.0 * 0.25)) / 3.0f32.sqrt()),
            "{}",
            out[0]
        );
        assert_eq!(past, vec![1.0]);
    }

    #[test]
    fn convolve_static_single_tap() {
        let (out, past) = convolve_static(&[2.0], &[0.5], &[], 1, 1, 1);
        assert!(
            close(out[0], (2.0 + 2.0 * 0.5) / 2.0f32.sqrt()),
            "{}",
            out[0]
        );
        assert!(past.is_empty());
    }

    #[test]
    fn attention_sink_absorbs_probability_mass() {
        // One KV row with value 10, sink logit = 0, score logit = 0.
        // p_kv = 1/(1+1) = 0.5, so output = 5. A huge sink drives it to 0.
        let out = attention_softmax_with_sink(&[0.0], 0.0, &[10.0], 1, 1);
        assert!(close(out[0], 5.0), "{}", out[0]);
        let out = attention_softmax_with_sink(&[0.0], 1e6, &[10.0], 1, 1);
        assert!(close(out[0], 0.0), "huge sink: {}", out[0]);
        let out = attention_softmax_with_sink(&[0.0], -1e6, &[10.0], 1, 1);
        assert!(close(out[0], 10.0), "tiny sink: {}", out[0]);
    }

    #[test]
    fn situ_matches_source_formula() {
        let got = situ(1.0, 2.0, 4.0, 25.0);
        let want = 4.0 * 0.25f32.tanh() * sigmoid(1.0) * 25.0 * (2.0f32 / 25.0f32).tanh();
        assert!(close(got, want), "{} vs {}", got, want);
        assert_eq!(
            situ(1.0, 2.0, 4.0, 0.0),
            4.0 * 0.25f32.tanh() * sigmoid(1.0) * 2.0
        );
    }

    #[test]
    fn sigmoid_topk_renorms_then_scales() {
        let out = sigmoid_topk(&[1.0, 2.0, 3.0], None, 2, 8.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 2);
        assert_eq!(out[1].0, 1);
        // weights = sigmoid(logit) / sum(top-8 sigmoids) * 8; they sum to 8.
        let sum: f32 = out.iter().map(|x| x.1).sum();
        assert!(
            close(sum, 8.0),
            "renorm-then-scale must sum to route_scale: {sum}"
        );
        let p3 = sigmoid(3.0);
        let p2 = sigmoid(2.0);
        assert!(close(out[0].1, p3 / (p3 + p2) * 8.0));
        assert!(close(out[1].1, p2 / (p3 + p2) * 8.0));
    }

    #[test]
    fn sigmoid_topk_uses_bias_only_for_selection() {
        // A large bias on expert 0 makes it selected, but its weight is the
        // unbiased sigmoid, not sigmoid+bias.
        let out = sigmoid_topk(&[1.0, 2.0], Some(&[2.0, 0.0]), 1, 8.0);
        assert_eq!(out[0].0, 0);
        assert!(
            close(out[0].1, 8.0),
            "single selected expert weight = full scale"
        );
    }

    #[test]
    fn softcap_is_cap_times_tanh() {
        assert!(close(softcap(30.0, 30.0), 30.0 * 1.0f32.tanh()));
    }

    #[test]
    fn relative_bias_subtracts_sink_before_gather() {
        // table rows [1,2,3,4] and [5,6,7,8]; R = [2,3]; sink = 1.
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = relative_bias(&[2.0, 3.0], &table, &[1.0], &[0, 2], 2, 1, 4);
        // d=0: 1*2 + 5*3 - 1 = 16; d=2: 3*2 + 7*3 - 1 = 26.
        assert_eq!(out, vec![16.0, 26.0]);
    }
}
