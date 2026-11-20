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
}
