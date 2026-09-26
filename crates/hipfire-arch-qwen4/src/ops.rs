// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Equation-level Qwen4 operations.
//!
//! These helpers are deliberately CPU-testable and mirror the operand shapes
//! used by the ordinary HIP path.  They contain no source I/O and no scheduler
//! policy.  GPU lowering consumes the same shape/ordering contracts through the
//! Qwen4-specific dispatch and compute modules.

use crate::state::{ReferenceGdnLayerState, StateError};
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug)]
pub enum OpError {
    Length {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    Shape(&'static str),
    State(StateError),
    Empty,
    InvalidTopK {
        requested: usize,
        available: usize,
    },
    Capacity {
        capacity: usize,
        actual: usize,
    },
    InvalidChunk {
        total: usize,
        chunk: usize,
    },
}

impl From<StateError> for OpError {
    fn from(value: StateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length {
                what,
                expected,
                actual,
            } => write!(f, "{what}: expected {expected}, got {actual}"),
            Self::Shape(what) => write!(f, "invalid Qwen4 shape: {what}"),
            Self::State(error) => write!(f, "Qwen4 state operation: {error}"),
            Self::Empty => write!(f, "Qwen4 operation received an empty input"),
            Self::InvalidTopK {
                requested,
                available,
            } => write!(f, "top-k {requested} exceeds {available}"),
            Self::Capacity { capacity, actual } => {
                write!(f, "capacity {capacity} exceeded by {actual}")
            }
            Self::InvalidChunk { total, chunk } => {
                write!(f, "invalid chunk {chunk} for total {total}")
            }
        }
    }
}

impl std::error::Error for OpError {}

#[inline]
pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

#[inline]
pub fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

/// Grouped RMS normalization used by HC and PLE branches.
///
/// Qwen4 calls this "zero-centered" because its learned scale is stored as a
/// zero-centered parameter: the runtime multiplier is `(1 + weight)`.  There
/// is no mean subtraction in the pinned upstream equation.
pub fn zero_centered_rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), OpError> {
    if input.is_empty() {
        return Err(OpError::Empty);
    }
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(OpError::Length {
            what: "zero-centered RMSNorm",
            expected: input.len(),
            actual: weight.len().max(output.len()),
        });
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_square + epsilon.max(1.0e-12)).sqrt().recip();
    for ((dst, &value), &scale) in output.iter_mut().zip(input).zip(weight) {
        *dst = value * inv_rms * (1.0 + scale);
    }
    Ok(())
}

/// Ordinary RMSNorm with the Qwen4 zero-centered learned-scale convention.
pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), OpError> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(OpError::Length {
            what: "RMSNorm",
            expected: input.len(),
            actual: weight.len().max(output.len()),
        });
    }
    let mean_square = input.iter().map(|v| v * v).sum::<f32>() / input.len().max(1) as f32;
    let inv = (mean_square + epsilon.max(1.0e-12)).sqrt().recip();
    for ((dst, &value), &scale) in output.iter_mut().zip(input).zip(weight) {
        *dst = value * inv * (1.0 + scale);
    }
    Ok(())
}

/// Prepare one HC residual branch: `(1 + w) * zero_centered_norm(x)`.
pub fn hc_prepare(
    input: &[f32],
    branch_weights: &[f32],
    epsilon: f32,
    normalized: &mut [f32],
    branch: &mut [f32],
) -> Result<(), OpError> {
    if input.len() != branch_weights.len()
        || input.len() != normalized.len()
        || input.len() != branch.len()
    {
        return Err(OpError::Length {
            what: "HC prepare",
            expected: input.len(),
            actual: branch_weights.len().max(normalized.len()).max(branch.len()),
        });
    }
    zero_centered_rms_norm(input, branch_weights, epsilon, normalized)?;
    for ((dst, normed), weight) in branch
        .iter_mut()
        .zip(normalized.iter().copied())
        .zip(branch_weights.iter().copied())
    {
        *dst = normed * (1.0 + weight);
    }
    Ok(())
}
/// Exact learned HC read/mix equation for one or more token rows.
///
/// `hyper_input` and `normalized` are flattened `[tokens, branches, hidden]`;
/// `down` is `[rank, branches*hidden]`; `up` is
/// `[branches*hidden, rank]`; `mixed` is `[tokens, hidden]`.
pub fn hc_read(
    hyper_input: &[f32],
    down: &[f32],
    up: &[f32],
    norm_weight: &[f32],
    branches: usize,
    hidden: usize,
    rank: usize,
    epsilon: f32,
    normalized: &mut [f32],
    mixed: &mut [f32],
) -> Result<(), OpError> {
    let wide = branches
        .checked_mul(hidden)
        .ok_or(OpError::Shape("HC width overflow"))?;
    if hyper_input.len() != normalized.len() || hyper_input.len() != wide {
        return Err(OpError::Shape("HC input must be one flattened stream row"));
    }
    if norm_weight.len() != wide
        || down.len() != rank * wide
        || up.len() != wide * rank
        || mixed.len() != hidden
    {
        return Err(OpError::Shape("HC learned projection dimensions"));
    }
    for branch in 0..branches {
        let start = branch * hidden;
        let end = start + hidden;
        let mean_square =
            hyper_input[start..end].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv_rms = (mean_square + epsilon.max(1.0e-12)).sqrt().recip();
        for j in 0..hidden {
            let column = start + j;
            normalized[column] = hyper_input[column] * inv_rms * (1.0 + norm_weight[column]);
        }
    }
    let mut low = vec![0.0f32; rank];
    for (r, value) in low.iter_mut().enumerate() {
        let row = &down[r * wide..(r + 1) * wide];
        *value = silu(
            row.iter()
                .zip(normalized.iter())
                .map(|(&weight, &x)| weight * x)
                .sum::<f32>()
                / branches as f32,
        );
    }
    mixed.fill(0.0);
    for branch in 0..branches {
        for j in 0..hidden {
            let column = branch * hidden + j;
            let gate = (0..rank)
                .map(|r| up[column * rank + r] * low[r])
                .sum::<f32>();
            mixed[j] += sigmoid(gate) * normalized[column];
        }
    }
    for value in mixed {
        *value /= branches as f32;
    }
    Ok(())
}

/// Exact HC write/injection equation.  The learned block-injection matrix is
/// `[branches, branches*hidden]`; the shared mixed hidden row is added to each
/// original stream with `2*sigmoid(dot / branches)`.
pub fn hc_write(
    hyper_input: &[f32],
    normalized: &[f32],
    mixed: &[f32],
    block_inject: &[f32],
    branches: usize,
    hidden: usize,
    output: &mut [f32],
) -> Result<(), OpError> {
    let wide = branches
        .checked_mul(hidden)
        .ok_or(OpError::Shape("HC width overflow"))?;
    if hyper_input.len() != wide
        || normalized.len() != wide
        || output.len() != wide
        || mixed.len() != hidden
        || block_inject.len() != branches * wide
    {
        return Err(OpError::Shape("HC write dimensions"));
    }
    output.copy_from_slice(hyper_input);
    for branch in 0..branches {
        let row = &block_inject[branch * wide..(branch + 1) * wide];
        let gate = 2.0
            * sigmoid(
                row.iter()
                    .zip(normalized)
                    .map(|(&weight, &x)| weight * x)
                    .sum::<f32>()
                    / branches as f32,
            );
        for j in 0..hidden {
            output[branch * hidden + j] += gate * mixed[j];
        }
    }
    Ok(())
}

/// Final learned HC read mixer.  This retains the normalized multi-stream
/// row and returns the collapsed hidden row for the language head.
pub fn hc_final_read(
    hyper_input: &[f32],
    down: &[f32],
    up: &[f32],
    norm_weight: &[f32],
    branches: usize,
    hidden: usize,
    rank: usize,
    epsilon: f32,
    normalized: &mut [f32],
    mixed: &mut [f32],
) -> Result<(), OpError> {
    hc_read(
        hyper_input,
        down,
        up,
        norm_weight,
        branches,
        hidden,
        rank,
        epsilon,
        normalized,
        mixed,
    )
}

/// Inject the HC bottleneck feedback.  The down projection is divided by four,
/// then SiLU, then the up projection is gated by sigmoid before branch means.
pub fn hc_inject(
    branch_input: &[f32],
    down_projection: &[f32],
    up_projection: &[f32],
    block_inject: &[f32],
    branch_count: usize,
    hidden: usize,
    lowrank: usize,
    output: &mut [f32],
) -> Result<(), OpError> {
    if branch_input.len() != hidden || output.len() != hidden {
        return Err(OpError::Length {
            what: "HC inject residual",
            expected: hidden,
            actual: branch_input.len().max(output.len()),
        });
    }
    if down_projection.len() != branch_count * lowrank * hidden {
        return Err(OpError::Shape(
            "HC down projection must be [branches*rank, hidden]",
        ));
    }
    if up_projection.len() != hidden * branch_count * lowrank {
        return Err(OpError::Shape(
            "HC up projection must be [hidden, branches*rank]",
        ));
    }
    if block_inject.len() != branch_count * branch_count * lowrank {
        return Err(OpError::Shape(
            "HC block inject must be [branches, branches*rank]",
        ));
    }
    // The public helper represents one flattened branch.  Keep the operation
    // deterministic and explicit: down/4 -> SiLU -> up -> sigmoid gate.
    let mut bottleneck = vec![0.0f32; branch_count * lowrank];
    for j in 0..branch_count * lowrank {
        let row = &down_projection[j * hidden..(j + 1) * hidden];
        bottleneck[j] = row
            .iter()
            .zip(branch_input)
            .map(|(&w, &x)| w * x)
            .sum::<f32>()
            / 4.0;
        bottleneck[j] = silu(bottleneck[j]);
    }
    let mut projected = vec![0.0f32; hidden];
    for i in 0..hidden {
        let row = &up_projection[i * branch_count * lowrank..(i + 1) * branch_count * lowrank];
        projected[i] = row
            .iter()
            .zip(&bottleneck)
            .map(|(&w, &x)| w * x)
            .sum::<f32>();
    }
    for i in 0..hidden {
        let gate = block_inject[i % block_inject.len()];
        output[i] = branch_input[i] + sigmoid(gate) * projected[i];
    }
    Ok(())
}

/// Final HC mixer: values are averaged across the four branches.  There is no
/// final write projection and no extra RMSNorm in this operation.
pub fn hc_final_mix(branches: &[&[f32]], output: &mut [f32]) -> Result<(), OpError> {
    if branches.is_empty() {
        return Err(OpError::Empty);
    }
    if branches.iter().any(|branch| branch.len() != output.len()) {
        return Err(OpError::Shape("HC final branches must share hidden width"));
    }
    output.fill(0.0);
    let scale = 1.0 / branches.len() as f32;
    for branch in branches {
        for (dst, &value) in output.iter_mut().zip(*branch) {
            *dst += value * scale;
        }
    }
    Ok(())
}

/// Gated DeltaNet recurrence.  Q/K have 16 heads and V has 48 heads; each V
/// head repeats its corresponding Q/K head three times.  `gate` is the
/// already-computed log decay and `beta` is the sigmoid update coefficient.
pub fn gdn_step(
    state: &mut ReferenceGdnLayerState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    output: &mut [f32],
) -> Result<(), OpError> {
    let q_width = state.key_heads * state.key_dim;
    let v_width = state.value_heads * state.value_dim;
    if q.len() != q_width || k.len() != q_width || v.len() != v_width || output.len() != v_width {
        return Err(OpError::Shape(
            "GDN Q/K/V widths do not match 16/48 head geometry",
        ));
    }
    if gate.len() != state.value_heads || beta.len() != state.value_heads {
        return Err(OpError::Shape(
            "GDN gate/beta must have one scalar per value head",
        ));
    }
    let repeat = state.value_heads / state.key_heads;
    let query_scale = (state.key_dim as f32).sqrt().recip();
    for value_head in 0..state.value_heads {
        let key_head = value_head / repeat;
        let q_slice = &q[key_head * state.key_dim..(key_head + 1) * state.key_dim];
        let k_slice = &k[key_head * state.key_dim..(key_head + 1) * state.key_dim];
        let q_norm = l2_norm_eps(q_slice, 1.0e-6);
        let k_norm = l2_norm_eps(k_slice, 1.0e-6);
        let decay = gate[value_head].exp();
        let update = beta[value_head];
        let state_base = value_head * state.value_dim * state.key_dim;
        let value_base = value_head * state.value_dim;
        for idx in 0..state.value_dim * state.key_dim {
            state.recurrent[state_base + idx] *= decay;
        }
        for value_dim in 0..state.value_dim {
            let kv_mem = (0..state.key_dim)
                .map(|key_dim| {
                    state.recurrent[state_base + key_dim * state.value_dim + value_dim]
                        * k_norm[key_dim]
                })
                .sum::<f32>();
            let delta = (v[value_base + value_dim] - kv_mem) * update;
            for key_dim in 0..state.key_dim {
                let idx = state_base + key_dim * state.value_dim + value_dim;
                state.recurrent[idx] += k_norm[key_dim] * delta;
            }
            output[value_base + value_dim] = (0..state.key_dim)
                .map(|key_dim| {
                    state.recurrent[state_base + key_dim * state.value_dim + value_dim]
                        * q_norm[key_dim]
                        * query_scale
                })
                .sum::<f32>();
        }
    }
    Ok(())
}

fn l2_norm_eps(values: &[f32], epsilon: f32) -> Vec<f32> {
    let norm = (values.iter().map(|v| v * v).sum::<f32>() + epsilon).sqrt();
    values.iter().map(|value| value / norm).collect()
}

fn l2_norm(values: &[f32]) -> Vec<f32> {
    l2_norm_eps(values, 1.0e-12)
}

/// Prefix-HalfSplit RoPE.  The rotated half is the first `rotary_dim` values;
/// each half is rotated against its corresponding partner, not interleaved
/// adjacent pairs.
pub fn rope_prefix_halfsplit(
    values: &mut [f32],
    position: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<(), OpError> {
    if rotary_dim % 2 != 0 || rotary_dim > values.len() {
        return Err(OpError::Shape("Prefix-HalfSplit rotary dimension"));
    }
    let half = rotary_dim / 2;
    for i in 0..half {
        let exponent = (2 * i) as f32 / rotary_dim as f32;
        let inv_freq = theta.powf(-exponent);
        let angle = position as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let left = values[i];
        let right = values[i + half];
        values[i] = left * cos - right * sin;
        values[i + half] = left * sin + right * cos;
    }
    Ok(())
}

/// Average a complete indexer block in F32, apply grouped RMS normalization,
/// and apply HalfSplit RoPE at the block-start position.
pub fn qsa_pool_block(
    keys: &[f32],
    block_size: usize,
    block_start: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<Vec<f32>, OpError> {
    if block_size == 0 || keys.len() != block_size * head_dim {
        return Err(OpError::Shape("QSA pool block"));
    }
    let mut pooled = vec![0.0f32; head_dim];
    for row in keys.chunks_exact(head_dim) {
        for (dst, &value) in pooled.iter_mut().zip(row) {
            *dst += value / block_size as f32;
        }
    }
    let mean_square = pooled.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    let inv = (mean_square + 1.0e-6).sqrt().recip();
    for value in &mut pooled {
        *value *= inv;
    }
    rope_prefix_halfsplit(&mut pooled, block_start, head_dim.min(64), rope_theta)?;
    Ok(pooled)
}

/// Score pooled QSA blocks using four summed ReLU query-head dot products and
/// `sqrt(128)` scaling.  Ties are resolved by lower block index.
pub fn qsa_select_blocks(
    query_heads: &[f32],
    pooled_keys: &[f32],
    block_positions: &[usize],
    budget: usize,
    head_dim: usize,
) -> Result<Vec<usize>, OpError> {
    if query_heads.len() != 4 * head_dim {
        return Err(OpError::Shape("QSA indexer query must have four heads"));
    }
    if pooled_keys.len() != block_positions.len() * head_dim {
        return Err(OpError::Shape("QSA pooled keys/positions mismatch"));
    }
    let mut scored = Vec::with_capacity(block_positions.len());
    for (block, &position) in block_positions.iter().enumerate() {
        let key = &pooled_keys[block * head_dim..(block + 1) * head_dim];
        let mut score = 0.0f32;
        for head in 0..4 {
            let query = &query_heads[head * head_dim..(head + 1) * head_dim];
            let dot = query.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>();
            score += dot.max(0.0);
        }
        scored.push((score / (128.0f32).sqrt(), position));
    }
    scored.sort_by(|(left_score, left_pos), (right_score, right_pos)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_pos.cmp(right_pos))
    });
    Ok(scored
        .into_iter()
        .take(budget.min(block_positions.len()))
        .map(|(_, position)| position)
        .collect())
}

/// Append incomplete-tail token positions after block selection.  Capacity is
/// exactly `budget + compress_ratio - 1` for the pinned Qwen4 geometry.
pub fn qsa_append_incomplete_tail(
    selected: &mut Vec<usize>,
    partial_positions: &[usize],
    budget: usize,
    compress_ratio: usize,
) -> Result<(), OpError> {
    let capacity =
        budget
            .checked_add(compress_ratio.saturating_sub(1))
            .ok_or(OpError::Capacity {
                capacity: usize::MAX,
                actual: selected.len(),
            })?;
    selected.extend_from_slice(partial_positions);
    if selected.len() > capacity {
        return Err(OpError::Capacity {
            capacity,
            actual: selected.len(),
        });
    }
    Ok(())
}

/// Full QSA attention over persistent K/V.  Q heads share each KV head in a
/// 24:2 ratio; the attention scale is `1/sqrt(256)`.  `q_with_gate` stores
/// interleaved Q and gate halves per head.
pub fn qsa_attention(
    q_with_gate: &[f32],
    full_keys: &[f32],
    full_values: &[f32],
    selected_indices: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) -> Result<(), OpError> {
    let q_width = n_heads * head_dim;
    if q_with_gate.len() != q_width * 2 || output.len() != q_width {
        return Err(OpError::Shape("QSA interleaved Q/gate width"));
    }
    let kv_width = n_kv_heads * head_dim;
    if full_keys.len() % kv_width != 0 || full_values.len() != full_keys.len() {
        return Err(OpError::Shape("QSA persistent K/V width"));
    }
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(OpError::Shape("QSA 24:2 head grouping"));
    }
    output.fill(0.0);
    let scale = 1.0 / (head_dim as f32).sqrt();
    for head in 0..n_heads {
        let kv_head = head / (n_heads / n_kv_heads);
        let q_off = head * head_dim * 2;
        let q = &q_with_gate[q_off..q_off + head_dim];
        let gate = &q_with_gate[q_off + head_dim..q_off + 2 * head_dim];
        let mut scores = Vec::with_capacity(selected_indices.len());
        let mut max_score = f32::NEG_INFINITY;
        for &index in selected_indices {
            let key_off = index * kv_width + kv_head * head_dim;
            if key_off + head_dim > full_keys.len() {
                return Err(OpError::Shape("QSA selected index outside persistent K/V"));
            }
            let score = q
                .iter()
                .zip(&full_keys[key_off..key_off + head_dim])
                .map(|(&qv, &kv)| qv * kv)
                .sum::<f32>()
                * scale;
            max_score = max_score.max(score);
            scores.push((index, score));
        }
        if scores.is_empty() {
            continue;
        }
        let normalizer = scores
            .iter()
            .map(|(_, score)| (*score - max_score).exp())
            .sum::<f32>()
            .max(1.0e-12);
        let out = &mut output[head * head_dim..(head + 1) * head_dim];
        for ((index, _), weight) in scores.iter().zip(
            scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp() / normalizer),
        ) {
            let value_off = index * kv_width + kv_head * head_dim;
            for (channel, (dst, &value)) in out
                .iter_mut()
                .zip(&full_values[value_off..value_off + head_dim])
                .enumerate()
            {
                *dst += weight * sigmoid(gate[channel]) * value;
            }
        }
    }
    Ok(())
}

/// Stable float32 softmax/top-10 route used by the CPU contract and as the
/// reference for the sealed HIP executor.
pub fn route_top10(logits: &[f32]) -> Result<([usize; 10], [f32; 10]), OpError> {
    if logits.len() < 10 {
        return Err(OpError::InvalidTopK {
            requested: 10,
            available: logits.len(),
        });
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = logits
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, (value - max_logit).exp()))
        .collect::<Vec<_>>();
    probs.sort_by(|(left_i, left_v), (right_i, right_v)| {
        right_v
            .partial_cmp(left_v)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_i.cmp(right_i))
    });
    let normalizer = probs
        .iter()
        .take(10)
        .map(|(_, value)| *value)
        .sum::<f32>()
        .max(1.0e-12);
    let mut indices = [0usize; 10];
    let mut weights = [0.0f32; 10];
    for (slot, (index, value)) in probs.into_iter().take(10).enumerate() {
        indices[slot] = index;
        weights[slot] = value / normalizer;
    }
    Ok((indices, weights))
}

/// Expand one routed expert's gated output and add it exactly once with the
/// normalized route weight.  Shared-expert addition is kept separate so the
/// caller cannot accidentally apply route weights twice.
pub fn combine_routed_top10(
    weights: &[f32; 10],
    expert_outputs: &[&[f32]],
    shared_output: &[f32],
    shared_gate: f32,
    output: &mut [f32],
) -> Result<(), OpError> {
    if expert_outputs.len() != 10 || output.len() != shared_output.len() {
        return Err(OpError::Shape("top-10 combine dimensions"));
    }
    if expert_outputs
        .iter()
        .any(|value| value.len() != output.len())
    {
        return Err(OpError::Shape("top-10 expert output width"));
    }
    output.fill(0.0);
    for (weight, expert) in weights.iter().zip(expert_outputs) {
        for (dst, &value) in output.iter_mut().zip(*expert) {
            *dst += *weight * value;
        }
    }
    let gate = sigmoid(shared_gate);
    for (dst, &value) in output.iter_mut().zip(shared_output) {
        *dst += gate * value;
    }
    Ok(())
}

/// Exact chunk tiling helper used by bounded prefill.  It refuses zero-sized
/// chunks and never silently switches to an unbounded/per-token fallback.
pub fn prefill_chunks(total: usize, chunk: usize) -> Result<Vec<(usize, usize)>, OpError> {
    if total == 0 {
        return Ok(Vec::new());
    }
    if chunk == 0 {
        return Err(OpError::InvalidChunk { total, chunk });
    }
    let mut ranges = Vec::with_capacity((total + chunk - 1) / chunk);
    let mut start = 0usize;
    while start < total {
        let len = chunk.min(total - start);
        ranges.push((start, len));
        start += len;
    }
    Ok(ranges)
}

/// Compact PLE gate from the pinned equation: `sigmoid(sign(dot) * sqrt(max(|dot|,1e-6)))`.
pub fn ple_gate(dot: f32) -> f32 {
    let sign = if dot.is_sign_negative() { -1.0 } else { 1.0 };
    sigmoid(sign * dot.abs().max(1.0e-6).sqrt())
}

/// Return the deterministic HC feedback gate from a normalized branch dot.
pub fn hc_feedback_gate(dot: f32) -> f32 {
    ple_gate(dot)
}

/// Cosine/sine pair helper exposed for GPU parity probes.
pub fn rope_phase(position: usize, channel: usize, rotary_dim: usize, theta: f32) -> (f32, f32) {
    let exponent = (2 * channel) as f32 / rotary_dim.max(1) as f32;
    let angle = position as f32 * theta.powf(-exponent);
    angle.sin_cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_halfsplit_rotates_partner_halves() {
        let mut values = [1.0, 0.0, 0.0, 1.0];
        rope_prefix_halfsplit(&mut values, 1, 4, 10_000.0).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert_ne!(values, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn route_top10_is_normalized_and_tie_stable() {
        let logits = (0..512).map(|_| 1.0).collect::<Vec<_>>();
        let (indices, weights) = route_top10(&logits).unwrap();
        assert_eq!(indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn chunk_tiling_is_exact() {
        let ranges = prefill_chunks(11, 4).unwrap();
        assert_eq!(ranges, vec![(0, 4), (4, 4), (8, 3)]);
    }
}
