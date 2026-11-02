// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Logit-space sampling: top-p, temperature, repeat_penalty, and
//! single-token attractor blocking. Wraps GPU-dispatched sampling
//! kernels. One entry point for any caller (daemon, examples, future
//! arch ports).
//!
//! # Why this module
//!
//! Sampling primitives (top-p kernel call, repeat-penalty window upload,
//! attractor `-INF` writes, RNG threading) used to live across
//! `hipfire_runtime::llama` (CPU primitives + GPU launch wrappers) and
//! `hipfire-daemon/src/main.rs` (call-site glue). New arch ports either
//! reached into llama.rs internals or duplicated the host-side prep.
//! This module gives every caller one entry point: [`sample`], with
//! [`SamplerConfig`] holding the policy knobs.
//!
//! # Behavior preservation
//!
//! [`sample`] is a pure call-site refactor. It delegates to the same
//! `Gpu::sample_top_p` kernel and the same `memcpy_htod` for the repeat
//! window and attractor `-INF` writes that the daemon used inline. The
//! same `(logits, history, temp, top_p, repeat_penalty, repeat_window,
//! blocked_tokens, rng_state)` tuple produces the same `next_token`
//! before and after PR 3.
//!
//! # Conditional vs unconditional blocking
//!
//! The unclosed-opener attractor block (#111) decides at the call site
//! which token to block (the opener) based on a depth count over recent
//! history. The decision lives at the call site; the resulting set of
//! token IDs is passed in as [`SamplerConfig::blocked_tokens`]. The
//! sampler treats them as unconditional `-INF` writes — it does not
//! reimplement the depth counter.

use crate::llama;
use rdna_compute::{Gpu, GpuTensor};

/// Re-exports of the CPU-side sampling primitives that still live in
/// `hipfire_runtime::llama`. Other examples (`infer_qwen35`, `run`, etc.)
/// continue to call them via the `llama::` path; this module exposes
/// them via `sampler::` so new code has a single import path.
pub use crate::llama::{
    apply_ngram_block, apply_repeat_penalty, apply_repeat_penalty_candidates,
    apply_special_token_attractor_block, apply_unclosed_attractor_block, argmax, sample_top_k_p,
    sample_top_p as sample_top_p_cpu, sample_top_p_from_candidates, sampler_rng_restore,
    sampler_rng_snapshot, SamplingConfig,
};

/// Sampler policy knobs for a single token sample.
///
/// `temperature == 0.0` is the greedy path (the kernel falls back to
/// argmax internally). `top_p == 1.0` disables nucleus truncation.
/// `repeat_penalty == 1.0` (with any `repeat_window`) is a no-op.
///
/// `blocked_tokens` are unconditional `-INF` writes applied directly to
/// the on-GPU logits buffer before the sampling kernel launches. The
/// daemon populates this list per-token from its unclosed-opener depth
/// counter (#111); a future caller could populate it from anywhere.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0.0 = greedy (kernel argmax fast path).
    pub temperature: f32,
    /// 1.0 = no nucleus truncation.
    pub top_p: f32,
    /// 1.0 = repeat-penalty disabled.
    pub repeat_penalty: f32,
    /// Tokens of recent history visible to the repeat-penalty kernel.
    /// Effective window is `min(history.len(), repeat_window)` and is
    /// also clipped to the GPU `repeat_buf` capacity by the caller.
    pub repeat_window: usize,
    /// OpenAI `presence_penalty`: flat logit subtraction applied once to any
    /// token that occurred within `repeat_window`. 0.0 = disabled. Unlike the
    /// recency-weighted `repeat_penalty`, this is constant across the window,
    /// so it suppresses block-level repetition loops a short recency-weighted
    /// window cannot see (matches llama.cpp / Lemonade semantics).
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty`: logit subtraction scaled by the token's
    /// occurrence count within `repeat_window`. 0.0 = disabled.
    pub frequency_penalty: f32,
    /// Token IDs whose logit is unconditionally set to `-INF` before
    /// sampling. Used for the unclosed-opener attractor block (#111).
    pub blocked_tokens: Vec<u32>,
    /// Request-driven top-K candidate cutoff. `None` preserves the kernel's
    /// legacy hard-coded K=20 candidate gather exactly (the gather pool is
    /// always 20; a `Some(k)` only narrows the post-softmax nucleus to the
    /// top-`min(k, 20)` candidates). Values `>= 20` are no-ops.
    pub top_k: Option<u32>,
    /// Request-driven min-p cutoff: drop candidates whose probability is
    /// below `min_p * max_prob`. `None` (or `Some(0.0)`) disables the cut,
    /// byte-identical to the legacy path.
    pub min_p: Option<f32>,
}

impl SamplerConfig {
    /// Greedy: temperature=0, top_p=1, repeat_penalty=1, no blocks.
    /// The kernel takes the argmax fast path; RNG state is unused.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
            top_k: None,
            min_p: None,
        }
    }

    /// This config with its penalty stage neutralized (repeat 1, presence and
    /// frequency 0) and everything else kept: what the host still applies to
    /// a row a [`PenaltyTable`] prepass already penalized.
    pub fn without_penalties(&self) -> Self {
        Self {
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ..self.clone()
        }
    }
}

impl Default for SamplerConfig {
    /// Daemon-default: temperature=0.3, top_p=0.95, repeat_penalty=1.05.
    /// Mirrors the user-validated `RP=1.05` floor (CLAUDE.md memory:
    /// `feedback_repeat_penalty_default.md`). `repeat_window=128`
    /// matches `hipfire_runtime::llama::SamplingConfig::text_thinking()`.
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.95,
            repeat_penalty: 1.05,
            repeat_window: 128,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
            top_k: None,
            min_p: None,
        }
    }
}

/// Sample one token from a GPU-resident `logits` tensor.
///
/// Pre-dispatch host work, in order (matches the daemon's pre-PR3
/// inline sequence so byte-identical token streams are preserved):
///
///  1. Upload the trailing `min(history.len(), repeat_window,
///     repeat_buf_capacity)` tokens of `history` into `repeat_buf`.
///  2. Write `-INF` to `logits` at every offset in
///     `cfg.blocked_tokens` (one 4-byte H2D copy each).
///  3. Launch `Gpu::sample_top_p` (top-K + softmax + top-p + RNG +
///     argmax-on-greedy, all on GPU). One 8-byte D2H syncs the
///     `(token, new_rng)` result.
///
/// `rng_state` is mutated in place. For greedy (`temperature == 0.0`)
/// the value is unused but is still threaded through the kernel.
///
/// # Buffer types
///
/// `logits` is the model's output logits tensor (shape `[vocab_size]`,
/// dtype F32). `sample_buf` and `repeat_buf` are scratch buffers from
/// `llama::ForwardScratch`; the caller owns them. This matches the
/// existing pre-PR3 daemon signature exactly — we do not redesign the
/// argument shape.
pub fn sample(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    vocab_size: usize,
    history: &[u32],
    cfg: &SamplerConfig,
    rng_state: &mut u32,
) -> u32 {
    // Step 1: upload the repeat-penalty window. The kernel reads
    // `repeat_tokens[0..effective_window]`, so we only have to upload
    // the tokens that will actually be read. An empty scope is a no-op
    // (matches the first-sample case in the daemon, which used to
    // skip the htod when `bytes0` was empty).
    let buf_cap_tokens = repeat_buf.buf.size() / 4;
    let window = cfg.repeat_window.min(buf_cap_tokens);
    let scope_start = history.len().saturating_sub(window);
    let scope = &history[scope_start..];
    if !scope.is_empty() {
        let bytes: Vec<u8> = scope.iter().flat_map(|t| t.to_ne_bytes()).collect();
        let _ = gpu.hip.memcpy_htod(&repeat_buf.buf, &bytes);
    }

    // Step 2: apply unconditional blocked tokens. One 4-byte H2D per
    // token. The daemon path used `gpu_block_attractor_unclosed` which
    // wrote `-INF` to a single offset only when the depth counter
    // tripped; here the caller has already done the depth math and
    // accumulated the token IDs into `cfg.blocked_tokens`.
    if !cfg.blocked_tokens.is_empty() {
        let neg_inf: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        for &tok in &cfg.blocked_tokens {
            if (tok as usize) < vocab_size {
                let _ = gpu
                    .hip
                    .memcpy_htod_offset(&logits.buf, (tok as usize) * 4, &neg_inf);
            }
        }
    }

    // Step 3: GPU sample. The kernel does:
    //   - top-K = 20 from raw logits
    //   - apply repeat_penalty over `repeat_buf[0..scope.len()]`
    //   - softmax(top-K) with temperature scaling
    //   - top-p truncation
    //   - RNG draw + argmax-on-greedy fallback
    //   - writeback (token_id, new_rng) to `sample_buf`
    //   - 8-byte D2H sync (returned by the wrapper)
    let (tok, new_rng) = gpu
        .sample_top_p_pf(
            logits,
            sample_buf,
            repeat_buf,
            vocab_size,
            cfg.temperature,
            cfg.top_p,
            *rng_state,
            scope.len(),
            cfg.repeat_penalty,
            cfg.presence_penalty,
            cfg.frequency_penalty,
            cfg.top_k,
            cfg.min_p,
        )
        .expect("sample_top_p kernel launch / readback failed");
    *rng_state = new_rng;
    tok
}

/// CPU-only fallback: same policy as [`sample`] but operates on a host
/// `logits` slice. Used by the Qwen4 AR producer, the VL path, and the
/// grammar-active Qwen3.5 branches, where the argmax/top-p selection runs
/// after CPU-side policy mutations that have no GPU equivalent. Callers that
/// want a positional n-gram ban apply [`llama::apply_ngram_block`] before
/// calling this function.
///
/// Penalties and blocked tokens, then [`llama::sample_top_k_p`], which honours
/// `cfg.top_k` and `cfg.min_p` with the GPU kernel's semantics. Both `None`
/// (and `top_k == Some(20)`, `min_p == Some(0.0)`) reproduce the legacy
/// top-20 nucleus byte-for-byte.
pub fn sample_cpu(logits: &mut [f32], history: &[u32], cfg: &SamplerConfig) -> u32 {
    apply_logit_policy_cpu(logits, history, cfg);
    llama::sample_top_k_p(logits, cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p)
}

/// The logit mutations of [`sample_cpu`], in its order: repeat penalty,
/// presence/frequency penalties, then blocked tokens. Shared by the host AR
/// producer and every speculative verifier that must reproduce its
/// distribution, so one arithmetic serves both.
pub fn apply_logit_policy_cpu(logits: &mut [f32], history: &[u32], cfg: &SamplerConfig) {
    if cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0 {
        llama::apply_repeat_penalty(logits, history, cfg.repeat_window, cfg.repeat_penalty);
    }
    // OpenAI-style subtractive presence/frequency penalties over the same
    // window (mirrors the GPU `sample_top_p` kernel). logit -= freq*count +
    // presence, applied once per unique token. Keeps the GPU and CPU
    // (grammar-active) decode paths consistent.
    if (cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0) && cfg.repeat_window > 0 {
        let start = history.len().saturating_sub(cfg.repeat_window);
        let window = &history[start..];
        let mut counts: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for &t in window {
            *counts.entry(t).or_insert(0.0) += 1.0;
        }
        for (tok, count) in counts {
            if (tok as usize) < logits.len() {
                logits[tok as usize] -= cfg.frequency_penalty * count + cfg.presence_penalty;
            }
        }
    }
    for &tok in &cfg.blocked_tokens {
        if (tok as usize) < logits.len() {
            logits[tok as usize] = f32::NEG_INFINITY;
        }
    }
}

/// [`apply_logit_policy_cpu`] restricted to `(ids[i], values[i])` candidate
/// pairs, where `ids` are in-vocabulary token ids (never ranks): each
/// candidate's value ends bit-identical to what the dense helper leaves at
/// `logits[ids[i]]` for the same history and config. No allocation beyond
/// the repeat-penalty count map the dense helper also builds.
pub fn apply_logit_policy_candidates_cpu(
    ids: &[u32],
    values: &mut [f32],
    history: &[u32],
    cfg: &SamplerConfig,
) {
    debug_assert_eq!(ids.len(), values.len());
    if cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0 {
        llama::apply_repeat_penalty_candidates(
            ids,
            values,
            history,
            cfg.repeat_window,
            cfg.repeat_penalty,
        );
    }
    if (cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0) && cfg.repeat_window > 0 {
        let start = history.len().saturating_sub(cfg.repeat_window);
        let window = &history[start..];
        for (value, &id) in values.iter_mut().zip(ids) {
            // Same f32 count construction as the dense helper: one `+= 1.0`
            // per occurrence, in window order.
            let mut count = 0.0f32;
            for &t in window {
                if t == id {
                    count += 1.0;
                }
            }
            if count > 0.0 {
                *value -= cfg.frequency_penalty * count + cfg.presence_penalty;
            }
        }
    }
    for (value, &id) in values.iter_mut().zip(ids) {
        if cfg.blocked_tokens.contains(&id) {
            *value = f32::NEG_INFINITY;
        }
    }
}

/// [`PenaltyTable`] flag: divide a positive logit by the entry's repeat
/// factor, multiply any other logit by it.
pub const PENALTY_TABLE_REPEAT: u32 = 1;
/// [`PenaltyTable`] flag: subtract the entry's presence/frequency amount.
pub const PENALTY_TABLE_SUBTRACT: u32 = 2;

/// The penalty stage of [`apply_logit_policy_cpu`] (repeat, then presence /
/// frequency; blocked tokens are not part of it) for one or more logit rows,
/// as a token table the GPU prepass `Gpu::apply_penalty_table` applies in
/// place.
///
/// Every number is computed here with the CPU policy's own f32 arithmetic:
/// per distinct in-vocabulary token of a row's window, the repeat factor
/// `penalty.powf(count * recency).min(1.5)` (recency of the closest
/// occurrence) and the amount `frequency * count + presence`. The device
/// then only divides or multiplies and subtracts, so a row it penalizes is
/// bit-identical to [`apply_logit_policy_cpu`] over the same history and
/// config (blocked tokens excluded; apply those with
/// [`SamplerConfig::without_penalties`]). [`Self::apply_row_cpu`] is the
/// same arithmetic on the host.
///
/// Rows share one buffer: `row_ends[r]` is the cumulative entry count
/// through row `r`, and each entry is `[token, factor bits, amount bits]`.
#[derive(Debug, Default)]
pub struct PenaltyTable {
    flags: u32,
    row_ends: Vec<u32>,
    entries: Vec<u32>,
    /// Per-row scratch: token -> (count, closest recency).
    counts: std::collections::HashMap<u32, (u32, f32)>,
}

impl PenaltyTable {
    /// The flags of `cfg`'s penalty stage under [`apply_logit_policy_cpu`]'s
    /// own gates; 0 when it has none.
    pub fn flags_for(cfg: &SamplerConfig) -> u32 {
        let mut flags = 0;
        if cfg.repeat_window > 0 {
            if cfg.repeat_penalty != 1.0 {
                flags |= PENALTY_TABLE_REPEAT;
            }
            if cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0 {
                flags |= PENALTY_TABLE_SUBTRACT;
            }
        }
        flags
    }

    /// Start an empty table for `cfg`'s penalty stage.
    pub fn reset(&mut self, cfg: &SamplerConfig) {
        self.flags = Self::flags_for(cfg);
        self.row_ends.clear();
        self.entries.clear();
    }

    /// Append the next row: `cfg`'s penalty stage over `history` (its
    /// trailing `repeat_window` tokens) for a `vocab`-wide logit row. `cfg`
    /// must be the config the table was [`Self::reset`] for.
    pub fn push_row(&mut self, history: &[u32], cfg: &SamplerConfig, vocab: usize) {
        debug_assert_eq!(self.flags, Self::flags_for(cfg));
        if self.flags != 0 {
            let recent = &history[history.len().saturating_sub(cfg.repeat_window)..];
            let window_len = recent.len() as f32;
            self.counts.clear();
            for (i, &t) in recent.iter().enumerate() {
                let recency = (i as f32 + 1.0) / window_len;
                let entry = self.counts.entry(t).or_insert((0, 0.0));
                entry.0 += 1;
                if recency > entry.1 {
                    entry.1 = recency;
                }
            }
            for (&t, &(count, recency)) in &self.counts {
                if (t as usize) >= vocab {
                    continue;
                }
                let factor = if self.flags & PENALTY_TABLE_REPEAT != 0 {
                    cfg.repeat_penalty.powf(count as f32 * recency).min(1.5)
                } else {
                    1.0
                };
                let amount = if self.flags & PENALTY_TABLE_SUBTRACT != 0 {
                    cfg.frequency_penalty * count as f32 + cfg.presence_penalty
                } else {
                    0.0
                };
                self.entries
                    .extend_from_slice(&[t, factor.to_bits(), amount.to_bits()]);
            }
        }
        self.row_ends.push((self.entries.len() / 3) as u32);
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    pub fn rows(&self) -> usize {
        self.row_ends.len()
    }

    /// No row has an entry: applying the table changes nothing.
    pub fn is_noop(&self) -> bool {
        self.flags == 0 || self.entries.is_empty()
    }

    /// Cumulative entry count through each row.
    pub fn row_ends(&self) -> &[u32] {
        &self.row_ends
    }

    /// `[token, factor bits, amount bits]` per entry, rows in order.
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    /// Apply row `row` to a host logit row: the device kernel's arithmetic.
    pub fn apply_row_cpu(&self, row: usize, logits: &mut [f32]) {
        let begin = if row == 0 {
            0
        } else {
            self.row_ends[row - 1] as usize
        };
        for entry in self.entries[3 * begin..3 * self.row_ends[row] as usize].chunks_exact(3) {
            let Some(value) = logits.get_mut(entry[0] as usize) else {
                continue;
            };
            if self.flags & PENALTY_TABLE_REPEAT != 0 {
                let factor = f32::from_bits(entry[1]);
                if *value > 0.0 {
                    *value /= factor;
                } else {
                    *value *= factor;
                }
            }
            if self.flags & PENALTY_TABLE_SUBTRACT != 0 {
                *value -= f32::from_bits(entry[2]);
            }
        }
    }
}

/// Compute the unclosed-opener attractor blocked-token list (#111).
///
/// Counts unclosed openers in the trailing `window` tokens of `history`
/// (as `opens - closes`, floored at zero). When the running depth
/// reaches `threshold`, the opener is appended to `out`. The
/// downstream sampler will write `-INF` to that token's logit so the
/// next sample cannot stack another nested opener.
///
/// `pairs` is a slice of `(open, close)` pairs (e.g.
/// `(<tool_call>, </tool_call>)` and `(<think>, </think>)`); only
/// pairs whose `open` clears the threshold contribute. With
/// `threshold = 2`, a second consecutive opener without an intervening
/// closer is the last one the decoder is allowed to emit.
///
/// Pure, no GPU work; the caller passes the result into
/// [`SamplerConfig::blocked_tokens`].
pub fn collect_unclosed_attractor_blocks(
    history: &[u32],
    pairs: &[(u32, u32)],
    window: usize,
    threshold: usize,
    out: &mut Vec<u32>,
) {
    if window == 0 || threshold == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let recent = &history[start..];
    for &(open_id, close_id) in pairs {
        let mut depth: i32 = 0;
        for &t in recent {
            if t == open_id {
                depth += 1;
            } else if t == close_id && depth > 0 {
                depth -= 1;
            }
        }
        if depth >= threshold as i32 {
            out.push(open_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_config_fields() {
        let g = SamplerConfig::greedy();
        assert_eq!(g.temperature, 0.0);
        assert_eq!(g.top_p, 1.0);
        assert_eq!(g.repeat_penalty, 1.0);
        assert_eq!(g.repeat_window, 0);
        assert!(g.blocked_tokens.is_empty());
    }

    #[test]
    fn default_config_fields() {
        let d = SamplerConfig::default();
        assert!((d.temperature - 0.3).abs() < 1e-6);
        assert!((d.top_p - 0.95).abs() < 1e-6);
        assert!((d.repeat_penalty - 1.05).abs() < 1e-6);
        assert_eq!(d.repeat_window, 128);
        assert!(d.blocked_tokens.is_empty());
    }

    #[test]
    fn sample_cpu_greedy_picks_argmax() {
        // sample_cpu with greedy SamplerConfig should return the
        // argmax of the logits — even when blocked_tokens or
        // repeat_penalty would otherwise mutate the slice.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let cfg = SamplerConfig::greedy();
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 3);
    }

    #[test]
    fn sample_cpu_blocks_tokens() {
        // A blocked token should never be the argmax even if it
        // started as the largest logit. The blocker is unconditional
        // (a -INF write) so the next-best token wins.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![3];
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 1);
    }

    #[test]
    fn collect_unclosed_blocks_appends_when_depth_reached() {
        // history has 2 unclosed `<tool_call>` (id=10) — depth=2
        // hits threshold=2, so 10 should be in `out`. `<think>`
        // (id=20, close=21) has 1 unclosed → below threshold, not
        // appended.
        let history = [10u32, 99, 10, 5, 20, 7];
        let pairs = [(10u32, 11u32), (20u32, 21u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert_eq!(out, vec![10]);
    }

    #[test]
    fn collect_unclosed_blocks_zero_threshold_is_noop() {
        let history = [10u32, 10, 10];
        let pairs = [(10u32, 11u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_unclosed_blocks_balanced_open_close_does_not_block() {
        // 2 opens, 2 closes → depth=0, never trips.
        let history = [10u32, 5, 11, 10, 7, 11];
        let pairs = [(10u32, 11u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn sample_cpu_blocked_tokens_out_of_range_skipped() {
        // Out-of-range token IDs are silently skipped — the GPU path
        // does the same `(tok as usize) < vocab_size` guard.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![999, 1234];
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 3); // argmax unchanged
    }

    // ---- G1: CPU logit-policy exactness against a FROZEN reference --------
    //
    // `ref_policy` is a longhand, deliberately independent copy of the
    // pre-extraction `sample_cpu` penalty block plus the old
    // `llama::apply_repeat_penalty` arithmetic. It calls NO production helper
    // (no `llama::apply_repeat_penalty*`, no `apply_logit_policy_*`), so a
    // future edit to the shared helpers cannot move the reference with it.

    /// Small deterministic LCG (Knuth MMIX constants), high 32 bits.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        /// Uniform-ish in [-8, 8).
        fn next_logit(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32 * 16.0 - 8.0
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    /// Frozen reference of the old sample_cpu penalty block. Order: repeat →
    /// presence/frequency → blocked tokens.
    fn ref_policy(
        logits: &mut [f32],
        history: &[u32],
        repeat_penalty: f32,
        repeat_window: usize,
        presence: f32,
        frequency: f32,
        blocked: &[u32],
    ) {
        // --- old `llama::apply_repeat_penalty`, gated by sample_cpu ---
        if repeat_penalty != 1.0 && repeat_window > 0 {
            let start = history.len().saturating_sub(repeat_window);
            let recent = &history[start..];
            let window_len = recent.len() as f32;
            let mut uniq: Vec<u32> = recent.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            for &t in &uniq {
                // count = occurrences in window; recency = closest (latest)
                // occurrence's (i+1)/window_len — never a sum.
                let mut count = 0u32;
                let mut recency = 0.0f32;
                for (i, &x) in recent.iter().enumerate() {
                    if x == t {
                        count += 1;
                        let r = (i as f32 + 1.0) / window_len;
                        if r > recency {
                            recency = r;
                        }
                    }
                }
                if (t as usize) < logits.len() {
                    let effective = repeat_penalty.powf(count as f32 * recency).min(1.5);
                    if logits[t as usize] > 0.0 {
                        logits[t as usize] /= effective;
                    } else {
                        logits[t as usize] *= effective;
                    }
                }
            }
        }
        // --- old presence/frequency block ---
        if (presence > 0.0 || frequency > 0.0) && repeat_window > 0 {
            let start = history.len().saturating_sub(repeat_window);
            let window = &history[start..];
            let mut uniq: Vec<u32> = window.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            for &t in &uniq {
                let mut count = 0.0f32;
                for &x in window {
                    if x == t {
                        count += 1.0;
                    }
                }
                if (t as usize) < logits.len() {
                    logits[t as usize] -= frequency * count + presence;
                }
            }
        }
        for &tok in blocked {
            if (tok as usize) < logits.len() {
                logits[tok as usize] = f32::NEG_INFINITY;
            }
        }
    }

    struct PolicyCase {
        name: &'static str,
        repeat_penalty: f32,
        repeat_window: usize,
        presence: f32,
        frequency: f32,
        blocked: &'static [u32],
    }

    impl PolicyCase {
        fn cfg(&self) -> SamplerConfig {
            let mut c = SamplerConfig::greedy();
            c.repeat_penalty = self.repeat_penalty;
            c.repeat_window = self.repeat_window;
            c.presence_penalty = self.presence;
            c.frequency_penalty = self.frequency;
            c.blocked_tokens = self.blocked.to_vec();
            c
        }
        fn active(&self) -> bool {
            self.repeat_window > 0
                && (self.repeat_penalty != 1.0 || self.presence > 0.0 || self.frequency > 0.0)
        }
    }

    const POLICY_CASES: &[PolicyCase] = &[
        PolicyCase {
            name: "window0_everything_set",
            repeat_penalty: 1.1,
            repeat_window: 0,
            presence: 1.5,
            frequency: 0.5,
            blocked: &[],
        },
        PolicyCase {
            name: "neutral",
            repeat_penalty: 1.0,
            repeat_window: 64,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "repeat_1_1_w16",
            repeat_penalty: 1.1,
            repeat_window: 16,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "repeat_1_1_w1",
            repeat_penalty: 1.1,
            repeat_window: 1,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "repeat_1_1_w1000_short_history",
            repeat_penalty: 1.1,
            repeat_window: 1000,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "repeat_lt_1",
            repeat_penalty: 0.9,
            repeat_window: 16,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "repeat_capped",
            repeat_penalty: 2.0,
            repeat_window: 24,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "presence_only_repeat_1",
            repeat_penalty: 1.0,
            repeat_window: 16,
            presence: 1.5,
            frequency: 0.0,
            blocked: &[],
        },
        PolicyCase {
            name: "frequency_only",
            repeat_penalty: 1.0,
            repeat_window: 16,
            presence: 0.0,
            frequency: 0.5,
            blocked: &[],
        },
        PolicyCase {
            name: "presence_frequency_w1",
            repeat_penalty: 1.0,
            repeat_window: 1,
            presence: 1.5,
            frequency: 0.5,
            blocked: &[],
        },
        PolicyCase {
            name: "combined",
            repeat_penalty: 1.1,
            repeat_window: 32,
            presence: 1.5,
            frequency: 0.5,
            blocked: &[],
        },
        PolicyCase {
            name: "blocked_only",
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence: 0.0,
            frequency: 0.0,
            blocked: &[3, 7, 17, 5000],
        },
        PolicyCase {
            name: "blocked_with_combined",
            repeat_penalty: 1.1,
            repeat_window: 32,
            presence: 1.5,
            frequency: 0.5,
            blocked: &[3, 7, 17, 5000],
        },
    ];

    /// Dense logits with positive, negative, +0.0 and -0.0 entries.
    fn synth_logits(rng: &mut Lcg, vocab: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..vocab).map(|_| rng.next_logit()).collect();
        for i in (0..vocab).step_by(13) {
            v[i] = 0.0;
        }
        for i in (5..vocab).step_by(29) {
            v[i] = -0.0;
        }
        v
    }

    /// Histories around a window `w`: empty, 1, short, w-1, w, w+1, 3w. Tokens
    /// come mostly from a small pool (heavy repetition) plus some OOV ids.
    fn synth_histories(rng: &mut Lcg, vocab: usize, w: usize) -> Vec<Vec<u32>> {
        let pool = (vocab as u32 / 4).max(2);
        let mut lens = vec![0usize, 1, 3, 40];
        if w > 0 {
            lens.extend([w - 1, w, w + 1, 3 * w]);
        }
        lens.sort_unstable();
        lens.dedup();
        let mut out = Vec::new();
        for len in lens {
            let h: Vec<u32> = (0..len)
                .map(|i| {
                    if i % 17 == 9 {
                        vocab as u32 + rng.below(50) // OOV: dense skips these
                    } else {
                        rng.below(pool)
                    }
                })
                .collect();
            out.push(h);
        }
        out
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: logits[{i}] got {g} ({:#010x}) want {w} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    #[test]
    fn policy_dense_matches_frozen_reference_bitwise() {
        let mut rng = Lcg(0x5eed_0001);
        for case in POLICY_CASES {
            let cfg = case.cfg();
            for &vocab in &[64usize, 151, 300] {
                for hist in synth_histories(&mut rng, vocab, case.repeat_window.min(1000)) {
                    let orig = synth_logits(&mut rng, vocab);
                    let mut want = orig.clone();
                    ref_policy(
                        &mut want,
                        &hist,
                        case.repeat_penalty,
                        case.repeat_window,
                        case.presence,
                        case.frequency,
                        case.blocked,
                    );
                    let mut got = orig.clone();
                    apply_logit_policy_cpu(&mut got, &hist, &cfg);
                    let what = format!("{} vocab={vocab} hist_len={}", case.name, hist.len());
                    assert_bits_eq(&got, &want, &what);

                    // Non-vacuity: active policies change something whenever a
                    // nonzero, finite, unblocked in-vocab token is in-window;
                    // inactive ones change nothing (outside blocked ids).
                    let blocked_in = |i: usize| case.blocked.contains(&(i as u32));
                    let start = hist.len().saturating_sub(case.repeat_window);
                    let touched = hist[start..].iter().any(|&t| {
                        (t as usize) < vocab && !blocked_in(t as usize) && orig[t as usize] != 0.0
                    });
                    let changed =
                        (0..vocab).any(|i| !blocked_in(i) && got[i].to_bits() != orig[i].to_bits());
                    if case.active() && touched {
                        assert!(changed, "{what}: active policy changed nothing");
                    }
                    if !case.active() {
                        assert!(!changed, "{what}: inactive policy mutated logits");
                    }
                    for &b in case.blocked {
                        if (b as usize) < vocab {
                            assert_eq!(got[b as usize], f32::NEG_INFINITY, "{what}: block {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn policy_candidates_match_dense_reference_bitwise() {
        let mut rng = Lcg(0xc0ffee_02);
        for case in POLICY_CASES {
            let cfg = case.cfg();
            for &vocab in &[64usize, 151, 300] {
                for hist in synth_histories(&mut rng, vocab, case.repeat_window.min(1000)) {
                    let orig = synth_logits(&mut rng, vocab);
                    let mut dense = orig.clone();
                    ref_policy(
                        &mut dense,
                        &hist,
                        case.repeat_penalty,
                        case.repeat_window,
                        case.presence,
                        case.frequency,
                        case.blocked,
                    );

                    // Candidates: random ids (mostly absent from history),
                    // every in-vocab history id in reverse order (repeats
                    // included), blocked ids, and duplicated ids; arbitrary
                    // order, never ranks.
                    let mut ids: Vec<u32> = (0..40).map(|_| rng.below(vocab as u32)).collect();
                    ids.extend(hist.iter().rev().copied().filter(|&t| (t as usize) < vocab));
                    ids.extend(
                        case.blocked
                            .iter()
                            .copied()
                            .filter(|&b| (b as usize) < vocab),
                    );
                    let dup: Vec<u32> = ids.iter().take(6).copied().collect();
                    ids.extend(dup);
                    let n = ids.len();
                    for i in 0..n {
                        let j = rng.below(n as u32) as usize;
                        ids.swap(i, j);
                    }

                    let mut vals: Vec<f32> = ids.iter().map(|&id| orig[id as usize]).collect();
                    apply_logit_policy_candidates_cpu(&ids, &mut vals, &hist, &cfg);
                    for (k, &id) in ids.iter().enumerate() {
                        assert_eq!(
                            vals[k].to_bits(),
                            dense[id as usize].to_bits(),
                            "{} vocab={vocab} hist_len={} cand#{k} id={id}: got {} want {}",
                            case.name,
                            hist.len(),
                            vals[k],
                            dense[id as usize]
                        );
                    }
                }
            }
        }
    }

    /// The GPU prepass arithmetic (`PenaltyTable::apply_row_cpu`, then the
    /// penalty-free policy for blocked tokens) equals the frozen reference
    /// and `apply_logit_policy_cpu`, bit for bit, with every history of a
    /// case packed as consecutive rows of one table.
    #[test]
    fn penalty_table_rows_match_policy_bitwise() {
        let mut rng = Lcg(0x7ab1_e003);
        let mut table = PenaltyTable::default();
        for case in POLICY_CASES {
            let cfg = case.cfg();
            let rest = cfg.without_penalties();
            for &vocab in &[64usize, 151, 300] {
                let hists = synth_histories(&mut rng, vocab, case.repeat_window.min(1000));
                table.reset(&cfg);
                for hist in &hists {
                    table.push_row(hist, &cfg, vocab);
                }
                assert_eq!(table.rows(), hists.len());
                assert_eq!(table.flags() == 0, !case.active(), "{}: flags", case.name);
                for (row, hist) in hists.iter().enumerate() {
                    let orig = synth_logits(&mut rng, vocab);
                    let mut want = orig.clone();
                    ref_policy(
                        &mut want,
                        hist,
                        case.repeat_penalty,
                        case.repeat_window,
                        case.presence,
                        case.frequency,
                        case.blocked,
                    );
                    let mut policy = orig.clone();
                    apply_logit_policy_cpu(&mut policy, hist, &cfg);
                    let mut got = orig.clone();
                    table.apply_row_cpu(row, &mut got);
                    apply_logit_policy_cpu(&mut got, hist, &rest);
                    let what = format!(
                        "{} vocab={vocab} row={row} hist_len={}",
                        case.name,
                        hist.len()
                    );
                    assert_bits_eq(&got, &want, &what);
                    assert_bits_eq(&policy, &want, &what);
                }
            }
        }
    }

    /// A neutral policy builds empty rows; distinct tokens only, in-vocab only.
    #[test]
    fn penalty_table_entries_are_distinct_in_vocab_tokens() {
        let mut table = PenaltyTable::default();
        let neutral = SamplerConfig::greedy();
        table.reset(&neutral);
        table.push_row(&[1, 2, 3], &neutral, 10);
        assert_eq!(table.row_ends(), &[0]);
        assert!(table.is_noop());

        let mut cfg = SamplerConfig::greedy();
        cfg.repeat_window = 4;
        cfg.presence_penalty = 1.5;
        table.reset(&cfg);
        // Window keeps [2, 2, 99, 3]: 99 is out of a 10-wide vocab.
        table.push_row(&[7, 2, 2, 99, 3], &cfg, 10);
        table.push_row(&[], &cfg, 10);
        assert_eq!(table.flags(), PENALTY_TABLE_SUBTRACT);
        assert_eq!(table.row_ends(), &[2, 2]);
        let mut tokens: Vec<u32> = table.entries().chunks_exact(3).map(|e| e[0]).collect();
        tokens.sort_unstable();
        assert_eq!(tokens, vec![2, 3]);
    }

    #[test]
    fn policy_repeat_uses_closest_occurrence_not_sum() {
        // window = whole history (len 4). Token 5: once at idx0 (recency
        // 0.25). Token 1: idx1,idx2 -> count 2, closest recency 0.75, so
        // penalty^(2*0.75) = penalty^1.5 (a recency SUM would give ^2.5).
        let mut cfg = SamplerConfig::greedy();
        cfg.repeat_penalty = 1.1;
        cfg.repeat_window = 4;
        let hist = [5u32, 1, 1, 2];
        let orig: Vec<f32> = vec![0.5, 2.0, -3.0, 4.0, 1.0, 6.0, -1.0, 0.25];
        let mut got = orig.clone();
        apply_logit_policy_cpu(&mut got, &hist, &cfg);
        let p = 1.1f32;
        let mut want = orig.clone();
        want[5] /= p.powf(1.0 * 0.25);
        want[1] /= p.powf(2.0 * 0.75);
        want[2] *= p.powf(1.0 * 1.0);
        assert_bits_eq(&got, &want, "closest occurrence");
        // Untouched ids stay bit-identical.
        for i in [0usize, 3, 4, 6, 7] {
            assert_eq!(got[i].to_bits(), orig[i].to_bits());
        }
    }

    #[test]
    fn policy_repeat_effective_penalty_is_capped_at_1_5() {
        // 2.0^(6*1.0) = 64 -> capped to 1.5 for both signs.
        let mut cfg = SamplerConfig::greedy();
        cfg.repeat_penalty = 2.0;
        cfg.repeat_window = 8;
        let hist = [9u32, 3, 3, 3, 3, 3, 3];
        let orig: Vec<f32> = (0..16).map(|i| if i == 3 { 6.0 } else { 1.0 }).collect();
        let mut got = orig.clone();
        apply_logit_policy_cpu(&mut got, &hist, &cfg);
        assert_eq!(got[3].to_bits(), (6.0f32 / 1.5).to_bits());
        let mut neg = orig.clone();
        neg[3] = -6.0;
        apply_logit_policy_cpu(&mut neg, &hist, &cfg);
        assert_eq!(neg[3].to_bits(), (-6.0f32 * 1.5).to_bits());
    }

    #[test]
    fn policy_presence_frequency_exact_values() {
        // presence 1.5 + frequency 0.5, token 2 seen 3x inside the window,
        // token 4 seen 1x outside it (window 4 over 6 tokens), OOV 99 skipped.
        let mut cfg = SamplerConfig::greedy();
        cfg.repeat_window = 4;
        cfg.presence_penalty = 1.5;
        cfg.frequency_penalty = 0.5;
        let hist = [4u32, 4, 2, 2, 99, 2];
        let orig: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut got = orig.clone();
        apply_logit_policy_cpu(&mut got, &hist, &cfg);
        let mut want = orig.clone();
        want[2] -= 0.5 * 3.0 + 1.5;
        assert_bits_eq(&got, &want, "presence+frequency");

        let mut pres_only = SamplerConfig::greedy();
        pres_only.repeat_window = 4;
        pres_only.presence_penalty = 1.5;
        let mut got = orig.clone();
        apply_logit_policy_cpu(&mut got, &hist, &pres_only);
        let mut want = orig.clone();
        want[2] -= 0.0 * 3.0 + 1.5;
        assert_bits_eq(&got, &want, "presence only");
    }

    #[test]
    fn sample_cpu_matches_reference_policy_plus_sample_top_k_p() {
        // sample_cpu == frozen policy + unchanged llama::sample_top_k_p draw:
        // same seeded RNG, same logits stream, same growing history -> same
        // tokens and same final RNG state. The CPU RNG is process-global.
        let _g = crate::llama::sampler_rng_test_guard();
        let vocab = 200usize;
        let steps = 40usize;

        struct Cfg {
            name: &'static str,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            min_p: Option<f32>,
            policy: &'static PolicyCase,
        }
        let by_name = |n: &str| POLICY_CASES.iter().find(|c| c.name == n).unwrap();
        let configs = [
            Cfg {
                name: "neutral_default_draw",
                temperature: 0.7,
                top_p: 0.95,
                top_k: None,
                min_p: None,
                policy: by_name("neutral"),
            },
            Cfg {
                name: "neutral_top_k_min_p",
                temperature: 0.9,
                top_p: 0.9,
                top_k: Some(5),
                min_p: Some(0.05),
                policy: by_name("neutral"),
            },
            Cfg {
                name: "repeat",
                temperature: 0.7,
                top_p: 0.95,
                top_k: None,
                min_p: None,
                policy: by_name("repeat_1_1_w16"),
            },
            Cfg {
                name: "combined",
                temperature: 0.8,
                top_p: 0.92,
                top_k: None,
                min_p: None,
                policy: by_name("combined"),
            },
            Cfg {
                name: "combined_blocked_top_k",
                temperature: 1.0,
                top_p: 1.0,
                top_k: Some(8),
                min_p: None,
                policy: by_name("blocked_with_combined"),
            },
            Cfg {
                name: "presence_w1",
                temperature: 0.6,
                top_p: 0.9,
                top_k: None,
                min_p: Some(0.02),
                policy: by_name("presence_frequency_w1"),
            },
        ];

        for c in &configs {
            let mut cfg = c.policy.cfg();
            cfg.temperature = c.temperature;
            cfg.top_p = c.top_p;
            cfg.top_k = c.top_k;
            cfg.min_p = c.min_p;

            let run = |use_sample_cpu: bool| -> (Vec<u32>, u32) {
                crate::llama::reset_cpu_sampler_rng(0x1234_5678);
                let mut lrng = Lcg(0xfeed_0003);
                // Prompt-like prefix so penalties have in-window history from step 0.
                let mut hist: Vec<u32> = (0..12).map(|_| lrng.below(30)).collect();
                let mut toks = Vec::with_capacity(steps);
                for _ in 0..steps {
                    let mut logits = synth_logits(&mut lrng, vocab);
                    // Pull mass toward a few low ids so repeats actually recur.
                    for i in 0..6 {
                        logits[i] += 3.0;
                    }
                    let tok = if use_sample_cpu {
                        sample_cpu(&mut logits, &hist, &cfg)
                    } else {
                        ref_policy(
                            &mut logits,
                            &hist,
                            c.policy.repeat_penalty,
                            c.policy.repeat_window,
                            c.policy.presence,
                            c.policy.frequency,
                            c.policy.blocked,
                        );
                        crate::llama::sample_top_k_p(
                            &logits,
                            c.temperature,
                            c.top_p,
                            c.top_k,
                            c.min_p,
                        )
                    };
                    toks.push(tok);
                    hist.push(tok);
                }
                (toks, crate::llama::sampler_rng_snapshot())
            };

            let (a, a_state) = run(true);
            let (b, b_state) = run(false);
            assert_eq!(a, b, "{}: token stream diverged", c.name);
            assert_eq!(a_state, b_state, "{}: RNG state diverged", c.name);
            // The stream must not be degenerate.
            assert!(
                a.iter().any(|&t| t != a[0]),
                "{}: constant token stream proves nothing",
                c.name
            );
        }
    }
}
