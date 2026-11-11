//! MTP-only speculative decode: standalone trunk + native MTP head loop.
//!
//! The MTP head ([`crate::mtp_head`], Task 9) is a single transformer-decoder
//! block that takes (last-committed-token, trunk-post-norm-hidden) and emits
//! a single-token next-prediction. This module composes that head into a
//! greedy lossless spec-decode cycle:
//!
//! ```text
//! cycle:
//!   1. Snapshot trunk DN state (for rollback after batched verify).
//!   2. MTP-chain `max_n` candidates by feeding each step's `t_mtp_out` and
//!      its predicted token back into mtp_head_forward.
//!   3. Run trunk verify on `[last_committed, c1, ..., c_max_n]`
//!      (max_n + 1 tokens). Capture per-position post-output-norm hidden via
//!      `per_token_hidden_out`.
//!   4. Trunk lm_head over all positions → batched argmax → accept-prefix
//!      where MTP candidate matches trunk argmax.
//!   5. bonus_token = trunk argmax at slot `accept_count` (always commits
//!      unless we early-stopped on EOS inside the accepted prefix).
//!   6. Update `state.prev_hidden` from `verify_hidden[advance - 1]`.
//!   7. Roll back trunk DN state to snapshot, replay accepted committed
//!      tokens (= `verify_tokens[..advance]`) so trunk dn/kv lands at
//!      position `cur_pos + advance` for the next cycle.
//!   8. MTP head KV cache: NO explicit rollback. Stale slots from rejected
//!      candidates get overwritten in-order by next cycle's MTP chain
//!      because each next-cycle MTP step writes slot `pos` BEFORE attending
//!      `[0..pos+1)`.
//! ```
//!
//! Greedy-only (temp=0). No DDTree, no PLD, no rejection sampling — that's
//! Task 11 territory.

use crate::mtp_head::{
    self, Qwen35MtpHead, Qwen35MtpHeadBatchedScratch, Qwen35MtpHeadKvCache, Qwen35MtpHeadScratch,
};
use crate::qwen35::prefill::{fp8_verify_open, fp8_verify_target};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::qwen35::{LayerType, LayerWeights, PrefillBatchScratch, StateQuant};
use hipfire_dispatch::families::moe::{RoutedExpertWeights, NARROW_VERIFY_MAX_ROWS};
/// Debug tracing for the slot-engine MTP cycle (`HIPFIRE_MTP_TRACE=1`).
/// Prints seed/candidates/argmax/advance per cycle to stderr.
pub(crate) fn mtp_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MTP_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Per-window phase timing for Qwen3.x MTP (`HIPFIRE_MTP_PHASE_TIMING=1`,
/// `QWEN35_MTP_PHASE` lines); off = no `Instant`, no sync.
pub(crate) fn mtp_phase_timing_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MTP_PHASE_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false)
    });
    *ON
}

/// Host wall time and launch/upload counts of penalty work in one MTP window.
#[derive(Clone, Copy, Debug, Default)]
pub struct MtpPenaltyTiming {
    pub host_us: f64,
    pub launches: u32,
    pub uploads: u32,
}

impl MtpPenaltyTiming {
    /// Start a timed penalty section; `None` (no clock read) unless phase timing is on.
    fn start() -> Option<Instant> {
        mtp_phase_timing_enabled().then(Instant::now)
    }

    /// Account a section begun by [`Self::start`]; no-op when it returned `None`.
    fn finish(&mut self, t0: Option<Instant>, uploads: u32, launches: u32) {
        if let Some(t0) = t0 {
            self.host_us += t0.elapsed().as_secs_f64() * 1e6;
            self.uploads += uploads;
            self.launches += launches;
        }
    }
}

/// A10 fault seam (oracle only): while armed, every MTP verify cycle
/// rejects all candidates, so each cycle advances exactly one trunk token
/// (the τ=1 full-reject path). Armed through [`arm_mtp_full_reject`], not the
/// environment, so the oracle can toggle it between requests on a live engine.
static MTP_FULL_REJECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Arm or disarm the forced full-reject fault seam. Test-facing only.
pub fn arm_mtp_full_reject(on: bool) {
    MTP_FULL_REJECT.store(on, std::sync::atomic::Ordering::Release);
}
use crate::speculative::{
    apply_topp_trunc, sample_categorical, sample_residual, sample_residual_point_mass,
};
use crate::speculative::{DeltaNetSnapshot, GdnTape, ModelSlot};
use hip_bridge::{Event, Graph, GraphExec, HipResult, Stream};
use hipfire_runtime::llama;
use hipfire_runtime::llama::KvCache;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

/// Cross-request batched MTP cycle (continuous batching).
#[path = "mtp_cb.rs"]
pub mod cb;

// ─── Sampling primitives (host-side, used by temp>0 spec-decode path) ────

/// Sampling configuration matching the Unsloth-recommended MTP defaults
/// (temp=1.0, top_p=0.95, top_k=20, min_p=0.0 for Qwen3.5/3.6 thinking
/// mode). `temp == 0.0` disables sampling and the spec-step falls back
/// to greedy argmax-match.
#[derive(Copy, Clone, Debug)]
pub struct MtpSamplingConfig {
    pub temp: f32,
    pub top_k: usize, // AR candidate cap (`llama::ar_candidate_cap`); 0 = none
    pub top_p: f32,   // 1.0 = disabled (no nucleus cutoff)
    pub min_p: f32,   // 0.0 = disabled (no min-prob cutoff)
    /// Repeat penalty (1.0 = off). With the presence/frequency penalties it
    /// is applied by the AR GPU sampler over the last `repeat_window`
    /// generated tokens; verify (and full-vocab draft) rows reproduce it.
    pub repeat_penalty: f32,
    /// Penalty window in tokens (0 = penalties off), already clamped to the
    /// AR repeat buffer capacity by the caller (≤ [`PENALTY_REPEAT_CAP`]).
    pub repeat_window: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for MtpSamplingConfig {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl MtpSamplingConfig {
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }

    /// True iff the AR GPU sampler's penalty prepass would run for this
    /// request: exactly the kernel gate (`SlotSampleParams::penalized`).
    pub fn penalized(&self) -> bool {
        self.repeat_window > 0
            && (self.repeat_penalty > 1.0
                || self.presence_penalty > 0.0
                || self.frequency_penalty > 0.0)
    }
}

/// Maximum penalty window (tokens) the MTP penalty history buffer
/// ([`MtpSpecState::mtp_sample_repeat_buf`]) is sized for; mirrors the AR
/// repeat buffer's 2048-token capacity.
pub const PENALTY_REPEAT_CAP: usize = 2048;

fn mtp_device_token_chain_enabled_from_env() -> bool {
    // Default on: this path is token-identical in greedy mode and removes the
    // proposal-loop host data dependency needed for later graph capture.
    match hipfire_config::developer_var("HIPFIRE_MTP_DEVICE_TOKEN_CHAIN") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

fn mtp_snapshot_overlap_enabled_from_env() -> bool {
    // Default off: current gfx1201 benches show this stream split regresses,
    // but the hook is useful as an opt-in cross-arch experiment.
    match hipfire_config::developer_var("HIPFIRE_MTP_SNAPSHOT_OVERLAP") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on" || v == "yes"
        }
        Err(_) => false,
    }
}

fn mtp_gpu_greedy_accept_enabled_from_env() -> bool {
    // Default on for greedy device-token-chain MTP: candidates and verify
    // argmaxes are already on-device, so the prefix accept + bonus selection
    // can stay there until a compact two-int result is needed on host.
    match hipfire_config::developer_var("HIPFIRE_MTP_GPU_ACCEPT") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

fn mtp_q8_verify_wmma_enabled_from_env_value(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
    }
}

fn mtp_q8_verify_wmma_enabled_from_env() -> bool {
    // Default on for MTP q8 verify: the chunked dispatch only routes to gfx12
    // WMMA when the arch/shape supports it, otherwise it falls back to scalar.
    // Prompt-sweep parity is clean; opt out with HIPFIRE_MTP_Q8_VERIFY_WMMA=0.
    mtp_q8_verify_wmma_enabled_from_env_value(
        hipfire_config::developer_var("HIPFIRE_MTP_Q8_VERIFY_WMMA")
            .ok()
            .as_deref(),
    )
}

/// Default draft-confidence cutoff (adaptive-K) per arch.
///
/// p_min=0.6 was measured as a win on gfx1100 (2026-06-16 sweep: +11% on
/// high-τ code) — but that sweep predates the POST-OUTPUT-NORM hidden fix
/// for the slot-engine head inputs (2026-09-02): with correctly-scaled head
/// inputs the draft chain's confidence distribution shifted upward, and the
/// 0.6 gate now mostly truncates chains that would have verified. Measured
/// on ornith-1.5-9b (gfx1101): slot engine 69.4 → 78.1 tok/s with p_min=0
/// (also keeps the verify width stable at mtp_k+1 rows, which the verify
/// graph key needs), sequential path τ 1.20 → 1.47 / 67.0 → 69.0 tok/s.
/// Default is therefore 0.0 (off) everywhere; re-enable per arch via
/// HIPFIRE_MTP_P_MIN (e.g. =0.6) if a future head wants it.
fn default_mtp_p_min(arch: &str) -> f32 {
    let _ = arch;
    if let Some(v) = hipfire_config::developer_var("HIPFIRE_MTP_P_MIN").ok() {
        return v
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .unwrap_or(0.0);
    }
    0.0
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MtpProposalGraphPolicy {
    Off,
    Auto,
    On,
}

fn mtp_proposal_graph_policy_from_env_value(value: Option<&str>) -> MtpProposalGraphPolicy {
    // Opt-in for now: q8 proposal graph capture is token-identical, but
    // remains neutral/slightly negative on the canonical gfx1201 A3B K=5
    // smoke after PR #5/#6 lowered verify cost (2026-05-28: 192.99 tok/s
    // unset vs 191.44 tok/s graph=on, same output md5).
    match value {
        None => MtpProposalGraphPolicy::Off,
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => MtpProposalGraphPolicy::Off,
            "1" | "true" | "on" | "yes" => MtpProposalGraphPolicy::On,
            "auto" | "" => MtpProposalGraphPolicy::Auto,
            _ => MtpProposalGraphPolicy::Auto,
        },
    }
}

fn mtp_proposal_graph_policy_from_env() -> MtpProposalGraphPolicy {
    mtp_proposal_graph_policy_from_env_value(
        hipfire_config::developer_var("HIPFIRE_MTP_PROPOSAL_GRAPH")
            .ok()
            .as_deref(),
    )
}

fn mtp_proposal_graph_eligible_for(
    policy: MtpProposalGraphPolicy,
    use_device_token_chain: bool,
    use_full_vocab: bool,
    kv_mode: crate::mtp_head::MtpKvMode,
) -> bool {
    policy != MtpProposalGraphPolicy::Off
        && use_device_token_chain
        && !use_full_vocab
        && kv_mode == crate::mtp_head::MtpKvMode::Q8
}

fn mtp_device_token_chain_eligible_for(
    embd_format: llama::EmbeddingFormat,
    use_sampling: bool,
    use_p_min: bool,
) -> bool {
    !use_sampling
        && !use_p_min
        && matches!(
            embd_format,
            llama::EmbeddingFormat::HFQ4G256 | llama::EmbeddingFormat::Q8_0
        )
}

/// Reproducible xorshift64* RNG. Cheap, fast, no `rand` dep.
#[derive(Copy, Clone, Debug)]
pub struct MtpRng(u64);

impl MtpRng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_uniform_f32(&mut self) -> f32 {
        // 24-bit mantissa: [0.0, 1.0).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Reconstruct from a raw internal state (NO seed re-multiply), for resuming
    /// a stream across calls; pairs with [`state`](Self::state). Zero → 1.
    pub fn from_state(s: u64) -> Self {
        Self(s.max(1))
    }
    /// Raw internal state, for persisting/resuming the stream across calls.
    pub fn state(&self) -> u64 {
        self.0
    }
}

/// Sample one token from a row of logits with temp/top_k/top_p/min_p.
/// Returns `(token_id, p_sampled)` where `p_sampled` is the normalized
/// probability of the chosen token under the truncated distribution.
///
/// All arithmetic is host-side f64 for the softmax stability + cumulative
/// sums, then cast back to f32 for the returned probability. The vocab is
/// typically 32K (compressed) or 248K (full); a few μs per call.
pub fn sample_from_logits(logits: &[f32], cfg: &MtpSamplingConfig, rng: &mut MtpRng) -> (u32, f32) {
    assert!(!logits.is_empty(), "sample_from_logits: empty logits row");
    assert!(
        cfg.temp > 0.0,
        "sample_from_logits: temp must be > 0; caller must branch on greedy"
    );

    // 1. Build (id, scaled_logit) pairs.
    let inv_temp = 1.0 / cfg.temp as f64;
    let mut pairs: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l as f64 * inv_temp))
        .collect();

    // 2. Optional top_k: partial-sort to keep top_k by logit (descending).
    //
    // hunt3 M-B: a NaN logit makes `partial_cmp` return None; `.unwrap()` would
    // panic (the panic class M-B exists to remove — matches the deepseek4
    // sampling.rs sibling sites). `.unwrap_or(Less)` averts the panic. Note a
    // NaN may NOT sort to the tail (with Less the NaN is ordered toward the
    // front), so the kept top-k set can transiently include it; it is then
    // neutralized downstream — exp(NaN)=NaN poisons the softmax sum and the
    // multinomial loop's `r < acc` stays false, so sampling falls through to
    // the last surviving candidate (line ~297) instead of panicking.
    if cfg.top_k > 0 && cfg.top_k < pairs.len() {
        pairs.select_nth_unstable_by(cfg.top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less)
        });
        pairs.truncate(cfg.top_k);
    }

    // 3. Sort by logit descending for top_p and min_p filtering + sampling.
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

    // 4. Softmax with max-subtraction stability.
    let max_logit = pairs[0].1;
    let mut sum_exp = 0.0_f64;
    let mut probs: Vec<f64> = pairs
        .iter()
        .map(|(_, l)| {
            let e = (l - max_logit).exp();
            sum_exp += e;
            e
        })
        .collect();
    for p in probs.iter_mut() {
        *p /= sum_exp;
    }

    // 5. min_p: drop tokens with normalized prob < min_p * top_prob.
    if cfg.min_p > 0.0 {
        let top_p_val = probs[0];
        let thresh = (cfg.min_p as f64) * top_p_val;
        let cutoff = probs
            .iter()
            .position(|&p| p < thresh)
            .unwrap_or(probs.len());
        pairs.truncate(cutoff.max(1));
        probs.truncate(cutoff.max(1));
        // Renormalize after min_p.
        let s: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }

    // 6. top_p (nucleus): cumulative-prob cutoff.
    if cfg.top_p < 1.0 {
        let mut cum = 0.0_f64;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p as f64 {
                cutoff = i + 1;
                break;
            }
        }
        pairs.truncate(cutoff);
        probs.truncate(cutoff);
        // Renormalize after top_p.
        let s: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }

    // 7. Multinomial sample.
    let r = rng.next_uniform_f32() as f64;
    let mut acc = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return (pairs[i].0, probs[i] as f32);
        }
    }
    // Numerical edge case: fall through to last.
    let last = probs.len() - 1;
    (pairs[last].0, probs[last] as f32)
}

/// Compute the temperature-scaled softmax probability of a single token at
/// index `idx` in a logits row. Used by the residual-acceptance rule to
/// gather `p_target(c_k)` and `p_draft(c_k)` under the same temperature
/// scaling that draft sampling uses. Numerically stable via
/// max-subtraction (single pass over logits).
///
/// Note: this is the **un-truncated** softmax prob (no top_k/top_p filter).
/// For the residual ratio `p_target(c_k) / p_draft(c_k)` we want both probs
/// from the same distribution shape; using raw temp-scaled softmax keeps
/// the math clean even if draft sampling itself was truncated.
pub fn softmax_prob_at_temp(logits: &[f32], idx: usize, temp: f32) -> f32 {
    assert!(temp > 0.0, "softmax_prob_at_temp: temp must be > 0");
    let inv_t = (1.0 / temp) as f64;
    // First pass: find max (for stability) of temp-scaled logits.
    let mut max_scaled = f64::NEG_INFINITY;
    for &l in logits.iter() {
        let s = l as f64 * inv_t;
        if s > max_scaled {
            max_scaled = s;
        }
    }
    let mut sum_exp = 0.0_f64;
    let mut target_e = 0.0_f64;
    for (i, &l) in logits.iter().enumerate() {
        let e = ((l as f64) * inv_t - max_scaled).exp();
        sum_exp += e;
        if i == idx {
            target_e = e;
        }
    }
    (target_e / sum_exp) as f32
}

// ─── Public state ────────────────────────────────────────────────────────

/// All persistent buffers needed by [`spec_step_mtp`]. Allocate once per
/// generation; reuse across cycles. Frees at end of generation via
/// [`MtpSpecState::free_gpu`].
pub struct MtpSpecState {
    /// Persistent post-output-norm hidden snapshot of the trunk at the
    /// position of the most-recently-committed token. Fed as `prev_hidden`
    /// into the MTP head's first step each cycle. Shape: `[dim]` F32.
    pub prev_hidden: GpuTensor,

    /// Trunk position whose hidden `prev_hidden` holds, or `None` after a
    /// reset. The MTP head pairs token `p` with the hidden of position
    /// `p - 1`, so a fill starting at `start_pos` may reuse `prev_hidden`
    /// only when this is `Some(start_pos - 1)`.
    pub prev_hidden_pos: Option<usize>,

    /// Per-cycle scratch: post-output-norm hidden states from the trunk
    /// verify, one row per verify position. Shape: `[(max_n + 1) × dim]` F32.
    pub verify_hidden: GpuTensor,

    /// Per-cycle scratch: trunk lm_head output over all verify positions.
    /// Shape: `[(max_n + 1) × vocab]` F32.
    pub verify_logits: GpuTensor,

    /// Per-cycle scratch: FWHT-rotated `verify_hidden` for MQ-family
    /// lm_head. Shape: `[(max_n + 1) × dim]` F32. Allocated unconditionally;
    /// unused for non-MQ lm_head dtypes.
    pub verify_rot: GpuTensor,

    /// GPU-side argmax buffer for batched argmax over verify positions.
    /// Shape: `[(max_n + 1)]` i32 stored as F32 slots.
    pub verify_argmax: GpuTensor,

    /// Trunk DN snapshot for rollback before replay.
    pub trunk_snap: DeltaNetSnapshot,

    /// Side stream used to overlap `trunk_snap.save_from` with the MTP
    /// proposal loop. The event orders the side-stream copy after any
    /// prior-cycle main-stream work that may still be updating DN state.
    pub trunk_snap_stream: Option<Stream>,
    pub trunk_snap_start_event: Option<Event>,

    /// Persistent batch scratch for the trunk's batched verify forward.
    /// Sized to `max_n + 1` so verify always fits in one chunk.
    pub trunk_pbs: qwen35::PrefillBatchScratch,

    /// Innovation tape captured during trunk verify. After rollback, replaying
    /// this advances only the DeltaNet recurrence for the accepted prefix.
    pub trunk_gdn_tape: GdnTape,

    /// MTP head per-call scratch (Task 9). One per slot; reused across cycles.
    pub mtp_scratch: Qwen35MtpHeadScratch,

    /// MTP head KV cache (Task 9). Single-layer F32, rolling overwrite.
    pub mtp_kv: Qwen35MtpHeadKvCache,

    /// Per-step `t_mtp_out` capture buffer for the K-step batched lm_head
    /// chain (Task 10b). Shape `[max_n, n_embd]` row-major. After the
    /// chain runs, the end-of-chain batched lm_head consumes this whole
    /// buffer in one GEMM call. Step k writes its `t_mtp_out` into row k
    /// (offset `k * n_embd * 4` bytes); step k+1 reads row k as its
    /// "next-token embedding override" + `prev_hidden`.
    pub mtp_t_outs: GpuTensor,

    /// Per-step batched-rmsnorm output for the K-step batched lm_head.
    /// Shape `[max_n, n_embd]`. Caller of `mtp_head_apply_lm_head_batched`
    /// owns this scratch.
    pub mtp_lm_tmp: GpuTensor,

    /// Per-step FWHT-rotated x scratch for MagnumQuant lm_heads (MQ4/MQ3/MQ6).
    /// Shape `[max_n, n_embd]`. Unused for non-MQ lm_head dtypes.
    pub mtp_lm_rot: GpuTensor,

    /// Per-step batched lm_head logits output. Shape `[max_n, vocab]`.
    /// Caller `argmax_f32_batched` reads this for the K predicted tokens.
    pub mtp_lm_logits: GpuTensor,

    /// GPU-side argmax destination for the K-step batched argmax over
    /// `mtp_lm_logits`. Shape `[max_n]` i32 stored as F32 slots.
    pub mtp_lm_argmax: GpuTensor,

    /// Greedy device-token chain for compressed-serial MTP. Slot 0 is seeded
    /// with `last_committed`; step k writes slot k+1 after argmax/remap, and
    /// step k+1 embeds slot k directly from GPU memory. Shape `[max_n + 1]`
    /// i32 stored in F32-typed slots.
    pub mtp_token_chain: GpuTensor,

    /// Single-token embedding scratch used by the device-token-chain path.
    /// The existing MTP block forward consumes it through its embedding
    /// override hook, avoiding a larger forward-path refactor.
    pub mtp_token_embed: GpuTensor,

    /// Per-step absolute MTP positions for proposal-graph replay. Shape
    /// `[max_n]` i32 stored in F32 slots. The captured graph reads each slot
    /// by pointer, while the host refreshes the values before every launch.
    pub mtp_positions: GpuTensor,

    /// Takeover head-KV fill input rows, `[(verify_capacity + 1) × dim]` F32.
    /// Row 0 holds the pre-window `prev_hidden` (h_{cur_pos-1}); row `i > 0`
    /// holds verify hidden row `i - 1`, so row `i` pairs with verify token `i`
    /// exactly as decode pairs `(tok_p, h_{p-1})`.
    pub takeover_fill_hidden: GpuTensor,

    /// Batched head scratch + rotation scratch for the takeover fill, sized to
    /// `verify_capacity + 1` rows. Allocated by the first takeover window that
    /// takes the batched route.
    takeover_fill_batched: Option<(Qwen35MtpHeadBatchedScratch, GpuTensor)>,

    /// Captured q8 compressed proposal graph for the greedy device-token-chain
    /// path. It belongs to this state because the graph bakes scratch/weight
    /// pointers into captured kernel nodes.
    mtp_proposal_graph: Option<Graph>,
    ///
    /// Bounded per-state capture of the served compressed-serial narrow MTP
    /// trunk-verify forward (`n_verify = drafts + 1`, i.e. 2..=4 rows). The
    /// single live entry owns its instantiated graph plus the kernarg blobs
    /// its nodes point at. Destroyed before any buffer the capture references
    /// is freed (see `reset` / `free_gpu`), and torn down wholesale whenever
    /// scratch growth invalidates live pointers.
    mtp_verify_graph: Option<(Graph, GraphExec, Vec<Vec<u8>>)>,
    /// `n_verify` the live `mtp_verify_graph` entry was captured for.
    mtp_verify_graph_n: usize,
    /// Capture-identity guard for the live narrow MTP verify graph entry.
    ///
    /// Ownership invariant (stable for this state's lifetime): the per-state
    /// PBS / tape / verify buffers and every loaded weight are never freed or
    /// reallocated while the state lives; a reset/free destroys THIS graph
    /// before freeing any of those resources. So a stored pointer stays valid
    /// until the graph is destroyed — no dangling, and reused memory only ever
    /// appears after an intervening destroy.
    ///
    /// The key guards what a capture bakes in: model identity (output-norm
    /// buffer), attention width, the mutable KV mapping (`physical_cap`,
    /// `max_seq`, first K/V buffers), the recurrent owner (first DN matrix),
    /// and the global conversion/attention scratch pools that may grow between
    /// MTP windows. Those pools are grow-never-shrink: a moved pointer identifies
    /// growth, so pointer identity is sufficient — no buffer contents are hashed
    /// here and no converted activation values are cached. `None` (nothing
    /// captured) matches no live entry; this is not an exhaustive fingerprint
    /// for arbitrary public-field mutation nor a per-window config string.
    mtp_verify_graph_key: Option<MtpVerifyGraphKey>,
    /// `true` once one direct warm execution for the pending `n_verify` has
    /// completed (JIT + lazy scratch growth outside any capture). The next
    /// eligible window captures instead of warming again.
    mtp_verify_graph_warmed: bool,
    /// `n_verify` the pending warmup ran for. A window at a different width
    /// restarts the warm-then-capture sequence for its own width.
    mtp_verify_graph_warm_n: usize,
    /// Latched `true` on capture or launch failure for this state's lifetime.
    /// Unsupported windows use eager execution without setting this latch.
    mtp_verify_graph_disabled: bool,
    /// Windows the captured graph actually replayed (oracle replay proof).
    mtp_verify_graph_replays: u64,
    /// Successful captures (oracle capture proof; bounded small).
    mtp_verify_graph_captures: u64,
    /// Last window's verify mode tag: "direct" | "warmup" | "capture" |
    /// "replay" (oracle mode proof without log scraping).
    mtp_verify_graph_mode: &'static str,
    mtp_proposal_graph_exec: Option<GraphExec>,
    mtp_proposal_graph_blobs: Vec<Vec<u8>>,
    mtp_proposal_graph_seq_cap: usize,
    mtp_proposal_graph_warmed: bool,
    mtp_proposal_graph_disabled: bool,

    /// Optional FastMTP-style compressed batched-logits output, shape
    /// `[max_n * compressed_vocab_size]`. Populated by
    /// [`spec_step_mtp_compressed`] when the head ships a sidecar.
    /// Allocated lazily via [`MtpSpecState::ensure_compressed_lm_logits`].
    pub mtp_lm_logits_compressed: Option<GpuTensor>,

    /// Per-step top-2 index scratch for [`spec_step_mtp_compressed_serial`]
    /// with `p_min > 0`. Shape `[2]` i32 stored as F32 slots. Reused across
    /// K-chain steps within a cycle.
    pub mtp_topk_idx: GpuTensor,

    /// Per-step top-2 log-softmax-prob scratch matching `mtp_topk_idx`.
    /// Shape `[2]` F32. `top_logp[0]` is log P(argmax); compared against
    /// `log(p_min)` for the chain-truncation early-exit.
    pub mtp_topk_logp: GpuTensor,

    // ─── GPU sampling scratches (only used when sampling.temp > 0) ──────
    /// Result buffer for `gpu.sample_top_p`: shape `[2]` u32-as-F32. Slot 0
    /// = sampled token id, slot 1 = updated rng state.
    pub mtp_sample_result: GpuTensor,
    /// Repeat/presence/frequency history for the exact-AR penalty prepass:
    /// `PENALTY_REPEAT_CAP + verify_capacity + 1` u32 token ids stored in the
    /// F32-typed tensor (uploaded as raw bytes). Holds `state.penalty.tokens()`
    /// — base history (≤ window ≤ `PENALTY_REPEAT_CAP`) plus the window's
    /// drafts. Only written when `sampling.penalized()`.
    pub mtp_sample_repeat_buf: GpuTensor,
    /// Single-element index buf for the per-step p_draft gather. Shape `[1]`
    /// i32-as-F32. H2D'd with the sampled token id, fed to
    /// `softmax_prob_gather_batched_f32` with n_rows=1.
    pub mtp_gather_idx_draft: GpuTensor,
    /// Output buf for the per-step p_draft gather. Shape `[1]` F32.
    pub mtp_gather_prob_draft: GpuTensor,
    /// Batched index buf for the K-candidate p_target gather. Shape `[max_n]`
    /// i32-as-F32. H2D'd with `candidates[0..drafts_generated]`.
    pub mtp_gather_idx_verify: GpuTensor,
    /// Batched output buf for the K-candidate p_target gather. Shape `[max_n]`
    /// F32. D2H'd after `softmax_prob_gather_batched_f32(n_rows=drafts_generated)`.
    pub mtp_gather_prob_verify: GpuTensor,
    /// On-device rng state for `gpu.sample_top_p`. Threaded through both
    /// draft and bonus sampling calls within a cycle and across cycles.
    /// Reseeded by [`Self::set_sampling`].
    pub gpu_rng_state: u32,

    /// Maximum number of MTP candidates per cycle (chain depth).
    pub max_n: usize,

    /// Verify capacity (max external candidate window) sizing trunk verify
    /// buffers (`verify_hidden` / `verify_logits` / `verify_rot` /
    /// `verify_argmax`) and `trunk_pbs` / `trunk_gdn_tape`. Must be >=
    /// `max_n`; proposal buffers (`mtp_t_outs`, `mtp_lm_*`, etc.) remain sized
    /// to `max_n` so ordinary MTP draft depth never inflates with PLD capacity.
    pub verify_capacity: usize,

    /// Optional draft-confidence cutoff for [`spec_step_mtp_compressed_serial`].
    /// When `p_min > 0`, the K-chain truncates early at step k if the draft's
    /// top-1 softmax prob falls below `p_min` — analog of llama.cpp's
    /// `--spec-draft-p-min` (PR #22673). Default 0.0 = disabled. Set via
    /// [`Self::set_p_min`]. The current step's candidate IS kept (we already
    /// computed it); only steps k+1..max_n are skipped.
    pub p_min: f32,

    /// Sampling config (temp/top_k/top_p/min_p). When `temp == 0.0` (default)
    /// the spec-step uses the legacy greedy/argmax-match accept rule. When
    /// `temp > 0`, the K-chain samples per step + the trunk applies
    /// residual-style acceptance (matches Unsloth/llama.cpp #22673 canonical
    /// MTP path). Set via [`Self::set_sampling`].
    pub sampling: MtpSamplingConfig,

    /// RNG state for sampling. Seeded via [`Self::set_sampling`].
    /// Single global stream across all spec-decode positions in a generation.
    pub rng: MtpRng,

    /// Per-request exact-AR penalty history (window 0 = penalties inactive;
    /// the drafter installs `PenaltyHistory::new(w)` per request and calls
    /// `begin_window(emitted)` before every step). Rows are penalized from it
    /// by [`mtp_verify_accept`] and the full-vocab sampled draft.
    pub penalty: hipfire_runtime::spec_sampling::PenaltyHistory,

    /// Per-window host-side penalty cost, filled only under `HIPFIRE_MTP_PHASE_TIMING=1`.
    pub penalty_timing: MtpPenaltyTiming,
}

impl MtpSpecState {
    /// Allocate all per-generation buffers, sizing the trunk DN snapshot
    /// against the live `target.dn_state`. Defaults MTP head KV to Q8 — see
    /// [`Self::new_for_slot_with_kv_mode`] for explicit selection.
    pub fn new_for_slot(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
    ) -> HipResult<Self> {
        Self::new_for_slot_with_kv_mode(gpu, target, head, max_n, crate::mtp_head::MtpKvMode::Q8)
    }

    /// Component-level constructor: takes `&Qwen35Config` and `&DeltaNetState`
    /// directly instead of a `&ModelSlot`. Used by the slot engine where
    /// per-slot state lives in `Rig`'s field layout, not in a `ModelSlot`.
    pub fn new_for_components(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        dn_state: &DeltaNetState,
        head: &Qwen35MtpHead,
        max_n: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
    ) -> HipResult<Self> {
        Self::new_for_components_with_verify_capacity(
            gpu, config, dn_state, head, max_n, max_n, kv_mode,
        )
    }

    /// Component-level constructor with explicit `verify_capacity`.
    pub fn new_for_components_with_verify_capacity(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        dn_state: &DeltaNetState,
        head: &Qwen35MtpHead,
        max_n: usize,
        verify_capacity: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
    ) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "MtpSpecState::new_for_components: max_n must be ≥ 1 (chain depth)"
        );
        assert!(
            verify_capacity >= max_n,
            "MtpSpecState: verify_capacity {verify_capacity} must be >= max_n {max_n}"
        );
        let dim = config.dim;
        let vocab = config.vocab_size;
        assert_eq!(
            head.config.n_embd, dim,
            "MtpSpecState: trunk dim={dim} but head n_embd={}",
            head.config.n_embd,
        );
        assert_eq!(
            head.config.vocab_size, vocab,
            "MtpSpecState: trunk vocab={vocab} but head vocab={}",
            head.config.vocab_size,
        );

        let prev_hidden = gpu.alloc_tensor(&[dim], DType::F32)?;
        let verify_hidden = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;
        let verify_logits = gpu.alloc_tensor(&[(verify_capacity + 1) * vocab], DType::F32)?;
        let verify_rot = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;
        let verify_argmax = gpu.alloc_tensor(&[verify_capacity + 1], DType::F32)?;
        let trunk_snap = DeltaNetSnapshot::new_for(gpu, dn_state)?;
        let (trunk_snap_stream, trunk_snap_start_event) = if mtp_snapshot_overlap_enabled_from_env()
        {
            (
                Some(gpu.hip.stream_create()?),
                Some(gpu.hip.event_create()?),
            )
        } else {
            (None, None)
        };
        let trunk_pbs = qwen35::PrefillBatchScratch::new(gpu, config, verify_capacity + 1)?;
        let trunk_gdn_tape = GdnTape::new_for_config(gpu, config, verify_capacity + 1)?;
        let mtp_scratch = Qwen35MtpHeadScratch::new(gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new_with_kv_mode(gpu, &head.config, kv_mode)?;

        let mtp_t_outs = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_tmp = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_rot = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_logits = gpu.alloc_tensor(&[max_n * vocab], DType::F32)?;
        let mtp_lm_argmax = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_token_chain = gpu.alloc_tensor(&[max_n + 1], DType::F32)?;
        let mtp_token_embed = gpu.alloc_tensor(&[dim], DType::F32)?;
        let mtp_positions = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_topk_idx = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_topk_logp = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_sample_result = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_sample_repeat_buf =
            gpu.alloc_tensor(&[PENALTY_REPEAT_CAP + verify_capacity + 1], DType::F32)?;
        let mtp_gather_idx_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_prob_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_idx_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_gather_prob_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;

        // Takeover fill rows (land MTP fix): row 0 holds the pre-window hidden.
        let takeover_fill_hidden = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;
        Ok(Self {
            prev_hidden,
            verify_hidden,
            verify_logits,
            verify_rot,
            verify_argmax,
            trunk_snap,
            trunk_snap_stream,
            trunk_snap_start_event,
            trunk_pbs,
            trunk_gdn_tape,
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_tmp,
            mtp_lm_rot,
            mtp_lm_logits,
            mtp_lm_argmax,
            mtp_token_chain,
            mtp_token_embed,
            mtp_positions,
            mtp_proposal_graph: None,
            mtp_verify_graph: None,
            mtp_verify_graph_n: 0,
            mtp_verify_graph_key: None,
            mtp_verify_graph_warmed: false,
            mtp_verify_graph_warm_n: 0,
            mtp_verify_graph_disabled: false,
            mtp_verify_graph_replays: 0,
            mtp_verify_graph_captures: 0,
            mtp_verify_graph_mode: "direct",
            mtp_proposal_graph_exec: None,
            mtp_proposal_graph_blobs: Vec::new(),
            mtp_proposal_graph_seq_cap: 0,
            mtp_proposal_graph_warmed: false,
            mtp_proposal_graph_disabled: false,
            mtp_lm_logits_compressed: None,
            mtp_topk_idx,
            mtp_topk_logp,
            mtp_sample_result,
            mtp_sample_repeat_buf,
            mtp_gather_idx_draft,
            mtp_gather_prob_draft,
            mtp_gather_idx_verify,
            mtp_gather_prob_verify,
            gpu_rng_state: 42,
            max_n,
            verify_capacity,
            p_min: default_mtp_p_min(gpu.arch.as_str()),
            sampling: MtpSamplingConfig::default(),
            rng: MtpRng::new(42),
            penalty: hipfire_runtime::spec_sampling::PenaltyHistory::default(),
            penalty_timing: MtpPenaltyTiming::default(),
            prev_hidden_pos: None,
            takeover_fill_hidden,
            takeover_fill_batched: None,
        })
    }

    /// Like [`Self::new_for_slot`] but allocates the MTP head's KV cache in
    /// the requested format. Used by `mtp_only_demo` to A/B kv-mode variants
    /// (q8/asym3/fwht4) per the 2026-05-16 feat/fwht prose-τ findings.
    pub fn new_for_slot_with_kv_mode(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
    ) -> HipResult<Self> {
        Self::new_for_slot_with_kv_mode_and_verify_capacity(
            gpu, target, head, max_n, max_n, kv_mode,
        )
    }

    /// Like [`Self::new_for_slot_with_kv_mode`] but allows `verify_capacity`
    /// to oversize trunk verify buffers (`verify_hidden` / `verify_logits` /
    /// `verify_rot` / `verify_argmax`, `trunk_pbs`, `trunk_gdn_tape`) so an
    /// external PLD/n-gram window larger than native MTP depth can still be
    /// verified in one batch. `max_n` remains the proposal-buffer/alloc depth
    /// and the per-call K clamp for ordinary MTP; `verify_capacity` is only
    /// the verify-batch ceiling. `verify_capacity` must be >= `max_n`.
    pub fn new_for_slot_with_kv_mode_and_verify_capacity(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
        verify_capacity: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
    ) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "MtpSpecState::new_for_slot: max_n must be ≥ 1 (chain depth)"
        );
        assert!(
            verify_capacity >= max_n,
            "MtpSpecState: verify_capacity {verify_capacity} must be >= max_n {max_n}"
        );
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        assert_eq!(
            head.config.n_embd, dim,
            "MtpSpecState: trunk dim={dim} but head n_embd={}",
            head.config.n_embd,
        );
        assert_eq!(
            head.config.vocab_size, vocab,
            "MtpSpecState: trunk vocab={vocab} but head vocab={}",
            head.config.vocab_size,
        );

        let prev_hidden = gpu.alloc_tensor(&[dim], DType::F32)?;
        let verify_hidden = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;
        let verify_logits = gpu.alloc_tensor(&[(verify_capacity + 1) * vocab], DType::F32)?;
        let verify_rot = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;
        let verify_argmax = gpu.alloc_tensor(&[verify_capacity + 1], DType::F32)?;
        let trunk_snap = DeltaNetSnapshot::new_for(gpu, &target.dn_state)?;
        // Snapshot overlap resources are construction-time only; set the env
        // before creating MtpSpecState when comparing the opt-in path.
        let (trunk_snap_stream, trunk_snap_start_event) = if mtp_snapshot_overlap_enabled_from_env()
        {
            (
                Some(gpu.hip.stream_create()?),
                Some(gpu.hip.event_create()?),
            )
        } else {
            (None, None)
        };
        let trunk_pbs = qwen35::PrefillBatchScratch::new(gpu, &target.config, verify_capacity + 1)?;
        let trunk_gdn_tape = GdnTape::new_for_config(gpu, &target.config, verify_capacity + 1)?;
        let mtp_scratch = Qwen35MtpHeadScratch::new(gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new_with_kv_mode(gpu, &head.config, kv_mode)?;

        // Per-step batched-lm_head scratch (Task 10b). Sized to max_n, not
        // verify_capacity — ordinary MTP depth never inflates.
        let mtp_t_outs = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_tmp = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_rot = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_logits = gpu.alloc_tensor(&[max_n * vocab], DType::F32)?;
        let mtp_lm_argmax = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_token_chain = gpu.alloc_tensor(&[max_n + 1], DType::F32)?;
        let mtp_token_embed = gpu.alloc_tensor(&[dim], DType::F32)?;
        let mtp_positions = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let takeover_fill_hidden = gpu.alloc_tensor(&[(verify_capacity + 1) * dim], DType::F32)?;

        // Per-step top-2 scratches for p_min early-exit (16 B total, always
        // allocated — only used when state.p_min > 0).
        let mtp_topk_idx = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_topk_logp = gpu.alloc_tensor(&[2], DType::F32)?;

        // GPU sampling scratches (always allocated; only used when temp > 0).
        // All are tiny so the unconditional alloc is fine.
        let mtp_sample_result = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_sample_repeat_buf =
            gpu.alloc_tensor(&[PENALTY_REPEAT_CAP + verify_capacity + 1], DType::F32)?;
        let mtp_gather_idx_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_prob_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_idx_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_gather_prob_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;

        Ok(Self {
            prev_hidden,
            prev_hidden_pos: None,
            verify_hidden,
            verify_logits,
            verify_rot,
            verify_argmax,
            trunk_snap,
            trunk_snap_stream,
            trunk_snap_start_event,
            trunk_pbs,
            trunk_gdn_tape,
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_tmp,
            mtp_lm_rot,
            mtp_lm_logits,
            mtp_lm_argmax,
            mtp_token_chain,
            mtp_token_embed,
            mtp_positions,
            takeover_fill_hidden,
            takeover_fill_batched: None,
            mtp_proposal_graph: None,
            mtp_verify_graph: None,
            mtp_verify_graph_n: 0,
            mtp_verify_graph_key: None,
            mtp_verify_graph_warmed: false,
            mtp_verify_graph_warm_n: 0,
            mtp_verify_graph_disabled: false,
            mtp_verify_graph_replays: 0,
            mtp_verify_graph_captures: 0,
            mtp_verify_graph_mode: "direct",
            mtp_proposal_graph_exec: None,
            mtp_proposal_graph_blobs: Vec::new(),
            mtp_proposal_graph_seq_cap: 0,
            mtp_proposal_graph_warmed: false,
            mtp_proposal_graph_disabled: false,
            mtp_lm_logits_compressed: None,
            mtp_topk_idx,
            mtp_topk_logp,
            mtp_sample_result,
            mtp_sample_repeat_buf,
            mtp_gather_idx_draft,
            mtp_gather_prob_draft,
            mtp_gather_idx_verify,
            mtp_gather_prob_verify,
            gpu_rng_state: 42,
            max_n,
            verify_capacity,
            p_min: default_mtp_p_min(gpu.arch.as_str()),
            sampling: MtpSamplingConfig::default(),
            rng: MtpRng::new(42),
            penalty: hipfire_runtime::spec_sampling::PenaltyHistory::default(),
            penalty_timing: MtpPenaltyTiming::default(),
        })
    }

    /// Device bytes the first request's lazily created `MtpSpecState` takes
    /// ([`Self::new_for_slot_with_kv_mode_and_verify_capacity`] with
    /// `MtpKvMode::Q8`, then the compressed-vocab scratches when the head
    /// ships them, as `Qwen35MtpDrafter::ensure_state` builds it), from shape
    /// arithmetic alone. `trunk_snapshot_bytes` is
    /// `DeltaNetSnapshot::bytes_for(dn_state)` (the `trunk_snap` owner); the
    /// head KV and flash partials are sized by `head.config.max_seq`.
    pub fn requested_bytes(
        arch: &str,
        config: &Qwen35Config,
        trunk_snapshot_bytes: usize,
        head_config: &mtp_head::Qwen35MtpHeadConfig,
        compressed_vocab: Option<usize>,
        max_n: usize,
        verify_capacity: usize,
    ) -> Result<usize, String> {
        let dim = config.dim;
        let vocab = config.vocab_size;
        let rows = verify_capacity + 1;
        let cvs = compressed_vocab;
        let f32_elems = dim // prev_hidden
            + rows * dim // verify_hidden
            + rows * vocab // verify_logits
            + rows * dim // verify_rot
            + rows // verify_argmax
            + 3 * max_n * dim // mtp_t_outs, mtp_lm_tmp, mtp_lm_rot
            + max_n * vocab // mtp_lm_logits
            + max_n // mtp_lm_argmax
            + (max_n + 1) // mtp_token_chain
            + dim // mtp_token_embed
            + max_n // mtp_positions
            + rows * dim // takeover_fill_hidden
            + 2 + 2 + 2 // topk idx/logp, sample result
            + PENALTY_REPEAT_CAP + rows // mtp_sample_repeat_buf
            + 1 + 1 + 2 * max_n // gather idx/prob draft + verify
            + cvs.map_or(0, |c| max_n * c); // mtp_lm_logits_compressed
        let pbs = PrefillBatchScratch::projected_allocation_bytes(config, rows, true)
            .map_err(|e| format!("MTP trunk PBS sizing: {e}"))?;
        let pbs = usize::try_from(pbs).map_err(|_| "MTP trunk PBS sizing overflow".to_string())?;
        Ok(f32_elems * 4
            + trunk_snapshot_bytes
            + pbs
            + GdnTape::requested_bytes(config, rows, true)
            + Qwen35MtpHeadScratch::requested_bytes(arch, head_config, cvs)
            + Qwen35MtpHeadKvCache::q8_requested_bytes(head_config))
    }

    /// Configure sampling (temp/top_p/top_k/min_p) and reseed the per-state RNG.
    /// `cfg.temp == 0.0` keeps the legacy greedy path. `cfg.temp > 0` enables
    /// residual-acceptance sampling per the Unsloth/llama.cpp MTP recipe.
    /// The repeat/presence/frequency penalties in `cfg` are kept as-is; they take
    /// effect (see [`MtpSamplingConfig::penalized`]) only together with a
    /// `state.penalty` history the drafter installs and advances per window.
    /// Reseeds BOTH the host RNG (used for the residual accept rule) and the
    /// on-device RNG (used by `gpu.sample_top_p`).
    pub fn set_sampling(&mut self, cfg: MtpSamplingConfig, seed: u64) {
        assert!(
            cfg.temp >= 0.0,
            "set_sampling: temp must be >= 0.0, got {}",
            cfg.temp
        );
        assert!(
            cfg.top_p > 0.0 && cfg.top_p <= 1.0,
            "set_sampling: top_p must be in (0,1], got {}",
            cfg.top_p
        );
        assert!(
            cfg.min_p >= 0.0 && cfg.min_p <= 1.0,
            "set_sampling: min_p must be in [0,1], got {}",
            cfg.min_p
        );
        self.sampling = cfg;
        self.rng = MtpRng::new(seed);
        // GPU rng uses u32; mix the lower + upper halves so different seeds
        // produce different on-device streams.
        self.gpu_rng_state = ((seed >> 32) as u32) ^ (seed as u32) | 1;
    }

    /// Set the draft-confidence threshold for compressed-serial K-chain
    /// truncation. Pass `0.0` (default) to disable. Typical values: 0.5-0.8.
    /// llama.cpp's PR #22673 ships `0.75` as the recommended starting point.
    pub fn set_p_min(&mut self, p_min: f32) {
        assert!(
            (0.0..=1.0).contains(&p_min),
            "MtpSpecState::set_p_min: p_min must be in [0.0, 1.0], got {p_min}"
        );
        self.p_min = p_min;
    }

    /// Allocate `mtp_lm_logits_compressed` shape `[max_n * cvs]` for the
    /// compressed batched-lm_head dispatch path. Idempotent — reallocates
    /// only if the existing tensor has the wrong size. Call once after
    /// loading a head with a sidecar (cvs = `head.weights.compressed_vocab_size`).
    pub fn ensure_compressed_lm_logits(&mut self, gpu: &mut Gpu, cvs: usize) -> HipResult<()> {
        let needed = self.max_n * cvs;
        if let Some(existing) = self.mtp_lm_logits_compressed.as_ref() {
            if existing.numel() == needed {
                return Ok(());
            }
            if let Some(old) = self.mtp_lm_logits_compressed.take() {
                let _ = gpu.free_tensor(old);
            }
        }
        self.mtp_lm_logits_compressed = Some(gpu.alloc_tensor(&[needed], DType::F32)?);
        Ok(())
    }

    /// Capture the trunk's post-output-norm hidden after a `forward_scratch`
    /// call. Source is `target.scratch.tmp` (same buffer lm_head reads from).
    /// Call once after prefill's last `forward_scratch` to seed the cycle.
    pub fn capture_prev_hidden_from_scratch_tmp(
        &mut self,
        gpu: &Gpu,
        target_scratch_tmp: &GpuTensor,
        dim: usize,
        pos: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf,
            0,
            &target_scratch_tmp.buf,
            0,
            dim * 4,
        )?;
        self.prev_hidden_pos = Some(pos);
        Ok(())
    }

    /// Capture the trunk's post-output-norm hidden from a row of
    /// `verify_hidden` (per-position output of the batched verify forward).
    /// Used after spec_step accepts to set up next cycle's MTP `prev_hidden`
    /// from the verify slot corresponding to the last committed token,
    /// avoiding the cost of a separate single-token forward.
    pub fn capture_prev_hidden_from_verify_row(
        &mut self,
        gpu: &Gpu,
        cur_pos: usize,
        row: usize,
        dim: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf,
            0,
            &self.verify_hidden.buf,
            row * dim * 4,
            dim * 4,
        )?;
        self.prev_hidden_pos = Some(cur_pos + row);
        Ok(())
    }

    /// Drafter-local reset for a fresh conversation. Zeros the MTP head KV
    /// (stale absolute positions from the prior turn must not be re-read) and
    /// destroys any captured proposal graph (its node positions are turn-stale).
    /// The trunk's KV + DeltaNet reset is the daemon's concern (it owns the
    /// bundle and calls `reset_recurrent`); this clears only head-local state.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        destroy_mtp_proposal_graph(gpu, self);
        destroy_mtp_verify_graph(gpu, self);
        self.mtp_proposal_graph_warmed = false;
        self.mtp_proposal_graph_seq_cap = 0;
        self.mtp_kv.reset(gpu)?;
        self.prev_hidden_pos = None;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        gpu.bind_thread_or_warn();
        // Drain launches and destroy their nodes before releasing captured buffers.
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_synchronize(stream);
        }
        if let Some(exec) = self.mtp_proposal_graph_exec {
            let _ = gpu.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.mtp_proposal_graph {
            let _ = gpu.hip.graph_destroy(graph);
        }
        drop(self.mtp_proposal_graph_blobs);
        if let Some((graph, exec, _blobs)) = self.mtp_verify_graph {
            let _ = gpu.hip.graph_exec_destroy(exec);
            let _ = gpu.hip.graph_destroy(graph);
        }
        let _ = gpu.free_tensor(self.prev_hidden);
        let _ = gpu.free_tensor(self.verify_hidden);
        let _ = gpu.free_tensor(self.verify_logits);
        let _ = gpu.free_tensor(self.verify_rot);
        let _ = gpu.free_tensor(self.verify_argmax);
        let _ = gpu.free_tensor(self.mtp_t_outs);
        let _ = gpu.free_tensor(self.mtp_lm_tmp);
        let _ = gpu.free_tensor(self.mtp_lm_rot);
        let _ = gpu.free_tensor(self.mtp_lm_logits);
        let _ = gpu.free_tensor(self.mtp_lm_argmax);
        let _ = gpu.free_tensor(self.mtp_token_chain);
        let _ = gpu.free_tensor(self.mtp_token_embed);
        let _ = gpu.free_tensor(self.mtp_positions);
        let _ = gpu.free_tensor(self.takeover_fill_hidden);
        if let Some((scratch, rot)) = self.takeover_fill_batched {
            scratch.free_gpu(gpu);
            let _ = gpu.free_tensor(rot);
        }
        if let Some(lc) = self.mtp_lm_logits_compressed {
            let _ = gpu.free_tensor(lc);
        }
        let _ = gpu.free_tensor(self.mtp_topk_idx);
        let _ = gpu.free_tensor(self.mtp_topk_logp);
        let _ = gpu.free_tensor(self.mtp_sample_result);
        let _ = gpu.free_tensor(self.mtp_sample_repeat_buf);
        let _ = gpu.free_tensor(self.mtp_gather_idx_draft);
        let _ = gpu.free_tensor(self.mtp_gather_prob_draft);
        let _ = gpu.free_tensor(self.mtp_gather_idx_verify);
        let _ = gpu.free_tensor(self.mtp_gather_prob_verify);
        // DeltaNetSnapshot holds DeviceBuffers which have no Drop impl —
        // use free_gpu to release the GPU allocations rather than bare drop.
        self.trunk_snap.free_gpu(gpu);
        if let Some(event) = self.trunk_snap_start_event {
            let _ = gpu.hip.event_destroy(event);
        }
        if let Some(stream) = self.trunk_snap_stream {
            let _ = gpu.hip.stream_destroy(stream);
        }
        self.trunk_pbs.free_gpu(gpu);
        self.trunk_gdn_tape.free_gpu(gpu);
        self.mtp_scratch.free_gpu(gpu);
        // Qwen35MtpHeadKvCache::free_gpu does `drop(inner)` which does not
        // release GPU memory (llama::KvCache has no Drop). Call the inner
        // KvCache's own free_gpu directly to properly hipFree each tensor.
        let _ = self.mtp_kv.inner.free_gpu(gpu);
    }
}

// ─── One spec-decode cycle ───────────────────────────────────────────────

/// Result of one MTP spec-decode cycle.
#[derive(Debug, Clone)]
pub struct MtpSpecResult {
    /// Tokens committed THIS cycle (excludes the seed `last_committed` from
    /// the previous cycle; includes the bonus when no early EOS).
    /// Length == `advance`.
    pub committed: Vec<u32>,
    /// Number of MTP candidates that matched trunk argmax (not counting the
    /// bonus). 0 = MTP wasted, max_n = full acceptance.
    pub accept_count: usize,
    /// True iff any committed token equals `eos_token_id` (caller stops).
    pub hit_eos: bool,
    /// Position advance (= committed.len()). Caller's `cur_pos += advance`.
    pub advance: usize,
    /// Number of MTP draft candidates ACTUALLY generated this cycle (≤ max_n).
    /// Equals max_n unless p_min early-exit truncated the chain. The trunk
    /// verify batch is sized `drafts_generated + 1`.
    pub drafts_generated: usize,
    /// True iff `p_min` early-exit fired this cycle (caller can sum across
    /// cycles to compute the "% cycles truncated" stat).
    pub chain_truncated: bool,
    /// True iff this cycle was a full-accept (advance == drafts_generated + 1
    /// without EOS) so the post-verify replay was skipped — verify's KV +
    /// DN state already match what replay would have produced. Currently
    /// only set by [`spec_step_mtp_compressed_serial`]; other spec_step
    /// variants always replay and report `false`.
    pub replay_skipped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GreedyTrunkSpineAccept {
    committed: Vec<u32>,
    accept_count: usize,
    hit_eos: bool,
}

fn build_trunk_spine_verify_tokens(last_committed: u32, candidates: &[u32]) -> Vec<u32> {
    let mut verify_tokens = Vec::with_capacity(candidates.len() + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(candidates);
    verify_tokens
}

fn embed_device_token_into(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    out: &GpuTensor,
    token_id: &GpuTensor,
    dim: usize,
) -> HipResult<()> {
    match weights.embd_format {
        llama::EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256_batched(&weights.token_embd, out, token_id, 1, dim)
        }
        llama::EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8_batched(&weights.token_embd, out, token_id, 1, dim)
        }
        other => panic!("device-token MTP chain does not support embedding format {other:?}"),
    }
}

fn upload_mtp_proposal_graph_inputs(
    gpu: &mut Gpu,
    state: &MtpSpecState,
    last_committed: u32,
    cur_pos: usize,
    max_n: usize,
) -> HipResult<()> {
    assert!(
        cur_pos + max_n <= state.mtp_kv.max_seq,
        "MTP proposal positions [{}..{}) exceed kv max_seq {}",
        cur_pos,
        cur_pos + max_n,
        state.mtp_kv.max_seq,
    );

    let seed_token = last_committed as i32;
    let seed_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(&seed_token as *const i32 as *const u8, 4) };
    gpu.hip
        .memcpy_htod(&state.mtp_token_chain.buf, seed_bytes)?;

    let positions: Vec<i32> = (0..max_n).map(|k| (cur_pos + k) as i32).collect();
    let pos_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, max_n * 4) };
    gpu.hip.memcpy_htod(&state.mtp_positions.buf, pos_bytes)
}

fn mtp_proposal_graph_seq_cap(cur_pos: usize, max_n: usize, kv_max_seq: usize) -> usize {
    let needed = cur_pos + max_n;
    needed.next_power_of_two().max(256).min(kv_max_seq)
}

#[allow(clippy::too_many_arguments)]
fn run_mtp_proposal_graph_body_q8(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    max_n: usize,
    dim: usize,
    cvs: usize,
    seq_cap: usize,
) -> HipResult<()> {
    debug_assert_eq!(state.mtp_kv.kv_mode, crate::mtp_head::MtpKvMode::Q8);
    let dim_bytes = dim * 4;
    let logits_c = state
        .mtp_scratch
        .logits_compressed
        .as_ref()
        .expect("proposal graph requires compressed logits scratch");
    let vocab_map_gpu = head
        .weights
        .lm_head_draft_vocab_map_gpu
        .as_ref()
        .expect("proposal graph requires GPU vocab_map");
    let argmax_view = state.mtp_lm_argmax.sub_offset(0, 1);

    for k in 0..max_n {
        let token_slot = state.mtp_token_chain.sub_offset(k, 1);
        embed_device_token_into(gpu, weights, &state.mtp_token_embed, &token_slot, dim)?;

        let pos_slot = state.mtp_positions.sub_offset(k, 1);
        if k == 0 {
            mtp_head::mtp_head_forward_block_only_with_pos_buf(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,
                &state.prev_hidden,
                Some(&state.mtp_token_embed),
                &pos_slot.buf,
                cur_pos + k,
                seq_cap,
                weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only_with_pos_buf(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,
                &prev_row,
                Some(&state.mtp_token_embed),
                &pos_slot.buf,
                cur_pos + k,
                seq_cap,
                weights,
            )?;
        }
        mtp_head::mtp_head_apply_lm_head_draft(gpu, head, &state.mtp_scratch)?;
        gpu.argmax_token_chain_f32(
            logits_c,
            &argmax_view,
            &state.mtp_token_chain,
            Some(vocab_map_gpu),
            cvs,
            k + 1,
        )?;

        if k + 1 < max_n {
            gpu.memcpy_dtod_at_auto(
                &state.mtp_t_outs.buf,
                k * dim_bytes,
                &state.mtp_scratch.t_mtp_out.buf,
                0,
                dim_bytes,
            )?;
        }
    }
    Ok(())
}

/// Shared MTP hipGraph capture begin/end/abort used by both the proposal
/// graph and the narrow verify graph. One boring trio instead of two
/// copy/pasted wrappers: the HIP mechanics are identical, only the recorded
/// body differs.
fn mtp_graph_begin_capture(gpu: &mut Gpu) -> HipResult<()> {
    gpu.graphs.capture_blobs.clear();
    gpu.graphs.capture_mode = true;
    let stream = gpu
        .active_stream
        .as_ref()
        .expect("MTP graph capture requires an explicit stream");
    gpu.hip.stream_begin_capture(stream, 0)
}

/// End capture and instantiate. An instantiation failure destroys the
/// returned graph before propagating: the half-built graph must never leak
/// nor enter an entry, and its kernarg blobs die with the failed capture.
fn mtp_graph_end_capture(gpu: &mut Gpu) -> HipResult<(Graph, GraphExec, Vec<Vec<u8>>)> {
    gpu.graphs.capture_mode = false;
    let stream = gpu.active_stream.as_ref().unwrap();
    let graph = gpu.hip.stream_end_capture(stream)?;
    match gpu.hip.graph_instantiate(&graph) {
        Ok(exec) => {
            let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
            Ok((graph, exec, blobs))
        }
        Err(e) => {
            let _ = gpu.hip.graph_destroy(graph);
            gpu.graphs.capture_blobs.clear();
            Err(e)
        }
    }
}

/// Abort an in-flight capture. Always closes the actual HIP capture — even
/// when our marker is already false — because `stream_end_capture` is the
/// only call that exits the hardware capture state. A graph returned for a
/// discarded capture is destroyed, never stored: no blobs are released
/// while capture nodes live.
fn mtp_graph_abort_capture(gpu: &mut Gpu) {
    if let Some(stream) = gpu.active_stream.as_ref() {
        if let Ok(graph) = gpu.hip.stream_end_capture(stream) {
            let _ = gpu.hip.graph_destroy(graph);
        }
    }
    gpu.graphs.capture_mode = false;
    gpu.graphs.capture_blobs.clear();
}

fn destroy_mtp_verify_graph(gpu: &mut Gpu, state: &mut MtpSpecState) {
    if state.mtp_verify_graph.is_some() {
        gpu.bind_thread_or_warn();
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_synchronize(stream);
        }
    }
    if let Some((graph, exec, _blobs)) = state.mtp_verify_graph.take() {
        let _ = gpu.hip.graph_exec_destroy(exec);
        let _ = gpu.hip.graph_destroy(graph);
    }
    state.mtp_verify_graph_n = 0;
    state.mtp_verify_graph_key = None;
    state.mtp_verify_graph_warmed = false;
    state.mtp_verify_graph_warm_n = 0;
}
/// Narrow grouped Path-2 eligibility for the served compressed-serial MTP
/// verify graph. All conditions must hold; anything else fails closed to the
/// exact eager forward:
/// - greedy single-GPU chain (`use_sampling == false`, no external PLD
///   candidates, no snapshot-overlap side stream, no tree mask);
/// - actual `n_verify` in 2..=4 (the measured K=3 envelope plus K=1/K=2);
/// - exact gfx1201 (the only validated single-GPU arm);
/// - Q8 KV with mapped capacity and window end at most 4096; no eager-only
///   multirow-attention crossover, nested capture, or retained PM4 route;
/// - every layer batch-admissible AND every MoE layer on grouped Path 2
///   (broken all-indexed Path 1 stays eager; never fixed here);
/// - at least one routed projection is authoritatively host-mapped; other
///   layers may remain resident;
/// - `HIPFIRE_GRAPH` master control and `HIPFIRE_VERIFY_GRAPH` verify control
///   both admit (explicit opt-outs respected; the gfx1201+MQ4G256 quarantine
///   in `dflash_verify_graph_env_eligible` applies unchanged).
fn mtp_verify_graph_eligible(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &KvCache,
    state: &MtpSpecState,
    n_verify: usize,
    use_sampling: bool,
    is_external: bool,
    overlap_trunk_snap: bool,
    cur_pos: usize,
) -> bool {
    if state.mtp_verify_graph_disabled {
        return false;
    }
    if use_sampling || is_external || overlap_trunk_snap {
        return false;
    }
    // Lower bound of 2 is load-bearing: a 1-row window is decode, not a verify
    // window, so it must not take the verify-graph route.
    if !(2..=NARROW_VERIFY_MAX_ROWS).contains(&n_verify) {
        return false;
    }
    if gpu.arch.as_str() != "gfx1201" {
        return false;
    }
    if cur_pos.checked_add(n_verify).is_none_or(|end| end > 4096)
        || !kv_cache.quant_q8
        || kv_cache.physical_cap > 4096
        || gpu.graphs.capture_mode
        || gpu.replay.is_recording()
        || gpu.replay.prepared_pm4_route_active()
        || config.num_experts == 0
        || weights.ep_shard.is_some()
        || weights.pager.is_some()
        || gpu.graphs.host_cpu_weights
    {
        return false;
    }
    // The eager-only multirow route is suppressed during capture. Respect its
    // developer-configurable threshold rather than changing verifier math.
    if n_verify == 4
        && hipfire_dispatch::pipeline::batched_attention::fa_pertoken_min_ctx(gpu.arch.as_str())
            .is_some_and(|threshold| cur_pos + n_verify > threshold)
    {
        return false;
    }
    if gpu.flags.graph_forward == Some(false) {
        return false;
    }
    let verify_graph_env = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH").ok();
    if !crate::speculative::dflash_verify_graph_env_eligible(
        gpu.arch.as_str(),
        weights.output.gpu_dtype,
        verify_graph_env.as_deref(),
    ) {
        return false;
    }
    let mut has_host_projection = false;
    for lw in &weights.layers {
        if qwen35::qwen35_layer_batch_admissible(lw, config, gpu.arch.as_str()).is_err() {
            return false;
        }
        // Batch-aware Path-2 check: mirrors the live MoE program's own
        // `resolve_with_batch_ctx` (SpeculativeVerify, batch = n_verify), so
        // eligibility agrees with the exact route the eager forward selects
        // — including the narrow-verify and graded-mixed rules.
        let ffn_opt: Option<&crate::qwen35::MoeFfnWeights> = match lw {
            crate::qwen35::LayerWeights::DeltaNetMoe(l) => Some(&l.ffn),
            crate::qwen35::LayerWeights::FullAttnMoe(l) => Some(&l.ffn),
            _ => None,
        };
        if let Some(ffn) = ffn_opt {
            let (per_expert_gate_up, per_expert_down) = ffn.per_expert_tier_tables();
            let (host_gate_up, host_down) = ffn.host_mapped_projections();
            let dtypes = hipfire_dispatch::families::moe::MoeDtypes {
                router: ffn.router.gpu_dtype,
                shared: Some(hipfire_dispatch::families::moe::MoeSharedDtypes {
                    selector: ffn.shared_expert_gate.gpu_dtype,
                    gate: ffn.shared_expert.gate.gpu_dtype,
                    up: ffn.shared_expert.up.gpu_dtype,
                    down: ffn.shared_expert.down.gpu_dtype,
                }),
                experts_all_gate_up_mq4: ffn
                    .experts
                    .iter()
                    .all(|e| matches!(e.gate_up.gpu_dtype, DType::MQ4G256 | DType::MQ4G256V2)),
                routed_gate_up: ffn
                    .experts
                    .first()
                    .map(|e| e.gate_up.gpu_dtype)
                    .unwrap_or(DType::F32),
                routed_down: ffn
                    .experts
                    .first()
                    .map(|e| e.down.gpu_dtype)
                    .unwrap_or(DType::F32),
                routed_has_mixed_experts: per_expert_gate_up.is_some() || per_expert_down.is_some(),
                has_paro_shared: ffn.paro_shared.is_some(),
                per_expert_gate_up,
                per_expert_down,
            };
            let res = hipfire_dispatch::families::moe::MoePrefillResolution::resolve_with_batch_ctx(
                &dtypes,
                &gpu.arch_caps,
                &gpu.flags,
                n_verify,
                true,
                host_gate_up || host_down,
            );
            if !res.use_path2 {
                return false;
            }
            has_host_projection |= host_gate_up || host_down;
        }
    }
    has_host_projection
}

/// Capture-identity guard for the narrow MTP verify graph, built by
/// [`mtp_verify_graph_key`].
///
/// A `Copy` value so it compares and stores by value (in an `Option` field) with
/// no allocation; derive-based equality is a strict per-field compare, never a hash.
/// The ownership invariant that makes pointer identity sufficient here — captured
/// buffers are destroyed before any resource they point at can be freed or reallocated
/// — is documented on the `mtp_verify_graph_key` field of [`MtpSpecState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MtpVerifyGraphKey {
    /// Verify batch width this entry was captured for (`n_verify`).
    n_verify: usize,
    /// Model identity: output-norm weight buffer pointer (never hashed by content).
    model: usize,
    /// KV cache physical capacity (`kv_cache.physical_cap`).
    kv_capacity: usize,
    /// KV cache max sequence length (`kv_cache.max_seq`).
    kv_max_seq: usize,
    /// First K-lane buffer, or `0` when unallocated. Moves only on pool growth.
    kv_k: usize,
    /// First V-lane buffer, or `0` when unallocated. Moves only on pool growth.
    kv_v: usize,
    /// First DeltaNet state matrix buffer, or `0` when unallocated. Identifies the
    /// recurrent owner; moves only on pool growth.
    recurrent: usize,
    /// Global conversion/attention scratch pointers in `ScratchState` index order.
    /// Those pools are grow-never-shrink, so a moved (or newly zeroed) slot identifies
    /// growth or free — no contents hashed and no converted activation values cached
    /// here. Unused cells stay `0`.
    scratch: [usize; 32],
}

/// Capture-identity key for the narrow MTP verify graph.
///
/// Returns a [`MtpVerifyGraphKey`]; the stored field is `None` until one exists and is
/// reset to `None` on destroy, so `None` never matches a live entry (as the old sentinel
/// `0` did). Scope per the `mtp_verify_graph_key` field doc: this guards model identity,
/// attention width, the mutable KV mapping, the recurrent owner, and the global
/// conversion/attention scratch pools that may grow between MTP windows. Globals are
/// grow-never-shrink, so pointer changes identify growth — no per-window config Debug
/// strings and no buffer contents are hashed here.
fn mtp_verify_graph_key(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    kv_cache: &KvCache,
    dn_state: &DeltaNetState,
    n_verify: usize,
) -> MtpVerifyGraphKey {
    // Local pointer extractors — no allocation; a missing pool yields 0.
    fn tensor_ptr(t: &Option<GpuTensor>) -> usize {
        t.as_ref().map_or(0, |t| t.buf.as_ptr() as usize)
    }
    fn buffer_ptr(b: &Option<hip_bridge::DeviceBuffer>) -> usize {
        b.as_ref().map_or(0, |b| b.as_ptr() as usize)
    }

    let mut key = MtpVerifyGraphKey {
        n_verify,
        model: weights.output_norm.buf.as_ptr() as usize,
        kv_capacity: kv_cache.physical_cap,
        kv_max_seq: kv_cache.max_seq,
        kv_k: kv_cache
            .k_gpu
            .first()
            .map_or(0, |t| t.buf.as_ptr() as usize),
        kv_v: kv_cache
            .v_gpu
            .first()
            .map_or(0, |t| t.buf.as_ptr() as usize),
        recurrent: dn_state
            .s_matrices
            .first()
            .map_or(0, |t| t.buf.as_ptr() as usize),
        scratch: [0usize; 32],
    };

    // Global conversion/attention scratch pools (grow-never-shrink) in ScratchState
    // order. `gdn_pair_counters` is a DeviceBuffer despite its tensor-like role, so it
    // uses buffer_ptr; every slot defaults to 0 and only moves on pool growth.
    key.scratch[0] = tensor_ptr(&gpu.scratch.mq_signs1);
    key.scratch[1] = tensor_ptr(&gpu.scratch.mq_signs2);
    key.scratch[2] = tensor_ptr(&gpu.scratch.mq_signs1_128);
    key.scratch[3] = tensor_ptr(&gpu.scratch.mq_signs2_128);
    key.scratch[4] = tensor_ptr(&gpu.scratch.mq_x_rot);
    key.scratch[5] = tensor_ptr(&gpu.scratch.gemv_residual_tmp);
    key.scratch[6] = tensor_ptr(&gpu.scratch.paro_x_scratch);
    key.scratch[7] = buffer_ptr(&gpu.scratch.gdn_pair_counters);
    key.scratch[8] = tensor_ptr(&gpu.scratch.argmax_host);
    key.scratch[9] = buffer_ptr(&gpu.scratch.mq_x_rot_fp8);
    key.scratch[10] = buffer_ptr(&gpu.scratch.mq_x_q8);
    key.scratch[11] = buffer_ptr(&gpu.scratch.mq_x_scales);
    key.scratch[12] = buffer_ptr(&gpu.scratch.mq_rmsnorm_wavegrid_scratch);
    key.scratch[13] = buffer_ptr(&gpu.scratch.fp16_x_scratch);
    key.scratch[14] = buffer_ptr(&gpu.scratch.qsa_select_scores);
    key.scratch[15] = buffer_ptr(&gpu.scratch.gdn_state_f32);
    key.scratch[16] = buffer_ptr(&gpu.scratch.fp8_x_scratch);
    key.scratch[17] = buffer_ptr(&gpu.scratch.q8_1_mmq_x_scratch);
    key.scratch[18] = buffer_ptr(&gpu.scratch.int4_mmq_x_scratch);
    key.scratch[19] = buffer_ptr(&gpu.scratch.int8_mmq_x_scratch);
    key.scratch[20] = buffer_ptr(&gpu.scratch.mq4v2_fp8_x_scratch);
    key.scratch[21] = buffer_ptr(&gpu.scratch.mq4v2_fp8_half_sums_scratch);
    key.scratch[22] = buffer_ptr(&gpu.scratch.mq4v2_fp8_row_scales_scratch);
    key.scratch[23] = buffer_ptr(&gpu.scratch.fp8_f2_weights);
    key.scratch[24] = buffer_ptr(&gpu.scratch.ksplit_det_partials);
    key.scratch[25] = buffer_ptr(&gpu.scratch.sample_partials);
    key.scratch[26] = buffer_ptr(&gpu.scratch.fa2_q16_scratch);
    key.scratch[27] = buffer_ptr(&gpu.scratch.fa2_fp8_q_scratch);

    // paro_fused_scratch is an optional Vec of 4 tensor buffers (indices 0..3).
    if let Some(bufs) = &gpu.scratch.paro_fused_scratch {
        for (i, t) in bufs.iter().take(4).enumerate() {
            key.scratch[28 + i] = t.buf.as_ptr() as usize;
        }
    }

    key
}

impl MtpSpecState {
    /// Windows the captured MTP verify graph actually replayed. Oracle
    /// replay proof: gate on `> 0` with changing route/token inputs.
    pub fn mtp_verify_graph_replays(&self) -> u64 {
        self.mtp_verify_graph_replays
    }
    /// Successful MTP verify graph captures (bounded small). Oracle
    /// capture proof alongside `mtp_verify_graph_replays`.
    pub fn mtp_verify_graph_captures(&self) -> u64 {
        self.mtp_verify_graph_captures
    }
    /// Last window's MTP verify mode: "direct" | "warmup" | "capture" |
    /// "replay". Oracle mode proof without log scraping.
    pub fn mtp_verify_graph_mode(&self) -> &'static str {
        self.mtp_verify_graph_mode
    }
}

fn destroy_mtp_proposal_graph(gpu: &mut Gpu, state: &mut MtpSpecState) {
    if let Some(exec) = state.mtp_proposal_graph_exec.take() {
        let _ = gpu.hip.graph_exec_destroy(exec);
    }
    if let Some(graph) = state.mtp_proposal_graph.take() {
        let _ = gpu.hip.graph_destroy(graph);
    }
    state.mtp_proposal_graph_blobs.clear();
    state.mtp_proposal_graph_seq_cap = 0;
}

fn greedy_trunk_spine_accept(
    candidates: &[u32],
    argmax_per_pos: &[u32],
    eos_token_id: u32,
) -> GreedyTrunkSpineAccept {
    assert!(
        argmax_per_pos.len() >= candidates.len() + 1,
        "greedy_trunk_spine_accept: need at least candidates+1 argmax rows \
         (got {}, candidates={})",
        argmax_per_pos.len(),
        candidates.len(),
    );
    // Delegate to the shared greedy accept rule. MTP enables EOS-early-stop
    // (eos=Some): an accepted candidate equal to EOS stops the prefix with no
    // bonus. The trunk argmax IS the `target_pick`.
    let acc =
        hipfire_runtime::spec::accept_greedy_prefix(candidates, argmax_per_pos, Some(eos_token_id));
    GreedyTrunkSpineAccept {
        committed: acc.committed,
        accept_count: acc.accepted,
        hit_eos: acc.hit_eos,
    }
}

fn assemble_greedy_accept_from_gpu_result(
    candidates: &[u32],
    accept_count: usize,
    bonus_token_or_no_bonus: i32,
    eos_token_id: u32,
) -> GreedyTrunkSpineAccept {
    assert!(
        accept_count <= candidates.len(),
        "gpu greedy accept returned accept_count={} for {} candidates",
        accept_count,
        candidates.len()
    );

    let mut committed: Vec<u32> = Vec::with_capacity(candidates.len() + 1);
    committed.extend_from_slice(&candidates[..accept_count]);

    let hit_eos = if bonus_token_or_no_bonus < 0 {
        assert!(
            accept_count > 0 && candidates[accept_count - 1] == eos_token_id,
            "gpu greedy accept returned no bonus without an accepted EOS"
        );
        true
    } else {
        let bonus = bonus_token_or_no_bonus as u32;
        committed.push(bonus);
        bonus == eos_token_id
    };

    GreedyTrunkSpineAccept {
        committed,
        accept_count,
        hit_eos,
    }
}

/// Pure bound check for external PLD/ngram candidates. The window must be
/// non-empty, and its length must fit the allocated verify batch capacity
/// (`state.verify_capacity + 1` tokens including the seed `last_committed`).
/// This is the only pure, GPU-free validation the core exposes — no token
/// sampling, no device work.
fn mtp_external_candidates_within_capacity(candidates: &[u32], verify_capacity: usize) -> bool {
    !candidates.is_empty() && candidates.len() <= verify_capacity
}

/// Pure token assembly for the shared trunk verify: `[last_committed, candidates...]`.
/// Thin wrapper around [`build_trunk_spine_verify_tokens`] kept as a pure
/// helper so tests can assert bounds without touching GPU.
fn mtp_assemble_verify_tokens(last_committed: u32, candidates: &[u32]) -> Vec<u32> {
    build_trunk_spine_verify_tokens(last_committed, candidates)
}

/// Tokens whose MTP-head KV a takeover window fills (pure).
///
/// After shared verify/rollback the trunk holds `verify_tokens[..advance]`
/// (`[last_committed, candidates..]`) at `cur_pos..cur_pos + advance`, with or
/// without a bonus and including the accepted-EOS case. Decode writes head
/// slot `p` from `(tok_p, h_{p-1})`, so the fill writes exactly those rows:
/// row 0 pairs `last_committed` with the pre-window hidden, row `i > 0` pairs
/// `candidates[i - 1]` with verify hidden row `i - 1`. The pending token at
/// `cur_pos + advance` is written by the next window, as in decode.
fn mtp_takeover_fill_tokens(last_committed: u32, candidates: &[u32], advance: usize) -> Vec<u32> {
    assert!(
        advance >= 1 && advance <= candidates.len() + 1,
        "takeover fill: advance {advance} out of 1..={}",
        candidates.len() + 1
    );
    let mut tokens = Vec::with_capacity(advance);
    tokens.push(last_committed);
    tokens.extend_from_slice(&candidates[..advance - 1]);
    tokens
}

/// Enqueue the target lm_head over every MTP verify row.
///
/// All MTP entry points share this dispatcher. In particular, MQ V2 must not
/// fall through to `n_verify` sequential GEMVs now that the validated gfx11 /
/// gfx12 batched lm_head family exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_trunk_verify_lm_head(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    verify_hidden: &GpuTensor,
    verify_rot: &GpuTensor,
    logits_view: &GpuTensor,
    n_verify: usize,
    dim: usize,
    vocab: usize,
) -> HipResult<()> {
    // Fail closed: inside the PM FP8 verify scope only the dense MQ4G256V2
    // head has an FP8 verify arm (no GEMV / other-dtype fallback).
    if gpu.fp8_verify_scope && w_out.gpu_dtype != DType::MQ4G256V2 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "MTP verify lm_head: fp8 verify scope requires a dense MQ4G256V2 head, got {:?}",
                w_out.gpu_dtype
            ),
        ));
    }
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            if mtp_q8_verify_wmma_enabled_from_env() {
                gpu.gemm_q8_0_batched_chunked(
                    &w_out.buf,
                    verify_hidden,
                    logits_view,
                    w_out.m,
                    w_out.k,
                    n_verify,
                )?;
            } else {
                gpu.gemm_q8_0_batched_f32_chunked(
                    &w_out.buf,
                    verify_hidden,
                    logits_view,
                    w_out.m,
                    w_out.k,
                    n_verify,
                )?;
            }
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf,
                verify_hidden,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf,
                verify_hidden,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ4G256V2 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            // Inside the PM FP8 verify scope rdna routes this head to the FP8
            // `set` op at every row count (`gemm_mq4g256v2_batched_lmhead`).
            gpu.gemm_mq4g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ6G256V2 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_mq6g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ5G256V2 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_mq5g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ3G256V2 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_mq3g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        DType::MQ2G256V2 => {
            let rot = verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_mq2g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                logits_view,
                w_out.m,
                w_out.k,
                n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = verify_hidden.sub_offset(i * dim, dim);
                let logits_row = logits_view.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }
    Ok(())
}

/// Output of the MTP draft phase — K serial head-forward steps.
/// Consumed by either `mtp_shared_verify_accept_rollback_inner` (single-slot
/// path) or `mtp_batched_verify_accept_from_batch` (slot engine batched path).
pub struct MtpDraftOutput {
    pub candidates: Vec<u32>,
    pub drafts_generated: usize,
    pub chain_truncated: bool,
    pub use_sampling: bool,
    pub sampling: MtpSamplingConfig,
    pub draft_probs: Vec<f32>,
    pub draft_softmaxes: Vec<Vec<f32>>,
    pub use_device_token_chain: bool,
    pub cur_pos: usize,
    pub last_committed: u32,
}

impl MtpDraftOutput {
    /// Assemble `[seed, cand_1, ..., cand_K]` verify tokens.
    pub fn verify_tokens(&self) -> Vec<u32> {
        mtp_assemble_verify_tokens(self.last_committed, &self.candidates)
    }

    /// Number of verify rows = drafts_generated + 1 (seed).
    pub fn n_verify(&self) -> usize {
        self.drafts_generated + 1
    }
}

/// Accept decision of one verify window ([`mtp_verify_accept`]).
pub(crate) struct MtpVerifyAccepted {
    pub committed: Vec<u32>,
    pub accept_count: usize,
    pub hit_eos: bool,
}

fn mtp_penalty_require_window(
    penalty: &hipfire_runtime::spec_sampling::PenaltyHistory,
) -> HipResult<()> {
    if penalty.window() == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "MTP penalties active but state.penalty has window 0 (drafter must install \
             PenaltyHistory::new(window) and begin_window per step)",
        ));
    }
    Ok(())
}

/// One H2D of the penalty history (`penalty.tokens()`: base history then every
/// pushed draft) to offset 0 of `repeat_buf`, as raw u32 ids in the F32-typed
/// tensor (same byte upload as `sampler::sample`'s repeat buffer).
fn mtp_penalty_upload_history(
    gpu: &Gpu,
    penalty: &hipfire_runtime::spec_sampling::PenaltyHistory,
    repeat_buf: &GpuTensor,
) -> HipResult<()> {
    let tokens = penalty.tokens();
    if tokens.is_empty() {
        return Ok(());
    }
    if tokens.len() > repeat_buf.numel() {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "MTP penalty history of {} tokens exceeds repeat buffer capacity {} (window {})",
                tokens.len(),
                repeat_buf.numel(),
                penalty.window()
            ),
        ));
    }
    // SAFETY: u32 slice reinterpreted as its own bytes; length = len * 4.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
    gpu.hip.memcpy_htod(&repeat_buf.buf, bytes)
}

/// Start of a window's sampled draft chain: drop any stale drafts and upload
/// the base history.
fn mtp_penalty_begin_drafts(
    gpu: &Gpu,
    penalty: &mut hipfire_runtime::spec_sampling::PenaltyHistory,
    repeat_buf: &GpuTensor,
) -> HipResult<()> {
    mtp_penalty_require_window(penalty)?;
    penalty.rewind_drafts();
    mtp_penalty_upload_history(gpu, penalty, repeat_buf)
}

/// Keep a drawn draft: extend the history and upload just that id at the new
/// last slot of `repeat_buf`.
fn mtp_penalty_append_draft(
    gpu: &Gpu,
    penalty: &mut hipfire_runtime::spec_sampling::PenaltyHistory,
    repeat_buf: &GpuTensor,
    token: u32,
) -> HipResult<()> {
    mtp_penalty_require_window(penalty)?;
    penalty.push_draft(token);
    let len = penalty.tokens().len();
    if len > repeat_buf.numel() {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "MTP penalty history of {len} tokens exceeds repeat buffer capacity {} (window {})",
                repeat_buf.numel(),
                penalty.window()
            ),
        ));
    }
    gpu.hip
        .memcpy_htod_offset(&repeat_buf.buf, (len - 1) * 4, &token.to_ne_bytes())
}

/// Penalize one logits row in place against `penalty.row(row)` (a sub-range of
/// the uploaded history). Same kernel and arguments as the AR GPU sampler's
/// Phase 0; empty history is a no-op like AR's `window == 0` skip.
fn mtp_penalty_apply_row(
    gpu: &mut Gpu,
    penalty: &hipfire_runtime::spec_sampling::PenaltyHistory,
    repeat_buf: &GpuTensor,
    logits_row: &GpuTensor,
    row: usize,
    vocab: usize,
    sampling: &MtpSamplingConfig,
) -> HipResult<()> {
    let r = penalty.row_range(row);
    if r.is_empty() {
        return Ok(());
    }
    let history = repeat_buf.sub_offset(r.start, r.len());
    gpu.apply_repeat_penalty_row(
        logits_row,
        &history,
        vocab,
        r.len(),
        sampling.repeat_penalty,
        sampling.presence_penalty,
        sampling.frequency_penalty,
    )
}

/// Penalize all `n_verify` rows of `logits_view` (row `k` scores `c_k` after
/// `[seed, c_0..c_{k-1}]`; history `emitted ‖ c[..k]`). Rewinds and re-pushes
/// the window's candidates from `candidates[..n_verify - 1]` (so rejected /
/// pruned draft-phase pushes never survive), uploads the history once, then
/// launches one prepass per non-empty row. No-op unless `sampling.penalized()`.
fn mtp_apply_verify_penalties(
    gpu: &mut Gpu,
    state: &mut MtpSpecState,
    logits_view: &GpuTensor,
    candidates: &[u32],
    n_verify: usize,
    vocab: usize,
    sampling: &MtpSamplingConfig,
) -> HipResult<()> {
    if !sampling.penalized() {
        return Ok(());
    }
    let t0 = MtpPenaltyTiming::start();
    mtp_penalty_require_window(&state.penalty)?;
    debug_assert!(n_verify >= 1 && candidates.len() + 1 >= n_verify);
    state.penalty.rewind_drafts();
    for &c in &candidates[..n_verify - 1] {
        state.penalty.push_draft(c);
    }
    mtp_penalty_upload_history(gpu, &state.penalty, &state.mtp_sample_repeat_buf)?;
    for k in 0..n_verify {
        let row = logits_view.sub_offset(k * vocab, vocab);
        mtp_penalty_apply_row(
            gpu,
            &state.penalty,
            &state.mtp_sample_repeat_buf,
            &row,
            k,
            vocab,
            sampling,
        )?;
    }
    if t0.is_some() {
        let uploads = u32::from(!state.penalty.tokens().is_empty());
        let launches = (0..n_verify)
            .filter(|&k| !state.penalty.row_range(k).is_empty())
            .count() as u32;
        state.penalty_timing.finish(t0, uploads, launches);
    }
    Ok(())
}

/// Trunk lm_head over the verify rows, the greedy/sampled accept rule, and the
/// `prev_hidden` capture from the last kept row. Everything in an MTP verify
/// that runs on the drafter's device; the trunk rollback is the caller's
/// (single slot: [`mtp_accept_and_rollback`]; dense TP: per rank in
/// `mtp_dense_tp`).
///
/// `state.verify_hidden` must hold the post-norm trunk hidden of all
/// `n_verify` rows.
///
/// When `sampling.penalized()`, every verify row `k` (logits after
/// `[seed, c_0..c_{k-1}]`, scoring `c_k`) is penalized in place with the
/// history `state.penalty.row(k)` = `emitted ‖ c[..k]` before the accept
/// rule runs, so sampled accept/residual/bonus and greedy argmax-match both
/// see the penalty-adjusted AR distribution. The caller must have run
/// `state.penalty.begin_window(emitted)` for this window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_verify_accept(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    dim: usize,
    vocab: usize,
    state: &mut MtpSpecState,
    n_verify: usize,
    candidates: &[u32],
    drafts_generated: usize,
    use_sampling: bool,
    sampling: MtpSamplingConfig,
    draft_probs: &[f32],
    draft_softmaxes: &[Vec<f32>],
    is_external: bool,
    use_device_token_chain: bool,
    cur_pos: usize,
    eos_token_id: u32,
) -> HipResult<MtpVerifyAccepted> {
    mtp_verify_accept_impl(
        gpu,
        w_out,
        dim,
        vocab,
        state,
        n_verify,
        candidates,
        drafts_generated,
        use_sampling,
        sampling,
        draft_probs,
        draft_softmaxes,
        is_external,
        use_device_token_chain,
        cur_pos,
        eos_token_id,
        false,
    )
}

/// [`mtp_verify_accept`] with the head evaluation optional: `head_ready`
/// means the caller already wrote this window's verify-head logits into
/// `state.verify_logits` rows `0..n_verify` (cross-request batched head).
#[allow(clippy::too_many_arguments)]
fn mtp_verify_accept_impl(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    dim: usize,
    vocab: usize,
    state: &mut MtpSpecState,
    n_verify: usize,
    candidates: &[u32],
    drafts_generated: usize,
    use_sampling: bool,
    sampling: MtpSamplingConfig,
    draft_probs: &[f32],
    draft_softmaxes: &[Vec<f32>],
    is_external: bool,
    use_device_token_chain: bool,
    cur_pos: usize,
    eos_token_id: u32,
    head_ready: bool,
) -> HipResult<MtpVerifyAccepted> {
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    if !head_ready {
        mtp_trunk_verify_lm_head(
            gpu,
            w_out,
            &state.verify_hidden,
            &state.verify_rot,
            &logits_view,
            n_verify,
            dim,
            vocab,
        )?;
    }

    // Exact-AR penalties on every verify row the verdict can read (incl. the
    // bonus row), before softmax (sampled) or argmax (greedy). No-op, no
    // upload/launch, unless `sampling.penalized()`.
    mtp_apply_verify_penalties(
        gpu,
        state,
        &logits_view,
        candidates,
        n_verify,
        vocab,
        &sampling,
    )?;

    let mut accept_count = 0usize;
    let hit_eos;
    let mut committed: Vec<u32> = Vec::with_capacity(drafts_generated + 1);

    // Sampled verify applies to every drafter, including the external
    // (ngram/PLD takeover) one: its candidates are deterministic proposals,
    // i.e. a point-mass draft q = δ(c), handled by `mtp_sampled_accept` with
    // `point_mass = true`. Greedy requests still take the argmax branch below.
    if use_sampling {
        hit_eos = mtp_sampled_accept(
            gpu,
            state,
            &logits_view,
            draft_softmaxes,
            draft_probs,
            candidates,
            drafts_generated,
            vocab,
            is_external,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            eos_token_id,
            &mut committed,
            &mut accept_count,
        )?;
    } else {
        let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
        gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;

        let use_gpu_accept =
            !is_external && use_device_token_chain && mtp_gpu_greedy_accept_enabled_from_env();
        let accepted = if use_gpu_accept {
            let candidate_device = state.mtp_token_chain.sub_offset(1, drafts_generated);
            let accept_result = state.verify_argmax.sub_offset(0, 2);
            gpu.greedy_accept_from_argmax_i32(
                &argmax_v,
                &candidate_device,
                &accept_result,
                drafts_generated,
                eos_token_id,
            )?;
            let mut accept_host: [i32; 2] = [0, 0];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(accept_host.as_mut_ptr() as *mut u8, 8)
                };
                gpu.hip.memcpy_dtoh(bytes, &accept_result.buf)?;
            }
            assemble_greedy_accept_from_gpu_result(
                &candidates[..drafts_generated],
                accept_host[0] as usize,
                accept_host[1],
                eos_token_id,
            )
        } else {
            let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        argmax_v_host.as_mut_ptr() as *mut u8,
                        n_verify * 4,
                    )
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
            }
            let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();
            greedy_trunk_spine_accept(candidates, &argmax_per_pos, eos_token_id)
        };
        committed = accepted.committed;
        accept_count = accepted.accept_count;
        hit_eos = accepted.hit_eos;
        if mtp_phase_timing_enabled() {
            eprintln!(
                "QWEN35_MTP_ACCEPT {{\"event\":\"mtp_accept\",\"pos\":{cur_pos},\"drafts\":{drafts_generated},\"accepted\":{accept_count}}}",
            );
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= drafts_generated + 1);

    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, cur_pos, prev_hidden_row, dim)?;

    Ok(MtpVerifyAccepted {
        committed,
        accept_count,
        hit_eos,
    })
}

/// Core accept-and-rollback logic shared between the single-slot path
/// (`mtp_shared_verify_accept_rollback_inner`) and the batched slot-engine
/// path (`mtp_batched_verify_accept_from_batch`).
///
/// Caller responsibilities BEFORE calling this:
/// - Populate `state.verify_hidden` with the trunk's hidden states for all
///   `n_verify` rows (single-slot: trunk forward writes them; batched: copied
///   from `pbs.x_batch`).
/// - Save DN snapshot to `state.trunk_snap` (for rollback on rejection).
/// - Assemble `verify_tokens` = `[seed, cand_1, ..., cand_K]`.
#[allow(clippy::too_many_arguments)]
fn mtp_accept_and_rollback(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &mut KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    state: &mut MtpSpecState,
    n_verify: usize,
    verify_tokens: &[u32],
    candidates: &[u32],
    drafts_generated: usize,
    chain_truncated: bool,
    use_sampling: bool,
    sampling: MtpSamplingConfig,
    draft_probs: &[f32],
    draft_softmaxes: &[Vec<f32>],
    is_external: bool,
    use_device_token_chain: bool,
    tape_captured: bool,
    cur_pos: usize,
    eos_token_id: u32,
    head_ready: bool,
) -> HipResult<MtpSpecResult> {
    let trunk_weights: &Qwen35Weights = weights;
    let MtpVerifyAccepted {
        committed,
        accept_count,
        hit_eos,
    } = mtp_verify_accept_impl(
        gpu,
        &trunk_weights.output,
        config.dim,
        config.vocab_size,
        state,
        n_verify,
        candidates,
        drafts_generated,
        use_sampling,
        sampling,
        draft_probs,
        draft_softmaxes,
        is_external,
        use_device_token_chain,
        cur_pos,
        eos_token_id,
        head_ready,
    )?;
    let advance = committed.len();

    // ── KV / DN rollback (or skip on full accept) ─────────────────────
    let full_accept_no_eos = advance == drafts_generated + 1 && !hit_eos;
    let replay_skipped = full_accept_no_eos;
    if !full_accept_no_eos {
        state.trunk_snap.restore_to(dn_state, gpu)?;
        if tape_captured {
            state
                .trunk_gdn_tape
                .replay_gdn(gpu, trunk_weights, config, dn_state, advance)?;
        } else {
            if advance >= 2 {
                let replay = &verify_tokens[..advance];
                // Rollback replay inside the PM FP8 verify scope: n >= 2 rows
                // reproduce the verify's FP8 rows (row-count independent).
                let fp8 = fp8_verify_target(trunk_weights, config, kv_cache);
                crate::speculative::fp8_verify_scoped(gpu, fp8, |gpu| {
                    qwen35::forward_prefill_batch(
                        gpu,
                        trunk_weights,
                        config,
                        replay,
                        cur_pos,
                        kv_cache,
                        dn_state,
                        scratch,
                        None,
                        None,
                        None,
                        None,
                    )
                })?;
            } else {
                gpu.graphs.ar_graph_eligible = false;
                qwen35::forward_scratch(
                    gpu,
                    trunk_weights,
                    config,
                    verify_tokens[0],
                    cur_pos,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            }
        }
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated,
        chain_truncated,
        replay_skipped,
    })
}

/// Run one narrow MTP trunk-verify forward through the per-state hipGraph
/// capture/replay cache, preserving the exact eager math (same
/// `forward_prefill_batch_with_pbs_opts` call with `DflashFusionCtx::Off`,
/// same PBS, same `verify_hidden` capture, same GDN tape.
///
/// Per-window lifecycle:
/// - refresh KV mapped capacity and compute the capture-identity key BEFORE
///   anything is uploaded or captured; an identity mismatch (graph or warmup
///   pending), a width change against the live graph, or a width change
///   against a pending warmup tears the entry down so warm-then-capture
///   restarts for this window;
/// - an existing graph replays on the active stream; any launch error destroys
///   it and latches disabled (no eager fallback);
/// - the first window at a fresh width runs the exact eager forward once to
///   finish JIT + lazy scratch growth outside any capture, then records that
///   it warmed without executing twice or restoring the pre-verify snapshot;
/// - the next window captures the forward and immediately replays the fresh
///   graph once so this window's outputs are real (first launch counts);
/// - any capture failure aborts, latches disabled, logs, and falls back to a
///   single eager run. No `trunk_snap` restore here: a capture only recorded
///   nodes, it never executed them.
/// Tokens and positions are device-buffer inputs refreshed before every replay
/// — no baked `cur_pos` scalar, no transient host pointer, no baked lifetime.
/// Disjoint state borrows go through ordinary references; this function uses
/// no raw pointers.
#[allow(clippy::too_many_arguments)]
fn mtp_verify_forward_graphed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &mut KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    state: &mut MtpSpecState,
    verify_tokens: &[u32],
    cur_pos: usize,
    n_verify: usize,
) -> HipResult<()> {
    // 1. Preflight KV mapping for this window BEFORE computing identity,
    //    uploading inputs, or capturing: lazy scratch/KV growth must be
    //    reflected in the key and never happen inside a capture.
    // If this window needs more mapped capacity than reserved, tear down any
    // cached entry FIRST: ensure_mapped_capacity below may grow/remap memory a
    // live graph references — destruction orders its frees ahead of that growth.
    if cur_pos
        .checked_add(n_verify)
        .is_none_or(|end| end > kv_cache.physical_cap)
    {
        destroy_mtp_verify_graph(gpu, state);
    }
    kv_cache.ensure_mapped_capacity(gpu, cur_pos + n_verify)?;

    // Capture-identity fingerprint for this width (see mtp_verify_graph_key).
    let key = mtp_verify_graph_key(gpu, weights, kv_cache, dn_state, n_verify);
    let graph_present = state.mtp_verify_graph.is_some();
    // Invalidate on an identity mismatch while a graph or warmup is pending, a
    // width change against the live graph, or a width change against a pending
    // warmup. `graph_n`/`warm_n` are compared only under their own guard so a
    // pre-capture `graph_n == 0` never forces an infinite warm loop.
    if ((graph_present || state.mtp_verify_graph_warmed) && state.mtp_verify_graph_key != Some(key))
        || (graph_present && state.mtp_verify_graph_n != n_verify)
        || (state.mtp_verify_graph_warmed && state.mtp_verify_graph_warm_n != n_verify)
    {
        destroy_mtp_verify_graph(gpu, state);
    }

    // Refresh the device-buffer inputs OUTSIDE any capture so every replay sees
    // fresh tokens/positions; the recorded graph only references these bytes.
    qwen35::upload_prefill_batch_inputs(gpu, &state.trunk_pbs, verify_tokens, cur_pos)?;

    // Shared eager Path-2 body (identical math to the non-graph path). Takes
    // only the live mutable handles and captures weights/config/inputs so the
    // argument list is written once and reused by warmup + capture fallback.
    let run_eager = |gpu: &mut Gpu,
                     kv_cache: &mut KvCache,
                     dn_state: &mut DeltaNetState,
                     scratch: &Qwen35Scratch,
                     state: &mut MtpSpecState|
     -> HipResult<()> {
        qwen35::forward_prefill_batch_with_pbs_opts(
            gpu,
            weights,
            config,
            verify_tokens,
            cur_pos,
            kv_cache,
            dn_state,
            scratch,
            None,
            Some(&state.verify_hidden),
            Some(&mut state.trunk_gdn_tape),
            None,
            Some(&state.trunk_pbs),
            None,
            None,
            false,
            qwen35::DflashFusionCtx::Off,
        )
    };

    // 2. Existing graph: replay on the active stream. A launch failure is fatal
    //    (no eager fallback): destroy + latch disabled + propagate.
    if state.mtp_verify_graph.is_some() {
        let stream = gpu.active_stream.as_ref().unwrap();
        let exec = &state.mtp_verify_graph.as_ref().unwrap().1;
        if let Err(e) = gpu.hip.graph_launch(exec, stream) {
            destroy_mtp_verify_graph(gpu, state);
            state.mtp_verify_graph_disabled = true;
            return Err(e);
        }
        state.mtp_verify_graph_replays += 1;
        state.mtp_verify_graph_mode = "replay";
        return Ok(());
    }

    // 3. No warmed width for this n: run the exact eager forward once to finish
    //    JIT + lazy scratch growth outside any capture, then remember it warmed.
    //    Do NOT restore trunk_snap and do NOT execute twice — this single warmup
    //    is the window's output; the recurrent state already advanced past it.
    if !state.mtp_verify_graph_warmed || state.mtp_verify_graph_warm_n != n_verify {
        run_eager(gpu, kv_cache, dn_state, scratch, state)?;
        let stream = gpu.active_stream.as_ref().unwrap();
        gpu.hip.stream_synchronize(stream)?;
        state.mtp_verify_graph_warmed = true;
        state.mtp_verify_graph_warm_n = n_verify;
        // Recompute identity AFTER warmup: the execution may have moved scratch
        // (lazy growth) or DN pointers that the next capture bakes into nodes.
        state.mtp_verify_graph_key = Some(mtp_verify_graph_key(
            gpu, weights, kv_cache, dn_state, n_verify,
        ));
        state.mtp_verify_graph_mode = "warmup";
        return Ok(());
    }

    // 4. Capture the forward: begin -> exact single-chunk captured body -> end
    //    in one IIFE so a single match handles every failure path.
    let run_capture = |gpu: &mut Gpu,
                       kv_cache: &mut KvCache,
                       dn_state: &mut DeltaNetState,
                       scratch: &Qwen35Scratch,
                       state: &mut MtpSpecState|
     -> HipResult<(Graph, GraphExec, Vec<Vec<u8>>)> {
        mtp_graph_begin_capture(gpu)?;
        qwen35::forward_prefill_batch_single_chunk_captured_opts(
            gpu,
            weights,
            config,
            verify_tokens,
            cur_pos,
            kv_cache,
            dn_state,
            scratch,
            &state.trunk_pbs,
            None,
            Some(&state.verify_hidden),
            Some(&mut state.trunk_gdn_tape),
            None,
            false,
            qwen35::DflashFusionCtx::Off,
        )?;
        mtp_graph_end_capture(gpu)
    };

    match run_capture(gpu, kv_cache, dn_state, scratch, state) {
        // 5. Capture success: store identity + counters (mode = capture), then
        //    replay once so this window's outputs are real (first launch counts).
        //    No steady-state sync — the accept path already orders its GPU work.
        //    A failed first launch destroys the entry, latches disabled, propagates.
        Ok((graph, exec, blobs)) => {
            let n_blobs = blobs.len();
            state.mtp_verify_graph = Some((graph, exec, blobs));
            state.mtp_verify_graph_n = n_verify;
            state.mtp_verify_graph_key = Some(key);
            state.mtp_verify_graph_warmed = false;
            state.mtp_verify_graph_captures += 1;
            state.mtp_verify_graph_mode = "capture";
            let launch = {
                let stream = gpu.active_stream.as_ref().unwrap();
                let exec = &state.mtp_verify_graph.as_ref().unwrap().1;
                gpu.hip.graph_launch(exec, stream)
            };
            match launch {
                Ok(()) => {
                    state.mtp_verify_graph_replays += 1;
                    eprintln!(
                        "[mtp-verify-graph] captured for n={n_verify} with {} blobs",
                        n_blobs
                    );
                    Ok(())
                }
                Err(e) => {
                    destroy_mtp_verify_graph(gpu, state);
                    state.mtp_verify_graph_disabled = true;
                    Err(e)
                }
            }
        }
        // Capture failed: abort the in-flight capture, latch disabled, log, and
        // fall back to one eager run using the same disjoint field borrows. The
        // capture only recorded nodes — it never executed them — so no trunk_snap
        // restore is needed here; do not hide the eager path's own error.
        Err(e) => {
            mtp_graph_abort_capture(gpu);
            state.mtp_verify_graph_disabled = true;
            eprintln!("[mtp-verify-graph] disabled after capture failure: {e}");
            state.mtp_verify_graph_mode = "direct";
            run_eager(gpu, kv_cache, dn_state, scratch, state)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mtp_shared_verify_accept_rollback_inner(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &mut KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    candidates: &[u32],
    drafts_generated: usize,
    chain_truncated: bool,
    overlap_trunk_snap: bool,
    use_sampling: bool,
    sampling: MtpSamplingConfig,
    draft_probs: &[f32],
    draft_softmaxes: &[Vec<f32>],
    is_external: bool,
    use_device_token_chain: bool,
) -> HipResult<MtpSpecResult> {
    let dim = config.dim;
    let vocab = config.vocab_size;
    let trunk_weights: &Qwen35Weights = weights;

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 2. Trunk verify (shared) ──────────────────────────────────────
    let verify_tokens = mtp_assemble_verify_tokens(last_committed, candidates);
    let n_verify = verify_tokens.len();
    debug_assert_eq!(n_verify, drafts_generated + 1);
    debug_assert!(n_verify <= state.verify_capacity + 1);

    if overlap_trunk_snap {
        gpu.hip
            .stream_synchronize(state.trunk_snap_stream.as_ref().unwrap())?;
    } else {
        state.trunk_snap.save_from(dn_state, gpu)?;
    }

    let tape_captured = qwen35::prefill_batch_pbs_eligible(
        trunk_weights,
        config,
        dn_state,
        n_verify,
        gpu.arch.as_str(),
        /* moe_router_logits_present — dense trunk: arm never matched */ true,
    );
    // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): the target verify forward and
    // its verify lm_head run the one E4M3 verify arithmetic (== the batched MTP
    // CB verify); the head runs here (`head_ready` below), scope closed on every path.
    // The scope is a no-op for the narrow-verify graph route, which is MoE-trunk
    // only while the scope's target is a dense MQ4G256V2 trunk.
    let prev = fp8_verify_open(gpu, trunk_weights, config, kv_cache)?;
    let graph_route = mtp_verify_graph_eligible(
        gpu,
        trunk_weights,
        config,
        kv_cache,
        state,
        n_verify,
        use_sampling,
        is_external,
        overlap_trunk_snap,
        cur_pos,
    ) && tape_captured;
    let fwd = if graph_route {
        mtp_verify_forward_graphed(
            gpu,
            trunk_weights,
            config,
            kv_cache,
            dn_state,
            scratch,
            state,
            &verify_tokens,
            cur_pos,
            n_verify,
        )
    } else {
        destroy_mtp_verify_graph(gpu, state);
        state.mtp_verify_graph_mode = "direct";
        let verify_tape: Option<&mut GdnTape> = if tape_captured {
            Some(&mut state.trunk_gdn_tape)
        } else {
            None
        };
        qwen35::forward_prefill_batch_with_pbs_opts(
            gpu,
            trunk_weights,
            config,
            &verify_tokens,
            cur_pos,
            kv_cache,
            dn_state,
            scratch,
            None,
            Some(&state.verify_hidden),
            verify_tape,
            None,
            Some(&state.trunk_pbs),
            None,
            None,
            false,
            qwen35::DflashFusionCtx::Off,
        )
    };
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    let head = fwd.and_then(|()| {
        mtp_trunk_verify_lm_head(
            gpu,
            &trunk_weights.output,
            &state.verify_hidden,
            &state.verify_rot,
            &logits_view,
            n_verify,
            dim,
            vocab,
        )
    });
    gpu.fp8_verify_exit(prev);
    head?;

    mtp_accept_and_rollback(
        gpu,
        weights,
        config,
        kv_cache,
        dn_state,
        scratch,
        state,
        n_verify,
        &verify_tokens,
        candidates,
        drafts_generated,
        chain_truncated,
        use_sampling,
        sampling,
        draft_probs,
        draft_softmaxes,
        is_external,
        use_device_token_chain,
        tape_captured,
        cur_pos,
        eos_token_id,
        true, // head_ready: the verify lm_head ran above inside the scope
    )
}

/// DeltaNet state repair after a partial MTP accept, replaying conv1d +
/// qk-norm + GDN from the verify step's per-layer `(qkv, alpha, beta)` tape.
///
/// A batched verify forward advances the slot's `DeltaNetState` through all
/// `m = k + 1` rows in place. On a partial accept the state must instead
/// reflect only the committed prefix. The old repair re-ran the ENTIRE trunk
/// as a prefill over the accepted tokens — attention, FFN, norms and all —
/// even though the KV rows for the accepted prefix were already correct.
///
/// The DeltaNet layers' per-row inputs cannot be re-read from `pbs` after
/// the forward: those batch buffers are per-LAYER scratch, overwritten by
/// each subsequent layer. Verify steps therefore TAPE their rows per layer
/// as they compute (see `MtpVerifyCapture` in forward_slots); this replay
/// mirrors `GdnTape::replay_gdn_inner` with a row offset — conv1d (advances
/// the ring), fused QK norm (+ repeat-interleave), then the GDN recurrence —
/// over the first `advance` taped rows. ~5 launches per DeltaNet layer
/// instead of a full trunk pass; the KV cache is untouched.
///
/// `tape_row_off` is the slot's first row in the tape (`slot * (mtp_k + 1)`).
/// `advance >= 1` (greedy accept always commits at least one row).
#[allow(clippy::too_many_arguments)]
pub fn mtp_dn_repair_from_tape(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &mut DeltaNetState,
    tape: &crate::speculative::GdnTape,
    tape_row_off: usize,
    advance: usize,
) -> HipResult<()> {
    assert!(
        advance >= 1,
        "mtp_dn_repair_from_tape: advance must be >= 1"
    );
    assert!(
        tape_row_off + advance <= tape.max_n,
        "mtp_dn_repair_from_tape: rows {}..{} exceed tape max_n {}",
        tape_row_off,
        tape_row_off + advance,
        tape.max_n
    );
    if !matches!(dn_state.quant, StateQuant::Q8) {
        return Err(hip_bridge::HipError::new(
            0,
            "mtp_dn_repair_from_tape: DeltaNetState must be StateQuant::Q8 \
             (the multi-slot batched path is Q8-only)",
        ));
    }

    let n_v_heads = tape.n_v_heads;
    let n_key_heads = tape.n_key_heads;
    let hd = tape.key_head_dim;
    let v_dim = tape.v_dim;
    let k_dim = tape.k_dim;
    let value_head_dim = tape.value_head_dim;

    let mut la_idx = 0usize;
    for (layer_idx, lt) in config.layer_types.iter().enumerate() {
        if *lt != LayerType::LinearAttention {
            continue;
        }
        let conv_weight = match &weights.layers[layer_idx] {
            LayerWeights::DeltaNet(l) => &l.conv_weight,
            LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "mtp_dn_repair_from_tape: LA layer type/weights mismatch",
                ));
            }
        };

        // Taped rows for this slot, at the slot's stride offset.
        let qkv_rows =
            tape.qkv_bufs[la_idx].sub_offset(tape_row_off * tape.qkv_dim, advance * tape.qkv_dim);
        let alpha_rows =
            tape.alpha_bufs[la_idx].sub_offset(tape_row_off * n_v_heads, advance * n_v_heads);
        let beta_rows =
            tape.beta_bufs[la_idx].sub_offset(tape_row_off * n_v_heads, advance * n_v_heads);

        // 1. conv1d + SiLU + split — advances the conv ring state.
        gpu.conv1d_silu_split_f32_n(
            &tape.q_raw_scratch,
            &tape.k_raw_scratch,
            &tape.v_scratch,
            &qkv_rows,
            conv_weight,
            &dn_state.conv_states[la_idx],
            k_dim,
            v_dim,
            advance,
        )?;

        // 2. L2 norm(Q) + L2 norm(K) + scale(Q).
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &tape.q_raw_scratch,
            &tape.k_raw_scratch,
            n_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            advance,
        )?;

        // 3. Repeat-interleave if GQA.
        if n_key_heads < n_v_heads {
            let ratio = n_v_heads / n_key_heads;
            gpu.repeat_interleave_qk_f32_batched(
                &tape.q_raw_scratch,
                &tape.k_raw_scratch,
                &tape.q_scratch,
                &tape.k_scratch,
                n_key_heads,
                ratio,
                hd,
                advance,
            )?;
        } else {
            let bytes = advance * k_dim * 4;
            gpu.hip
                .memcpy_dtod_at(&tape.q_scratch.buf, 0, &tape.q_raw_scratch.buf, 0, bytes)?;
            gpu.hip
                .memcpy_dtod_at(&tape.k_scratch.buf, 0, &tape.k_raw_scratch.buf, 0, bytes)?;
        }

        // 4. GDN recurrence — advances S/scales/EF residual.
        gpu.gated_delta_net_q8_batch_seq(
            &tape.q_scratch,
            &tape.k_scratch,
            &tape.v_scratch,
            &alpha_rows,
            &beta_rows,
            &dn_state.s_matrices[la_idx],
            &dn_state.s_scales[la_idx],
            &tape.attn_scratch,
            advance,
            n_v_heads,
            value_head_dim,
            dn_state.ef_residual(la_idx),
        )?;
        la_idx += 1;
    }
    Ok(())
}

/// Batched verify/accept for the vLLM-style slot-engine MTP path.
///
/// Differences vs the previous implementation:
///
///
/// * the trunk lm_head runs directly over a VIEW of `pbs.x_batch` at the
///   slot's row offset — no `verify_hidden` staging copy, and no separate
///   per-slot last-row GEMV upstream (the caller passes `lm_head_skip` to
///   `forward_batch_slots_graphed_opts` so the batched forward skips it);
/// * greedy accept reads one packed `n_verify`-int D2H instead of routing
///   through the GPU device-token-chain accept;
/// * a partial accept repairs the DeltaNet state with
///   [`mtp_dn_repair_from_tape`] replaying the verify tape over the
///   accepted rows (~5 launches per DN layer) instead of
///   re-running a full trunk `forward_prefill_batch` replay;
/// * `prev_hidden` for the next draft cycle is captured from the same
///   `x_batch` view.
///
/// The KV cache is not touched: the verify forward already wrote the
/// accepted prefix's rows at the right positions, and rows past the new
/// frontier are overwritten before any later forward can read them.
#[allow(clippy::too_many_arguments)]
pub fn mtp_batched_verify_accept_from_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &mut DeltaNetState,
    pbs: &PrefillBatchScratch,
    state: &mut MtpSpecState,
    draft: &MtpDraftOutput,
    hidden_row_offset: usize,
    verify_tape: &crate::speculative::GdnTape,
    tape_stride: usize,
    slot: usize,
    eos_token_id: u32,
    aux_eot_id: Option<u32>,
    commit_budget: usize,
) -> HipResult<MtpSpecResult> {
    let dim = config.dim;
    let vocab = config.vocab_size;
    let n_verify = draft.n_verify();
    let drafts_generated = draft.drafts_generated;

    // `pbs.x_batch` holds the PRE-final-norm residual — the same buffer
    // `final_logits_per_slot` norms per slot before its AR GEMV — while the
    // trunk lm_head consumes POST-output-norm hidden (the convention the
    // sequential verify path gets from `forward_prefill_batch`'s
    // `per_token_hidden_out`). Norm the verify rows into the state's post-norm
    // staging so acceptance compares against exactly what the AR path would
    // sample; skipping this skews every argmax by the per-channel γ scale and
    // breaks the MTP ≡ AR-at-greedy identity (the bonus token in particular).
    let verify_raw = pbs
        .x_batch
        .sub_offset(hidden_row_offset * dim, n_verify * dim);
    let verify_hidden = state.verify_hidden.sub_offset(0, n_verify * dim);
    gpu.rmsnorm_batched(
        &verify_raw,
        &weights.output_norm,
        &verify_hidden,
        n_verify,
        dim,
        config.norm_eps,
    )?;

    // Trunk lm_head over all verify rows (batched GEMM; handles every
    // admissible lm_head dtype incl. the FWHT-rotate for MQ-family).
    let w_out = &weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    mtp_trunk_verify_lm_head(
        gpu,
        w_out,
        &verify_hidden,
        &state.verify_rot,
        &logits_view,
        n_verify,
        dim,
        vocab,
    )?;

    // Penalized request (never the serve slot-engine default): the verdict's
    // argmax is the penalty-adjusted AR one, row k under `emitted ‖ c[..k]`.
    mtp_apply_verify_penalties(
        gpu,
        state,
        &logits_view,
        &draft.candidates,
        n_verify,
        vocab,
        &draft.sampling,
    )?;

    // Greedy accept: argmax per verify row on GPU, one packed D2H, host-side
    // prefix rule. (The slot-engine MTP path is greedy-only, matching the
    // retired `is_external = true` behaviour.)
    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, n_verify * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();
    if mtp_trace_enabled() {
        eprintln!(
            "[mtp-trace] accept cur_pos={} n_verify={} drafts={} candidates={:?} argmax={:?}",
            draft.cur_pos,
            n_verify,
            drafts_generated,
            &draft.candidates[..drafts_generated.min(draft.candidates.len())],
            argmax_per_pos
        );
    }
    // A10 fault seam (oracle only, see `arm_mtp_full_reject`): the committed
    // token is the trunk's own argmax, so greedy output must equal the
    // accepting MTP / AR sequence; the cell proves rejected draft rows never
    // become cache-visible (spec §4.6.2) and the repair path handles a
    // zero-length accepted prefix.
    let fault_full_reject = MTP_FULL_REJECT.load(std::sync::atomic::Ordering::Acquire);
    let accepted = if fault_full_reject {
        greedy_trunk_spine_accept(&[], &argmax_per_pos[..1], eos_token_id)
    } else {
        greedy_trunk_spine_accept(&draft.candidates, &argmax_per_pos, eos_token_id)
    };

    let mut committed = accepted.committed;
    let mut hit_eos = accepted.hit_eos;

    // Clamp the burst to what the engine can actually KEEP. The accept rule
    // above only early-stops on `eos_token_id`, but the commit path also
    // terminates on the auxiliary eot, on `max_tokens`, and on the context
    // cap — and the DN repair below replays exactly `advance` rows. If a
    // terminator fires mid-burst, committing the full accepted burst would
    // leave the recurrent state advanced past the session store for the rest
    // of the conversation (KV rewinds via `begin_turn`; DN cannot rewind).
    // Clamping here keeps `advance` == the rows the store will actually hold.
    //
    // `commit_budget` (min of the request's remaining max_tokens and its
    // remaining context room, at least 1) bounds how many tokens may be
    // committed; an auxiliary eot inside the kept prefix truncates the burst
    // to end on it, mirroring how the accept rule itself ends on eos.
    let mut keep = committed.len().min(commit_budget.max(1));
    for (i, t) in committed[..keep].iter().enumerate() {
        if i + 1 < keep && Some(*t) == aux_eot_id {
            keep = i + 1;
            hit_eos = true;
            break;
        }
    }
    committed.truncate(keep);

    let accept_count = accepted.accept_count;
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= drafts_generated + 1);
    if mtp_trace_enabled() {
        eprintln!(
            "[mtp-trace] result advance={} accept_count={} committed={:?} hit_eos={} (budget {})",
            advance, accept_count, committed, hit_eos, commit_budget
        );
    }

    // Next cycle's MTP head input: the trunk hidden at the last committed
    // row. `verify_hidden` already holds the POST-output-norm rows (normed
    // above), which is the convention the head was exported with and the one
    // the sequential path feeds it — so copy the row directly.
    let prev_hidden_row = advance - 1;
    gpu.hip.memcpy_dtod_at(
        &state.prev_hidden.buf,
        0,
        &verify_hidden.buf,
        prev_hidden_row * dim * 4,
        dim * 4,
    )?;

    // State repair: full accept leaves the verify-advanced state exactly
    // right; anything less rewinds the DN snapshot and replays only the
    // DeltaNet recurrence over the accepted rows.
    let full_accept_no_eos = advance == drafts_generated + 1 && !hit_eos;
    if !full_accept_no_eos {
        state.trunk_snap.restore_to(dn_state, gpu)?;
        mtp_dn_repair_from_tape(
            gpu,
            weights,
            config,
            dn_state,
            verify_tape,
            slot * tape_stride,
            advance,
        )?;
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated,
        chain_truncated: draft.chain_truncated,
        replay_skipped: full_accept_no_eos,
    })
}

/// [`ModelSlot`]-based wrapper for [`mtp_shared_verify_accept_rollback_inner`].
#[allow(clippy::too_many_arguments)]
fn mtp_shared_verify_accept_rollback(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    candidates: &[u32],
    drafts_generated: usize,
    chain_truncated: bool,
    overlap_trunk_snap: bool,
    use_sampling: bool,
    sampling: MtpSamplingConfig,
    draft_probs: &[f32],
    draft_softmaxes: &[Vec<f32>],
    is_external: bool,
    use_device_token_chain: bool,
) -> HipResult<MtpSpecResult> {
    mtp_shared_verify_accept_rollback_inner(
        gpu,
        &target.weights,
        &target.config,
        &mut target.kv_cache,
        &mut target.dn_state,
        &mut target.scratch,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        candidates,
        drafts_generated,
        chain_truncated,
        overlap_trunk_snap,
        use_sampling,
        sampling,
        draft_probs,
        draft_softmaxes,
        is_external,
        use_device_token_chain,
    )
}

/// Which trunk prefill route the MTP prompt fill drives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpPromptRoute {
    /// The trunk prefills exactly like AR's serve prefill (same outer chunk
    /// plan, widened chunks, GDN chunk scan, standard dispatch workload) and
    /// hands its hidden rows to the head. KV, DeltaNet state and the
    /// first-token logits match an AR prefill of the same tokens.
    #[default]
    ArRoute,
    /// Legacy route (`HIPFIRE_MTP_OWN_PREFILL=1`): `prefill_max_batch`-row
    /// trunk chunks captured as a speculative verify (sequential GDN
    /// recurrence, `SpeculativeVerify` dispatch workload).
    Own,
}

impl MtpPromptRoute {
    /// Resolve from the config-owned opt-out (`HIPFIRE_MTP_OWN_PREFILL`).
    pub fn from_own_prefill(own_prefill: bool) -> Self {
        if own_prefill {
            Self::Own
        } else {
            Self::ArRoute
        }
    }
}

/// DS4-style Qwen MTP prefill fill.
///
/// Runs trunk prefill while capturing post-output-norm hidden for every
/// prompt token, then runs the MTP block-only path over those same prompt
/// positions. The MTP layer enters decode with a warm private KV cache instead
/// of starting at the first generated token.
///
/// The trunk chunk plan follows `route` (see [`MtpPromptRoute`]). The head
/// fills in slices of at most `head_rows` rows aligned to the start of the
/// fill: `prefill_max_batch` by default, or (default ArRoute on gfx1201 with
/// MQ4G256 `eh_proj`/`wk`/`wv`) the wider lean-scratch capacity fitting the
/// legacy scratch+rotation byte budget. Each row is independent, so a given
/// set of hidden rows yields the same MTP KV for any slice width.
/// Callers observe a committed boundary after each trunk chunk. Adaptive KV
/// never reaches this path (qwen generate refuses speculation under
/// `kv_adaptive`).
///
/// MTP private-cache offsets stay absolute: each slice fills MTP KV at
/// `start_pos + off + i` (same schedule as the trunk), so multi-chunk prefill
/// is position-identical to a single whole-prompt fill. End state: full trunk
/// KV + full MTP private KV + last-token logits.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrunkSpinePrefillTimings {
    pub trunk_prefill_secs: f64,
    pub mtp_prompt_fill_secs: f64,
}

/// Absolute trunk positions at which a multi-chunk MTP prefill has committed
/// KV after each `<= chunk_max` slice. Empty when `prompt_len == 0`.
///
/// Used by adaptive serve hooks and unit tests so downshift sites stay locked
/// to the same schedule the prefill helper actually writes.
pub fn mtp_prefill_committed_boundaries(
    prompt_len: usize,
    start_pos: usize,
    chunk_max: usize,
) -> Vec<usize> {
    debug_assert!(chunk_max >= 1);
    let chunk_max = chunk_max.max(1);
    if prompt_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(prompt_len.div_ceil(chunk_max));
    let mut off = 0usize;
    while off < prompt_len {
        let end = (off + chunk_max).min(prompt_len);
        out.push(start_pos + end);
        off = end;
    }
    out
}

/// Temporary batched MTP fill capacity. `None` means allocate nothing
/// (empty prompt). Otherwise bounded by actual prompt rows, never the
/// architecture ceiling for a smaller prompt.
fn mtp_prompt_fill_scratch_rows(prompt_len: usize, prefill_max_batch: usize) -> Option<usize> {
    if prompt_len == 0 {
        None
    } else {
        Some(prefill_max_batch.min(prompt_len).max(1))
    }
}

/// Whether the head's prompt fill takes the batched route (and so allocates the
/// per-request fill scratch + rotation buffer at all).
pub fn mtp_head_fills_batched(head: &Qwen35MtpHead) -> bool {
    mtp_head::mtp_prompt_fill_uses_batched(
        mtp_head::MtpKvMode::Q8,
        &[
            head.weights.eh_proj.gpu_dtype,
            head.weights.wq.gpu_dtype,
            head.weights.wk.gpu_dtype,
            head.weights.wv.gpu_dtype,
        ],
    )
}

/// Temporary device bytes [`prefill_trunk_and_mtp_cache_parts`] allocates per
/// request for a prompt of at least `prefill_max_batch` tokens: the pairing
/// hidden window `(rows + 1) * dim` F32 and, when `batched`, the head fill
/// scratch plus rotation buffer at the legacy `rows = prefill_max_batch`. The
/// lean wide-slice layout (`new_for_prompt_fill`) never exceeds this legacy
/// byte budget (its row count is chosen against it), so this bounds both.
pub fn mtp_prompt_fill_requested_bytes(
    config: &Qwen35Config,
    head_config: &mtp_head::Qwen35MtpHeadConfig,
    batched: bool,
    prefill_max_batch: usize,
) -> usize {
    let rows = prefill_max_batch.max(1);
    let hidden = (rows + 1) * config.dim * 4;
    if !batched {
        return hidden;
    }
    let legacy_rotation_k = (2 * config.dim)
        .max(head_config.n_ff)
        .max(config.dim)
        .max(head_config.n_head * head_config.head_dim);
    hidden
        + Qwen35MtpHeadBatchedScratch::legacy_prompt_fill_bytes(
            head_config,
            rows,
            legacy_rotation_k,
        )
}

#[cfg(test)]
mod lazy_state_bytes_tests {
    use super::*;
    use crate::mtp_head::{Qwen35MtpFfnKind, Qwen35MtpHeadConfig};

    fn config_27b() -> Qwen35Config {
        Qwen35Config {
            dim: 5120,
            n_layers: 64,
            vocab_size: 248320,
            norm_eps: 1e-6,
            eos_token: 2,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            rope_theta: 500000.0,
            partial_rotary_factor: 0.25,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0, 0, 0],
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            conv_kernel_dim: 4,
            hidden_dim: 17408,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_shared_expert: false,
            norm_topk_prob: false,
            layer_types: (0..48)
                .map(|_| LayerType::LinearAttention)
                .chain((0..16).map(|_| LayerType::FullAttention))
                .collect(),
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
            reap_keep: None,
        }
    }

    fn head_27b(max_seq: usize) -> Qwen35MtpHeadConfig {
        Qwen35MtpHeadConfig {
            n_embd: 5120,
            n_head: 24,
            n_head_kv: 4,
            head_dim: 256,
            n_ff: 17408,
            ffn_kind: Qwen35MtpFfnKind::Dense,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: false,
            vocab_size: 248320,
            rope_theta: 10_000_000.0,
            n_rot: 64,
            rms_norm_eps: 1e-6,
            max_seq,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn gdn_tape_bytes_match_shape_arithmetic() {
        let c = config_27b();
        // 48 LA layers x rows x (qkv 10240 + 2 x 48 heads) + 2 x k_dim(2048)
        // + 4 x v_dim(6144) per row, F32.
        let rows = 8usize;
        let capture = (48 * rows * (10240 + 96) + rows * (2 * 2048 + 4 * 6144)) * 4;
        assert_eq!(GdnTape::requested_bytes(&c, rows, false), capture);
        // Replay arming adds 4 shared regions of 48 x rows x 6144 F32 plus the
        // two 48-row pointer tables.
        let replay = 4 * 48 * rows * 6144 * 4
            + 48 * (std::mem::size_of::<rdna_compute::dflash_gdn_replay::DflashReplayPreLayer>()
                + std::mem::size_of::<rdna_compute::dflash_gdn_replay::GdnLayerTable>());
        assert_eq!(GdnTape::requested_bytes(&c, rows, true), capture + replay);
    }

    #[test]
    fn spec_state_bytes_match_shape_arithmetic() {
        let c = config_27b();
        let h = head_27b(95664);
        let (max_n, vc) = (5usize, 5usize);
        let rows = vc + 1;
        let snap = 123_456_789usize;
        let got = MtpSpecState::requested_bytes("gfx1100", &c, snap, &h, None, max_n, vc).unwrap();
        let small = 5120
            + 3 * rows * 5120
            + rows * 248320
            + rows
            + 3 * max_n * 5120
            + max_n * 248320
            + max_n
            + (max_n + 1)
            + 5120
            + max_n
            + 6
            + PENALTY_REPEAT_CAP
            + rows
            + 2
            + 2 * max_n;
        let expect = small * 4
            + snap
            + PrefillBatchScratch::projected_allocation_bytes(&c, rows, true).unwrap() as usize
            + GdnTape::requested_bytes(&c, rows, true)
            + Qwen35MtpHeadScratch::requested_bytes("gfx1100", &h, None)
            + Qwen35MtpHeadKvCache::q8_requested_bytes(&h);
        assert_eq!(got, expect);
        // A compressed-vocab head adds the sidecar logits scratch pair.
        let cvs = 32768usize;
        let with =
            MtpSpecState::requested_bytes("gfx1100", &c, snap, &h, Some(cvs), max_n, vc).unwrap();
        assert_eq!(with, got + (max_n * cvs + cvs) * 4);
        // Head KV: K + V, 95664 x 4 heads x 8 blocks x 34 B.
        assert_eq!(
            Qwen35MtpHeadKvCache::q8_requested_bytes(&h),
            2 * 95664 * 32 * 34
        );
    }

    #[test]
    fn prompt_fill_bytes_match_shape_arithmetic() {
        let c = config_27b();
        let h = head_27b(95664);
        let hidden = 513 * 5120 * 4;
        assert_eq!(mtp_prompt_fill_requested_bytes(&c, &h, false, 512), hidden);
        // Legacy layout: 565_252 B/row of scratch + a 17408-wide rotation row.
        assert_eq!(
            mtp_prompt_fill_requested_bytes(&c, &h, true, 512),
            hidden + 512 * 565_252 + 512 * 17408 * 4
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_trunk_and_mtp_cache(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt_tokens: &[u32],
    start_pos: usize,
    route: MtpPromptRoute,
) -> HipResult<TrunkSpinePrefillTimings> {
    // No-op boundary: same final state as the historical single-call path.
    prefill_trunk_and_mtp_cache_with_boundary(
        gpu,
        target,
        head,
        state,
        prompt_tokens,
        start_pos,
        route,
        |_gpu, _target, _kv, _committed_pos| Ok(()),
    )
}

/// Like [`prefill_trunk_and_mtp_cache`], but invokes `on_committed_boundary`
/// after each trunk chunk and its MTP fill commit, with the trunk DeltaNet
/// state and the exclusive end position of the committed prefix
/// (`start_pos + tokens_written_so_far`).
///
/// The callback runs while `target.kv_cache` holds the just-written prefix.
/// Failures propagate and abort the remaining prefill (caller must free MTP
/// state).
#[allow(clippy::too_many_arguments)]
pub fn prefill_trunk_and_mtp_cache_with_boundary<F>(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt_tokens: &[u32],
    start_pos: usize,
    route: MtpPromptRoute,
    on_committed_boundary: F,
) -> HipResult<TrunkSpinePrefillTimings>
where
    F: FnMut(&mut Gpu, &DeltaNetState, &KvCache, usize) -> HipResult<()>,
{
    let mut parts = MtpPrefillTarget {
        weights: &target.weights,
        config: &target.config,
        kv_cache: &mut target.kv_cache,
        dn_state: &mut target.dn_state,
        scratch: &target.scratch,
    };
    prefill_trunk_and_mtp_cache_parts(
        gpu,
        &mut parts,
        head,
        state,
        prompt_tokens,
        start_pos,
        route,
        on_committed_boundary,
    )
}

/// The trunk owners a prompt prefill writes: a [`ModelSlot`]'s, or a
/// continuous-batching request's own KV/DeltaNet over the shared weights.
pub struct MtpPrefillTarget<'a> {
    pub weights: &'a Qwen35Weights,
    pub config: &'a Qwen35Config,
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub scratch: &'a Qwen35Scratch,
}

/// [`prefill_trunk_and_mtp_cache_with_boundary`] over explicit trunk owners.
#[allow(clippy::too_many_arguments)]
pub fn prefill_trunk_and_mtp_cache_parts<F>(
    gpu: &mut Gpu,
    target: &mut MtpPrefillTarget<'_>,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt_tokens: &[u32],
    start_pos: usize,
    route: MtpPromptRoute,
    mut on_committed_boundary: F,
) -> HipResult<TrunkSpinePrefillTimings>
where
    F: FnMut(&mut Gpu, &DeltaNetState, &KvCache, usize) -> HipResult<()>,
{
    let Some(head_rows) =
        mtp_prompt_fill_scratch_rows(prompt_tokens.len(), qwen35::prefill_max_batch(gpu))
    else {
        return Ok(TrunkSpinePrefillTimings::default());
    };
    // Trunk outer-chunk ceiling. ArRoute copies AR's serve prefill caller
    // (`hipfire-generate` `ar.rs`, no-eviction branch) and queries it before
    // the MTP buffers below take memory, as AR's query sees it.
    let trunk_chunk_max = match route {
        MtpPromptRoute::Own => head_rows,
        MtpPromptRoute::ArRoute => match qwen35::prefill::ordinary_prefill_chunk_limit_deducting(
            gpu,
            &target.weights,
            &target.config,
            &target.dn_state,
            &target.kv_cache,
            qwen35::prefill::request_kv_growth_deduction(
                gpu,
                &target.kv_cache,
                start_pos + prompt_tokens.len(),
            ),
        ) {
            Ok(limit) => limit,
            Err(e) => {
                eprintln!("mtp prefill: chunk-limit query failed ({e}); keeping legacy ceiling");
                qwen35::prefill_max_batch(gpu)
            }
        },
    };

    let dim = target.config.dim;
    let dim_bytes = dim * 4;
    // Pure support/eligibility metadata, decided before any head or hidden
    // buffer takes memory. The head fills in `head_rows` slices whatever the
    // trunk chunk size.
    let use_batched_mtp_fill = mtp_head::mtp_prompt_fill_uses_batched(
        state.mtp_kv.kv_mode,
        &[
            head.weights.eh_proj.gpu_dtype,
            head.weights.wq.gpu_dtype,
            head.weights.wk.gpu_dtype,
            head.weights.wv.gpu_dtype,
        ],
    );
    // Lean wide-slice fill: only the default single-GPU ArRoute on exact
    // gfx1201 with plain MQ4G256 eh_proj/wk/wv (the projections the kv_only
    // fill actually runs). Everything else keeps the legacy scratch and rows.
    let fill_fast_wide = use_batched_mtp_fill
        && mtp_head::mtp_fill_fast_enabled()
        && route == MtpPromptRoute::ArRoute
        && gpu.arch == "gfx1201"
        && [&head.weights.eh_proj, &head.weights.wk, &head.weights.wv]
            .iter()
            .all(|w| w.gpu_dtype == DType::MQ4G256 && w.awq_scale.is_none());
    let legacy_rotation_k = (2 * dim)
        .max(head.config.n_ff)
        .max(dim)
        .max(head.config.n_head * head.config.head_dim);
    let fill_rotation_k = head
        .weights
        .eh_proj
        .k
        .max(head.weights.wk.k)
        .max(head.weights.wv.k);
    // Slice capacity: the legacy rows unless the lean layout fits more rows in
    // the legacy scratch+rotation byte budget (never above the trunk chunk
    // ceiling queried above with the legacy rows, nor the prompt length).
    let head_rows = if fill_fast_wide {
        Qwen35MtpHeadBatchedScratch::prompt_fill_rows_for_budget(
            &head.config,
            head_rows,
            trunk_chunk_max,
            prompt_tokens.len(),
            legacy_rotation_k,
            fill_rotation_k,
        )
    } else {
        head_rows
    };
    let rotation_k = if fill_fast_wide {
        fill_rotation_k
    } else {
        legacy_rotation_k
    };
    // The head pairs token `p` with the trunk hidden of position `p - 1`, as
    // decode does. The pairing buffer holds one outer chunk plus its partner
    // row, not the whole prompt (a whole-prompt F32 buffer is 20 KiB/token on
    // a 5120-dim trunk, allocated per request and charged nowhere): row 0
    // holds the partner of the chunk's first token, row `j + 1` the hidden of
    // chunk position `j`, so chunk row `j` pairs `chunk[j]` with row `j`.
    // After each chunk its last hidden moves to row 0 for the next chunk.
    let chunk_cap = match route {
        MtpPromptRoute::Own => head_rows,
        MtpPromptRoute::ArRoute => trunk_chunk_max.max(1),
    };
    let window_rows = prompt_tokens.len().min(chunk_cap);
    let prompt_hidden = gpu.alloc_tensor(&[(window_rows + 1) * dim], DType::F32)?;
    let mut mtp_prefill_scratch = if use_batched_mtp_fill {
        let created = if fill_fast_wide {
            Qwen35MtpHeadBatchedScratch::new_for_prompt_fill(gpu, &head.config, head_rows)
        } else {
            Qwen35MtpHeadBatchedScratch::new(gpu, &head.config, head_rows)
        };
        match created {
            Ok(scratch) => Some(scratch),
            Err(error) => {
                let _ = gpu.free_tensor(prompt_hidden);
                return Err(error);
            }
        }
    } else {
        None
    };
    let mtp_prefill_rot = if use_batched_mtp_fill {
        match gpu.alloc_tensor(&[head_rows * rotation_k], DType::F32) {
            Ok(tensor) => Some(tensor),
            Err(error) => {
                if let Some(scratch) = mtp_prefill_scratch.take() {
                    scratch.free_gpu(gpu);
                }
                let _ = gpu.free_tensor(prompt_hidden);
                return Err(error);
            }
        }
    } else {
        None
    };

    let result = (|| -> HipResult<TrunkSpinePrefillTimings> {
        let mut trunk_prefill_secs = 0.0f64;
        let mut mtp_prompt_fill_secs = 0.0f64;
        let mut off = 0usize;
        // Partner of the first token is h_{start_pos-1}. Reuse `prev_hidden`
        // only when it holds that position; otherwise (cold start, or a warm
        // fill whose predecessor hidden is gone) pair the token with a zero
        // hidden, which the head's hnorm maps to zero.
        if start_pos > 0 && state.prev_hidden_pos == Some(start_pos - 1) {
            gpu.hip
                .memcpy_dtod_at(&prompt_hidden.buf, 0, &state.prev_hidden.buf, 0, dim_bytes)?;
        } else if mtp_head::mtp_fill_fast_enabled() {
            gpu.bind_thread()?;
            // D32 counts f32 words: zero only the first partner-hidden row.
            gpu.hip
                .memset_d32_async(&prompt_hidden.buf, 0, dim, gpu.active_stream.as_ref())?;
        } else {
            gpu.hip
                .memcpy_htod(&prompt_hidden.buf, &vec![0u8; dim_bytes])?;
        }
        while off < prompt_tokens.len() {
            let remaining = prompt_tokens.len() - off;
            let outer = match route {
                MtpPromptRoute::Own => remaining.min(head_rows),
                MtpPromptRoute::ArRoute => {
                    qwen35::prefill::ordinary_serve_prefill_chunk_len(remaining, trunk_chunk_max)
                        .unwrap_or(remaining.min(trunk_chunk_max).max(1))
                }
            };
            let end = off + outer;
            let chunk = &prompt_tokens[off..end];
            let chunk_start_pos = start_pos + off;
            let committed_pos = start_pos + end;

            // Per-chunk hidden destination: rows `1..` of the pairing window.
            let chunk_hidden = prompt_hidden.sub_offset(dim, chunk.len() * dim);

            let t_trunk = Instant::now();
            match route {
                MtpPromptRoute::ArRoute => qwen35::forward_prefill_batch_capture_hidden(
                    gpu,
                    &target.weights,
                    &target.config,
                    chunk,
                    chunk_start_pos,
                    &mut target.kv_cache,
                    &mut target.dn_state,
                    &target.scratch,
                    &chunk_hidden,
                )?,
                MtpPromptRoute::Own => qwen35::forward_prefill_batch(
                    gpu,
                    &target.weights,
                    &target.config,
                    chunk,
                    chunk_start_pos,
                    &mut target.kv_cache,
                    &mut target.dn_state,
                    &target.scratch,
                    None,
                    Some(&chunk_hidden),
                    None,
                    None,
                )?,
            }
            trunk_prefill_secs += t_trunk.elapsed().as_secs_f64();

            let t_mtp_fill = Instant::now();
            if let (Some(scratch), Some(rot)) =
                (mtp_prefill_scratch.as_mut(), mtp_prefill_rot.as_ref())
            {
                // Slices end on multiples of `head_rows` from the fill start,
                // independent of the trunk chunk boundaries.
                let mut slice_off = off;
                while slice_off < end {
                    let slice_end = ((slice_off / head_rows + 1) * head_rows).min(end);
                    let slice = &prompt_tokens[slice_off..slice_end];
                    let slice_hidden =
                        prompt_hidden.sub_offset((slice_off - off) * dim, slice.len() * dim);
                    let positions: Vec<i32> = (start_pos + slice_off..start_pos + slice_end)
                        .map(|position| position as i32)
                        .collect();
                    mtp_head::mtp_head_forward_block_batched(
                        gpu,
                        head,
                        scratch,
                        &mut state.mtp_kv,
                        slice,
                        &slice_hidden,
                        &positions,
                        slice.len(),
                        &target.weights,
                        Some(rot),
                        true,
                    )?;
                    slice_off = slice_end;
                }
            } else {
                for (i, &token) in chunk.iter().enumerate() {
                    let hidden_row = prompt_hidden.sub_offset(i * dim, dim);
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        token,
                        &hidden_row,
                        None,
                        chunk_start_pos + i,
                        &target.weights,
                    )?;
                }
            }
            mtp_prompt_fill_secs += t_mtp_fill.elapsed().as_secs_f64();
            // Carry the chunk's last hidden (row `chunk.len()`) to row 0: it
            // partners the next chunk's first token and, after the last
            // chunk, seeds `prev_hidden`. Ordered after the fill on the
            // active stream when there is one.
            match gpu.active_stream.as_ref() {
                Some(stream) => gpu.hip.memcpy_dtod_async_at(
                    &prompt_hidden.buf,
                    0,
                    &prompt_hidden.buf,
                    chunk.len() * dim_bytes,
                    dim_bytes,
                    stream,
                )?,
                None => gpu.hip.memcpy_dtod_at(
                    &prompt_hidden.buf,
                    0,
                    &prompt_hidden.buf,
                    chunk.len() * dim_bytes,
                    dim_bytes,
                )?,
            }

            // Committed boundary: trunk + MTP private KV now cover [0, committed_pos).
            on_committed_boundary(gpu, target.dn_state, &*target.kv_cache, committed_pos)?;
            // `prefill_progress` for the request's prompt prefill (live only
            // inside a `prompt_phase`; silent for decode-time replays).
            hipfire_runtime::prefill_progress::report(end, prompt_tokens.len());
            off = end;
        }

        let last = prompt_tokens.len() - 1;
        gpu.hip
            .memcpy_dtod_at(&state.prev_hidden.buf, 0, &prompt_hidden.buf, 0, dim_bytes)?;
        state.prev_hidden_pos = Some(start_pos + last);
        Ok(TrunkSpinePrefillTimings {
            trunk_prefill_secs,
            mtp_prompt_fill_secs,
        })
    })();

    if let Some(rot) = mtp_prefill_rot {
        let _ = gpu.free_tensor(rot);
    }
    if let Some(scratch) = mtp_prefill_scratch {
        scratch.free_gpu(gpu);
    }
    let _ = gpu.free_tensor(prompt_hidden);
    result
}

/// Explicit DS4-shaped Qwen MTP spec step.
///
/// This is the discrete-token trunk spine: serial MTP proposals, one trunk
/// batched verify, greedy/sampling accept, then rollback/replay only when the
/// full verify state cannot be kept. It intentionally delegates to the mature
/// compressed-serial implementation, whose name describes the lm_head storage
/// mode rather than the spine architecture.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_trunk_spine(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let k = state.max_n;
    spec_step_mtp_compressed_serial_with_k(
        gpu,
        target,
        head,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        k,
    )
}

/// Budget-aware trunk spine: draft at most `k` candidates (k may be 0).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_trunk_spine_with_k(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    k: usize,
) -> HipResult<MtpSpecResult> {
    spec_step_mtp_compressed_serial_with_k(
        gpu,
        target,
        head,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        k,
    )
}

/// One MTP-only spec-decode cycle (greedy, temp=0).
///
/// Preconditions:
/// - Trunk has been forwarded through position `cur_pos - 1`. Trunk's
///   `dn_state` is post-position-`cur_pos - 1` ready to accept the next
///   token at position `cur_pos`. `target.kv_cache` has K/V written at
///   positions `[0..cur_pos)`. `target.scratch.tmp` is unused / available.
/// - `state.prev_hidden` holds the trunk's post-output-norm hidden at
///   position `cur_pos - 1` (i.e., the position whose argmax produced
///   `last_committed`).
/// - MTP head KV cache is "rolling": stale slots from prior rejected
///   chains may exist at slots `>= cur_pos`; they will be overwritten in
///   place by this cycle's chain.
/// - `last_committed` is the token id at logical position `cur_pos - 1`.
///   It IS verified again as block[0] of this cycle's verify input — its
///   K/V row in trunk's cache stays consistent (idempotent re-write at
///   position cur_pos - 1? No — at position cur_pos - 1 the token was
///   written at the PREVIOUS cycle's last verify or replay. block[0]
///   here is at position cur_pos: trunk's verify will write trunk KV
///   slot cur_pos with last_committed's K/V. NOTE: this means
///   `last_committed`'s "logical" position for THIS cycle's verify is
///   cur_pos, NOT cur_pos - 1.
///
/// So the precise semantics: `cur_pos` is the position where this cycle's
/// verify block[0] (= `last_committed` re-emit) lands. The PREVIOUS cycle
/// committed `last_committed` at position `cur_pos - 1`? Or at position
/// `cur_pos`?
///
/// Actually re-reading spec_step_dflash @ 2560-2583: the seed token is
/// re-fed at position `position` (the `position` parameter), and verify
/// runs `block` at positions `[position..position + B)`. So the seed is
/// at `position`, NOT `position - 1`. This is intentional: each cycle
/// re-runs the seed forward to land trunk K/V in slot `position` (because
/// after the previous cycle's replay, trunk advanced to `position` but
/// did NOT yet write a K/V slot at `position` for the bonus_token — the
/// bonus is "deferred" to be written here as block[0]).
///
/// We mirror that convention. Caller supplies `cur_pos` = position where
/// `last_committed` (= prev cycle's bonus) sits. After cycle, advance by
/// `result.advance`. NEW commits include indices [cur_pos+1..cur_pos+advance]
/// PLUS we also "re-confirm" cur_pos itself via block[0]. The caller's
/// emitted-token list should NOT include `last_committed` again.
///
/// Postconditions:
/// - Trunk dn_state advanced to `cur_pos + advance` (state ready for next
///   cycle's verify at position `cur_pos + advance`).
/// - Trunk kv_cache has K/V at positions `[0..cur_pos + advance)`.
/// - `state.prev_hidden` updated to trunk's post-output-norm hidden at
///   position `cur_pos + advance - 1` (= the LAST committed token).
/// - MTP KV cache may have stale slots at `>= cur_pos + advance`; OK.
/// - `result.hit_eos`: true iff any committed token equals `eos_token_id`.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Ensure stream exists for any async memset (downstream batched kernels
    // assume it). Mirrors spec_step_dflash convention.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. MTP candidate chain (Approach B: feature-only K-step) ────────
    //
    // ARCHITECTURALLY LOSSY: each step k (k > 0) feeds the previous step's
    // `t_mtp_out` directly as both the "embedding of the predicted token"
    // (bypassing the embedding table — see `mtp_head_forward_block_only`'s
    // `next_token_embed` doc comment) AND as `prev_hidden`. This lets us
    // chain all K block forwards back-to-back WITHOUT running lm_head per
    // step, then collapse the K argmaxes into a single batched lm_head GEMM.
    //
    // Trade-off (vs the earlier discrete-token-roundtrip path that called
    // `mtp_head_forward` K times): saves K-1 separate GEMV launches into
    // the trunk's vocab×n_embd lm_head matrix. Each saved launch is ~12 ms
    // for the 27B-3.5 lm_head (vocab=152K). On gfx1100 K=3 the projected
    // win is ~22 ms/cycle = ~37%-1.5× lift if τ holds.
    //
    // Lossless guarantee preserved at trunk-verify: any incorrect MTP
    // candidate (from the lossy chain or otherwise) is rejected by trunk
    // argmax check; the cycle just degenerates to AR + bonus token. τ may
    // drop relative to the discrete-token-roundtrip path because the head
    // was trained with `embed[token]`, not raw `t_mtp_out`, as its
    // step-input distribution.
    //
    // MTP step k writes its own KV slot at position cur_pos + k (same as
    // before — a per-step single-layer transformer cache).
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        // Step 0 reads from state.prev_hidden (trunk's last post-norm
        // hidden, snapshot from the previous cycle / prefill) and embeds
        // last_committed via the trunk's table. Step k > 0 reads from row
        // (k-1) of mtp_t_outs and uses that same row as the embedding
        // override (lossy substitute for embed[predicted_token_{k-1}]).
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                last_committed,
                &state.prev_hidden,
                None,
                cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,               // sentinel: ignored when next_token_embed is Some(_)
                &prev_row,       // prev_hidden = step k-1's t_mtp_out
                Some(&prev_row), // embed override = step k-1's t_mtp_out (lossy)
                cur_pos + k,
                trunk_weights,
            )?;
        }
        // Capture this step's t_mtp_out into row k of mtp_t_outs so the
        // next step can read it AND so the end-of-chain batched lm_head
        // can ingest the whole stack.
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf,
            k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf,
            0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched lm_head over all K t_mtp_outs ────────────────────────
    //
    // Single GEMM (N=max_n) replacing K separate GEMV calls. shared_head_norm
    // is applied row-wise via rmsnorm_batched inside the helper. MQ4
    // dispatch path includes the FWHT rotation pre-pass (rotate_x_mq_batched).
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = state.mtp_lm_logits.sub_offset(0, max_n * vocab);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu,
        head,
        &trunk_weights.output,
        &t_outs_view,
        &lm_tmp_view,
        &lm_rot_view,
        &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over (max_n) rows → single D2H of K i32 ids.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, vocab, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, max_n * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let candidates: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 2. Build trunk verify input: [last_committed, c1, ..., c_max_n] ──
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    debug_assert_eq!(verify_tokens.len(), max_n + 1);

    // ── 3. Snapshot trunk DN state (rollback target before verify) ───────
    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    // ── 4. Trunk batched verify, capturing per-position hidden ───────────
    //
    // `forward_prefill_batch_with_pbs` writes per-position post-output-norm
    // hidden into `state.verify_hidden` when per_token_hidden_out is Some.
    // Trunk KV cache and dn_state advance by max_n+1 (= verify_tokens.len()).
    //
    // GDN-tape capture gate: the batched (PBS) verify path writes the GDN
    // innovation tape; the per-token fallback does not. Gate on the forward's
    // OWN eligibility predicate so the rollback's cheap-vs-full replay choice
    // tracks exactly whether the tape was written this cycle. Mirror
    // compressed_serial (lines 2604-2616) and DFlash (speculative.rs:3452-3456).
    let n_verify = max_n + 1;
    let moe_router_logits_present = state.trunk_pbs.moe_router_logits_batch.is_some();
    let tape_captured = qwen35::prefill_batch_pbs_eligible(
        trunk_weights,
        &target.config,
        &target.dn_state,
        n_verify,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    let verify_tape: Option<&mut GdnTape> = if tape_captured {
        Some(&mut state.trunk_gdn_tape)
    } else {
        None
    };
    // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): trunk verify forward + verify
    // lm_head under the one E4M3 verify arithmetic; closed on every path.
    let prev = fp8_verify_open(gpu, trunk_weights, &target.config, &target.kv_cache)?;
    let fwd = qwen35::forward_prefill_batch_with_pbs_opts(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None, // hidden_rb: not used here (DFlash drafter not in the loop)
        Some(&state.verify_hidden),
        verify_tape,
        None, // tree_verify
        Some(&state.trunk_pbs),
        None,  // mask_override
        None,  // max_layer
        false, // MTP computes all verify logits from verify_hidden below
        qwen35::DflashFusionCtx::Off,
    );

    // ── 5. Per-position lm_head + batched argmax ─────────────────────────
    //
    // Mirrors verify_dflash_block's batched lm_head dispatch by dtype.
    // Sized to (max_n + 1). Greedy-only, so we use GPU-side batched argmax
    // to avoid the (max_n + 1) × vocab D2H.
    // n_verify is hoisted above (before tape_captured computation).
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    let head = fwd.and_then(|()| {
        mtp_trunk_verify_lm_head(
            gpu,
            w_out,
            &state.verify_hidden,
            &state.verify_rot,
            &logits_view,
            n_verify,
            dim,
            vocab,
        )
    });
    gpu.fp8_verify_exit(prev);
    head?;

    // GPU-side batched argmax over (max_n + 1) rows.
    let argmax_view = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_view, vocab, n_verify)?;
    let mut argmax_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, n_verify * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 6. Accept-prefix ────────────────────────────────────────────────
    //
    // For k in [0..max_n): if MTP candidate `candidates[k]` matches trunk's
    // argmax at slot k (= trunk's prediction of token at position
    // cur_pos + k + 1), accept. Stop on first miss OR EOS.
    //
    // bonus_token = argmax_per_pos[accept_count] (always commits unless
    // an accepted MTP candidate was already EOS).
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);

    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        // Bonus: trunk's argmax at the first rejection slot (or after the
        // full accepted chain). Always commits one extra token.
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1, "spec_step_mtp must commit at least the bonus");
    debug_assert!(advance <= max_n + 1);

    // ── 7. Capture prev_hidden for next cycle ────────────────────────────
    //
    // Next cycle's MTP step 0 needs trunk's post-output-norm hidden whose
    // argmax PRODUCED the next cycle's last_committed (= this cycle's
    // last committed token). Verify-slot `i` holds the post-norm hidden at
    // trunk position `cur_pos + i`, which produces argmax_per_pos[i] for
    // position `cur_pos + i + 1`. The last committed token sits at:
    //
    //   - If hit_eos triggered inside the MTP-accepted chain (no bonus):
    //     advance == accept_count, last token at position cur_pos + advance
    //     (= cur_pos + accept_count), which is the i = advance - 1 slot.
    //   - Otherwise: last token = bonus at position cur_pos + accept_count + 1,
    //     advance == accept_count + 1, slot = advance - 1.
    //
    // Both cases reduce to slot `advance - 1`.
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, cur_pos, prev_hidden_row, dim)?;

    // ── 8. Roll back trunk DN state + replay accepted committed tokens ───
    //
    // After verify, trunk dn_state is at position cur_pos + max_n + 1.
    // We need it at cur_pos + advance for next cycle. Restore snapshot
    // (back to cur_pos), then replay first `advance` of `verify_tokens`.
    //
    // LEVER 1 (W3e): When the verify took the batched (PBS) path it populated
    // `state.trunk_gdn_tape` with the GDN innovation (qkvza projection +
    // post-sigmoid α/β) for each LA layer at each verify position. We replay
    // ONLY the GDN recurrence from the tape (~0.30ms) instead of a full B=2
    // trunk forward_prefill_batch (~12.7ms). This is the same GDN-only repair
    // that DFlash uses (speculative.rs:3465 dflash_use_gdn_tape_replay) and that
    // compressed_serial already wires at lines 2942-2987. Mirror that exactly.
    //
    // Safety: replay_gdn does NOT touch KV cache. Per the "MTP KV cache rollback:
    // NOT NEEDED" argument below, the accepted prefix's KV slots
    // [cur_pos..cur_pos+advance) are already correct from the verify forward (the
    // verify wrote the same tokens to the same positions as the replay would).
    // Stale slots beyond cur_pos+advance are overwritten in-order by next cycle
    // before they are attended. Skipping the KV re-write is safe for both the
    // full-replay and tape-replay paths.
    //
    // Instrument first cycle: print tape_captured so we can confirm the MoE
    // batched verify populates the tape (HIPFIRE_DEBUG_BATCH=1 already prints
    // eligibility; this one-shot print on cycle 0 is unconditional for W3e).
    // HIPFIRE_MTP_TAPE_REPLAY=0 forces the original full-replay path (A/B gate).
    let use_tape_replay = tape_captured
        && advance >= 2
        && hipfire_config::developer_var("HIPFIRE_MTP_TAPE_REPLAY")
            .ok()
            .as_deref()
            != Some("0");
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if use_tape_replay {
        // Batched verify populated the tape this cycle — cheap GDN-only replay.
        // Only fire for advance >= 2: for advance=1 the baseline uses forward_scratch
        // (decode-mode GEMV for wqkv), while the tape captured the batched WMMA
        // result. These differ at ULP level. To preserve byte-identical streams,
        // fall through to forward_scratch below when advance==1.
        state.trunk_gdn_tape.replay_gdn(
            gpu,
            trunk_weights,
            &target.config,
            &mut target.dn_state,
            advance,
        )?;
    } else {
        // Either tape not captured, HIPFIRE_MTP_TAPE_REPLAY=0 (A/B gate), or advance=1
        // (decode-mode-GEMV vs batch-WMMA ULP mismatch — use decode path for identity).
        if advance >= 2 {
            let replay = &verify_tokens[..advance];
            // Rollback replay inside the PM FP8 verify scope (see `mtp_accept_and_rollback`).
            let fp8 = fp8_verify_target(trunk_weights, &target.config, &target.kv_cache);
            crate::speculative::fp8_verify_scoped(gpu, fp8, |gpu| {
                qwen35::forward_prefill_batch(
                    gpu,
                    trunk_weights,
                    &target.config,
                    replay,
                    cur_pos,
                    &mut target.kv_cache,
                    &mut target.dn_state,
                    &target.scratch,
                    None,
                    None,
                    None,
                    None,
                )
            })?;
        } else {
            // advance == 1: only the seed (last_committed) is "replayed". This
            // happens when EOS hits with accept_count==0 AND bonus==eos
            // (very rare), or when the bonus was committed alone (also implies
            // accept_count==0, which is the common AR-degraded case).
            gpu.graphs.ar_graph_eligible = false; // spec re-seed: never the plain-AR graph
            qwen35::forward_scratch(
                gpu,
                trunk_weights,
                &target.config,
                verify_tokens[0],
                cur_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
            )?;
        }
    }

    // ── 9. MTP KV cache rollback: NOT NEEDED ────────────────────────────
    //
    // Argument: MTP step k of the original chain wrote slot cur_pos + k
    // with K/V derived from input (k==0 ? last_committed : candidates[k-1]).
    // For k in [0..accept_count], the actual committed token at position
    // cur_pos + k matches that input (last_committed for k=0, candidates[k-1]
    // = committed[k-1] for k≥1 because c_k was accepted), so those slots
    // hold correct K/V. For k in [accept_count+1..max_n), the chain wrote
    // K/V from rejected candidates — STALE. But the next cycle starts at
    // cur_pos_new = cur_pos + advance = cur_pos + accept_count + 1
    // (or cur_pos + accept_count if EOS-no-bonus), and step 0 of that
    // next cycle WRITES slot cur_pos_new BEFORE attending [0..cur_pos_new+1).
    // Stale slots [cur_pos + advance, cur_pos + max_n) are at-or-beyond
    // cur_pos_new = cur_pos + advance, so step 0 next cycle writes the
    // first stale slot before reading it. Step k next cycle writes the
    // (k+1)-th formerly-stale slot before its attention reads it. The
    // chain progresses left-to-right, overwriting stale slots in-order
    // exactly as needed. NO explicit rollback required.

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated: state.max_n,
        chain_truncated: false,
        replay_skipped: false,
    })
}

// ─── FastMTP-style compressed spec step (K = max_n) ───────────────────────
//
// Drop-in alternative to [`spec_step_mtp`] when the loaded MTP head ships a
// compressed `lm_head_draft` sidecar. Chains K = state.max_n MTP block
// forwards (k=0 normal, k>0 with lossy embedding override = previous step's
// `t_mtp_out`), then runs ONE batched compressed lm_head dispatch over the K
// stacked outputs (vs K full-vocab dispatches in the naive K-step path).
//
// Per-cycle BW (27B-3.5, K=3):
//   K full-vocab lm_head GEMVs: K * 635 MB ≈ 1.9 GB MQ4 streamed in
//   1 batched compressed lm_head GEMM: 84 MB MQ4 + K * 5120 * 4 B activations ≈ 84 MB
//   Saved: ~1.8 GB BW per cycle; trunk verify still runs full lm_head (n=K+1 batched).
//
// K=1 retains the original K=1 fast path; K>1 chains lossily (same OOD risk
// the existing `spec_step_mtp` warns about — τ may degrade vs discrete-token
// roundtrip). Lossless greedy preserved at the trunk-verify layer regardless.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Precondition: head must have compressed sidecar loaded + caller must
    // have allocated the compressed batched-logits scratch.
    let lm_head_draft = head
        .weights
        .lm_head_draft
        .as_ref()
        .expect("spec_step_mtp_compressed requires head loaded with --vocab-sidecar");
    let vocab_map = head
        .weights
        .lm_head_draft_vocab_map
        .as_ref()
        .expect("compressed head missing vocab_map");
    let cvs = head
        .weights
        .compressed_vocab_size
        .expect("compressed head missing compressed_vocab_size");
    let logits_c_batched = state.mtp_lm_logits_compressed.as_ref().expect(
        "MtpSpecState::mtp_lm_logits_compressed not allocated; \
                 call ensure_compressed_lm_logits(gpu, cvs) after head load",
    );
    assert!(
        logits_c_batched.numel() >= max_n * cvs,
        "mtp_lm_logits_compressed too small: {} < max_n*cvs ({})",
        logits_c_batched.numel(),
        max_n * cvs,
    );

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. Chain K MTP block forwards (lossy embedding override for k>0) ──
    //
    // Identical structure to spec_step_mtp's K-step lossy chain. Step 0
    // reads from state.prev_hidden + embeds last_committed via trunk's
    // token table. Step k>0 reads from row (k-1) of mtp_t_outs and uses
    // that row as the embedding override (continuous t_mtp_out as a lossy
    // substitute for embed[predicted_token_{k-1}]) so we don't need to
    // know the chain's interim tokens before running step k+1.
    //
    // The chain's ARCHITECTURAL LOSS comes from this OOD-for-the-head
    // continuous embedding — the MTP head was trained with discrete-token
    // roundtrips. τ may degrade. Trunk verify catches any wrong candidate
    // (lossless greedy preserved), but acceptance rate suffers.
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                last_committed,
                &state.prev_hidden,
                None,
                cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0, // ignored when next_token_embed = Some
                &prev_row,
                Some(&prev_row), // lossy embed override
                cur_pos + k,
                trunk_weights,
            )?;
        }
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf,
            k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf,
            0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched COMPRESSED lm_head over all K t_mtp_outs ──────────────
    //
    // Single GEMM with M=cvs (vs M=vocab=248K in the plain path) — this is
    // the FastMTP-style ~7.5x BW reduction lever. shared_head_norm applied
    // row-wise inside the helper. MQ4 path includes the FWHT pre-pass.
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = logits_c_batched.sub_offset(0, max_n * cvs);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu,
        head,
        lm_head_draft,
        &t_outs_view,
        &lm_tmp_view,
        &lm_rot_view,
        &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over K compressed logit rows → K i32 ids → remap.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, cvs, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, max_n * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    for raw in argmax_host {
        let draft_idx = raw as usize;
        assert!(
            draft_idx < cvs,
            "draft argmax {draft_idx} out of compressed vocab {cvs}"
        );
        candidates.push(vocab_map[draft_idx]);
    }

    // ── 2. Trunk verify on [last_committed, c1, ..., c_max_n] ─────────────
    let verify_tokens = build_trunk_spine_verify_tokens(last_committed, &candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): trunk verify forward + verify
    // lm_head under the one E4M3 verify arithmetic; closed on every path.
    let prev = fp8_verify_open(gpu, trunk_weights, &target.config, &target.kv_cache)?;
    let fwd = qwen35::forward_prefill_batch_with_pbs_opts(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,
        Some(&state.verify_hidden),
        None,
        None,
        Some(&state.trunk_pbs),
        None,  // mask_override
        None,  // max_layer
        false, // MTP computes all verify logits from verify_hidden below
        qwen35::DflashFusionCtx::Off,
    );

    // ── 3. Trunk batched lm_head over verify positions ─────────────────────
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    let head = fwd.and_then(|()| {
        mtp_trunk_verify_lm_head(
            gpu,
            w_out,
            &state.verify_hidden,
            &state.verify_rot,
            &logits_view,
            n_verify,
            dim,
            vocab,
        )
    });
    gpu.fp8_verify_exit(prev);
    head?;

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    // ── 4. Accept-prefix (K-general) ──────────────────────────────────────
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);
    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= max_n + 1);

    // ── 5. Capture prev_hidden from verify slot advance-1 ─────────────────
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, cur_pos, prev_hidden_row, dim)?;

    // ── 6. Roll back trunk DN state + replay accepted ─────────────────────
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        // Rollback replay inside the PM FP8 verify scope (see `mtp_accept_and_rollback`).
        let fp8 = fp8_verify_target(trunk_weights, &target.config, &target.kv_cache);
        crate::speculative::fp8_verify_scoped(gpu, fp8, |gpu| {
            qwen35::forward_prefill_batch(
                gpu,
                trunk_weights,
                &target.config,
                replay,
                cur_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
                None,
                None,
                None,
                None,
            )
        })?;
    } else {
        gpu.graphs.ar_graph_eligible = false; // spec re-seed: never the plain-AR graph
        qwen35::forward_scratch(
            gpu,
            trunk_weights,
            &target.config,
            verify_tokens[0],
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
        )?;
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated: state.max_n,
        chain_truncated: false,
        replay_skipped: false,
    })
}

/// F5c: distribution-preserving SAMPLED (temp>0) accept rule for full-vocab
/// MTP, mirroring the proven, coherence-validated DFlash convention
/// (`speculative.rs:3728-3855`) EXACTLY:
///
///   * INDEPENDENT per-side nuclei. The DRAFT truncated to its OWN top_p
///     nucleus at draft time (`draft_softmaxes[k]`, `draft_probs[k]`); the
///     TARGET truncates to its OWN top_p nucleus here (tau/Z from the target
///     verify logits, NOT the draft's tau). The two supports differ.
///   * Accept `u * p_d <= p_t`, with `p_d = draft_probs[k]` (the truncated
///     draft prob at the drafted token) and `p_t = target_row_k[candidate]`
///     (the truncated target prob at the same token). cactus_delta == 0 on the
///     serve path, so accept_prob == p_t — NO CACTUS bump.
///   * On rejection, the bonus is drawn from the residual
///     `(target_nucleus − draft_nucleus)₊` via `sample_residual`, across the
///     MISMATCHED nuclei. (NOT sample_top_p over the raw verify logits.)
///   * On full accept, the bonus is `sample_categorical` over the target
///     nucleus at position `*accept_count` (== drafts_generated).
///
/// `logits_view` is an OWNED `GpuTensor` (the caller's
/// `state.verify_logits.sub_offset(..)`), so it borrows nothing from `state`
/// and composes cleanly with the `&mut state` / `&mut gpu` here.
///
/// Returns `hit_eos`. Pushes committed tokens into `committed` and bumps
/// `*accept_count` for each accepted draft (matching the legacy accept block's
/// bookkeeping).
///
/// Full-vocab only: `candidates[k]` is already the trunk token id (vocab_map is
/// None in full-vocab mode), so it indexes the target softmax row directly.
///
/// `point_mass`: the draft is deterministic (ngram/PLD takeover), q = δ(c_k).
/// Leviathan then accepts with probability `p_t[c_k]` (`draft_probs[k]` MUST be
/// 1.0) and, on reject, draws from `p_t` with `c_k` removed and renormalized
/// via `sample_residual_point_mass` — identical to `sample_residual` against a
/// one-hot, without allocating a vocab row. `draft_softmaxes` is unused (may be
/// empty). RNG draw count/order is identical to the learned-draft path.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn mtp_sampled_accept(
    gpu: &mut Gpu,
    state: &mut MtpSpecState,
    logits_view: &GpuTensor,
    draft_softmaxes: &[Vec<f32>],
    draft_probs: &[f32],
    candidates: &[u32],
    drafts_generated: usize,
    vocab: usize,
    point_mass: bool,
    temp: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    eos_token_id: u32,
    committed: &mut Vec<u32>,
    accept_count: &mut usize,
) -> HipResult<bool> {
    // Target nucleus rows for ALL n_verify = drafts_generated + 1 positions.
    //
    // NOTE (deviation from the F5c spec text, which said batch=drafts_generated):
    // the all-accept bonus is sampled at row `*accept_count`, which equals
    // `drafts_generated` on full accept — beyond the first `drafts_generated`
    // rows. DFlash computes target softmax for ALL `b` (= drafts_generated + 1)
    // rows for exactly this reason (speculative.rs:3730-3756 uses `b`). We
    // follow the CODE/DFlash and softmax all `n_verify` rows so the all-accept
    // bonus row exists. `logits_view` is already sized `n_verify * vocab`.
    let n_verify = drafts_generated + 1;
    let probs_gpu = gpu.alloc_tensor(&[n_verify * vocab], DType::F32)?;
    let tau_gpu = gpu.alloc_tensor(&[n_verify], DType::F32)?;
    let z_gpu = gpu.alloc_tensor(&[n_verify], DType::F32)?;
    gpu.softmax_temp_topp_batched_into_f32(
        logits_view,
        &probs_gpu,
        &tau_gpu,
        &z_gpu,
        vocab,
        n_verify,
        temp,
        top_p,
        top_k,
        min_p,
    )?;
    let host_probs = gpu.download_f32(&probs_gpu)?;
    let tau = gpu.download_f32(&tau_gpu)?;
    let z = gpu.download_f32(&z_gpu)?;
    let _ = gpu.free_tensor(probs_gpu);
    let _ = gpu.free_tensor(tau_gpu);
    let _ = gpu.free_tensor(z_gpu);
    debug_assert_eq!(host_probs.len(), n_verify * vocab);

    // Build a single position's TRUNCATED target row on demand. The softmax
    // kernel's tau/Z encode AR's law (top_k pool, min_p, nucleus over the pool
    // mass), so apply_topp_trunc yields AR's distribution for the row.
    let target_row_at = |pos: usize| -> Vec<f32> {
        let mut row = host_probs[pos * vocab..(pos + 1) * vocab].to_vec();
        apply_topp_trunc(&mut row, tau[pos], z[pos]);
        row
    };

    let mut hit_eos = false;
    let mut rejected = false;
    for k in 0..drafts_generated {
        let target_row_k = target_row_at(k);
        let p_d = draft_probs[k].max(f32::MIN_POSITIVE);
        let p_t = target_row_k[candidates[k] as usize];
        let u = state.rng.next_uniform_f32();
        if u * p_d <= p_t {
            // Accept.
            committed.push(candidates[k]);
            *accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            // Reject: bonus from the residual (target_nucleus − draft_nucleus)₊
            // across the MISMATCHED per-side nuclei. Both rows are full-vocab
            // length `vocab` (different zero-support), so sample_residual's
            // element-wise subtraction is well-formed.
            let u2 = state.rng.next_uniform_f32();
            let bonus = if point_mass {
                sample_residual_point_mass(&target_row_k, candidates[k], u2)
            } else {
                sample_residual(&target_row_k, &draft_softmaxes[k], u2)
            };
            committed.push(bonus);
            if bonus == eos_token_id {
                hit_eos = true;
            }
            rejected = true;
            break;
        }
    }

    if !rejected && !hit_eos {
        // All `drafts_generated` accepted (and none was EOS): sample the bonus
        // from the target nucleus at position `*accept_count` (== drafts_generated).
        let row = target_row_at(*accept_count);
        let u = state.rng.next_uniform_f32();
        let bonus = sample_categorical(&row, u);
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }

    Ok(hit_eos)
}

// ─── FastMTP-style compressed SERIAL spec step (K serial roundtrips) ──────
//
// Variant of [`spec_step_mtp_compressed`] that runs K MTP steps as
// DISCRETE-TOKEN ROUNDTRIPS (not lossy chain): each step argmaxes its
// compressed logits, remaps via vocab_map to a full-vocab token id, and
// uses that token id (embedded via trunk's `token_embd` table) as the
// next step's input. Mirrors the original Task 10 "serial lm_head" path
// — but with COMPRESSED lm_head dispatches (~7.5x BW saving each).
//
// Per-cycle BW (27B-3.5, K=3, projection):
//   3 full lm_head GEMVs:        3 × 635 MB = 1.9 GB MQ4  (Task 10 baseline)
//   3 compressed lm_head GEMVs:  3 × 84 MB  = 252 MB MQ4  (this path)
//   Saved: ~1.65 GB BW per cycle → ~21-27 ms per cycle on gfx1100.
//
// Task 10 plain serial measured 39.68 tok/s τ=3.08; this path projects
// ~50-58 tok/s at the same τ (gate is 60). A trained sidecar (hiptrx
// trunk-argmax distillation) lifting τ from 3.08 → 3.5+ should clear.
//
// Trade-off vs the batched lossy chain: K serial dispatches launch K
// times instead of 1 batched, costing extra launch overhead and per-step
// argmax+D2H. On a 248K-vocab model this is dwarfed by the BW savings.
// On smaller-vocab models the lossy batched path may still be faster.
/// Legacy entry: drafts `state.max_n` candidates. Prefer
/// [`spec_step_mtp_compressed_serial_with_k`] when a per-call budget is known.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let k = state.max_n;
    spec_step_mtp_compressed_serial_with_k(
        gpu,
        target,
        head,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        k,
    )
}

/// Budget-aware compressed-serial MTP step: draft at most `k` candidates.
/// `k == 0` verifies `[last_committed]` only and emits the single bonus token.
/// MTP draft phase: K serial head-forward steps producing candidate tokens.
///
/// Extracted from the former `spec_step_mtp_compressed_serial_inner` so the
/// slot engine can batch trunk verify across slots. The single-slot path
/// calls this then `mtp_shared_verify_accept_rollback_inner`; the slot engine
/// calls this per-slot, then does a batched `forward_batch_slots_graphed`,
/// then `mtp_batched_verify_accept_from_batch` per-slot.
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_phase_inner(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    k: usize,
    skip_proposal_graph: bool,
) -> HipResult<MtpDraftOutput> {
    // Per-call k is authoritative (remaining max_emit from MtpSpeculator). Never
    // draft more than state.max_n, and k==0 means verify [seed] only + bonus.
    let max_n = k.min(state.max_n);
    let dim = config.dim;
    let vocab = config.vocab_size;
    let trunk_weights: &Qwen35Weights = weights;

    // Two modes for the K-step draft lm_head dispatch:
    //
    //   compressed (FastMTP-style): use head's own lm_head_draft (32K-row
    //     top-K vocab slice) plus vocab_map for the draft→full id remap.
    //     Small per-step GEMV (~0.09 ms BW on MQ4) but softmax is over the
    //     32K compressed vocab, which dilutes top-1 prob signal (compresses
    //     the distribution shape) and breaks --mtp-p-min.
    //
    //   full-vocab (bundled .mq4-mtp): use the trunk's own output (lm_head)
    //     directly as the draft head. Per-step GEMV is larger (~0.69 ms BW)
    //     but the softmax is over the real 248K vocab — top-1 prob is the
    //     true confidence the trunk would see, and the argmax id IS the
    //     committed token id (no vocab_map remap). Cost difference per cycle
    //     is small (~3 ms more across K=5) since trunk verify dominates.
    //     This is the architecture Unsloth/llama.cpp #22673 ship.
    //
    // Mode is selected by whether the loaded head carries a compressed sidecar.
    // Bundled .mq4-mtp files drop the sidecar; load_mtp_head_bundled returns
    // a head with lm_head_draft: None.
    let use_full_vocab = head.weights.lm_head_draft.is_none();
    let (vocab_map_opt, cvs): (Option<&Vec<u32>>, usize) = if use_full_vocab {
        (None, vocab)
    } else {
        let _ = head
            .weights
            .lm_head_draft
            .as_ref()
            .expect("compressed head missing lm_head_draft");
        let vm = head
            .weights
            .lm_head_draft_vocab_map
            .as_ref()
            .expect("compressed head missing vocab_map");
        let c = head
            .weights
            .compressed_vocab_size
            .expect("compressed head missing compressed_vocab_size");
        let _ = state.mtp_scratch.logits_compressed.as_ref().expect(
            "Qwen35MtpHeadScratch::logits_compressed not allocated; \
                     call mtp_scratch.ensure_compressed_logits(gpu, cvs) after head load",
        );
        (Some(vm), c)
    };

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    let dim_bytes = dim * 4;
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    let argmax_view = state.mtp_lm_argmax.sub_offset(0, 1);
    // Full-vocab per-step logits scratch: first row of mtp_lm_logits is
    // shape [vocab] which is exactly what we need per step. Allocated as
    // [max_n * vocab] elsewhere so the storage is already there.
    let full_vocab_logits_view = state.mtp_lm_logits.sub_offset(0, vocab);

    // p_min early-exit: when state.p_min > 0, each step uses
    // topk_logsumexp_batched (K=2) instead of argmax. We then check
    // top_logp[0] (log-softmax prob of argmax) against log(p_min) and
    // truncate the chain when the draft's own confidence drops below
    // threshold. Mirrors llama.cpp's --spec-draft-p-min (PR #22673).
    //
    // Semantics: the current step's candidate IS kept (we already computed
    // it); only steps k+1..max_n are skipped. This matches the upstream
    // implementation — a low-confidence draft still gets to face trunk
    // verify, but we stop spending compute speculating further.
    let p_min = state.p_min;
    let sampling = state.sampling;
    // The K-step draft pick and the trunk-verify accept rule run in exactly ONE
    // of three modes per step. Modeling it as `DraftMode` makes the illegal
    // `p_min + sampling` combination unrepresentable downstream — it was
    // previously a runtime `panic!` deep inside the step. The legacy
    // `use_p_min` / `use_sampling` / `log_p_min` bindings are derived from this
    // single source of truth below so they can never disagree.
    //
    // Sampling vs greedy: when sampling.temp > 0, the K-chain SAMPLES from each
    // draft distribution (instead of argmax) and trunk verify applies a residual
    // acceptance rule (instead of strict argmax-match). Matches Unsloth/llama.cpp
    // #22673 canonical MTP recipe (temp=1.0, top_p=0.95, top_k=20). p_min stops
    // speculating further once a draft's confidence drops below the cutoff.
    #[derive(Debug, Clone, Copy)]
    enum DraftMode {
        /// temp == 0, p_min == 0: argmax draft + strict argmax-match accept.
        Greedy,
        /// temp == 0, p_min > 0: top-2 logsumexp draft with a confidence cutoff.
        PMin { log_p_min: f32 },
        /// temp > 0: sampled draft + residual acceptance (temp/top_p in state.sampling).
        Sampled,
        /// temp > 0, p_min > 0: sampled draft (residual acceptance, as `Sampled`)
        /// PLUS the `PMin` confidence cutoff — truncate the K-chain early when the
        /// head's RAW top prob drops below p_min. Stays lossless (a shorter
        /// speculative chain only lowers tau); the τ lever for sampled MTP.
        SampledPMin { log_p_min: f32 },
    }
    let draft_mode = match (sampling.is_greedy(), p_min > 0.0) {
        // temp>0 + p_min>0: sampled draft WITH the p_min confidence cutoff. The
        // sampled draft samples from its own nucleus (not topk_logsumexp), and the
        // cutoff reads the head's raw top prob (computed alongside), so the two
        // compose — this used to be a hard Err (the kernels didn't share a path).
        (false, true) => DraftMode::SampledPMin {
            log_p_min: p_min.ln(),
        },
        (false, false) => DraftMode::Sampled,
        (true, true) => DraftMode::PMin {
            log_p_min: p_min.ln(),
        },
        (true, false) => DraftMode::Greedy,
    };
    let use_p_min = matches!(draft_mode, DraftMode::PMin { .. });
    let use_sampling = matches!(
        draft_mode,
        DraftMode::Sampled | DraftMode::SampledPMin { .. }
    );
    let use_sampled_pmin = matches!(draft_mode, DraftMode::SampledPMin { .. });
    let log_p_min = match draft_mode {
        DraftMode::PMin { log_p_min } | DraftMode::SampledPMin { log_p_min } => log_p_min,
        _ => f32::NEG_INFINITY,
    };
    let mut chain_truncated = false;
    let mut draft_probs: Vec<f32> = if use_sampling {
        Vec::with_capacity(max_n)
    } else {
        Vec::new()
    };
    // F5b: per-step TRUNCATED draft nucleus distributions (full-vocab sampled
    // path). Mirrors DFlash `draft_softmaxes` (speculative.rs:3494-3504): each
    // entry is the draft's OWN top_p-truncated, renormalized softmax row over
    // the full vocab, captured at draft time. Consumed by `mtp_sampled_accept`
    // for the residual bonus draw (sample_residual subtracts THIS draft nucleus
    // from the target nucleus). Empty on greedy/p_min paths.
    let mut draft_softmaxes: Vec<Vec<f32>> = if use_sampling {
        Vec::with_capacity(max_n)
    } else {
        Vec::new()
    };
    // Cached host buffer for per-step draft logits readback (sampling only).
    // Sized to whichever vocab the dispatch path uses (compressed cvs or
    // full 248K vocab). Allocated once outside the loop.
    let draft_logits_host: Vec<f32> = if use_sampling {
        let n = if use_full_vocab { vocab } else { cvs };
        vec![0.0; n]
    } else {
        Vec::new()
    };
    // k==0 one-token path: never run device-token chain / proposal graph
    // (both assume max_n ≥ 1 for candidate buffer views).
    let use_device_token_chain = max_n > 0
        && mtp_device_token_chain_enabled_from_env()
        && mtp_device_token_chain_eligible_for(trunk_weights.embd_format, use_sampling, use_p_min);
    if use_device_token_chain && !use_full_vocab {
        assert!(
            head.weights.lm_head_draft_vocab_map_gpu.is_some(),
            "compressed MTP device-token chain requires GPU vocab_map"
        );
    }
    let proposal_graph_policy = mtp_proposal_graph_policy_from_env();
    let use_proposal_graph = !skip_proposal_graph
        && max_n > 0
        && !state.mtp_proposal_graph_disabled
        && mtp_proposal_graph_eligible_for(
            proposal_graph_policy,
            use_device_token_chain,
            use_full_vocab,
            state.mtp_kv.kv_mode,
        );
    let proposal_graph_seq_cap = if use_proposal_graph {
        mtp_proposal_graph_seq_cap(cur_pos, max_n, state.mtp_kv.max_seq)
    } else {
        0
    };
    if use_proposal_graph
        && state.mtp_proposal_graph_exec.is_some()
        && proposal_graph_seq_cap > state.mtp_proposal_graph_seq_cap
    {
        destroy_mtp_proposal_graph(gpu, state);
        state.mtp_proposal_graph_warmed = true;
    }
    if use_device_token_chain {
        if use_proposal_graph {
            upload_mtp_proposal_graph_inputs(gpu, state, last_committed, cur_pos, max_n)?;
        } else {
            let seed_token = last_committed as i32;
            let seed_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(&seed_token as *const i32 as *const u8, 4) };
            gpu.hip
                .memcpy_htod(&state.mtp_token_chain.buf, seed_bytes)?;
        }
    }

    // ── 1. K serial discrete-token roundtrips ─────────────────────────────
    let mut proposal_graph_ran = false;
    if use_proposal_graph {
        if let Some(exec) = state.mtp_proposal_graph_exec.as_ref() {
            let stream = gpu.active_stream.as_ref().unwrap();
            gpu.hip.graph_launch(exec, stream)?;
            proposal_graph_ran = true;
        } else if state.mtp_proposal_graph_warmed {
            let capture_result: HipResult<(Graph, GraphExec, Vec<Vec<u8>>)> = (|| {
                mtp_graph_begin_capture(gpu)?;
                if let Err(e) = run_mtp_proposal_graph_body_q8(
                    gpu,
                    trunk_weights,
                    head,
                    state,
                    cur_pos,
                    max_n,
                    dim,
                    cvs,
                    proposal_graph_seq_cap,
                ) {
                    mtp_graph_abort_capture(gpu);
                    return Err(e);
                }
                mtp_graph_end_capture(gpu)
            })();
            match capture_result {
                Ok((graph, exec, blobs)) => {
                    state.mtp_proposal_graph = Some(graph);
                    state.mtp_proposal_graph_exec = Some(exec);
                    state.mtp_proposal_graph_blobs = blobs;
                    state.mtp_proposal_graph_seq_cap = proposal_graph_seq_cap;
                    let stream = gpu.active_stream.as_ref().unwrap();
                    gpu.hip
                        .graph_launch(state.mtp_proposal_graph_exec.as_ref().unwrap(), stream)?;
                    proposal_graph_ran = true;
                }
                Err(e) if proposal_graph_policy == MtpProposalGraphPolicy::On => return Err(e),
                Err(e) => {
                    mtp_graph_abort_capture(gpu);
                    state.mtp_proposal_graph_disabled = true;
                    eprintln!("[mtp-proposal-graph] disabled after capture failure: {}", e);
                }
            }
        } else {
            state.mtp_proposal_graph_warmed = true;
        }
    }

    if !proposal_graph_ran {
        for k in 0..max_n {
            let next_tok = if use_device_token_chain {
                0
            } else if k == 0 {
                last_committed
            } else {
                candidates[k - 1]
            };
            let next_token_embed = if use_device_token_chain {
                let token_slot = state.mtp_token_chain.sub_offset(k, 1);
                embed_device_token_into(
                    gpu,
                    trunk_weights,
                    &state.mtp_token_embed,
                    &token_slot,
                    dim,
                )?;
                Some(&state.mtp_token_embed)
            } else {
                None
            };

            // Forward: in compressed mode, mtp_head_forward_compressed runs
            // block_only + rmsnorm + small GEMV against lm_head_draft in one
            // call. In full-vocab mode we run block_only here and do the
            // rmsnorm + trunk-lm_head GEMV manually below.
            if use_full_vocab {
                if k == 0 {
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &state.prev_hidden,
                        next_token_embed,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                } else {
                    let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &prev_row,
                        next_token_embed,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                }
                // rmsnorm(t_mtp_out, shared_head_norm) → tmp, then trunk lm_head
                // GEMV → full_vocab_logits_view. weight_gemv dispatches the right
                // kernel (gemv_mq4g256_with_rotate / gemv_q8_0 / etc.) on the
                // trunk's output dtype.
                gpu.rmsnorm_f32(
                    &state.mtp_scratch.t_mtp_out,
                    &head.weights.shared_head_norm,
                    &state.mtp_scratch.tmp,
                    head.config.rms_norm_eps,
                )?;
                llama::weight_gemv(
                    gpu,
                    &trunk_weights.output,
                    &state.mtp_scratch.tmp,
                    &full_vocab_logits_view,
                )?;
            } else if use_device_token_chain {
                if k == 0 {
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &state.prev_hidden,
                        next_token_embed,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                } else {
                    let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &prev_row,
                        next_token_embed,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                }
                mtp_head::mtp_head_apply_lm_head_draft(gpu, head, &state.mtp_scratch)?;
            } else {
                if k == 0 {
                    mtp_head::mtp_head_forward_compressed(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &state.prev_hidden,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                } else {
                    let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                    mtp_head::mtp_head_forward_compressed(
                        gpu,
                        head,
                        &state.mtp_scratch,
                        &mut state.mtp_kv,
                        next_tok,
                        &prev_row,
                        cur_pos + k,
                        trunk_weights,
                    )?;
                }
            }

            // Pick the logits buffer to argmax over (mode-dependent).
            let logits_for_argmax: &GpuTensor = if use_full_vocab {
                &full_vocab_logits_view
            } else {
                state.mtp_scratch.logits_compressed.as_ref().unwrap()
            };
            let argmax_vocab = cvs; // = full vocab in full-vocab mode, = compressed vocab in compressed mode

            let draft_idx: usize;
            if use_sampling {
                // ── F5b: DFlash-convention sampled DRAFT (full-vocab only) ──
                //
                // Distribution-preserving spec-decode requires the draft to be
                // sampled from its OWN top_p nucleus AND that exact truncated,
                // renormalized distribution to be stored for the later residual
                // bonus draw. The prior fused `sample_top_p` (top_k=20+top_p) +
                // UNtruncated `softmax_prob_gather` violated this twice: it added
                // a top_k=20 cut the convention does not use, and it gathered
                // p_draft over the un-truncated softmax even though the sample
                // came from a nucleus — silently shipping a token attractor.
                //
                // Mirrors DFlash speculative.rs:3457-3504 EXACTLY, applied to the
                // single draft row of this step:
                //   GPU softmax+nucleus tau/Z → download probs/tau/z →
                //   apply_topp_trunc (host, same cut the target side uses) →
                //   host-RNG sample_categorical → store the TRUNCATED row.
                //
                // Compressed-vocab sampled IS supported (cvs-sampled): softmax
                // over the COMPRESSED draft row (argmax_vocab wide, ~15× cheaper
                // than full vocab — the cvs win), sample a compressed token,
                // remap to the full trunk id via vocab_map, and EMBED the
                // compressed nucleus into a full-vocab vector so the residual
                // bonus draw (sample_residual, on reject) subtracts it from the
                // full-vocab target correctly. In full-vocab mode argmax_vocab ==
                // vocab and the embed is a no-op (row already full-width). The
                // accept ratio uses only the SCALAR draft prob, so it is
                // mode-agnostic; losslessness holds because the residual covers
                // the full vocab (unmapped slots keep the full target mass).

                // Exact-AR penalties on the FULL-vocab draft row: the draft is
                // drawn from (and `draft_softmaxes`/`draft_probs` store) the
                // penalized q under history `emitted ‖ candidates[..k]` — the
                // same history the target verify row k uses. Compressed-vocab
                // and greedy draft rows stay unpenalized (q need not be
                // penalized for the verdict to be exact).
                if use_full_vocab && sampling.penalized() {
                    let t0 = MtpPenaltyTiming::start();
                    let mut uploads = 0;
                    if k == 0 {
                        mtp_penalty_begin_drafts(
                            gpu,
                            &mut state.penalty,
                            &state.mtp_sample_repeat_buf,
                        )?;
                        uploads = u32::from(!state.penalty.tokens().is_empty());
                    }
                    mtp_penalty_apply_row(
                        gpu,
                        &state.penalty,
                        &state.mtp_sample_repeat_buf,
                        logits_for_argmax,
                        k,
                        vocab,
                        &sampling,
                    )?;
                    let launches = u32::from(!state.penalty.row_range(k).is_empty());
                    state.penalty_timing.finish(t0, uploads, launches);
                }
                // GPU softmax(+nucleus) over the single draft row (full OR
                // compressed width). idiom copied from speculative.rs:3457-3492.
                let probs_gpu = gpu.alloc_tensor(&[argmax_vocab], DType::F32)?;
                let tau_gpu = gpu.alloc_tensor(&[1], DType::F32)?;
                let z_gpu = gpu.alloc_tensor(&[1], DType::F32)?;
                gpu.softmax_temp_topp_batched_into_f32(
                    logits_for_argmax,
                    &probs_gpu,
                    &tau_gpu,
                    &z_gpu,
                    argmax_vocab,
                    /* rows */ 1,
                    sampling.temp,
                    sampling.top_p,
                    sampling.top_k,
                    sampling.min_p,
                )?;
                let mut probs = gpu.download_f32(&probs_gpu)?;
                let tau = gpu.download_f32(&tau_gpu)?;
                let z = gpu.download_f32(&z_gpu)?;
                let _ = gpu.free_tensor(probs_gpu);
                let _ = gpu.free_tensor(tau_gpu);
                let _ = gpu.free_tensor(z_gpu);
                debug_assert_eq!(probs.len(), argmax_vocab);

                // SampledPMin reads the head's RAW top prob (max over the
                // UNtruncated softmax = the head's certainty at this step) for the
                // chain-truncation cutoff below — NOT the sampled token's prob.
                let top_prob = if use_sampled_pmin {
                    probs.iter().cloned().fold(0.0f32, f32::max)
                } else {
                    0.0
                };

                // Host truncation cut. The softmax kernel's tau/Z already
                // encode AR's law (top_k pool, min_p, nucleus over the pool
                // mass), so apply_topp_trunc applies the SAME cut the target
                // side gets.
                apply_topp_trunc(&mut probs, tau[0], z[0]);

                // Host-RNG categorical sample from the SAME truncated nucleus we
                // store — draft token and stored distribution agree (load-bearing
                // for the residual). top_p-only under the DFlash convention.
                let u = state.rng.next_uniform_f32();
                draft_idx = sample_categorical(&probs, u) as usize;
                assert!(
                    draft_idx < argmax_vocab,
                    "draft sample {draft_idx} out of draft vocab {argmax_vocab}"
                );

                draft_probs.push(probs[draft_idx]);

                // Remap compressed → full trunk id (no-op in full-vocab mode);
                // feeds the next MTP head forward via candidates[k-1].
                let token_id = match vocab_map_opt {
                    Some(vm) => vm[draft_idx],
                    None => draft_idx as u32,
                };
                candidates.push(token_id);
                if use_full_vocab && sampling.penalized() {
                    let t0 = MtpPenaltyTiming::start();
                    mtp_penalty_append_draft(
                        gpu,
                        &mut state.penalty,
                        &state.mtp_sample_repeat_buf,
                        token_id,
                    )?;
                    state.penalty_timing.finish(t0, 1, 0);
                }

                // Store the draft nucleus at FULL-vocab width for the residual.
                // Full-vocab: the row already is full-width. Compressed: scatter
                // each kept compressed prob to its full-vocab slot (vocab_map[j]);
                // the unmapped slots stay 0, so residual = (target − draft)₊
                // returns the full target mass there (lossless over full vocab).
                if use_full_vocab {
                    draft_softmaxes.push(probs);
                } else {
                    let vm = vocab_map_opt.expect("compressed mode carries a vocab_map");
                    let mut full = vec![0.0f32; vocab];
                    for (j, &pj) in probs.iter().enumerate() {
                        if pj > 0.0 {
                            full[vm[j] as usize] = pj;
                        }
                    }
                    draft_softmaxes.push(full);
                }
                // draft_logits_host is unused on this GPU-softmax path.
                let _ = &draft_logits_host;

                // SampledPMin: confidence-prune the K-chain. Keep this candidate;
                // skip the remaining draft steps if the head was unsure here. A
                // shorter speculative chain only lowers tau — it never changes the
                // committed (target) distribution, so this stays lossless. Mirrors
                // the PMin cutoff below, but on the sampled draft + raw top prob.
                if use_sampled_pmin && top_prob < p_min {
                    chain_truncated = true;
                    break;
                }
            } else if use_p_min {
                // Top-2 with log-softmax probs: 8 B idx + 8 B logp D2H per step.
                gpu.topk_logsumexp_batched_f32(
                    logits_for_argmax,
                    &state.mtp_topk_idx,
                    &state.mtp_topk_logp,
                    argmax_vocab,
                    /* k */ 2,
                    /* b */ 1,
                )?;
                let mut idx_host: [i32; 2] = [0, 0];
                let mut logp_host: [f32; 2] = [0.0, 0.0];
                {
                    let idx_bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, 8)
                    };
                    gpu.hip.memcpy_dtoh(idx_bytes, &state.mtp_topk_idx.buf)?;
                    let logp_bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(logp_host.as_mut_ptr() as *mut u8, 8)
                    };
                    gpu.hip.memcpy_dtoh(logp_bytes, &state.mtp_topk_logp.buf)?;
                }
                draft_idx = idx_host[0] as usize;
                assert!(
                    draft_idx < argmax_vocab,
                    "draft argmax {draft_idx} out of argmax_vocab {argmax_vocab}"
                );
                // Compressed: remap via vocab_map. Full-vocab: idx IS the token id.
                let token_id = match vocab_map_opt {
                    Some(vm) => vm[draft_idx],
                    None => draft_idx as u32,
                };
                candidates.push(token_id);

                // Check confidence AFTER pushing — keep this candidate, just
                // skip future steps if confidence is below threshold.
                if logp_host[0] < log_p_min {
                    chain_truncated = true;
                    break;
                }
            } else if use_device_token_chain {
                let vocab_map_gpu = if use_full_vocab {
                    None
                } else {
                    head.weights.lm_head_draft_vocab_map_gpu.as_ref()
                };
                gpu.argmax_token_chain_f32(
                    logits_for_argmax,
                    &argmax_view,
                    &state.mtp_token_chain,
                    vocab_map_gpu,
                    argmax_vocab,
                    k + 1,
                )?;
            } else {
                gpu.argmax_f32_batched(logits_for_argmax, &argmax_view, argmax_vocab, 1)?;
                let mut argmax_host: [i32; 1] = [0];
                {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, 4)
                    };
                    gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
                }
                draft_idx = argmax_host[0] as usize;
                assert!(
                    draft_idx < argmax_vocab,
                    "draft argmax {draft_idx} out of argmax_vocab {argmax_vocab}"
                );
                let token_id = match vocab_map_opt {
                    Some(vm) => vm[draft_idx],
                    None => draft_idx as u32,
                };
                candidates.push(token_id);
            }

            if k + 1 < max_n {
                gpu.memcpy_dtod_at_auto(
                    &state.mtp_t_outs.buf,
                    k * dim_bytes,
                    &state.mtp_scratch.t_mtp_out.buf,
                    0,
                    dim_bytes,
                )?;
            }
        }
    }
    let drafts_generated = if use_device_token_chain {
        debug_assert!(
            !use_sampling && !use_p_min,
            "device-token chain assumes an untruncated greedy chain"
        );
        let candidate_view = state.mtp_token_chain.sub_offset(1, max_n);
        let mut candidate_host: Vec<i32> = vec![0; max_n];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(candidate_host.as_mut_ptr() as *mut u8, max_n * 4)
            };
            gpu.hip.memcpy_dtoh(bytes, &candidate_view.buf)?;
        }
        candidates.extend(candidate_host.into_iter().map(|t| t as u32));
        max_n
    } else {
        candidates.len()
    };

    Ok(MtpDraftOutput {
        candidates,
        drafts_generated,
        chain_truncated,
        use_sampling,
        sampling,
        draft_probs,
        draft_softmaxes,
        use_device_token_chain,
        cur_pos,
        last_committed,
    })
}

/// Single-slot spec step: draft K candidates via [`mtp_draft_phase_inner`],
/// then verify + accept + rollback via [`mtp_shared_verify_accept_rollback_inner`].
/// This is the path used by `ModelSlot` / `MtpSpeculator` and the TUI.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial_inner(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &mut KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &mut Qwen35Scratch,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    k: usize,
    skip_proposal_graph: bool,
) -> HipResult<MtpSpecResult> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let overlap_trunk_snap = mtp_snapshot_overlap_enabled_from_env()
        && state.trunk_snap_stream.is_some()
        && state.trunk_snap_start_event.is_some();
    if overlap_trunk_snap {
        let snap_event = state.trunk_snap_start_event.as_ref().unwrap();
        let snap_stream = state.trunk_snap_stream.as_ref().unwrap();
        {
            let active_stream = gpu.active_stream.as_ref().unwrap();
            gpu.hip.event_record(snap_event, Some(active_stream))?;
            gpu.hip.stream_wait_event(snap_stream, snap_event)?;
        }
        state
            .trunk_snap
            .save_from_async_on(dn_state, gpu, snap_stream)?;
    }

    let draft = mtp_draft_phase_inner(
        gpu,
        weights,
        config,
        head,
        state,
        cur_pos,
        last_committed,
        k,
        skip_proposal_graph,
    )?;

    mtp_shared_verify_accept_rollback_inner(
        gpu,
        weights,
        config,
        kv_cache,
        dn_state,
        scratch,
        state,
        draft.cur_pos,
        draft.last_committed,
        eos_token_id,
        &draft.candidates,
        draft.drafts_generated,
        draft.chain_truncated,
        overlap_trunk_snap,
        draft.use_sampling,
        draft.sampling,
        &draft.draft_probs,
        &draft.draft_softmaxes,
        false,
        draft.use_device_token_chain,
    )
}

/// [`ModelSlot`]-based wrapper for [`spec_step_mtp_compressed_serial_inner`].
/// Uses proposal graphs (the single-slot path's default).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial_with_k(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    k: usize,
) -> HipResult<MtpSpecResult> {
    spec_step_mtp_compressed_serial_inner(
        gpu,
        &target.weights,
        &target.config,
        &mut target.kv_cache,
        &mut target.dn_state,
        &mut target.scratch,
        head,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        k,
        /* skip_proposal_graph */ false,
    )
}

/// External ngram-mod/PLD candidate verify with MTP-head KV fill.
///
/// MTP-head-bypass path. The daemon supplies a non-empty candidate window
/// `candidates` (already filtered for length `<= state.verify_capacity`).
/// Greedy requests (`state.sampling.temp <= 0`) verify with greedy trunk
/// acceptance; sampled requests verify with the point-mass q = δ(c_k) through
/// the same verify/penalty path as native MTP (`mtp_verify_accept` with
/// `is_external = true`): each candidate is accepted with probability
/// `p_target(c_k)`, a reject draws from the target with `c_k` removed, and a
/// full accept draws the bonus from the next verify row. The request's
/// penalties shape every verify row exactly like the AR sampler; the caller
/// must have run `state.penalty.begin_window(emitted)` for this window.
///
/// * Bypasses all head proposal / graph / device-chain work.
/// * Sets `drafts_generated = candidates.len()` and `chain_truncated=false`.
/// * Before shared verify, stages the pre-window `state.prev_hidden`
///   (h_{cur_pos-1}) as row 0 of `takeover_fill_hidden`.
/// * After shared verify/rollback, fills the MTP-head KV for exactly the
///   `advance` rows the trunk kept ([`mtp_takeover_fill_tokens`]) with
///   decode's `(tok_p, h_{p-1})` pairing, from the staged hidden and the
///   verify hidden rows. A rejected tail is never written, so native MTP can
///   draft on the next window as if those rows had been decoded one by one.
///
/// External candidates must be non-empty and `len <= state.verify_capacity`.
/// Panics if those bounds are violated so daemon bugs surface loudly.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial_with_takeover_candidates(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    candidates: &[u32],
) -> HipResult<MtpSpecResult> {
    assert!(
        mtp_external_candidates_within_capacity(candidates, state.verify_capacity),
        "spec_step_mtp_compressed_serial_with_takeover_candidates: candidates len {} out of bounds (must be 1..=verify_capacity {}, verify_capacity={}, max_n={})",
        candidates.len(),
        state.verify_capacity,
        state.verify_capacity,
        state.max_n
    );
    let verify_sampling = state.sampling;
    assert!(
        candidates
            .iter()
            .all(|&token| (token as usize) < target.config.vocab_size),
        "spec_step_mtp_compressed_serial_with_takeover_candidates: candidate token out of vocabulary"
    );
    let verify_end = cur_pos
        .checked_add(candidates.len() + 1)
        .expect("spec_step_mtp_compressed_serial_with_takeover_candidates: position overflow");
    assert!(
        verify_end <= state.mtp_kv.max_seq,
        "spec_step_mtp_compressed_serial_with_takeover_candidates: verify end {verify_end} exceeds MTP KV capacity {}",
        state.mtp_kv.max_seq
    );

    // No proposal work exists to overlap with the snapshot on this path.
    // Force the shared verifier to take a fresh synchronous snapshot.
    let overlap_trunk_snap = false;

    let drafts_generated = candidates.len();
    let chain_truncated = false;

    let dim = target.config.dim;
    let dim_bytes = dim * 4;

    // Shared verify overwrites `state.prev_hidden` with verify row advance-1.
    // Stage the pre-window partner of `last_committed` as fill row 0. That is
    // h_{cur_pos-1}, the same (tok_p, h_{p-1}) pairing the prompt fill and
    // decode use.
    debug_assert_eq!(
        state.prev_hidden_pos,
        cur_pos.checked_sub(1),
        "takeover fill: prev_hidden must hold h_{{cur_pos-1}}"
    );
    gpu.hip.memcpy_dtod_at(
        &state.takeover_fill_hidden.buf,
        0,
        &state.prev_hidden.buf,
        0,
        dim_bytes,
    )?;

    // Verify is the AR target: the request's penalties shape the verify rows
    // exactly like the AR sampler would. Sampled requests accept against the
    // deterministic draft's point mass (draft_prob 1.0, no draft softmax).
    let use_sampling = !verify_sampling.is_greedy();
    let draft_probs = vec![1.0f32; drafts_generated];
    let mut result = mtp_shared_verify_accept_rollback(
        gpu,
        target,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        candidates,
        drafts_generated,
        chain_truncated,
        overlap_trunk_snap,
        use_sampling,
        verify_sampling,
        &draft_probs,
        &[],
        true,
        false,
    )?;

    mtp_takeover_fill_head_kv(
        gpu,
        head,
        state,
        &target.weights,
        cur_pos,
        &mtp_takeover_fill_tokens(last_committed, candidates, result.advance),
    )?;

    // External windows are never truncated; ensure result fields cohere even
    // if the shared helper drifted.
    result.drafts_generated = drafts_generated;
    result.chain_truncated = false;
    Ok(result)
}

/// Write MTP-head KV slots `cur_pos..cur_pos + tokens.len()` from
/// `(tokens[i], takeover_fill_hidden[i])` after a takeover verify.
///
/// Row 0 of `takeover_fill_hidden` must already hold the pre-window hidden;
/// this copies verify hidden rows `0..n-1` into rows `1..n`. Uses the batched
/// KV-only head pass (same gate and kernels as the batched prompt fill) and
/// falls back to per-row decode forwards otherwise. Rows are independent:
/// each slot's K/V depends only on its own token and hidden.
fn mtp_takeover_fill_head_kv(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    trunk_weights: &Qwen35Weights,
    cur_pos: usize,
    tokens: &[u32],
) -> HipResult<()> {
    let n = tokens.len();
    let dim = head.config.n_embd;
    let dim_bytes = dim * 4;
    if n > 1 {
        gpu.hip.memcpy_dtod_at(
            &state.takeover_fill_hidden.buf,
            dim_bytes,
            &state.verify_hidden.buf,
            0,
            (n - 1) * dim_bytes,
        )?;
    }
    let use_batched = mtp_head::mtp_prompt_fill_uses_batched(
        state.mtp_kv.kv_mode,
        &[
            head.weights.eh_proj.gpu_dtype,
            head.weights.wq.gpu_dtype,
            head.weights.wk.gpu_dtype,
            head.weights.wv.gpu_dtype,
        ],
    );
    if use_batched {
        if state.takeover_fill_batched.is_none() {
            let rows = state.verify_capacity + 1;
            let scratch = Qwen35MtpHeadBatchedScratch::new(gpu, &head.config, rows)?;
            let widest_k = (2 * dim)
                .max(head.config.n_ff)
                .max(head.config.n_head * head.config.head_dim);
            match gpu.alloc_tensor(&[rows * widest_k], DType::F32) {
                Ok(rot) => state.takeover_fill_batched = Some((scratch, rot)),
                Err(error) => {
                    scratch.free_gpu(gpu);
                    return Err(error);
                }
            }
        }
        let (scratch, rot) = state
            .takeover_fill_batched
            .as_mut()
            .expect("takeover fill scratch allocated above");
        let positions: Vec<i32> = (cur_pos..cur_pos + n).map(|p| p as i32).collect();
        let hidden = state.takeover_fill_hidden.sub_offset(0, n * dim);
        mtp_head::mtp_head_forward_block_batched(
            gpu,
            head,
            scratch,
            &mut state.mtp_kv,
            tokens,
            &hidden,
            &positions,
            n,
            trunk_weights,
            Some(rot),
            true,
        )
    } else {
        for (i, &token) in tokens.iter().enumerate() {
            let hidden_row = state.takeover_fill_hidden.sub_offset(i * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                token,
                &hidden_row,
                None,
                cur_pos + i,
                trunk_weights,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_prefill_boundaries_chunk_at_prefill_max_batch() {
        let b = mtp_prefill_committed_boundaries(600, 0, qwen35::PREFILL_MAX_BATCH);
        assert_eq!(
            b,
            vec![256, 512, 600],
            "adaptive maybe_downshift must see every committed end ≤ PREFILL_MAX_BATCH apart"
        );
        // Empty prompt → no boundaries (no KV writes).
        assert!(mtp_prefill_committed_boundaries(0, 0, qwen35::PREFILL_MAX_BATCH).is_empty());
        // start_pos offset is absolute trunk position.
        assert_eq!(
            mtp_prefill_committed_boundaries(300, 10, 256),
            vec![266, 310]
        );
        // Single short chunk → one boundary at full length.
        assert_eq!(
            mtp_prefill_committed_boundaries(40, 0, qwen35::PREFILL_MAX_BATCH),
            vec![40]
        );
    }

    #[test]
    fn mtp_prompt_fill_scratch_rows_track_prompt_not_arch_ceiling() {
        assert_eq!(mtp_prompt_fill_scratch_rows(0, 384), None);
        assert_eq!(mtp_prompt_fill_scratch_rows(1, 384), Some(1));
        assert_eq!(mtp_prompt_fill_scratch_rows(383, 384), Some(383));
        assert_eq!(mtp_prompt_fill_scratch_rows(384, 384), Some(384));
        assert_eq!(mtp_prompt_fill_scratch_rows(385, 384), Some(384));
        assert_eq!(mtp_prompt_fill_scratch_rows(1, 256), Some(1));
        assert_eq!(mtp_prompt_fill_scratch_rows(512, 256), Some(256));
    }

    #[test]
    fn mtp_prefill_boundaries_never_exceed_chunk_stride() {
        let chunk = qwen35::PREFILL_MAX_BATCH;
        let bounds = mtp_prefill_committed_boundaries(10_000, 0, chunk);
        let mut prev = 0usize;
        for &pos in &bounds {
            assert!(pos > prev);
            assert!(
                pos - prev <= chunk,
                "boundary gap {} exceeds chunk_max {}",
                pos - prev,
                chunk
            );
            prev = pos;
        }
        assert_eq!(*bounds.last().unwrap(), 10_000);
    }

    #[test]
    fn trunk_spine_verify_tokens_are_seed_plus_candidates() {
        let tokens = build_trunk_spine_verify_tokens(42, &[7, 8, 9]);

        assert_eq!(tokens, vec![42, 7, 8, 9]);
    }

    #[test]
    fn trunk_spine_accepts_longest_prefix_then_bonus() {
        let accepted = greedy_trunk_spine_accept(&[11, 12, 13], &[11, 99, 55, 66], 2);

        assert_eq!(accepted.committed, vec![11, 99]);
        assert_eq!(accepted.accept_count, 1);
        assert!(!accepted.hit_eos);
    }

    #[test]
    fn trunk_spine_stops_without_bonus_when_accepted_candidate_is_eos() {
        let accepted = greedy_trunk_spine_accept(&[11, 2, 13], &[11, 2, 55, 66], 2);

        assert_eq!(accepted.committed, vec![11, 2]);
        assert_eq!(accepted.accept_count, 2);
        assert!(accepted.hit_eos);
    }

    #[test]
    fn gpu_greedy_accept_result_matches_host_accept_semantics() {
        let mismatch_bonus = assemble_greedy_accept_from_gpu_result(&[11, 12, 13], 1, 99, 2);
        assert_eq!(mismatch_bonus.committed, vec![11, 99]);
        assert_eq!(mismatch_bonus.accept_count, 1);
        assert!(!mismatch_bonus.hit_eos);

        let accepted_eos = assemble_greedy_accept_from_gpu_result(&[11, 2, 13], 2, -1, 2);
        assert_eq!(accepted_eos.committed, vec![11, 2]);
        assert_eq!(accepted_eos.accept_count, 2);
        assert!(accepted_eos.hit_eos);

        let bonus_eos = assemble_greedy_accept_from_gpu_result(&[11, 12, 13], 3, 2, 2);
        assert_eq!(bonus_eos.committed, vec![11, 12, 13, 2]);
        assert_eq!(bonus_eos.accept_count, 3);
        assert!(bonus_eos.hit_eos);
    }

    #[test]
    fn device_token_chain_is_greedy_only_and_embedding_gated() {
        assert!(mtp_device_token_chain_eligible_for(
            llama::EmbeddingFormat::HFQ4G256,
            false,
            false
        ));
        assert!(mtp_device_token_chain_eligible_for(
            llama::EmbeddingFormat::Q8_0,
            false,
            false
        ));
        assert!(!mtp_device_token_chain_eligible_for(
            llama::EmbeddingFormat::HFQ4G128,
            false,
            false
        ));
        assert!(!mtp_device_token_chain_eligible_for(
            llama::EmbeddingFormat::HFQ4G256,
            true,
            false
        ));
        assert!(!mtp_device_token_chain_eligible_for(
            llama::EmbeddingFormat::HFQ4G256,
            false,
            true
        ));
    }

    #[test]
    fn proposal_graph_policy_is_opt_in_and_q8_device_chain_only() {
        assert!(mtp_q8_verify_wmma_enabled_from_env_value(None));
        assert!(!mtp_q8_verify_wmma_enabled_from_env_value(Some("0")));
        assert!(!mtp_q8_verify_wmma_enabled_from_env_value(Some("off")));
        assert!(mtp_q8_verify_wmma_enabled_from_env_value(Some("1")));
        assert!(mtp_q8_verify_wmma_enabled_from_env_value(Some("ON")));

        assert_eq!(
            mtp_proposal_graph_policy_from_env_value(None),
            MtpProposalGraphPolicy::Off
        );
        assert_eq!(
            mtp_proposal_graph_policy_from_env_value(Some("0")),
            MtpProposalGraphPolicy::Off
        );
        assert_eq!(
            mtp_proposal_graph_policy_from_env_value(Some("off")),
            MtpProposalGraphPolicy::Off
        );
        assert_eq!(
            mtp_proposal_graph_policy_from_env_value(Some("1")),
            MtpProposalGraphPolicy::On
        );
        assert_eq!(
            mtp_proposal_graph_policy_from_env_value(Some("ON")),
            MtpProposalGraphPolicy::On
        );

        assert!(mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::Auto,
            true,
            false,
            crate::mtp_head::MtpKvMode::Q8,
        ));
        assert!(!mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::Auto,
            false,
            false,
            crate::mtp_head::MtpKvMode::Q8,
        ));
        assert!(!mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::Auto,
            true,
            true,
            crate::mtp_head::MtpKvMode::Q8,
        ));
        assert!(!mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::Auto,
            true,
            false,
            crate::mtp_head::MtpKvMode::Asym3,
        ));
        assert!(mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::On,
            true,
            false,
            crate::mtp_head::MtpKvMode::Q8,
        ));
        assert!(!mtp_proposal_graph_eligible_for(
            MtpProposalGraphPolicy::Off,
            true,
            false,
            crate::mtp_head::MtpKvMode::Q8,
        ));

        assert_eq!(mtp_proposal_graph_seq_cap(27, 5, 4096), 256);
        assert_eq!(mtp_proposal_graph_seq_cap(260, 5, 4096), 512);
        assert_eq!(mtp_proposal_graph_seq_cap(3000, 5, 4096), 4096);
    }

    #[test]
    fn external_candidates_within_capacity_bounds() {
        // Pure bound check — no GPU.
        assert!(!mtp_external_candidates_within_capacity(&[], 4));
        assert!(mtp_external_candidates_within_capacity(&[1], 4));
        assert!(mtp_external_candidates_within_capacity(&[1, 2, 3, 4], 4));
        assert!(!mtp_external_candidates_within_capacity(
            &[1, 2, 3, 4, 5],
            4
        ));
        assert!(!mtp_external_candidates_within_capacity(&[], 0));
        // verify_capacity vs max_n: external window may be larger than max_n.
        // e.g., max_n=2, verify_capacity=5 allows 5 candidates.
        assert!(mtp_external_candidates_within_capacity(&[1, 2, 3, 4, 5], 5));
        assert!(!mtp_external_candidates_within_capacity(
            &[1, 2, 3, 4, 5, 6],
            5
        ));
    }

    #[test]
    fn assemble_verify_tokens_is_seed_plus_candidates() {
        assert_eq!(mtp_assemble_verify_tokens(99, &[]), vec![99]);
        assert_eq!(mtp_assemble_verify_tokens(7, &[1, 2, 3]), vec![7, 1, 2, 3]);
        // matches build_trunk_spine_verify_tokens contract
        assert_eq!(
            mtp_assemble_verify_tokens(42, &[7, 8, 9]),
            build_trunk_spine_verify_tokens(42, &[7, 8, 9])
        );
    }

    #[test]
    fn takeover_fill_rows_are_the_trunk_kept_prefix() {
        let cands = [11, 12, 13, 14];
        // Zero accept: bonus only, the seed row is the only trunk-kept row.
        assert_eq!(mtp_takeover_fill_tokens(7, &cands, 1), vec![7]);
        // Partial accept (2 drafts + bonus): rejected tail is never filled.
        assert_eq!(mtp_takeover_fill_tokens(7, &cands, 3), vec![7, 11, 12]);
        // Full accept: seed + every draft; the bonus is pending, not filled.
        assert_eq!(
            mtp_takeover_fill_tokens(7, &cands, 5),
            vec![7, 11, 12, 13, 14]
        );
    }
}
