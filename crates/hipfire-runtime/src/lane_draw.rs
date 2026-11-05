// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-wide device draws (`HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE`): every sampled
//! lane of a step is drawn by ONE launch of the lane-wide verify kernel
//! (`rdna_compute::lane_leviathan`), with one async H2D of the step's inputs
//! and one D2H of `{tokens, accepted, rng}` per lane.
//!
//! A job is one lane's `rows` consecutive resident logits rows: `rows - 1`
//! drafts verified with Leviathan acceptance (`min(1, p/q)`, residual
//! `(p - q)+` on the first rejection, the bonus from `p` when every draft was
//! accepted), or `rows == 1`, a plain AR draw from `p`. `p` is the shared
//! device sampler's law over the row (blocked tokens, the recency-weighted
//! repeat / presence / frequency penalty over the trailing
//! [`LANE_DRAW_REPEAT_CAP`]-capped window of `history ++ drafts[..j]`, top-k
//! gather, temperature softmax, cap, min-p, nucleus); `q` is each draft's
//! distribution as it was drawn (a deterministic draft, an argmax or a copy
//! candidate, is the point mass). Every emitted token is distributed as `p`
//! whatever `q` is. A rejected draft is never its own replacement, so the
//! emitted tokens end at the first one that differs from its draft (or an
//! accepted EOS, or the bonus).
//!
//! Uniforms come from the lane's xorshift32 state (the AR sampler's stream);
//! the advanced state is returned. The kernel never mutates the logits.

use crate::sampler::SamplerConfig;
use hip_bridge::{HipError, HipResult};
use rdna_compute::lane_leviathan::{
    LaneLeviathanLayout, LLV_H_CAP, LLV_LANE_WORDS, LLV_NO_EOS, LLV_OUT_WORDS, LLV_P_WORDS,
    LLV_Q_MAX, LLV_Q_WORDS, LLV_ROWS_CAP, LLV_ROW_WORDS, LL_BLOCKED, LL_DRAFTS, LL_EOS, LL_FREQ,
    LL_MIN_P, LL_NBLOCKED, LL_NROWS, LL_OUT, LL_PRES, LL_Q, LL_REP, LL_RNG, LL_ROW0, LL_TEMP,
    LL_TOP_K, LL_TOP_P, LL_VOCAB, LO_ACCEPTED, LO_DONE, LO_NTOK, LO_RNG, LO_TICKET, LO_TOKENS,
    LR_J, LR_LANE, LR_PEN_LEN, LR_PEN_OFF, LR_PTR_HI, LR_PTR_LO,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

/// Penalty-window capacity of a device draw: the singleton sampler's
/// `repeat_buf` holds 64 tokens (`llama::ForwardScratch`), so a lane's device
/// draw windows its history as the serial device sampler does.
pub const LANE_DRAW_REPEAT_CAP: usize = 64;

/// Most rows of one job (`drafts + 1`).
pub const LANE_DRAW_MAX_ROWS: usize = LLV_ROWS_CAP;

/// Most `(token, prob)` pairs of one draft's `q`.
pub const LANE_DRAW_MAX_Q: usize = LLV_Q_MAX;

/// One lane's draw.
pub struct LaneDrawJob<'a> {
    /// `[rows * vocab]` f32: the lane's consecutive target rows (a view into
    /// the step's logits). Read only.
    pub logits: &'a GpuTensor,
    /// `1..=LANE_DRAW_MAX_ROWS`; `1` is a plain AR draw.
    pub rows: usize,
    pub vocab: usize,
    /// `rows - 1` drafted tokens.
    pub drafts: &'a [u32],
    /// Per draft, the distribution it was drawn from as `(token, prob)`
    /// pairs (at most [`LANE_DRAW_MAX_Q`]); an empty entry, or an empty
    /// slice for the whole window, is the point mass on the draft.
    pub q: &'a [Vec<(u32, f32)>],
    /// Committed tokens before row 0 (penalty window source).
    pub history: &'a [u32],
    pub cfg: &'a SamplerConfig,
    /// The lane's xorshift32 state (never 0).
    pub rng_state: u32,
    pub eos: Option<u32>,
}

/// A lane's draw: every accepted draft, then the token that ended the window
/// (replacement or bonus; absent after an accepted EOS draft), the accepted
/// draft count (an accepted EOS draft included) and the advanced RNG state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneDraw {
    pub tokens: Vec<u32>,
    pub accepted: usize,
    pub advanced_rng: u32,
}

/// Device region, p-table scratch and pinned stage of [`lane_draws`], grown on
/// demand and reused across calls.
pub struct LaneDrawArena {
    region: Option<GpuTensor>,
    ptab: Option<GpuTensor>,
    host: *mut c_void,
    words: usize,
    ptab_rows: usize,
    stage: Vec<u32>,
}

// SAFETY: `host` is the arena's exclusively owned pinned allocation, touched
// only through `&mut self` while no copy is in flight (every call ends with a
// stream sync).
unsafe impl Send for LaneDrawArena {}

impl Default for LaneDrawArena {
    fn default() -> Self {
        Self::new()
    }
}

impl LaneDrawArena {
    pub fn new() -> Self {
        Self {
            region: None,
            ptab: None,
            host: std::ptr::null_mut(),
            words: 0,
            ptab_rows: 0,
            stage: Vec::new(),
        }
    }

    /// The p tables of the last call's rows (`LLV_P_WORDS` floats per row, in
    /// job-row order); oracle access.
    pub fn ptab(&self) -> Option<&GpuTensor> {
        self.ptab.as_ref()
    }

    /// Free the device and pinned allocations (no work may be in flight).
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        if let Some(t) = self.region.take() {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.ptab.take() {
            let _ = gpu.free_tensor(t);
        }
        if !self.host.is_null() {
            let _ = gpu.hip.host_free(self.host);
        }
    }

    fn ensure(&mut self, gpu: &mut Gpu, words: usize, rows: usize) -> HipResult<()> {
        if self.words < words {
            let cap = words.next_power_of_two().max(4096);
            let dev = gpu.zeros(&[cap], DType::F32)?;
            let host = match gpu.hip.host_malloc(cap * 4, 0) {
                Ok(p) => p,
                Err(e) => {
                    let _ = gpu.free_tensor(dev);
                    return Err(e);
                }
            };
            if let Some(old) = self.region.replace(dev) {
                let _ = gpu.free_tensor(old);
            }
            if !self.host.is_null() {
                let _ = gpu.hip.host_free(self.host);
            }
            self.host = host;
            self.words = cap;
        }
        if self.ptab_rows < rows {
            let cap = rows.next_power_of_two().max(64);
            let t = gpu.zeros(&[cap * LLV_P_WORDS], DType::F32)?;
            if let Some(old) = self.ptab.replace(t) {
                let _ = gpu.free_tensor(old);
            }
            self.ptab_rows = cap;
        }
        Ok(())
    }
}

/// Whether [`lane_draws`] runs on `gpu`.
pub fn lane_draws_supported(gpu: &Gpu) -> bool {
    gpu.lane_leviathan_supported()
}

fn penalty_active(cfg: &SamplerConfig) -> bool {
    (cfg.repeat_penalty > 1.0 || cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0)
        && cfg.repeat_window > 0
}

fn malformed(i: usize, why: String) -> HipError {
    HipError::new(0, &format!("lane_draws: lane {i}: {why}"))
}

fn check_job(i: usize, job: &LaneDrawJob<'_>) -> HipResult<()> {
    if job.rows == 0 || job.rows > LANE_DRAW_MAX_ROWS {
        return Err(malformed(
            i,
            format!("{} rows (1..={LANE_DRAW_MAX_ROWS})", job.rows),
        ));
    }
    if job.drafts.len() + 1 != job.rows {
        return Err(malformed(
            i,
            format!("{} drafts for {} rows", job.drafts.len(), job.rows),
        ));
    }
    if job.vocab == 0 || job.vocab > i32::MAX as usize || job.logits.numel() < job.rows * job.vocab
    {
        return Err(malformed(
            i,
            format!(
                "vocab {} over a {}-float view",
                job.vocab,
                job.logits.numel()
            ),
        ));
    }
    if !job.q.is_empty() && job.q.len() != job.drafts.len() {
        return Err(malformed(
            i,
            format!(
                "{} draft distributions for {} drafts",
                job.q.len(),
                job.drafts.len()
            ),
        ));
    }
    for (j, q) in job.q.iter().enumerate() {
        if q.len() > LANE_DRAW_MAX_Q || q.iter().any(|&(_, p)| !p.is_finite() || p < 0.0) {
            return Err(malformed(
                i,
                format!(
                    "draft {j}: q of {} entries (at most {LANE_DRAW_MAX_Q}, finite >= 0)",
                    q.len()
                ),
            ));
        }
    }
    if job.rng_state == 0 {
        return Err(malformed(i, "the stuck xorshift32 state 0".to_string()));
    }
    if job.eos == Some(LLV_NO_EOS) {
        return Err(malformed(i, format!("EOS id {LLV_NO_EOS} is reserved")));
    }
    let scope = if penalty_active(job.cfg) {
        job.cfg.repeat_window.min(LANE_DRAW_REPEAT_CAP)
    } else {
        0
    };
    if scope + job.cfg.blocked_tokens.len() > LLV_H_CAP {
        return Err(malformed(
            i,
            format!(
                "{} penalty + {} blocked tokens per row (at most {LLV_H_CAP})",
                scope,
                job.cfg.blocked_tokens.len()
            ),
        ));
    }
    Ok(())
}

/// Pack `[out | lanes | rows | inputs]` into `w` (cleared first).
fn pack(jobs: &[LaneDrawJob<'_>], w: &mut Vec<u32>) -> LaneLeviathanLayout {
    let rows: usize = jobs.iter().map(|j| j.rows).sum();
    let layout = LaneLeviathanLayout {
        lanes: jobs.len(),
        rows,
    };
    w.clear();
    w.resize(layout.inputs_off(), 0);
    let mut grow = 0usize;
    let mut tail: Vec<u32> = Vec::new();
    for (li, job) in jobs.iter().enumerate() {
        let cfg = job.cfg;
        let drafts_off = w.len();
        w.extend_from_slice(job.drafts);
        let q_off = w.len();
        for (j, &d) in job.drafts.iter().enumerate() {
            let mut rec = [0u32; LLV_Q_WORDS];
            match job.q.get(j).filter(|q| !q.is_empty()) {
                Some(q) => {
                    rec[0] = q.len() as u32;
                    for (k, &(t, p)) in q.iter().enumerate() {
                        rec[1 + k] = t;
                        rec[1 + LLV_Q_MAX + k] = p.to_bits();
                    }
                }
                None => {
                    rec[0] = 1;
                    rec[1] = d;
                    rec[1 + LLV_Q_MAX] = 1.0f32.to_bits();
                }
            }
            w.extend_from_slice(&rec);
        }
        let blocked_off = w.len();
        w.extend_from_slice(&cfg.blocked_tokens);
        // Row j's scope is the trailing `window` of history ++ drafts[..j]: one
        // contiguous tail whose sub-slices are the rows' scopes.
        let window = if penalty_active(cfg) {
            cfg.repeat_window.min(LANE_DRAW_REPEAT_CAP)
        } else {
            0
        };
        let tail_start = job.history.len().saturating_sub(window);
        tail.clear();
        if window > 0 {
            tail.extend_from_slice(&job.history[tail_start..]);
            tail.extend_from_slice(job.drafts);
        }
        let tail_off = w.len();
        w.extend_from_slice(&tail);
        let lr = layout.lanes_off() + li * LLV_LANE_WORDS;
        w[lr + LL_ROW0] = grow as u32;
        w[lr + LL_NROWS] = job.rows as u32;
        w[lr + LL_RNG] = job.rng_state;
        w[lr + LL_EOS] = job.eos.unwrap_or(LLV_NO_EOS);
        w[lr + LL_TEMP] = cfg.temperature.to_bits();
        w[lr + LL_TOP_P] = cfg.top_p.to_bits();
        w[lr + LL_TOP_K] = cfg.top_k.map_or(20, |k| k.min(i32::MAX as u32) as i32) as u32;
        w[lr + LL_MIN_P] = cfg.min_p.unwrap_or(0.0).to_bits();
        w[lr + LL_REP] = cfg.repeat_penalty.to_bits();
        w[lr + LL_PRES] = cfg.presence_penalty.to_bits();
        w[lr + LL_FREQ] = cfg.frequency_penalty.to_bits();
        w[lr + LL_VOCAB] = job.vocab as u32;
        w[lr + LL_DRAFTS] = drafts_off as u32;
        w[lr + LL_Q] = q_off as u32;
        w[lr + LL_BLOCKED] = blocked_off as u32;
        w[lr + LL_NBLOCKED] = cfg.blocked_tokens.len() as u32;
        w[lr + LL_OUT] = layout.out_off(li) as u32;
        let base = job.logits.buf.as_ptr() as u64;
        for j in 0..job.rows {
            let (pen_off, pen_len) = if window > 0 {
                let end = job.history.len() + j;
                let start = end.saturating_sub(window);
                (tail_off + (start - tail_start), end - start)
            } else {
                (tail_off, 0)
            };
            let rr = layout.rows_off() + grow * LLV_ROW_WORDS;
            let ptr = base + (j * job.vocab * 4) as u64;
            w[rr + LR_PTR_LO] = ptr as u32;
            w[rr + LR_PTR_HI] = (ptr >> 32) as u32;
            w[rr + LR_LANE] = li as u32;
            w[rr + LR_J] = j as u32;
            w[rr + LR_PEN_OFF] = pen_off as u32;
            w[rr + LR_PEN_LEN] = pen_len as u32;
            grow += 1;
        }
    }
    layout
}

/// Draw every job in ONE launch: one async H2D of the packed inputs, the
/// lane-wide verify kernel over all rows, one D2H of every lane's
/// `{tokens, accepted, rng}`, one stream sync. Draws are in job order.
pub fn lane_draws(
    gpu: &mut Gpu,
    arena: &mut LaneDrawArena,
    jobs: &[LaneDrawJob<'_>],
) -> HipResult<Vec<LaneDraw>> {
    for (i, job) in jobs.iter().enumerate() {
        check_job(i, job)?;
    }
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if !lane_draws_supported(gpu) {
        return Err(HipError::new(
            0,
            &format!(
                "lane_draws: the lane-wide verify kernel does not run on {}",
                gpu.arch
            ),
        ));
    }
    gpu.bind_thread()?;
    let mut stage = std::mem::take(&mut arena.stage);
    let layout = pack(jobs, &mut stage);
    let result = (|| {
        arena.ensure(gpu, stage.len(), layout.rows)?;
        // SAFETY: `ensure` sized the pinned stage to >= `stage.len()` u32s; the
        // previous call's final sync left no copy reading or writing it.
        let host = unsafe { std::slice::from_raw_parts_mut(arena.host as *mut u32, stage.len()) };
        host.copy_from_slice(&stage);
        let region = arena.region.as_ref().expect("ensure allocated the region");
        let ptab = arena.ptab.as_ref().expect("ensure allocated the p tables");
        // SAFETY: `stage.len()` initialized u32s of the live pinned stage.
        let bytes = unsafe { std::slice::from_raw_parts(arena.host as *const u8, stage.len() * 4) };
        match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.memcpy_htod_async(&region.buf, bytes, stream)?,
            None => gpu.hip.memcpy_htod_async_default(&region.buf, bytes)?,
        }
        gpu.lane_leviathan_verify_enqueue(region, layout, ptab)?;
        let out_bytes = layout.out_words() * 4;
        // SAFETY: the out prefix fits the stage (`out_words <= stage.len()`).
        let dst = unsafe { std::slice::from_raw_parts_mut(arena.host as *mut u8, out_bytes) };
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.hip.memcpy_dtoh_async(dst, &region.buf, Some(stream))?;
                gpu.hip.stream_synchronize(stream)?;
            }
            None => gpu.hip.memcpy_dtoh(dst, &region.buf)?,
        }
        gpu.take_device_wait_error()?;
        // SAFETY: the D2H just refreshed the out prefix.
        let out =
            unsafe { std::slice::from_raw_parts(arena.host as *const u32, layout.out_words()) };
        let mut draws = Vec::with_capacity(jobs.len());
        for (i, job) in jobs.iter().enumerate() {
            let o = &out[layout.out_off(i)..layout.out_off(i) + LLV_OUT_WORDS];
            let ntok = o[LO_NTOK] as usize;
            let accepted = o[LO_ACCEPTED] as usize;
            let ok = o[LO_DONE] == 1
                && o[LO_TICKET] as usize == job.rows
                && (1..=job.rows).contains(&ntok)
                && accepted <= job.drafts.len()
                && accepted <= ntok
                && o[LO_TOKENS..LO_TOKENS + accepted] == job.drafts[..accepted]
                && o[LO_RNG] != 0;
            if !ok {
                return Err(malformed(
                    i,
                    format!(
                        "corrupt result record {:?}",
                        &o[..LO_TOKENS + ntok.min(job.rows)]
                    ),
                ));
            }
            draws.push(LaneDraw {
                tokens: o[LO_TOKENS..LO_TOKENS + ntok].to_vec(),
                accepted,
                advanced_rng: o[LO_RNG],
            });
        }
        Ok(draws)
    })();
    arena.stage = stage;
    result
}

// ───────────────────────── host model of the kernel law ─────────────────────────
//
// The same arithmetic as `lane_leviathan.hip` on the host (f32 throughout; only
// `exp` may differ from the device `expf` by an ulp). The GPU oracle checks the
// device against it and CPU tests use it as the device stand-in.

/// One row's target distribution: a single token (`point`), or the nucleus
/// `(token, unnormalized prob)` in rank order with its mass.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceP {
    pub point: Option<u32>,
    pub entries: Vec<(u32, f32)>,
    pub mass: f32,
}

impl ReferenceP {
    /// `p(token)` (normalized).
    pub fn prob(&self, token: u32) -> f32 {
        match self.point {
            Some(t) => f32::from(u8::from(t == token)),
            None => self
                .entries
                .iter()
                .find(|e| e.0 == token)
                .map_or(0.0, |e| e.1 / self.mass),
        }
    }
}

fn ord_key(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The penalty scope of row `j`: the trailing capped window of
/// `history ++ drafts[..j]` (empty with no active penalty).
pub fn reference_scope(history: &[u32], drafts: &[u32], j: usize, cfg: &SamplerConfig) -> Vec<u32> {
    if !penalty_active(cfg) {
        return Vec::new();
    }
    let window = cfg.repeat_window.min(LANE_DRAW_REPEAT_CAP);
    let mut all = history.to_vec();
    all.extend_from_slice(&drafts[..j]);
    all[all.len().saturating_sub(window)..].to_vec()
}

/// The p of one logits `row` with penalty `scope` under `cfg` (the shared
/// device sampler's law).
pub fn reference_p(row: &[f32], scope: &[u32], cfg: &SamplerConfig) -> ReferenceP {
    let mut v = row.to_vec();
    for &t in &cfg.blocked_tokens {
        if (t as usize) < v.len() {
            v[t as usize] = f32::NEG_INFINITY;
        }
    }
    if penalty_active(cfg) && !scope.is_empty() {
        let wl = scope.len() as f32;
        for (i, &tok) in scope.iter().enumerate() {
            if tok as usize >= v.len() || scope[i + 1..].contains(&tok) {
                continue;
            }
            let count = scope.iter().filter(|&&t| t == tok).count();
            let x = &mut v[tok as usize];
            if cfg.repeat_penalty > 1.0 {
                let recency = (i as f32 + 1.0) / wl;
                let eff = cfg.repeat_penalty.powf(count as f32 * recency).min(1.5);
                *x = if *x > 0.0 { *x / eff } else { *x * eff };
            }
            if cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0 {
                *x -= cfg.frequency_penalty * count as f32 + cfg.presence_penalty;
            }
        }
    }
    let top_k_req = cfg.top_k.map_or(20, |k| k.min(i32::MAX as u32) as i32);
    let g_width = if top_k_req <= 20 { 20 } else { 64 };
    let mut cand: Vec<(u32, f32)> = v
        .iter()
        .enumerate()
        .filter(|(_, &x)| x > -1e29)
        .map(|(i, &x)| (i as u32, x))
        .collect();
    cand.sort_by(|a, b| ord_key(b.1).cmp(&ord_key(a.1)).then(a.0.cmp(&b.0)));
    cand.truncate(g_width);
    let g = cand.len();
    let point = |t: u32| ReferenceP {
        point: Some(t),
        entries: Vec::new(),
        mass: 0.0,
    };
    if g == 0 {
        return point(0);
    }
    if cfg.temperature <= 1e-6 {
        return point(cand[0].0);
    }
    let inv = 1.0 / cfg.temperature;
    let maxv = cand[0].1;
    let probs: Vec<f32> = cand
        .iter()
        .map(|&(_, x)| ((x - maxv) * inv).exp())
        .collect();
    let get = |i: usize| if i < g { probs[i] } else { 0.0 };
    let mut cap = g_width;
    if top_k_req > 0 && (top_k_req as usize) < g_width {
        cap = top_k_req as usize;
    }
    let min_p = cfg.min_p.unwrap_or(0.0);
    if min_p > 0.0 {
        let floor = min_p * get(0);
        if let Some(c) = (1..cap).find(|&i| get(i) < floor) {
            cap = c;
        }
    }
    let cap = cap.max(1);
    let mut sum = 0.0f32;
    for i in 0..cap {
        sum += get(i);
    }
    if sum <= 0.0 {
        return point(cand[0].0);
    }
    let thresh = cfg.top_p * sum;
    let mut ts = 0.0f32;
    let mut k = cap;
    for i in 0..cap {
        ts += get(i);
        if ts >= thresh {
            k = i + 1;
            break;
        }
    }
    let k = k.min(g);
    ReferenceP {
        point: None,
        entries: (0..k).map(|i| (cand[i].0, probs[i])).collect(),
        mass: ts,
    }
}

fn xorshift32(mut s: u32) -> u32 {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    s
}

fn q_prob(q: &[(u32, f32)], d: u32, t: u32) -> f32 {
    if q.is_empty() {
        return f32::from(u8::from(t == d));
    }
    q.iter().find(|e| e.0 == t).map_or(0.0, |e| e.1)
}

fn draw_p(p: &ReferenceP, u: f32) -> u32 {
    let r = u * p.mass;
    let mut cum = 0.0f32;
    for &(t, x) in &p.entries {
        cum += x;
        if cum >= r {
            return t;
        }
    }
    p.entries.last().map_or(0, |e| e.0)
}

fn draw_residual(p: &ReferenceP, q: &[(u32, f32)], d: u32, u: f32) -> u32 {
    let res = |&(t, x): &(u32, f32)| x / p.mass - q_prob(q, d, t);
    let mut mass = 0.0f32;
    for e in p.entries.iter().filter(|e| e.0 != d) {
        let r = res(e);
        if r > 0.0 {
            mass += r;
        }
    }
    let mut last = d;
    if mass > 0.0 {
        let target = u * mass;
        let mut acc = 0.0f32;
        for e in p.entries.iter().filter(|e| e.0 != d) {
            let r = res(e);
            if !(r > 0.0) {
                continue;
            }
            acc += r;
            if target < acc {
                return e.0;
            }
            last = e.0;
        }
        return last;
    }
    let rest: f32 = p.entries.iter().filter(|e| e.0 != d).map(|e| e.1).sum();
    let target = u * rest;
    let mut cum = 0.0f32;
    for e in p.entries.iter().filter(|e| e.0 != d) {
        cum += e.1;
        last = e.0;
        if target < cum {
            return last;
        }
    }
    last
}

/// The lane verdict over per-row p tables `ps` (the kernel's finalize).
pub fn reference_verdict(
    ps: &[ReferenceP],
    drafts: &[u32],
    q: &[Vec<(u32, f32)>],
    rng_state: u32,
    eos: Option<u32>,
) -> LaneDraw {
    let mut rng = rng_state;
    let mut tokens = Vec::new();
    let mut accepted = 0usize;
    for (j, p) in ps.iter().enumerate().take(drafts.len() + 1) {
        if j < drafts.len() {
            let d = drafts[j];
            if let Some(t) = p.point {
                tokens.push(t);
                if t == d {
                    accepted += 1;
                    if eos == Some(d) {
                        break;
                    }
                    continue;
                }
                break;
            }
            let qd: &[(u32, f32)] = q.get(j).map_or(&[], Vec::as_slice);
            let qx = q_prob(qd, d, d);
            let px = p.prob(d);
            rng = xorshift32(rng);
            let u = (rng >> 8) as f32 * (1.0 / 16_777_216.0);
            if qx > 0.0 && u * qx < px {
                tokens.push(d);
                accepted += 1;
                if eos == Some(d) {
                    break;
                }
                continue;
            }
            rng = xorshift32(rng);
            let u2 = (rng >> 8) as f32 * (1.0 / 16_777_216.0);
            tokens.push(draw_residual(p, qd, d, u2));
            break;
        }
        match p.point {
            Some(t) => tokens.push(t),
            None => {
                rng = xorshift32(rng);
                tokens.push(draw_p(p, rng as f32 / 4294967295.0));
            }
        }
        break;
    }
    LaneDraw {
        tokens,
        accepted,
        advanced_rng: rng,
    }
}

/// One lane's draw from host `rows` (`drafts.len() + 1` rows of `vocab`):
/// [`reference_p`] per row, then [`reference_verdict`].
#[allow(clippy::too_many_arguments)]
pub fn reference_draw(
    rows: &[f32],
    vocab: usize,
    drafts: &[u32],
    q: &[Vec<(u32, f32)>],
    history: &[u32],
    cfg: &SamplerConfig,
    rng_state: u32,
    eos: Option<u32>,
) -> LaneDraw {
    let ps: Vec<ReferenceP> = (0..=drafts.len())
        .map(|j| {
            reference_p(
                &rows[j * vocab..(j + 1) * vocab],
                &reference_scope(history, drafts, j, cfg),
                cfg,
            )
        })
        .collect();
    reference_verdict(&ps, drafts, q, rng_state, eos)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null-backed `n`-float tensor: metadata only, never handed to HIP.
    fn meta_tensor(n: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut(), n * 4) },
            shape: vec![n],
            dtype: DType::F32,
        }
    }

    fn cfg_pen(window: usize) -> SamplerConfig {
        SamplerConfig {
            temperature: 1.0,
            top_p: 0.95,
            repeat_penalty: 1.3,
            repeat_window: window,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: vec![7, 9],
            top_k: None,
            min_p: None,
        }
    }

    /// The packed scope of row j is the trailing (capped) window of
    /// `history ++ drafts[..j]`; point-mass q records carry the draft.
    #[test]
    fn pack_scopes_and_point_masses() {
        let logits = meta_tensor(4 * 10);
        for window in [1usize, 3, 5, 64, 200] {
            for h in [0usize, 2, 6, 80] {
                let history: Vec<u32> = (0..h as u32).map(|x| 100 + x).collect();
                let drafts = [11u32, 12, 13];
                let cfg = cfg_pen(window);
                let q = vec![vec![], vec![(12, 0.5), (40, 0.5)], vec![]];
                let job = LaneDrawJob {
                    logits: &logits,
                    rows: 4,
                    vocab: 10,
                    drafts: &drafts,
                    q: &q,
                    history: &history,
                    cfg: &cfg,
                    rng_state: 5,
                    eos: Some(2),
                };
                check_job(0, &job).unwrap();
                let mut w = Vec::new();
                let layout = pack(std::slice::from_ref(&job), &mut w);
                assert_eq!(layout.rows, 4);
                let cap = window.min(LANE_DRAW_REPEAT_CAP);
                for j in 0..4 {
                    let rr = layout.rows_off() + j * LLV_ROW_WORDS;
                    let (off, len) = (w[rr + LR_PEN_OFF] as usize, w[rr + LR_PEN_LEN] as usize);
                    let mut all = history.clone();
                    all.extend_from_slice(&drafts[..j]);
                    let want = &all[all.len().saturating_sub(cap)..];
                    assert_eq!(&w[off..off + len], want, "w={window} h={h} j={j}");
                }
                let lr = layout.lanes_off();
                let q0 = w[lr + LL_Q] as usize;
                assert_eq!(&w[q0..q0 + 2], &[1, 11]);
                assert_eq!(w[q0 + 1 + LLV_Q_MAX], 1.0f32.to_bits());
                let q1 = q0 + LLV_Q_WORDS;
                assert_eq!(&w[q1..q1 + 3], &[2, 12, 40]);
                let b = w[lr + LL_BLOCKED] as usize;
                assert_eq!(&w[b..b + 2], &[7, 9]);
                assert_eq!(w[lr + LL_EOS], 2);
                assert_eq!(w[lr + LL_TOP_K], 20);
            }
        }
    }

    #[test]
    fn malformed_jobs_are_refused() {
        let logits = meta_tensor(3 * 10);
        let cfg = cfg_pen(4);
        let ok_q: Vec<Vec<(u32, f32)>> = vec![];
        let base = LaneDrawJob {
            logits: &logits,
            rows: 3,
            vocab: 10,
            drafts: &[1, 2],
            q: &ok_q,
            history: &[],
            cfg: &cfg,
            rng_state: 1,
            eos: None,
        };
        check_job(0, &base).unwrap();
        assert!(check_job(0, &LaneDrawJob { rows: 2, ..base }).is_err());
        assert!(check_job(
            0,
            &LaneDrawJob {
                rng_state: 0,
                ..base
            }
        )
        .is_err());
        assert!(check_job(0, &LaneDrawJob { vocab: 11, ..base }).is_err());
        let wrong_q = vec![vec![(1, 1.0)]];
        assert!(check_job(
            0,
            &LaneDrawJob {
                q: &wrong_q,
                ..base
            }
        )
        .is_err());
        let wide_q = vec![(0..65).map(|t| (t, 0.01)).collect::<Vec<_>>(), vec![]];
        assert!(check_job(0, &LaneDrawJob { q: &wide_q, ..base }).is_err());
        assert!(check_job(
            0,
            &LaneDrawJob {
                eos: Some(u32::MAX),
                ..base
            }
        )
        .is_err());
    }
}
