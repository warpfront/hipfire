//! Online draft tuning for chain DFlash (developer-only, `HIPFIRE_DFLASH_ONLINE_TUNE`).
//!
//! The drafter's greedy proposal at each block row is re-ranked among the
//! draft's own top-K logits by a small per-request model trained online from
//! tokens the target has already verified. A DFlash block row's draft logits
//! depend only on the committed context and the mask tokens, never on the
//! other drafted tokens, so every row is a clean supervised example once the
//! emitted stream reaches its position: `(draft top-K at row, true token)`.
//!
//! What the draft cannot see is its own chain: row `r` is drafted without
//! knowing the tokens proposed at rows `< r`. The re-ranker supplies that
//! from session statistics — n-gram continuations (orders 1..=3) of the
//! proposed chain over the verified stream — plus a rank prior and a
//! depth-scaled draft-logit term:
//! `score_j = scale·(l_j − l_0) + w·f_j`, trained by Adagrad on the
//! softmax-over-candidates cross-entropy of the rows a greedy chain actually
//! reaches (all earlier rows correct). Features use the stream as of the
//! block seed only, at proposal and at training time.
//!
//! Verify stays the target's greedy argmax, so every emitted token is the
//! target's; only the proposal (and therefore acceptance τ) moves. (Batched
//! verify is not bit-identical to AR, so a different acceptance pattern can
//! still flip a near-tie — the same caveat as the adaptive block width.)
//!
//! Negative results (offline replay, `examples/dflash_online_replay.rs`,
//! Qwen3.5-9B + 9B DFlash draft, gfx1151): a rank-16 LoRA on the LM head over
//! the draft hidden, a per-token bias, and the target's stale verify argmax
//! past the rejection all lowered or did not move τ. Within one request the
//! LoRA only memorizes (in-sample after 4 epochs essay τ 1.10 → 1.88, online
//! 1.10 → 0.94). With its weights carried across requests (leave-one-out over
//! 10 sessions) a low-rate LoRA adds +0.6 pp τ (×1.062 → ×1.068), but it needs
//! a 245 KB hidden D2H and ~1 ms of host work per cycle, loses 0.5 pp cold,
//! and its learning rate is sharply peaked. It is not worth carrying.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{HashMap, VecDeque};

/// Draft candidates kept per row (top-K kernel limit is 16).
pub const K: usize = 16;
/// n-gram context orders (bigram .. 4-gram).
const ORDERS: usize = 3;
/// Dense features: rank one-hots (3), per order (MLE, seen), recency, depth·gap,
/// suffix-match length, injected-candidate flag, equals chain token 1 / 2 back,
/// equals the previous row's top-1 / the next row's top-1 / top-2.
const NF: usize = 15 + 2 * ORDERS;
const RECENT: usize = 64;
/// Longest suffix match considered, and occurrences of the nearest token scanned.
const MAXM: usize = 32;
const MAX_OCC: usize = 64;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Collect acceptance-ceiling statistics only; proposals stay argmax.
    Stats,
    /// Train online and re-rank proposals.
    On,
    /// Evaluation dumps: the caller drafts a token the target never picks, so
    /// every cycle commits one target token and the dump holds the draft's
    /// top-K at *every* position — `dflash_online_replay --simulate` then
    /// replays any policy with its own block starts (fixed-start replay of a
    /// normal session cannot: a policy that accepts more starts later blocks
    /// elsewhere).
    Sweep,
}

impl Mode {
    pub fn from_env() -> Option<Mode> {
        match hipfire_config::developer_var("HIPFIRE_DFLASH_ONLINE_TUNE")
            .ok()?
            .as_str()
        {
            "stats" => Some(Mode::Stats),
            "1" | "on" => Some(Mode::On),
            "sweep" => Some(Mode::Sweep),
            _ => None,
        }
    }
}

/// Learner hyper-parameters. `HIPFIRE_DFLASH_ONLINE_HP="key=val,..."`
/// overrides the defaults (developer sweeps; the replay tool takes the same string).
#[derive(Clone, Debug)]
pub struct Hyper {
    pub lr: f32,
    /// Adagrad initial accumulator: steps are ~`lr·g` until gradients accumulate.
    pub acc0: f32,
    /// Minimum score lead over the argmax candidate required to override it.
    pub margin: f32,
    /// Suffix-match length at which a continuation outside the draft top-K
    /// replaces the last candidate (0 = never inject).
    pub inject: f32,
    /// Keep the learner's weights across requests. Only the structural
    /// weights carry; session text statistics (n-grams, suffix history)
    /// always reset, so no request's text leaks into another's proposals.
    /// Short requests otherwise never leave the cold start (replay, first 32
    /// cycles: x1.018 cold vs x1.058 carried).
    pub carry: bool,
}

impl Default for Hyper {
    fn default() -> Self {
        Self {
            lr: 0.3,
            acc0: 1.0,
            margin: 0.0,
            inject: 1.0,
            carry: true,
        }
    }
}

impl Hyper {
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut h = Self::default();
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad hp `{kv}`"))?;
            let v = v.parse::<f32>().map_err(|e| format!("hp {k}: {e}"))?;
            match k {
                "lr" => h.lr = v,
                "acc0" => h.acc0 = v,
                "margin" => h.margin = v,
                "inject" => h.inject = v,
                "carry" => h.carry = v != 0.0,
                _ => return Err(format!("unknown hp `{k}`")),
            }
        }
        Ok(h)
    }

    fn from_env() -> Self {
        let spec = hipfire_config::developer_var("HIPFIRE_DFLASH_ONLINE_HP").unwrap_or_default();
        Self::parse(&spec).unwrap_or_else(|e| panic!("HIPFIRE_DFLASH_ONLINE_HP: {e}"))
    }
}

/// One drafted block awaiting labels from the emitted stream.
struct Cycle {
    /// Absolute position of the seed; row `i` (0-based) predicts `start + 1 + i`.
    start: usize,
    ids: Vec<[u32; K]>,
    vals: Vec<[f32; K]>,
    /// Label rank per row in the draft top-K (`K` = absent), once known.
    ranks: Vec<Option<u8>>,
    /// Candidate index the tuner proposed per row.
    picked: Vec<u8>,
    /// Row had a suffix-match continuation injected as candidate `K-1`.
    injected: Vec<bool>,
}

#[derive(Default)]
struct Stats {
    cycles: u64,
    accepted: u64,
    finalized: u64,
    /// Sum over finalized cycles of the longest row prefix whose label rank < 2^j.
    oracle: [u64; 5],
    /// Sum over finalized cycles of the tuner's picked-prefix length.
    picked_prefix: u64,
    /// Overrides of the argmax inside the reached prefix / and right.
    overrides: u64,
    overrides_right: u64,
    /// Host time in `propose_from_logits` (incl. waiting for the draft
    /// forward, top-K + D2H), its pure-host re-rank part, and `observe`.
    propose_ns: u64,
    rerank_ns: u64,
    observe_ns: u64,
}

/// Session n-gram counts over the known emitted stream.
#[derive(Default)]
struct Ngrams {
    /// `[order-1]`: (context, next) and context counts, keyed by hash.
    next: [HashMap<u64, u32>; ORDERS],
    ctx: [HashMap<u64, u32>; ORDERS],
    /// Positions of each token, ascending.
    occ: HashMap<u32, Vec<usize>>,
    /// Positions `< upto` have their n-grams ending there counted.
    upto: usize,
}

/// Hash of the context `prev[..k]` (`prev[0]` = nearest token).
fn ctx_hash(prev: &[u32]) -> u64 {
    prev.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
        (h ^ t as u64).wrapping_mul(0x0100_0000_01b3)
    })
}
fn next_hash(ctx: u64, v: u32) -> u64 {
    ctx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ v as u64
}

impl Ngrams {
    fn add(&mut self, known: &[u32], q: usize) {
        let v = known[q];
        if v == NONE {
            return;
        }
        let mut prev = [NONE; ORDERS];
        for k in 0..ORDERS.min(q) {
            prev[k] = known[q - 1 - k];
            if prev[k] == NONE {
                break;
            }
            let c = ctx_hash(&prev[..=k]);
            *self.next[k].entry(next_hash(c, v)).or_default() += 1;
            *self.ctx[k].entry(c).or_default() += 1;
        }
    }

    fn advance(&mut self, known: &[u32]) {
        if self.upto > known.len() {
            *self = Ngrams::default();
        }
        for q in self.upto.max(1)..known.len() {
            self.add(known, q);
        }
        for (q, &t) in known.iter().enumerate().skip(self.upto) {
            if t != NONE {
                self.occ.entry(t).or_default().push(q);
            }
        }
        self.upto = known.len().max(self.upto);
    }
}

/// Draft top-1 of the previous row and top-1/top-2 of the next row (`NONE`
/// at the block edges): parallel-drafted rows often land one position early
/// or late.
fn neighbors(ids: &[[u32; K]], r: usize) -> [u32; 3] {
    let prev = r.checked_sub(1).map_or(NONE, |p| ids[p][0]);
    let next = ids.get(r + 1);
    [
        prev,
        next.map_or(NONE, |n| n[0]),
        next.map_or(NONE, |n| n[1]),
    ]
}

/// Per row: `(token, softmax over the row's top-K draft logits)`.
fn row_probs(ids: &[[u32; K]], vals: &[[f32; K]]) -> Vec<[(u32, f32); K]> {
    ids.iter()
        .zip(vals)
        .map(|(i, v)| {
            let e: [f32; K] = std::array::from_fn(|j| (v[j] - v[0]).exp());
            let z: f32 = e.iter().sum();
            std::array::from_fn(|j| (i[j], e[j] / z))
        })
        .collect()
}

/// Rows r-1, r+1, r+2 of `row_probs` (empty past the block edges).
fn nb_probs(p: &[[(u32, f32); K]], r: usize) -> [&[(u32, f32)]; 3] {
    let get = |i: Option<usize>| i.and_then(|i| p.get(i)).map_or(&[][..], |x| &x[..]);
    [get(r.checked_sub(1)), get(Some(r + 1)), get(Some(r + 2))]
}

/// Dense re-ranker: `score_j = scale·(l_j − l_0) + w·f_j`, Adagrad. One weight
/// set for every row depth: per-depth sets learn too slowly from a cold start
/// (own-start simulation: 3 buckets ×1.072, 2 ×1.079, 1 ×1.083); depth enters
/// through the depth-scaled logit-gap feature.
struct Learner {
    hp: Hyper,
    scale: f32,
    g_scale: f32,
    w: [f32; NF],
    gw: [f32; NF],
}

fn adagrad(p: &mut f32, acc: &mut f32, g: f32, lr: f32) {
    *acc += g * g;
    *p -= lr * g / acc.sqrt();
}

impl Learner {
    fn new(hp: Hyper) -> Self {
        Self {
            scale: 1.0,
            g_scale: hp.acc0,
            w: [0.0; NF],
            gw: [hp.acc0; NF],
            hp,
        }
    }

    fn scores(&self, vals: &[f32; K], f: &[[f32; NF]; K]) -> [f32; K] {
        std::array::from_fn(|j| {
            self.scale * (vals[j] - vals[0])
                + self.w.iter().zip(&f[j]).map(|(w, x)| w * x).sum::<f32>()
        })
    }

    fn pick(&self, vals: &[f32; K], f: &[[f32; NF]; K]) -> usize {
        let s = self.scores(vals, f);
        let best = (0..K).fold(0, |b, j| if s[j] > s[b] { j } else { b });
        if s[best] - s[0] > self.hp.margin {
            best
        } else {
            0
        }
    }

    fn train(&mut self, vals: &[f32; K], f: &[[f32; NF]; K], y: usize) {
        let s = self.scores(vals, f);
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: [f32; K] = std::array::from_fn(|j| (s[j] - m).exp());
        let sum: f32 = e.iter().sum();
        let g: [f32; K] = std::array::from_fn(|j| e[j] / sum - (j == y) as u32 as f32);
        let gs: f32 = (0..K).map(|j| g[j] * (vals[j] - vals[0])).sum();
        adagrad(&mut self.scale, &mut self.g_scale, gs, self.hp.lr);
        for (k, (wk, gk_acc)) in self.w.iter_mut().zip(self.gw.iter_mut()).enumerate() {
            let gk: f32 = (0..K).map(|j| g[j] * f[j][k]).sum();
            adagrad(wk, gk_acc, gk, self.hp.lr);
        }
    }
}

/// Host top-`K` per row of `rows × vocab` logits, in the top-K kernel's
/// layout (`rows × K` ids and values) — for drafters whose logits are already
/// on the host (generic DFlash).
pub fn top_k_rows(logits: &[f32], vocab: usize, rows: usize) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(rows * K);
    let mut vals = Vec::with_capacity(rows * K);
    for row in logits.chunks_exact(vocab).take(rows) {
        // Descending (value, id) list; a candidate enters only if it beats the
        // current K-th, so the scan is one compare per logit.
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(K + 1);
        for (i, &x) in row.iter().enumerate() {
            if top.len() == K && x <= top[K - 1].0 {
                continue;
            }
            let at = top.partition_point(|&(v, _)| v >= x);
            top.insert(at, (x, i as u32));
            top.truncate(K);
        }
        ids.extend(top.iter().map(|t| t.1));
        vals.extend(top.iter().map(|t| t.0));
    }
    (ids, vals)
}

pub struct OnlineDraftTuner {
    pub mode: Mode,
    /// Known emitted tokens by absolute position (`NONE` = unknown).
    known: Vec<u32>,
    ngrams: Ngrams,
    pending: VecDeque<Cycle>,
    stats: Stats,
    learner: Learner,
    /// Device top-K ids (i32 in an F32 tensor) and values, `[max_rows × K]`.
    dev: Option<(GpuTensor, GpuTensor)>,
    /// `HIPFIRE_DFLASH_ONLINE_DUMP=<file>`: raw propose/observe records for
    /// the offline replay example (`dflash_online_replay`).
    dump: Option<std::io::BufWriter<std::fs::File>>,
}

impl OnlineDraftTuner {
    pub fn new(mode: Mode, hp: Hyper) -> Self {
        Self {
            mode,
            known: Vec::new(),
            ngrams: Ngrams::default(),
            pending: VecDeque::new(),
            stats: Stats::default(),
            learner: Learner::new(hp),
            dev: None,
            dump: None,
        }
    }

    pub fn from_env() -> Option<Self> {
        let mut t = Self::new(Mode::from_env()?, Hyper::from_env());
        if let Ok(path) = hipfire_config::developer_var("HIPFIRE_DFLASH_ONLINE_DUMP") {
            let f = std::fs::File::create(&path)
                .unwrap_or_else(|e| panic!("HIPFIRE_DFLASH_ONLINE_DUMP {path}: {e}"));
            t.dump = Some(std::io::BufWriter::new(f));
        }
        Some(t)
    }

    fn dump_record(&mut self, tag: u8, head: &[u64], words: &[u32]) {
        use std::io::Write;
        if let Some(d) = self.dump.as_mut() {
            let mut b = vec![tag];
            b.extend(head.iter().flat_map(|x| x.to_le_bytes()));
            b.extend(words.iter().flat_map(|x| x.to_le_bytes()));
            let _ = d.write_all(&b);
            let _ = d.flush();
        }
    }

    /// Draft proposals for `rows` rows of `logits` (`[rows × vocab]`, device):
    /// top-K on device, small D2H, host re-rank.
    pub fn propose_from_logits(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        vocab: usize,
        rows: usize,
        start: usize,
        seed: u32,
    ) -> rdna_compute::HipResult<Vec<u32>> {
        let t0 = std::time::Instant::now();
        if self.dev.as_ref().is_none_or(|(i, _)| i.numel() < rows * K) {
            if let Some((i, v)) = self.dev.take() {
                gpu.free_tensor(i)?;
                gpu.free_tensor(v)?;
            }
            let n = rows.max(16) * K;
            self.dev = Some((
                gpu.alloc_tensor(&[n], DType::F32)?,
                gpu.alloc_tensor(&[n], DType::F32)?,
            ));
        }
        let (di, dv) = self.dev.as_ref().unwrap();
        let di = di.sub_offset(0, rows * K);
        let dv = dv.sub_offset(0, rows * K);
        gpu.topk_values_batched_f32_verified_regrid(logits, &di, &dv, vocab, K, rows)?;
        let mut ids = vec![0u32; rows * K];
        // SAFETY: `ids` holds exactly rows*K u32; any byte pattern is valid.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, rows * K * 4) };
        gpu.hip.memcpy_dtoh(bytes, &di.buf)?;
        let vals = gpu.download_f32(&dv)?;
        let t1 = std::time::Instant::now();
        let out = self.propose(start, seed, &ids, &vals, rows);
        self.stats.rerank_ns += t1.elapsed().as_nanos() as u64;
        self.stats.propose_ns += t0.elapsed().as_nanos() as u64;
        Ok(out)
    }

    pub fn free_gpu(&mut self, gpu: &mut Gpu) {
        if let Some((i, v)) = self.dev.take() {
            let _ = gpu.free_tensor(i);
            let _ = gpu.free_tensor(v);
        }
    }

    /// New request: forget the session text; the weights carry unless
    /// `Hyper::carry` is off.
    pub fn reset(&mut self) {
        self.report("request end");
        self.dump_record(b'R', &[], &[]);
        self.known.clear();
        self.ngrams = Ngrams::default();
        self.pending.clear();
        self.stats = Stats::default();
        if !self.learner.hp.carry {
            self.learner = Learner::new(self.learner.hp.clone());
        }
    }

    /// Prompt tokens at absolute positions `0..prompt.len()` become the start
    /// of the session stream (n-gram and suffix-match history only: they are
    /// never training labels, no block drafted them).
    pub fn seed_prompt(&mut self, prompt: &[u32]) {
        self.dump_record(b'S', &[prompt.len() as u64], prompt);
        self.known.clear();
        self.known.extend_from_slice(prompt);
        self.ngrams = Ngrams::default();
        self.ngrams.advance(&self.known);
        self.pending.clear();
    }

    fn tok(&self, p: isize) -> u32 {
        if p < 0 {
            NONE
        } else {
            self.known.get(p as usize).copied().unwrap_or(NONE)
        }
    }

    /// Per order: (`prev[..k]` → `v`, `prev[..k]`) occurrences ending at
    /// positions in `(start, ngrams.upto)` — those the as-of-`start` view must
    /// not see.
    fn late(&self, start: usize, prev: &[u32; ORDERS], v: u32) -> [[u32; 2]; ORDERS] {
        let mut c = [[0u32; 2]; ORDERS];
        for q in start + 1..self.ngrams.upto {
            for k in 0..ORDERS.min(q) {
                if self.known[q - 1 - k] != prev[k] {
                    break;
                }
                c[k][1] += 1;
                c[k][0] += (self.known[q] == v) as u32;
            }
        }
        c
    }

    /// Continuations of the context `ctx` (nearest token first) seen in the
    /// stream as of `start`: `(next token, matched suffix length)`, longest
    /// match per token, most recent occurrences first.
    fn suffix_matches(&self, start: usize, ctx: &[u32]) -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = Vec::new();
        let Some(occ) = self.ngrams.occ.get(&ctx[0]).filter(|_| ctx[0] != NONE) else {
            return out;
        };
        for &q in occ.iter().rev().filter(|&&q| q < start).take(MAX_OCC) {
            let u = self.known[q + 1];
            if u == NONE {
                continue;
            }
            let mut l = 1;
            while l < ctx.len() && l <= q && ctx[l] != NONE && self.known[q - l] == ctx[l] {
                l += 1;
            }
            match out.iter_mut().find(|(t, _)| *t == u) {
                Some(e) => e.1 = e.1.max(l),
                None => out.push((u, l)),
            }
        }
        out
    }

    /// Dense features for the K candidates of a row predicting position
    /// `start + 1 + row` given the preceding tokens `ctx` (nearest first),
    /// using the session statistics as of `start` (no peeking past the seed).
    #[allow(clippy::too_many_arguments)]
    fn features(
        &self,
        start: usize,
        ctx: &[u32; MAXM],
        ids: &[u32; K],
        vals: &[f32; K],
        row: usize,
        injected: bool,
        nb: [u32; 3],
        nbp: [&[(u32, f32)]; 3],
    ) -> [[f32; NF]; K] {
        let ng = &self.ngrams;
        let prev: &[u32; ORDERS] = ctx[..ORDERS].try_into().unwrap();
        let recent_lo = (start + 1).saturating_sub(RECENT);
        let recent_hi = (start + 1).min(self.known.len());
        let depth = row as f32 / K as f32;
        let sfx = self.suffix_matches(start, ctx);
        std::array::from_fn(|j| {
            let v = ids[j];
            let mut f = [0f32; NF];
            if (1..=3).contains(&j) {
                f[j - 1] = 1.0;
            }
            let late = self.late(start, prev, v);
            for k in 0..ORDERS {
                if prev[k] == NONE {
                    break;
                }
                let c = ctx_hash(&prev[..=k]);
                let n = ng.next[k]
                    .get(&next_hash(c, v))
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(late[k][0]);
                let d = ng.ctx[k]
                    .get(&c)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(late[k][1]);
                if n > 0 {
                    f[3 + 2 * k] = n as f32 / d as f32;
                    f[4 + 2 * k] = 1.0;
                }
            }
            if recent_lo < recent_hi && self.known[recent_lo..recent_hi].contains(&v) {
                f[3 + 2 * ORDERS] = 1.0;
            }
            f[4 + 2 * ORDERS] = (vals[j] - vals[0]) * depth;
            if let Some(&(_, l)) = sfx.iter().find(|(t, _)| *t == v) {
                f[5 + 2 * ORDERS] = ((1 + l) as f32).ln() / ((1 + MAXM) as f32).ln();
            }
            f[6 + 2 * ORDERS] = (injected && j == K - 1) as u32 as f32;
            f[7 + 2 * ORDERS] = (v == ctx[0]) as u32 as f32;
            f[8 + 2 * ORDERS] = (v == ctx[1]) as u32 as f32;
            for (i, &n) in nb.iter().enumerate() {
                f[9 + 2 * ORDERS + i] = (v == n) as u32 as f32;
            }
            for (i, row) in nbp.iter().enumerate() {
                f[12 + 2 * ORDERS + i] = row.iter().find(|(t, _)| *t == v).map_or(0.0, |x| x.1);
            }
            f
        })
    }

    /// Choose the proposal for each row from its top-K.
    /// `ids`/`vals` are `rows × K` row-major as produced by the top-K kernel.
    pub fn propose(
        &mut self,
        start: usize,
        seed: u32,
        ids: &[u32],
        vals: &[f32],
        rows: usize,
    ) -> Vec<u32> {
        self.dump_record(b'P', &[start as u64, seed as u64, rows as u64], ids);
        self.dump_record(
            b'L',
            &[],
            &vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
        if self.known.len() <= start {
            self.known.resize(start + 1, NONE);
        }
        self.known[start] = seed;
        self.ngrams.advance(&self.known);
        let mut cyc = Cycle {
            start,
            ids: Vec::with_capacity(rows),
            vals: Vec::with_capacity(rows),
            ranks: vec![None; rows],
            picked: Vec::with_capacity(rows),
            injected: Vec::with_capacity(rows),
        };
        let mut out: Vec<u32> = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut order: [usize; K] = std::array::from_fn(|j| j);
            let v = &vals[r * K..(r + 1) * K];
            order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
            cyc.ids.push(std::array::from_fn(|j| ids[r * K + order[j]]));
            cyc.vals.push(std::array::from_fn(|j| v[order[j]]));
        }
        for r in 0..rows {
            let mut row_ids = cyc.ids[r];
            let row_vals = cyc.vals[r];
            let nb = neighbors(&cyc.ids, r);
            let probs = row_probs(&cyc.ids, &cyc.vals);
            let mut injected = false;
            let pick = if self.mode == Mode::On {
                // Proposal chain: the rows before this one are assumed accepted.
                let p = (start + 1 + r) as isize;
                let ctx: [u32; MAXM] = std::array::from_fn(|k| {
                    let q = p - 1 - k as isize;
                    if q > start as isize {
                        out[q as usize - start - 1]
                    } else {
                        self.tok(q)
                    }
                });
                if self.learner.hp.inject > 0.0 {
                    let best = self.suffix_matches(start, &ctx).into_iter().fold(
                        None,
                        |b: Option<(u32, usize)>, m| {
                            if b.is_none_or(|b| m.1 > b.1) {
                                Some(m)
                            } else {
                                b
                            }
                        },
                    );
                    if let Some((u, l)) = best {
                        if l as f32 >= self.learner.hp.inject && !row_ids.contains(&u) {
                            row_ids[K - 1] = u;
                            injected = true;
                        }
                    }
                }
                let f = self.features(
                    start,
                    &ctx,
                    &row_ids,
                    &row_vals,
                    r,
                    injected,
                    nb,
                    nb_probs(&probs, r),
                );
                self.learner.pick(&row_vals, &f)
            } else {
                0
            };
            out.push(row_ids[pick]);
            cyc.picked.push(pick as u8);
            cyc.injected.push(injected);
            cyc.ids[r] = row_ids;
        }
        self.pending.push_back(cyc);
        out
    }

    /// Record the tokens committed after the seed at `start` (accepted drafts
    /// plus the bonus) and train on every pending row they label.
    pub fn observe(&mut self, start: usize, committed_after_seed: &[u32], accepted: usize) {
        let t0 = std::time::Instant::now();
        self.dump_record(
            b'O',
            &[
                start as u64,
                accepted as u64,
                committed_after_seed.len() as u64,
            ],
            committed_after_seed,
        );
        self.stats.cycles += 1;
        self.stats.accepted += accepted as u64;
        // A rewind invalidates everything past the seed.
        self.known.truncate(start + 1);
        self.pending.retain(|c| c.start <= start);
        let end = start + 1 + committed_after_seed.len();
        if self.known.len() < end {
            self.known.resize(end, NONE);
        }
        self.known[start + 1..end].copy_from_slice(committed_after_seed);
        self.ngrams.advance(&self.known);

        while let Some(c) = self.pending.front_mut() {
            for (r, rank) in c.ranks.iter_mut().enumerate() {
                if rank.is_none() {
                    if let Some(&tok) = self.known.get(c.start + 1 + r).filter(|&&t| t != NONE) {
                        let pos = c.ids[r].iter().position(|&id| id == tok).unwrap_or(K);
                        *rank = Some(pos as u8);
                    }
                }
            }
            if c.ranks.iter().any(Option::is_none) {
                break;
            }
            let c = self.pending.pop_front().unwrap();
            self.finalize(&c);
        }
        self.stats.observe_ns += t0.elapsed().as_nanos() as u64;
        if self.stats.cycles.is_multiple_of(64) {
            self.report("running");
        }
    }

    fn finalize(&mut self, c: &Cycle) {
        let ranks: Vec<usize> = c.ranks.iter().map(|r| r.unwrap() as usize).collect();
        // Train the rows a greedy chain reaches: every earlier label was the argmax.
        for (r, &y) in ranks.iter().enumerate() {
            if y >= K {
                break;
            }
            let p = (c.start + 1 + r) as isize;
            let ctx: [u32; MAXM] = std::array::from_fn(|k| self.tok(p - 1 - k as isize));
            let f = self.features(
                c.start,
                &ctx,
                &c.ids[r],
                &c.vals[r],
                r,
                c.injected[r],
                neighbors(&c.ids, r),
                nb_probs(&row_probs(&c.ids, &c.vals), r),
            );
            self.learner.train(&c.vals[r], &f, y);
            if y != 0 {
                break;
            }
        }
        let s = &mut self.stats;
        s.finalized += 1;
        for (j, o) in s.oracle.iter_mut().enumerate() {
            *o += ranks.iter().take_while(|&&r| r < (1 << j)).count() as u64;
        }
        let prefix = ranks
            .iter()
            .zip(&c.picked)
            .take_while(|(r, p)| **r == **p as usize)
            .count();
        s.picked_prefix += prefix as u64;
        for (r, &p) in ranks.iter().zip(&c.picked).take(prefix + 1) {
            if p != 0 {
                s.overrides += 1;
                s.overrides_right += (*r == p as usize) as u64;
            }
        }
    }

    /// `(finalized cycles, argmax τ, tuner-picked τ)` at the recorded block
    /// starts — the offline replay's figure of merit.
    pub fn summary(&self) -> (u64, f64, f64) {
        let s = &self.stats;
        let f = s.finalized.max(1) as f64;
        (
            s.finalized,
            s.oracle[0] as f64 / f,
            s.picked_prefix as f64 / f,
        )
    }

    pub fn report(&self, tag: &str) {
        let s = &self.stats;
        if s.cycles == 0 {
            return;
        }
        let f = s.finalized.max(1) as f64;
        let o: Vec<String> = s
            .oracle
            .iter()
            .enumerate()
            .map(|(j, &v)| format!("k{}={:.3}", 1 << j, v as f64 / f))
            .collect();
        let l = &self.learner;
        eprintln!(
            "[dflash-online] {tag}: mode={:?} cycles={} tau={:.3} finalized={} picked_tau={:.3} overrides={}/{} oracle[{}] host_us/cycle[propose={:.0} rerank={:.0} observe={:.0}] scale={:.3} w={:.2?}",
            self.mode,
            s.cycles,
            s.accepted as f64 / s.cycles as f64,
            s.finalized,
            s.picked_prefix as f64 / f,
            s.overrides_right,
            s.overrides,
            o.join(" "),
            s.propose_ns as f64 / 1e3 / s.cycles as f64,
            s.rerank_ns as f64 / 1e3 / s.cycles as f64,
            s.observe_ns as f64 / 1e3 / s.cycles as f64,
            l.scale,
            l.w,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Features recomputed at training time, after the emitted stream has
    /// moved past the block, must equal what the proposal saw: any token at
    /// or after the seed leaking into n-gram, suffix or recency statistics
    /// would let the learner train on the future and inflate replay τ.
    #[test]
    fn training_features_match_proposal_features() {
        let mut t = OnlineDraftTuner::new(Mode::Stats, Hyper::default());
        let prompt: Vec<u32> = (0..40).map(|i| [7, 8, 9, 7, 8, 10][i % 6]).collect();
        t.seed_prompt(&prompt);
        let (start, seed) = (prompt.len(), 9u32);
        // The truth continues the pattern, so the late positions repeat the
        // very n-grams the rows ask about.
        let truth = [7u32, 8, 10, 7, 8, 9];
        let rows = truth.len();
        let row_ids: [u32; K] = std::array::from_fn(|j| {
            if j < 4 {
                [7, 8, 9, 10][j]
            } else {
                1000 + j as u32
            }
        });
        let row_vals: [f32; K] = std::array::from_fn(|j| 10.0 - j as f32);
        let ids: Vec<u32> = (0..rows).flat_map(|_| row_ids).collect();
        let vals: Vec<f32> = (0..rows).flat_map(|_| row_vals).collect();
        t.propose(start, seed, &ids, &vals, rows);
        let all_ids = vec![row_ids; rows];
        let feats = |t: &OnlineDraftTuner| -> Vec<[[f32; NF]; K]> {
            (0..rows)
                .map(|r| {
                    let p = (start + 1 + r) as isize;
                    let ctx: [u32; MAXM] = std::array::from_fn(|k| {
                        let q = p - 1 - k as isize;
                        if q > start as isize {
                            truth[q as usize - start - 1]
                        } else {
                            t.tok(q)
                        }
                    });
                    t.features(
                        start,
                        &ctx,
                        &row_ids,
                        &row_vals,
                        r,
                        false,
                        neighbors(&all_ids, r),
                        [&[]; 3],
                    )
                })
                .collect()
        };
        let before = feats(&t);
        t.observe(start, &truth, rows - 1);
        assert_eq!(
            t.known.len(),
            start + 1 + rows,
            "stream advanced past the block"
        );
        assert_eq!(before, feats(&t));
        // Not vacuous: row 0's context ends in the seed 9 and the prompt holds
        // the bigram 9 -> 7, so candidate 7 has its bigram feature set.
        assert!(before[0][0][4] > 0.0, "prompt bigram feature must fire");
    }
}
