// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `session` subcommand: a session-cache hit is BYTE-IDENTICAL to a
//! cold prefill on the GPU.
//!
//! Opts: `model` (HFQ), `tokens` (token file, see `logits::read_tokens`), optional
//! `split` (length of the turn-1 prompt A; default `len - len/4`, nudged off the
//! prefill grid), `decode` (greedy tokens decoded after every prefill, default 16),
//! `budget-mib` (session cache device budget, default 4096),
//! `prefill serial|batched` (default batched).
//!
//! With T = tokens, A = T[..split], U = an unrelated prompt of A's length that
//! shares no token position with T:
//!
//! 1. `cold_reference`: prefill T through the (fresh) attached cache with `reused = 0`,
//!    keep the final logits, the persistent slab, and the greedy decode (ids and
//!    per-step logits). Nothing is committed.
//! 2. `turn1_publish`: prefill A (cold), greedy-decode (dirty decoded rows above A),
//!    commit.
//! 3. `resident_followup`: `session_plan(T)` must be a non-zero grid point (the resident
//!    snapshot at A's grid end); prefill T from it; logits, slab, decode ids and decode
//!    logits must equal the cold reference bit-for-bit. Commit.
//! 4. `pool_after_unrelated`: prefill + decode + commit U (overwrites latent rows and
//!    drops the resident), then `session_plan(T)` must be a non-zero stride snapshot;
//!    the same bit-exact comparison (chain restore + mirror re-expansion).
//! 5. `budget`: the cache's `pool_bytes` never exceeded `budget_bytes` at any sample
//!    and the pool actually held snapshots.
//!
//! Result: `{"pass":bool,"cases":[{"name","pass","detail",..},..],...}`. Every pass is
//! a measured comparison; a case that cannot run is a failure, never a pass.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use hipfire_arch_raon::bundle::{PrefillMode, RaonBundle};
use hipfire_arch_raon::session::SESSION_STRIDE_TOKENS;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::{json, Value};

use super::logits::{check_tokens, load_bundle, parse_prefill, read_tokens};

/// Tokens decoded after the unrelated prompt (dirty rows only; not compared).
const UNRELATED_DECODE: usize = 4;

/// First maximum (lowest index on ties); the oracle's greedy rule.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn parse_usize(
    opts: &BTreeMap<String, String>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    match opts.get(key) {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("session: --{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

/// One prefill + greedy decode, captured for comparison.
struct Captured {
    /// Final prompt row logits (`bundle.download_logits` right after the prefill).
    logits: Vec<f32>,
    /// Persistent slab right after the prefill, before any decode.
    slab: Vec<u8>,
    /// Greedy ids decoded after the prefill.
    ids: Vec<u32>,
    /// Logits after each decode step, flattened `[ids.len(), vocab]`.
    step_logits: Vec<f32>,
    /// Prefill wall time incl. the logits download (the stream-ordered sync).
    prefill_ms: f64,
}

fn finite(v: &[f32], what: &str) -> Result<(), String> {
    match v.iter().position(|x| !x.is_finite()) {
        Some(i) => Err(format!("session: non-finite {what}[{i}] = {}", v[i])),
        None => Ok(()),
    }
}

/// `session_prefill(prompt, reused)` then `decode` greedy tokens.
fn prefill_and_decode(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    reused: usize,
    decode: usize,
) -> Result<Captured, String> {
    bundle.release_session_for_growth(gpu, prompt.len() + decode + 1);
    let t0 = Instant::now();
    bundle.session_prefill(gpu, prompt, reused)?;
    let mut cur = bundle.download_logits(gpu)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    finite(&cur, "prefill logits")?;
    let logits = cur.clone();
    let slab = bundle.state.download_persist(gpu)?;
    let mut ids = Vec::with_capacity(decode);
    let mut step_logits = Vec::with_capacity(decode * cur.len());
    for _ in 0..decode {
        let tok = argmax(&cur) as u32;
        ids.push(tok);
        bundle.decode_step(gpu, tok)?;
        cur = bundle.download_logits(gpu)?;
        finite(&cur, "decode logits")?;
        step_logits.extend_from_slice(&cur);
    }
    Ok(Captured {
        logits,
        slab,
        ids,
        step_logits,
        prefill_ms,
    })
}

/// Bitwise comparison of two f32 vectors: `(equal, report)`.
fn cmp_f32(a: &[f32], b: &[f32], row_len: usize) -> (bool, Value) {
    let mut max_abs = 0.0f64;
    let mut first: Option<usize> = None;
    let mut n_mismatch = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            n_mismatch += 1;
            if first.is_none() {
                first = Some(i);
            }
            let d = (x as f64 - y as f64).abs();
            max_abs = max_abs.max(if d.is_nan() { f64::INFINITY } else { d });
        }
    }
    let len_equal = a.len() == b.len();
    let equal = len_equal && n_mismatch == 0;
    let mut r = json!({
        "equal": equal,
        "len_a": a.len(),
        "len_b": b.len(),
        "n_mismatch": n_mismatch,
        "max_abs_diff": max_abs,
    });
    if let Some(i) = first {
        r["first_mismatch"] = json!(i);
        r["first_mismatch_row"] = json!(i / row_len.max(1));
        r["first_mismatch_a"] = json!(a[i]);
        r["first_mismatch_b"] = json!(b[i]);
    }
    (equal, r)
}

/// Bytewise comparison of two slabs: `(equal, report)`.
fn cmp_bytes(a: &[u8], b: &[u8]) -> (bool, Value) {
    let mut first: Option<usize> = None;
    let mut n_mismatch = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            n_mismatch += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    let equal = a.len() == b.len() && n_mismatch == 0;
    let mut r = json!({
        "equal": equal,
        "len_a": a.len(),
        "len_b": b.len(),
        "n_mismatch": n_mismatch,
    });
    if let Some(i) = first {
        r["first_mismatch"] = json!(i);
    }
    (equal, r)
}

/// Device pool occupancy samples (`pool_bytes <= budget_bytes`).
#[derive(Default)]
struct BudgetTrack {
    samples: usize,
    violations: Vec<String>,
    max_pool_bytes: u64,
    budget_bytes: u64,
    max_device_total: u64,
}

impl BudgetTrack {
    fn sample(&mut self, bundle: &RaonBundle, label: &str) -> Result<(), String> {
        let u = bundle
            .session_usage()
            .ok_or("session: no session cache attached (usage unavailable)")?;
        self.samples += 1;
        self.budget_bytes = u.budget_bytes;
        self.max_pool_bytes = self.max_pool_bytes.max(u.pool_bytes);
        self.max_device_total = self.max_device_total.max(u.device_total_bytes());
        if u.pool_bytes > u.budget_bytes {
            self.violations.push(format!(
                "{label}: pool_bytes {} > budget_bytes {}",
                u.pool_bytes, u.budget_bytes
            ));
        }
        Ok(())
    }

    fn case(&self) -> Value {
        let pass = self.samples > 0 && self.violations.is_empty() && self.max_pool_bytes > 0;
        let detail = if !self.violations.is_empty() {
            self.violations.join("; ")
        } else if self.max_pool_bytes == 0 {
            "the pool never held a snapshot (nothing was published)".to_string()
        } else {
            format!(
                "max pool_bytes {} <= budget_bytes {} over {} samples",
                self.max_pool_bytes, self.budget_bytes, self.samples
            )
        };
        json!({
            "name": "budget",
            "pass": pass,
            "detail": detail,
            "samples": self.samples,
            "max_pool_bytes": self.max_pool_bytes,
            "budget_bytes": self.budget_bytes,
            "max_device_total_bytes": self.max_device_total,
            "violations": self.violations,
        })
    }
}

/// Shared description of one hit-vs-cold comparison case.
struct HitCase<'a> {
    name: &'a str,
    reused: usize,
    n_tokens: usize,
    /// Prefill step width (1 on the serial route): a restore point must be a multiple.
    wstep: usize,
    cold: &'a Captured,
    hit: &'a Captured,
    vocab: usize,
    extra: Value,
}

impl HitCase<'_> {
    fn verdict(&self) -> Value {
        let (logits_eq, logits) = cmp_f32(&self.cold.logits, &self.hit.logits, self.vocab);
        let (slab_eq, slab) = cmp_bytes(&self.cold.slab, &self.hit.slab);
        let ids_eq = self.cold.ids == self.hit.ids;
        let ids_first = self
            .cold
            .ids
            .iter()
            .zip(self.hit.ids.iter())
            .position(|(a, b)| a != b);
        let (dec_eq, dec) = cmp_f32(&self.cold.step_logits, &self.hit.step_logits, self.vocab);
        let mut problems: Vec<String> = Vec::new();
        if self.reused == 0 {
            problems.push("plan returned 0: no snapshot was reused".into());
        }
        if self.reused % self.wstep != 0 {
            problems.push(format!(
                "reused {} is not on the prefill grid (step {})",
                self.reused, self.wstep
            ));
        }
        if self.reused >= self.n_tokens {
            problems.push(format!(
                "reused {} leaves no token to prefill (n_tokens {})",
                self.reused, self.n_tokens
            ));
        }
        if !logits_eq {
            problems.push("final logits differ from cold".into());
        }
        if !slab_eq {
            problems.push("persistent slab differs from cold".into());
        }
        if !ids_eq {
            problems.push(format!("decode ids differ (first at step {ids_first:?})"));
        }
        if !dec_eq {
            problems.push("decode logits differ from cold".into());
        }
        let pass = problems.is_empty();
        let detail = if pass {
            format!(
                "hit reused {}/{} tokens: logits, slab, {} decode ids and decode logits bit-exact vs cold ({:.1} ms vs cold {:.1} ms)",
                self.reused,
                self.n_tokens,
                self.hit.ids.len(),
                self.hit.prefill_ms,
                self.cold.prefill_ms
            )
        } else {
            problems.join("; ")
        };
        let mut v = json!({
            "name": self.name,
            "pass": pass,
            "detail": detail,
            "reused": self.reused,
            "n_tokens": self.n_tokens,
            "prefilled_tokens": self.n_tokens.saturating_sub(self.reused),
            "cold_prefill_ms": self.cold.prefill_ms,
            "hit_prefill_ms": self.hit.prefill_ms,
            "speedup": self.cold.prefill_ms / self.hit.prefill_ms.max(1e-9),
            "logits": logits,
            "slab": slab,
            "decode_ids_equal": ids_eq,
            "decode_ids_cold": self.cold.ids,
            "decode_ids_hit": self.hit.ids,
            "decode_logits": dec,
        });
        if let (Some(obj), Some(extra)) = (v.as_object_mut(), self.extra.as_object()) {
            for (k, x) in extra {
                obj.insert(k.clone(), x.clone());
            }
        }
        v
    }
}

fn finish(cases: Vec<Value>, mut head: Value) -> Value {
    let pass = !cases.is_empty()
        && cases
            .iter()
            .all(|c| c.get("pass").and_then(Value::as_bool) == Some(true));
    head["pass"] = json!(pass);
    head["cases"] = json!(cases);
    head
}

pub fn run(_fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let model = opts.get("model").ok_or("session: missing --model")?.clone();
    let tokens_path = opts.get("tokens").ok_or("session: missing --tokens")?;
    let decode = parse_usize(opts, "decode", 16)?;
    if decode == 0 {
        return Err("session: --decode must be >= 1".into());
    }
    let budget_mib = parse_usize(opts, "budget-mib", 4096)?;
    if budget_mib == 0 {
        return Err("session: --budget-mib must be >= 1".into());
    }
    let budget_bytes = (budget_mib as u64) << 20;
    let mode = parse_prefill(opts)?;
    let t_all = read_tokens(tokens_path)?;

    // Cache identity: the production domain recipe (HFQ digest + tokenizer + template).
    let hfq = HfqFile::open(Path::new(&model)).map_err(|e| format!("open {model}: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let chat_template = hfq.chat_template();

    let mut gpu = Gpu::init().map_err(|e| format!("session: gpu init: {e:?}"))?;
    let domain = CacheDomain::for_model(
        &hfq,
        &tokenizer,
        chat_template.as_deref(),
        "raon1_mini",
        gpu.device_id,
    );
    let max_seq = (t_all.len() + decode + 64).max(512);
    let mut bundle = load_bundle(&mut gpu, ModelSource::Hfq(hfq), &model, max_seq)?;
    let r = run_inner(
        &mut bundle,
        &mut gpu,
        domain,
        budget_bytes,
        mode,
        &t_all,
        decode,
        opts,
    );
    bundle.free_all(&mut gpu);
    r
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    domain: CacheDomain,
    budget_bytes: u64,
    mode: PrefillMode,
    t_all: &[u32],
    decode: usize,
    opts: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab;
    check_tokens(t_all, vocab, "tokens")?;
    if vocab <= 2048 {
        return Err(format!(
            "session: vocab {vocab} too small for the unrelated-prompt remap"
        ));
    }
    bundle.set_prefill_mode(mode);
    let w = bundle.prefill_chunk_rows();
    let wstep = match mode {
        PrefillMode::Serial => 1,
        PrefillMode::Batched => w,
    };
    let n = t_all.len();
    if n < 3 * w + 1 {
        return Err(format!(
            "session: {n} tokens < 3*W+1 = {} (W = prefill_chunk_rows = {w}); supply a longer --tokens",
            3 * w + 1
        ));
    }
    // Stride S: smallest multiple of the step width >= SESSION_STRIDE_TOKENS.
    let stride = SESSION_STRIDE_TOKENS.div_ceil(wstep) * wstep;
    let mut split = parse_usize(opts, "split", n - n / 4)?;
    if opts.get("split").is_none() && wstep > 1 && split % wstep == 0 {
        split -= 1;
    }
    if split == 0 || split >= n {
        return Err(format!("session: --split {split} must be in 1..{n}"));
    }
    if wstep > 1 && split % wstep == 0 {
        return Err(format!(
            "session: --split {split} is a multiple of the prefill width {wstep}; the follow-up would have no tail"
        ));
    }
    if split <= stride {
        return Err(format!(
            "session: split {split} <= stride {stride}: no pool snapshot would be published for pool_after_unrelated; supply more tokens"
        ));
    }
    let t = t_all;
    let a = &t[..split];
    // U: T rotated by n/2 then remapped away from T at every position.
    let span = (vocab - 1024) as u64;
    let u: Vec<u32> = (0..split)
        .map(|i| {
            let src = t[(i + n / 2) % n] as u64;
            let mut id = ((src * 7919 + 13) % span + 1024) as u32;
            if id == t[i] {
                id = ((id as u64 - 1024 + 1) % span + 1024) as u32;
            }
            id
        })
        .collect();
    debug_assert!(u.iter().zip(t.iter()).all(|(x, y)| x != y));

    let cache = SessionCache::new(domain, budget_bytes);
    bundle.attach_session_cache(cache);
    if bundle.session_cache().is_none() {
        return Err("session: attach_session_cache left no cache attached".into());
    }
    let snapshot_bytes = bundle.session_snapshot_bytes();
    eprintln!(
        "session: n={n} split={split} W={w} wstep={wstep} stride={stride} decode={decode} budget={} MiB snapshot={} MiB",
        budget_bytes >> 20,
        snapshot_bytes >> 20
    );

    let head = json!({
        "n_tokens": n,
        "split": split,
        "prefill_chunk_rows": w,
        "prefill_step_rows": wstep,
        "stride": stride,
        "decode": decode,
        "vocab": vocab,
        "budget_bytes": budget_bytes,
        "snapshot_bytes": snapshot_bytes,
        "prefill": match mode {
            PrefillMode::Serial => "serial",
            PrefillMode::Batched => "batched",
        },
    });
    let mut cases: Vec<Value> = Vec::new();
    let mut budget = BudgetTrack::default();
    let plan =
        |bundle: &RaonBundle, p: &[u32]| ArchModel::session_plan(bundle, p, SessionRoute::Ar);

    // 1. Cold reference over a fresh cache.
    bundle.reset(gpu)?;
    let cold_plan = plan(bundle, t);
    if cold_plan != 0 {
        cases.push(json!({
            "name": "cold_reference",
            "pass": false,
            "detail": format!("fresh cache planned reused={cold_plan} (expected 0)"),
            "reused": cold_plan,
        }));
        cases.push(budget.case());
        return Ok(finish(cases, head));
    }
    let cold = prefill_and_decode(bundle, gpu, t, 0, decode)?;
    budget.sample(bundle, "after cold prefill")?;
    cases.push(json!({
        "name": "cold_reference",
        "pass": true,
        "detail": format!("cold prefill of {n} tokens in {:.1} ms, {} decode ids", cold.prefill_ms, cold.ids.len()),
        "reused": 0,
        "n_tokens": n,
        "cold_prefill_ms": cold.prefill_ms,
        "decode_ids": cold.ids,
    }));

    // 2. Turn 1: prompt A (the next begin drops the uncommitted cold captures).
    let a_plan = plan(bundle, a);
    if a_plan % wstep != 0 || a_plan >= a.len() {
        cases.push(json!({
            "name": "turn1_publish",
            "pass": false,
            "detail": format!("plan(A) = {a_plan} is not a restorable grid point (step {wstep}, |A| = {})", a.len()),
            "reused": a_plan,
        }));
        cases.push(budget.case());
        return Ok(finish(cases, head));
    }
    let turn1 = prefill_and_decode(bundle, gpu, a, a_plan, decode)?;
    budget.sample(bundle, "after turn1 prefill+decode")?;
    ArchModel::session_commit(bundle);
    budget.sample(bundle, "after turn1 commit")?;
    let pool_after_turn1 = bundle.session_usage().map(|u| u.pool_bytes).unwrap_or(0);
    cases.push(json!({
        "name": "turn1_publish",
        "pass": pool_after_turn1 > 0,
        "detail": if pool_after_turn1 > 0 {
            format!("A ({} tokens, reused {a_plan}) committed; pool holds {pool_after_turn1} bytes", a.len())
        } else {
            "turn 1 committed but the pool holds no snapshot".to_string()
        },
        "reused": a_plan,
        "n_tokens": a.len(),
        "prefill_ms": turn1.prefill_ms,
        "pool_bytes": pool_after_turn1,
    }));

    // 3. Resident follow-up: T extends A.
    let expected_resident = split / wstep * wstep;
    let reused = plan(bundle, t);
    let mut resident_cap = None;
    if reused != 0 && reused % wstep == 0 && reused < n {
        resident_cap = Some(prefill_and_decode(bundle, gpu, t, reused, decode)?);
        budget.sample(bundle, "after resident_followup prefill+decode")?;
        ArchModel::session_commit(bundle);
        budget.sample(bundle, "after resident_followup commit")?;
    }
    match &resident_cap {
        Some(hit) => cases.push(
            HitCase {
                name: "resident_followup",
                reused,
                n_tokens: n,
                wstep,
                cold: &cold,
                hit,
                vocab,
                extra: json!({
                    "expected_reused": expected_resident,
                    "reused_matches_expected": reused == expected_resident,
                }),
            }
            .verdict(),
        ),
        None => cases.push(json!({
            "name": "resident_followup",
            "pass": false,
            "detail": format!("plan(T) = {reused} is not a restorable grid point (expected {expected_resident}, step {wstep}, n {n}); nothing to compare"),
            "reused": reused,
            "expected_reused": expected_resident,
            "n_tokens": n,
        })),
    }

    // 4. Unrelated prompt, then T again from the pool.
    let u_plan = plan(bundle, &u);
    if u_plan % wstep != 0 || u_plan >= u.len() {
        cases.push(json!({
            "name": "pool_after_unrelated",
            "pass": false,
            "detail": format!("plan(U) = {u_plan} is not a restorable grid point (step {wstep}, |U| = {})", u.len()),
            "reused": u_plan,
        }));
        cases.push(budget.case());
        return Ok(finish(cases, head));
    }
    let _ = prefill_and_decode(bundle, gpu, &u, u_plan, decode.min(UNRELATED_DECODE))?;
    budget.sample(bundle, "after unrelated prefill+decode")?;
    ArchModel::session_commit(bundle);
    budget.sample(bundle, "after unrelated commit")?;
    let reused = plan(bundle, t);
    let mut pool_cap = None;
    if reused != 0 && reused % wstep == 0 && reused < n {
        pool_cap = Some(prefill_and_decode(bundle, gpu, t, reused, decode)?);
        budget.sample(bundle, "after pool_after_unrelated prefill+decode")?;
        ArchModel::session_commit(bundle);
        budget.sample(bundle, "after pool_after_unrelated commit")?;
    }
    match &pool_cap {
        Some(hit) => cases.push(
            HitCase {
                name: "pool_after_unrelated",
                reused,
                n_tokens: n,
                wstep,
                cold: &cold,
                hit,
                vocab,
                extra: json!({
                    "unrelated_plan": u_plan,
                    "unrelated_tokens": u.len(),
                    "stride": stride,
                    "reused_is_stride_multiple": reused % stride == 0,
                }),
            }
            .verdict(),
        ),
        None => cases.push(json!({
            "name": "pool_after_unrelated",
            "pass": false,
            "detail": format!("plan(T) = {reused} after the unrelated prompt is not a restorable grid point (stride {stride}, step {wstep}, n {n}); nothing to compare"),
            "reused": reused,
            "n_tokens": n,
        })),
    }

    // 5. Budget.
    cases.push(budget.case());
    Ok(finish(cases, head))
}
