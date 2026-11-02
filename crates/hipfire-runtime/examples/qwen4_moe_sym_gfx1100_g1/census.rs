// SPDX-License-Identifier: Apache-2.0
//! CPU-only hook-key census (`--collect-hook-keys OUT.json`).
//!
//! Walks EVERY planned candidate-independent CPU case of the G1 device run with the same scenario functions, sample
//! sets, seeds and CPU twin (`fold_twin`) the device run uses, and drives each GateUp output pair `(g, u)` through the
//! gfx1100 adapter `epilogue::grouped_prefill_epilogue_audit(GateUp, m = 2, [g, u], [0], hooks)` (the original public
//! `grouped_prefill_epilogue_with` plus the reviewed device MulUp NaN rule; the adapter adds no hook query), and audits it
//! against the public twin: every non-NaN-product output MUST be byte-identical (`mul_up_audit.nonnan_changed == 0`; the
//! census is not `complete` otherwise) and the NaN-result outputs are counted by rule branch. There is no host-math
//! substitute and no `CpuOnlySelfCheckHooks`/`PortableEpilogue` anywhere on this path.
//!
//! `CensusHooks` answers only from the qualified table (base probe table + divide extras + `--intrinsic-table`s) and
//! records EVERY query (known or missing) into the REQUIRED key sets. A missing key is also recorded as MISSING and the
//! query raises the typed control token `MissingQuery` through `std::panic::resume_unwind` (the panic hook is bypassed);
//! `CensusHooks::pair` catches exactly that token around ONE scalar epilogue call, counts the pair as suspended and
//! continues with the next pair. Any other panic is resumed. Nothing downstream of a missing value is derived: the
//! suspended scalar contributes no later key and no value. Because later keys (ldexp after exp2, divide after ldexp)
//! depend on hardware outputs, a census with missing keys is a FRONTIER (`complete: false`); closure is iterative.
//!
//! Down (`DecodeKind::Down`) has no intrinsic, but its CPU cases are still run: once the GateUp frontier is closed the
//! census folds every synthetic Down scenario, every real Down scenario and every chain Down scenario (chain inputs from
//! `--census-chain-dir`, validated through `ChainDir::tok_for` against the GateUp H recomputed from the closed table)
//! with the real `fold_twin` and the original Down epilogue, asserting zero hook queries per case. Down folds do not depend
//! on the hooks, so they are deferred while keys are missing. Real chain H/A4 depend on the closed GateUp table and are
//! prepared outside (`--prepare-chain` + the A4 helper): a closed frontier without chains reports `chain_inputs_ready:
//! false` / `needs_chain_preparation` and `complete: false`; rerun with `--census-chain-dir` for the final all-case census.

use crate::epilogue::{self, MulUpStats};
use crate::model::*;
use crate::probe_table::{QualifiedHooks, INTRINSIC_KEYS_SCHEMA};
use crate::scenarios::RealRef;
use crate::scenarios::{self, Scenario};
use crate::{chain, samples};
use hipfire_runtime::mq_i4_reference::{DecodeKind, EpilogueIntrinsics};
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::Path;
use std::time::Instant;

pub const CENSUS_SCHEMA: &str = "consumer24.gfx1100.hook-census.v1";

/// Typed control token: the intrinsic key is not in the qualified table; the scalar epilogue call is suspended.
#[derive(Debug, Clone, Copy)]
pub struct MissingQuery;

#[derive(Default)]
struct KeySets {
    exp2: HashSet<u32>,
    ldexp: HashSet<(u32, i32)>,
    divide: HashSet<(u32, u32)>,
}

impl KeySets {
    fn total(&self) -> usize {
        self.exp2.len() + self.ldexp.len() + self.divide.len()
    }
    /// The `consumer24.gfx1100.intrinsic-keys.v1` document (sorted, deterministic).
    fn keys_json(&self) -> Value {
        let mut e: Vec<u32> = self.exp2.iter().copied().collect();
        e.sort_unstable();
        let mut l: Vec<(u32, i32)> = self.ldexp.iter().copied().collect();
        l.sort_unstable();
        let mut d: Vec<(u32, u32)> = self.divide.iter().copied().collect();
        d.sort_unstable();
        json!({"schema": INTRINSIC_KEYS_SCHEMA, "exp2": e,
               "ldexp": l.iter().map(|(v, x)| json!({"value_bits": v, "exponent": x})).collect::<Vec<_>>(),
               "divide": d.iter().map(|(n, q)| json!({"numerator_bits": n, "denominator_bits": q})).collect::<Vec<_>>()})
    }
    fn counts(&self) -> Value {
        json!({"exp2": self.exp2.len(), "ldexp": self.ldexp.len(), "divide": self.divide.len(), "total": self.total()})
    }
}

pub struct CensusHooks<'a> {
    table: &'a QualifiedHooks,
    required: RefCell<KeySets>,
    missing: RefCell<KeySets>,
    queries: Cell<u64>,
    hits: Cell<u64>,
    misses: Cell<u64>,
    first_missing: RefCell<Option<String>>,
    mul: Cell<MulUpStats>,
}

impl<'a> CensusHooks<'a> {
    pub fn new(table: &'a QualifiedHooks) -> Self {
        Self {
            table,
            required: RefCell::default(),
            missing: RefCell::default(),
            queries: Cell::new(0),
            hits: Cell::new(0),
            misses: Cell::new(0),
            first_missing: RefCell::new(None),
            mul: Cell::new(MulUpStats::default()),
        }
    }

    pub fn queries(&self) -> u64 {
        self.queries.get()
    }

    /// Cumulative MulUp audit over every resolved GateUp pair so far.
    pub fn mul_stats(&self) -> MulUpStats {
        self.mul.get()
    }

    pub fn missing_total(&self) -> usize {
        self.missing.borrow().total()
    }

    fn miss(&self, what: impl FnOnce() -> String) -> ! {
        self.misses.set(self.misses.get() + 1);
        if self.first_missing.borrow().is_none() {
            *self.first_missing.borrow_mut() = Some(what());
        }
        resume_unwind(Box::new(MissingQuery))
    }

    /// One GateUp output pair through the original grouped epilogue (m = 2, one live slot). `true` = fully resolved;
    /// `false` = suspended on a missing key (that scalar produced no value).
    pub fn pair(&self, g: f32, u: f32) -> bool {
        let r = catch_unwind(AssertUnwindSafe(|| {
            epilogue::grouped_prefill_epilogue_audit(DecodeKind::GateUp, 2, &[g, u], &[0], self)
        }));
        match r {
            Ok((_, st)) => {
                let mut t = self.mul.get();
                t += st;
                self.mul.set(t);
                true
            }
            Err(p) => {
                if p.is::<MissingQuery>() {
                    false
                } else {
                    resume_unwind(p)
                }
            }
        }
    }
}

impl EpilogueIntrinsics for CensusHooks<'_> {
    fn exp2(&self, x: f32) -> f32 {
        let k = x.to_bits();
        self.queries.set(self.queries.get() + 1);
        self.required.borrow_mut().exp2.insert(k);
        match self.table.get_exp2(k) {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(r)
            }
            None => {
                self.missing.borrow_mut().exp2.insert(k);
                self.miss(|| format!("exp2 arg={k:#010x}"))
            }
        }
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        let k = x.to_bits();
        self.queries.set(self.queries.get() + 1);
        self.required.borrow_mut().ldexp.insert((k, e));
        match self.table.get_ldexp(k, e) {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(r)
            }
            None => {
                self.missing.borrow_mut().ldexp.insert((k, e));
                self.miss(|| format!("ldexp arg={k:#010x} exponent={e}"))
            }
        }
    }
    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        let k = (numerator.to_bits(), denominator.to_bits());
        self.queries.set(self.queries.get() + 1);
        self.required.borrow_mut().divide.insert(k);
        match self.table.get_divide(k.0, k.1) {
            Some(r) => {
                self.hits.set(self.hits.get() + 1);
                f32::from_bits(r)
            }
            None => {
                self.missing.borrow_mut().divide.insert(k);
                self.miss(|| format!("divide num={:#010x} den={:#010x}", k.0, k.1))
            }
        }
    }
}

pub struct Inputs<'a> {
    pub table: &'a QualifiedHooks,
    pub seed: u32,
    /// `(codec name, sample dir)`: `w2` / `w3` (their `mq4` companion and the mixed scenarios are traversed too) or `mq4`.
    pub sample_dirs: &'a [(String, String)],
    /// `(codec name, prepared chain dir)`, optional: when given and the frontier is closed the chain Down cases are folded.
    pub chains: &'a [(String, String)],
    /// Provenance the prepared chain roots are validated against (same values the device run pins).
    pub probe_sha: Option<String>,
    pub divide_sha: Option<String>,
    pub intrinsic_sha: Vec<String>,
    pub out: &'a str,
    pub head: Value,
}

/// CPU twin dense sums of one GateUp scenario (exactly the device run's `build_dense` with `fold_twin`), then every
/// referenced `(expert, x row)` vector's `(g, u)` pairs through the gfx1100 adapter (original epilogue + MulUp NaN rule).
fn census_scenario(
    hooks: &CensusHooks,
    sc: &Scenario,
    codec: Codec,
    scope: &str,
    source: &str,
) -> Res<(Value, u64)> {
    let t0 = Instant::now();
    let kind = Kind::GateUp;
    let experts = sc.build_experts(codec, kind)?;
    let x = sc.x_set(kind);
    let expert_of = |e: i32| e as usize;
    let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
    let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
        fold_twin(k, e, r, x)
    });
    let (m, oc) = (kind.m(), kind.out_cols());
    let q0 = hooks.queries();
    let mk0 = hooks.missing_total();
    let ms0 = hooks.mul_stats();
    let (mut vectors, mut n_pairs, mut suspended) = (0u64, 0u64, 0u64);
    for v in dense.values() {
        if v.len() != m {
            return Err(format!(
                "{}: dense vector has {} values, expected {m}",
                sc.name,
                v.len()
            ));
        }
        vectors += 1;
        for i in 0..oc {
            n_pairs += 1;
            if !hooks.pair(v[i], v[oc + i]) {
                suspended += 1;
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[census] {scope} {} {} pairs={n_pairs} suspended={suspended} ({secs:.1}s)",
        codec.name(),
        sc.name
    );
    Ok((
        json!({"scope": scope, "source": source, "codec": codec.name(), "kind": "gate_up", "scenario": sc.name, "class": format!("{:?}", sc.class),
               "production": sc.production, "feeds_chain": sc.feeds_chain, "expert_x_vectors": vectors, "pairs": n_pairs, "suspended_pairs": suspended,
               "resolved_pairs": n_pairs - suspended, "epilogue_queries": hooks.queries() - q0, "new_missing_keys": hooks.missing_total() - mk0,
               "mul_up": hooks.mul_stats().since(&ms0).to_json(), "seconds": secs}),
        suspended,
    ))
}

/// CPU twin of one Down scenario (real `fold_twin` dense sums, the chain input when `sc.chain_x` is set) through the
/// gfx1100 adapter's Down path (the original `grouped_prefill_epilogue_with(Down, ...)`) via `assemble`; the epilogue must make ZERO hook queries.
fn census_down(
    hooks: &CensusHooks,
    sc: &Scenario,
    codec: Codec,
    scope: &str,
    source: &str,
) -> Res<Value> {
    let t0 = Instant::now();
    let kind = Kind::Down;
    let experts = sc.build_experts(codec, kind)?;
    let x = sc.x_set(kind);
    let expert_of = |e: i32| e as usize;
    let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
    let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
        fold_twin(k, e, r, x)
    });
    let q0 = hooks.queries();
    let exp = assemble(kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false);
    if hooks.queries() != q0 {
        return Err(format!("{scope} {}: the Down epilogue queried an intrinsic ({} queries); the Down-needs-no-hooks premise is false", sc.name, hooks.queries() - q0));
    }
    let live = exp.row_class.iter().filter(|&&c| c == 0).count();
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[census] {scope} {} {} down live_rows={live} queries=0 ({secs:.1}s)",
        codec.name(),
        sc.name
    );
    Ok(
        json!({"scope": scope, "source": source, "codec": codec.name(), "kind": "down", "scenario": sc.name, "class": format!("{:?}", sc.class),
              "production": sc.production, "chain_input": sc.chain_x.is_some(), "expert_x_vectors": dense.len(), "live_rows": live, "pairs": 0, "suspended_pairs": 0,
              "new_missing_keys": 0, "epilogue_queries": 0, "down_hook_free": true, "seconds": secs}),
    )
}

struct Acc {
    scenarios: Vec<Value>,
    chain_sources: Vec<Value>,
    pairs: u64,
    suspended: u64,
}

impl Acc {
    fn gate_up(
        &mut self,
        hooks: &CensusHooks,
        sc: &Scenario,
        codec: Codec,
        scope: &str,
        source: &str,
        chain_name: Option<(&str, String, &str)>,
    ) -> Res<()> {
        let (v, sus) = census_scenario(hooks, sc, codec, scope, source)?;
        self.pairs += v["pairs"].as_u64().unwrap_or(0);
        self.suspended += sus;
        if let Some((mode, name, dir)) = chain_name {
            self.chain_sources.push(json!({"name": name, "mode": mode, "scenario": sc.name, "codec": codec.name(), "scope": scope, "sample_dir": dir,
                "suspended_pairs": sus, "census_resolved": sus == 0,
                "state": if sus == 0 { "GateUp H keys closed under this table; re-prepare after closure (hooks changed)" } else { "blocked: unresolved GateUp keys; no H, no chain, no A4 can be formed yet" },
                "requires_reprep_after_closure": true}));
        }
        self.scenarios.push(v);
        Ok(())
    }
}

/// Standalone-fixture census: the exact `fixture_values` / slots / pass list of `run_fixture`. GateUp passes go pair by
/// pair through the original epilogue; Down passes must make ZERO hook queries.
fn census_fixture(hooks: &CensusHooks, seed: u32, acc: &mut Acc) -> Res<()> {
    let finite = crate::finite_bf16_list();
    for kind in [Kind::GateUp, Kind::Down] {
        let (m, oc) = (kind.m(), kind.out_cols());
        let p_rows = crate::fixture_p_rows(kind);
        for (pi, pass) in crate::FIXTURE_PASSES.iter().enumerate() {
            let t0 = Instant::now();
            let sums = crate::fixture_values(kind, pass, p_rows, seed ^ (pi as u32 * 977), &finite);
            let slots = crate::fixture_slots(p_rows);
            let q0 = hooks.queries();
            let mk0 = hooks.missing_total();
            let ms0 = hooks.mul_stats();
            let (mut n_pairs, mut suspended) = (0u64, 0u64);
            match kind {
                Kind::GateUp => {
                    for (p, &s) in slots.iter().enumerate() {
                        if s < 0 {
                            continue;
                        }
                        for i in 0..oc {
                            n_pairs += 1;
                            if !hooks.pair(sums[p * m + i], sums[p * m + oc + i]) {
                                suspended += 1;
                            }
                        }
                    }
                }
                Kind::Down => {
                    let _ = epilogue::grouped_prefill_epilogue(
                        DecodeKind::Down,
                        m,
                        &sums,
                        &slots,
                        hooks,
                    );
                    if hooks.queries() != q0 {
                        return Err(format!("fixture down/{pass}: the Down epilogue queried an intrinsic ({} queries); the Down-needs-no-hooks premise is false", hooks.queries() - q0));
                    }
                }
            }
            acc.pairs += n_pairs;
            acc.suspended += suspended;
            let tag = if kind == Kind::GateUp {
                "gate_up"
            } else {
                "down"
            };
            eprintln!(
                "[census] fixture {tag} {pass} pairs={n_pairs} suspended={suspended} ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
            acc.scenarios.push(json!({"scope": "fixture", "source": "standalone_epilogue_fixture", "codec": Value::Null, "kind": tag, "scenario": format!("fixture_{tag}_{pass}"),
                "class": if *pass == "exceptional_inputs" { "Exceptional" } else { "Finite" }, "grouped_rows": p_rows, "pairs": n_pairs, "suspended_pairs": suspended,
                "resolved_pairs": n_pairs - suspended, "epilogue_queries": hooks.queries() - q0, "down_hook_free": kind == Kind::Down && hooks.queries() == q0,
                "new_missing_keys": hooks.missing_total() - mk0, "mul_up": hooks.mul_stats().since(&ms0).to_json(),
                "seconds": t0.elapsed().as_secs_f64()}));
        }
    }
    Ok(())
}

struct ChainScope {
    mode: &'static str,
    /// Codec named in the chain entry (`{mode}-{codec}-{scenario}`): the run codec, the cold codec in mixed mode.
    chain_codec: Codec,
    dir: String,
    /// `(codec, manifest sha256)` pairs a prepared chain root must carry; the first is `chain_codec`'s.
    manifests: Vec<(String, String)>,
    downs: HashMap<(u64, u64, Codec), RealRef>,
}

impl ChainScope {
    fn scope_name(&self) -> String {
        format!(
            "down/chain/{}/{}/{}",
            self.chain_codec.name(),
            self.mode,
            self.dir
        )
    }
}

struct Scope {
    name: String,
    source: &'static str,
    build_codec: Codec,
    gate_up: Vec<Scenario>,
    down: Vec<Scenario>,
    chain: Option<ChainScope>,
}

/// Every planned scope of the G1 device run, built with the same scenario functions and sample sets.
fn plan(inp: &Inputs, sample_sets: &mut Vec<Value>) -> Res<Vec<Scope>> {
    let mut v: Vec<Scope> = Vec::new();
    let have_e8p = inp.sample_dirs.iter().any(|(k, _)| k == "e8p");
    let have_legacy = inp.sample_dirs.iter().any(|(k, _)| k == "w2" || k == "w3") || !have_e8p;
    // the e8p profile still runs the (candidate-independent, already table-closed) mq4 synthetic scope: the reviewed
    // MulUp/epilogue keys are shared, and the hot MQ4 kernel of every mixed run is part of the G1 matrix
    let mut synth: Vec<Codec> = if have_legacy {
        vec![Codec::Mq4, Codec::W2, Codec::W3]
    } else {
        vec![Codec::Mq4]
    };
    if have_e8p {
        synth.push(Codec::E8p);
    }
    for codec in synth {
        let list = scenarios::synthetic(inp.seed);
        v.push(Scope {
            name: format!("synthetic/{}", codec.name()),
            source: "synthetic",
            build_codec: codec,
            gate_up: list.clone(),
            down: list,
            chain: None,
        });
    }
    for (cname, dir) in inp.sample_dirs {
        let codec = Codec::parse(cname)?;
        let dpath = Path::new(dir);
        let set = samples::load(dpath, codec)?;
        sample_sets.push(json!({"codec": codec.name(), "sample_dir": dir, "manifest_sha256": set.manifest_sha256, "provisional": set.provisional,
            "artifact_sha256": set.artifact_sha256, "meta": set.meta}));
        let hot = if codec == Codec::Mq4 {
            None
        } else {
            Some(samples::load(dpath, Codec::Mq4)?)
        };
        if let Some(h) = &hot {
            sample_sets.push(json!({"codec": "mq4", "sample_dir": dir, "manifest_sha256": h.manifest_sha256, "provisional": h.provisional,
                "artifact_sha256": h.artifact_sha256, "meta": h.meta}));
        }
        let mut downs: HashMap<(u64, u64, Codec), RealRef> = HashMap::new();
        for r in set
            .down
            .iter()
            .chain(hot.iter().flat_map(|h| h.down.iter()))
        {
            downs.insert((r.layer, r.expert, r.codec), r.clone());
        }
        let mut manifests = vec![(codec.name().to_string(), set.manifest_sha256.clone())];
        if let Some(h) = &hot {
            manifests.push(("mq4".to_string(), h.manifest_sha256.clone()));
        }
        v.push(Scope {
            name: format!("real/{}/{}", codec.name(), dir),
            source: "real",
            build_codec: codec,
            gate_up: scenarios::real(&set.gate_up, inp.seed),
            down: scenarios::real(&set.down, inp.seed),
            chain: Some(ChainScope {
                mode: "percodec",
                chain_codec: codec,
                dir: dir.clone(),
                manifests: manifests.clone(),
                downs: downs.clone(),
            }),
        });
        if let Some(h) = &hot {
            v.push(Scope {
                name: format!("real/mq4/{dir}"),
                source: "real",
                build_codec: Codec::Mq4,
                gate_up: scenarios::real(&h.gate_up, inp.seed),
                down: scenarios::real(&h.down, inp.seed),
                chain: Some(ChainScope {
                    mode: "percodec",
                    chain_codec: Codec::Mq4,
                    dir: dir.clone(),
                    manifests: vec![("mq4".to_string(), h.manifest_sha256.clone())],
                    downs: downs.clone(),
                }),
            });
            v.push(Scope {
                name: format!("mixed/{}/{dir}", codec.name()),
                source: "mixed",
                build_codec: Codec::Mq4,
                gate_up: scenarios::mixed(&set.gate_up, &h.gate_up, inp.seed),
                down: Vec::new(),
                chain: Some(ChainScope {
                    mode: "mixed",
                    chain_codec: codec,
                    dir: dir.clone(),
                    manifests,
                    downs,
                }),
            });
        }
    }
    Ok(v)
}

/// CPU GateUp expectation (the chain's source H): same dense fold and original grouped epilogue as `assemble` in the
/// device run, hooks from the qualified table. Only called once the frontier is closed.
fn gate_up_expectation(hooks: &CensusHooks, sc: &Scenario, codec: Codec) -> Res<Expectation> {
    let kind = Kind::GateUp;
    let experts = sc.build_experts(codec, kind)?;
    let x = sc.x_set(kind);
    let expert_of = |e: i32| e as usize;
    let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
    let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
        fold_twin(k, e, r, x)
    });
    Ok(assemble(
        kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false,
    ))
}

/// Exact text of the `incomplete_reasons` entry when no `--census-chain-dir` was given at a closed frontier.
pub const REASON_CHAINS_ABSENT: &str = "chain inputs absent: pass --census-chain-dir mq4=DIR --census-chain-dir w2=DIR --census-chain-dir w3=DIR (chains prepared with the closed tables and the A4 helper)";

/// Writes the census (exit code 0 whenever the file is written; read `complete`).
///
/// Phase A (always): every GateUp scenario of every scope + the fixture, pair by pair (frontier discovery).
/// Phase B (only when the frontier is closed, i.e. `hooks_closed`): the Down CPU twin folds (synthetic, real) through the
/// original Down epilogue with a zero-query assertion (Down folds do not depend on the hooks, so they are deferred
/// instead of repeated on every frontier round).
/// Phase C (only when closed and `--census-chain-dir` given): for every prepared chain, the GateUp expectation is
/// recomputed from the closed table, validated against the chain entry (`ChainDir::tok_for`: H bytes, hashes, rows,
/// seeds, tiles) and the resulting chain Down scenario is folded and run through the Down epilogue (zero queries).
/// `complete` = frontier closed, every REQUIRED scope (`required_scopes`) complete, and both `w2` and `w3` sample dirs given.
pub fn run(inp: Inputs) -> Res<i32> {
    let hooks = CensusHooks::new(inp.table);
    let mut acc = Acc {
        scenarios: Vec::new(),
        chain_sources: Vec::new(),
        pairs: 0,
        suspended: 0,
    };
    let t_all = Instant::now();
    let mut sample_sets: Vec<Value> = Vec::new();
    let scopes = plan(&inp, &mut sample_sets)?;

    // prepared chain directories (a directory that cannot load yet is a state, not an error)
    let mut chain_dirs: Vec<(Codec, chain::ChainDir)> = Vec::new();
    let mut chain_dir_info: Vec<Value> = Vec::new();
    for (c, d) in inp.chains {
        let codec = Codec::parse(c)?;
        match chain::ChainDir::load(Path::new(d)) {
            Ok(cd) => {
                let rc = cd
                    .root
                    .get("codec")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if rc.as_deref() != Some(codec.name()) {
                    chain_dir_info.push(json!({"codec": c, "dir": d, "loaded": false, "error": format!("chain root codec {rc:?} != {c}")}));
                } else {
                    chain_dir_info.push(
                        json!({"codec": c, "dir": d, "loaded": true, "names": cd.names().len()}),
                    );
                    chain_dirs.push((codec, cd));
                }
            }
            Err(e) => {
                chain_dir_info.push(json!({"codec": c, "dir": d, "loaded": false, "error": e}))
            }
        }
    }
    // chain dir matching each scope (codec + sample manifest) and whether the scope's chain is required; an mq4 real
    // scope is only required when it is the one the given mq4 chain belongs to (or no mq4 chain was given at all)
    let mq4_chain_given = inp.chains.iter().any(|(c, _)| c == "mq4");
    let chain_plan: Vec<Option<(Option<usize>, bool)>> = scopes
        .iter()
        .map(|s| {
            s.chain.as_ref().map(|cs| {
                let found = chain_dirs.iter().position(|(c, cd)| {
                    *c == cs.chain_codec
                        && cd
                            .root
                            .get("sample_manifest_sha256")
                            .and_then(|m| m.get(cs.chain_codec.name()))
                            .and_then(Value::as_str)
                            == Some(cs.manifests[0].1.as_str())
                });
                let required =
                    !(s.name.starts_with("real/mq4/") && mq4_chain_given && found.is_none());
                (found, required)
            })
        })
        .collect();

    // required scopes (names the caller must find complete) and their expected case counts
    let mut required_scopes: Vec<String> = Vec::new();
    let mut expected: HashMap<String, usize> = HashMap::new();
    for s in &scopes {
        required_scopes.push(s.name.clone());
        expected.insert(s.name.clone(), s.gate_up.len());
    }
    required_scopes.push("fixture".into());
    expected.insert("fixture".into(), crate::FIXTURE_PASSES.len() * 2);
    for s in &scopes {
        if !s.down.is_empty() {
            required_scopes.push(format!("down/{}", s.name));
            expected.insert(format!("down/{}", s.name), s.down.len());
        }
    }
    for (s, cp) in scopes.iter().zip(&chain_plan) {
        if let (Some(cs), Some((_, true))) = (&s.chain, cp) {
            required_scopes.push(cs.scope_name());
            expected.insert(
                cs.scope_name(),
                s.gate_up
                    .iter()
                    .filter(|g| g.feeds_chain && !g.production)
                    .count(),
            );
        }
    }

    // Phase A: GateUp keys
    for sc_scope in &scopes {
        for sc in &sc_scope.gate_up {
            let chain_name = match &sc_scope.chain {
                Some(cs) if sc.feeds_chain && !sc.production => Some((
                    cs.mode,
                    chain::chain_name(cs.mode, cs.chain_codec, &sc.name),
                    cs.dir.as_str(),
                )),
                _ => None,
            };
            acc.gate_up(
                &hooks,
                sc,
                sc_scope.build_codec,
                &sc_scope.name,
                sc_scope.source,
                chain_name,
            )?;
        }
    }
    census_fixture(&hooks, inp.seed, &mut acc)?;
    let gate_up_phase_pairs = acc.pairs;
    let hooks_closed = hooks.missing_total() == 0 && acc.suspended == 0;

    // Phase B: Down folds (synthetic + real); Phase C: chain Down folds
    let mut down_done = false;
    let mut needs_prep: Vec<Value> = Vec::new();
    let mut chain_failures: Vec<Value> = Vec::new();
    let mut chain_down_cases = 0u64;
    let mut chain_dirs_used: Vec<Value> = Vec::new();
    if hooks_closed {
        for sc_scope in &scopes {
            for sc in &sc_scope.down {
                let v = census_down(
                    &hooks,
                    sc,
                    sc_scope.build_codec,
                    &format!("down/{}", sc_scope.name),
                    &format!("{}_down", sc_scope.source),
                )?;
                acc.scenarios.push(v);
            }
        }
        down_done = true;
        for (sc_scope, cp) in scopes.iter().zip(&chain_plan) {
            let (Some(cs), Some((found, required))) = (&sc_scope.chain, cp) else {
                continue;
            };
            let Some(idx) = found else {
                if *required {
                    needs_prep.push(json!({"scope": cs.scope_name(), "mode": cs.mode, "codec": cs.chain_codec.name(), "sample_dir": cs.dir,
                        "reason": "no prepared chain dir matches this scope (codec + sample manifest); run --prepare-chain with the closed tables and the A4 helper, then pass --census-chain-dir CODEC=DIR"}));
                }
                continue;
            };
            let cd = &chain_dirs[*idx].1;
            let exp_root = chain::RootExpect {
                source_sha256: crate::source_hashes(),
                probe_table_sha256: inp.probe_sha.clone(),
                divide_table_sha256: inp.divide_sha.clone(),
                intrinsic_tables_sha256: inp.intrinsic_sha.clone(),
                hooks: chain::HOOKS_QUALIFIED,
                sample_manifests: cs.manifests.clone(),
                seed: inp.seed,
                oracle_binary_sha256: crate::exe_sha(),
            };
            let problems = cd.root_problems(&exp_root);
            if !problems.is_empty() {
                needs_prep.push(json!({"scope": cs.scope_name(), "mode": cs.mode, "codec": cs.chain_codec.name(), "sample_dir": cs.dir,
                    "reason": "prepared chain root does not match this oracle/tables (re-prepare)", "problems": problems}));
                continue;
            }
            chain_dirs_used.push(json!({"scope": cs.scope_name(), "dir": cs.dir, "chain_codec": cs.chain_codec.name()}));
            for sc in sc_scope
                .gate_up
                .iter()
                .filter(|s| s.feeds_chain && !s.production)
            {
                let name = chain::chain_name(cs.mode, cs.chain_codec, &sc.name);
                let exp = gate_up_expectation(&hooks, sc, sc_scope.build_codec)?;
                let tok = cd.tok_for(&name, &exp.h, &exp.row_class, &chain::tiles_sha(&sc.tiles));
                match tok.and_then(|t| scenarios::chain_down(sc, &cs.downs, t)) {
                    Ok(Some(ds)) => {
                        let v = census_down(&hooks, &ds, sc_scope.build_codec, &cs.scope_name(), "chain_down")?;
                        chain_down_cases += 1;
                        acc.scenarios.push(v);
                    }
                    Ok(None) => chain_failures.push(json!({"scope": cs.scope_name(), "chain": name, "status": "skipped: an expert has no Down tensor in the sample set"})),
                    Err(e) => chain_failures.push(json!({"scope": cs.scope_name(), "chain": name, "error": e})),
                }
            }
        }
    }
    let chain_inputs_ready =
        hooks_closed && needs_prep.is_empty() && chain_failures.is_empty() && chain_down_cases > 0;

    let required = hooks.required.borrow();
    let missing = hooks.missing.borrow();
    let have = |c: &str| inp.sample_dirs.iter().any(|(k, _)| k == c);
    // legacy profile (mq4/w2/w3 scopes): both cold sample dirs. e8p profile (any `e8p=DIR`): the e8p dirs and only the
    // e8p/mq4 synthetic + real + mixed scopes; w2/w3 are then optional extras but, when one is given, both are required.
    let scopes_complete = if have("e8p") {
        (!have("w2") && !have("w3")) || (have("w2") && have("w3"))
    } else {
        have("w2") && have("w3")
    };

    // per-scope roll-up: every required scope, in order (an absent / partial scope is reported incomplete, never dropped)
    let mut scope_rows: Vec<Value> = Vec::new();
    let mut scope_problems: Vec<String> = Vec::new();
    for n in &required_scopes {
        let rows: Vec<&Value> = acc
            .scenarios
            .iter()
            .filter(|s| s["scope"].as_str() == Some(n.as_str()))
            .collect();
        let sum = |k: &str| rows.iter().map(|r| r[k].as_u64().unwrap_or(0)).sum::<u64>();
        let sus = sum("suspended_pairs");
        let want = expected.get(n).copied().unwrap_or(0);
        let complete_s = rows.len() == want && want > 0 && sus == 0;
        let reason = if complete_s {
            Value::Null
        } else if sus > 0 {
            json!(format!("{sus} suspended pairs (missing keys)"))
        } else if !hooks_closed && n.starts_with("down/") {
            json!("deferred until every GateUp key is closed")
        } else {
            json!(format!("{} of {want} cases ran", rows.len()))
        };
        if !complete_s && hooks_closed {
            scope_problems.push(format!("required scope {n} incomplete: {reason}"));
        }
        scope_rows.push(json!({"name": n, "cases": rows.len(), "expected_cases": want, "pairs": sum("pairs"), "suspended_pairs": sus, "complete": complete_s,
            "reason": reason, "missing_count": sum("new_missing_keys"),
            "missing_count_meaning": "distinct missing keys first seen in this scope (a key shared with an earlier scope is counted there)"}));
    }
    let mut reasons: Vec<String> = Vec::new();
    if missing.total() > 0 {
        reasons.push(format!("{} intrinsic keys missing from the qualified table (probe them on the device, merge, rerun)", missing.total()));
    }
    let mul = hooks.mul_stats();
    if mul.nonnan_changed > 0 {
        reasons.push(format!("MulUp adapter audit FAILED: {} outputs with a non-NaN host product differ from the public twin (must be 0)", mul.nonnan_changed));
    }
    if !scopes_complete {
        reasons.push("--census-sample-dir w2=DIR and w3=DIR are both required for a complete census (an e8p-only census needs --census-sample-dir e8p=DIR)".into());
    }
    if !hooks_closed {
        reasons.push(
            "frontier: Down folds and chain Down folds run only once every GateUp key is closed"
                .into(),
        );
    } else {
        if inp.chains.is_empty() {
            reasons.push(REASON_CHAINS_ABSENT.into());
        } else {
            for n in &needs_prep {
                reasons.push(format!(
                    "chain scope {}: {}",
                    n["scope"].as_str().unwrap_or(""),
                    n["reason"].as_str().unwrap_or("")
                ));
            }
            for f in &chain_failures {
                reasons.push(format!(
                    "chain scope {}: {} {}",
                    f["scope"].as_str().unwrap_or(""),
                    f["chain"].as_str().unwrap_or(""),
                    f["error"].as_str().or(f["status"].as_str()).unwrap_or("")
                ));
            }
        }
        reasons.extend(scope_problems);
    }
    let complete = reasons.is_empty() && hooks_closed && scopes_complete && down_done;
    let required_path = format!("{}.required-keys.json", inp.out);
    let missing_path = format!("{}.missing-keys.json", inp.out);
    let write = |p: &str, v: &Value| -> Res<()> {
        std::fs::write(p, serde_json::to_vec_pretty(v).map_err(err)?)
            .map_err(|e| format!("{p}: {e}"))
    };
    write(&required_path, &required.keys_json())?;
    write(&missing_path, &missing.keys_json())?;
    let mut doc = inp.head;
    doc["schema"] = json!(CENSUS_SCHEMA);
    doc["complete"] = json!(complete);
    doc["hooks_closed"] = json!(hooks_closed);
    doc["incomplete_reasons"] = json!(reasons);
    doc["missing_count"] = json!(missing.total());
    doc["missing_counts"] = missing.counts();
    doc["required_counts"] = required.counts();
    doc["first_missing"] = json!(hooks.first_missing.borrow().clone());
    doc["keys"] = required.keys_json();
    doc["missing"] = missing.keys_json();
    doc["keys_note"] = json!("`keys` = FULL required set (every query that was made, known or missing); `missing` = the frontier (same intrinsic-keys.v1 shape; this is what the device observer probes). Keys downstream of a suspended scalar are unknown until the frontier is resolved.");
    doc["keys_file"] = json!(required_path);
    doc["missing_file"] = json!(missing_path);
    doc["queries"] = json!({"total": hooks.queries.get(), "table_hits": hooks.hits.get(), "table_misses": hooks.misses.get(),
        "dedup": "required/missing key sets are deduplicated by exact operand bits; every output pair is still evaluated (no pair-level dedup)",
        "audit_double_call": "every resolved GateUp pair makes TWO public-twin calls with the same hooks (the adapter's capture call and the audit-only direct `grouped_prefill_epilogue_with` call), so total/table_hits count 2 queries per intrinsic per pair; the key sets are unaffected"});
    doc["epilogue_pairs"] = json!({"total": gate_up_phase_pairs, "suspended": acc.suspended, "resolved": gate_up_phase_pairs - acc.suspended,
        "note": "a suspended pair hit a missing key at its FIRST unresolved query; keys downstream of it are unknown until the frontier is probed"});
    doc["mul_up_audit"] = json!({"totals": mul.to_json(), "nonnan_differences_vs_public_twin": mul.nonnan_changed, "audit_pass": mul.nonnan_changed == 0,
        "scope": "every resolved GateUp pair of every GateUp scenario and fixture pass (suspended pairs excluded); Down is delegated to the public twin",
        "rule": "q NaN -> q|0x00400000; else u NaN -> u|0x00400000; else 0*inf/inf*0 -> 0xffc00000 (device pin); else the public twin's output",
        "changed_nan_pairs_note": "NaN-result outputs whose BF16 differs from the DIRECT public twin's own output (an audit-only extra public call with the same hooks); q_nan/u_nan/invalid_zero_inf count NaN-result outputs by branch; capture_vs_direct_pre_correction counts cells where the capture run before correction already differs from the direct call"});
    doc["scopes_requested"] = json!(inp
        .sample_dirs
        .iter()
        .map(|(c, d)| json!({"codec": c, "sample_dir": d}))
        .collect::<Vec<_>>());
    doc["required_scopes"] = json!(required_scopes);
    doc["scopes"] = Value::Array(scope_rows);
    doc["sample_sets"] = Value::Array(sample_sets);
    doc["scenarios"] = Value::Array(acc.scenarios);
    doc["chain_sources"] = Value::Array(acc.chain_sources);
    doc["down_folds"] = json!({"done": down_done, "state": if down_done { "ran: synthetic + real + chain Down CPU folds through the original Down epilogue, zero hook queries asserted per case" } else { "deferred until hooks_closed (Down folds do not depend on the hooks)" }});
    doc["chain_inputs_ready"] = json!(chain_inputs_ready);
    doc["needs_chain_preparation"] = json!(needs_prep);
    doc["chain_stage"] = json!({"ready": complete, "hooks_closed": hooks_closed, "chain_inputs_ready": chain_inputs_ready, "chain_down_cases_folded": chain_down_cases,
        "chain_dirs_given": chain_dir_info, "chain_dirs_validated": chain_dirs_used, "chain_failures": chain_failures,
        "state": if !hooks_closed { "blocked_until_hooks_closed" } else if complete { "ready" } else { "needs_chain_preparation" },
        "note": "`ready` = the whole census is complete. Chain H/A4 depend on the closed GateUp table: when hooks_closed and `needs_chain_preparation`/chain_failures is non-empty, run --prepare-chain (every codec/sample dir) with the closed tables and the A4 helper, then rerun the census with --census-chain-dir CODEC=DIR"});
    doc["seconds"] = json!(t_all.elapsed().as_secs_f64());
    doc["exit_codes"] = json!({"0": "census file written (read `complete`)", "1": "harness error"});
    write(inp.out, &doc)?;
    println!(
        "{}",
        json!({"schema": CENSUS_SCHEMA, "out": inp.out, "complete": complete, "hooks_closed": hooks_closed, "chain_inputs_ready": chain_inputs_ready,
               "missing_count": missing.total(), "required_count": required.total(), "first_missing": hooks.first_missing.borrow().clone(),
               "missing_file": missing_path, "keys_file": required_path, "suspended_pairs": acc.suspended, "pairs": gate_up_phase_pairs, "mul_up_audit": mul.to_json(),
               "needs_chain_preparation": needs_prep.len(), "reasons": reasons})
    );
    Ok(0)
}
