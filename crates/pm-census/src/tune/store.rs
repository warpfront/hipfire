// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R5 evidence-keyed result store: append-only JSONL of [`TuneRecordV1`], a
//! deterministic [`TableV1`] export, and exact-key `lookup`.
//!
//! * No runtime reads this today; there is no startup benchmarking.
//! * An unrepresented key is a miss and the caller keeps the existing incumbent
//!   selector. Lookup is exact key equality, never nearest/prefix.
//! * Malformed lines are refusals (reported in `malformed`), never hits.
//! * If two promoted records share a key, the one appearing LAST in the file wins
//!   (later evidence supersedes earlier).

use super::{
    canonical_sha256, ProductSelectV1, PromotedVariantV1, Refusal, TableV1, TuneKeyV1,
    TuneRecordV1, RECORD_SCHEMA, TABLE_SCHEMA,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// Append one canonical JSON line (file created if missing).
pub fn append(path: &Path, rec: &TuneRecordV1) -> Result<(), String> {
    let v = serde_json::to_value(rec).map_err(|e| format!("serialize record: {e}"))?;
    let mut line = crate::noise::canonical(&v);
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    f.write_all(line.as_bytes())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate(rec: &TuneRecordV1) -> Result<(), String> {
    if rec.schema != RECORD_SCHEMA {
        return Err(format!("schema {:?}, expected {RECORD_SCHEMA}", rec.schema));
    }
    let k = &rec.key;
    for (n, v) in [
        ("arch", &k.arch),
        ("kernel", &k.kernel),
        ("shape", &k.shape),
        ("numeric_contract", &k.numeric_contract),
        ("mode", &k.mode),
    ] {
        if v.is_empty() {
            return Err(format!("key.{n} is empty"));
        }
    }
    for (n, v) in [
        ("baseline_code_sha256", &k.baseline_code_sha256),
        ("candidate_co_sha256", &k.candidate_co_sha256),
        ("candidate_code_sha256", &k.candidate_code_sha256),
        ("pm_loop_sha256", &k.pm_loop_sha256),
    ] {
        if !is_hex(v, 64) {
            return Err(format!("key.{n} is not 64 lowercase hex"));
        }
    }
    if !is_hex(&k.commit, 40) {
        return Err("key.commit is not 40 lowercase hex".to_owned());
    }
    Ok(())
}

/// Parse a store: `(records, malformed "line N: err")`.
pub fn import(text: &str) -> (Vec<TuneRecordV1>, Vec<String>) {
    let mut recs = Vec::new();
    let mut bad = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let n = i + 1;
        match serde_json::from_str::<TuneRecordV1>(line) {
            Err(e) => bad.push(format!("line {n}: {e}")),
            Ok(r) => match validate(&r) {
                Ok(()) => recs.push(r),
                Err(e) => bad.push(format!("line {n}: {e}")),
            },
        }
    }
    (recs, bad)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `M=..,N=..,K=..` -> `(m, k)`.
fn parse_mk(shape: &str) -> Result<(u64, u64), String> {
    let (mut m, mut k) = (None, None);
    for part in shape.split(',') {
        if let Some((name, val)) = part.split_once('=') {
            let v = val.trim().parse::<u64>().ok();
            match name.trim().to_ascii_uppercase().as_str() {
                "M" => m = v,
                "K" => k = v,
                _ => {}
            }
        }
    }
    match (m, k) {
        (Some(m), Some(k)) => Ok((m, k)),
        _ => Err(format!("shape {shape:?} has no parseable M=..,K=.. pair")),
    }
}

/// Find a `*.co` under `dir` (depth <= 3) whose sha256 is `want`.
fn find_code_object(dir: &Path, want: &str, depth: u32) -> Option<std::path::PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    let mut subdirs = Vec::new();
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if p.extension().is_some_and(|x| x == "co")
            && std::fs::read(&p).is_ok_and(|b| sha256_hex(&b) == want)
        {
            return Some(p);
        }
    }
    if depth > 0 {
        for d in subdirs {
            if let Some(p) = find_code_object(&d, want, depth - 1) {
                return Some(p);
            }
        }
    }
    None
}

/// Build the product selector of one record and copy its code object to
/// `<table_dir>/objects/<sha256>.co`. `Err((true, _))` = unparsable shape
/// (malformed); `Err((false, _))` = export refusal.
fn product_for(rec: &TuneRecordV1, table_dir: &Path) -> Result<ProductSelectV1, (bool, String)> {
    let d = &rec.evidence.discovery;
    let (m, k) = parse_mk(&rec.evidence.candidate.shape).map_err(|e| (true, e))?;
    let symbol = d
        .candidate_launch_symbol
        .clone()
        .ok_or((false, "discovery has no candidate_launch_symbol".to_owned()))?;
    let incumbent_image_sha256 = d
        .baseline_co_sha256
        .clone()
        .ok_or((false, "discovery has no baseline_co_sha256".to_owned()))?;
    let want = &rec.key.candidate_co_sha256;
    let rec_dir = Path::new(&d.record_path).parent().ok_or((
        false,
        format!("record_path {:?} has no directory", d.record_path),
    ))?;
    // `tune run` keeps records in `<out>/records` and `pm_loop --out` objects in
    // `<out>/obj`; a hand-run `pm_loop --out DIR` keeps both in DIR.
    let obj_dir = rec_dir.parent().map(|p| p.join("obj"));
    let src = find_code_object(rec_dir, want, 3)
        .or_else(|| {
            obj_dir
                .as_deref()
                .and_then(|o| find_code_object(o, want, 3))
        })
        .ok_or((
            false,
            format!(
                "no *.co with sha256 {want} under {} or its sibling obj/",
                rec_dir.display()
            ),
        ))?;
    let objects = table_dir.join("objects");
    std::fs::create_dir_all(&objects)
        .map_err(|e| (false, format!("create {}: {e}", objects.display())))?;
    let dst = objects.join(format!("{want}.co"));
    std::fs::copy(&src, &dst).map_err(|e| (false, format!("copy {}: {e}", src.display())))?;
    Ok(ProductSelectV1 {
        symbol,
        arch: rec.key.arch.clone(),
        m,
        k,
        incumbent_image_sha256,
        code_object: format!("objects/{want}.co"),
        code_object_sha256: want.clone(),
    })
}

/// Deterministic table: promoted rows (last record per key wins) sorted by key,
/// refused winners sorted by key, malformed lines, and the rows digest.
/// Rows carry no `product` block (no code-object directory); see [`export_to`].
pub fn export(text: &str) -> TableV1 {
    export_impl(text, None)
}

/// [`export`] with product blocks: each promoted row's code object is copied to
/// `<table_dir>/objects/<sha256>.co`. A row whose code object cannot be found
/// goes to `refused` with stage `export`; an unparsable shape goes to `malformed`.
pub fn export_to(text: &str, table_dir: &Path) -> TableV1 {
    export_impl(text, Some(table_dir))
}

fn export_impl(text: &str, table_dir: Option<&Path>) -> TableV1 {
    let (recs, mut malformed) = import(text);
    let mut rows: BTreeMap<TuneKeyV1, PromotedVariantV1> = BTreeMap::new();
    let mut refused: Vec<(TuneKeyV1, Refusal)> = Vec::new();
    for r in recs {
        match &r.promotion {
            Ok(p) => {
                let product = match table_dir {
                    None => None,
                    Some(dir) => match product_for(&r, dir) {
                        Ok(pr) => Some(pr),
                        Err((true, e)) => {
                            malformed.push(format!("key {}: {e}", r.candidate.id()));
                            rows.remove(&r.key);
                            continue;
                        }
                        Err((false, e)) => {
                            rows.remove(&r.key);
                            refused.push((
                                r.key.clone(),
                                Refusal {
                                    stage: "export".to_owned(),
                                    reason: e,
                                    passed: p.passed.clone(),
                                },
                            ));
                            continue;
                        }
                    },
                };
                rows.insert(
                    r.key.clone(),
                    PromotedVariantV1 {
                        key: r.key.clone(),
                        candidate: r.candidate.clone(),
                        speedup: p.speedup,
                        product,
                    },
                );
            }
            Err(refusal) => {
                if r.verdict == "winner" {
                    refused.push((r.key.clone(), refusal.clone()));
                }
            }
        }
    }
    refused.sort_by(|a, b| a.0.cmp(&b.0));
    let rows: Vec<PromotedVariantV1> = rows.into_values().collect();
    let rows_digest = canonical_sha256(&serde_json::to_value(&rows).expect("rows serialize"));
    TableV1 {
        schema: TABLE_SCHEMA.to_owned(),
        rows,
        refused,
        malformed,
        rows_digest,
        purpose: None,
    }
}

/// Single-row `kld_trial` table for one record (promoted or not), writing
/// `<out_dir>/table.json` and `<out_dir>/objects/<sha256>.co`. The speedup is
/// the inverse of the discovery `block_ratio`.
pub fn trial_table(rec: &TuneRecordV1, out_dir: &Path) -> Result<TableV1, String> {
    let ratio = rec
        .evidence
        .discovery
        .block_ratio
        .filter(|r| *r > 0.0)
        .ok_or("discovery has no usable block_ratio")?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;
    let product = product_for(rec, out_dir).map_err(|(_, e)| e)?;
    let rows = vec![PromotedVariantV1 {
        key: rec.key.clone(),
        candidate: rec.candidate.clone(),
        speedup: 1.0 / ratio,
        product: Some(product),
    }];
    let rows_digest = canonical_sha256(&serde_json::to_value(&rows).expect("rows serialize"));
    let table = TableV1 {
        schema: TABLE_SCHEMA.to_owned(),
        rows,
        refused: Vec::new(),
        malformed: Vec::new(),
        rows_digest,
        purpose: Some("kld_trial".to_owned()),
    };
    write_json(&out_dir.join("table.json"), &table)?;
    Ok(table)
}

/// Pretty JSON + newline.
pub fn write_json<T: serde::Serialize>(path: &Path, v: &T) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(path, s + "\n").map_err(|e| format!("write {}: {e}", path.display()))
}

/// Exact key equality only.
pub fn lookup<'a>(t: &'a TableV1, key: &TuneKeyV1) -> Option<&'a PromotedVariantV1> {
    t.rows
        .binary_search_by(|r| r.key.cmp(key))
        .ok()
        .map(|i| &t.rows[i])
}

/// `pm-census tune export --store FILE [--out FILE]`. With `--out`, rows get
/// product blocks and code objects are copied next to the table.
pub fn cli(args: &[String]) -> Result<(), String> {
    let mut store: Option<&str> = None;
    let mut out: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--store" => store = Some(val),
            "--out" => out = Some(val),
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    let store = store.ok_or("usage: pm-census tune export --store FILE [--out FILE]")?;
    let text = std::fs::read_to_string(store).map_err(|e| format!("read {store}: {e}"))?;
    let table = match out {
        Some(o) => {
            let dir = match Path::new(o).parent() {
                Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
                _ => std::path::PathBuf::from("."),
            };
            std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
            export_to(&text, &dir)
        }
        None => export(&text),
    };
    let v = serde_json::to_value(&table).map_err(|e| format!("serialize table: {e}"))?;
    println!("{}", crate::noise::canonical(&v));
    if let Some(o) = out {
        let pretty = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
        std::fs::write(o, pretty + "\n").map_err(|e| format!("write {o}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{AttemptV1, CandidateSpecV1, ClaimPolicyV1, PromotionV1, TuneEvidenceV1};

    fn h(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn key(tag: char) -> TuneKeyV1 {
        TuneKeyV1 {
            arch: "gfx1100".into(),
            kernel: "gemv".into(),
            shape: "4096x4096".into(),
            numeric_contract: "exact".into(),
            mode: "eager".into(),
            baseline_code_sha256: h('a'),
            candidate_co_sha256: h('b'),
            candidate_code_sha256: h(tag),
            commit: "c".repeat(40),
            pm_loop_sha256: h('d'),
        }
    }

    fn spec() -> CandidateSpecV1 {
        CandidateSpecV1 {
            kernel: "gemv".into(),
            arch: "gfx1100".into(),
            shape: "4096x4096".into(),
            knobs: BTreeMap::new(),
        }
    }

    fn attempt() -> AttemptV1 {
        AttemptV1 {
            candidate_id: "c".into(),
            phase: "discovery".into(),
            replicate: 0,
            record_path: "r".into(),
            exit_code: 0,
            error_stage: None,
            error_message: None,
            m7_status: None,
            correctness_pass: None,
            identical_to_baseline: None,
            baseline_launch_symbol: None,
            candidate_launch_symbol: None,
            baseline_co_sha256: None,
            candidate_co_sha256: None,
            baseline_code_sha256: None,
            candidate_code_sha256: None,
            separable: None,
            direction: None,
            sign_p: None,
            block_ratio: None,
            material: None,
            rounds_run: None,
            noise_floor: None,
            exceeds_noise_floor: None,
            device_uuid: None,
            fabric_job: None,
        }
    }

    fn rec(tag: char, verdict: &str, promotion: Result<PromotionV1, Refusal>) -> TuneRecordV1 {
        TuneRecordV1 {
            schema: RECORD_SCHEMA.into(),
            key: key(tag),
            domain_hash: h('e'),
            candidate: spec(),
            verdict: verdict.into(),
            evidence: TuneEvidenceV1 {
                candidate: spec(),
                discovery: attempt(),
                confirms: vec![],
                noise_floor: None,
                aa_runs: 0,
                kld_state: None,
                kld_token: None,
            },
            promotion,
        }
    }

    fn promoted(tag: char, speedup: f64) -> TuneRecordV1 {
        rec(
            tag,
            "winner",
            Ok(PromotionV1 {
                passed: vec!["m7".into()],
                speedup,
            }),
        )
    }

    fn refusal() -> Refusal {
        Refusal {
            stage: "kld_state".into(),
            reason: "unwired".into(),
            passed: vec![],
        }
    }

    fn text(recs: &[TuneRecordV1]) -> String {
        recs.iter()
            .map(|r| crate::noise::canonical(&serde_json::to_value(r).unwrap()) + "\n")
            .collect()
    }

    #[test]
    fn hit_and_mutation_matrix() {
        let t = export(&text(&[promoted('1', 1.1)]));
        let k = key('1');
        assert_eq!(lookup(&t, &k).unwrap().speedup, 1.1);
        let muts: Vec<Box<dyn Fn(&mut TuneKeyV1)>> = vec![
            Box::new(|k| k.arch.push('x')),
            Box::new(|k| k.kernel.push('x')),
            Box::new(|k| k.shape.push('x')),
            Box::new(|k| k.numeric_contract = "numeric".into()),
            Box::new(|k| k.mode = "graph".into()),
            Box::new(|k| k.baseline_code_sha256 = h('9')),
            Box::new(|k| k.candidate_co_sha256 = h('9')),
            Box::new(|k| k.candidate_code_sha256 = h('9')),
            Box::new(|k| k.commit = "9".repeat(40)),
            Box::new(|k| k.pm_loop_sha256 = h('9')),
        ];
        assert_eq!(muts.len(), 10);
        for (i, m) in muts.iter().enumerate() {
            let mut q = k.clone();
            m(&mut q);
            assert_ne!(q, k, "mutation {i} changed nothing");
            assert!(lookup(&t, &q).is_none(), "mutation {i} hit");
        }
    }

    #[test]
    fn refused_and_not_winner_routing() {
        let t = export(&text(&[
            promoted('1', 1.1),
            rec('2', "winner", Err(refusal())),
            rec('3', "not_winner", Err(refusal())),
        ]));
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.refused.len(), 1);
        assert_eq!(t.refused[0].0, key('2'));
        assert!(lookup(&t, &key('2')).is_none());
        assert!(lookup(&t, &key('3')).is_none());
        assert!(t.malformed.is_empty());
    }

    #[test]
    fn malformed_lines_do_not_affect_rows() {
        let good = text(&[promoted('1', 1.1)]);
        let mut wrong_schema = promoted('2', 1.2);
        wrong_schema.schema = "nope".into();
        let mut bad_commit = promoted('3', 1.3);
        bad_commit.key.commit = "abc".into();
        let mut all = good.clone();
        all.push_str("{\"schema\": trunc\n");
        all.push_str("garbage\n\n");
        all.push_str(&text(&[wrong_schema, bad_commit]));
        let t = export(&all);
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.malformed.len(), 4);
        assert!(t.malformed[0].starts_with("line 2:"));
        assert_eq!(t.rows_digest, export(&good).rows_digest);
    }

    #[test]
    fn deterministic_and_order_independent() {
        let a = promoted('1', 1.1);
        let b = promoted('2', 1.2);
        let r = rec('4', "winner", Err(refusal()));
        let t1 = export(&text(&[a.clone(), b.clone(), r.clone()]));
        let t1b = export(&text(&[a.clone(), b.clone(), r.clone()]));
        let t2 = export(&text(&[r, b, a]));
        let c = |t: &TableV1| crate::noise::canonical(&serde_json::to_value(t).unwrap());
        assert_eq!(c(&t1), c(&t1b));
        assert_eq!(t1.rows_digest, t2.rows_digest);
        assert_eq!(c(&t1), c(&t2));
    }

    #[test]
    fn later_duplicate_supersedes() {
        let t = export(&text(&[promoted('1', 1.1), promoted('1', 1.3)]));
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.rows[0].speedup, 1.3);
    }

    #[test]
    fn append_roundtrip() {
        let p = std::env::temp_dir().join(format!(
            "pm-census-store-test-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        let a = promoted('1', 1.1);
        let b = rec('2', "winner", Err(refusal()));
        append(&p, &a).unwrap();
        append(&p, &b).unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        let (recs, bad) = import(&s);
        assert!(bad.is_empty());
        assert_eq!(recs, vec![a, b]);
        let _ = ClaimPolicyV1::default();
    }

    fn sha_of(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-store-{}-{}-{:?}",
            name,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Promoted record whose discovery record dir holds `co_bytes` as a `.co`
    /// (or nothing when `co_bytes` is None).
    fn product_rec(dir: &Path, co_bytes: Option<&[u8]>, shape: &str) -> TuneRecordV1 {
        let co = b"code-object-bytes";
        let mut r = promoted('1', 1.1);
        r.key.candidate_co_sha256 = sha_of(co);
        r.candidate.shape = shape.into();
        r.evidence.candidate.shape = shape.into();
        let rd = dir.join("rec");
        std::fs::create_dir_all(&rd).unwrap();
        if let Some(b) = co_bytes {
            std::fs::write(rd.join("cand.co"), b).unwrap();
        }
        std::fs::write(rd.join("other.co"), b"unrelated").unwrap();
        let d = &mut r.evidence.discovery;
        d.record_path = rd.join("record.json").display().to_string();
        d.candidate_launch_symbol = Some("sym_b1s".into());
        d.baseline_co_sha256 = Some(h('7'));
        d.block_ratio = Some(0.8);
        r
    }

    #[test]
    fn export_to_fills_product_and_copies_object() {
        let d = tmpdir("prod");
        let co = b"code-object-bytes";
        let r = product_rec(&d, Some(co), "M=17408,N=512,K=5120");
        let out = d.join("table");
        let t = export_to(&text(std::slice::from_ref(&r)), &out);
        assert_eq!(t.rows.len(), 1, "{:?}", t.refused);
        let p = t.rows[0].product.as_ref().unwrap();
        assert_eq!(p.symbol, "sym_b1s");
        assert_eq!(p.arch, "gfx1100");
        assert_eq!((p.m, p.k), (17408, 5120));
        assert_eq!(p.incumbent_image_sha256, h('7'));
        assert_eq!(p.code_object_sha256, sha_of(co));
        assert_eq!(p.code_object, format!("objects/{}.co", sha_of(co)));
        assert_eq!(std::fs::read(out.join(&p.code_object)).unwrap(), co);

        // missing code object -> refused at export
        let d2 = tmpdir("prod-missing");
        let r2 = product_rec(&d2, None, "M=1,N=2,K=3");
        let t2 = export_to(&text(&[r2]), &d2.join("table"));
        assert!(t2.rows.is_empty());
        assert_eq!(t2.refused.len(), 1);
        assert_eq!(t2.refused[0].1.stage, "export");

        // unparsable shape -> malformed, not a row
        let d3 = tmpdir("prod-shape");
        let r3 = product_rec(&d3, Some(co), "4096x4096");
        let t3 = export_to(&text(&[r3]), &d3.join("table"));
        assert!(t3.rows.is_empty());
        assert_eq!(t3.malformed.len(), 1);

        // pure export carries no product block
        assert!(export(&text(&[r])).rows[0].product.is_none());
    }

    #[test]
    fn export_to_finds_object_in_tune_run_sibling_obj_dir() {
        // `tune run` layout: <out>/records/<rec>.json, <out>/obj/<tag>.<sha16>.co
        let d = tmpdir("prod-obj");
        let co = b"tune-run-object";
        let mut r = product_rec(&d, None, "M=17408,N=8192,K=5120");
        r.key.candidate_co_sha256 = sha_of(co);
        std::fs::create_dir_all(d.join("obj/x")).unwrap();
        std::fs::write(d.join("obj/x/candidate.co"), co).unwrap();
        let t = export_to(&text(&[r]), &d.join("table"));
        assert_eq!(t.rows.len(), 1, "{:?}", t.refused);
        let p = t.rows[0].product.as_ref().unwrap();
        assert_eq!(
            std::fs::read(d.join("table").join(&p.code_object)).unwrap(),
            co
        );
    }

    #[test]
    fn trial_table_single_row_even_if_unpromoted() {
        let d = tmpdir("trial");
        let mut r = product_rec(&d, Some(b"code-object-bytes"), "M=64,N=8,K=128");
        r.promotion = Err(refusal());
        r.verdict = "not_winner".into();
        let out = d.join("trial");
        let t = trial_table(&r, &out).unwrap();
        assert_eq!(t.purpose.as_deref(), Some("kld_trial"));
        assert_eq!(t.rows.len(), 1);
        assert!((t.rows[0].speedup - 1.25).abs() < 1e-12);
        let on_disk: TableV1 =
            serde_json::from_str(&std::fs::read_to_string(out.join("table.json")).unwrap())
                .unwrap();
        assert_eq!(on_disk, t);
        assert!(out
            .join(&t.rows[0].product.as_ref().unwrap().code_object)
            .is_file());

        let mut none = r.clone();
        none.evidence.discovery.block_ratio = None;
        assert!(trial_table(&none, &d.join("t2")).is_err());
    }
}
