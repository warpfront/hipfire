// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R7 model-level KLD/state evidence: HFKSEQ v2 reader, paired per-chunk KLD
//! statistics, evidence assembly (re-reading and hash-checking both kldseq
//! files) and the `pm-census tune kld|kld-power|promote|trial-table` CLI.
//!
//! `tune kld` defaults to the R7 per-chunk Student-t statistic (`--stat
//! per-chunk`); `--stat token-bootstrap` runs the R9 token-level statistic of
//! [`super::ktok`] (statistic, gate and the power calculation are documented
//! there) and `tune kld-power` prints the sample-size arithmetic alone.

use super::{
    ktok, promote, store, ClaimPolicyV1, KldRunV1, KldStateEvidenceV1, PairedKldV1, TuneRecordV1,
    RECORD_SCHEMA,
};
use std::path::Path;

const MAGIC: &[u8; 8] = b"HFKSEQ\0\0";
const HEADER: usize = 20;

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Parse HFKSEQ v2 bytes into `(mean_kld, p99_kld, mean_nll)` per chunk.
pub fn parse_kldseq(b: &[u8]) -> Result<Vec<(f64, f64, f64)>, String> {
    if b.len() < HEADER {
        return Err(format!("kldseq too short ({} bytes)", b.len()));
    }
    if &b[..8] != MAGIC {
        return Err("kldseq bad magic (want HFKSEQ\\0\\0)".to_owned());
    }
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
    let version = u32_at(8);
    if version != 2 {
        return Err(format!("kldseq version {version}, want 2"));
    }
    let n = u32_at(12) as usize;
    let want = n
        .checked_mul(24)
        .and_then(|x| x.checked_add(HEADER))
        .ok_or("kldseq n_chunk overflows")?;
    if b.len() != want {
        return Err(format!(
            "kldseq length {} != {want} for n_chunk {n}",
            b.len()
        ));
    }
    let f = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
    Ok((0..n)
        .map(|i| {
            let o = HEADER + i * 24;
            (f(o), f(o + 8), f(o + 16))
        })
        .collect())
}

/// Read and parse an HFKSEQ v2 file.
pub fn read_kldseq(path: &Path) -> Result<Vec<(f64, f64, f64)>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_kldseq(&b).map_err(|e| format!("{}: {e}", path.display()))
}

/// Two-sided 95% Student-t critical value for `df` degrees of freedom.
fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281, 2.2010,
        2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739,
        2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
    ];
    if df <= 30 {
        return T[df - 1];
    }
    // Cornish-Fisher expansion of the t quantile around z = 1.959964.
    let z: f64 = 1.959964;
    let d = df as f64;
    let z3 = z.powi(3);
    let z5 = z.powi(5);
    let z7 = z.powi(7);
    z + (z3 + z) / (4.0 * d)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * d * d)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * d * d * d)
}

/// Paired per-chunk mean-KLD statistics, `cand - inc`, with a 95% two-sided
/// Student-t interval of the mean difference.
pub fn paired(inc: &[(f64, f64, f64)], cand: &[(f64, f64, f64)]) -> Result<PairedKldV1, String> {
    if inc.len() != cand.len() {
        return Err(format!(
            "chunk count differs: incumbent {} vs candidate {}",
            inc.len(),
            cand.len()
        ));
    }
    let n = inc.len();
    if n < 2 {
        return Err(format!("need at least 2 paired chunks, got {n}"));
    }
    let d: Vec<f64> = inc.iter().zip(cand).map(|(i, c)| c.0 - i.0).collect();
    if d.iter().any(|x| !x.is_finite()) {
        return Err("non-finite per-chunk KLD".to_owned());
    }
    let nf = n as f64;
    let mean = d.iter().sum::<f64>() / nf;
    let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    let sd = var.sqrt();
    let half = t_crit_95(n - 1) * sd / nf.sqrt();
    Ok(PairedKldV1 {
        n,
        mean_diff: mean,
        sd_diff: sd,
        ci95_lo: mean - half,
        ci95_hi: mean + half,
    })
}

pub(super) fn read_verified(run: &KldRunV1) -> Result<Vec<u8>, String> {
    let b = std::fs::read(&run.kldseq_path)
        .map_err(|e| format!("read {} ({}): {e}", run.kldseq_path, run.role))?;
    let sha = sha256_hex(&b);
    if sha != run.kldseq_sha256 {
        return Err(format!(
            "{} kldseq sha256 {sha} != recorded {}",
            run.role, run.kldseq_sha256
        ));
    }
    Ok(b)
}

/// Assemble evidence: both kldseq files are re-read and hash-verified against
/// the run JSON; paired statistics and byte-exactness are computed here.
pub fn build_evidence(
    grade: &str,
    inc: KldRunV1,
    cand: KldRunV1,
) -> Result<KldStateEvidenceV1, String> {
    if grade != "screen" && grade != "promotion" {
        return Err(format!("grade {grade:?}, want screen|promotion"));
    }
    let bi = read_verified(&inc)?;
    let bc = read_verified(&cand)?;
    let pi = parse_kldseq(&bi).map_err(|e| format!("incumbent: {e}"))?;
    let pc = parse_kldseq(&bc).map_err(|e| format!("candidate: {e}"))?;
    let paired = paired(&pi, &pc)?;
    Ok(KldStateEvidenceV1 {
        grade: grade.to_owned(),
        incumbent: inc,
        candidate: cand,
        paired,
        exact: bi == bc,
    })
}

fn load<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse {path}: {e}"))
}

fn flags(
    args: &[String],
    allowed: &[&str],
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut m = std::collections::BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        let name = f.strip_prefix("--").filter(|n| allowed.contains(n));
        let Some(name) = name else {
            return Err(format!("unknown argument {f}"));
        };
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("{f} needs a value"))?;
        m.insert(name.to_owned(), v.clone());
        i += 2;
    }
    Ok(m)
}

fn req<'a>(m: &'a std::collections::BTreeMap<String, String>, k: &str) -> Result<&'a str, String> {
    m.get(k)
        .map(String::as_str)
        .ok_or_else(|| format!("--{k} is required"))
}

/// Latest record (file order) whose candidate id is `id`.
fn latest_record(store_path: &str, id: &str) -> Result<TuneRecordV1, String> {
    let text =
        std::fs::read_to_string(store_path).map_err(|e| format!("read {store_path}: {e}"))?;
    let (recs, _) = store::import(&text);
    recs.into_iter()
        .rfind(|r| r.candidate.id() == id)
        .ok_or_else(|| format!("no record for candidate id {id} in {store_path}"))
}

const USAGE: &str = "usage:\n  pm-census tune kld --grade screen|promotion --incumbent RUN.json --candidate RUN.json --out EV.json\n  pm-census tune kld --stat token-bootstrap --grade screen|promotion --stratum LABEL=INC_RUN.json,CAND_RUN.json [--stratum ...] --out EV.json [--margin 0.0005] [--alpha 0.05] [--power 0.8] [--seed N]\n  pm-census tune kld-power --sd SD --margin M [--alpha 0.05] [--power 0.8] [--delta 0] [--tokens-per-chunk 1023] [--available N]\n  pm-census tune trial-table --store S.jsonl --candidate-id ID --out DIR\n  pm-census tune promote --store S.jsonl --candidate-id ID [--kld EV.json] [--kld-token EV.json] [--policy P.json]";

/// Like [`flags`] but a flag may repeat (values kept in order).
fn flags_multi(
    args: &[String],
    allowed: &[&str],
) -> Result<std::collections::BTreeMap<String, Vec<String>>, String> {
    let mut m: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        let name = f.strip_prefix("--").filter(|n| allowed.contains(n));
        let Some(name) = name else {
            return Err(format!("unknown argument {f}"));
        };
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("{f} needs a value"))?;
        m.entry(name.to_owned()).or_default().push(v.clone());
        i += 2;
    }
    Ok(m)
}

/// The single value of a flag that must not repeat.
fn one<'a>(
    m: &'a std::collections::BTreeMap<String, Vec<String>>,
    k: &str,
) -> Result<Option<&'a str>, String> {
    match m.get(k).map(Vec::as_slice) {
        None => Ok(None),
        Some([v]) => Ok(Some(v.as_str())),
        Some(_) => Err(format!("--{k} given more than once")),
    }
}

fn need<'a>(
    m: &'a std::collections::BTreeMap<String, Vec<String>>,
    k: &str,
) -> Result<&'a str, String> {
    one(m, k)?.ok_or_else(|| format!("--{k} is required"))
}

fn num<T: std::str::FromStr>(v: Option<&str>, k: &str, default: T) -> Result<T, String> {
    match v {
        None => Ok(default),
        Some(s) => s.parse().map_err(|_| format!("--{k}: cannot parse {s:?}")),
    }
}

/// `LABEL=INCUMBENT_RUN.json,CANDIDATE_RUN.json`.
fn parse_stratum(s: &str) -> Result<(String, KldRunV1, KldRunV1), String> {
    let (label, paths) = s
        .split_once('=')
        .ok_or_else(|| format!("--stratum {s:?}: want LABEL=INC_RUN.json,CAND_RUN.json"))?;
    let (inc, cand) = paths
        .split_once(',')
        .ok_or_else(|| format!("--stratum {s:?}: want LABEL=INC_RUN.json,CAND_RUN.json"))?;
    Ok((label.to_owned(), load(inc)?, load(cand)?))
}

/// `pm-census tune kld --stat token-bootstrap ...`: build and write the
/// evidence, print the statistic, per-stratum diagnostics, the power
/// calculation at the measured stratified chunk sd, and the gate preview.
fn kld_token_cli(m: &std::collections::BTreeMap<String, Vec<String>>) -> Result<(), String> {
    if m.contains_key("incumbent") || m.contains_key("candidate") {
        return Err(
            "--incumbent/--candidate belong to --stat per-chunk; use --stratum LABEL=INC.json,CAND.json"
                .to_owned(),
        );
    }
    let strata = m
        .get("stratum")
        .ok_or("--stratum is required (repeat once per reference corpus)")?
        .iter()
        .map(|s| parse_stratum(s))
        .collect::<Result<Vec<_>, _>>()?;
    let margin: f64 = num(one(m, "margin")?, "margin", 0.0005)?;
    let alpha: f64 = num(one(m, "alpha")?, "alpha", 0.05)?;
    let power: f64 = num(one(m, "power")?, "power", 0.8)?;
    let seed: u64 = num(one(m, "seed")?, "seed", ktok::DEFAULT_SEED)?;
    let ev = ktok::build_token_evidence(need(m, "grade")?, strata, ktok::BOOT_B, seed)?;
    let tokens_per_chunk = ev.strata[0].tokens_per_chunk;
    let pw = ktok::kld_power(
        ev.stat.sd_chunk,
        margin,
        alpha,
        power,
        0.0,
        tokens_per_chunk,
        Some(ev.stat.n_chunks),
    )?;
    store::write_json(Path::new(need(m, "out")?), &ev)?;
    let gate = ktok::judge_bounds(&ev, margin);
    let strata_json: Vec<serde_json::Value> = ev
        .strata
        .iter()
        .map(|s| {
            serde_json::json!({
                "label": s.label,
                "n_chunks": s.n_chunks,
                "tokens_per_chunk": s.tokens_per_chunk,
                "mean_diff": s.mean_diff,
                "sd_token": s.sd_token,
                "sd_chunk": s.sd_chunk,
                "deff": s.deff,
                "top1pct_abs_share": s.top1pct_abs_share,
                "exact": s.exact,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "stat": ev.stat,
        "strata": strata_json,
        "power": pw,
        "gate": {
            "kld_margin": margin,
            "pass": gate.is_ok(),
            "reason": gate.err(),
        },
    });
    println!(
        "{}",
        serde_json::to_string(&doc).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// `pm-census tune kld|kld-power|trial-table|promote ...` (`args[0]` is the subcommand).
pub fn cli(args: &[String]) -> Result<(), String> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(USAGE.to_owned());
    };
    match sub.as_str() {
        "kld" => {
            let m = flags_multi(
                rest,
                &[
                    "grade",
                    "stat",
                    "incumbent",
                    "candidate",
                    "out",
                    "stratum",
                    "margin",
                    "alpha",
                    "power",
                    "seed",
                ],
            )?;
            match one(&m, "stat")?.unwrap_or("per-chunk") {
                "per-chunk" => {
                    for k in ["stratum", "margin", "alpha", "power", "seed"] {
                        if m.contains_key(k) {
                            return Err(format!("--{k} applies to --stat token-bootstrap only"));
                        }
                    }
                    let ev = build_evidence(
                        need(&m, "grade")?,
                        load(need(&m, "incumbent")?)?,
                        load(need(&m, "candidate")?)?,
                    )?;
                    store::write_json(Path::new(need(&m, "out")?), &ev)?;
                    println!(
                        "{}",
                        serde_json::to_string(&ev.paired).map_err(|e| e.to_string())?
                    );
                    Ok(())
                }
                "token-bootstrap" => kld_token_cli(&m),
                other => Err(format!("--stat {other:?}, want per-chunk|token-bootstrap")),
            }
        }
        "kld-power" => {
            let m = flags_multi(
                rest,
                &[
                    "sd",
                    "margin",
                    "alpha",
                    "power",
                    "delta",
                    "tokens-per-chunk",
                    "available",
                ],
            )?;
            let available = one(&m, "available")?
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| format!("--available: cannot parse {s:?}"))
                })
                .transpose()?;
            let p = ktok::kld_power(
                num(Some(need(&m, "sd")?), "sd", 0.0)?,
                num(Some(need(&m, "margin")?), "margin", 0.0)?,
                num(one(&m, "alpha")?, "alpha", 0.05)?,
                num(one(&m, "power")?, "power", 0.8)?,
                num(one(&m, "delta")?, "delta", 0.0)?,
                num(one(&m, "tokens-per-chunk")?, "tokens-per-chunk", 1023)?,
                available,
            )?;
            println!("{}", serde_json::to_string(&p).map_err(|e| e.to_string())?);
            Ok(())
        }
        "trial-table" => {
            let m = flags(rest, &["store", "candidate-id", "out"])?;
            let rec = latest_record(req(&m, "store")?, req(&m, "candidate-id")?)?;
            let out = Path::new(req(&m, "out")?);
            let t = store::trial_table(&rec, out)?;
            println!("{}", serde_json::to_string(&t).map_err(|e| e.to_string())?);
            Ok(())
        }
        "promote" => {
            let m = flags(
                rest,
                &["store", "candidate-id", "kld", "kld-token", "policy"],
            )?;
            let store_path = req(&m, "store")?;
            let mut rec = latest_record(store_path, req(&m, "candidate-id")?)?;
            let policy: ClaimPolicyV1 = match m.get("policy") {
                Some(p) => load(p)?,
                None => ClaimPolicyV1::default(),
            };
            if let Some(k) = m.get("kld") {
                rec.evidence.kld_state = Some(load(k)?);
            }
            if let Some(k) = m.get("kld-token") {
                rec.evidence.kld_token = Some(load(k)?);
            }
            let ev = &rec.evidence;
            let numeric = std::iter::once(&ev.discovery)
                .chain(ev.confirms.iter())
                .any(|a| a.identical_to_baseline != Some(true));
            if numeric {
                rec.key.numeric_contract = "numeric".to_owned();
            }
            rec.schema = RECORD_SCHEMA.to_owned();
            rec.verdict = if promote::is_winner(&rec.evidence, &policy) {
                "winner"
            } else {
                "not_winner"
            }
            .to_owned();
            rec.promotion = promote::can_promote(&rec.evidence, &policy);
            store::append(Path::new(store_path), &rec)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&rec.promotion).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        _ => Err(USAGE.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    pub(crate) fn kldseq_bytes(means: &[f64]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&(means.len() as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        for m in means {
            v.extend_from_slice(&m.to_le_bytes());
            v.extend_from_slice(&(m * 2.0).to_le_bytes());
            v.extend_from_slice(&1.5f64.to_le_bytes());
        }
        v
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-kld-{}-{}-{:?}",
            name,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub(crate) fn run(role: &str, path: &Path, bytes: &[u8]) -> KldRunV1 {
        std::fs::write(path, bytes).unwrap();
        KldRunV1 {
            role: role.into(),
            model_path: "m.hfq".into(),
            model_sha256: "m".repeat(64),
            reference_path: "r.bin".into(),
            reference_sha256: "r".repeat(64),
            evaluator_path: "eval".into(),
            evaluator_sha256: "e".repeat(64),
            hipfire_commit: "c".repeat(40),
            scoring_mode: "prefill".into(),
            max_chunks: 32,
            kv_mode: "q8".into(),
            kv_v: "q8".into(),
            env: BTreeMap::from([
                ("HIPFIRE_GRAPH".to_owned(), "0".to_owned()),
                ("HIPFIRE_NORMALIZE_PROMPT".to_owned(), "0".to_owned()),
            ]),
            fresh_home: true,
            tune_table_path: None,
            tune_table_sha256: None,
            selected_code_object_sha256: None,
            kldseq_path: path.display().to_string(),
            kldseq_sha256: sha256_hex(bytes),
            ktok_path: None,
            ktok_sha256: None,
            device_uuid: "u".into(),
            fabric_job: Some("j".into()),
            slice_mean_kld: 0.01,
        }
    }

    #[test]
    fn kldseq_round_trip_and_corruption() {
        let b = kldseq_bytes(&[0.1, 0.2, 0.3]);
        let v = parse_kldseq(&b).unwrap();
        assert_eq!(v, vec![(0.1, 0.2, 1.5), (0.2, 0.4, 1.5), (0.3, 0.6, 1.5)]);
        let d = tmp("rt");
        let p = d.join("a.kldseq");
        std::fs::write(&p, &b).unwrap();
        assert_eq!(read_kldseq(&p).unwrap(), v);
        assert!(read_kldseq(&d.join("missing")).is_err());

        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(parse_kldseq(&bad).unwrap_err().contains("magic"));
        let mut bad = b.clone();
        bad[8] = 3;
        assert!(parse_kldseq(&bad).unwrap_err().contains("version"));
        assert!(parse_kldseq(&b[..b.len() - 1])
            .unwrap_err()
            .contains("length"));
        let mut long = b.clone();
        long.push(0);
        assert!(parse_kldseq(&long).unwrap_err().contains("length"));
        assert!(parse_kldseq(&b[..10]).unwrap_err().contains("short"));
        let mut huge = b.clone();
        huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_kldseq(&huge).is_err());
    }

    #[test]
    fn paired_matches_hand_computation() {
        // diffs 1,2,3,4 -> mean 2.5, var 5/3, sd 1.2910, t(3)=3.1824
        let inc: Vec<_> = (0..4).map(|_| (0.0, 0.0, 0.0)).collect();
        let cand: Vec<_> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|m| (*m, 0.0, 0.0))
            .collect();
        let p = paired(&inc, &cand).unwrap();
        assert_eq!(p.n, 4);
        assert!((p.mean_diff - 2.5).abs() < 1e-12);
        assert!((p.sd_diff - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        let half = 3.1824 * (5.0f64 / 3.0).sqrt() / 2.0;
        assert!((p.ci95_hi - (2.5 + half)).abs() < 1e-9);
        assert!((p.ci95_lo - (2.5 - half)).abs() < 1e-9);
        assert!(paired(&inc[..1], &cand[..1]).is_err());
        assert!(paired(&inc, &cand[..3]).is_err());
    }

    #[test]
    fn t_quantile_continuity() {
        // Expansion at df=31 stays within the tabulated trend and above z.
        assert!((t_crit_95(31) - 2.0395).abs() < 5e-4);
        assert!((t_crit_95(60) - 2.0003).abs() < 5e-4);
        assert!((t_crit_95(120) - 1.9799).abs() < 5e-4);
        assert!(t_crit_95(10_000) > 1.959964);
    }

    #[test]
    fn build_evidence_exact_hash_and_grade() {
        let d = tmp("ev");
        let b = kldseq_bytes(&[0.1, 0.2, 0.3]);
        let inc = run("incumbent", &d.join("i.kldseq"), &b);
        let cand = run("candidate", &d.join("c.kldseq"), &b);
        let ev = build_evidence("promotion", inc.clone(), cand.clone()).unwrap();
        assert!(ev.exact);
        assert_eq!(ev.paired.mean_diff, 0.0);
        assert_eq!(ev.paired.n, 3);

        let other = kldseq_bytes(&[0.1, 0.2, 0.4]);
        let cand2 = run("candidate", &d.join("c2.kldseq"), &other);
        let ev = build_evidence("screen", inc.clone(), cand2).unwrap();
        assert!(!ev.exact);
        assert!(ev.paired.mean_diff > 0.0);

        assert!(build_evidence("bogus", inc.clone(), cand.clone()).is_err());
        // tampered file no longer matches the recorded hash
        std::fs::write(d.join("c.kldseq"), &other).unwrap();
        let e = build_evidence("promotion", inc, cand).unwrap_err();
        assert!(e.contains("sha256"), "{e}");
    }
}
