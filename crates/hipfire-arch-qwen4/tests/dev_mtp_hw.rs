// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer-profile native MTP head hardware tests against the S0 CPU
//! `mtp_step0.npz` reference.
//!
//! Every GPU test is `#[ignore]`d: it needs a real HIP GPU and a developer
//! artifact. Environment:
//!
//! * `HIPFIRE_QWEN4_DEV_ARTIFACT` - absolute path of the `.hfq` under test.
//! * `HIPFIRE_QWEN4_DEV_MTP_REF` - absolute directory written by
//!   `s3/export_mtp_ref.py` (`tokens.u32`, `backbone.f32`, `logits.f32`,
//!   `sample_hidden.f32`, `multi_hidden.f32`, `positions.i64`, `index.json`).
//! * `HIPFIRE_QWEN4_DEV_REPORT` - absolute directory; each test writes
//!   `<test>.json` there. The predeclared tolerance is written into the report
//!   before any comparison runs.
//! * `HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL` - (refusal test only) substring the
//!   `attach_mtp` error must contain.
//!
//! MTP input path: DIRECT. S0 step 0 conditions the head on the reference
//! trunk's pre-final-mixer streams (`mtp.in.backbone_multi_stream`, rows for
//! fixed tokens 0..T-2) with draft input tokens 1..T-1. This test feeds those
//! reference streams row by row to the production head step
//! (`Qwen4Bundle::mtp_dev_row`: `forward_token`, Predict, fresh QSA selection,
//! caller-supplied backbone hidden), so the head is checked in isolation
//! against the reference's own input. The head's positions run 0..R-1 where
//! S0 runs 1..R; RoPE scores are relative and the compress/pool windows are
//! sequence-indexed in both, so the shift only moves absolute rotation
//! angles (rounding level). The report records both position bases.
//!
//! Nothing here is a mock: a test passes only when the real GPU head
//! produced the compared rows.

#![cfg(feature = "qwen4-dev-profiles")]

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::Gpu;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const ARTIFACT_ENV: &str = "HIPFIRE_QWEN4_DEV_ARTIFACT";
const REF_ENV: &str = "HIPFIRE_QWEN4_DEV_MTP_REF";
const REPORT_ENV: &str = "HIPFIRE_QWEN4_DEV_REPORT";
const REFUSAL_ENV: &str = "HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL";

/// Predeclared tolerance; written verbatim into every report before comparing.
/// Looser than the trunk's 2^-6 because the head consumes backbone streams
/// that are BF16-rounded reference values and runs its own F32 chain.
const TOLERANCE: &str = "MTP logits row PASS iff max_j |gpu-ref| <= 2^-5 * max(1, max_j|ref_j|) and argmax(gpu)==argmax(ref) whenever ref top1-top2 margin > 2^-5 * max(1, max_j|ref_j|); sample_hidden and multi_hidden are reported (max abs err, BF16 ulps, exact fraction) but not gating";
const TOL_REL: f64 = 1.0 / 32.0;

const MAX_SEQ_REQUEST: usize = 64;

fn env_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("{name} must be set"));
    let path = PathBuf::from(value);
    assert!(path.is_absolute(), "{name} must be an absolute path");
    path
}

// ---------------------------------------------------------------- loading

/// A loaded developer bundle on its GPU.
struct Loaded {
    bundle: Qwen4Bundle,
    gpu: Gpu,
    max_seq: usize,
    vocab: usize,
    hidden: usize,
    hc_count: usize,
    rope_theta: f64,
    mtp_rope_theta: Option<f64>,
    ffn_kind: &'static str,
    state_format: String,
    kv_backend: String,
}

/// Every bundle-construction step lives here: admission, manifest fulfilment,
/// state format, assembly, and optionally the forward attach. The error names
/// the failing stage. The MTP head is attached by the caller (the refusal test
/// needs the attach error itself).
fn load_dev(attach_forward: bool) -> Result<Loaded, String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let path = env_path(ARTIFACT_ENV);
    let mut hfq = HfqFile::open(&path).map_err(|e| format!("open: {e:?}"))?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(|e| format!("admission: {e}"))?;
    // The executor admission the loader runs before weight fulfillment.
    hipfire_arch_qwen4::bundle::admit_executable_profile(&receipt.config)
        .map_err(|e| format!("executor admission: {e}"))?;
    let max_seq = MAX_SEQ_REQUEST.min(receipt.config.max_position_embeddings);
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| format!("mesh: {e:?}"))?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| format!("weights: {e:?}"))?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )
    .map_err(|e| format!("state format: {e:?}"))?;
    let (vocab, hidden, hc_count) = (
        receipt.config.vocab_size,
        receipt.config.hidden_size,
        receipt.config.hc_count,
    );
    let rope_theta = receipt.config.rope_theta;
    let mtp_rope_theta = receipt.config.mtp.as_ref().map(|mtp| mtp.rope_theta);
    let ffn_kind = if receipt.config.ffn.is_dense() {
        "dense"
    } else {
        "moe"
    };
    let backend = Qwen4KvBackend::automatic(&gpu);
    let recipe = receipt.recipe().clone();
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| format!("assemble: {e}"))?;
    if attach_forward {
        bundle
            .attach_forward(&mut gpu, max_seq)
            .map_err(|e| format!("attach_forward: {e}"))?;
    }
    Ok(Loaded {
        bundle,
        gpu,
        max_seq,
        vocab,
        hidden,
        hc_count,
        rope_theta,
        mtp_rope_theta,
        ffn_kind,
        state_format: format!("{state_format:?}"),
        kv_backend: backend.name().to_string(),
    })
}

impl Loaded {
    fn free(self) {
        let Loaded {
            bundle, mut gpu, ..
        } = self;
        bundle.free_gpu(&mut gpu).expect("free bundle");
    }
}

// -------------------------------------------------------------- reference

struct Reference {
    rows: usize,
    vocab: usize,
    hidden: usize,
    wide: usize,
    positions: Vec<i64>,
    tokens: Vec<u32>,
    backbone: Vec<f32>,
    logits: Vec<f32>,
    sample_hidden: Vec<f32>,
    multi_hidden: Vec<f32>,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len() % 4 == 0, "{}: not f32", path.display());
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_reference() -> Reference {
    let dir = env_path(REF_ENV);
    let index: Value = serde_json::from_slice(
        &std::fs::read(dir.join("index.json")).expect("read ref index.json"),
    )
    .expect("parse ref index.json");
    let num = |key: &str| index[key].as_u64().unwrap_or_else(|| panic!("index {key}")) as usize;
    let (rows, vocab, hidden, wide) = (num("rows"), num("vocab"), num("hidden"), num("wide"));
    let file = |key: &str| dir.join(index["files"][key].as_str().expect("index file"));
    let tokens: Vec<u32> = std::fs::read(file("tokens"))
        .expect("read tokens")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let positions: Vec<i64> = std::fs::read(file("positions"))
        .expect("read positions")
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let backbone = read_f32(&file("backbone"));
    let logits = read_f32(&file("logits"));
    let sample_hidden = read_f32(&file("sample_hidden"));
    let multi_hidden = read_f32(&file("multi_hidden"));
    assert_eq!(tokens.len(), rows, "token count");
    assert_eq!(positions.len(), rows, "position count");
    assert_eq!(backbone.len(), rows * wide, "backbone shape");
    assert_eq!(logits.len(), rows * vocab, "logits shape");
    assert_eq!(sample_hidden.len(), rows * hidden, "sample_hidden shape");
    assert_eq!(multi_hidden.len(), rows * wide, "multi_hidden shape");
    Reference {
        rows,
        vocab,
        hidden,
        wide,
        positions,
        tokens,
        backbone,
        logits,
        sample_hidden,
        multi_hidden,
    }
}

// ----------------------------------------------------------------- compare

struct Cmp {
    pass: bool,
    rows: Vec<Value>,
    rows_pass: usize,
    max_abs_err: f64,
    max_err_bf16_ulps: f64,
    bit_exact_fraction: f64,
    nonfinite_gpu_values: usize,
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (j, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = j;
        }
    }
    best
}

/// BF16 ulp of `x` (the spacing of BF16 values at `x`'s magnitude).
fn bf16_ulp(x: f32) -> f64 {
    let exp = ((x.to_bits() >> 23) & 0xff) as i32;
    if exp == 0 {
        2f64.powi(-133)
    } else {
        2f64.powi(exp - 127 - 7)
    }
}

/// Round-to-nearest-even f32 -> BF16 bits.
fn bf16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let bits = x.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

/// Compare `gpu` rows against `reference` rows of width `width` under the
/// predeclared tolerance ([`TOLERANCE`]). `rank` also applies the argmax
/// clause (logits rows only).
fn compare(gpu: &[f32], reference: &[f32], width: usize, rank: bool) -> Cmp {
    assert_eq!(gpu.len(), reference.len(), "compared shapes differ");
    assert!(width >= 2 && gpu.len() % width == 0);
    let mut rows = Vec::new();
    let (mut rows_pass, mut exact, mut nonfinite) = (0usize, 0usize, 0usize);
    let (mut max_abs, mut max_ulps) = (0f64, 0f64);
    for (r, (g, x)) in gpu
        .chunks_exact(width)
        .zip(reference.chunks_exact(width))
        .enumerate()
    {
        let scale = x.iter().fold(1f64, |m, v| m.max(v.abs() as f64));
        let tol = TOL_REL * scale;
        let mut row_err = 0f64;
        for (&gv, &xv) in g.iter().zip(x) {
            let err = if gv.is_finite() {
                (gv as f64 - xv as f64).abs()
            } else {
                nonfinite += 1;
                f64::INFINITY
            };
            row_err = row_err.max(err);
            max_ulps = max_ulps.max(err / bf16_ulp(xv));
            if bf16_bits(gv) == (xv.to_bits() >> 16) as u16 {
                exact += 1;
            }
        }
        max_abs = max_abs.max(row_err);
        let (ag, ar) = (argmax(g), argmax(x));
        let top1 = x[ar] as f64;
        let top2 = x
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != ar)
            .fold(f64::NEG_INFINITY, |m, (_, v)| m.max(*v as f64));
        let margin = top1 - top2;
        let argmax_ok = !rank || margin <= tol || ag == ar;
        let pass = row_err <= tol && argmax_ok;
        rows_pass += pass as usize;
        rows.push(json!({
            "row": r, "pass": pass, "max_abs_err": row_err, "tol": tol,
            "ref_margin": margin, "argmax_gpu": ag, "argmax_ref": ar,
        }));
    }
    let total = gpu.len();
    Cmp {
        pass: rows_pass == rows.len(),
        rows_pass,
        rows,
        max_abs_err: max_abs,
        max_err_bf16_ulps: max_ulps,
        bit_exact_fraction: exact as f64 / total as f64,
        nonfinite_gpu_values: nonfinite,
    }
}

impl Cmp {
    fn json(&self) -> Value {
        json!({
            "pass": self.pass,
            "rows": self.rows.len(),
            "rows_pass": self.rows_pass,
            "max_abs_err": self.max_abs_err,
            "max_err_bf16_ulps_of_ref": self.max_err_bf16_ulps,
            "bit_exact_fraction_bf16": self.bit_exact_fraction,
            "nonfinite_gpu_values": self.nonfinite_gpu_values,
            "per_row": self.rows,
        })
    }
}

// ------------------------------------------------------------------ report

struct Report {
    path: PathBuf,
    map: Map<String, Value>,
    finalized: bool,
}

impl Report {
    /// Starts the report and writes it, tolerance first, before any compare.
    fn new(test: &str) -> Self {
        let dir = env_path(REPORT_ENV);
        std::fs::create_dir_all(&dir).expect("create report dir");
        let mut map = Map::new();
        map.insert("test".into(), json!(test));
        map.insert("tolerance".into(), json!(TOLERANCE));
        map.insert("status".into(), json!("started"));
        map.insert(
            "mtp_input_path".into(),
            json!("direct: S0 reference backbone streams fed row by row to the production MTP head step (Qwen4Bundle::mtp_dev_row)"),
        );
        for (key, env) in [
            ("artifact", ARTIFACT_ENV),
            ("reference_dir", REF_ENV),
            ("expected_refusal", REFUSAL_ENV),
            ("hipfire_mtp_draft_head", "HIPFIRE_MTP_DRAFT_HEAD"),
        ] {
            map.insert(
                key.into(),
                std::env::var(env).map_or(Value::Null, Value::String),
            );
        }
        let report = Report {
            path: dir.join(format!("{test}.json")),
            map,
            finalized: false,
        };
        report.write();
        report
    }

    fn context(&mut self, l: &Loaded) {
        self.set("gpu_arch", json!(l.gpu.arch.to_string()));
        self.set("gpu_device_id", json!(l.gpu.device_id));
        self.set("vocab", json!(l.vocab));
        self.set("hidden", json!(l.hidden));
        self.set("hc_count", json!(l.hc_count));
        self.set("ffn_kind", json!(l.ffn_kind));
        self.set("max_seq_used", json!(l.max_seq));
        self.set("rope_theta", json!(l.rope_theta));
        self.set("mtp_rope_theta", json!(l.mtp_rope_theta));
        self.set("state_format", json!(l.state_format));
        self.set("qsa_kv_backend", json!(l.kv_backend));
        self.write();
    }

    fn set(&mut self, key: &str, value: Value) {
        self.map.insert(key.into(), value);
    }

    fn write(&self) {
        let text = serde_json::to_string_pretty(&Value::Object(self.map.clone())).unwrap();
        std::fs::write(&self.path, text + "\n")
            .unwrap_or_else(|e| panic!("write {}: {e}", self.path.display()));
    }

    fn done(mut self, pass: bool) {
        self.set("status", json!(if pass { "PASS" } else { "FAIL" }));
        self.set("pass", json!(pass));
        self.finalized = true;
        self.write();
        assert!(pass, "see {}", self.path.display());
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        // A panic before `done` leaves an honest FAIL record.
        if !self.finalized {
            self.map.insert("status".into(), json!("FAIL"));
            self.map.insert("pass".into(), json!(false));
            self.write();
        }
    }
}

// ------------------------------------------------------------------- tests

/// The head's logits (and diagnostics) over the S0 step-0 rows.
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn dev_mtp_head_matches_s0_step0() {
    let mut report = Report::new("dev_mtp_head_matches_s0_step0");
    let reference = load_reference();
    let mut loaded = load_dev(true).unwrap_or_else(|e| panic!("load: {e}"));
    report.context(&loaded);
    assert_eq!(reference.vocab, loaded.vocab, "reference vocab");
    assert_eq!(reference.hidden, loaded.hidden, "reference hidden");
    assert_eq!(
        reference.wide,
        loaded.hc_count * loaded.hidden,
        "reference width"
    );
    assert!(
        reference.rows <= loaded.max_seq,
        "fixture rows exceed the head capacity"
    );
    loaded
        .bundle
        .attach_mtp(&mut loaded.gpu, loaded.max_seq)
        .unwrap_or_else(|e| panic!("attach_mtp: {e}"));
    report.set("rows", json!(reference.rows));
    report.set("ref_positions", json!(reference.positions));
    report.set(
        "gpu_positions",
        json!((0..reference.rows).collect::<Vec<_>>()),
    );
    report.write();

    let run = |loaded: &mut Loaded| {
        loaded.bundle.reset(&mut loaded.gpu).expect("reset");
        let (mut logits, mut sample, mut multi, mut drafts) = (vec![], vec![], vec![], vec![]);
        for row in 0..reference.rows {
            let out = loaded
                .bundle
                .mtp_dev_row(
                    &mut loaded.gpu,
                    reference.tokens[row],
                    &reference.backbone[row * reference.wide..(row + 1) * reference.wide],
                    row,
                )
                .unwrap_or_else(|e| panic!("mtp row {row}: {e}"));
            assert_eq!(out.logits.len(), reference.vocab, "logits width");
            assert_eq!(out.sample_hidden.len(), reference.hidden, "sample width");
            assert_eq!(out.multi_hidden.len(), reference.wide, "multi width");
            logits.extend(out.logits);
            sample.extend(out.sample_hidden);
            multi.extend(out.multi_hidden);
            drafts.push(out.draft_token);
        }
        (logits, sample, multi, drafts)
    };
    let (logits, sample, multi, drafts) = run(&mut loaded);
    let (again, ..) = run(&mut loaded);

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let rerun_identical = bits(&logits) == bits(&again);
    let cmp = compare(&logits, &reference.logits, reference.vocab, true);
    let sample_cmp = compare(&sample, &reference.sample_hidden, reference.hidden, false);
    let multi_cmp = compare(&multi, &reference.multi_hidden, reference.wide, false);
    let draft_is_gpu_argmax = logits
        .chunks_exact(reference.vocab)
        .zip(&drafts)
        .all(|(row, &draft)| argmax(row) as u32 == draft);
    report.set("logits", cmp.json());
    report.set("sample_hidden", sample_cmp.json());
    report.set("multi_hidden", multi_cmp.json());
    report.set("rerun_logits_bit_identical", json!(rerun_identical));
    report.set("draft_tokens", json!(drafts));
    report.set("draft_token_is_gpu_argmax", json!(draft_is_gpu_argmax));
    println!(
        "MTP logits: max_abs_err {:.6} ({:.2} bf16 ulps of ref), exact {:.4}, rows {}/{}, rerun identical {rerun_identical}",
        cmp.max_abs_err,
        cmp.max_err_bf16_ulps,
        cmp.bit_exact_fraction,
        cmp.rows_pass,
        cmp.rows.len()
    );
    let pass = cmp.pass && cmp.nonfinite_gpu_values == 0 && rerun_identical;
    loaded.free();
    report.done(pass);
}

/// `attach_mtp` refuses with `HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL`: a RoPE base
/// the QSA kernels do not implement (rope-1e4 originals) or a head-free
/// profile. The forward is not attached: the trunk refuses rope-1e4 profiles
/// first, and this checks the head's own admission.
#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn dev_mtp_attach_refuses() {
    let mut report = Report::new("dev_mtp_attach_refuses");
    let expected = std::env::var(REFUSAL_ENV)
        .unwrap_or_else(|_| panic!("{REFUSAL_ENV} must name the expected refusal substring"));
    let mut loaded = load_dev(false).unwrap_or_else(|e| panic!("load: {e}"));
    report.context(&loaded);
    let result = loaded.bundle.attach_mtp(&mut loaded.gpu, loaded.max_seq);
    let message = match &result {
        Ok(()) => None,
        Err(error) => Some(error.to_string()),
    };
    report.set("attach_error", json!(message));
    let pass = message.as_deref().is_some_and(|m| m.contains(&expected));
    loaded.free();
    report.done(pass);
}
