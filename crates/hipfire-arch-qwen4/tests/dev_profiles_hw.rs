// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer-profile (tiny / 5120-wide dense Qwen4) hardware tests against the
//! S0 CPU references.
//!
//! Every GPU test is `#[ignore]`d: it needs a real HIP GPU and a developer
//! artifact. Environment:
//!
//! * `HIPFIRE_QWEN4_DEV_ARTIFACT` - absolute path of the `.hfq` under test.
//! * `HIPFIRE_QWEN4_DEV_REF` - absolute path of the raw reference directory
//!   written by `s3/export_logits_ref.py` (`tokens.u32`, `output_logits.f32`,
//!   optional `prefill_chunk0_logits.f32` / `chunked_chunk{i}_logits.f32`,
//!   `index.json`).
//! * `HIPFIRE_QWEN4_DEV_REPORT` - absolute directory; each test writes
//!   `<test>.json` there. The predeclared tolerance is written into the report
//!   before any comparison runs.
//! * `HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL` - (refusal test only) substring the
//!   load error must contain.
//!
//! Single-token decode runs on the replay backend chosen once per process by
//! `HIPFIRE_REPLAY_BACKEND` (`hip` | `auto`); the report records it.
//!
//! Nothing here is a mock: a test passes only when the real GPU forward
//! produced the compared rows.

#![cfg(feature = "qwen4-dev-profiles")]

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::session_cache::{SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const ARTIFACT_ENV: &str = "HIPFIRE_QWEN4_DEV_ARTIFACT";
const REF_ENV: &str = "HIPFIRE_QWEN4_DEV_REF";
const REPORT_ENV: &str = "HIPFIRE_QWEN4_DEV_REPORT";
const REFUSAL_ENV: &str = "HIPFIRE_QWEN4_DEV_EXPECT_REFUSAL";

/// Predeclared tolerance; written verbatim into every report before comparing.
const TOLERANCE: &str = "logits row PASS iff max_j |gpu-ref| <= 2^-6 * max(1, max_j |ref_j|) and argmax(gpu)==argmax(ref) whenever ref top1-top2 margin > 2^-6 * max(1, max_j|ref_j|)";
const TOL_REL: f64 = 1.0 / 64.0;

const FIXTURE_MAX_SEQ: usize = 64;
const LONG_MAX_SEQ: usize = 1024;
const DECODE_GREEDY: usize = 16;
const LONG_TOKENS: usize = 513;
const LONG_GREEDY: usize = 8;

fn env_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("{name} must be set"));
    let path = PathBuf::from(value);
    assert!(path.is_absolute(), "{name} must be an absolute path");
    path
}

// ---------------------------------------------------------------- loading

/// The assembled bundle plus the config scalars the tests report.
struct Parts {
    bundle: Qwen4Bundle,
    vocab: usize,
    max_seq: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    eos: u32,
    num_layers: usize,
    state_format: String,
    kv_backend: String,
}

/// A loaded developer bundle on its GPU.
struct Loaded {
    bundle: Qwen4Bundle,
    gpu: Gpu,
    vocab: usize,
    max_seq: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    eos: u32,
    num_layers: usize,
    state_format: String,
    kv_backend: String,
    chunk_rows: usize,
}

/// Every bundle-construction step lives here: admission, manifest fulfilment,
/// state format, assembly, forward attach. The error names the failing stage.
/// `max_seq_request` is clamped to the artifact's `max_position_embeddings`.
fn try_load(gpu: &mut Gpu, max_seq_request: usize) -> Result<Parts, String> {
    let path = env_path(ARTIFACT_ENV);
    let mut hfq = HfqFile::open(&path).map_err(|e| format!("open: {e:?}"))?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(|e| format!("admission: {e}"))?;
    // The executor admission the loader runs between artifact admission and
    // weight fulfillment (hipfire-loader carriers.rs): refusals land here,
    // before any device allocation.
    hipfire_arch_qwen4::bundle::admit_executable_profile(&receipt.config)
        .map_err(|e| format!("executor admission: {e}"))?;
    let max_seq = max_seq_request.min(receipt.config.max_position_embeddings);
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| format!("mesh: {e:?}"))?;
    let expected = WeightOrigin::for_single(&mesh, gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        gpu,
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
        gpu,
        &receipt.config,
    )
    .map_err(|e| format!("state format: {e:?}"))?;
    let vocab = receipt.config.vocab_size;
    let max_position_embeddings = receipt.config.max_position_embeddings;
    let rope_theta = receipt.config.rope_theta;
    let eos = receipt.config.eos_token_id;
    let num_layers = receipt.config.num_hidden_layers;
    let backend = Qwen4KvBackend::automatic(gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| format!("assemble: {e}"))?;
    bundle
        .attach_forward(gpu, max_seq)
        .map_err(|e| format!("attach_forward: {e}"))?;
    Ok(Parts {
        bundle,
        vocab,
        max_seq,
        max_position_embeddings,
        rope_theta,
        eos,
        num_layers,
        state_format: format!("{state_format:?}"),
        kv_backend: backend.name().to_string(),
    })
}

/// Load the artifact named by `HIPFIRE_QWEN4_DEV_ARTIFACT` on a fresh GPU.
fn load_dev(max_seq_request: usize) -> Loaded {
    let mut gpu = Gpu::init().expect("GPU");
    let parts = try_load(&mut gpu, max_seq_request).unwrap_or_else(|e| panic!("load: {e}"));
    let chunk_rows = parts.bundle.spec_chunk_rows().expect("forward attached");
    Loaded {
        bundle: parts.bundle,
        gpu,
        vocab: parts.vocab,
        max_seq: parts.max_seq,
        max_position_embeddings: parts.max_position_embeddings,
        rope_theta: parts.rope_theta,
        eos: parts.eos,
        num_layers: parts.num_layers,
        state_format: parts.state_format,
        kv_backend: parts.kv_backend,
        chunk_rows,
    }
}

impl Loaded {
    fn reset(&mut self) {
        self.bundle.reset(&mut self.gpu).expect("reset");
    }

    fn alloc(&mut self, rows: usize) -> GpuTensor {
        self.gpu
            .zeros(&[rows * self.vocab], DType::F32)
            .expect("alloc logits")
    }

    fn read(&mut self, tensor: &GpuTensor) -> Vec<f32> {
        self.gpu.hip.device_synchronize().expect("sync");
        self.gpu.download_f32(tensor).expect("download logits")
    }

    /// All-row logits of `tokens` (`forward_chunk`); continues the live state.
    fn chunk_all(&mut self, tokens: &[u32]) -> Vec<f32> {
        let t = self.alloc(tokens.len());
        self.bundle
            .forward_chunk(&mut self.gpu, tokens, &t, None)
            .expect("forward_chunk");
        let out = self.read(&t);
        self.gpu.free_tensor(t).expect("free logits");
        out
    }

    /// Logits after one decode step of `token` (`forward_token`).
    fn token(&mut self, token: u32) -> Vec<f32> {
        let t = self.alloc(1);
        self.bundle
            .forward_token(&mut self.gpu, token, &t, None)
            .expect("forward_token");
        let out = self.read(&t);
        self.gpu.free_tensor(t).expect("free logits");
        out
    }

    /// Reset, prefill `tokens` (final row), then `n` greedy decode steps.
    /// Returns the prefill logits, the greedy ids and the logits after each
    /// step (`forward_token_or_argmax(None)` consumes the argmax of the
    /// previous logits, so id `i` is the argmax of logits `i`).
    fn prefill_greedy(&mut self, tokens: &[u32], n: usize) -> (Vec<f32>, Vec<u32>, Vec<Vec<f32>>) {
        self.reset();
        let t = self.alloc(1);
        self.bundle
            .forward_chunk_final(&mut self.gpu, tokens, &t, None)
            .expect("forward_chunk_final");
        let first = self.read(&t);
        let (ids, steps) = self.greedy_on(&t, n);
        self.gpu.free_tensor(t).expect("free logits");
        (first, ids, steps)
    }

    /// `n` greedy steps continuing from the logits already in `t`.
    fn greedy_on(&mut self, t: &GpuTensor, n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut ids = Vec::with_capacity(n);
        let mut steps = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self
                .bundle
                .forward_token_or_argmax(&mut self.gpu, None, t)
                .expect("greedy decode");
            ids.push(id);
            steps.push(self.read(t));
        }
        (ids, steps)
    }

    fn free(self) {
        let Loaded {
            bundle, mut gpu, ..
        } = self;
        bundle.free_gpu(&mut gpu).expect("free bundle");
    }
}

// -------------------------------------------------------------- reference

struct Reference {
    tokens: Vec<u32>,
    output: Vec<f32>,
    prefill_chunk0: Option<Vec<f32>>,
    /// S0 `chunked.chunk{i}.logits`: chunk sizes and the concatenated rows.
    chunked: Option<(Vec<usize>, Vec<f32>)>,
    vocab: usize,
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
    let vocab = index["vocab"].as_u64().expect("index vocab") as usize;
    let n = index["n_tokens"].as_u64().expect("index n_tokens") as usize;
    let tokens: Vec<u32> = std::fs::read(dir.join("tokens.u32"))
        .expect("read tokens.u32")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(tokens.len(), n, "token count");
    let output = read_f32(&dir.join(index["output_logits"]["file"].as_str().expect("file")));
    assert_eq!(output.len(), n * vocab, "output logits shape");
    let prefill_chunk0 = index["prefill_chunk0_logits"]["file"]
        .as_str()
        .map(|f| read_f32(&dir.join(f)));
    if let Some(rows) = &prefill_chunk0 {
        assert_eq!(rows.len(), n * vocab, "prefill.chunk0 logits shape");
    }
    let chunked = index["chunked"]["sizes"].as_array().map(|sizes| {
        let sizes: Vec<usize> = sizes.iter().map(|s| s.as_u64().unwrap() as usize).collect();
        let files = index["chunked"]["files"].as_array().expect("chunked files");
        let mut rows = Vec::new();
        for (size, file) in sizes.iter().zip(files) {
            let part = read_f32(&dir.join(file.as_str().unwrap()));
            assert_eq!(part.len(), size * vocab, "chunked chunk shape");
            rows.extend(part);
        }
        (sizes, rows)
    });
    Reference {
        tokens,
        output,
        prefill_chunk0,
        chunked,
        vocab,
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

/// Compare `gpu` rows against `reference` rows under the predeclared
/// tolerance ([`TOLERANCE`]).
fn compare(gpu: &[f32], reference: &[f32], vocab: usize) -> Cmp {
    assert_eq!(gpu.len(), reference.len(), "compared shapes differ");
    assert!(vocab >= 2 && gpu.len() % vocab == 0);
    let mut rows = Vec::new();
    let (mut rows_pass, mut exact, mut nonfinite) = (0usize, 0usize, 0usize);
    let (mut max_abs, mut max_ulps) = (0f64, 0f64);
    for (r, (g, x)) in gpu
        .chunks_exact(vocab)
        .zip(reference.chunks_exact(vocab))
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
        let argmax_ok = margin <= tol || ag == ar;
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha_f32(streams: &[&[f32]]) -> String {
    let mut h = Sha256::new();
    for s in streams {
        for v in *s {
            h.update(v.to_le_bytes());
        }
    }
    hex(&h.finalize())
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
        for (key, env) in [
            ("artifact", ARTIFACT_ENV),
            ("reference_dir", REF_ENV),
            ("hipfire_replay_backend", "HIPFIRE_REPLAY_BACKEND"),
            ("hipfire_prefill_chunk_rows", "HIPFIRE_PREFILL_CHUNK_ROWS"),
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
        let arch = l.gpu.arch.to_string();
        self.set("gpu_arch", json!(arch));
        self.set("gpu_device_id", json!(l.gpu.device_id));
        self.set("vocab", json!(l.vocab));
        self.set("num_layers", json!(l.num_layers));
        self.set("max_seq_used", json!(l.max_seq));
        self.set("max_position_embeddings", json!(l.max_position_embeddings));
        self.set("rope_theta", json!(l.rope_theta));
        self.set("eos_token_id", json!(l.eos));
        self.set("spec_chunk_rows", json!(l.chunk_rows));
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
        assert!(
            pass,
            "{} FAILED; see {}",
            self.map["test"],
            self.path.display()
        );
    }

    fn skipped(mut self, reason: &str) {
        self.set("status", json!("SKIPPED"));
        self.set("pass", Value::Null);
        self.set("skip_reason", json!(reason));
        self.finalized = true;
        self.write();
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        if !self.finalized {
            self.map.insert(
                "status".into(),
                json!("ABORTED (panic or early exit before the verdict)"),
            );
            self.map.insert("pass".into(), json!(false));
            self.write();
        }
    }
}

// ---------------------------------------------------------- state digests

/// SHA-256 of every part of the live state at `position`: the metadata, each
/// fixed part, then the valid rows of each row stream.
fn state_digests(l: &mut Loaded, position: usize) -> Vec<[u8; 32]> {
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(&mut l.bundle, SessionRoute::Ar, position)
            .expect("state parts");
    l.gpu.hip.device_synchronize().expect("sync");
    let mut digests = vec![Sha256::digest(&meta).into()];
    let ranges = layout
        .fixed
        .iter()
        .map(|part| (part.buf, part.offset, part.bytes))
        .chain(
            layout
                .rows
                .iter()
                .map(|stream| (stream.buf, 0, stream.rows * stream.row_bytes)),
        );
    for (buf, offset, bytes) in ranges {
        let mut host = vec![0u8; bytes];
        l.gpu
            .hip
            .memcpy_dtoh_at(&mut host, buf, offset)
            .expect("download state part");
        digests.push(Sha256::digest(&host).into());
    }
    digests
}

fn digests_json(d: &[[u8; 32]]) -> Value {
    json!(d.iter().map(|x| hex(x)).collect::<Vec<_>>())
}

/// Host copy of the live state at one position: the snapshot metadata, every
/// fixed part and the valid rows of every row stream, in layout order.
struct HostState {
    meta: Vec<u8>,
    fixed: Vec<Vec<u8>>,
    rows: Vec<Vec<u8>>,
}

fn capture_state(l: &mut Loaded, position: usize) -> HostState {
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(&mut l.bundle, SessionRoute::Ar, position)
            .expect("snapshot parts");
    l.gpu.hip.device_synchronize().expect("sync");
    let mut fixed = Vec::new();
    for part in &layout.fixed {
        let mut host = vec![0u8; part.bytes];
        l.gpu
            .hip
            .memcpy_dtoh_at(&mut host, part.buf, part.offset)
            .expect("download fixed part");
        fixed.push(host);
    }
    let mut rows = Vec::new();
    for stream in &layout.rows {
        let mut host = vec![0u8; stream.rows * stream.row_bytes];
        l.gpu
            .hip
            .memcpy_dtoh_at(&mut host, stream.buf, 0)
            .expect("download row stream");
        rows.push(host);
    }
    HostState { meta, fixed, rows }
}

/// Restore `state` through the architecture's `SessionState` protocol:
/// `restore_parts` (prepare + destination layout), copy the bytes, then
/// `finish_restore`.
fn restore_state(l: &mut Loaded, state: &HostState) {
    {
        let layout =
            SessionState::restore_parts(&mut l.bundle, &mut l.gpu, SessionRoute::Ar, &state.meta)
                .expect("restore_parts");
        assert_eq!(layout.fixed.len(), state.fixed.len(), "fixed part count");
        assert_eq!(layout.rows.len(), state.rows.len(), "row stream count");
        for (part, bytes) in layout.fixed.iter().zip(&state.fixed) {
            assert_eq!(part.bytes, bytes.len(), "fixed part size");
            l.gpu
                .hip
                .memcpy_htod_offset(part.buf, part.offset, bytes)
                .expect("upload fixed part");
        }
        for (stream, bytes) in layout.rows.iter().zip(&state.rows) {
            assert_eq!(
                stream.rows * stream.row_bytes,
                bytes.len(),
                "row stream size"
            );
            l.gpu
                .hip
                .memcpy_htod_offset(stream.buf, 0, bytes)
                .expect("upload row stream");
        }
    }
    l.gpu.hip.device_synchronize().expect("sync");
    SessionState::finish_restore(&mut l.bundle, &mut l.gpu, SessionRoute::Ar, &state.meta)
        .expect("finish_restore");
}

// ------------------------------------------------------------------- tests

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn prefill_logits_match_reference() {
    let mut report = Report::new("prefill_logits_match_reference");
    let reference = load_reference();
    let mut l = load_dev(FIXTURE_MAX_SEQ);
    report.context(&l);
    l.reset();
    let gpu_rows = l.chunk_all(&reference.tokens);
    let vs_output = compare(&gpu_rows, &reference.output, reference.vocab);
    report.set("vs_forward_output_logits", vs_output.json());
    let mut pass = vs_output.pass;
    match &reference.prefill_chunk0 {
        Some(rows) => {
            let vs_prefill = compare(&gpu_rows, rows, reference.vocab);
            pass &= vs_prefill.pass;
            report.set("vs_prefill_chunk0_logits", vs_prefill.json());
        }
        None => report.set(
            "vs_prefill_chunk0_logits",
            json!("not present in this fixture (no attention layer)"),
        ),
    }
    l.free();
    report.done(pass);
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn decode_logits_match_reference() {
    let mut report = Report::new("decode_logits_match_reference");
    let reference = load_reference();
    let mut l = load_dev(FIXTURE_MAX_SEQ);
    report.context(&l);
    l.reset();
    let mut gpu_rows = Vec::new();
    for &token in &reference.tokens {
        gpu_rows.extend(l.token(token));
    }
    let cmp = compare(&gpu_rows, &reference.output, reference.vocab);
    report.set("teacher_forced_vs_forward_output_logits", cmp.json());
    let pass = cmp.pass;
    l.free();
    report.done(pass);
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn chunk_splits_match_reference() {
    let mut report = Report::new("chunk_splits_match_reference");
    let reference = load_reference();
    let mut l = load_dev(FIXTURE_MAX_SEQ);
    report.context(&l);
    let n = reference.tokens.len();
    let splits: Vec<Vec<usize>> = vec![
        vec![1; n],
        vec![2, 2, 2, 2, 1],
        vec![4, 4, 1],
        vec![8, 1],
        vec![6, 1, 1, 1],
    ];
    l.reset();
    let whole = l.chunk_all(&reference.tokens);
    let mut results = Vec::new();
    let mut pass = true;
    for split in &splits {
        assert_eq!(split.iter().sum::<usize>(), n, "split covers the fixture");
        l.reset();
        let mut rows = Vec::new();
        let mut at = 0;
        for &size in split {
            rows.extend(l.chunk_all(&reference.tokens[at..at + size]));
            at += size;
        }
        let vs_output = compare(&rows, &reference.output, reference.vocab);
        let mut entry = json!({
            "split": split,
            "vs_forward_output_logits": vs_output.json(),
        });
        let mut ok = vs_output.pass;
        if let Some((sizes, chunked)) = &reference.chunked {
            if sizes == split {
                let vs_chunked = compare(&rows, chunked, reference.vocab);
                ok &= vs_chunked.pass;
                entry["vs_s0_chunked_logits"] = vs_chunked.json();
            }
        }
        let identical_rows = rows
            .chunks_exact(reference.vocab)
            .zip(whole.chunks_exact(reference.vocab))
            .filter(|(a, b)| bits(a) == bits(b))
            .count();
        entry["bit_identical_to_whole_prefill"] = json!(bits(&rows) == bits(&whole));
        entry["rows_bit_identical_to_whole_prefill"] = json!(identical_rows);
        entry["pass"] = json!(ok);
        pass &= ok;
        results.push(entry);
    }
    report.set(
        "s0_chunked_reference",
        match &reference.chunked {
            Some((sizes, _)) => json!({"sizes": sizes}),
            None => json!("not present in this fixture (no attention layer)"),
        },
    );
    report.set("splits", Value::Array(results));
    l.free();
    report.done(pass);
}

/// Deterministic token stream over `vocab` with the EOS id periodically.
fn long_stream(len: usize, vocab: usize, eos: u32) -> Vec<u32> {
    let mut state: u64 = 0x5eed_0513;
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 37 == 36 {
                eos
            } else {
                ((state >> 33) % vocab as u64) as u32
            }
        })
        .collect()
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn long_chunk_boundaries_self_consistent() {
    let mut report = Report::new("long_chunk_boundaries_self_consistent");
    let mut l = load_dev(LONG_MAX_SEQ);
    report.context(&l);
    if l.max_position_embeddings < LONG_MAX_SEQ {
        let reason = format!(
            "artifact max_position_embeddings {} < {LONG_MAX_SEQ}",
            l.max_position_embeddings
        );
        l.free();
        report.skipped(&reason);
        return;
    }
    report.set(
        "label",
        json!("self-consistency (no S0 reference at this length)"),
    );
    let tokens = long_stream(LONG_TOKENS, l.vocab, l.eos);
    report.set("token_count", json!(tokens.len()));
    report.set(
        "token_stream_sha256",
        json!(hex(&Sha256::digest(
            tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<u8>>()
        ))),
    );
    let (whole, whole_ids, _) = l.prefill_greedy(&tokens, LONG_GREEDY);
    report.set("whole_prefill_greedy_ids", json!(whole_ids));
    let mut results = Vec::new();
    let mut pass = true;
    for split in [[511usize, 2], [512, 1], [1, 512], [256, 257]] {
        assert_eq!(split.iter().sum::<usize>(), tokens.len());
        l.reset();
        let t = l.alloc(1);
        let mut at = 0;
        for &size in &split {
            l.bundle
                .forward_chunk_final(&mut l.gpu, &tokens[at..at + size], &t, None)
                .expect("forward_chunk_final");
            at += size;
        }
        let final_row = l.read(&t);
        let (ids, _) = l.greedy_on(&t, LONG_GREEDY);
        l.gpu.free_tensor(t).expect("free logits");
        let cmp = compare(&final_row, &whole, l.vocab);
        let ok = cmp.pass && ids == whole_ids;
        pass &= ok;
        results.push(json!({
            "split": split,
            "final_row_vs_whole_prefill": cmp.json(),
            "greedy_ids": ids,
            "greedy_ids_equal_to_whole": ids == whole_ids,
            "final_row_bitwise_equal_to_whole": bits(&final_row) == bits(&whole),
            "pass": ok,
        }));
    }
    report.set("splits", Value::Array(results));
    l.free();
    report.done(pass);
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn greedy_is_deterministic() {
    let mut report = Report::new("greedy_is_deterministic");
    let reference = load_reference();
    let mut l = load_dev(FIXTURE_MAX_SEQ);
    report.context(&l);
    let (first_a, ids_a, steps_a) = l.prefill_greedy(&reference.tokens, DECODE_GREEDY);
    let (first_b, ids_b, steps_b) = l.prefill_greedy(&reference.tokens, DECODE_GREEDY);
    let stream = |first: &Vec<f32>, steps: &Vec<Vec<f32>>| {
        let mut all: Vec<&[f32]> = vec![first.as_slice()];
        all.extend(steps.iter().map(|s| s.as_slice()));
        sha_f32(&all)
    };
    let (sha_a, sha_b) = (stream(&first_a, &steps_a), stream(&first_b, &steps_b));
    let ids_equal = ids_a == ids_b;
    let logits_equal = bits(&first_a) == bits(&first_b)
        && steps_a.len() == steps_b.len()
        && steps_a
            .iter()
            .zip(&steps_b)
            .all(|(a, b)| bits(a) == bits(b));
    report.set("greedy_ids", json!(ids_a));
    report.set("greedy_ids_second_run", json!(ids_b));
    report.set("logits_stream_sha256", json!(sha_a));
    report.set("logits_stream_sha256_second_run", json!(sha_b));
    report.set(
        "logits_stream_layout",
        json!("sha256 of f32le: prefill final row, then logits after each of the 16 greedy decode steps"),
    );
    report.set("ids_equal", json!(ids_equal));
    report.set("logits_bitwise_equal", json!(logits_equal));
    l.free();
    report.done(ids_equal && logits_equal && sha_a == sha_b);
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn reset_and_restore_are_exact() {
    let mut report = Report::new("reset_and_restore_are_exact");
    let reference = load_reference();
    let mut l = load_dev(FIXTURE_MAX_SEQ);
    report.context(&l);
    let n = reference.tokens.len();
    assert!(n >= 9, "fixture needs at least 9 tokens");

    // (i) prefill, reset, prefill again.
    l.reset();
    let rows_a = l.chunk_all(&reference.tokens);
    let digest_a = state_digests(&mut l, n);
    l.reset();
    let rows_b = l.chunk_all(&reference.tokens);
    let digest_b = state_digests(&mut l, n);
    let reset_logits = bits(&rows_a) == bits(&rows_b);
    let reset_state = digest_a == digest_b;
    report.set(
        "reset_prefill_reset_prefill",
        json!({
            "logits_bitwise_equal": reset_logits,
            "state_digests_equal": reset_state,
            "state_part_count": digest_a.len(),
            "state_digests_first": digests_json(&digest_a),
            "state_digests_second": digests_json(&digest_b),
        }),
    );

    // (ii) snapshot after a 5-token prefix, decode 4, restore, decode 4 again.
    let (prefix, tail) = reference.tokens.split_at(5);
    let tail = &tail[..4];
    l.reset();
    l.chunk_all(prefix);
    let at_snapshot = state_digests(&mut l, prefix.len());
    let saved = capture_state(&mut l, prefix.len());
    let mut run_a = Vec::new();
    for &t in tail {
        run_a.push(l.token(t));
    }
    let end_a = state_digests(&mut l, prefix.len() + tail.len());
    restore_state(&mut l, &saved);
    let after_restore = state_digests(&mut l, prefix.len());
    let mut run_b = Vec::new();
    for &t in tail {
        run_b.push(l.token(t));
    }
    let end_b = state_digests(&mut l, prefix.len() + tail.len());
    let restore_state_equal = at_snapshot == after_restore;
    let restore_logits =
        run_a.len() == run_b.len() && run_a.iter().zip(&run_b).all(|(a, b)| bits(a) == bits(b));
    let restore_end_state = end_a == end_b;
    report.set(
        "snapshot_restore",
        json!({
            "mechanism": "SessionState::snapshot_parts -> host copy; restore_parts -> upload -> finish_restore (Qwen4Bundle, route Ar)",
            "prefix_tokens": prefix.len(),
            "decoded_tokens": tail.len(),
            "state_after_restore_equals_state_at_snapshot": restore_state_equal,
            "decode_logits_bitwise_equal": restore_logits,
            "end_state_digests_equal": restore_end_state,
            "state_part_count": at_snapshot.len(),
            "snapshot_host_bytes": saved.meta.len()
                + saved.fixed.iter().map(Vec::len).sum::<usize>()
                + saved.rows.iter().map(Vec::len).sum::<usize>(),
        }),
    );
    l.free();
    report.done(
        reset_logits && reset_state && restore_state_equal && restore_logits && restore_end_state,
    );
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_QWEN4_DEV_ARTIFACT"]
fn refused_profiles_fail_before_allocation() {
    let mut report = Report::new("refused_profiles_fail_before_allocation");
    let expected = std::env::var(REFUSAL_ENV)
        .unwrap_or_else(|_| panic!("{REFUSAL_ENV} must name the expected refusal substring"));
    report.set("expected_refusal_substring", json!(expected));
    let mut gpu = Gpu::init().expect("GPU");
    report.set("gpu_arch", json!(gpu.arch.to_string()));
    let mem = |gpu: &Gpu| {
        gpu.hip.device_synchronize().expect("sync");
        let (free, total) = gpu.device_mem_info().expect("device_mem_info");
        total.saturating_sub(free)
    };
    let used_before = mem(&gpu);
    let outcome = try_load(&mut gpu, FIXTURE_MAX_SEQ);
    match outcome {
        Ok(parts) => {
            let Parts { bundle, .. } = parts;
            bundle.free_gpu(&mut gpu).expect("free bundle");
            report.set("loaded_unexpectedly", json!(true));
            report.done(false);
        }
        Err(error) => {
            let used_after = mem(&gpu);
            let delta = used_after as i128 - used_before as i128;
            let contains = error.to_lowercase().contains(&expected.to_lowercase());
            let contains_exact = error.contains(&expected);
            let flat = delta.abs() <= (1 << 20);
            report.set("loaded_unexpectedly", json!(false));
            report.set("error", json!(error));
            report.set("error_contains_substring_case_insensitive", json!(contains));
            report.set("error_contains_substring_exact_case", json!(contains_exact));
            report.set("gpu_used_bytes_before", json!(used_before));
            report.set("gpu_used_bytes_after", json!(used_after));
            report.set("gpu_used_delta_bytes", json!(delta as i64));
            report.set("delta_within_1mib", json!(flat));
            report.done(contains && flat);
        }
    }
}
