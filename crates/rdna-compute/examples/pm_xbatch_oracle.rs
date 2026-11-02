// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the PM `gemv_mq4g256v2_xbatch_pm` (gfx1201).
//!
//! `verify --co FILE --out FILE.json`: for every case and B = 1..8, the
//! candidate's whole y allocation (guards, unused columns and the B written
//! columns) must equal the expectation built from the incumbent singleton:
//! hipcc `gemv_mq4g256v2_xbatch` launched with B = 1 on each column `x[b]`.
//! The hipcc x-batch at the same B (launched as the production wrapper does,
//! in chunks of 4) must equal it too, and for the real gate/up matrices so must
//! the production singleton `fused_gate_up_mq4g256v2`. Cases: every real H2
//! matrix captured under `oracle-inputs` (QKV, QKVZA, gate/up, wo, down,
//! lm_head) at its real shape, plus sub-matrices of real gate/down bytes with
//! K = 1..9 groups, real K - 256 and odd/non-multiple-of-4 M. Columns are 8
//! distinct real activation vectors (captured x inputs, then captured
//! projection outputs, as long as K needs).
//!
//! `time --co FILE --out FILE.json`: HIP-event medians (5 warmups, 15
//! samples) at real gate (17408), QKVZA qkv (10240) and z (6144) shapes,
//! B = 1/2/4/8: candidate (one launch), hipcc x-batch (production chunking)
//! and B x the singleton `Gpu::gemv_mq4g256v2`. One process per run; run it
//! three times for three fresh processes.

use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;

const CAP: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-inputs";
const SYMBOL: &str = "gemv_mq4g256v2_xbatch_pm";
const XBATCH: &str = "gemv_mq4g256v2_xbatch";
/// The production x-batch source string (`kernels::GEMV_MQ4G256V2_XBATCH_SRC`).
const XBATCH_SRC: &str = concat!(
    "#define HIPFIRE_MQ4G256V2_XBATCH_MAX 4\n",
    "#define HIPFIRE_MQ4G256V2_XBATCH_KERNEL gemv_mq4g256v2_xbatch\n",
    include_str!("../../../kernels/src/gemv_mq4g256v2_xbatch.hip")
);
const XBATCH_MAX: usize = 4;
const MAXB: usize = 8;
const GROUP: usize = 136;
const GUARD: usize = 256;
const SENTINEL: u8 = 0x7f;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One captured file, checked against its capture's meta.json hash.
fn capture(rel: &str) -> Result<Vec<u8>> {
    let path = Path::new(CAP).join(rel);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().ok_or("capture without directory")?;
    let meta: Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(err)?).map_err(err)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("capture name")?;
    let want = meta["files"][name]["sha256"]
        .as_str()
        .ok_or(format!("{rel}: no hash"))?;
    if sha256_hex(&bytes) != want {
        return Err(format!("{rel}: sha256 mismatch"));
    }
    Ok(bytes)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Real activation sources in preference order: captured x inputs, then
/// captured projection outputs (long enough for K = 6144 / 17408).
const ACTIVATIONS: [&str; 17] = [
    "fused_gate_up_mq4g256v2/occ0/arg16.pre.bin",
    "fused_gate_up_mq4g256v2/occ63/arg16.pre.bin",
    "fused_qkv_mq4g256v2/occ0/arg24.pre.bin",
    "fused_qkv_mq4g256v2/occ15/arg24.pre.bin",
    "fused_qkvza_mq4g256v2/occ0/arg32.pre.bin",
    "fused_qkvza_mq4g256v2/occ47/arg32.pre.bin",
    "gemv_mq4g256v2_multirow_r2/occ0/arg8.pre.bin",
    "gemv_mq4g256v2_residual/occ0/arg8.pre.bin",
    "gemv_mq4g256v2_residual/occ127/arg8.pre.bin",
    "fused_gate_up_mq4g256v2/occ0/arg24.post.bin",
    "fused_gate_up_mq4g256v2/occ63/arg24.post.bin",
    "fused_gate_up_mq4g256v2/occ0/arg32.post.bin",
    "fused_gate_up_mq4g256v2/occ63/arg32.post.bin",
    "fused_gate_up_mq4g256v2/occ0/arg24.pre.bin",
    "fused_gate_up_mq4g256v2/occ63/arg24.pre.bin",
    "fused_gate_up_mq4g256v2/occ0/arg32.pre.bin",
    "fused_gate_up_mq4g256v2/occ63/arg32.pre.bin",
];

struct Activations {
    vectors: Vec<(String, Vec<f32>)>,
}

impl Activations {
    fn load() -> Result<Self> {
        let mut vectors = Vec::new();
        for rel in ACTIVATIONS {
            vectors.push((rel.to_string(), f32s(&capture(rel)?)));
        }
        Ok(Self { vectors })
    }
    /// `[MAXB][k]` row-major: the first eight sources holding k values.
    fn columns(&self, k: usize) -> Result<(Vec<f32>, Vec<String>)> {
        let mut x = Vec::with_capacity(MAXB * k);
        let mut names = Vec::new();
        for (name, v) in self.vectors.iter().filter(|(_, v)| v.len() >= k).take(MAXB) {
            x.extend_from_slice(&v[..k]);
            names.push(name.clone());
        }
        if names.len() != MAXB {
            return Err(format!(
                "only {} real activation vectors hold K={k}",
                names.len()
            ));
        }
        Ok((x, names))
    }
}

struct Matrix {
    name: String,
    source: String,
    m: usize,
    k: usize,
    bytes: Vec<u8>,
}

fn real_matrix(name: &str, rel: &str, k: usize) -> Result<Matrix> {
    let bytes = capture(rel)?;
    let row = k / 256 * GROUP;
    if bytes.len() % row != 0 {
        return Err(format!(
            "{rel}: {} bytes is not whole K={k} rows",
            bytes.len()
        ));
    }
    Ok(Matrix {
        name: name.into(),
        source: rel.into(),
        m: bytes.len() / row,
        k,
        bytes,
    })
}

/// Rows `start..start+m`, groups `0..groups` of a real matrix.
fn sub_matrix(src: &Matrix, start: usize, m: usize, groups: usize) -> Matrix {
    let row = src.k / 256 * GROUP;
    let mut bytes = Vec::with_capacity(m * groups * GROUP);
    for r in start..start + m {
        bytes.extend_from_slice(&src.bytes[r * row..r * row + groups * GROUP]);
    }
    Matrix {
        name: format!(
            "{}[rows {start}..{}, groups 0..{groups}]",
            src.name,
            start + m
        ),
        source: src.source.clone(),
        m,
        k: groups * 256,
        bytes,
    }
}

struct Candidate {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
    sha256: String,
}

fn load_candidate(gpu: &Gpu, path: &str) -> Result<Candidate> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let module = gpu.hip.module_load_data(&bytes).map_err(err)?;
    let func = gpu.hip.module_get_function(&module, SYMBOL).map_err(err)?;
    Ok(Candidate {
        _module: module,
        func,
        sha256: sha256_hex(&bytes),
    })
}

fn kernarg(a: u64, x: u64, y: u64, m: usize, k: usize, b: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(40);
    v.extend_from_slice(&a.to_le_bytes());
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    for d in [m, k, b] {
        v.extend_from_slice(&(d as i32).to_le_bytes());
    }
    v.resize(40, 0);
    v
}

/// Device copies of one matrix and its eight columns, plus a guarded y.
struct Problem {
    m: usize,
    k: usize,
    a: GpuTensor,
    x: GpuTensor,
    /// `GUARD | MAXB * m f32 | GUARD` bytes.
    y: GpuTensor,
}

impl Problem {
    fn new(gpu: &mut Gpu, mat: &Matrix, x: &[f32]) -> Result<Self> {
        let a = gpu
            .upload_raw(&mat.bytes, &[mat.bytes.len()])
            .map_err(err)?;
        let x = gpu.upload_f32(x, &[x.len()]).map_err(err)?;
        let y = gpu
            .zeros(&[(2 * GUARD) / 4 + MAXB * mat.m], DType::F32)
            .map_err(err)?;
        Ok(Self {
            m: mat.m,
            k: mat.k,
            a,
            x,
            y,
        })
    }
    fn a_ptr(&self) -> u64 {
        self.a.buf.as_ptr() as u64
    }
    fn x_ptr(&self, col: usize) -> u64 {
        self.x.buf.as_ptr() as u64 + (col * self.k * 4) as u64
    }
    fn y_ptr(&self, col: usize) -> u64 {
        self.y.buf.as_ptr() as u64 + (GUARD + col * self.m * 4) as u64
    }
    fn y_bytes(&self) -> usize {
        2 * GUARD + MAXB * self.m * 4
    }
    fn reset_y(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.y.buf, SENTINEL as i32, self.y_bytes())
            .map_err(err)
    }
    fn download_y(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.y.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.y.buf).map_err(err)?;
        v.truncate(self.y_bytes());
        Ok(v)
    }
    fn x_view(&self, col: usize) -> GpuTensor {
        self.x.sub_offset(col * self.k, self.k)
    }
    fn y_view(&self, col: usize) -> GpuTensor {
        self.y.sub_offset(GUARD / 4 + col * self.m, self.m)
    }
    /// Candidate over columns `0..b`, one launch.
    fn launch_pm(&self, gpu: &Gpu, cand: &Candidate, b: usize) -> Result<()> {
        let mut ka = kernarg(
            self.a_ptr(),
            self.x_ptr(0),
            self.y_ptr(0),
            self.m,
            self.k,
            b,
        );
        // SAFETY: the kernarg follows the candidate's ABI and addresses live allocations.
        unsafe {
            gpu.hip.launch_kernel_blob(
                &cand.func,
                [self.m as u32, 1, 1],
                [32, 1, 1],
                0,
                None,
                &mut ka,
            )
        }
        .map_err(err)
    }
    /// hipcc x-batch over columns `first..first+b`, chunked like the
    /// production wrapper (`Gpu::gemv_v2_xbatch_rows`, XBATCH_MAX 4).
    fn launch_xbatch(&self, gpu: &Gpu, first: usize, b: usize) -> Result<()> {
        let mut start = 0;
        while start < b {
            let rows = XBATCH_MAX.min(b - start);
            let mut ka = kernarg(
                self.a_ptr(),
                self.x_ptr(first + start),
                self.y_ptr(first + start),
                self.m,
                self.k,
                rows,
            );
            gpu.launch_kernel_blob(XBATCH, [self.m as u32, 1, 1], [32, 1, 1], 0, &mut ka)
                .map_err(err)?;
            start += rows;
        }
        Ok(())
    }
}

struct Tally {
    cases: Vec<Value>,
    launches: usize,
    mismatched_bytes: usize,
    cross_mismatched_bytes: usize,
}

fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

/// Verify one matrix: singleton references, then candidate and hipcc x-batch
/// for every B. `gate_up`: the matching other half for the production
/// singleton cross-check (`Some((up, is_gate))`).
fn verify_matrix(
    gpu: &mut Gpu,
    cand: &Candidate,
    mat: &Matrix,
    acts: &Activations,
    gate_up: Option<(&Matrix, bool)>,
    tally: &mut Tally,
) -> Result<()> {
    let (x, names) = acts.columns(mat.k)?;
    let p = Problem::new(gpu, mat, &x)?;
    // Reference: hipcc x-batch with B = 1 on each column.
    p.reset_y(gpu)?;
    for col in 0..MAXB {
        p.launch_xbatch(gpu, col, 1)?;
    }
    let reference = p.download_y(gpu)?;
    let col_range = |col: usize| GUARD + col * p.m * 4..GUARD + (col + 1) * p.m * 4;
    let mut sentinel = vec![SENTINEL; p.y_bytes()];
    let mut by_b = Vec::new();
    let mut case_bytes = 0;
    let mut cross_bytes = 0;
    for b in 1..=MAXB {
        // Expected: columns 0..b from the singleton reference, sentinel elsewhere.
        let r = col_range(b - 1);
        sentinel[r.clone()].copy_from_slice(&reference[r]);
        let expected = &sentinel;
        p.reset_y(gpu)?;
        p.launch_pm(gpu, cand, b)?;
        let got = p.download_y(gpu)?;
        let (bad, first) = diff(&got, expected);
        p.reset_y(gpu)?;
        p.launch_xbatch(gpu, 0, b)?;
        let inc = p.download_y(gpu)?;
        let (inc_bad, _) = diff(&inc, expected);
        tally.launches += 1 + b.div_ceil(XBATCH_MAX);
        case_bytes += bad;
        cross_bytes += inc_bad;
        by_b.push(json!({"B": b, "candidate_mismatch_bytes": bad, "first_mismatch_byte": first, "hipcc_xbatch_vs_singleton_mismatch_bytes": inc_bad}));
    }
    let mut extra = json!(null);
    if let Some((other, is_gate)) = gate_up {
        // Production singleton fused_gate_up on each column (gate, up halves).
        let o = gpu
            .upload_raw(&other.bytes, &[other.bytes.len()])
            .map_err(err)?;
        let scratch = gpu.zeros(&[other.m], DType::F32).map_err(err)?;
        p.reset_y(gpu)?;
        for col in 0..MAXB {
            let (xv, yv) = (p.x_view(col), p.y_view(col));
            if is_gate {
                gpu.fused_gate_up_hfq4g256_mq4v2(&p.a, &o, &xv, &yv, &scratch, p.m, other.m, p.k)
                    .map_err(err)?;
            } else {
                gpu.fused_gate_up_hfq4g256_mq4v2(&o, &p.a, &xv, &scratch, &yv, other.m, p.m, p.k)
                    .map_err(err)?;
            }
        }
        let fused = p.download_y(gpu)?;
        let (bad, _) = diff(&fused, &reference);
        cross_bytes += bad;
        extra = json!({"fused_gate_up_singleton_vs_xbatch_singleton_mismatch_bytes": bad});
    }
    tally.mismatched_bytes += case_bytes;
    tally.cross_mismatched_bytes += cross_bytes;
    eprintln!(
        "{:<60} M={:<6} K={:<5} candidate mismatches {case_bytes} cross {cross_bytes}",
        mat.name, p.m, p.k
    );
    tally.cases.push(json!({
        "matrix": mat.name, "source": mat.source, "M": p.m, "K": p.k, "groups": p.k / 256,
        "tail_groups": (p.k / 256) % 4, "columns": names, "by_B": by_b,
        "candidate_mismatch_bytes": case_bytes, "cross_mismatch_bytes": cross_bytes, "production_singleton": extra,
    }));
    Ok(())
}

fn verify(gpu: &mut Gpu, cand: &Candidate) -> Result<Value> {
    let acts = Activations::load()?;
    let gate = real_matrix("gate", "fused_gate_up_mq4g256v2/occ0/arg0.pre.bin", 5120)?;
    let up = real_matrix("up", "fused_gate_up_mq4g256v2/occ0/arg8.pre.bin", 5120)?;
    let down = real_matrix("down", "gemv_mq4g256v2_residual/occ127/arg0.pre.bin", 17408)?;
    let mut tally = Tally {
        cases: Vec::new(),
        launches: 0,
        mismatched_bytes: 0,
        cross_mismatched_bytes: 0,
    };
    verify_matrix(gpu, cand, &gate, &acts, Some((&up, true)), &mut tally)?;
    verify_matrix(gpu, cand, &up, &acts, Some((&gate, false)), &mut tally)?;
    verify_matrix(gpu, cand, &down, &acts, None, &mut tally)?;
    for (name, rel, k) in [
        ("qkv.q", "fused_qkv_mq4g256v2/occ0/arg0.pre.bin", 5120),
        ("qkv.k", "fused_qkv_mq4g256v2/occ0/arg8.pre.bin", 5120),
        ("qkv.v", "fused_qkv_mq4g256v2/occ0/arg16.pre.bin", 5120),
        ("qkvza.qkv", "fused_qkvza_mq4g256v2/occ0/arg0.pre.bin", 5120),
        ("qkvza.z", "fused_qkvza_mq4g256v2/occ0/arg8.pre.bin", 5120),
        ("qkvza.b", "fused_qkvza_mq4g256v2/occ0/arg16.pre.bin", 5120),
        ("qkvza.a", "fused_qkvza_mq4g256v2/occ0/arg24.pre.bin", 5120),
        ("wo", "gemv_mq4g256v2_residual/occ0/arg0.pre.bin", 6144),
        (
            "lm_head",
            "gemv_mq4g256v2_multirow_r2/occ0/arg0.pre.bin",
            5120,
        ),
    ] {
        let mat = real_matrix(name, rel, k)?;
        verify_matrix(gpu, cand, &mat, &acts, None, &mut tally)?;
    }
    // Tails and odd M from real bytes: K = 1..9 groups and real K - 256.
    let ms = [1usize, 3, 6, 7, 4097, 4099];
    for (i, groups) in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 19, 23, 67]
        .into_iter()
        .enumerate()
    {
        let src = if groups <= 20 { &gate } else { &down };
        for (j, &m) in ms.iter().enumerate() {
            let start = (977 * (i * ms.len() + j)) % (src.m - m);
            let sub = sub_matrix(src, start, m, groups);
            verify_matrix(gpu, cand, &sub, &acts, None, &mut tally)?;
        }
    }
    Ok(json!({
        "cases": tally.cases.len(), "launch_groups": tally.launches,
        "candidate_mismatch_bytes": tally.mismatched_bytes,
        "cross_mismatch_bytes": tally.cross_mismatched_bytes,
        "detail": tally.cases,
    }))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_arm(gpu: &mut Gpu, mut arm: impl FnMut(&mut Gpu) -> Result<()>) -> Result<Vec<f64>> {
    let (start, stop) = (
        gpu.hip.event_create().map_err(err)?,
        gpu.hip.event_create().map_err(err)?,
    );
    for _ in 0..5 {
        arm(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let mut samples = Vec::new();
    for _ in 0..15 {
        gpu.hip.event_record(&start, None).map_err(err)?;
        arm(gpu)?;
        gpu.hip.event_record(&stop, None).map_err(err)?;
        gpu.hip.event_synchronize(&stop).map_err(err)?;
        samples.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1000.0);
    }
    gpu.hip.event_destroy(start).map_err(err)?;
    gpu.hip.event_destroy(stop).map_err(err)?;
    Ok(samples)
}

fn time(gpu: &mut Gpu, cand: &Candidate) -> Result<Value> {
    let acts = Activations::load()?;
    let mut shapes = Vec::new();
    for (name, rel) in [
        ("gate", "fused_gate_up_mq4g256v2/occ0/arg0.pre.bin"),
        ("qkvza.qkv", "fused_qkvza_mq4g256v2/occ0/arg0.pre.bin"),
        ("qkvza.z", "fused_qkvza_mq4g256v2/occ0/arg8.pre.bin"),
    ] {
        let mat = real_matrix(name, rel, 5120)?;
        let (x, _) = acts.columns(mat.k)?;
        let p = Problem::new(gpu, &mat, &x)?;
        let mut rows = Vec::new();
        for b in [1usize, 2, 4, 8] {
            // Exactness of the timed buffers (candidate vs hipcc x-batch).
            p.reset_y(gpu)?;
            p.launch_xbatch(gpu, 0, b)?;
            let inc = p.download_y(gpu)?;
            p.reset_y(gpu)?;
            p.launch_pm(gpu, cand, b)?;
            let (bad, _) = diff(&p.download_y(gpu)?, &inc);
            let mut pm = time_arm(gpu, |g| p.launch_pm(g, cand, b))?;
            let mut xb = time_arm(gpu, |g| p.launch_xbatch(g, 0, b))?;
            let mut single = time_arm(gpu, |g| {
                for col in 0..b {
                    g.gemv_mq4g256v2(&p.a, &p.x_view(col), &p.y_view(col), p.m, p.k)
                        .map_err(err)?;
                }
                Ok(())
            })?;
            let (mp, mx, ms) = (median(&mut pm), median(&mut xb), median(&mut single));
            eprintln!("{name:<10} B={b}: pm {mp:8.2} us | hipcc xbatch {mx:8.2} us | {b} x singleton {ms:8.2} us | mismatches {bad}");
            rows.push(json!({
                "B": b, "pm_us": pm, "xbatch_us": xb, "singleton_x_b_us": single,
                "pm_median_us": mp, "xbatch_median_us": mx, "singleton_x_b_median_us": ms,
                "pm_vs_xbatch": mp / mx, "pm_vs_singleton_x_b": mp / ms, "mismatch_bytes_vs_xbatch": bad,
            }));
        }
        shapes.push(json!({"shape": name, "source": rel, "M": p.m, "K": p.k, "rows": rows}));
    }
    Ok(json!({"warmups": 5, "samples": 15, "unit": "us (HIP events)", "shapes": shapes}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (Some(mode), Some(co), Some(out)) = (args.get(1).cloned(), flag("--co"), flag("--out"))
    else {
        eprintln!("usage: pm_xbatch_oracle verify|time --co FILE --out FILE.json");
        std::process::exit(2);
    };
    let run = || -> Result<Value> {
        let mut gpu = Gpu::init().map_err(err)?;
        gpu.ensure_kernel_public(XBATCH, XBATCH_SRC, XBATCH)
            .map_err(err)?;
        let cand = load_candidate(&gpu, &co)?;
        let body = match mode.as_str() {
            "verify" => verify(&mut gpu, &cand)?,
            "time" => time(&mut gpu, &cand)?,
            m => return Err(format!("unknown mode {m}")),
        };
        Ok(json!({
            "schema": "pm-xbatch-oracle-v1", "mode": mode, "symbol": SYMBOL,
            "candidate": co, "candidate_sha256": cand.sha256,
            "incumbent": XBATCH, "incumbent_source_sha256": sha256_hex(XBATCH_SRC.as_bytes()),
            "pid": std::process::id(), "result": body,
        }))
    };
    match run() {
        Ok(v) => {
            std::fs::write(&out, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
            let bad = v["result"]["candidate_mismatch_bytes"]
                .as_u64()
                .unwrap_or(0)
                + v["result"]["cross_mismatch_bytes"].as_u64().unwrap_or(0);
            eprintln!("wrote {out}");
            std::process::exit(if bad == 0 { 0 } else { 1 });
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
