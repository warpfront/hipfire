// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of `Gpu::gemv_mq2g256v2_xbatch` (the MTP draft head's batched ranking
//! GEMV over the MQ2G256V2 copy): for every shape and B = 1..8, the whole `y` of the
//! x-batched kernel (production chunks of eight on the gfx1201 PeaceMaker route,
//! four on hipcc; `HIPFIRE_QWEN4_MTP_DRAFT_HEAD_PM=0` selects hipcc)
//! must equal the bytes of `Gpu::gemv_mq2g256v2` launched once per column. Random
//! weights (finite f16 scale / zero-point headers, random packed 2-bit payloads) and
//! random activations; shapes cover K = 256 .. 4352 and every M residue mod 4.
//!
//! On a GPU the PeaceMaker module covers (gfx1201), every case also runs the module
//! and the hipcc kernel explicitly (`gemv_mq2g256v2_xbatch_with`) and compares their
//! whole `y` (written rows and the 0x7f guard past them). `--captures MANIFEST` adds
//! the WideNet F05 served captures (real ranking-copy weights and draft rows; each
//! case's PM and hipcc `y` against each other and against the captured `y`).
//!
//! `mq2_xbatch_oracle [--captures MANIFEST] --out FILE.json`; exit 0 only when every
//! case matches.
//!
//! DIRECT single-launch proof and bench of an arbitrary emitted PM object (no Gpu wrapper, so
//! no 4-row chunking): the object is loaded with `hipModuleLoadData` and launched through the
//! normal kernel ABI (`A, x, y, M, K, B`, grid `ceil(M/rows)`, 32 lanes). Default object is the
//! shipped `GEMV_MQ2_XBATCH_PM_GFX1201`; `--object PATH` replaces it, `--sweep-dir DIR` runs
//! every `*.hxaco` in DIR (rows taken from an `r1`/`r2`/`r4` token in the file stem, else
//! `--rows`). Options: `--rows N` (output rows per block, default 4), `--max-b N` (largest B
//! one launch must cover, default 8; pass 4 for a cap4 object), `--symbol S` (default
//! `gemv_mq2g256v2_xbatch_pm`).
//!
//! Correctness (runs unless `--bench --no-oracle`): for every finite random shape (K = 256 .. 4352,
//! every M residue, small and served-size shapes) and B = 1..=max-b, ONE object launch into
//! a y buffer guarded by 0x7f on both sides must equal per-column `Gpu::gemv_mq2g256v2` bytes.
//! Additional M7 K768/K1280 and M33 K768 cases exercise half-header NaN payloads/Inf/zeros/
//! subnormals/extremes and activation NaN/Inf/zero/subnormal/extreme words.
//!
//! Gating: finite random shapes must equal per-column singleton bytes. NaN/Inf edge cases are
//! gated direct == hipcc-chunked (and == the `--baseline-object` PM chunked by 4 when given),
//! raw bytes; the singleton-vs-xbatch NaN-payload disagreement (operand order of compiled
//! add/mul: hardware returns the first NaN source, so commuted operands change the payload)
//! is recorded as `preexisting_singleton_*` with counts and first words, not gated.
//!
//! `--bench` times the object's single launch (B = 1..8) against the TODAY cap4 baseline:
//! `--baseline-object PATH` (REQUIRED with `--bench`; must be the pre-change shipped cap4
//! object, run with 4 rows per block in chunks of 4 so B = 5..8 is two launches; the new
//! kernel is never timed as the baseline). Synthetic M=100276 K=2560 weights, 3 rotating copies
//! (3 x 72 MB > the 64 MB MALL), 18 calls per HIP graph, `--bench-reps N` replays (default 15),
//! median. JSON per B: `single_us`, `chunked_us`, `saving_us = chunked - single`; no fit. The
//! single-launch y must equal the chunked y bytes. `--no-oracle` skips the legacy wrapper
//! sweep and, with `--bench`, correctness (timing mode). Correctness mode and timing mode are
//! separate invocations:
//!   mq2_xbatch_oracle --sweep-dir DIR --out verify.json
//!   mq2_xbatch_oracle --sweep-dir DIR --bench --no-oracle \
//!       --baseline-object baseline_cap4.hxaco --out time.json

use rdna_compute::{DType, Gpu};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

const BENCH_CALLS: usize = 18;
const BENCH_COPIES: usize = 3;

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// xorshift64* stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// `m` rows of `k / 256` 72-byte groups: two (scale, zero-point) f16 pairs, then 64
/// packed bytes.
fn weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = m * (k / 256);
    let mut bytes = Vec::with_capacity(groups * 72);
    for _ in 0..groups {
        for _ in 0..2 {
            let scale = 0.01 + 0.5 * (rng.f32().abs());
            let zero = -0.5 * scale * 3.0 + 0.05 * rng.f32();
            bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f16_bits(zero).to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    }
    bytes
}

/// Whole-`y` bytes of the module (`pm`) or the hipcc kernel over `b` rows, the
/// buffer pre-filled with 0x7f.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    a: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    b: usize,
    pm: bool,
) -> Result<Vec<u8>> {
    gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
    gpu.gemv_mq2g256v2_xbatch_with(a, x, y, m, k, b, pm)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.download_raw_bytes(y).map_err(err)
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// One case per F05 capture of `manifest` (blobs under `<manifest dir>/blobs/`).
fn capture_cases(
    gpu: &mut Gpu,
    manifest: &str,
    cases: &mut Vec<serde_json::Value>,
) -> Result<usize> {
    let text = std::fs::read_to_string(manifest).map_err(err)?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(err)?;
    let dir = std::path::Path::new(manifest)
        .parent()
        .ok_or("manifest has no directory")?
        .join("blobs");
    let blob = |sha: &str| -> Result<Vec<u8>> {
        std::fs::read(dir.join(&sha[..2]).join(sha)).map_err(err)
    };
    let mut bad_total = 0;
    let mut seen = 0;
    for case in value["cases"].as_array().ok_or("no cases")? {
        if case["family"] != "F05" {
            continue;
        }
        seen += 1;
        let s = &case["scalars"];
        let (m, k, b) = (
            s["M"].as_u64().ok_or("M")? as usize,
            s["K"].as_u64().ok_or("K")? as usize,
            s["B"].as_u64().ok_or("B")? as usize,
        );
        let region = |id: &str| -> Result<&serde_json::Value> {
            case["regions"]
                .as_array()
                .and_then(|r| r.iter().find(|r| r["id"] == id))
                .ok_or_else(|| format!("region {id}"))
        };
        let a_host = blob(region("r0")?["pre_blob"].as_str().ok_or("r0")?)?;
        let x_host = blob(region("r1")?["pre_blob"].as_str().ok_or("r1")?)?;
        let y_captured = blob(region("r2")?["post_blob"].as_str().ok_or("r2")?)?;
        let a = gpu.upload_raw(&a_host, &[a_host.len()]).map_err(err)?;
        let x = gpu.upload_raw(&x_host, &[x_host.len()]).map_err(err)?;
        let y = gpu.zeros(&[(b + 1) * m], DType::F32).map_err(err)?;
        let hip = launch(gpu, &a, &x, &y, m, k, b, false)?;
        let pm = launch(gpu, &a, &x, &y, m, k, b, true)?;
        let written = b * m * 4;
        let pm_vs_hip = diff(&pm, &hip);
        let pm_vs_capture = diff(&pm[..written], &y_captured[..written.min(y_captured.len())]);
        bad_total += pm_vs_hip + pm_vs_capture;
        cases.push(json!({
            "capture": case["case_id"], "m": m, "k": k, "b": b,
            "pm_vs_hipcc_bytes": pm_vs_hip, "pm_vs_captured_y_bytes": pm_vs_capture,
        }));
        gpu.free_tensor(a).map_err(err)?;
        gpu.free_tensor(x).map_err(err)?;
        gpu.free_tensor(y).map_err(err)?;
    }
    if seen == 0 {
        return Err("no F05 captures in the manifest".into());
    }
    Ok(bad_total)
}

/// Per-call us of each replay (replay order) of `calls` launches captured in one graph (`reps`
/// synchronized replays after three untimed ones); `copies` weight copies rotate across calls.
fn graph_samples(
    gpu: &mut Gpu,
    reps: usize,
    copies: usize,
    mut f: impl FnMut(&mut Gpu, usize) -> Result<()>,
) -> Result<Vec<f64>> {
    for i in 0..copies {
        f(gpu, i)?;
    }
    gpu.ensure_capture_stream().map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.begin_stream_capture().map_err(err)?;
    for i in 0..BENCH_CALLS {
        f(gpu, i)?;
    }
    let graph = gpu.end_stream_capture().map_err(err)?;
    let exec = gpu.hip.graph_instantiate(&graph).map_err(err)?;
    for _ in 0..3 {
        gpu.launch_graph(&exec).map_err(err)?;
    }
    let mut s = Vec::new();
    for _ in 0..reps {
        gpu.hip.device_synchronize().map_err(err)?;
        let t = Instant::now();
        gpu.launch_graph(&exec).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        s.push(t.elapsed().as_secs_f64() * 1e6 / BENCH_CALLS as f64);
    }
    gpu.hip.graph_exec_destroy(exec).map_err(err)?;
    gpu.hip.graph_destroy(graph).map_err(err)?;
    Ok(s)
}

/// Median us per call of `calls` launches captured in one graph (`reps` synchronized replays).
fn graph_us(
    gpu: &mut Gpu,
    reps: usize,
    f: impl FnMut(&mut Gpu, usize) -> Result<()>,
) -> Result<f64> {
    let mut s = graph_samples(gpu, reps, BENCH_COPIES, f)?;
    s.sort_by(f64::total_cmp);
    Ok(s[s.len() / 2])
}

const GUARD_F: usize = 64;
const DEFAULT_SYMBOL: &str = "gemv_mq2g256v2_xbatch_pm";

/// One emitted object loaded through the HIP module loader.
struct Obj {
    label: String,
    sha: String,
    rows: usize,
    /// rip-search candidate id and axes (manifest objects only).
    id: Option<String>,
    axes: serde_json::Value,
    /// How the bytes reached `hipModuleLoadData`: the raw image, or ELF wrapped in a bundle.
    wrap: &'static str,
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn load_obj(gpu: &Gpu, label: &str, bytes: &[u8], symbol: &str, rows: usize) -> Result<Obj> {
    if !matches!(rows, 1 | 2 | 4) {
        return Err(format!("{label}: --rows must be 1, 2 or 4, got {rows}"));
    }
    let module = gpu.hip.module_load_data(bytes).map_err(err)?;
    let func = gpu.hip.module_get_function(&module, symbol).map_err(err)?;
    Ok(Obj {
        label: label.to_string(),
        sha: sha256_hex(bytes),
        rows,
        id: None,
        axes: serde_json::Value::Null,
        wrap: "raw",
        _module: module,
        func,
    })
}

/// [`load_obj`] of `bytes`; when the loader refuses the raw image (a bare ELF) on gfx1201,
/// retry once with the same `__CLANG_OFFLOAD_BUNDLE__` writer the repo uses for `.hxaco`.
/// `sha` stays the sha256 of the bytes given.
fn load_elf_obj(gpu: &Gpu, label: &str, bytes: &[u8], symbol: &str, rows: usize) -> Result<Obj> {
    match load_obj(gpu, label, bytes, symbol, rows) {
        Ok(o) => Ok(o),
        Err(raw_err) if gpu.arch == "gfx1201" && matches!(rows, 1 | 2 | 4) => {
            use peacemaker_isa::{arch::Arch, native};
            let wrapped = native::bundle(bytes, Arch::Gfx1201, native::DEFAULT_HOST_TARGET);
            let mut o = load_obj(gpu, label, &wrapped, symbol, rows)
                .map_err(|e| format!("{label}: raw load: {raw_err}; bundled load: {e}"))?;
            o.sha = sha256_hex(bytes);
            o.wrap = "bundle";
            Ok(o)
        }
        Err(e) => Err(e),
    }
}

/// ONE launch of `obj` over columns `first..first + b` (x/y column strides `k`/`m`), on the
/// active stream (so it is captured by `graph_us`).
#[allow(clippy::too_many_arguments)]
fn launch_obj(
    gpu: &Gpu,
    obj: &Obj,
    a: &rdna_compute::GpuTensor,
    x: u64,
    y: u64,
    m: usize,
    k: usize,
    b: usize,
) -> Result<()> {
    let mut ka = Vec::with_capacity(40);
    for p in [a.buf.as_ptr() as u64, x, y] {
        ka.extend_from_slice(&p.to_le_bytes());
    }
    for d in [m, k, b] {
        ka.extend_from_slice(&(d as i32).to_le_bytes());
    }
    ka.resize(40, 0);
    // SAFETY: kernarg follows the A,x,y,M,K,B ABI and addresses live allocations.
    unsafe {
        gpu.hip.launch_kernel_blob(
            &obj.func,
            [m.div_ceil(obj.rows) as u32, 1, 1],
            [32, 1, 1],
            0,
            gpu.active_stream.as_ref(),
            &mut ka,
        )
    }
    .map_err(err)
}

/// Baseline: the cap4 object in chunks of 4 columns (production chunking).
#[allow(clippy::too_many_arguments)]
fn launch_chunked(
    gpu: &Gpu,
    obj: &Obj,
    a: &rdna_compute::GpuTensor,
    x: u64,
    y: u64,
    m: usize,
    k: usize,
    b: usize,
) -> Result<()> {
    launch_chunked_by(gpu, obj, a, x, y, m, k, b, 4)
}

/// `obj` in chunks of `chunk` columns (`chunk >= b`: one launch).
#[allow(clippy::too_many_arguments)]
fn launch_chunked_by(
    gpu: &Gpu,
    obj: &Obj,
    a: &rdna_compute::GpuTensor,
    x: u64,
    y: u64,
    m: usize,
    k: usize,
    b: usize,
    chunk: usize,
) -> Result<()> {
    let mut start = 0;
    while start < b {
        let rows = chunk.min(b - start);
        launch_obj(
            gpu,
            obj,
            a,
            x + (start * k * 4) as u64,
            y + (start * m * 4) as u64,
            m,
            k,
            rows,
        )?;
        start += rows;
    }
    Ok(())
}

/// Half bit patterns for header scale/zero-point: distinct NaN payloads (quiet, signalling,
/// negative), +-Inf, +-0, subnormals, extremes, min normal, ordinary values.
const EDGE_HALF: [u16; 20] = [
    0x7e00, 0x7c01, 0xfe55, 0x7dff, 0xfc01, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03ff,
    0x83ff, 0x7bff, 0xfbff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3555,
];

/// MQ2 weights whose every (scale, zero-point) half is drawn from `EDGE_HALF`; payloads random.
fn edge_weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = m * (k / 256);
    let mut bytes = Vec::with_capacity(groups * 72);
    for _ in 0..groups {
        for _ in 0..4 {
            let h = EDGE_HALF[(rng.next() >> 33) as usize % EDGE_HALF.len()];
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    }
    bytes
}

/// `[8][k]` activations, one pattern per column: finite; sparse NaN payloads; +-Inf (Inf-Inf
/// possible); signed zeros; subnormals; +-f32::MAX; dense distinct NaN payloads; Inf+NaN mix.
fn edge_x(rng: &mut Rng, k: usize) -> Vec<f32> {
    const SPECIAL: [u32; 14] = [
        0x7fc0_0001,
        0xffc0_1234,
        0x7f80_0001,
        0xff80_0abc,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0080_0000,
    ];
    let mut x = Vec::with_capacity(8 * k);
    for col in 0..8usize {
        for i in 0..k {
            let r = rng.next();
            let pick = |set: &[u32]| f32::from_bits(set[(r >> 33) as usize % set.len()]);
            let sparse = (r & 15) == 0;
            let v = match col {
                0 => rng.f32() * 3.0,
                1 if sparse => pick(&SPECIAL[..4]),
                2 if sparse => pick(&SPECIAL[4..6]),
                3 => pick(&SPECIAL[4..6]),
                4 => pick(&SPECIAL[6..8]),
                5 => pick(&SPECIAL[8..11]),
                6 => f32::from_bits(
                    0x7fc0_0000 | ((i as u32 * 2654435761 ^ col as u32) & 0x003f_ffff),
                ),
                7 if (r & 7) == 0 => pick(&SPECIAL),
                7 => pick(&SPECIAL[11..13]),
                _ => rng.f32() * 3.0,
            };
            x.push(v);
        }
    }
    x
}

/// First `limit` differing f32 words of the written region (`b` columns of `m` rows), as
/// `{col,row,reference,got}` hex bit patterns; `other` (when given) adds the third arm's word.
fn first_words(
    reference: &[u8],
    got: &[u8],
    other: Option<&[u8]>,
    m: usize,
    written: usize,
    limit: usize,
) -> Vec<serde_json::Value> {
    let word = |v: &[u8], i: usize| {
        u32::from_le_bytes([v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]])
    };
    let mut out = Vec::new();
    for i in 0..written / 4 {
        if reference[4 * i..4 * i + 4] != got[4 * i..4 * i + 4] {
            let mut e = json!({
                "col": i / m, "row": i % m,
                "reference": format!("{:#010x}", word(reference, i)),
                "got": format!("{:#010x}", word(got, i)),
            });
            if let Some(o) = other {
                e["other"] = json!(format!("{:#010x}", word(o, i)));
            }
            out.push(e);
            if out.len() == limit {
                break;
            }
        }
    }
    out
}

/// (differing words, both-NaN words, words where at least one side is non-NaN).
fn nan_class(reference: &[u8], got: &[u8]) -> (usize, usize, usize) {
    let w = |v: &[u8], i: usize| {
        f32::from_bits(u32::from_le_bytes([
            v[4 * i],
            v[4 * i + 1],
            v[4 * i + 2],
            v[4 * i + 3],
        ]))
    };
    let (mut n, mut both, mut other) = (0, 0, 0);
    for i in 0..reference.len() / 4 {
        if reference[4 * i..4 * i + 4] != got[4 * i..4 * i + 4] {
            n += 1;
            if w(reference, i).is_nan() && w(got, i).is_nan() {
                both += 1;
            } else {
                other += 1;
            }
        }
    }
    (n, both, other)
}

/// Direct correctness of `obj`: single launch at B = 1..=max_b vs per-column singleton bytes
/// (guards both sides included); hipcc in chunks of <= 4 vs the same bytes.
fn direct_cases(
    gpu: &mut Gpu,
    obj: &Obj,
    base: Option<&Obj>,
    shapes: Option<&[(usize, usize)]>,
    b_list: &[usize],
) -> Result<(Vec<serde_json::Value>, usize)> {
    let mut rng = Rng(0xD1EC7_5EED_0001);
    let mut cases = Vec::new();
    let mut bad_total = 0usize;
    // (kind, m, k): "random" = finite synthetic; "edge_hdr" = edge half headers + edge x;
    // "edge_x" = finite headers + edge x.
    let mut specs: Vec<(&str, usize, usize)> = [
        (1usize, 256usize),
        (2, 512),
        (3, 1024),
        (4, 256),
        (5, 256),
        (6, 512),
        (7, 256),
        (8, 256),
        (9, 768),
        (15, 256),
        (31, 768),
        (130, 2560),
        (1001, 2560),
        (4093, 2048),
        (33, 4352),
        (100003, 2560),
        (100276, 2560),
    ]
    .into_iter()
    .map(|(m, k)| ("random", m, k))
    .collect();
    if let Some(shapes) = shapes {
        // --shape replaces the random-finite corpus; the (tiny) NaN/Inf edge cases below stay.
        specs = shapes.iter().map(|&(m, k)| ("random", m, k)).collect();
    }
    for (m, k) in [(7usize, 768usize), (7, 1280), (33, 768)] {
        specs.push(("edge_hdr", m, k));
        specs.push(("edge_x", m, k));
    }
    for (kind, m, k) in specs {
        let a_bytes = if kind == "edge_hdr" {
            edge_weights(&mut rng, m, k)
        } else {
            weights(&mut rng, m, k)
        };
        let a = gpu.upload_raw(&a_bytes, &[a_bytes.len()]).map_err(err)?;
        let x_host: Vec<f32> = if kind == "random" {
            (0..8 * k).map(|_| rng.f32() * 3.0).collect()
        } else {
            edge_x(&mut rng, k)
        };
        let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
        let y_ref = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
        let y = gpu.zeros(&[8 * m + 2 * GUARD_F], DType::F32).map_err(err)?;
        let y_view = y.sub_offset(GUARD_F, 8 * m);
        for column in 0..8 {
            gpu.gemv_mq2g256v2(
                &a,
                &x.sub_offset(column * k, k),
                &y_ref.sub_offset(column * m, m),
                m,
                k,
            )
            .map_err(err)?;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        let reference = gpu.download_raw_bytes(&y_ref).map_err(err)?;
        for &b in b_list {
            let written = b * m * 4;
            let guard = GUARD_F * 4;
            // single launch of the object
            gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
            launch_obj(
                gpu,
                obj,
                &a,
                x.buf.as_ptr() as u64,
                y_view.buf.as_ptr() as u64,
                m,
                k,
                b,
            )?;
            gpu.hip.device_synchronize().map_err(err)?;
            let got = gpu.download_raw_bytes(&y).map_err(err)?;
            let guards_bad = |v: &[u8]| {
                v[..guard].iter().filter(|&&q| q != 0x7f).count()
                    + v[guard + written..].iter().filter(|&&q| q != 0x7f).count()
            };
            let bad_guard = guards_bad(&got);
            // hipcc kernel, chunks of <= 4, same guards
            gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
            gpu.gemv_mq2g256v2_xbatch_with(&a, &x, &y_view, m, k, b, false)
                .map_err(err)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let hip = gpu.download_raw_bytes(&y).map_err(err)?;
            // incumbent PM object (cap4, chunks of 4), when --baseline-object supplied
            let old = match base {
                Some(base) => {
                    gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
                    launch_chunked(
                        gpu,
                        base,
                        &a,
                        x.buf.as_ptr() as u64,
                        y_view.buf.as_ptr() as u64,
                        m,
                        k,
                        b,
                    )?;
                    gpu.hip.device_synchronize().map_err(err)?;
                    Some(gpu.download_raw_bytes(&y).map_err(err)?)
                }
                None => None,
            };
            let (r_w, g_w, h_w) = (
                &reference[..written],
                &got[guard..guard + written],
                &hip[guard..guard + written],
            );
            let o_w = old.as_ref().map(|o| &o[guard..guard + written]);
            let sing_direct = diff(g_w, r_w);
            let sing_hipcc = diff(h_w, r_w) + guards_bad(&hip);
            let mut entry = json!({
                "kind": kind, "m": m, "k": k, "b": b,
                "single_launch_guard_overwritten_bytes": bad_guard,
                "first_direct_mismatch_words": first_words(r_w, g_w, Some(h_w), m, written, 8),
                "first_hipcc_mismatch_words": first_words(r_w, h_w, Some(g_w), m, written, 8),
            });
            let mut gate = bad_guard;
            if kind == "random" {
                // finite: per-column singleton is the gate
                gate += sing_direct + sing_hipcc;
                entry["single_launch_mismatch_bytes"] = json!(sing_direct);
                entry["hipcc_chunked_mismatch_bytes"] = json!(sing_hipcc);
                if let Some(o_w) = o_w {
                    let d = diff(g_w, o_w);
                    gate += d;
                    entry["direct_vs_baseline_mismatch_bytes"] = json!(d);
                }
            } else {
                // NaN/Inf edges: the gate is direct == hipcc chunked (== old PM chunked) raw bytes;
                // the singleton disagreement is a recorded pre-existing diagnostic, not gated.
                let vs_hip = diff(g_w, h_w);
                gate += vs_hip;
                entry["direct_vs_hipcc_chunked_mismatch_bytes"] = json!(vs_hip);
                entry["first_direct_vs_hipcc_words"] =
                    json!(first_words(h_w, g_w, None, m, written, 8));
                if let Some(o_w) = o_w {
                    let d = diff(g_w, o_w);
                    gate += d + guards_bad(old.as_ref().unwrap());
                    entry["direct_vs_baseline_mismatch_bytes"] = json!(d);
                    entry["baseline_guard_overwritten_bytes"] =
                        json!(guards_bad(old.as_ref().unwrap()));
                    entry["first_direct_vs_baseline_words"] =
                        json!(first_words(o_w, g_w, None, m, written, 8));
                    entry["preexisting_baseline_vs_hipcc_mismatch_bytes"] = json!(diff(o_w, h_w));
                    entry["preexisting_singleton_baseline_mismatch_bytes"] = json!(diff(o_w, r_w));
                }
                let (dw, dnan, dother) = nan_class(r_w, g_w);
                let (hw, hnan, hother) = nan_class(r_w, h_w);
                entry["preexisting_singleton_direct"] = json!({
                    "mismatch_bytes": sing_direct, "mismatch_words": dw,
                    "both_nan_words": dnan, "non_nan_involved_words": dother,
                });
                entry["preexisting_singleton_hipcc"] = json!({
                    "mismatch_bytes": sing_hipcc, "mismatch_words": hw,
                    "both_nan_words": hnan, "non_nan_involved_words": hother,
                });
            }
            entry["gated_bad_bytes"] = json!(gate);
            bad_total += gate;
            cases.push(entry);
        }
        gpu.free_tensor(a).map_err(err)?;
        gpu.free_tensor(x).map_err(err)?;
        gpu.free_tensor(y_ref).map_err(err)?;
        gpu.free_tensor(y).map_err(err)?;
        // `upload_raw` takes a fresh `hipMalloc` (never the pool) and `free_tensor` parks the
        // buffer in the pool, so without a drain every object leaked this case's ~72 MB
        // weights (hipMalloc OOM after ~222 objects in one process, j8589).
        gpu.drain_pool();
    }
    Ok((cases, bad_total))
}

/// B = 1..8 single launch of `obj` vs `base` (cap4, chunked by 4) at the bench shape.
fn bench_table(
    gpu: &mut Gpu,
    obj: &Obj,
    base: &Obj,
    reps: usize,
) -> Result<(serde_json::Value, usize)> {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (m, k) = (100276usize, 2560usize);
    let mut ws = Vec::new();
    for _ in 0..BENCH_COPIES {
        let bytes = weights(&mut rng, m, k);
        ws.push(gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?);
    }
    let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32() * 3.0).collect();
    let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
    let y = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
    let y2 = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
    let (xp, yp, y2p) = (
        x.buf.as_ptr() as u64,
        y.buf.as_ptr() as u64,
        y2.buf.as_ptr() as u64,
    );
    let mut rows_json = Vec::new();
    let mut bad = 0usize;
    println!(
        "mq2 xbatch bench: M={m} K={k} object={} rows={} vs baseline={} rows={} reps={reps} calls/replay={BENCH_CALLS} copies={BENCH_COPIES}",
        obj.label, obj.rows, base.label, base.rows
    );
    for b in 1..=8usize {
        // same bytes check on copy 0
        gpu.hip.memset(&y.buf, 0x7f, 8 * m * 4).map_err(err)?;
        gpu.hip.memset(&y2.buf, 0x7f, 8 * m * 4).map_err(err)?;
        launch_obj(gpu, obj, &ws[0], xp, yp, m, k, b)?;
        launch_chunked(gpu, base, &ws[0], xp, y2p, m, k, b)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let same = diff(
            &gpu.download_raw_bytes(&y).map_err(err)?,
            &gpu.download_raw_bytes(&y2).map_err(err)?,
        );
        bad += same;
        // interleaved arms: single, chunked
        let single = graph_us(gpu, reps, |g, i| {
            launch_obj(g, obj, &ws[i % BENCH_COPIES], xp, yp, m, k, b)
        })?;
        let chunked = graph_us(gpu, reps, |g, i| {
            launch_chunked(g, base, &ws[i % BENCH_COPIES], xp, y2p, m, k, b)
        })?;
        println!(
            "  B={b}: single {single:.2} us | chunked({} launch) {chunked:.2} us | saving {:.2} us | bytes_diff {same}",
            b.div_ceil(4), chunked - single
        );
        rows_json.push(json!({
            "b": b, "single_us": single, "chunked_us": chunked, "saving_us": chunked - single,
            "chunked_launches": b.div_ceil(4), "single_vs_chunked_mismatch_bytes": same,
        }));
    }
    for t in ws {
        gpu.free_tensor(t).map_err(err)?;
    }
    for t in [x, y, y2] {
        gpu.free_tensor(t).map_err(err)?;
    }
    gpu.drain_pool();
    Ok((
        json!({
            "m": m, "k": k, "reps": reps, "calls_per_replay": BENCH_CALLS,
            "weight_copies": BENCH_COPIES, "weight": "synthetic",
            "baseline": base.label, "baseline_sha256": base.sha, "baseline_rows": base.rows,
            "table": rows_json,
        }),
        bad,
    ))
}

/// Weight copies for one shape: `max(3, ceil(200 MB / copy bytes))` (formula), of which at
/// most `BENCH_CALLS` can be touched by one graph replay (the rest would stay idle), so that
/// many are allocated. Returns `(formula, allocated)`.
fn copies_for(m: usize, k: usize) -> (usize, usize) {
    let copy_bytes = (m * (k / 256) * 72).max(1);
    let formula = 200_000_000usize.div_ceil(copy_bytes).max(3);
    (formula, formula.min(BENCH_CALLS))
}

/// Linear-interpolated quantile of an ascending slice.
fn pct(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// (median = `s[len/2]` like `graph_us`, p10, p90) of per-call us replays.
fn stats(replays: &[f64]) -> (f64, f64, f64) {
    let mut s = replays.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], pct(&s, 0.10), pct(&s, 0.90))
}

/// Interleaved bench: per (shape, B) the arms run ref, c1, ref, c2, ..., ref. `base` is the
/// ref (chunks of `base_cap` columns); every `objects` entry is one single launch. Each object's
/// whole guarded `y` (0x7f memset included) must equal the ref's on weight copy 0 before it is
/// timed; a mismatching object is not timed in that cell.
#[allow(clippy::too_many_arguments)]
fn interleave_bench(
    gpu: &mut Gpu,
    objects: &[Obj],
    base: &Obj,
    base_cap: usize,
    shapes: &[(usize, usize)],
    b_list: &[usize],
    reps: usize,
    seed: u64,
) -> Result<(serde_json::Value, usize)> {
    let mut bad = 0usize;
    let mut shapes_json = Vec::new();
    for &(m, k) in shapes {
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(format!(
                "shape {m},{k}: K must be a positive multiple of 256"
            ));
        }
        let (formula, copies) = copies_for(m, k);
        let copy_bytes = m * (k / 256) * 72;
        let mut rng = Rng(seed);
        let mut ws = Vec::new();
        for _ in 0..copies {
            let bytes = weights(&mut rng, m, k);
            ws.push(gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?);
        }
        let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32() * 3.0).collect();
        let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
        let y_ref = gpu.zeros(&[8 * m + 2 * GUARD_F], DType::F32).map_err(err)?;
        let y_c = gpu.zeros(&[8 * m + 2 * GUARD_F], DType::F32).map_err(err)?;
        let (xp, yrp, ycp) = (
            x.buf.as_ptr() as u64,
            y_ref.sub_offset(GUARD_F, 8 * m).buf.as_ptr() as u64,
            y_c.sub_offset(GUARD_F, 8 * m).buf.as_ptr() as u64,
        );
        println!(
            "interleave: M={m} K={k} copies={copies} (formula {formula}, {copy_bytes} B each) reps={reps} calls/replay={BENCH_CALLS} ref={} cap={base_cap}",
            base.label
        );
        let guard = GUARD_F * 4;
        let mut cells = Vec::new();
        for &b in b_list {
            let written = b * m * 4;
            gpu.hip
                .memset(&y_ref.buf, 0x7f, y_ref.numel() * 4)
                .map_err(err)?;
            launch_chunked_by(gpu, base, &ws[0], xp, yrp, m, k, b, base_cap)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let ref_bytes = gpu.download_raw_bytes(&y_ref).map_err(err)?;
            let ref_guard_bad = ref_bytes[..guard].iter().filter(|&&q| q != 0x7f).count()
                + ref_bytes[guard + written..]
                    .iter()
                    .filter(|&&q| q != 0x7f)
                    .count();
            bad += ref_guard_bad;
            let mut mism = Vec::new();
            let mut timed = Vec::new();
            for (oi, obj) in objects.iter().enumerate() {
                gpu.hip
                    .memset(&y_c.buf, 0x7f, y_c.numel() * 4)
                    .map_err(err)?;
                launch_obj(gpu, obj, &ws[0], xp, ycp, m, k, b)?;
                gpu.hip.device_synchronize().map_err(err)?;
                let got = gpu.download_raw_bytes(&y_c).map_err(err)?;
                let d = diff(&got, &ref_bytes);
                bad += d;
                mism.push(d);
                if d == 0 {
                    timed.push(oi);
                }
            }
            let ref_arm = |gpu: &mut Gpu| -> Result<Vec<f64>> {
                graph_samples(gpu, reps, copies, |g, i| {
                    launch_chunked_by(g, base, &ws[i % copies], xp, yrp, m, k, b, base_cap)
                })
            };
            let mut ref_arms: Vec<Vec<f64>> = Vec::new();
            let mut obj_arms: Vec<Vec<f64>> = Vec::new();
            if !timed.is_empty() {
                ref_arms.push(ref_arm(gpu)?);
                for &oi in &timed {
                    let obj = &objects[oi];
                    obj_arms.push(graph_samples(gpu, reps, copies, |g, i| {
                        launch_obj(g, obj, &ws[i % copies], xp, ycp, m, k, b)
                    })?);
                    ref_arms.push(ref_arm(gpu)?);
                }
            }
            let ref_meds: Vec<f64> = ref_arms.iter().map(|a| stats(a).0).collect();
            let pooled: Vec<f64> = ref_arms.iter().flatten().copied().collect();
            let ref_json = if pooled.is_empty() {
                serde_json::Value::Null
            } else {
                let (pm, p10, p90) = stats(&pooled);
                let spread = ref_meds.iter().cloned().fold(f64::MIN, f64::max)
                    - ref_meds.iter().cloned().fold(f64::MAX, f64::min);
                json!({
                    "arms": ref_arms.len(), "reps": reps, "arm_medians_us": ref_meds,
                    "aa_spread_us": spread, "pooled_median_us": pm, "pooled_p10_us": p10,
                    "pooled_p90_us": p90, "guard_overwritten_bytes": ref_guard_bad,
                    "arm_replays_us": ref_arms,
                })
            };
            let mut objs_json = Vec::new();
            for (oi, obj) in objects.iter().enumerate() {
                let mut e = json!({
                    "object": obj.label, "sha256": obj.sha,
                    "hw_bitwise": mism[oi] == 0, "mismatch_bytes": mism[oi],
                    "timed": mism[oi] == 0,
                });
                if let Some(id) = &obj.id {
                    e["id"] = json!(id);
                }
                if let Some(j) = timed.iter().position(|&t| t == oi) {
                    let (med, p10, p90) = stats(&obj_arms[j]);
                    e["median_us"] = json!(med);
                    e["p10_us"] = json!(p10);
                    e["p90_us"] = json!(p90);
                    e["reps"] = json!(reps);
                    e["ref_before_median_us"] = json!(ref_meds[j]);
                    e["ref_after_median_us"] = json!(ref_meds[j + 1]);
                    e["replays_us"] = json!(obj_arms[j]);
                    println!(
                        "  B={b} {}: {med:.2} us (p10 {p10:.2} p90 {p90:.2}) ref {:.2}/{:.2}",
                        obj.label,
                        ref_meds[j],
                        ref_meds[j + 1]
                    );
                } else {
                    println!(
                        "  B={b} {}: NOT TIMED, {} mismatching bytes",
                        obj.label, mism[oi]
                    );
                }
                objs_json.push(e);
            }
            cells.push(json!({"b": b, "ref": ref_json, "objects": objs_json}));
        }
        for t in ws {
            gpu.free_tensor(t).map_err(err)?;
        }
        for t in [x, y_ref, y_c] {
            gpu.free_tensor(t).map_err(err)?;
        }
        shapes_json.push(json!({
            "m": m, "k": k, "weight_copies": copies, "weight_copies_formula": formula,
            "copy_bytes": copy_bytes, "cells": cells,
        }));
    }
    Ok((
        json!({
            "order": "ref,c1,ref,c2,...,ref", "reps": reps, "calls_per_replay": BENCH_CALLS,
            "untimed_warm_replays": 3, "median": "sorted[len/2]",
            "p10_p90": "linear-interpolated quantiles", "per_call_us_of_replay": true,
            "ref": base.label, "ref_sha256": base.sha, "ref_rows": base.rows, "ref_cap": base_cap,
            "shapes": shapes_json,
        }),
        bad,
    ))
}

fn parse_u64(s: &str) -> u64 {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    }
    .unwrap_or_else(|_| panic!("bad integer {s}"))
}

/// Rows per block from an `r1`/`r2`/`r4` token of a file stem.
fn rows_from_stem(stem: &str) -> Option<usize> {
    stem.split(|c: char| !c.is_ascii_alphanumeric())
        .find_map(|t| match t {
            "r1" => Some(1),
            "r2" => Some(2),
            "r4" => Some(4),
            _ => None,
        })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let out = flag("--out");
    let captures = flag("--captures");
    let do_bench = args.iter().any(|a| a == "--bench");
    let no_oracle = args.iter().any(|a| a == "--no-oracle");
    let object = flag("--object");
    let sweep_dir = flag("--sweep-dir");
    let baseline_object = flag("--baseline-object");
    let symbol = flag("--symbol").unwrap_or_else(|| DEFAULT_SYMBOL.to_string());
    let rows: usize = flag("--rows").map_or(4, |v| v.parse().expect("--rows N"));
    let max_b: usize = flag("--max-b").map_or(8, |v| v.parse().expect("--max-b N"));
    let bench_reps: usize = flag("--bench-reps").map_or(15, |v| v.parse().expect("--bench-reps N"));
    // rip-search flags; any of them selects schema v3 and the `inputs` record.
    let manifest_path = flag("--manifest");
    let interleave = args.iter().any(|a| a == "--interleave");
    let seed_flag = flag("--seed").map(|v| parse_u64(&v));
    let seed = seed_flag.unwrap_or(0x9E3779B97F4A7C15);
    let baseline_cap_flag = flag("--baseline-cap").map(|v| parse_u64(&v) as usize);
    let baseline_cap = baseline_cap_flag.unwrap_or(4);
    let baseline_rows: usize =
        flag("--baseline-rows").map_or(4, |v| v.parse().expect("--baseline-rows N"));
    let shape_args: Vec<(usize, usize)> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--shape")
        .map(|(i, _)| {
            let v = args.get(i + 1).expect("--shape M,K");
            let (m, k) = v.split_once(',').expect("--shape M,K");
            (
                m.parse().expect("--shape M,K"),
                k.parse().expect("--shape M,K"),
            )
        })
        .collect();
    let b_list_arg: Option<Vec<usize>> = flag("--b-list").map(|v| {
        v.split(',')
            .map(|b| b.parse().expect("--b-list 1,2,4,8"))
            .collect()
    });
    let new_flags = manifest_path.is_some()
        || interleave
        || seed_flag.is_some()
        || baseline_cap_flag.is_some()
        || !shape_args.is_empty()
        || b_list_arg.is_some();
    let b_list: Vec<usize> = b_list_arg.unwrap_or_else(|| (1..=max_b.min(8)).collect());
    if b_list.is_empty() || b_list.iter().any(|&b| b == 0 || b > 8 || b > max_b) {
        eprintln!("error: --b-list entries must be in 1..=min(8, --max-b)");
        std::process::exit(2);
    }
    if interleave && !(do_bench && baseline_object.is_some()) {
        eprintln!("error: --interleave requires --bench and --baseline-object PATH (the ref)");
        std::process::exit(2);
    }
    if do_bench && !interleave && (!shape_args.is_empty() || b_list.len() != max_b.min(8)) {
        eprintln!("error: --shape/--b-list with --bench require --interleave");
        std::process::exit(2);
    }
    if baseline_cap == 0 || baseline_cap > 8 {
        eprintln!("error: --baseline-cap must be in 1..=8");
        std::process::exit(2);
    }
    let shapes: Vec<(usize, usize)> = if shape_args.is_empty() {
        vec![(100276, 2560)]
    } else {
        shape_args.clone()
    };
    let run = || -> Result<(serde_json::Value, usize)> {
        let mut gpu = Gpu::init().map_err(err)?;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut cases = Vec::new();
        let mut pm_cases = Vec::new();
        let mut bad_total = 0usize;
        for (m, k) in [
            (1usize, 256usize),
            (2, 512),
            (3, 1024),
            (7, 256),
            (31, 768),
            (130, 2560),
            (1001, 2560),
            (4093, 2048),
            (33, 4352),
            (100003, 2560),
            (100276, 2560),
        ]
        .into_iter()
        .filter(|_| !no_oracle && sweep_dir.is_none() && manifest_path.is_none())
        {
            let a_bytes = weights(&mut rng, m, k);
            let a = gpu.upload_raw(&a_bytes, &[a_bytes.len()]).map_err(err)?;
            let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32() * 3.0).collect();
            let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
            let y_ref = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            let y = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            for column in 0..8 {
                gpu.gemv_mq2g256v2(
                    &a,
                    &x.sub_offset(column * k, k),
                    &y_ref.sub_offset(column * m, m),
                    m,
                    k,
                )
                .map_err(err)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let reference = gpu.download_raw_bytes(&y_ref).map_err(err)?;
            for b in 1..=8usize {
                gpu.hip.memset(&y.buf, 0x7f, 8 * m * 4).map_err(err)?;
                gpu.gemv_mq2g256v2_xbatch(&a, &x, &y, m, k, b)
                    .map_err(err)?;
                gpu.hip.device_synchronize().map_err(err)?;
                let got = gpu.download_raw_bytes(&y).map_err(err)?;
                let written = b * m * 4;
                let bad_written = got[..written]
                    .iter()
                    .zip(&reference[..written])
                    .filter(|(g, r)| g != r)
                    .count();
                let bad_guard = got[written..].iter().filter(|&&byte| byte != 0x7f).count();
                bad_total += bad_written + bad_guard;
                cases.push(json!({
                    "m": m, "k": k, "b": b,
                    "mismatch_bytes": bad_written, "overwritten_guard_bytes": bad_guard,
                }));
            }
            if gpu.mq2_xbatch_pm_covers(m, k) {
                for b in 1..=8usize {
                    let hip = launch(&mut gpu, &a, &x, &y, m, k, b, false)?;
                    let pm = launch(&mut gpu, &a, &x, &y, m, k, b, true)?;
                    let bad = diff(&pm, &hip);
                    bad_total += bad;
                    pm_cases.push(json!({"m": m, "k": k, "b": b, "pm_vs_hipcc_bytes": bad}));
                }
            }
            gpu.free_tensor(a).map_err(err)?;
            gpu.free_tensor(x).map_err(err)?;
            gpu.free_tensor(y_ref).map_err(err)?;
            gpu.free_tensor(y).map_err(err)?;
        }
        let pm_module = gpu.arch == "gfx1201";
        if pm_module
            && !no_oracle
            && sweep_dir.is_none()
            && manifest_path.is_none()
            && pm_cases.is_empty()
        {
            return Err("gfx1201: the PeaceMaker module covered no case".into());
        }
        if let Some(manifest) = captures.as_deref() {
            bad_total += capture_cases(&mut gpu, manifest, &mut pm_cases)?;
        }
        // Objects under test: --sweep-dir entries, else --object, else the shipped PM image.
        let mut objects: Vec<Obj> = Vec::new();
        let mut load_errors: Vec<serde_json::Value> = Vec::new();
        let mut manifest_sha = String::new();
        let mut manifest_skipped: Vec<serde_json::Value> = Vec::new();
        if let Some(mp) = manifest_path.as_deref() {
            // rip-search manifest.tsv: tab separated, header line; `id`, `axes` (JSON with ROWS),
            // `elf_sha256`, optional `symbol`. Objects: `<manifest dir>/objects/<id>.co`.
            let raw = std::fs::read(mp).map_err(|e| format!("{mp}: {e}"))?;
            manifest_sha = sha256_hex(&raw);
            let text = String::from_utf8(raw).map_err(|e| format!("{mp}: {e}"))?;
            let mut lines = text.lines().filter(|l| !l.trim().is_empty());
            let header: Vec<&str> = lines
                .next()
                .ok_or_else(|| format!("{mp}: empty manifest"))?
                .split('\t')
                .map(str::trim)
                .collect();
            let col = |n: &str| {
                header
                    .iter()
                    .position(|h| *h == n)
                    .ok_or_else(|| format!("{mp}: no `{n}` column"))
            };
            let (c_id, c_axes, c_sha) = (col("id")?, col("axes")?, col("elf_sha256")?);
            let c_sym = header.iter().position(|h| *h == "symbol");
            let dir = Path::new(mp).parent().unwrap_or(Path::new("."));
            let mut seen: std::collections::HashMap<String, String> = Default::default();
            for (n, line) in lines.enumerate() {
                let f: Vec<&str> = line.split('\t').collect();
                let get = |c: usize| f.get(c).copied().unwrap_or("").trim();
                let (id, sha) = (get(c_id), get(c_sha));
                if id.is_empty() || id.contains('/') || id.contains("..") {
                    return Err(format!("{mp}: row {}: bad id {id:?}", n + 1));
                }
                if sha.is_empty() {
                    manifest_skipped.push(json!({"id": id, "reason": "empty elf_sha256"}));
                    continue;
                }
                if let Some(first) = seen.get(sha) {
                    manifest_skipped.push(json!({"id": id, "reason": "dedup", "dedup_of": first}));
                    continue;
                }
                let axes: serde_json::Value = serde_json::from_str(get(c_axes))
                    .map_err(|e| format!("{mp}: {id}: axes: {e}"))?;
                let r = match &axes["ROWS"] {
                    serde_json::Value::Number(v) => v.as_u64().map(|v| v as usize),
                    serde_json::Value::String(s) => s.parse().ok(),
                    _ => None,
                }
                .ok_or_else(|| format!("{mp}: {id}: axes has no ROWS"))?;
                let path = dir.join("objects").join(format!("{id}.co"));
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let got = sha256_hex(&bytes);
                if got != sha {
                    return Err(format!(
                        "{}: sha256 {got} != manifest elf_sha256 {sha}; refusing to load",
                        path.display()
                    ));
                }
                seen.insert(sha.to_string(), id.to_string());
                let sym = c_sym.map(get).filter(|s| !s.is_empty()).unwrap_or(&symbol);
                match load_elf_obj(&gpu, id, &bytes, sym, r) {
                    Ok(mut o) => {
                        o.id = Some(id.to_string());
                        o.axes = axes;
                        objects.push(o);
                    }
                    Err(e) => {
                        eprintln!("load failed {id}: {e}");
                        bad_total += 1;
                        load_errors.push(json!({"object": id, "error": e}));
                    }
                }
            }
            if objects.is_empty() {
                return Err(format!("{mp}: no loadable object"));
            }
        } else if let Some(dir) = sweep_dir.as_deref() {
            let mut paths: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| format!("{dir}: {e}"))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "hxaco"))
                .collect();
            paths.sort();
            if paths.is_empty() {
                return Err(format!("{dir}: no *.hxaco"));
            }
            // Optional DIR/manifest.json: `[{"file","rows","symbol"?}]` or `{"objects":[..]}`;
            // entries override the stem token / --rows / --symbol for that file.
            let manifest: Vec<serde_json::Value> =
                match std::fs::read(Path::new(dir).join("manifest.json")) {
                    Ok(b) => {
                        let v: serde_json::Value = serde_json::from_slice(&b).map_err(err)?;
                        v.get("objects")
                            .unwrap_or(&v)
                            .as_array()
                            .cloned()
                            .ok_or("manifest.json: no object array")?
                    }
                    Err(_) => Vec::new(),
                };
            for p in paths {
                let name = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let ent = manifest.iter().find(|e| {
                    e["file"]
                        .as_str()
                        .is_some_and(|f| Path::new(f).file_name().is_some_and(|n| *n == *name))
                });
                let r = ent
                    .and_then(|e| e["rows"].as_u64())
                    .map(|v| v as usize)
                    .or_else(|| rows_from_stem(stem))
                    .unwrap_or(rows);
                let sym = ent.and_then(|e| e["symbol"].as_str()).unwrap_or(&symbol);
                let loaded = std::fs::read(&p)
                    .map_err(|e| format!("{}: {e}", p.display()))
                    .and_then(|bytes| load_obj(&gpu, &p.display().to_string(), &bytes, sym, r));
                match loaded {
                    Ok(o) => objects.push(o),
                    Err(e) => {
                        // a tuple that fails to load is recorded, never aborts the sweep
                        eprintln!("load failed {}: {e}", p.display());
                        bad_total += 1;
                        load_errors.push(json!({"object": p.display().to_string(), "error": e}));
                    }
                }
            }
        } else if let Some(path) = object.as_deref() {
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            objects.push(load_obj(&gpu, path, &bytes, &symbol, rows)?);
        } else {
            objects.push(load_obj(
                &gpu,
                "shipped:GEMV_MQ2_XBATCH_PM_GFX1201",
                rdna_compute::kernels::GEMV_MQ2_XBATCH_PM_GFX1201,
                &symbol,
                rows,
            )?);
        }
        let base = if do_bench || baseline_object.is_some() {
            let path = baseline_object
                .as_deref()
                .ok_or("--bench requires --baseline-object PATH (the pre-change cap4 object)")?;
            if !new_flags && baseline_rows != 4 {
                return Err("the cap4 baseline must run with 4 rows per block".into());
            }
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            Some(load_elf_obj(&gpu, path, &bytes, &symbol, baseline_rows)?)
        } else {
            None
        };
        let mut object_json = Vec::new();
        let mut bench_json = serde_json::Value::Null;
        for obj in &objects {
            let mut entry = json!({
                "object": obj.label, "sha256": obj.sha, "rows": obj.rows, "max_b": max_b,
            });
            if let Some(id) = &obj.id {
                entry["id"] = json!(id);
                entry["axes"] = obj.axes.clone();
                entry["load_wrap"] = json!(obj.wrap);
            }
            if !do_bench || !no_oracle {
                let shape_opt = (!shape_args.is_empty()).then_some(shape_args.as_slice());
                match direct_cases(&mut gpu, obj, base.as_ref(), shape_opt, &b_list) {
                    Ok((dc, bad)) => {
                        bad_total += bad;
                        entry["direct_mismatch_bytes"] = json!(bad);
                        entry["direct_cases"] = json!(dc);
                    }
                    Err(e) if sweep_dir.is_some() || manifest_path.is_some() => {
                        bad_total += 1;
                        entry["error"] = json!(e);
                        object_json.push(entry);
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            if let (true, Some(base)) = (do_bench && !interleave, base.as_ref()) {
                let (t, bad) = bench_table(&mut gpu, obj, base, bench_reps)?;
                bad_total += bad;
                if objects.len() == 1 {
                    bench_json = t.clone();
                }
                entry["bench"] = t;
            }
            object_json.push(entry);
        }
        let mut interleave_json = serde_json::Value::Null;
        if let (true, Some(base)) = (interleave, base.as_ref()) {
            let (t, bad) = interleave_bench(
                &mut gpu,
                &objects,
                base,
                baseline_cap,
                &shapes,
                &b_list,
                bench_reps,
                seed,
            )?;
            bad_total += bad;
            interleave_json = t;
        }
        let mut result = json!({
            "bench": bench_json,
            "schema": if new_flags { "mq2-xbatch-oracle-v3" } else { "mq2-xbatch-oracle-v2" },
            "arch": gpu.arch,
            "pm_image_sha256": rdna_compute::kernels::GEMV_MQ2_XBATCH_PM_GFX1201_SHA256,
            "objects": object_json, "load_errors": load_errors,
            "cases": cases, "pm_cases": pm_cases, "mismatch_bytes_total": bad_total,
        });
        if new_flags {
            let pci = gpu.hip.device_pci_bus_id(gpu.device_id).ok();
            let sysfs = |leaf: &str| {
                pci.as_ref().and_then(|p| {
                    std::fs::read_to_string(format!(
                        "/sys/bus/pci/devices/{}/{leaf}",
                        p.to_lowercase()
                    ))
                    .ok()
                    .map(|s| s.trim().to_string())
                })
            };
            result["inputs"] = json!({
                "manifest": manifest_path, "manifest_sha256": manifest_sha,
                "manifest_skipped": manifest_skipped,
                "objects": objects.iter().map(|o| json!({
                    "id": o.id, "object": o.label, "sha256": o.sha, "rows": o.rows,
                    "axes": o.axes, "load_wrap": o.wrap,
                })).collect::<Vec<_>>(),
                "baseline": base.as_ref().map(|b| json!({
                    "object": b.label, "sha256": b.sha, "rows": b.rows, "cap": baseline_cap,
                })),
                "shapes": shapes.iter().map(|&(m, k)| {
                    let (formula, copies) = copies_for(m, k);
                    json!({"m": m, "k": k, "weight_copies": copies, "weight_copies_formula": formula})
                }).collect::<Vec<_>>(),
                "b_list": b_list, "max_b": max_b, "reps": bench_reps, "seed": seed,
                "calls_per_replay": BENCH_CALLS, "interleave": interleave,
                "device": {
                    "arch": gpu.arch, "device_id": gpu.device_id, "pci_bus_id": pci,
                    "pci_vendor": sysfs("vendor"), "pci_device": sysfs("device"),
                    "pci_revision": sysfs("revision"), "product_name": sysfs("product_name"),
                },
            });
            result["interleave"] = interleave_json;
        }
        Ok((result, bad_total))
    };
    match run() {
        Ok((value, bad)) => {
            if let Some(out) = out {
                std::fs::write(&out, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
                eprintln!("wrote {out}");
            }
            eprintln!("mq2 xbatch oracle: {bad} mismatching bytes");
            std::process::exit(i32::from(bad != 0));
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
