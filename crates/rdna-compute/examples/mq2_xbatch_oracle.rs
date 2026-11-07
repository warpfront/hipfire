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

/// Median us per call of `calls` launches captured in one graph (`reps` synchronized replays).
fn graph_us(
    gpu: &mut Gpu,
    reps: usize,
    mut f: impl FnMut(&mut Gpu, usize) -> Result<()>,
) -> Result<f64> {
    for i in 0..BENCH_COPIES {
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
        _module: module,
        func,
    })
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
    let mut start = 0;
    while start < b {
        let rows = 4.min(b - start);
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
    max_b: usize,
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
        for b in 1..=max_b.min(8) {
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
        .filter(|_| !no_oracle && sweep_dir.is_none())
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
        if pm_module && !no_oracle && sweep_dir.is_none() && pm_cases.is_empty() {
            return Err("gfx1201: the PeaceMaker module covered no case".into());
        }
        if let Some(manifest) = captures.as_deref() {
            bad_total += capture_cases(&mut gpu, manifest, &mut pm_cases)?;
        }
        // Objects under test: --sweep-dir entries, else --object, else the shipped PM image.
        let mut objects: Vec<Obj> = Vec::new();
        let mut load_errors: Vec<serde_json::Value> = Vec::new();
        if let Some(dir) = sweep_dir.as_deref() {
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
            if flag("--baseline-rows").is_some_and(|v| v != "4") {
                return Err("the cap4 baseline must run with 4 rows per block".into());
            }
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            Some(load_obj(&gpu, path, &bytes, &symbol, 4)?)
        } else {
            None
        };
        let mut object_json = Vec::new();
        let mut bench_json = serde_json::Value::Null;
        for obj in &objects {
            let mut entry = json!({
                "object": obj.label, "sha256": obj.sha, "rows": obj.rows, "max_b": max_b,
            });
            if !do_bench || !no_oracle {
                match direct_cases(&mut gpu, obj, base.as_ref(), max_b) {
                    Ok((dc, bad)) => {
                        bad_total += bad;
                        entry["direct_mismatch_bytes"] = json!(bad);
                        entry["direct_cases"] = json!(dc);
                    }
                    Err(e) if sweep_dir.is_some() => {
                        bad_total += 1;
                        entry["error"] = json!(e);
                        object_json.push(entry);
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            if let (true, Some(base)) = (do_bench, base.as_ref()) {
                let (t, bad) = bench_table(&mut gpu, obj, base, bench_reps)?;
                bad_total += bad;
                if objects.len() == 1 {
                    bench_json = t.clone();
                }
                entry["bench"] = t;
            }
            object_json.push(entry);
        }
        Ok((
            json!({
                "bench": bench_json,
                "schema": "mq2-xbatch-oracle-v2", "arch": gpu.arch,
                "pm_image_sha256": rdna_compute::kernels::GEMV_MQ2_XBATCH_PM_GFX1201_SHA256,
                "objects": object_json, "load_errors": load_errors,
                "cases": cases, "pm_cases": pm_cases, "mismatch_bytes_total": bad_total,
            }),
            bad_total,
        ))
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
