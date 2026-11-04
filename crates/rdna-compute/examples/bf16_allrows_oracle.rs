// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle (+ bench): the one-launch all-rows BF16 GEMVs
//! (`HIPFIRE_QWEN4_BF16_ALLROWS`, module `qwen4_gemv_bf16_xf32_allrows`) for the
//! Flash-Next MoE call's router / selector / shared gate / up / shared down at
//! 9..=64 rows, on REAL weights (HFQM container, read-only) or `--synthetic`.
//!
//! Ground truth is the per-row singleton `gemv_bf16_xf32` (one launch per row and
//! part). Arms, each compared word for word against it (all live words plus the
//! guard words each side, outputs sentinel-poisoned):
//!   group           flag off, routed launcher (today's route; the incumbent for rows <= 8)
//!   allrows_routed  flag on, routed launcher (all-rows arm for 9..=64 rows)
//!   direct_<tile>   flag on, explicit tile entry, every `Bf16AllRowsTile::ALL`, rows 1..=64
//! for the x4 parts call and, per part, `gemm_bf16_xf32_multirow`. Rows 1 has no routed
//! launcher (both need >= 2 rows), so only the direct arms run there.
//!
//! Negative controls (each must be detected): row offset, poisoned pad row, wrong K.
//!
//! HC k4 section (`HIPFIRE_QWEN4_HC_K4_ALLROWS`, `gemv_bf16_xf32_k4_rows_allrows_m<TM>u<U>`): per layer
//! block (attn / mlp) the real HC down 320 x 10240 (with and without `hc_act`), the M = 4 write gate
//! (K 10240) and the up 10240 x 320, rows 1..=64, against the per-row `gemv_bf16_xf32_k4` singleton;
//! arms group (flags off), `lane_t8` (direct), routed, and every tile x token tile direct; the same
//! three controls on the HC down; `--bench` adds the HC before (`lane_t8`) / after table.
//!
//!   cargo run --release -p rdna-compute --features lab --example bf16_allrows_oracle -- \
//!     [--model PATH] [--layers 0,23,47] [--seeds N] [--synthetic] \
//!     [--bench] [--bench-reps N] [--no-oracle]
//! Exit nonzero on any mismatch, launch error, undetected control, or non-gfx1201 device.

use rdna_compute::gemv::{Bf16AllRowsTile, HcK4AllRowsTile};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.5/symtrunk/model/qwen3.8-flash-next-base.mq4-xts-symtrunk";
const GUARD: usize = 64;
const SENTINEL: u32 = 0x7fc0_beef;
const HIDDEN: usize = 2560;
const HC_K: usize = 10240;
const HC_RANK: usize = 320;
const SHARED_N: usize = 640;
const SHARD_N: usize = 384;
const MAXR: usize = 64;
const REP_ROWS: [usize; 17] = [1, 2, 7, 8, 9, 10, 15, 16, 17, 24, 30, 31, 32, 33, 48, 63, 64];
const BENCH_ROWS: [usize; 6] = [8, 16, 24, 30, 32, 48];
const CALLS_PER_SAMPLE: usize = 20;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---- minimal HFQM index reader (tensor name -> byte range) ---------------

struct Hfq {
    file: File,
    tensors: BTreeMap<String, (u64, u64, Vec<u32>, u8)>,
}

impl Hfq {
    fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut hdr = [0u8; 32];
        file.read_exact_at(&mut hdr, 0)
            .map_err(|e| format!("header {path}: {e}"))?;
        if &hdr[..4] != b"HFQM" {
            return Err(format!("{path}: not an HFQM container"));
        }
        let n = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let meta_off = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        let data_off = u64::from_le_bytes(hdr[24..32].try_into().unwrap());
        let mut meta = vec![0u8; (data_off - meta_off) as usize];
        file.read_exact_at(&mut meta, meta_off)
            .map_err(|e| format!("index {path}: {e}"))?;
        // Metadata JSON ends at its closing top-level brace; the tensor index follows.
        let (mut depth, mut in_str, mut esc, mut end) = (0i32, false, false, 0usize);
        for (i, &b) in meta.iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if b == b'\\' && in_str {
                esc = true;
                continue;
            }
            if b == b'"' {
                in_str = !in_str;
                continue;
            }
            if !in_str {
                if b == b'{' {
                    depth += 1;
                } else if b == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
            }
        }
        if end == 0 {
            return Err(format!("{path}: metadata JSON not terminated"));
        }
        let mut pos = end;
        let idx_n = u32::from_le_bytes(meta[pos..pos + 4].try_into().unwrap()) as usize;
        if idx_n != n {
            return Err(format!("{path}: tensor count {idx_n} != {n}"));
        }
        pos += 4;
        let mut off = data_off;
        let mut tensors = BTreeMap::new();
        for _ in 0..n {
            let nl = u16::from_le_bytes(meta[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&meta[pos..pos + nl]).to_string();
            pos += nl;
            let qt = meta[pos];
            let nd = meta[pos + 1] as usize;
            pos += 2;
            let shape: Vec<u32> = (0..nd)
                .map(|d| u32::from_le_bytes(meta[pos + 4 * d..pos + 4 * d + 4].try_into().unwrap()))
                .collect();
            pos += 4 * nd;
            pos += 4; // group size
            let size = u64::from_le_bytes(meta[pos..pos + 8].try_into().unwrap());
            pos += 8;
            tensors.insert(name, (off, size, shape, qt));
            off += size;
        }
        Ok(Hfq { file, tensors })
    }

    /// BF16 tensor `name` with the expected `[rows, cols]` shape, as raw u16 words.
    fn bf16(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        let (off, size, shape, qt) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} missing"))?;
        if *qt != 16 {
            return Err(format!("{name}: not BF16 (quant type {qt})"));
        }
        if shape.as_slice() != [rows as u32, cols as u32] || *size as usize != rows * cols * 2 {
            return Err(format!("{name}: shape {shape:?} / size {size} != [{rows}, {cols}]"));
        }
        let mut buf = vec![0u8; *size as usize];
        self.file
            .read_exact_at(&mut buf, *off)
            .map_err(|e| format!("{name}: read: {e}"))?;
        Ok(buf)
    }
}

/// One layer's MoE BF16 weights as raw bf16 bytes.
struct Layer {
    router: Vec<u8>,   // 512 x 2560
    selector: Vec<u8>, // 1 x 2560
    sgate: Vec<u8>,    // 640 x 2560
    sup: Vec<u8>,      // 640 x 2560
    sdown: Vec<u8>,    // 2560 x 640
}

fn real_layer(h: &Hfq, l: usize) -> Result<Layer, String> {
    let p = format!("model.language_model.layers.{l}.mlp");
    Ok(Layer {
        router: h.bf16(&format!("{p}.gate.weight"), 512, HIDDEN)?,
        selector: h.bf16(&format!("{p}.shared_expert_gate.weight"), 1, HIDDEN)?,
        sgate: h.bf16(&format!("{p}.shared_expert.gate_proj.weight"), SHARED_N, HIDDEN)?,
        sup: h.bf16(&format!("{p}.shared_expert.up_proj.weight"), SHARED_N, HIDDEN)?,
        sdown: h.bf16(&format!("{p}.shared_expert.down_proj.weight"), HIDDEN, SHARED_N)?,
    })
}

/// Random BF16 words (small magnitudes, occasional zeros) as little-endian bytes.
fn synth_bf16(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let v = if rng.below(50) == 0 {
            0.0
        } else {
            (rng.unit() * 2.0 - 1.0) * 0.1
        };
        b.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    b
}

fn synth_layer(rng: &mut Rng) -> Layer {
    Layer {
        router: synth_bf16(rng, 512 * HIDDEN),
        selector: synth_bf16(rng, HIDDEN),
        sgate: synth_bf16(rng, SHARED_N * HIDDEN),
        sup: synth_bf16(rng, SHARED_N * HIDDEN),
        sdown: synth_bf16(rng, HIDDEN * SHARED_N),
    }
}

fn load_layer(hfq: &mut Option<Hfq>, l: usize, rng: &mut Rng) -> Layer {
    if let Some(h) = hfq.as_ref() {
        match real_layer(h, l) {
            Ok(layer) => return layer,
            Err(e) => {
                println!("FALLBACK synthetic weights: layer {l}: {e}");
                *hfq = None;
            }
        }
    }
    synth_layer(rng)
}

/// `[2560, 384]` host rebuild from the first 384 columns of the `[2560, 640]` down.
fn down_shard(sdown: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(HIDDEN * SHARD_N * 2);
    for r in 0..HIDDEN {
        let s = r * SHARED_N * 2;
        v.extend_from_slice(&sdown[s..s + SHARD_N * 2]);
    }
    v
}

// ---- GPU helpers ----------------------------------------------------------

fn hip<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> T {
    r.unwrap_or_else(|e| panic!("{what}: {e:?}"))
}

fn upload_raw(gpu: &mut Gpu, bytes: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[bytes.len()], DType::Raw).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
    t
}

fn upload_bf16(gpu: &mut Gpu, bytes: &[u8], m: usize, k: usize) -> GpuTensor {
    assert_eq!(bytes.len(), m * k * 2, "bf16 weight bytes");
    let mut t = upload_raw(gpu, bytes);
    t.dtype = DType::BF16;
    t.shape = vec![m, k];
    t
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut t = upload_raw(gpu, &bytes);
    t.dtype = DType::F32;
    t.shape = vec![v.len()];
    t
}

/// `[GUARD][n][GUARD]` sentinel-filled F32 output; the tensor is the live window.
struct Out {
    whole: GpuTensor,
    live: GpuTensor,
    n: usize,
}

fn alloc_out(gpu: &mut Gpu, n: usize) -> Out {
    let whole = gpu
        .alloc_tensor(&[n + 2 * GUARD], DType::F32)
        .expect("alloc out");
    let live = whole.sub_offset(GUARD, n);
    let o = Out { whole, live, n };
    fill_sentinel(gpu, &o);
    o
}

fn fill_sentinel(gpu: &Gpu, o: &Out) {
    gpu.hip.device_synchronize().expect("sync");
    let bytes: Vec<u8> = SENTINEL.to_le_bytes().repeat(o.n + 2 * GUARD);
    gpu.hip.memcpy_htod(&o.whole.buf, &bytes).expect("sentinel");
}

fn fetch(gpu: &Gpu, o: &Out) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(&o.whole)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn free_outs(gpu: &mut Gpu, outs: Vec<Out>) {
    for o in outs {
        let _ = gpu.free_tensor(o.whole);
    }
}

fn mismatches(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// The words of a `[GUARD][n][GUARD]` output that a correct launch leaves: sentinels around
/// `live`.
fn expected(live: &[u32], pad_rows_after: usize) -> Vec<u32> {
    let mut v = vec![SENTINEL; GUARD];
    v.extend_from_slice(live);
    v.extend(std::iter::repeat_n(SENTINEL, pad_rows_after + GUARD));
    v
}

fn set_allrows(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).qwen4_bf16_allrows = on;
}

/// X `[rows * k]`: every row distinct; magnitudes and patterns per `family`.
fn gen_x(k: usize, family: usize, rows: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let base = match family {
            0 => 1.0f32,
            1 => 2f32.powi((r as i32 % 5) - 2),
            2 => 2f32.powi(((r as i32 * 7) % 21) - 10),
            _ => 0.25,
        };
        for _ in 0..k {
            let u = rng.unit() * 2.0 - 1.0;
            let v = match family {
                0 => u * base,
                // heavy tails: cubed values, occasional x64 spikes
                1 => {
                    let mut v = u * u * u * base;
                    if rng.below(40) == 0 {
                        v *= 64.0;
                    }
                    v
                }
                // zeros, signed zeros, subnormals, wide range
                2 => match rng.below(24) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(rng.next() as u32 & 0x807f_ffff),
                    _ => u * base,
                },
                // decode stream scale mix: bf16-rounded activations times 0.25
                _ => f32::from_bits((u * 3.0).to_bits() & 0xffff_0000) * base,
            };
            // an all-zero row and an all -0.0 row in every eight (family 0)
            x.push(match (family, r % 8) {
                (0, 5) => 0.0,
                (0, 6) => -0.0,
                _ => v,
            });
        }
    }
    x
}

#[derive(Default)]
struct Report {
    rows: BTreeMap<String, (usize, usize)>,
    failures: usize,
}

impl Report {
    fn record(&mut self, label: &str, detail: &str, bad: usize) {
        let e = self.rows.entry(label.to_string()).or_default();
        e.0 += 1;
        e.1 += bad;
        if bad != 0 {
            self.failures += 1;
            eprintln!("MISMATCH {label} {detail}: {bad} words");
        }
    }
}

type Parts<'a> = [(&'a GpuTensor, &'a GpuTensor, usize); 4];

/// Parts 0..n real, the rest m = 0 dummies (part 0's pointers).
fn parts4<'a>(ws: &[&'a GpuTensor], ys: &[&'a GpuTensor], ms: &[usize]) -> Parts<'a> {
    std::array::from_fn(|p| {
        if p < ws.len() {
            (ws[p], ys[p], ms[p])
        } else {
            (ws[0], ys[0], 0)
        }
    })
}

/// Per-row singleton reference of `nrows` rows: for each part `[nrows * m]` words.
fn reference(
    gpu: &mut Gpu,
    ws: &[GpuTensor],
    ms: &[usize],
    x: &GpuTensor,
    k: usize,
    nrows: usize,
) -> Vec<Vec<u32>> {
    set_allrows(gpu, true);
    ms.iter()
        .enumerate()
        .map(|(p, &m)| {
            let o = alloc_out(gpu, nrows * m);
            for r in 0..nrows {
                hip(
                    gpu.gemv_bf16_xf32(
                        &ws[p],
                        &x.sub_offset(r * k, k),
                        &o.live.sub_offset(r * m, m),
                        m,
                        k,
                    ),
                    "reference singleton",
                );
            }
            let v = fetch(gpu, &o)[GUARD..GUARD + nrows * m].to_vec();
            let _ = gpu.free_tensor(o.whole);
            v
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Arm {
    Group,
    Routed,
    Direct(Bf16AllRowsTile),
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Group => "group".into(),
            Arm::Routed => "allrows_routed".into(),
            Arm::Direct(t) => format!("direct_{}", t.entry().rsplit('_').next().unwrap()),
        }
    }
}

fn run_x4(gpu: &mut Gpu, arm: Arm, parts: Parts, x: &GpuTensor, k: usize, rows: usize) {
    set_allrows(gpu, !matches!(arm, Arm::Group));
    match arm {
        Arm::Group | Arm::Routed => hip(gpu.gemv_bf16_xf32_x4_rows(parts, x, k, rows), "x4 routed"),
        Arm::Direct(t) => hip(
            gpu.gemv_bf16_xf32_x4_rows_allrows_tile(t, parts, x, k, rows),
            "x4 direct",
        ),
    }
    set_allrows(gpu, true);
}

#[allow(clippy::too_many_arguments)]
fn run_mr(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) {
    set_allrows(gpu, matches!(arm, Arm::Routed));
    hip(gpu.gemm_bf16_xf32_multirow(w, x, y, m, k, rows), "multirow");
    set_allrows(gpu, true);
}

/// One call shape: `parts` = `(bf16 bytes [m, k], m)`; every part also runs alone through
/// `gemm_bf16_xf32_multirow`.
#[allow(clippy::too_many_arguments)]
fn check_case(
    gpu: &mut Gpu,
    rep: &mut Report,
    seeds: usize,
    tag: &str,
    label: &str,
    parts: &[(&[u8], usize)],
    k: usize,
    rng: &mut Rng,
) {
    let ms: Vec<usize> = parts.iter().map(|p| p.1).collect();
    let ws: Vec<GpuTensor> = parts
        .iter()
        .map(|(b, m)| upload_bf16(gpu, b, *m, k))
        .collect();
    let wrefs: Vec<&GpuTensor> = ws.iter().collect();
    for seed in 0..seeds {
        for family in 0..4 {
            let x = upload_f32(gpu, &gen_x(k, family, MAXR, rng));
            let refs = reference(gpu, &ws, &ms, &x, k, MAXR);
            let rows_list: Vec<usize> = if family == 0 {
                (1..=MAXR).collect()
            } else {
                REP_ROWS.to_vec()
            };
            for rows in rows_list {
                let detail = format!("{tag} rows={rows} seed={seed} family={family}");
                let mut arms: Vec<Arm> = if rows >= 2 {
                    vec![Arm::Group, Arm::Routed]
                } else {
                    vec![]
                };
                arms.extend(Bf16AllRowsTile::ALL.map(Arm::Direct));
                for &arm in &arms {
                    let outs: Vec<Out> = ms.iter().map(|&m| alloc_out(gpu, rows * m)).collect();
                    let ys: Vec<&GpuTensor> = outs.iter().map(|o| &o.live).collect();
                    run_x4(gpu, arm, parts4(&wrefs, &ys, &ms), &x, k, rows);
                    let bad: usize = outs
                        .iter()
                        .zip(&refs)
                        .zip(&ms)
                        .map(|((o, r), &m)| {
                            mismatches(&fetch(gpu, o), &expected(&r[..rows * m], 0))
                        })
                        .sum();
                    rep.record(&format!("{label} x4 {}", arm.name()), &detail, bad);
                    free_outs(gpu, outs);
                }
                if rows < 2 {
                    continue;
                }
                for (p, &m) in ms.iter().enumerate() {
                    for arm in [Arm::Group, Arm::Routed] {
                        let o = alloc_out(gpu, rows * m);
                        run_mr(gpu, arm, &ws[p], &x, &o.live, m, k, rows);
                        let bad = mismatches(&fetch(gpu, &o), &expected(&refs[p][..rows * m], 0));
                        rep.record(
                            &format!("{label} mr[m={m}] {}", arm.name()),
                            &detail,
                            bad,
                        );
                        free_outs(gpu, vec![o]);
                    }
                }
            }
            let _ = gpu.free_tensor(x);
        }
    }
    for w in ws {
        let _ = gpu.free_tensor(w);
    }
}

// ---- launch counting ------------------------------------------------------

/// Number of kernel launches `f` issues: a graph capture (nothing executes), counting the
/// kernarg blobs `launch_maybe_blob` records. Every kernel `f` needs must already be loaded.
fn launches(gpu: &mut Gpu, f: impl FnOnce(&mut Gpu)) -> usize {
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    let stream = gpu.active_stream.as_ref().expect("capture stream");
    hip(
        gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, stream),
        "begin capture",
    );
    f(gpu);
    let n = gpu.graph_blob_count();
    let stream = gpu.active_stream.as_ref().expect("capture stream");
    gpu.graphs
        .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
    n
}

/// Before any other launch: flag on loads only the all-rows entries for 9..=64 rows (no
/// `_x4_rows_r<R>`, no `multirow_pto2`), flag off loads the incumbent; then launch counts.
fn route_proof(gpu: &mut Gpu) {
    let (m, k) = (64usize, SHARD_N);
    let w = upload_bf16(gpu, &vec![0x3f; m * k * 2], m, k);
    let x = upload_f32(gpu, &vec![0.5f32; MAXR * k]);
    let y = alloc_out(gpu, MAXR * m);
    let parts = parts4(&[&w], &[&y.live], &[m]);
    let go = |gpu: &mut Gpu, on: bool, rows: usize| {
        set_allrows(gpu, on);
        hip(gpu.gemv_bf16_xf32_x4_rows(parts, &x, k, rows), "proof x4");
        hip(
            gpu.gemm_bf16_xf32_multirow(&w, &x, &y.live, m, k, rows),
            "proof multirow",
        );
    };
    for rows in [9, 30, MAXR] {
        go(gpu, true, rows);
    }
    let loaded = gpu.loaded_kernel_names();
    assert!(
        loaded
            .iter()
            .any(|n| n.starts_with("gemv_bf16_xf32_x4_rows_allrows_")),
        "flag on: no gemv_bf16_xf32_x4_rows_allrows_* entry loaded"
    );
    for n in &loaded {
        assert!(
            !n.starts_with("gemv_bf16_xf32_x4_rows_r") && !n.contains("gemm_bf16_xf32_multirow_pto2"),
            "flag on: incumbent {n} was loaded"
        );
    }
    go(gpu, false, 30);
    let loaded = gpu.loaded_kernel_names();
    for want in ["gemv_bf16_xf32_x4_rows_r8", "gemm_bf16_xf32_multirow_pto2"] {
        assert!(
            loaded.iter().any(|n| n == want),
            "flag off: incumbent {want} not loaded"
        );
    }
    let mut counts = [0usize; 4];
    for (i, on) in [false, true].into_iter().enumerate() {
        set_allrows(gpu, on);
        counts[2 * i] = launches(gpu, |g| {
            hip(g.gemv_bf16_xf32_x4_rows(parts, &x, k, 30), "count x4");
        });
        counts[2 * i + 1] = launches(gpu, |g| {
            hip(
                g.gemm_bf16_xf32_multirow(&w, &x, &y.live, m, k, 30),
                "count multirow",
            );
        });
    }
    set_allrows(gpu, true);
    assert_eq!(counts[0], 30usize.div_ceil(8), "flag off x4 launches at 30 rows");
    assert_eq!(counts[2], 1, "flag on x4 launches at 30 rows");
    assert_eq!(counts[3], 1, "flag on multirow launches at 30 rows");
    println!(
        "bf16_allrows_oracle: route proof ok (flag on -> *_allrows_* only, flag off -> incumbent); \
         launches at 30 rows: x4 off={} on={}, multirow off={} on={}",
        counts[0], counts[2], counts[1], counts[3]
    );
    for t in [x, w, y.whole] {
        let _ = gpu.free_tensor(t);
    }
}

// ---- negative controls ----------------------------------------------------

/// Run the three controls on the router weight; returns how many were detected.
fn controls(gpu: &mut Gpu, rep: &mut Report, router: &[u8], rng: &mut Rng) -> usize {
    let (m, k, rows) = (512usize, HIDDEN, 30usize);
    set_allrows(gpu, true);
    let w = upload_bf16(gpu, router, m, k);
    let mut hx = gen_x(k, 1, rows + 1, rng);
    let clean = upload_f32(gpu, &hx);
    hx[rows * k..].fill(f32::NAN);
    let poisoned = upload_f32(gpu, &hx);
    let want = reference(gpu, std::slice::from_ref(&w), &[m], &clean, k, rows).remove(0);
    let tile = Bf16AllRowsTile::pick(m, k, rows);
    let run = |gpu: &mut Gpu, x: &GpuTensor, y: &GpuTensor, kk: usize, r: usize| {
        hip(
            gpu.gemv_bf16_xf32_x4_rows_allrows_tile(tile, parts4(&[&w], &[y], &[m]), x, kk, r),
            "control launch",
        );
    };
    let mut detected = 0;
    let mut report = |name: &str, n: usize| {
        if n > 0 {
            detected += 1;
            println!("control {name}: detected ({n} words)");
        } else {
            println!("CONTROL NOT DETECTED: {name}");
        }
    };

    // (a) rows live rows over an x with a NaN pad row: bitwise the reference, pad output untouched.
    let o = alloc_out(gpu, (rows + 1) * m);
    run(gpu, &poisoned, &o.live, k, rows);
    let bad = mismatches(&fetch(gpu, &o), &expected(&want, m));
    rep.record("pad row: rows live, pad output untouched", "rows=30", bad);
    free_outs(gpu, vec![o]);
    // (b) the same launch over rows + 1: the NaN pad row must reach the last output row.
    let o = alloc_out(gpu, (rows + 1) * m);
    run(gpu, &poisoned, &o.live, k, rows + 1);
    let got = fetch(gpu, &o);
    report(
        "poisoned pad row",
        got[GUARD + rows * m..GUARD + (rows + 1) * m]
            .iter()
            .filter(|&&b| f32::from_bits(b).is_nan())
            .count(),
    );
    free_outs(gpu, vec![o]);
    // row offset: x shifted by one row.
    let o = alloc_out(gpu, rows * m);
    run(gpu, &clean.sub_offset(k, rows * k), &o.live, k, rows);
    report(
        "row offset",
        mismatches(&fetch(gpu, &o)[GUARD..GUARD + rows * m], &want),
    );
    free_outs(gpu, vec![o]);
    // wrong K: K - 8 against the full-K reference.
    let o = alloc_out(gpu, rows * m);
    run(gpu, &clean, &o.live, k - 8, rows);
    report(
        "wrong K",
        mismatches(&fetch(gpu, &o)[GUARD..GUARD + rows * m], &want),
    );
    free_outs(gpu, vec![o]);
    for t in [w, clean, poisoned] {
        let _ = gpu.free_tensor(t);
    }
    detected
}

// ---- HC k4 projection (HIPFIRE_QWEN4_HC_K4_ALLROWS) ----------------------------

/// One HC tensor of a layer block (`input_mix_weight_down` `[320, 10240]` or `..._up` `[10240, 320]`).
fn hc_tensor(h: &Option<Hfq>, l: usize, blk: &str, which: &str, m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, bool) {
    if let Some(h) = h {
        let name = format!("model.language_model.layers.{l}.{blk}_hyper_connection.input_mix_weight_{which}.weight");
        match h.bf16(&name, m, k) {
            Ok(b) => return (b, true),
            Err(e) => println!("FALLBACK synthetic HC weights: {e}"),
        }
    }
    (synth_bf16(rng, m * k), false)
}

fn set_hc(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).qwen4_hc_k4_allrows = on;
}

fn set_lane_hc(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).qwen4_lane_hc_allrows = on;
}

#[derive(Clone, Copy)]
enum HcArm {
    /// `HIPFIRE_QWEN4_HC_K4_ALLROWS` and `HIPFIRE_QWEN4_LANE_HC_ALLROWS` off: today's k4 route.
    Group,
    /// `gemv_bf16_xf32_k4_rows_lane_t8` (the H0 kernel) called directly.
    Lane,
    Routed,
    Direct(HcK4AllRowsTile, usize),
}

impl HcArm {
    fn name(self) -> String {
        match self {
            HcArm::Group => "group".into(),
            HcArm::Lane => "lane_t8".into(),
            HcArm::Routed => "allrows_routed".into(),
            HcArm::Direct(t, tpb) => {
                format!("direct_{}_tpb{tpb}", t.entry().trim_start_matches("gemv_bf16_xf32_k4_rows_allrows_"))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_hc(
    gpu: &mut Gpu,
    arm: HcArm,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
) {
    set_hc(gpu, !matches!(arm, HcArm::Group | HcArm::Lane));
    set_lane_hc(gpu, matches!(arm, HcArm::Lane));
    match arm {
        HcArm::Group | HcArm::Routed => hip(
            gpu.gemv_bf16_xf32_k4_rows(w, x, y, m, k, act, rows),
            "k4 routed",
        ),
        HcArm::Lane => hip(
            gpu.gemv_bf16_xf32_k4_rows_lane_allrows(w, x, y, m, k, act, rows),
            "k4 lane_t8",
        ),
        HcArm::Direct(t, tpb) => hip(
            gpu.gemv_bf16_xf32_k4_rows_allrows_tile(t, tpb, w, x, y, m, k, act, rows),
            "k4 direct",
        ),
    }
    set_hc(gpu, true);
    set_lane_hc(gpu, true);
}

/// Per-row `gemv_bf16_xf32_k4` singleton over `nrows` rows: `[nrows * m]` words.
fn hc_reference(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, m: usize, k: usize, act: Option<f32>, nrows: usize) -> Vec<u32> {
    let o = alloc_out(gpu, nrows * m);
    for r in 0..nrows {
        hip(
            gpu.gemv_bf16_xf32_k4(w, &x.sub_offset(r * k, k), &o.live.sub_offset(r * m, m), m, k, act),
            "k4 singleton",
        );
    }
    let v = fetch(gpu, &o)[GUARD..GUARD + nrows * m].to_vec();
    let _ = gpu.free_tensor(o.whole);
    v
}

/// One real HC matrix: every tile / token tile direct for rows 1..=64, the routed launcher, today's
/// route and the lane_t8 kernel, against the per-row k4 singleton.
#[allow(clippy::too_many_arguments)]
fn check_hc(
    gpu: &mut Gpu,
    rep: &mut Report,
    seeds: usize,
    label: &str,
    wbytes: &[u8],
    m: usize,
    k: usize,
    act: Option<f32>,
    rng: &mut Rng,
) {
    let w = upload_bf16(gpu, wbytes, m, k);
    for seed in 0..seeds {
        for family in 0..4 {
            let x = upload_f32(gpu, &gen_x(k, family, MAXR, rng));
            let want_full = hc_reference(gpu, &w, &x, m, k, act, MAXR);
            let rows_list: Vec<usize> = if family == 0 {
                (1..=MAXR).collect()
            } else {
                REP_ROWS.to_vec()
            };
            for rows in rows_list {
                let want = &want_full[..rows * m];
                let xv = x.sub_offset(0, rows * k);
                let mut arms = vec![HcArm::Routed];
                if rows > 8 {
                    arms.push(HcArm::Group);
                    arms.push(HcArm::Lane);
                }
                for t in HcK4AllRowsTile::ALL {
                    for tpb in HcK4AllRowsTile::TOKEN_TILES {
                        arms.push(HcArm::Direct(t, tpb));
                    }
                }
                for arm in arms {
                    if rows == 1 && matches!(arm, HcArm::Routed) {
                        continue; // rows == 1 is the singleton itself
                    }
                    let o = alloc_out(gpu, rows * m);
                    run_hc(gpu, arm, &w, &xv, &o.live, m, k, act, rows);
                    let bad = mismatches(&fetch(gpu, &o), &expected(want, 0));
                    rep.record(
                        &format!("HC {label} act={} {}", act.is_some(), arm.name()),
                        &format!("rows={rows} seed={seed} family={family}"),
                        bad,
                    );
                    free_outs(gpu, vec![o]);
                }
            }
            let _ = gpu.free_tensor(x);
        }
    }
    let _ = gpu.free_tensor(w);
}

/// HC negative controls on the real down: pad row, row offset, wrong K (K - 32).
fn hc_controls(gpu: &mut Gpu, rep: &mut Report, down: &[u8], rng: &mut Rng) -> usize {
    let (m, k, rows, act) = (HC_RANK, HC_K, 30usize, Some(0.25f32));
    let w = upload_bf16(gpu, down, m, k);
    let mut hx = gen_x(k, 1, rows + 1, rng);
    let clean = upload_f32(gpu, &hx);
    hx[rows * k..].fill(f32::NAN);
    let poisoned = upload_f32(gpu, &hx);
    let want = hc_reference(gpu, &w, &clean, m, k, act, rows);
    let (tile, tpb) = HcK4AllRowsTile::pick(m, rows);
    let run = |gpu: &mut Gpu, x: &GpuTensor, y: &GpuTensor, kk: usize, r: usize| {
        run_hc(gpu, HcArm::Direct(tile, tpb), &w, x, y, m, kk, act, r);
    };
    let mut detected = 0;
    let mut report = |name: &str, n: usize| {
        if n > 0 {
            detected += 1;
            println!("control HC {name}: detected ({n} words)");
        } else {
            println!("CONTROL NOT DETECTED: HC {name}");
        }
    };
    let o = alloc_out(gpu, (rows + 1) * m);
    run(gpu, &poisoned, &o.live, k, rows);
    let bad = mismatches(&fetch(gpu, &o), &expected(&want, m));
    rep.record("HC pad row: rows live, pad output untouched", "rows=30", bad);
    free_outs(gpu, vec![o]);
    let o = alloc_out(gpu, (rows + 1) * m);
    run(gpu, &poisoned, &o.live, k, rows + 1);
    let got = fetch(gpu, &o);
    report(
        "poisoned pad row",
        got[GUARD + rows * m..GUARD + (rows + 1) * m]
            .iter()
            .filter(|&&b| f32::from_bits(b).is_nan())
            .count(),
    );
    free_outs(gpu, vec![o]);
    let o = alloc_out(gpu, rows * m);
    run(gpu, &clean.sub_offset(k, rows * k), &o.live, k, rows);
    report("row offset", mismatches(&fetch(gpu, &o)[GUARD..GUARD + rows * m], &want));
    free_outs(gpu, vec![o]);
    let o = alloc_out(gpu, rows * m);
    run(gpu, &clean, &o.live, k - 32, rows);
    report("wrong K", mismatches(&fetch(gpu, &o)[GUARD..GUARD + rows * m], &want));
    free_outs(gpu, vec![o]);
    for t in [w, clean, poisoned] {
        let _ = gpu.free_tensor(t);
    }
    detected
}

/// HC route proof: with the flag on the routed 9..=64-row call loads an `_allrows_` k4 entry and
/// never `lane_t8` / `_tiled_r8`; flag off loads the incumbent; launch counts at 30 rows.
fn hc_route_proof(gpu: &mut Gpu) {
    let (m, k, rows) = (64usize, HcK4AllRowsTile::MIN_K, 30usize);
    let w = upload_bf16(gpu, &vec![0x3f; m * k * 2], m, k);
    let x = upload_f32(gpu, &vec![0.5f32; MAXR * k]);
    let y = alloc_out(gpu, MAXR * m);
    // Load every kernel first (a capture cannot compile).
    for arm in [HcArm::Group, HcArm::Lane, HcArm::Routed] {
        run_hc(gpu, arm, &w, &x, &y.live, m, k, Some(0.25), rows);
    }
    let loaded = gpu.loaded_kernel_names();
    assert!(
        loaded.iter().any(|n| n.starts_with("gemv_bf16_xf32_k4_rows_allrows_")),
        "flag on: no all-rows k4 entry loaded"
    );
    let n_on = launches(gpu, |g| run_hc(g, HcArm::Routed, &w, &x, &y.live, m, k, Some(0.25), rows));
    let n_group = launches(gpu, |g| run_hc(g, HcArm::Group, &w, &x, &y.live, m, k, Some(0.25), rows));
    let n_lane = launches(gpu, |g| run_hc(g, HcArm::Lane, &w, &x, &y.live, m, k, Some(0.25), rows));
    assert_eq!(n_on, 1, "routed HC k4 at 30 rows is one launch");
    println!(
        "bf16_allrows_oracle: HC route proof ok; launches at 30 rows: group (tiled r8 + tail) {n_group}, lane_t8 {n_lane}, all-rows {n_on}"
    );
    for t in [w, x, y.whole] {
        let _ = gpu.free_tensor(t);
    }
}

// ---- bench ----------------------------------------------------------------

/// Median over samples of microseconds per call: `CALLS_PER_SAMPLE` back-to-back calls are
/// captured once into a HIP graph and each sample is one synchronized graph replay, so the host
/// launch rate (about 5 us per launch) cannot floor the small kernels and the figure is device
/// time including the graph's inter-kernel gaps.
fn time_us(gpu: &mut Gpu, reps: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    hip(gpu.begin_stream_capture(), "begin capture");
    for _ in 0..CALLS_PER_SAMPLE {
        f(gpu);
    }
    let graph = hip(gpu.end_stream_capture(), "end capture");
    let exec = hip(gpu.hip.graph_instantiate(&graph), "instantiate");
    for _ in 0..3 {
        hip(gpu.launch_graph(&exec), "warm replay");
    }
    let mut s: Vec<f64> = (0..reps)
        .map(|_| {
            gpu.hip.device_synchronize().expect("sync");
            let t = Instant::now();
            hip(gpu.launch_graph(&exec), "replay");
            gpu.hip.device_synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e6 / CALLS_PER_SAMPLE as f64
        })
        .collect();
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(graph).expect("destroy graph");
    s.sort_by(f64::total_cmp);
    if reps % 2 == 1 {
        s[reps / 2]
    } else {
        (s[reps / 2 - 1] + s[reps / 2]) / 2.0
    }
}

const SWEEP_ROWS: [usize; 10] = [9, 12, 16, 20, 24, 30, 32, 40, 48, 64];
const BENCH_M: [usize; 5] = [512, 1, SHARD_N, SHARD_N, HIDDEN];
const BENCH_K: [usize; 5] = [HIDDEN, HIDDEN, HIDDEN, HIDDEN, SHARD_N];
const CASES: [&str; 9] = [
    "router 512x2560 mr",
    "selector 1x2560 mr",
    "gate 384x2560 mr",
    "up 384x2560 mr",
    "down 2560x384 mr",
    "V1 set (5 mr)",
    "V2 gate x4 512+1+384+384",
    "V2 down x4 2560xK384",
    "V2 set (2 x4)",
];

/// Router, selector, gate384, up384, down-shard weights with their activations / outputs.
struct Bench {
    w: Vec<GpuTensor>,
    xg: GpuTensor,
    xd: GpuTensor,
    o: Vec<Out>,
}

impl Bench {
    fn new(gpu: &mut Gpu, l: &Layer, rng: &mut Rng) -> Self {
        let shard = down_shard(&l.sdown);
        let bytes: [&[u8]; 5] = [
            &l.router,
            &l.selector,
            &l.sgate[..SHARD_N * HIDDEN * 2],
            &l.sup[..SHARD_N * HIDDEN * 2],
            &shard,
        ];
        let w = (0..5)
            .map(|i| upload_bf16(gpu, bytes[i], BENCH_M[i], BENCH_K[i]))
            .collect();
        let xg = upload_f32(gpu, &gen_x(HIDDEN, 0, MAXR, rng));
        let xd = upload_f32(gpu, &gen_x(SHARD_N, 0, MAXR, rng));
        let o = (0..5).map(|i| alloc_out(gpu, MAXR * BENCH_M[i])).collect();
        Bench { w, xg, xd, o }
    }

    fn x(&self, i: usize) -> &GpuTensor {
        if i == 4 { &self.xd } else { &self.xg }
    }

    fn mr(&self, gpu: &mut Gpu, i: usize, rows: usize) {
        hip(
            gpu.gemm_bf16_xf32_multirow(
                &self.w[i],
                self.x(i),
                &self.o[i].live,
                BENCH_M[i],
                BENCH_K[i],
                rows,
            ),
            "bench multirow",
        );
    }

    fn x4(&self, gpu: &mut Gpu, idx: &[usize], rows: usize) {
        let ws: Vec<&GpuTensor> = idx.iter().map(|&i| &self.w[i]).collect();
        let ys: Vec<&GpuTensor> = idx.iter().map(|&i| &self.o[i].live).collect();
        let ms: Vec<usize> = idx.iter().map(|&i| BENCH_M[i]).collect();
        hip(
            gpu.gemv_bf16_xf32_x4_rows(parts4(&ws, &ys, &ms), self.x(idx[0]), BENCH_K[idx[0]], rows),
            "bench x4",
        );
    }

    fn call(&self, gpu: &mut Gpu, case: usize, rows: usize) {
        match case {
            0..=4 => self.mr(gpu, case, rows),
            5 => (0..5).for_each(|i| self.mr(gpu, i, rows)),
            6 => self.x4(gpu, &[0, 1, 2, 3], rows),
            7 => self.x4(gpu, &[4], rows),
            _ => {
                self.x4(gpu, &[0, 1, 2, 3], rows);
                self.x4(gpu, &[4], rows);
            }
        }
    }

    /// One shape (`idx` weights as one x4 call, activation `x(idx[0])`) on the direct `tile` entry.
    fn tile_call(&self, gpu: &mut Gpu, tile: Bf16AllRowsTile, idx: &[usize], rows: usize) {
        let ws: Vec<&GpuTensor> = idx.iter().map(|&i| &self.w[i]).collect();
        let ys: Vec<&GpuTensor> = idx.iter().map(|&i| &self.o[i].live).collect();
        let ms: Vec<usize> = idx.iter().map(|&i| BENCH_M[i]).collect();
        hip(
            gpu.gemv_bf16_xf32_x4_rows_allrows_tile(
                tile,
                parts4(&ws, &ys, &ms),
                self.x(idx[0]),
                BENCH_K[idx[0]],
                rows,
            ),
            "bench tile",
        );
    }

    fn free(self, gpu: &mut Gpu) {
        for t in self.w {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.xg);
        let _ = gpu.free_tensor(self.xd);
        free_outs(gpu, self.o);
    }
}

fn bench(gpu: &mut Gpu, layer: &Layer, reps: usize, rng: &mut Rng) {
    let b = Bench::new(gpu, layer, rng);
    let mut t: HashMap<(usize, usize, bool), f64> = HashMap::new();
    println!(
        "bf16_allrows_oracle: bench, median of {reps} samples x {CALLS_PER_SAMPLE} back-to-back calls (one graph replay per sample, device time)"
    );
    println!("| case | rows | off us | on us | speedup | launches off->on |");
    println!("|---|---:|---:|---:|---:|---|");
    for (case, name) in CASES.iter().enumerate() {
        for rows in BENCH_ROWS {
            let mut us = [0.0f64; 2];
            let mut n = [0usize; 2];
            for (i, on) in [false, true].into_iter().enumerate() {
                set_allrows(gpu, on);
                us[i] = time_us(gpu, reps, |g| b.call(g, case, rows));
                n[i] = launches(gpu, |g| b.call(g, case, rows));
                t.insert((case, rows, on), us[i]);
            }
            println!(
                "| {name} | {rows} | {:.2} | {:.2} | {:.3}x | {}->{} |",
                us[0],
                us[1],
                us[0] / us[1],
                n[0],
                n[1]
            );
        }
    }
    set_allrows(gpu, true);
    println!("bf16_allrows_oracle: implied marginal ms per added row, 48 MoE layers");
    println!("  (8 -> 30 rows is the AR8 -> 30-row verify step; 16 -> 48 rows the wide span)");
    for (case, name) in [(5usize, "V1 set"), (8, "V2 set")] {
        let ms = |lo: usize, hi: usize, on: bool| {
            (t[&(case, hi, on)] - t[&(case, lo, on)]) / (hi - lo) as f64 * 48.0 / 1000.0
        };
        println!(
            "  {name}: 8->30 off {:.4} on {:.4} ms/row | 16->48 off {:.4} on {:.4} ms/row",
            ms(8, 30, false),
            ms(8, 30, true),
            ms(16, 48, false),
            ms(16, 48, true)
        );
    }
    println!("bf16_allrows_oracle: tile sweep (direct tile entries, us per call, best per row marked *)");
    let shapes: [(&str, &[usize]); 4] = [
        ("router 512x2560", &[0]),
        ("gate 384x2560", &[2]),
        ("down 2560x384", &[4]),
        ("gate-side x4 512+1+384+384", &[0, 1, 2, 3]),
    ];
    for (name, idx) in shapes {
        println!("{name}");
        print!("| rows |");
        for tile in Bf16AllRowsTile::ALL {
            print!(" {} |", tile.entry().trim_start_matches("gemv_bf16_xf32_x4_rows_allrows_"));
        }
        println!(" pick |");
        println!("|---:|{}---|", "---:|".repeat(Bf16AllRowsTile::ALL.len()));
        for rows in SWEEP_ROWS {
            let us: Vec<f64> = Bf16AllRowsTile::ALL
                .iter()
                .map(|&tile| time_us(gpu, reps, |g| b.tile_call(g, tile, idx, rows)))
                .collect();
            let best = us.iter().copied().fold(f64::INFINITY, f64::min);
            print!("| {rows} |");
            for v in &us {
                print!(" {:.2}{} |", v, if *v == best { "*" } else { "" });
            }
            let m_total: usize = idx.iter().map(|&i| BENCH_M[i]).sum();
            println!(
                " {} |",
                Bf16AllRowsTile::pick(m_total, BENCH_K[idx[0]], rows)
                    .entry()
                    .trim_start_matches("gemv_bf16_xf32_x4_rows_allrows_")
            );
        }
    }
    b.free(gpu);
    std::io::stdout().flush().expect("flush");
}

/// HC k4 bench: the down (320 x 10240, `hc_act`) and the M = 4 write gate (K 10240), as the V1
/// step runs them once each per HC block. "off" is what the 30-row step runs today (`lane_t8`
/// above 8 rows, the incumbent route at 8), "on" the routed all-rows launch.
fn hc_bench(gpu: &mut Gpu, down: &[u8], reps: usize, rng: &mut Rng) {
    let wd = upload_bf16(gpu, down, HC_RANK, HC_K);
    let wg = upload_bf16(gpu, &down[..4 * HC_K * 2], 4, HC_K);
    let x = upload_f32(gpu, &gen_x(HC_K, 0, MAXR, rng));
    let od = alloc_out(gpu, MAXR * HC_RANK);
    let og = alloc_out(gpu, MAXR * 4);
    let call = |g: &mut Gpu, case: usize, arm: HcArm, rows: usize| {
        let xv = x.sub_offset(0, rows * HC_K);
        if case != 1 {
            run_hc(g, arm, &wd, &xv, &od.live, HC_RANK, HC_K, Some(0.25), rows);
        }
        if case != 0 {
            run_hc(g, arm, &wg, &xv, &og.live, 4, HC_K, None, rows);
        }
    };
    let names = ["HC down 320x10240", "HC gate 4x10240", "HC set (down + gate)"];
    println!("bf16_allrows_oracle: HC k4 bench (graph replay, median of {reps} x {CALLS_PER_SAMPLE} calls)");
    println!("| case | rows | off us | on us | speedup | launches off->on |");
    println!("|---|---:|---:|---:|---:|---|");
    let mut t: HashMap<(usize, usize, bool), f64> = HashMap::new();
    for (case, name) in names.iter().enumerate() {
        for rows in BENCH_ROWS {
            let off = if rows > 8 { HcArm::Lane } else { HcArm::Group };
            let mut us = [0.0f64; 2];
            let mut n = [0usize; 2];
            for (i, arm) in [off, HcArm::Routed].into_iter().enumerate() {
                us[i] = time_us(gpu, reps, |g| call(g, case, arm, rows));
                n[i] = launches(gpu, |g| call(g, case, arm, rows));
                t.insert((case, rows, i == 1), us[i]);
            }
            println!(
                "| {name} | {rows} | {:.2} | {:.2} | {:.3}x | {}->{} |",
                us[0], us[1], us[0] / us[1], n[0], n[1]
            );
        }
    }
    println!("bf16_allrows_oracle: HC implied marginal ms per added row (96 HC blocks per step: 48 layers x attn/mlp)");
    let ms = |lo: usize, hi: usize, on: bool| {
        (t[&(2, hi, on)] - t[&(2, lo, on)]) / (hi - lo) as f64 * 96.0 / 1000.0
    };
    println!(
        "  HC set: 8->30 off {:.4} on {:.4} ms/row | 16->48 off {:.4} on {:.4} ms/row",
        ms(8, 30, false), ms(8, 30, true), ms(16, 48, false), ms(16, 48, true)
    );
    println!("bf16_allrows_oracle: HC tile sweep (direct entries, us per call, best per row marked *; columns tile/tpb)");
    let shapes: [(&str, usize); 2] = [("HC down 320x10240", 0), ("HC gate 4x10240", 1)];
    for (name, case) in shapes {
        println!("{name}");
        let cols: Vec<(HcK4AllRowsTile, usize)> = HcK4AllRowsTile::ALL
            .iter()
            .flat_map(|&t| HcK4AllRowsTile::TOKEN_TILES.iter().map(move |&p| (t, p)))
            .collect();
        print!("| rows | lane_t8 |");
        for (t, p) in &cols {
            print!(" {}/{p} |", t.entry().trim_start_matches("gemv_bf16_xf32_k4_rows_allrows_"));
        }
        println!(" pick |");
        println!("|---:|{}---|", "---:|".repeat(cols.len() + 1));
        for rows in SWEEP_ROWS {
            let mut us = vec![time_us(gpu, reps, |g| call(g, case, HcArm::Lane, rows))];
            for &(t, p) in &cols {
                us.push(time_us(gpu, reps, |g| call(g, case, HcArm::Direct(t, p), rows)));
            }
            let best = us.iter().copied().fold(f64::INFINITY, f64::min);
            print!("| {rows} |");
            for v in &us {
                print!(" {:.2}{} |", v, if *v == best { "*" } else { "" });
            }
            let (pt, pp) = HcK4AllRowsTile::pick(if case == 0 { HC_RANK } else { 4 }, rows);
            println!(" {}/{pp} |", pt.entry().trim_start_matches("gemv_bf16_xf32_k4_rows_allrows_"));
        }
    }
    for tns in [wd, wg, x, od.whole, og.whole] {
        let _ = gpu.free_tensor(tns);
    }
}

fn main() {
    let mut model = DEFAULT_MODEL.to_string();
    let mut layers: Vec<usize> = vec![0, 23, 47];
    let (mut seeds, mut synthetic, mut do_bench, mut oracle, mut reps) = (1usize, false, false, true, 7usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().expect("--model PATH"),
            "--layers" => {
                layers = args
                    .next()
                    .expect("--layers a,b,c")
                    .split(',')
                    .map(|s| s.parse().expect("layer"))
                    .collect()
            }
            "--seeds" => seeds = args.next().expect("--seeds N").parse().expect("N"),
            "--synthetic" => synthetic = true,
            "--bench" => do_bench = true,
            "--bench-reps" => reps = args.next().expect("--bench-reps N").parse().expect("N"),
            "--no-oracle" => oracle = false,
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(reps >= 5, "--bench-reps must be >= 5");
    assert!(!layers.is_empty(), "--layers needs at least one layer");
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "bf16_allrows_oracle: arch={} model={} layers={layers:?} seeds={seeds} synthetic={synthetic} \
         bench={do_bench} oracle={oracle}",
        gpu.arch,
        if synthetic { "-" } else { model.as_str() }
    );
    if gpu.arch != "gfx1201" {
        eprintln!("bf16_allrows_oracle: needs exact gfx1201");
        std::process::exit(2);
    }
    let mut hfq = if synthetic {
        None
    } else {
        match Hfq::open(&model) {
            Ok(h) => Some(h),
            Err(e) => {
                println!("FALLBACK synthetic weights: {e}");
                None
            }
        }
    };
    let mut rep = Report::default();
    let mut rng = Rng(0x0A11_B0F5_5EED_0001);
    if oracle {
        route_proof(&mut gpu);
        hc_route_proof(&mut gpu);
    }
    let mut bench_layer = None;
    for (i, &l) in layers.iter().enumerate() {
        if !oracle && i > 0 {
            break;
        }
        let lay = load_layer(&mut hfq, l, &mut rng);
        println!(
            "bf16_allrows_oracle: layer {l} weights: {}",
            if hfq.is_some() { "real" } else { "synthetic" }
        );
        if oracle {
            let tag = format!("L{l}");
            let gate_k = HIDDEN * 2;
            let (router, selector) = (lay.router.as_slice(), lay.selector.as_slice());
            let (sgate, sup, sdown) = (lay.sgate.as_slice(), lay.sup.as_slice(), lay.sdown.as_slice());
            let (sgate384, sup384) = (&sgate[..SHARD_N * gate_k], &sup[..SHARD_N * gate_k]);
            let shard = down_shard(sdown);
            let cases: [(&str, Vec<(&[u8], usize)>, usize); 6] = [
                (
                    "gate-side 512+1+640+640 K2560",
                    vec![(router, 512), (selector, 1), (sgate, 640), (sup, 640)],
                    HIDDEN,
                ),
                (
                    "gate-side EP2 512+1+384+384 K2560",
                    vec![(router, 512), (selector, 1), (sgate384, 384), (sup384, 384)],
                    HIDDEN,
                ),
                ("shared-down 2560 K640", vec![(sdown, HIDDEN)], SHARED_N),
                ("shared-down shard 2560 K384", vec![(shard.as_slice(), HIDDEN)], SHARD_N),
                (
                    "ragged 37+1+5+130 K2560",
                    vec![
                        (&router[..37 * gate_k], 37),
                        (selector, 1),
                        (&sgate[..5 * gate_k], 5),
                        (&sup[..130 * gate_k], 130),
                    ],
                    HIDDEN,
                ),
                ("router 512 K2560", vec![(router, 512)], HIDDEN),
            ];
            for (label, parts, k) in &cases {
                check_case(&mut gpu, &mut rep, seeds, &tag, label, parts, *k, &mut rng);
            }
            if i == 0 {
                // Scalar-tail (K % 8 != 0) and K % 256 != 0 vector shapes, synthetic weights.
                for (label, m, k) in [
                    ("scalar-tail m33 K2564", 33usize, 2564usize),
                    ("scalar-tail m7 K100", 7, 100),
                    ("vector m96 K1536", 96, 1536),
                ] {
                    let w = synth_bf16(&mut rng, m * k);
                    check_case(&mut gpu, &mut rep, seeds, "synthetic", label, &[(w.as_slice(), m)], k, &mut rng);
                }
                let detected = controls(&mut gpu, &mut rep, &lay.router, &mut rng);
                println!("bf16_allrows_oracle: controls detected: {detected}/3");
                if detected != 3 {
                    println!("bf16_allrows_oracle: FAIL (CONTROL NOT DETECTED)");
                    std::process::exit(1);
                }
            }
        }
        if oracle {
            for blk in ["attn", "mlp"] {
                let (down, real) = hc_tensor(&hfq, l, blk, "down", HC_RANK, HC_K, &mut rng);
                let (up, _) = hc_tensor(&hfq, l, blk, "up", HC_K, HC_RANK, &mut rng);
                println!("bf16_allrows_oracle: layer {l} {blk} HC weights: {}", if real { "real" } else { "synthetic" });
                let tag = format!("L{l} {blk}");
                // The decode read (hc_act, 0.25 stream scale) and the raw gate projection.
                check_hc(&mut gpu, &mut rep, seeds, &format!("{tag} down 320x10240"), &down, HC_RANK, HC_K, Some(0.25), &mut rng);
                check_hc(&mut gpu, &mut rep, seeds, &format!("{tag} down 320x10240"), &down, HC_RANK, HC_K, None, &mut rng);
                // The M = 4 write gate shape (first four down rows), K 10240.
                check_hc(&mut gpu, &mut rep, seeds, &format!("{tag} gate 4x10240"), &down[..4 * HC_K * 2], 4, HC_K, None, &mut rng);
                // The generic k4 shape: M 10240, K 320 (part 80).
                check_hc(&mut gpu, &mut rep, seeds, &format!("{tag} up 10240x320"), &up, HC_K, HC_RANK, Some(0.25), &mut rng);
                if i == 0 && blk == "attn" {
                    let detected = hc_controls(&mut gpu, &mut rep, &down, &mut rng);
                    println!("bf16_allrows_oracle: HC controls detected: {detected}/3");
                    if detected != 3 {
                        println!("bf16_allrows_oracle: FAIL (HC CONTROL NOT DETECTED)");
                        std::process::exit(1);
                    }
                }
            }
        }
        if i == 0 {
            bench_layer = Some(lay);
        }
    }
    if oracle {
        println!("bf16_allrows_oracle: per-case summary (runs, mismatching words)");
        for (label, (runs, bad)) in &rep.rows {
            println!("  {label:<70} runs={runs:<6} mismatch_words={bad}");
        }
        let total: usize = rep.rows.values().map(|(_, b)| b).sum();
        println!(
            "bf16_allrows_oracle: total mismatching words = {total}, failing cases = {}",
            rep.failures
        );
        if rep.failures != 0 || total != 0 {
            std::io::stdout().flush().expect("flush");
            std::process::exit(1);
        }
    }
    if do_bench {
        bench(&mut gpu, bench_layer.as_ref().expect("bench layer"), reps, &mut rng);
        let (down, _) = hc_tensor(&hfq, layers[0], "attn", "down", HC_RANK, HC_K, &mut rng);
        hc_bench(&mut gpu, &down, reps, &mut rng);
    }
    if oracle {
        println!("bf16_allrows_oracle: PASS");
    }
    std::io::stdout().flush().expect("flush");
}
