// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle: the output-tiled decode BF16 rows GEMVs
//! (`HIPFIRE_QWEN4_BF16_ROWS_OTILE`, module `qwen4_gemv_bf16_xf32_otile`)
//! against the incumbent `gemv_bf16_xf32_k4_rows_r<R>` / `_x4_rows_r<R>`
//! on REAL Flash-Next weights.
//!
//! Real tensors (read straight from the HFQM container by name): per layer
//! the attn/mlp HC down (320 x 10240, the k4 hc_act decode shape), HC up
//! (10240 x 320, generic k4/x4 shape), the MoE router (512 x 2560), shared
//! selector (1 x 2560), shared gate/up (640 x 2560, the four-part gate-side
//! x4 call) and shared down (2560 x 640). X is random (several families:
//! uniform, heavy-tailed, zeros and signed zeros, wide dynamic range, the
//! decode stream scale 0.25 for the HC epilogue). Rows 1..=8 and 32 (rows 1
//! and 32 are the pass-through to the incumbent route).
//!
//! Arms, each compared word for word (all live words plus guard words,
//! outputs sentinel-poisoned so an unwritten word is a mismatch):
//!   incumbent  flag off, routed launcher (the shipped kernels)
//!   routed     flag on, routed launcher (otile for 2..=8, pass-through else)
//!   direct     the explicit `*_otile` entry (rows 2..=8)
//! plus a graph-capture replay of the routed flag-on gate-side and HC down.
//!
//!   cargo run --release -p rdna-compute --example otile_bf16_rows_oracle -- \
//!     [--model PATH] [--layers 0,1,23,47] [--seeds N]
//! Exit nonzero on any mismatch, launch error, or non-gfx1201 device.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts";
const GUARD: usize = 64;
const SENTINEL: u32 = 0x7fc0_beef;
const HIDDEN: usize = 2560;
const HC_K: usize = 10240;
const HC_RANK: usize = 320;

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
    fn open(path: &str) -> Self {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut hdr = [0u8; 32];
        file.read_exact_at(&mut hdr, 0).expect("header");
        assert_eq!(&hdr[..4], b"HFQM", "not an HFQM container");
        let n = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let meta_off = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        let data_off = u64::from_le_bytes(hdr[24..32].try_into().unwrap());
        let mut meta = vec![0u8; (data_off - meta_off) as usize];
        file.read_exact_at(&mut meta, meta_off).expect("index");
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
        assert!(end > 0, "metadata JSON not terminated");
        let mut pos = end;
        let idx_n = u32::from_le_bytes(meta[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n, "tensor count");
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
        Hfq { file, tensors }
    }

    /// BF16 tensor `name` with the expected `[rows, cols]` shape, as raw u16 words.
    fn bf16(&self, name: &str, rows: usize, cols: usize) -> Vec<u8> {
        let (off, size, shape, qt) = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} missing"));
        assert_eq!(*qt, 16, "{name}: not BF16 (quant type {qt})");
        assert_eq!(
            shape.as_slice(),
            [rows as u32, cols as u32],
            "{name}: shape"
        );
        assert_eq!(*size as usize, rows * cols * 2, "{name}: size");
        let mut buf = vec![0u8; *size as usize];
        self.file
            .read_exact_at(&mut buf, *off)
            .expect("tensor read");
        buf
    }
}

// ---- GPU helpers ----------------------------------------------------------

fn upload_raw(gpu: &mut Gpu, bytes: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[bytes.len()], DType::Raw).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
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

fn mismatches(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn set_otile(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).qwen4_bf16_rows_otile = on;
}

/// X `[8 * k]`, row magnitudes and patterns per `family`.
fn gen_x(k: usize, family: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(8 * k);
    for r in 0..8 {
        let base = match family {
            0 => 1.0f32,
            1 => 2f32.powi((r as i32 % 5) - 2),
            2 => 2f32.powi(((r as i32 * 7) % 21) - 10),
            _ => 0.25,
        };
        for i in 0..k {
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
                // zeros, signed zeros, wide range
                2 => match rng.below(24) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(rng.next() as u32 & 0x807f_ffff), // subnormal
                    _ => u * base,
                },
                // decode stream scale mix: bf16-rounded activations times 0.25
                _ => {
                    let b = (u * 3.0).to_bits() & 0xffff_0000;
                    f32::from_bits(b) * base
                }
            };
            // keep r % 11 patterns of the existing oracles: an all-zero row, an all -0.0 row
            let v = if r == 5 && family == 0 {
                0.0
            } else if r == 6 && family == 0 {
                -0.0
            } else {
                v
            };
            let _ = i;
            x.push(v);
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

fn hip<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> T {
    r.unwrap_or_else(|e| panic!("{what}: {e:?}"))
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Incumbent,
    Routed,
    Direct,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Incumbent => "incumbent",
            Arm::Routed => "routed",
            Arm::Direct => "direct",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_k4(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
) {
    match arm {
        Arm::Incumbent => {
            set_otile(gpu, false);
            hip(
                gpu.gemv_bf16_xf32_k4_rows(w, x, y, m, k, act, rows),
                "k4 incumbent",
            );
        }
        Arm::Routed => {
            set_otile(gpu, true);
            hip(
                gpu.gemv_bf16_xf32_k4_rows(w, x, y, m, k, act, rows),
                "k4 routed",
            );
            set_otile(gpu, false);
        }
        Arm::Direct => hip(
            gpu.gemv_bf16_xf32_k4_rows_otile(w, x, y, m, k, act, rows),
            "k4 direct",
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_x4(
    gpu: &mut Gpu,
    arm: Arm,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
) {
    match arm {
        Arm::Incumbent => {
            set_otile(gpu, false);
            hip(
                gpu.gemv_bf16_xf32_x4_rows(parts, x, k, rows),
                "x4 incumbent",
            );
        }
        Arm::Routed => {
            set_otile(gpu, true);
            hip(gpu.gemv_bf16_xf32_x4_rows(parts, x, k, rows), "x4 routed");
            set_otile(gpu, false);
        }
        Arm::Direct => hip(
            gpu.gemv_bf16_xf32_x4_rows_otile(parts, x, k, rows),
            "x4 direct",
        ),
    }
}

/// One real k4 matrix: all rows, X families, incumbent vs routed/direct.
#[allow(clippy::too_many_arguments)]
fn check_k4(
    gpu: &mut Gpu,
    rep: &mut Report,
    label: &str,
    wbytes: &[u8],
    m: usize,
    k: usize,
    act: Option<f32>,
    seeds: usize,
    rng: &mut Rng,
) {
    let w = upload_raw(gpu, wbytes);
    for seed in 0..seeds {
        for family in 0..4 {
            let hx = gen_x(k, family, rng);
            let x = upload_f32(gpu, &hx);
            for rows in [1usize, 2, 3, 4, 5, 6, 7, 8, 32] {
                // rows 32 needs 32 X rows: tile the 8 generated rows
                let x = if rows > 8 {
                    let mut v = Vec::with_capacity(rows * k);
                    for r in 0..rows {
                        v.extend_from_slice(&hx[(r % 8) * k..(r % 8) * k + k]);
                    }
                    upload_f32(gpu, &v)
                } else {
                    x.sub_offset(0, rows * k)
                };
                let reference = alloc_out(gpu, rows * m);
                if rows == 1 {
                    set_otile(gpu, false);
                    hip(
                        gpu.gemv_bf16_xf32_k4(&w, &x, &reference.live, m, k, act),
                        "k4 single",
                    );
                } else {
                    run_k4(
                        gpu,
                        Arm::Incumbent,
                        &w,
                        &x,
                        &reference.live,
                        m,
                        k,
                        act,
                        rows,
                    );
                }
                let want = fetch(gpu, &reference);
                for arm in [Arm::Routed, Arm::Direct] {
                    if arm == Arm::Direct && !(2..=8).contains(&rows) {
                        continue;
                    }
                    let out = alloc_out(gpu, rows * m);
                    if rows == 1 {
                        set_otile(gpu, true);
                        hip(
                            gpu.gemv_bf16_xf32_k4(&w, &x, &out.live, m, k, act),
                            "k4 single (flag on)",
                        );
                        set_otile(gpu, false);
                    } else {
                        run_k4(gpu, arm, &w, &x, &out.live, m, k, act, rows);
                    }
                    let got = fetch(gpu, &out);
                    rep.record(
                        &format!("k4 {label} {} act={}", arm.name(), act.is_some()),
                        &format!("rows={rows} seed={seed} family={family}"),
                        mismatches(&got, &want),
                    );
                    let _ = gpu.free_tensor(out.whole);
                }
                let _ = gpu.free_tensor(reference.whole);
                if rows > 8 {
                    let _ = gpu.free_tensor(x);
                }
            }
            let _ = gpu.free_tensor(x);
        }
    }
    let _ = gpu.free_tensor(w);
}

/// A four-part x4 call over real tensors `(label, [(bytes, m); 4])`.
#[allow(clippy::too_many_arguments)]
fn check_x4(
    gpu: &mut Gpu,
    rep: &mut Report,
    label: &str,
    parts_bytes: &[(&[u8], usize)],
    k: usize,
    seeds: usize,
    rng: &mut Rng,
) {
    let ws: Vec<GpuTensor> = parts_bytes
        .iter()
        .map(|(b, _)| upload_raw(gpu, b))
        .collect();
    let ms: Vec<usize> = parts_bytes.iter().map(|(_, m)| *m).collect();
    for seed in 0..seeds {
        for family in 0..4 {
            let hx = gen_x(k, family, rng);
            for rows in [2usize, 3, 4, 5, 6, 7, 8, 32] {
                let mut xv = Vec::with_capacity(rows * k);
                for r in 0..rows {
                    xv.extend_from_slice(&hx[(r % 8) * k..(r % 8) * k + k]);
                }
                let x = upload_f32(gpu, &xv);
                // Unused matrices have m = 0 (reuse part 0's weights as the dummy pointer).
                let mk = |gpu: &mut Gpu| -> Vec<Out> {
                    (0..4)
                        .map(|p| alloc_out(gpu, rows * ms.get(p).copied().unwrap_or(0)))
                        .collect()
                };
                let reference = mk(gpu);
                {
                    let parts: [(&GpuTensor, &GpuTensor, usize); 4] = std::array::from_fn(|p| {
                        (
                            &ws[p.min(ws.len() - 1)],
                            &reference[p].live,
                            ms.get(p).copied().unwrap_or(0),
                        )
                    });
                    run_x4(gpu, Arm::Incumbent, parts, &x, k, rows);
                }
                let want: Vec<Vec<u32>> = reference.iter().map(|o| fetch(gpu, o)).collect();
                for arm in [Arm::Routed, Arm::Direct] {
                    if arm == Arm::Direct && !(2..=8).contains(&rows) {
                        continue;
                    }
                    let outs = mk(gpu);
                    {
                        let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                            std::array::from_fn(|p| {
                                (
                                    &ws[p.min(ws.len() - 1)],
                                    &outs[p].live,
                                    ms.get(p).copied().unwrap_or(0),
                                )
                            });
                        run_x4(gpu, arm, parts, &x, k, rows);
                    }
                    let bad: usize = outs
                        .iter()
                        .zip(&want)
                        .map(|(o, w)| mismatches(&fetch(gpu, o), w))
                        .sum();
                    rep.record(
                        &format!("x4 {label} {}", arm.name()),
                        &format!("rows={rows} seed={seed} family={family}"),
                        bad,
                    );
                    for o in outs {
                        let _ = gpu.free_tensor(o.whole);
                    }
                }
                for o in reference {
                    let _ = gpu.free_tensor(o.whole);
                }
                let _ = gpu.free_tensor(x);
            }
        }
    }
    for w in ws {
        let _ = gpu.free_tensor(w);
    }
}

/// Capture the flag-on routed launch into a HIP graph and replay it.
fn graph_check_k4(gpu: &mut Gpu, rep: &mut Report, wbytes: &[u8], rng: &mut Rng) {
    let (m, k, rows) = (320usize, HC_K, 8usize);
    let w = upload_raw(gpu, wbytes);
    let x = upload_f32(gpu, &gen_x(k, 1, rng));
    let reference = alloc_out(gpu, rows * m);
    run_k4(
        gpu,
        Arm::Incumbent,
        &w,
        &x,
        &reference.live,
        m,
        k,
        Some(0.25),
        rows,
    );
    let want = fetch(gpu, &reference);
    let out = alloc_out(gpu, rows * m);
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    run_k4(gpu, Arm::Routed, &w, &x, &out.live, m, k, Some(0.25), rows);
    let g = gpu.end_stream_capture().expect("end capture");
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    fill_sentinel(gpu, &out);
    gpu.launch_graph(&exec).expect("graph replay");
    let got = fetch(gpu, &out);
    rep.record(
        "k4 graph replay (routed, flag on)",
        "rows=8",
        mismatches(&got, &want),
    );
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
}

fn graph_check_x4(gpu: &mut Gpu, rep: &mut Report, parts_bytes: &[(&[u8], usize)], rng: &mut Rng) {
    let (k, rows) = (HIDDEN, 8usize);
    let ws: Vec<GpuTensor> = parts_bytes
        .iter()
        .map(|(b, _)| upload_raw(gpu, b))
        .collect();
    let ms: Vec<usize> = parts_bytes.iter().map(|(_, m)| *m).collect();
    let x = upload_f32(gpu, &gen_x(k, 1, rng));
    let reference: Vec<Out> = ms.iter().map(|&m| alloc_out(gpu, rows * m)).collect();
    let outs: Vec<Out> = ms.iter().map(|&m| alloc_out(gpu, rows * m)).collect();
    fn parts_of<'a>(
        ws: &'a [GpuTensor],
        o: &'a [Out],
        ms: &[usize],
    ) -> [(&'a GpuTensor, &'a GpuTensor, usize); 4] {
        std::array::from_fn(|p| (&ws[p], &o[p].live, ms[p]))
    }
    run_x4(
        gpu,
        Arm::Incumbent,
        parts_of(&ws, &reference, &ms),
        &x,
        k,
        rows,
    );
    let want: Vec<Vec<u32>> = reference.iter().map(|o| fetch(gpu, o)).collect();
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    run_x4(gpu, Arm::Routed, parts_of(&ws, &outs, &ms), &x, k, rows);
    let g = gpu.end_stream_capture().expect("end capture");
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    for o in &outs {
        fill_sentinel(gpu, o);
    }
    gpu.launch_graph(&exec).expect("graph replay");
    let bad: usize = outs
        .iter()
        .zip(&want)
        .map(|(o, w)| mismatches(&fetch(gpu, o), w))
        .sum();
    rep.record("x4 gate-side graph replay (routed, flag on)", "rows=8", bad);
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
}

/// Before any other launch: with the flag on the routed launchers must load the
/// `_otile_r<R>` entry of each row count 2..=8 and no incumbent `_r<R>` entry
/// (checked on the loaded-kernel set), and the flag off must load the incumbent.
fn route_proof(gpu: &mut Gpu) {
    let (m, k) = (320usize, HC_K);
    let w = upload_raw(gpu, &vec![0x3f; m * k * 2]);
    let ws_small = upload_raw(gpu, &vec![0x3f; 64 * HIDDEN * 2]);
    for rows in 2..=8usize {
        let x = upload_f32(gpu, &vec![0.5f32; rows * k]);
        let y = alloc_out(gpu, rows * m);
        set_otile(gpu, true);
        hip(
            gpu.gemv_bf16_xf32_k4_rows(&w, &x, &y.live, m, k, Some(0.25), rows),
            "proof k4",
        );
        let ya = alloc_out(gpu, rows * 64);
        let yb = alloc_out(gpu, 0);
        let parts: [(&GpuTensor, &GpuTensor, usize); 4] = [
            (&ws_small, &ya.live, 64),
            (&ws_small, &yb.live, 0),
            (&ws_small, &yb.live, 0),
            (&ws_small, &yb.live, 0),
        ];
        let x4 = upload_f32(gpu, &vec![0.5f32; rows * HIDDEN]);
        hip(
            gpu.gemv_bf16_xf32_x4_rows(parts, &x4, HIDDEN, rows),
            "proof x4",
        );
        set_otile(gpu, false);
        let loaded = gpu.loaded_kernel_names();
        for (otile, inc) in [
            (
                format!("gemv_bf16_xf32_k4_rows_otile_r{rows}"),
                format!("gemv_bf16_xf32_k4_rows_r{rows}"),
            ),
            (
                format!("gemv_bf16_xf32_x4_rows_otile_r{rows}"),
                format!("gemv_bf16_xf32_x4_rows_r{rows}"),
            ),
        ] {
            assert!(loaded.contains(&otile), "flag on: {otile} not loaded");
            assert!(
                !loaded.contains(&inc),
                "flag on: incumbent {inc} was loaded"
            );
        }
        for t in [x, y.whole, ya.whole, yb.whole, x4] {
            let _ = gpu.free_tensor(t);
        }
    }
    let _ = gpu.free_tensor(w);
    let _ = gpu.free_tensor(ws_small);
    // Flag off loads the incumbent entries.
    let x = upload_f32(gpu, &vec![0.5f32; 2 * HC_K]);
    let w = upload_raw(gpu, &vec![0x3f; 320 * HC_K * 2]);
    let y = alloc_out(gpu, 2 * 320);
    hip(
        gpu.gemv_bf16_xf32_k4_rows(&w, &x, &y.live, 320, HC_K, None, 2),
        "proof off",
    );
    assert!(
        gpu.loaded_kernel_names()
            .contains(&"gemv_bf16_xf32_k4_rows_r2".to_string()),
        "flag off: incumbent not loaded"
    );
    for t in [x, w, y.whole] {
        let _ = gpu.free_tensor(t);
    }
    println!("otile_bf16_rows_oracle: route proof ok (flag on -> *_otile_r2..r8 only, flag off -> incumbent)");
}

fn main() {
    let mut model = DEFAULT_MODEL.to_string();
    let mut layers: Vec<usize> = vec![0, 1, 23, 47];
    let mut seeds = 1usize;
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
            other => panic!("unknown argument {other}"),
        }
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "otile_bf16_rows_oracle: arch={} model={model} layers={layers:?} seeds={seeds}",
        gpu.arch
    );
    if gpu.arch != "gfx1201" {
        eprintln!("otile_bf16_rows_oracle: needs exact gfx1201");
        std::process::exit(2);
    }
    route_proof(&mut gpu);
    let hfq = Hfq::open(&model);
    let mut rep = Report::default();
    let mut rng = Rng(0x0711_1E5E_ED00_0001);
    for &l in &layers {
        let p = format!("model.language_model.layers.{l}");
        for blk in ["attn", "mlp"] {
            let down = hfq.bf16(
                &format!("{p}.{blk}_hyper_connection.input_mix_weight_down.weight"),
                HC_RANK,
                HC_K,
            );
            // The decode read: hc_act with the 0.25 stream scale; raw (None) is the gate projection.
            check_k4(
                &mut gpu,
                &mut rep,
                &format!("L{l} {blk} hc_down 320x10240"),
                &down,
                HC_RANK,
                HC_K,
                Some(0.25),
                seeds,
                &mut rng,
            );
            check_k4(
                &mut gpu,
                &mut rep,
                &format!("L{l} {blk} hc_down 320x10240"),
                &down,
                HC_RANK,
                HC_K,
                None,
                seeds,
                &mut rng,
            );
            let up = hfq.bf16(
                &format!("{p}.{blk}_hyper_connection.input_mix_weight_up.weight"),
                HC_K,
                HC_RANK,
            );
            // Generic shapes: M = 10240 (a multiple of every tile), K = 320.
            check_k4(
                &mut gpu,
                &mut rep,
                &format!("L{l} {blk} hc_up 10240x320"),
                &up,
                HC_K,
                HC_RANK,
                Some(0.25),
                seeds,
                &mut rng,
            );
            check_x4(
                &mut gpu,
                &mut rep,
                &format!("L{l} {blk} hc_up 10240x320"),
                &[(&up, HC_K)],
                HC_RANK,
                seeds,
                &mut rng,
            );
        }
        let router = hfq.bf16(&format!("{p}.mlp.gate.weight"), 512, HIDDEN);
        let selector = hfq.bf16(&format!("{p}.mlp.shared_expert_gate.weight"), 1, HIDDEN);
        let sgate = hfq.bf16(
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            640,
            HIDDEN,
        );
        let sup = hfq.bf16(
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            640,
            HIDDEN,
        );
        let sdown = hfq.bf16(
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            HIDDEN,
            640,
        );
        let gate_side: [(&[u8], usize); 4] =
            [(&router, 512), (&selector, 1), (&sgate, 640), (&sup, 640)];
        check_x4(
            &mut gpu,
            &mut rep,
            &format!("L{l} gate-side 512+1+640+640 K2560"),
            &gate_side,
            HIDDEN,
            seeds,
            &mut rng,
        );
        check_x4(
            &mut gpu,
            &mut rep,
            &format!("L{l} shared-down 2560 K640"),
            &[(&sdown, HIDDEN)],
            640,
            seeds,
            &mut rng,
        );
        // Ragged part sizes and a short K (scalar-free, K % 8 == 0 but not % 256).
        let ragged: [(&[u8], usize); 4] = [
            (&router[..(2560 * 2) * 37], 37),
            (&selector, 1),
            (&sgate[..(2560 * 2) * 5], 5),
            (&sup[..(2560 * 2) * 130], 130),
        ];
        check_x4(
            &mut gpu,
            &mut rep,
            &format!("L{l} ragged 37+1+5+130 K2560"),
            &ragged,
            HIDDEN,
            seeds,
            &mut rng,
        );
        if l == layers[0] {
            let down = hfq.bf16(
                &format!("{p}.mlp_hyper_connection.input_mix_weight_down.weight"),
                HC_RANK,
                HC_K,
            );
            graph_check_k4(&mut gpu, &mut rep, &down, &mut rng);
            graph_check_x4(&mut gpu, &mut rep, &gate_side, &mut rng);
        }
    }
    println!("otile_bf16_rows_oracle: per-case summary (runs, mismatching words)");
    for (label, (runs, bad)) in &rep.rows {
        println!("  {label:<64} runs={runs:<5} mismatch_words={bad}");
    }
    let total: usize = rep.rows.values().map(|(_, b)| b).sum();
    println!(
        "otile_bf16_rows_oracle: total mismatching words = {total}, failing cases = {}",
        rep.failures
    );
    if rep.failures != 0 || total != 0 {
        std::process::exit(1);
    }
    println!("otile_bf16_rows_oracle: PASS");
}
