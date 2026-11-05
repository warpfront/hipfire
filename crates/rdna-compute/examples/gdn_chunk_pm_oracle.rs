// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Hardware byte oracle of the chunked-GDN PM twins (`gated_delta_chunk_pm`, sealed by
//! `scripts/pm-hip-twin.sh`) against the shipped hipcc kernels on exact gfx1151 / gfx1100.
//!
//! For every (input kind, rows, entering state, variant, output dtype, Q8 mode) it launches the
//! hipcc incumbent twice (determinism control) and the twin once, on identical inputs and
//! identical kernargs, and compares the output buffer and the final state buffer BYTE FOR BYTE
//! (NaN payloads, signed zeros, denormals included). A negative control (one input LSB flipped)
//! proves the comparator can see a difference. Any mismatch or control failure exits 1.
//!
//! Variants: `F32` (all arches), `F32Arena` / `Q8` / `Q8Arena` (gfx1151 twins only; the arena arms
//! read `z` as BF16 from a pitched arena; the inline-Q8 arms run state mode 1 (product) and mode 2
//! (the slot plus an F32 capture of the final state, so the exact final state is compared too)).
//! Input kinds: `captured` (the product's value ranges: unit-norm BF16 q/k, negative log decay,
//! sigmoid beta; entering state zero and chained from a real 512-row incumbent run), `random`
//! (wide Gaussian everywhere, random entering state), `edge_finite` (signed zeros, denormals,
//! max-finite and tiny normals sprinkled into every operand and into the entering state, huge
//! +/- decay), `edge_nonfinite` (NaN with several payloads and +/-Inf sprinkled sparsely).
//!
//! `cargo run --release -p rdna-compute --features lab --example gdn_chunk_pm_oracle -- \
//!     [--rows 1,15,16,17,511,512,8191,8192] [--json OUT.json]`

use half::bf16;
use rayon::prelude::*;
use rdna_compute::gated_delta_chunk_pm::Variant;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const DIM: usize = 128;
const QKV_WIDTH: usize = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM; // 10240
const VALUE_WIDTH: usize = VALUE_HEADS * DIM; // 6144
const STATE_ELEMS: usize = VALUE_HEADS * DIM * DIM;
/// Q8 slot bytes: i8 codes for every head, then one F32 scale per (head, row).
const Q8_SLOT_BYTES: usize = VALUE_HEADS * DIM * DIM + VALUE_HEADS * DIM * 4;
/// Arena `z` pitch / first column (elements): deliberately not the dense value width.
const ARENA_PITCH: usize = VALUE_WIDTH + 96;
const ARENA_COL0: usize = 32;
const DEFAULT_ROWS: [usize; 8] = [1, 15, 16, 17, 511, 512, 8191, 8192];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / 16_777_216.0
    }
    fn gauss(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}
fn bf_f(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

const CHUNK: usize = 8192;

fn par_fill_u16(buf: &mut [u16], seed: u64, f: impl Fn(&mut Rng) -> u16 + Sync) {
    buf.par_chunks_mut(CHUNK).enumerate().for_each(|(i, c)| {
        let mut rng = Rng(seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        for v in c {
            *v = f(&mut rng);
        }
    });
}
fn par_fill_f32(buf: &mut [f32], seed: u64, f: impl Fn(&mut Rng) -> f32 + Sync) {
    buf.par_chunks_mut(CHUNK).enumerate().for_each(|(i, c)| {
        let mut rng = Rng(seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        for v in c {
            *v = f(&mut rng);
        }
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Captured,
    Random,
    EdgeFinite,
    EdgeNonFinite,
}
impl Kind {
    const ALL: [Kind; 4] = [
        Kind::Captured,
        Kind::Random,
        Kind::EdgeFinite,
        Kind::EdgeNonFinite,
    ];
    fn name(self) -> &'static str {
        match self {
            Kind::Captured => "captured",
            Kind::Random => "random",
            Kind::EdgeFinite => "edge_finite",
            Kind::EdgeNonFinite => "edge_nonfinite",
        }
    }
}

/// Host operands of one launch (BF16 as raw bits).
struct Inputs {
    rows: usize,
    projection: Vec<u16>,
    qn: Vec<u16>,
    kn: Vec<u16>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    z: Vec<f32>,
    z_arena: Vec<u16>,
    norm: Vec<u16>,
}

fn unit_heads_bf16(buf: &mut [u16]) {
    buf.par_chunks_mut(DIM).for_each(|h| {
        let n: f32 = h.iter().map(|&b| bf_f(b) * bf_f(b)).sum::<f32>().sqrt().max(1e-6);
        for b in h {
            *b = bf(bf_f(*b) / n);
        }
    });
}

fn gen_inputs(kind: Kind, rows: usize, seed: u64) -> Inputs {
    let mut projection = vec![0u16; rows * QKV_WIDTH];
    let mut qn = vec![0u16; rows * KEY_HEADS * DIM];
    let mut kn = vec![0u16; rows * KEY_HEADS * DIM];
    let mut gate = vec![0f32; rows * VALUE_HEADS];
    let mut beta = vec![0f32; rows * VALUE_HEADS];
    let mut z = vec![0f32; rows * VALUE_WIDTH];
    let mut z_arena = vec![0u16; rows * ARENA_PITCH];
    let mut norm = vec![0u16; DIM];
    let wide = kind == Kind::Random;
    let amp = if wide { 3.0 } else { 1.0 };
    par_fill_u16(&mut projection, seed ^ 1, |r| bf(amp * r.gauss()));
    par_fill_u16(&mut qn, seed ^ 2, |r| bf(amp * r.gauss()));
    par_fill_u16(&mut kn, seed ^ 3, |r| bf(amp * r.gauss()));
    if !wide {
        unit_heads_bf16(&mut qn);
        unit_heads_bf16(&mut kn);
    }
    if wide {
        par_fill_f32(&mut gate, seed ^ 4, |r| -0.5 * r.uniform() * r.uniform());
        par_fill_f32(&mut beta, seed ^ 5, |r| r.uniform());
    } else {
        // Qwen3.5 gate: -exp(A_log) * softplus(a + dt_bias): a negative log decay, mostly small.
        par_fill_f32(&mut gate, seed ^ 4, |r| {
            let sp = (1.0 + (r.gauss() * 1.2 - 0.5).exp()).ln();
            -(r.gauss() * 0.6 - 1.5).exp() * sp
        });
        par_fill_f32(&mut beta, seed ^ 5, |r| 1.0 / (1.0 + (-r.gauss()).exp()));
    }
    par_fill_f32(&mut z, seed ^ 6, |r| amp * r.gauss());
    par_fill_u16(&mut z_arena, seed ^ 7, |r| bf(amp * r.gauss()));
    for (i, n) in norm.iter_mut().enumerate() {
        *n = bf(1.0 + 0.1 * Rng(seed ^ 8 ^ i as u64).gauss());
    }
    let mut rng = Rng(seed ^ 0xED6E);
    match kind {
        Kind::EdgeFinite => {
            const B: [u16; 10] = [
                0x8000, 0x0001, 0x007f, 0x8001, 0x7f7f, 0xff7f, 0x0080, 0x8080, 0x0000, 0x3f80,
            ];
            const F: [f32; 10] = [
                -0.0, 1e-45, 1.1e-38, -1.1e-38, 3.0e38, -3.0e38, 1e-30, 0.0, 65504.0, -65520.0,
            ];
            let hit = |len: usize| (len / 500).max(8);
            for buf in [&mut projection, &mut qn, &mut kn, &mut z_arena] {
                for _ in 0..hit(buf.len()) {
                    let at = rng.below(buf.len());
                    buf[at] = B[rng.below(B.len())];
                }
            }
            for buf in [&mut z, &mut gate, &mut beta] {
                for _ in 0..hit(buf.len()) {
                    let at = rng.below(buf.len());
                    buf[at] = F[rng.below(F.len())];
                }
            }
            // Huge decays: exp(g) underflows to zero (g = -1e30) or overflows (g = +100).
            for _ in 0..(rows * VALUE_HEADS / 64).max(4) {
                let at = rng.below(gate.len());
                gate[at] = [-1.0e30, -200.0, 100.0, -0.0, 1e-30][rng.below(5)];
            }
        }
        Kind::EdgeNonFinite => {
            const B: [u16; 7] = [0x7fc0, 0xffc1, 0x7f81, 0x7f80, 0xff80, 0x7fff, 0xffff];
            const F: [f32; 6] = [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0xffff_ffff),
                f32::from_bits(0xff80_0001),
            ];
            for buf in [&mut projection, &mut qn, &mut kn, &mut z_arena] {
                for _ in 0..12 {
                    let at = rng.below(buf.len());
                    buf[at] = B[rng.below(B.len())];
                }
            }
            for buf in [&mut z, &mut gate, &mut beta] {
                for _ in 0..12 {
                    let at = rng.below(buf.len());
                    buf[at] = F[rng.below(F.len())];
                }
            }
        }
        _ => {}
    }
    Inputs {
        rows,
        projection,
        qn,
        kn,
        gate,
        beta,
        z,
        z_arena,
        norm,
    }
}

/// F32 entering state as Q8 slot bytes (`scale = peak / 127`, round to nearest; a zero row has scale 1).
fn q8_slot_of(state: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; Q8_SLOT_BYTES];
    let (codes, scales) = out.split_at_mut(VALUE_HEADS * DIM * DIM);
    for row in 0..VALUE_HEADS * DIM {
        // The kernel reads row `dv` of head `vh` as codes over the key axis; the F32 state is
        // `[head][key][value]`, so the row's codes are the strided column.
        let (vh, dv) = (row / DIM, row % DIM);
        let col = |k: usize| state[vh * DIM * DIM + k * DIM + dv];
        let peak = (0..DIM)
            .map(|k| col(k).abs())
            .filter(|v| v.is_finite())
            .fold(0f32, f32::max);
        let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
        scales[row * 4..row * 4 + 4].copy_from_slice(&scale.to_le_bytes());
        for k in 0..DIM {
            let v = col(k);
            let c = if v.is_finite() {
                (v / scale).round().clamp(-128.0, 127.0)
            } else {
                0.0
            };
            codes[row * DIM + k] = (c as i8) as u8;
        }
    }
    out
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Device {
    projection: GpuTensor,
    qn: GpuTensor,
    kn: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    z: GpuTensor,
    z_arena: GpuTensor,
    norm: GpuTensor,
}

fn upload(gpu: &mut Gpu, inp: &Inputs) -> Result<Device, String> {
    let e = |what: &str, e: rdna_compute::HipError| format!("{what}: {e}");
    let raw = |gpu: &Gpu, b: Vec<u8>, what: &str| -> Result<GpuTensor, String> {
        let n = b.len();
        gpu.upload_raw(&b, &[n]).map_err(|er| e(what, er))
    };
    Ok(Device {
        projection: raw(gpu, bytes_of_u16(&inp.projection), "projection")?,
        qn: raw(gpu, bytes_of_u16(&inp.qn), "qn")?,
        kn: raw(gpu, bytes_of_u16(&inp.kn), "kn")?,
        gate: raw(gpu, bytes_of_f32(&inp.gate), "gate")?,
        beta: raw(gpu, bytes_of_f32(&inp.beta), "beta")?,
        z: raw(gpu, bytes_of_f32(&inp.z), "z")?,
        z_arena: raw(gpu, bytes_of_u16(&inp.z_arena), "z_arena")?,
        norm: raw(gpu, bytes_of_u16(&inp.norm), "norm")?,
    })
}

fn free_device(gpu: &mut Gpu, d: Device) {
    for t in [
        d.projection,
        d.qn,
        d.kn,
        d.gate,
        d.beta,
        d.z,
        d.z_arena,
        d.norm,
    ] {
        let _ = gpu.free_tensor(t);
    }
}

fn read_back(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>, String> {
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let mut bytes = vec![0; t.buf.size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn arena(v: Variant) -> bool {
    matches!(v, Variant::F32Arena | Variant::Q8Arena)
}
fn is_q8(v: Variant) -> bool {
    matches!(v, Variant::Q8 | Variant::Q8Arena)
}

/// One kernel launch on fresh state / output buffers; returns (output bytes, state bytes).
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &mut Gpu,
    symbol: &str,
    variant: Variant,
    d: &Device,
    rows: usize,
    state_init: &[u8],
    state_bytes: usize,
    q8_mode: i32,
    out_bf16: bool,
    frame: u32,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut init = state_init.to_vec();
    init.resize(state_bytes, 0xFF);
    let state = gpu
        .upload_raw(&init, &[state_bytes])
        .map_err(|e| format!("state: {e}"))?;
    let out_bytes = rows * VALUE_WIDTH * 4;
    let out = gpu
        .upload_raw(&vec![0xFFu8; out_bytes], &[out_bytes])
        .map_err(|e| format!("output: {e}"))?;
    let mut a = hip_bridge::KernargBlob::new();
    a.push_ptr(d.projection.buf.as_ptr());
    a.push_ptr(d.qn.buf.as_ptr());
    a.push_ptr(d.kn.buf.as_ptr());
    a.push_ptr(d.gate.buf.as_ptr());
    a.push_ptr(d.beta.buf.as_ptr());
    a.push_ptr(state.buf.as_ptr());
    a.push_ptr(if arena(variant) {
        d.z_arena.buf.as_ptr()
    } else {
        d.z.buf.as_ptr()
    });
    a.push_ptr(d.norm.buf.as_ptr());
    a.push_ptr(out.buf.as_ptr());
    a.push_i32(rows as i32);
    a.push_i32(QKV_WIDTH as i32);
    a.push_i32(KEY_HEADS as i32);
    a.push_i32(VALUE_HEADS as i32);
    a.push_f32((DIM as f32).sqrt().recip());
    a.push_i32(i32::from(out_bf16));
    if is_q8(variant) {
        a.push_i32(q8_mode);
        a.push_u32(frame);
    }
    if arena(variant) {
        a.push_i32(ARENA_PITCH as i32);
        a.push_i32(ARENA_COL0 as i32);
    }
    a.pad_to(16);
    gpu.launch_blob_recorded(
        symbol,
        [VALUE_HEADS as u32, 1, 1],
        [256, 1, 1],
        0,
        a.as_mut_slice(),
        Default::default(),
    )
    .map_err(|e| format!("launch {symbol}: {e}"))?;
    let o = read_back(gpu, &out)?;
    let s = read_back(gpu, &state)?;
    let _ = gpu.free_tensor(out);
    let _ = gpu.free_tensor(state);
    Ok((o, s))
}

/// (differing bytes, differing 4-byte words, first differing byte).
fn diff(a: &[u8], b: &[u8]) -> (usize, usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let bytes = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let words = a
        .chunks(4)
        .zip(b.chunks(4))
        .filter(|(x, y)| x != y)
        .count();
    (bytes, words, a.iter().zip(b).position(|(x, y)| x != y))
}

#[derive(Default)]
struct Totals {
    comparisons: usize,
    out_bytes_compared: u64,
    state_bytes_compared: u64,
    out_byte_mismatches: usize,
    out_word_mismatches: usize,
    state_byte_mismatches: usize,
    state_word_mismatches: usize,
    control_nondeterministic: usize,
    negative_control_missed: usize,
    nonfinite_out_words: u64,
    failures: Vec<String>,
}

fn main() {
    let mut rows_list: Vec<usize> = DEFAULT_ROWS.to_vec();
    let mut json_out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rows" => {
                rows_list = args
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .map(|v| v.parse().expect("--rows a,b,c"))
                    .collect()
            }
            "--json" => json_out = args.next(),
            other => {
                eprintln!("usage: gdn_chunk_pm_oracle [--rows a,b,c] [--json OUT]  (unknown {other})");
                std::process::exit(2)
            }
        }
    }
    let mut gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("GPU init failed: {e}");
        std::process::exit(2)
    });
    let arch = gpu.arch.clone();
    if arch != "gfx1151" && arch != "gfx1100" {
        eprintln!("gdn_chunk_pm_oracle: needs gfx1151 or gfx1100, found {arch}");
        std::process::exit(2);
    }
    let variants: Vec<Variant> = Variant::ALL
        .into_iter()
        .filter(|v| rdna_compute::gated_delta_chunk_pm::twin_for(&arch, *v).is_some())
        .collect();
    // Receipts: module and image identity of every loaded symbol.
    let mut loaded: Vec<(Variant, &'static str, &'static str)> = Vec::new();
    for &v in &variants {
        let inc = gpu.lab_gdn_chunk_load(v, false).unwrap_or_else(|e| {
            eprintln!("load incumbent {v:?}: {e}");
            std::process::exit(2)
        });
        let twin = gpu.lab_gdn_chunk_load(v, true).unwrap_or_else(|e| {
            eprintln!("load twin {v:?}: {e}");
            std::process::exit(2)
        });
        for sym in [inc, twin] {
            let (module, container, device) = gpu.lab_kernel_device_identity(sym).unwrap();
            println!("receipt arch={arch} variant={v:?} symbol={sym} module={module} container_sha256={container} device_elf_sha256={device}");
        }
        loaded.push((v, inc, twin));
    }
    println!(
        "gdn_chunk_pm_oracle arch={arch} variants={variants:?} rows={rows_list:?} (hipcc incumbent x2 + twin x1 per comparison, exact bytes)"
    );
    let t0 = Instant::now();
    let mut totals = Totals::default();
    let mut lines: Vec<serde_json::Value> = Vec::new();
    let mut seed = 0x6D63_5F70_6D00u64;
    for kind in Kind::ALL {
        for &rows in &rows_list {
            seed = seed.wrapping_add(0x1000_0001);
            let inp = gen_inputs(kind, rows, seed);
            let dev = match upload(&mut gpu, &inp) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(2)
                }
            };
            // Entering states: captured gets zero and a state chained from a real 512-row
            // incumbent run; the others one random state (edge kinds with edge values in it).
            let mut states: Vec<(&'static str, Vec<f32>)> = Vec::new();
            match kind {
                Kind::Captured => {
                    states.push(("zero", vec![0f32; STATE_ELEMS]));
                    let prior = gen_inputs(Kind::Captured, 512, seed ^ 0xC4A1);
                    let pd = upload(&mut gpu, &prior).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let (_, s) = run(
                        &mut gpu,
                        loaded[0].1,
                        Variant::F32,
                        &pd,
                        512,
                        &vec![0u8; STATE_ELEMS * 4],
                        STATE_ELEMS * 4,
                        0,
                        false,
                        0,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    free_device(&mut gpu, pd);
                    states.push((
                        "chained",
                        s.chunks(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    ));
                }
                _ => {
                    let mut st = vec![0f32; STATE_ELEMS];
                    par_fill_f32(&mut st, seed ^ 0x57, |r| 0.3 * r.gauss());
                    let mut rng = Rng(seed ^ 0x58);
                    match kind {
                        Kind::EdgeFinite => {
                            for _ in 0..2000 {
                                let at = rng.below(st.len());
                                st[at] = [-0.0, 1e-45, -1.1e-38, 3.0e38, 1e-30, 0.0][rng.below(6)];
                            }
                        }
                        Kind::EdgeNonFinite => {
                            for _ in 0..4 {
                                let at = rng.below(st.len());
                                st[at] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.below(3)];
                            }
                        }
                        _ => {}
                    }
                    states.push(("random", st));
                }
            }
            for (state_name, st) in &states {
                let st_f32 = bytes_of_f32(st);
                let st_q8 = q8_slot_of(st);
                for &(variant, inc, twin) in &loaded {
                    let modes: &[i32] = if is_q8(variant) { &[1, 2] } else { &[0] };
                    for &mode in modes {
                        let (init, state_bytes) = if is_q8(variant) {
                            (
                                &st_q8,
                                if mode == 2 {
                                    Q8_SLOT_BYTES + STATE_ELEMS * 4
                                } else {
                                    Q8_SLOT_BYTES
                                },
                            )
                        } else {
                            (&st_f32, STATE_ELEMS * 4)
                        };
                        for out_bf16 in [false, true] {
                            let frame = 12_345 + rows as u32 - 1;
                            let go = |gpu: &mut Gpu, sym: &str| {
                                run(
                                    gpu, sym, variant, &dev, rows, init, state_bytes, mode, out_bf16,
                                    frame,
                                )
                            };
                            let res = (|| -> Result<(), String> {
                                let (o1, s1) = go(&mut gpu, inc)?;
                                let (o2, s2) = go(&mut gpu, inc)?;
                                let (ot, stt) = go(&mut gpu, twin)?;
                                let ctrl = diff(&o1, &o2).0 + diff(&s1, &s2).0;
                                let (ob, ow, ofirst) = diff(&o1, &ot);
                                let (sb, sw, sfirst) = diff(&s1, &stt);
                                totals.comparisons += 1;
                                totals.out_bytes_compared += o1.len() as u64;
                                totals.state_bytes_compared += s1.len() as u64;
                                totals.out_byte_mismatches += ob;
                                totals.out_word_mismatches += ow;
                                totals.state_byte_mismatches += sb;
                                totals.state_word_mismatches += sw;
                                totals.nonfinite_out_words += o1
                                    .chunks(4)
                                    .filter(|c| {
                                        !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite()
                                    })
                                    .count() as u64;
                                if ctrl != 0 {
                                    totals.control_nondeterministic += 1;
                                }
                                let tag = format!(
                                    "{} rows={rows} state={state_name} {variant:?} mode={mode} out_bf16={}",
                                    kind.name(),
                                    out_bf16 as u8
                                );
                                if ob + sb != 0 || ctrl != 0 {
                                    totals.failures.push(format!(
                                        "{tag}: out bytes {ob} (first {ofirst:?}) state bytes {sb} (first {sfirst:?}) control {ctrl}"
                                    ));
                                }
                                lines.push(serde_json::json!({
                                    "kind": kind.name(), "rows": rows, "state": state_name,
                                    "variant": format!("{variant:?}"), "q8_mode": mode, "out_bf16": out_bf16,
                                    "out_bytes": o1.len(), "state_bytes": s1.len(),
                                    "out_byte_mismatches": ob, "out_word_mismatches": ow,
                                    "state_byte_mismatches": sb, "state_word_mismatches": sw,
                                    "incumbent_repeat_byte_mismatches": ctrl,
                                }));
                                Ok(())
                            })();
                            if let Err(e) = res {
                                eprintln!("{e}");
                                std::process::exit(2);
                            }
                        }
                    }
                }
            }
            // Negative control: flip one LSB of a v element read by the kernel; the incumbent
            // outputs must then differ from the unflipped incumbent output (finite kinds only,
            // where the flipped element cannot be masked by a non-finite value).
            if matches!(kind, Kind::Captured | Kind::Random) {
                for &(variant, inc, _) in loaded.iter().filter(|(v, _, _)| *v == Variant::F32) {
                    let mut flipped = Inputs {
                        rows: inp.rows,
                        projection: inp.projection.clone(),
                        qn: inp.qn.clone(),
                        kn: inp.kn.clone(),
                        gate: inp.gate.clone(),
                        beta: inp.beta.clone(),
                        z: inp.z.clone(),
                        z_arena: inp.z_arena.clone(),
                        norm: inp.norm.clone(),
                    };
                    let at = 2 * KEY_HEADS * DIM + 5; // row 0, value head 0, column 5
                    flipped.projection[at] ^= 0x0001;
                    let fd = upload(&mut gpu, &flipped).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let zero = vec![0u8; STATE_ELEMS * 4];
                    let (o_ref, _) = run(&mut gpu, inc, variant, &dev, rows, &zero, STATE_ELEMS * 4, 0, false, 0).unwrap();
                    let (o_flip, _) = run(&mut gpu, inc, variant, &fd, rows, &zero, STATE_ELEMS * 4, 0, false, 0).unwrap();
                    if diff(&o_ref, &o_flip).0 == 0 {
                        totals.negative_control_missed += 1;
                        totals
                            .failures
                            .push(format!("negative control missed: {} rows={rows}", kind.name()));
                    }
                    free_device(&mut gpu, fd);
                }
            }
            free_device(&mut gpu, dev);
            println!(
                "done kind={} rows={rows}: {} comparisons so far, out byte mismatches {}, state byte mismatches {} ({:.0}s)",
                kind.name(),
                totals.comparisons,
                totals.out_byte_mismatches,
                totals.state_byte_mismatches,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let pass = totals.failures.is_empty();
    println!(
        "SUMMARY arch={arch} comparisons={} out_bytes_compared={} state_bytes_compared={} out_byte_mismatches={} out_word_mismatches={} state_byte_mismatches={} state_word_mismatches={} incumbent_repeat_nondeterministic={} negative_control_missed={} nonfinite_incumbent_out_words={}",
        totals.comparisons,
        totals.out_bytes_compared,
        totals.state_bytes_compared,
        totals.out_byte_mismatches,
        totals.out_word_mismatches,
        totals.state_byte_mismatches,
        totals.state_word_mismatches,
        totals.control_nondeterministic,
        totals.negative_control_missed,
        totals.nonfinite_out_words,
    );
    for f in totals.failures.iter().take(50) {
        println!("FAIL {f}");
    }
    if let Some(path) = json_out {
        let doc = serde_json::json!({
            "arch": arch, "rows": rows_list,
            "variants": variants.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>(),
            "comparisons": totals.comparisons,
            "out_byte_mismatches": totals.out_byte_mismatches,
            "out_word_mismatches": totals.out_word_mismatches,
            "state_byte_mismatches": totals.state_byte_mismatches,
            "state_word_mismatches": totals.state_word_mismatches,
            "incumbent_repeat_nondeterministic": totals.control_nondeterministic,
            "negative_control_missed": totals.negative_control_missed,
            "nonfinite_incumbent_out_words": totals.nonfinite_out_words,
            "failures": totals.failures,
            "cases": lines,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).expect("write json");
    }
    println!("{}", if pass { "RESULT PASS" } else { "RESULT FAIL" });
    std::process::exit(i32::from(!pass));
}
