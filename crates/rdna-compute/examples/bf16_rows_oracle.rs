// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G1 BF16 fixed-eight-row PM tile oracle, logical R=1..64, exact gfx1151/1201.
//!
//! References remain the actual architecture's `gemv_bf16_xf32`, singleton
//! `gemv_bf16_xf32_x4`, and singleton HC `gemv_bf16_xf32_k4` objects, never PM C1.
//! The routed arm requires the BF16 selector and eager verify scope; R>8 must
//! actually launch the new tile, not a loop or an obsolete r16/r32 object.
//! Direct PM launches include R=1, use the frozen 96B/48B kernargs, and use
//! grid [sum Mi,ceil(R/8),1], block32 (ordinary) or [M,ceil(R/8),1],
//! block128 (HC). Every observed launch requires its loaded module/object SHA.
//! Direct blobs use the recorder-aware launch funnel so each receipt observes
//! the submitted tile, not the preceding singleton reference launch.
//!
//! Production and ragged M/K shapes, scalar K%8 tails, raw/activated HC with
//! realistic and extreme scales, stable-address A/B rewrites and graph replay
//! are separate fatal byte gates. Inactive X rows and full-capacity Y tails
//! are poisoned; inputs/weights are checked immutable. All live words and
//! guards are compared, with first-mismatch diagnostics. HC quarters_out/in
//! handoff retains both alternating guarded scratch slots and stale-slot,
//! perturbed-input and swapped-weight negative controls.
//!
//! Synthetic cases are labelled synthetic, not captured model evidence.
//! `--captured-hc MANIFEST.json` additionally gates real captured HC operands.
//! Canonical `qwen4-rowtile-operands-v1` BF16 `hc_k4` manifests bind the
//! pinned source index, model/prompt, four W bindings and 64 sequential AR
//! recipes to the actual loaded singleton module/symbol/raw and container SHA.
//! Only the captured mode qualifies: `scale:null,hc_act:false` means raw
//! `None`; finite scale with `hc_act:true` means `Some(scale)`.
//! Legacy manifests retain `arch`, `source`, `m`, `k`, optional/null finite
//! `scale`, optional consistent `hc_act`, `weights`, `weights_sha256`, `x`,
//! `x_sha256`; paths are manifest-relative BF16 [M,K] and F32 [64,K].
//! Their extra raw exercise is labelled legacy-extra-HC, not captured mode.
//! `--captured-ordinary MANIFEST.json` (repeatable) gates canonical
//! `qwen4-rowtile-operands-v1` BF16 `single`/`x4` captures and their pinned
//! `qwen4-rowtile-source-v1` index, model/prompt, four immutable W bindings
//! and 64 eager AR recipes. The actual loaded single or x4 C1 object's
//! module/symbol/raw object SHA must match the source; reference-mode policies
//! are unchanged. Each capture runs routed/direct for every R=1..64 with
//! the existing full-word, inactive/tail/guard and immutable-input checks.
//! Without a manifest the final receipt explicitly withholds real-capture
//! qualification; it never claims a full PLAN G1 certificate.
//!
//! HIPFIRE_QWEN4_WIDE_TWIN_BF16=1 cargo run --release -p rdna-compute \
//!   --features lab --example bf16_rows_oracle -- \
//!   [--captured-ordinary MANIFEST.json] [--captured-hc MANIFEST.json]
//! Refused architecture/device, absent identity, launch errors and mismatches
//! exit nonzero. gfx1100 is refused until actual singleton/tile hooks exist.

use hip_bridge::KernargBlob;
use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::tensor_ops::{hyper_write_norm, HyperNextGates, HyperWrite};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const GUARD: usize = 64;
/// A quiet NaN with a payload no arithmetic produces.
const SENTINEL: u32 = 0x7fc0_beef;
const MAX_ROWS: usize = 64;
/// Include a whole extra tile after the logical capacity to catch R=64 overruns.
const OUTPUT_ROWS: usize = MAX_ROWS + 8;
const HC_K: usize = 10240;
const HC_PART: usize = 2560;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Round-to-nearest-even BF16 bits of a finite F32.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Random BF16 weights `[m * k]`. `exp_span > 0` mixes in +-0, subnormals,
/// exponents 2^-span..2^span and exact cancelling neighbours.
fn gen_weights(m: usize, k: usize, amp: f32, exp_span: i32, rng: &mut Rng) -> Vec<u16> {
    let mut w: Vec<u16> = Vec::with_capacity(m * k);
    for i in 0..m * k {
        let pick = rng.below(100);
        let sign: u16 = if rng.next() & 1 == 0 { 0 } else { 0x8000 };
        let wild = exp_span > 0;
        let bits = if wild && pick < 4 {
            sign
        } else if wild && pick < 7 {
            sign | (1 + rng.below(127) as u16)
        } else if wild && pick < 10 {
            let e = 127 + rng.below(2 * exp_span as u64 + 1) as i32 - exp_span;
            sign | ((e as u16) << 7) | rng.below(128) as u16
        } else if wild && pick < 13 && i > 0 {
            w[i - 1] ^ 0x8000
        } else {
            bf16_bits((rng.unit() * 2.0 - 1.0) * amp)
        };
        w.push(bits);
    }
    w
}

/// Random F32 activations `[rows * k]`; row magnitudes span 2^exp_lo..2^exp_hi.
fn gen_x(rows: usize, k: usize, exp_lo: i32, exp_hi: i32, rng: &mut Rng) -> Vec<f32> {
    let span = (exp_hi - exp_lo + 1) as u64;
    let mut x: Vec<f32> = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 2f32.powi(exp_lo + ((r as u64 * 5) % span) as i32);
        for i in 0..k {
            let v = if r % 11 == 5 {
                0.0
            } else if r % 11 == 8 {
                -0.0
            } else {
                let pick = rng.below(100);
                let sign = if rng.next() & 1 == 0 {
                    0u32
                } else {
                    0x8000_0000
                };
                if pick < 5 {
                    f32::from_bits(sign)
                } else if pick < 8 {
                    f32::from_bits(sign | (1 + rng.below(0x7f_ffff) as u32))
                } else if r % 7 == 3 && i % 2 == 1 {
                    x[r * k + i - 1]
                } else {
                    let mut v = (rng.unit() * 2.0 - 1.0) * scale;
                    if r % 5 == 0 && i % 2 == 1 {
                        v = -v;
                    }
                    v
                }
            };
            x.push(v);
        }
    }
    x
}

fn upload_raw_pooled(gpu: &mut Gpu, bytes: &[u8], context: &str) -> GpuTensor {
    let t = gpu.alloc_tensor(&[bytes.len()], DType::Raw).expect(context);
    if let Err(err) = gpu.hip.memcpy_htod(&t.buf, bytes) {
        gpu.free_tensor(t).expect("free failed Raw upload");
        panic!("{context}: {err:?}");
    }
    t
}

fn upload_bf16(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut t = upload_raw_pooled(gpu, &bytes, "upload bf16");
    t.dtype = DType::BF16;
    t.shape = vec![bits.len()];
    t
}

fn sentinel_bytes(words: usize) -> Vec<u8> {
    SENTINEL.to_le_bytes().repeat(words)
}

fn fill_sentinel(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .device_synchronize()
        .expect("sync before sentinel fill");
    gpu.hip
        .memcpy_htod(&t.buf, &sentinel_bytes(t.numel()))
        .expect("sentinel fill");
}

fn write_x(gpu: &Gpu, t: &GpuTensor, host: &[f32]) {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.hip.device_synchronize().expect("sync before X rewrite");
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("X rewrite");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A guarded output allocation `[GUARD][n][GUARD]`, all sentinel.
fn alloc_out(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let total = GUARD + n + GUARD;
    gpu.upload_f32(&vec![f32::from_bits(SENTINEL); total], &[total])
        .expect("alloc out")
}

/// The writable view of a guarded output (at least one word so a zero-M slot
/// still has a valid, never-written pointer).
fn out_view(out: &GpuTensor, n: usize) -> GpuTensor {
    out.sub_offset(GUARD, n.max(1))
}

fn guarded(expected: &[u32]) -> Vec<u32> {
    let mut v = vec![SENTINEL; GUARD];
    v.extend_from_slice(expected);
    v.extend(std::iter::repeat_n(SENTINEL, GUARD));
    v
}

fn mismatches(got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len(), "guarded length");
    if let Some((i, (g, w))) = got.iter().zip(want).enumerate().find(|(_, (g, w))| g != w) {
        eprintln!("first mismatch word={i} got=0x{g:08x} expected=0x{w:08x}");
    }
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

fn guarded_rows(expected: &[u32], rows: usize, m: usize) -> Bits {
    assert_eq!(expected.len(), rows * m);
    let mut full = vec![SENTINEL; OUTPUT_ROWS * m];
    full[..expected.len()].copy_from_slice(expected);
    guarded(&full)
}

fn check_k4(got: &[u32], reference: &[u32], rows: usize, m: usize) -> usize {
    let expected = guarded_rows(&reference[..rows * m], rows, m);
    if let Some(i) = got.iter().zip(&expected).position(|(g, e)| g != e) {
        let live = i.checked_sub(GUARD).filter(|&j| j < rows * m);
        eprintln!(
            "matrix=0 R={rows} M={m} first_word={i} row_output={:?} output_bytes={}",
            live.map(|j| (j / m, j % m)),
            rows * m * 4
        );
    }
    mismatches(got, &expected)
}

fn raw(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync immutable input");
    gpu.download_raw_bytes(t).expect("read immutable input")
}

/// Last-launch evidence comes from the actual dispatch, not a route predictor.
fn receipt(gpu: &Gpu, expected: &str, mode: &str, rows: usize, shape: &str) {
    assert_eq!(
        gpu.last_launched_kernel(),
        Some(expected),
        "missing/wrong actual route: {mode} {shape} R={rows}"
    );
    let (module, container_sha, sha) = gpu
        .lab_kernel_device_identity(expected)
        .expect("actual loaded object identity");
    assert!(
        !module.is_empty()
            && sha.len() == 64
            && sha.bytes().all(|b| b.is_ascii_hexdigit())
            && container_sha.len() == 64
            && container_sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid loaded object identity for {expected}"
    );
    println!("G1 arch={} device={} module={module} symbol={expected} object_sha256={sha} container_sha256={container_sha} R={rows} mode={mode:?} {shape}",
        gpu.arch, gpu.device_id);
}

fn tile_symbol(gpu: &Gpu, hc: bool) -> String {
    let family = if hc {
        "gemv_bf16_xf32_k4_rows"
    } else {
        "gemv_bf16_xf32_rows_x4"
    };
    format!("{family}_t8_pm_{}", gpu.arch)
}

/// Rewrite only inactive rows; preserve live input bits including signed zero.
fn poisoned_x(host: &[f32], k: usize, rows: usize) -> Vec<f32> {
    let mut poisoned = host.to_vec();
    poisoned[rows * k..MAX_ROWS * k].fill(f32::from_bits(SENTINEL));
    poisoned
}

fn assert_input(gpu: &Gpu, x: &GpuTensor, expected: &[f32]) {
    let want: Bits = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        mismatches(&fetch(gpu, x), &want),
        0,
        "input/tail bytes mutated"
    );
}

fn assert_weights(gpu: &Gpu, weights: &[&GpuTensor], before: &[Vec<u8>]) {
    for (i, (w, expected)) in weights.iter().zip(before).enumerate() {
        let got = raw(gpu, w);
        if let Some(j) = got.iter().zip(expected).position(|(g, e)| g != e) {
            panic!("immutable weight matrix={i} byte={j} mutated");
        }
        assert_eq!(got.len(), expected.len(), "immutable weight extent");
    }
}

type Bits = Vec<u32>;

#[derive(Default)]
struct Stats {
    runs: usize,
    bad: usize,
}

#[derive(Default)]
struct Report {
    rows: BTreeMap<String, Stats>,
    symbols: BTreeMap<String, usize>,
    failures: usize,
}

impl Report {
    fn record(&mut self, label: &str, detail: &str, bad: usize) {
        if bad != 0 {
            eprintln!("MISMATCH {label} {detail}: {bad} words differ");
            self.failures += 1;
        }
        let e = self.rows.entry(label.to_string()).or_default();
        e.runs += 1;
        e.bad += bad;
    }
    fn symbol(&mut self, sym: &str) {
        *self.symbols.entry(sym.to_string()).or_default() += 1;
    }
    fn fail(&mut self, what: &str) {
        eprintln!("FAIL {what}");
        self.failures += 1;
    }
}

// ───────────────────────────── ordinary BF16 x4 ─────────────────────────────

struct Mat {
    m: usize,
    k: usize,
    w: GpuTensor,
}

fn make_mats(
    gpu: &mut Gpu,
    ms: &[usize; 4],
    k: usize,
    rng: &mut Rng,
    captured: Option<&[Vec<u8>; 4]>,
) -> Vec<Mat> {
    let mut mats = Vec::new();
    for (i, &m) in ms.iter().enumerate() {
        let w = if m == 0 {
            // Zero-M slots have pinned empty captures but still need valid
            // never-read device pointers for the existing grouped ABI.
            upload_bf16(gpu, &[0u16; 8])
        } else if let Some(weights) = captured {
            let bytes = &weights[i];
            let mut w = upload_raw_pooled(gpu, bytes, "upload captured BF16 W");
            w.dtype = DType::BF16;
            w.shape = vec![bytes.len() / 2];
            w
        } else {
            upload_bf16(gpu, &gen_weights(m, k, 0.25, 40, rng))
        };
        mats.push(Mat { m, k, w });
    }
    mats
}

fn parts_of<'a>(
    mats: &'a [Mat],
    views: &'a [GpuTensor],
) -> [(&'a GpuTensor, &'a GpuTensor, usize); 4] {
    [
        (&mats[0].w, &views[0], mats[0].m),
        (&mats[1].w, &views[1], mats[1].m),
        (&mats[2].w, &views[2], mats[2].m),
        (&mats[3].w, &views[3], mats[3].m),
    ]
}

/// `rows` singleton `gemv_bf16_xf32_x4` launches (the decode object of a
/// router/shared group); per matrix `[rows * m]` bits.
fn ref_x4(
    gpu: &mut Gpu,
    mats: &[Mat],
    x: &GpuTensor,
    rows: usize,
    source: Option<&CapturedOrdinary>,
) -> Vec<Bits> {
    let k = mats[0].k;
    let x_before = raw(gpu, x);
    let weights: Vec<&GpuTensor> = mats.iter().map(|m| &m.w).collect();
    let w_before: Vec<Vec<u8>> = weights.iter().map(|w| raw(gpu, w)).collect();
    let outs: Vec<GpuTensor> = mats.iter().map(|m| alloc_out(gpu, rows * m.m)).collect();
    for r in 0..rows {
        let xv = x.sub_offset(r * k, k);
        let views: Vec<GpuTensor> = mats
            .iter()
            .zip(&outs)
            .map(|(m, y)| y.sub_offset(GUARD + r * m.m, m.m.max(1)))
            .collect();
        gpu.gemv_bf16_xf32_x4(parts_of(mats, &views), &xv, k)
            .expect("singleton gemv_bf16_xf32_x4");
        receipt(
            gpu,
            "gemv_bf16_xf32_x4",
            "actual-singleton-x4",
            1,
            &format!(
                "reference_row={r} M={:?} K={k}",
                mats.iter().map(|m| m.m).collect::<Vec<_>>()
            ),
        );
        if let Some(source) = source {
            source.assert_identity(gpu);
        }
    }
    let mut got = Vec::new();
    for (y, m) in outs.iter().zip(mats) {
        let all = fetch(gpu, y);
        let live = all[GUARD..GUARD + rows * m.m].to_vec();
        assert_eq!(
            mismatches(&all, &guarded(&live)),
            0,
            "singleton x4 output guards"
        );
        got.push(live);
    }
    assert_eq!(raw(gpu, x), x_before, "singleton x4 mutated X");
    assert_weights(gpu, &weights, &w_before);
    for y in outs {
        gpu.free_tensor(y).expect("free ref");
    }
    got
}

/// `rows` singleton `gemv_bf16_xf32` launches (the decode object of a single
/// projection) of one matrix.
fn ref_single(
    gpu: &mut Gpu,
    mat: &Mat,
    x: &GpuTensor,
    rows: usize,
    source: Option<&CapturedOrdinary>,
) -> Bits {
    let x_before = raw(gpu, x);
    let w_before = raw(gpu, &mat.w);
    let y = alloc_out(gpu, rows * mat.m);
    for r in 0..rows {
        gpu.gemv_bf16_xf32(
            &mat.w,
            &x.sub_offset(r * mat.k, mat.k),
            &y.sub_offset(GUARD + r * mat.m, mat.m),
            mat.m,
            mat.k,
        )
        .expect("singleton gemv_bf16_xf32");
        receipt(
            gpu,
            "gemv_bf16_xf32",
            "actual-singleton-single",
            1,
            &format!("reference_row={r} M={} K={}", mat.m, mat.k),
        );
        if let Some(source) = source {
            source.assert_identity(gpu);
        }
    }
    let all = fetch(gpu, &y);
    let bits = all[GUARD..GUARD + rows * mat.m].to_vec();
    assert_eq!(
        mismatches(&all, &guarded(&bits)),
        0,
        "singleton single output guards"
    );
    assert_eq!(raw(gpu, x), x_before, "singleton single mutated X");
    assert_weights(gpu, &[&mat.w], &[w_before]);
    gpu.free_tensor(y).expect("free ref");
    bits
}

#[derive(Clone, Copy)]
enum Arm {
    Routed,
    Direct,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Routed => "routed",
            Arm::Direct => "direct-t8",
        }
    }
    fn max_rows(self) -> usize {
        MAX_ROWS
    }
}

#[allow(clippy::too_many_arguments)]
fn direct_x4(
    gpu: &mut Gpu,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let func = tile_symbol(gpu, false);
    let mut b = KernargBlob::new();
    for p in &parts {
        b.push_ptr(p.0.buf.as_ptr());
    }
    b.push_ptr(x.buf.as_ptr());
    for p in &parts {
        b.push_ptr(p.1.buf.as_ptr());
    }
    for p in &parts {
        b.push_i32(p.2 as i32);
    }
    b.push_i32(k as i32);
    b.push_i32(rows as i32);
    b.pad_to(16);
    let blocks: usize = parts.iter().map(|p| p.2).sum();
    keep.push(b);
    let blob = keep.last_mut().expect("blob").as_mut_slice();
    gpu.launch_blob_recorded(
        &func,
        [blocks as u32, rows.div_ceil(8) as u32, 1],
        [32, 1, 1],
        0,
        blob,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_x4(
    gpu: &mut Gpu,
    arm: Arm,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let ms = parts.map(|p| p.2);
    let expected = match arm {
        Arm::Direct => tile_symbol(gpu, false),
        Arm::Routed if rows > 8 => tile_symbol(gpu, false),
        Arm::Routed if rows == 1 => "gemv_bf16_xf32_x4".into(),
        Arm::Routed => format!("gemv_bf16_xf32_x4_rows_r{rows}"),
    };
    match arm {
        Arm::Routed if rows == 1 => gpu
            .gemv_bf16_xf32_x4(parts, x, k)
            .map_err(|e| format!("{e:?}"))?,
        Arm::Routed => gpu
            .gemv_bf16_xf32_x4_rows(parts, x, k, rows)
            .map_err(|e| format!("{e:?}"))?,
        Arm::Direct => direct_x4(gpu, parts, x, k, rows, keep)?,
    }
    receipt(
        gpu,
        &expected,
        arm.name(),
        rows,
        &format!(
            "M={ms:?} K={k} X_bytes={} Y_bytes={:?} tile_grid=[{}, {}, 1] tile_block=32",
            rows * k * 4,
            ms.map(|m| rows * m * 4),
            ms.iter().sum::<usize>(),
            rows.div_ceil(8)
        ),
    );
    Ok(())
}

/// Capture `launch` into a HIP graph, optionally rewrite `swap` (a device
/// buffer and the host contents to write) at its stable address, refill the
/// sentinel and replay.
fn replay_graph(
    gpu: &mut Gpu,
    outs: &[GpuTensor],
    swap: Option<(&GpuTensor, &[f32])>,
    launch: impl FnOnce(&mut Gpu) -> Result<(), String>,
) {
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    let captured = launch(gpu);
    let g = gpu.end_stream_capture().expect("end capture");
    captured.expect("captured launch");
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    if let Some((xt, host)) = swap {
        write_x(gpu, xt, host);
    }
    for y in outs {
        fill_sentinel(gpu, y);
    }
    gpu.launch_graph(&exec).expect("graph replay");
    gpu.hip.device_synchronize().expect("sync");
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
}

fn check_x4(got: &[Bits], want: &[Bits], mats: &[Mat], rows: usize) -> usize {
    got.iter()
        .zip(want)
        .zip(mats)
        .enumerate()
        .map(|(matrix, ((g, w), m))| {
            let expected = guarded_rows(&w[..rows * m.m], rows, m.m);
            if let Some(i) = g.iter().zip(&expected).position(|(g, e)| g != e) {
                let live = i.checked_sub(GUARD).filter(|&j| j < rows * m.m);
                eprintln!(
                    "matrix={matrix} R={rows} first_word={i} row_output={:?} output_bytes={}",
                    live.map(|j| (j / m.m, j % m.m)),
                    rows * m.m * 4
                );
            }
            mismatches(g, &expected)
        })
        .sum()
}

/// One x4 launch of `mats` over `rows` rows into fresh sentinel outputs.
/// Returns the whole guarded outputs and (eager) the launched symbol.
#[allow(clippy::too_many_arguments)]
fn run_x4(
    gpu: &mut Gpu,
    arm: Arm,
    mats: &[Mat],
    x: &GpuTensor,
    rows: usize,
    graph: Option<&[f32]>,
    keep: &mut Vec<KernargBlob>,
) -> (Vec<Bits>, Option<String>) {
    let k = mats[0].k;
    let original: Vec<f32> = fetch(gpu, x).into_iter().map(f32::from_bits).collect();
    let capture_x = poisoned_x(&original, k, rows);
    let replay_x = graph.map(|host| poisoned_x(host, k, rows));
    let weights: Vec<&GpuTensor> = mats.iter().map(|m| &m.w).collect();
    let before: Vec<Vec<u8>> = weights.iter().map(|w| raw(gpu, w)).collect();
    write_x(gpu, x, &capture_x);
    let outs: Vec<GpuTensor> = mats
        .iter()
        .map(|m| alloc_out(gpu, OUTPUT_ROWS * m.m))
        .collect();
    let views: Vec<GpuTensor> = mats
        .iter()
        .zip(&outs)
        .map(|(m, y)| out_view(y, rows * m.m))
        .collect();
    match &replay_x {
        None => launch_x4(gpu, arm, parts_of(mats, &views), x, k, rows, keep).expect("x4 launch"),
        Some(host) => {
            let parts = parts_of(mats, &views);
            replay_graph(gpu, &outs, Some((x, host)), |g| {
                launch_x4(g, arm, parts, x, k, rows, keep)
            });
        }
    }
    let sym = Some(
        gpu.last_launched_kernel()
            .expect("missing actual x4 launch")
            .to_owned(),
    );
    receipt(
        gpu,
        sym.as_deref().expect("x4 receipt"),
        if graph.is_some() {
            "graph-captured-and-replayed-stable-X"
        } else {
            "eager-stable-X"
        },
        rows,
        &format!(
            "M={:?} K={k} X_address=0x{:x} inactive_X_rows={} Y_capacity_rows={OUTPUT_ROWS}",
            mats.iter().map(|m| m.m).collect::<Vec<_>>(),
            x.buf.as_ptr() as u64,
            MAX_ROWS - rows
        ),
    );
    assert_input(gpu, x, replay_x.as_deref().unwrap_or(&capture_x));
    assert_weights(gpu, &weights, &before);
    write_x(gpu, x, graph.unwrap_or(&original));
    let got: Vec<Bits> = outs.iter().map(|y| fetch(gpu, y)).collect();
    for y in outs {
        gpu.free_tensor(y).expect("free out");
    }
    (got, sym)
}

const X4_CASES: &[(&str, [usize; 4], usize)] = &[
    (
        "router|selector|shared_gate|shared_up K=2560",
        [512, 1, 512, 512],
        2560,
    ),
    ("single M=1 K=2560", [1, 0, 0, 0], 2560),
    ("single M=2 K=2560", [2, 0, 0, 0], 2560),
    ("single M=3 K=2560", [3, 0, 0, 0], 2560),
    ("single M=7 K=2560", [7, 0, 0, 0], 2560),
    ("single M=8 K=2560", [8, 0, 0, 0], 2560),
    ("single M=9 K=2560", [9, 0, 0, 0], 2560),
    ("single M=17 K=2560", [17, 0, 0, 0], 2560),
    ("mixed zero-M slots K=2560", [0, 13, 0, 9], 2560),
    ("odd boundaries K=2560", [9, 15, 1, 17], 2560),
    ("one full tile K=256", [9, 8, 0, 17], 256),
    ("partial tile K=320", [9, 8, 0, 17], 320),
    ("one lane in tail tile K=520", [9, 8, 0, 17], 520),
    ("tiny K=72", [9, 8, 0, 17], 72),
    ("K=6144", [16, 1, 0, 33], 6144),
    ("scalar domain K=7", [9, 8, 0, 17], 7),
    ("scalar domain K=100", [9, 8, 0, 17], 100),
    ("scalar domain K=333", [9, 8, 0, 17], 333),
    ("scalar domain K=2561", [9, 8, 0, 17], 2561),
    ("scalar domain K=258", [3, 2, 7, 9], 258),
    ("scalar domain K=259", [3, 2, 7, 9], 259),
    ("scalar domain K=262", [3, 2, 7, 9], 262),
];

fn run_x4_cases(gpu: &mut Gpu, report: &mut Report, seed: u64) {
    let mut keep: Vec<KernargBlob> = Vec::new();
    for (ci, &(name, ms, k)) in X4_CASES.iter().enumerate() {
        let mut rng = Rng((ci as u64 + 1 + seed * 4096).wrapping_mul(0xA24B_AED4_963E_E407));
        let mats = make_mats(gpu, &ms, k, &mut rng, None);
        let xa = gen_x(MAX_ROWS, k, -8, 8, &mut rng);
        let xb = gen_x(MAX_ROWS, k, -6, 10, &mut rng);
        let x = gpu.upload_f32(&xa, &[xa.len()]).expect("upload x");

        // The decode objects of this group on both X contents; the grouped
        // singleton and the single-projection singleton must agree.
        let ref_a = ref_x4(gpu, &mats, &x, MAX_ROWS, None);
        let single_a: Vec<Bits> = mats
            .iter()
            .map(|m| {
                if m.m == 0 {
                    Vec::new()
                } else {
                    ref_single(gpu, m, &x, MAX_ROWS, None)
                }
            })
            .collect();
        for (i, m) in mats.iter().enumerate() {
            if m.m != 0 {
                let bad = mismatches(&ref_a[i], &single_a[i]);
                report.record(
                    &format!("decode objects agree (x4 singleton vs gemv_bf16_xf32) {name}"),
                    &format!("slot {i}"),
                    bad,
                );
            }
        }
        write_x(gpu, &x, &xb);
        let ref_b = ref_x4(gpu, &mats, &x, MAX_ROWS, None);
        write_x(gpu, &x, &xa);
        if ref_a
            .iter()
            .zip(&ref_b)
            .zip(&mats)
            .any(|((a, b), m)| m.m != 0 && a == b)
        {
            report.fail(&format!(
                "{name}: X contents A and B give identical references (vacuous)"
            ));
        }

        let arms = [Arm::Routed, Arm::Direct];
        for arm in arms {
            let label = format!("synthetic seed={seed} x4 {} {name}", arm.name());
            for rows in 1..=arm.max_rows() {
                // Eager, X contents A.
                let (got, sym) = run_x4(gpu, arm, &mats, &x, rows, None, &mut keep);
                report.record(
                    &label,
                    &format!("rows={rows} eager A"),
                    check_x4(&got, &ref_a, &mats, rows),
                );
                report.symbol(sym.as_deref().expect("x4 actual launch evidence"));
                // Eager, X contents rewritten at the same address.
                write_x(gpu, &x, &xb);
                let (got, _) = run_x4(gpu, arm, &mats, &x, rows, None, &mut keep);
                report.record(
                    &label,
                    &format!("rows={rows} eager B"),
                    check_x4(&got, &ref_b, &mats, rows),
                );
                // Graph captured with B, X rewritten to A before the replay.
                if let Arm::Direct = arm {
                    let (got, _) = run_x4(gpu, arm, &mats, &x, rows, Some(&xa), &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} graph (B captured, A replayed)"),
                        check_x4(&got, &ref_a, &mats, rows),
                    );
                } else {
                    write_x(gpu, &x, &xa);
                }
                keep.clear();
            }
        }

        // Negative controls on the router group only.
        if ci == 0 {
            let rows = 20;
            let (got, _) = run_x4(gpu, Arm::Direct, &mats, &x, rows, None, &mut keep);
            let mut xp = xa.clone();
            for r in 0..MAX_ROWS {
                for j in (0..k).step_by(7) {
                    xp[r * k + j] += 3.0 + (j % 5) as f32;
                }
            }
            let xt = gpu.upload_f32(&xp, &[xp.len()]).expect("upload x'");
            let ref_p = ref_x4(gpu, &mats, &xt, rows, None);
            let bad: Vec<usize> = got
                .iter()
                .zip(&ref_p)
                .zip(&mats)
                .map(|((g, w), m)| mismatches(g, &guarded_rows(&w[..rows * m.m], rows, m.m)))
                .collect();
            println!(
                "bf16_rows_oracle: negative control perturbed-x mismatch words per output {bad:?}"
            );
            if bad.iter().any(|&n| n == 0) {
                report.fail("negative control: a perturbed-X reference agreed with the launch (vacuous oracle)");
            }
            let swapped = mismatches(
                &got[2],
                &guarded_rows(&ref_a[3][..rows * mats[3].m], rows, mats[3].m),
            );
            println!("bf16_rows_oracle: negative control swapped shared gate/up mismatch words {swapped}");
            if swapped == 0 {
                report.fail(
                    "negative control: swapped weights agreed with the launch (vacuous oracle)",
                );
            }
            gpu.free_tensor(xt).expect("free x'");
            keep.clear();
        }

        for m in mats {
            gpu.free_tensor(m.w).expect("free weights");
        }
        gpu.free_tensor(x).expect("free x");
    }
}

// ────────────────────────────────── HC k4 ───────────────────────────────────

/// `(m, k)` of the gates / read-down and boundary shapes (all K % 32 == 0).
const K4_SHAPES: [(&str, usize, usize); 11] = [
    ("gates M=4 K=10240", 4, 10240),
    ("read-down M=320 K=10240", 320, 10240),
    ("M=1 K=10240", 1, 10240),
    ("M=2 K=10240", 2, 10240),
    ("M=3 K=10240", 3, 10240),
    ("M=5 K=2048", 5, 2048),
    ("M=7 K=96 (partial lanes)", 7, 96),
    ("M=8 K=32 (minimum partition)", 8, 32),
    ("M=9 K=1056 (one lane in tail tile)", 9, 1056),
    ("M=33 K=1280", 33, 1280),
    ("M=16 K=1024", 16, 1024),
];

#[allow(clippy::too_many_arguments)]
fn direct_k4(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let func = tile_symbol(gpu, true);
    let mut b = KernargBlob::new();
    b.push_ptr(w.buf.as_ptr());
    b.push_ptr(x.buf.as_ptr());
    b.push_ptr(y.buf.as_ptr());
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_f32(act.unwrap_or(1.0));
    b.push_i32(i32::from(act.is_some()));
    b.push_i32(rows as i32);
    b.pad_to(16);
    keep.push(b);
    let blob = keep.last_mut().expect("blob").as_mut_slice();
    gpu.launch_blob_recorded(
        &func,
        [m as u32, rows.div_ceil(8) as u32, 1],
        [128, 1, 1],
        0,
        blob,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_k4(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let expected = match arm {
        Arm::Direct => tile_symbol(gpu, true),
        Arm::Routed if rows > 8 => tile_symbol(gpu, true),
        Arm::Routed if rows == 1 => "gemv_bf16_xf32_k4".into(),
        Arm::Routed => format!("gemv_bf16_xf32_k4_rows_r{rows}"),
    };
    match arm {
        Arm::Routed => gpu
            .gemv_bf16_xf32_k4_rows(w, x, y, m, k, act, rows)
            .map_err(|e| format!("{e:?}"))?,
        Arm::Direct => direct_k4(gpu, w, x, y, m, k, act, rows, keep)?,
    }
    receipt(gpu, &expected, arm.name(), rows,
        &format!("M={m} K={k} scale_bits={:?} X_bytes={} Y_bytes={} tile_grid=[{m}, {}, 1] tile_block=128",
            act.map(f32::to_bits), rows * k * 4, rows * m * 4, rows.div_ceil(8)));
    Ok(())
}

/// `rows` singleton `gemv_bf16_xf32_k4` launches; `[rows * m]` bits.
fn ref_k4(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
) -> Bits {
    ref_k4_bound(gpu, w, x, m, k, act, rows, None)
}

#[allow(clippy::too_many_arguments)]
fn ref_k4_bound(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    source: Option<&CapturedOrdinary>,
) -> Bits {
    let x_before = raw(gpu, x);
    let w_before = raw(gpu, w);
    let y = alloc_out(gpu, rows * m);
    for r in 0..rows {
        gpu.gemv_bf16_xf32_k4(
            w,
            &x.sub_offset(r * k, k),
            &y.sub_offset(GUARD + r * m, m),
            m,
            k,
            act,
        )
        .expect("singleton gemv_bf16_xf32_k4");
        receipt(
            gpu,
            "gemv_bf16_xf32_k4",
            "actual-singleton-k4",
            1,
            &format!(
                "reference_row={r} M={m} K={k} scale_bits={:?}",
                act.map(f32::to_bits)
            ),
        );
        if let Some(source) = source {
            source.assert_identity(gpu);
        }
    }
    let all = fetch(gpu, &y);
    let bits = all[GUARD..GUARD + rows * m].to_vec();
    assert_eq!(
        mismatches(&all, &guarded(&bits)),
        0,
        "singleton k4 output guards"
    );
    assert_eq!(raw(gpu, x), x_before, "singleton k4 mutated X");
    assert_weights(gpu, &[w], &[w_before]);
    gpu.free_tensor(y).expect("free ref");
    bits
}

#[allow(clippy::too_many_arguments)]
fn run_k4(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    graph: Option<&[f32]>,
    keep: &mut Vec<KernargBlob>,
) -> (Bits, Option<String>) {
    let original: Vec<f32> = fetch(gpu, x).into_iter().map(f32::from_bits).collect();
    let capture_x = poisoned_x(&original, k, rows);
    let replay_x = graph.map(|host| poisoned_x(host, k, rows));
    let before = raw(gpu, w);
    write_x(gpu, x, &capture_x);
    let out = alloc_out(gpu, OUTPUT_ROWS * m);
    let view = out_view(&out, rows * m);
    match &replay_x {
        None => launch_k4(gpu, arm, w, x, &view, m, k, act, rows, keep).expect("k4 launch"),
        Some(host) => {
            replay_graph(gpu, std::slice::from_ref(&out), Some((x, host)), |g| {
                launch_k4(g, arm, w, x, &view, m, k, act, rows, keep)
            });
        }
    }
    let sym = Some(
        gpu.last_launched_kernel()
            .expect("missing actual k4 launch")
            .to_owned(),
    );
    receipt(gpu, sym.as_deref().expect("k4 receipt"),
        if graph.is_some() { "graph-captured-and-replayed-stable-X" } else { "eager-stable-X" },
        rows, &format!("M={m} K={k} scale_bits={:?} X_address=0x{:x} inactive_X_rows={} Y_capacity_rows={OUTPUT_ROWS}",
            act.map(f32::to_bits), x.buf.as_ptr() as u64, MAX_ROWS - rows));
    assert_input(gpu, x, replay_x.as_deref().unwrap_or(&capture_x));
    assert_weights(gpu, &[w], &[before]);
    write_x(gpu, x, graph.unwrap_or(&original));
    let bits = fetch(gpu, &out);
    gpu.free_tensor(out).expect("free out");
    (bits, sym)
}

fn run_k4_cases(gpu: &mut Gpu, report: &mut Report, seed: u64) {
    let mut keep: Vec<KernargBlob> = Vec::new();
    for (ci, &(name, m, k)) in K4_SHAPES.iter().enumerate() {
        let mut rng = Rng((ci as u64 + 101 + seed * 4096).wrapping_mul(0xD1B5_4A32_D192_ED03));
        // exp_span 5 keeps products in the HC activation's interesting range.
        let wbits = gen_weights(m, k, 0.05, 5, &mut rng);
        let w = upload_bf16(gpu, &wbits);
        // The second seed stresses finite operands, not merely the scale.
        let (lo, hi) = if seed == 0 { (-6, 6) } else { (-60, 60) };
        let xa = gen_x(MAX_ROWS, k, lo, hi, &mut rng);
        let xb = gen_x(MAX_ROWS, k, lo + 1, hi + 1, &mut rng);
        let x = gpu.upload_f32(&xa, &[xa.len()]).expect("upload x");
        for act in [
            None,
            Some(0.25f32),
            Some(0.0),
            Some(-0.0),
            Some(-0.25),
            Some(f32::MIN_POSITIVE),
            Some(f32::from_bits(1)),
            Some(1.0e20),
            Some(f32::MAX),
        ] {
            let ref_a = ref_k4(gpu, &w, &x, m, k, act, MAX_ROWS);
            write_x(gpu, &x, &xb);
            let ref_b = ref_k4(gpu, &w, &x, m, k, act, MAX_ROWS);
            write_x(gpu, &x, &xa);
            if act.is_none_or(|s| s == 0.25) && ref_a == ref_b {
                report.fail(&format!(
                    "{name} act={act:?}: X contents A and B give identical references (vacuous)"
                ));
            }
            for arm in [Arm::Routed, Arm::Direct] {
                let label = format!("synthetic seed={seed} k4 {} {name} act={act:?}", arm.name());
                for rows in 1..=arm.max_rows() {
                    let (got, sym) = run_k4(gpu, arm, &w, &x, m, k, act, rows, None, &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} eager A"),
                        check_k4(&got, &ref_a, rows, m),
                    );
                    report.symbol(sym.as_deref().expect("k4 actual launch evidence"));
                    write_x(gpu, &x, &xb);
                    let (got, _) = run_k4(gpu, arm, &w, &x, m, k, act, rows, None, &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} eager B"),
                        check_k4(&got, &ref_b, rows, m),
                    );
                    if let Arm::Direct = arm {
                        let (got, _) =
                            run_k4(gpu, arm, &w, &x, m, k, act, rows, Some(&xa), &mut keep);
                        report.record(
                            &label,
                            &format!("rows={rows} graph (B captured, A replayed)"),
                            check_k4(&got, &ref_a, rows, m),
                        );
                    } else {
                        write_x(gpu, &x, &xa);
                    }
                    keep.clear();
                }
            }
        }
        // The activated and raw outputs must differ (the epilogue ran).
        let raw = ref_k4(gpu, &w, &x, m, k, None, 4);
        let act = ref_k4(gpu, &w, &x, m, k, Some(0.25), 4);
        if raw == act {
            report.fail(&format!(
                "{name}: hc_act epilogue is a no-op on this data (vacuous)"
            ));
        }
        gpu.free_tensor(x).expect("free x");
        gpu.free_tensor(w).expect("free weights");
    }
}

// ───────────────────── HC quarters_out -> quarters_in handoff ─────────────────────

fn host_bf16_vec(n: usize, lo: f32, hi: f32, rng: &mut Rng) -> Vec<u16> {
    (0..n)
        .map(|_| bf16_bits(lo + (hi - lo) * rng.unit()))
        .collect()
}

/// One `hyper_write_norm` launch (no `next`, no `clear`): gates from `gates`,
/// or from `q_in` (the `quarters_in` of a previous producer) when present.
/// Returns the whole guarded `output` and `normalized` bits.
fn hc_consume(
    gpu: &mut Gpu,
    streams: &GpuTensor,
    mix: &GpuTensor,
    wn: &GpuTensor,
    gates: &GpuTensor,
    q_in: Option<&GpuTensor>,
    rows: usize,
) -> (Bits, Bits) {
    let mut inputs = vec![streams, mix, wn, gates];
    if let Some(q) = q_in {
        inputs.push(q);
    }
    let before: Vec<Vec<u8>> = inputs.iter().map(|t| raw(gpu, t)).collect();
    let o = alloc_out(gpu, rows * HC_K);
    let ov = out_view(&o, rows * HC_K);
    let n = alloc_out(gpu, rows * HC_K);
    let nv = out_view(&n, rows * HC_K);
    hyper_write_norm(
        gpu,
        &HyperWrite {
            input: streams,
            normalized: &nv,
            mixed: mix,
            gates,
            output: &ov,
            branches: 4,
            hidden: HC_PART,
            state_bf16: false,
        },
        wn,
        &nv,
        q_in,
        None,
        None,
    )
    .expect("hyper_write_norm consumer");
    let symbol = gpu
        .last_launched_kernel()
        .expect("HC consumer route")
        .to_owned();
    receipt(
        gpu,
        &symbol,
        "hc-quarters-consumer",
        rows,
        "branches=4 hidden=2560",
    );
    assert_weights(gpu, &inputs, &before);
    let bits = (fetch(gpu, &o), fetch(gpu, &n));
    for output in [&bits.0, &bits.1] {
        assert_eq!(
            mismatches(output, &guarded(&output[GUARD..GUARD + rows * HC_K])),
            0,
            "HC consumer output guards"
        );
    }
    gpu.free_tensor(o).expect("free o");
    gpu.free_tensor(n).expect("free n");
    bits
}

fn run_hc_handoff(gpu: &mut Gpu, report: &mut Report) {
    let mut rng = Rng(0x4843_BF16);
    let hidden = HC_PART;
    let wn = upload_bf16(gpu, &host_bf16_vec(HC_K, -0.5, 0.5, &mut rng));
    let inject_bits = gen_weights(4, HC_K, 0.05, 0, &mut rng);
    let inject = upload_bf16(gpu, &inject_bits);
    // The four quarter slices of the gate rows: quarter b holds inject[g][b * 2560 ..].
    let mut qw: Vec<GpuTensor> = Vec::new();
    for b in 0..4 {
        let mut bits = Vec::with_capacity(4 * HC_PART);
        for g in 0..4 {
            bits.extend_from_slice(
                &inject_bits[g * HC_K + b * HC_PART..g * HC_K + (b + 1) * HC_PART],
            );
        }
        qw.push(upload_bf16(gpu, &bits));
    }
    let mut keep: Vec<KernargBlob> = Vec::new();
    for rows in 1..=MAX_ROWS {
        for slot in 0..2usize {
            let label = format!("hc handoff slot {slot}");
            let streams = gpu
                .upload_f32(&gen_x(rows, HC_K, -1, 1, &mut rng), &[rows * HC_K])
                .expect("streams");
            let mix = gpu
                .upload_f32(&gen_x(rows, hidden, -1, 0, &mut rng), &[rows * hidden])
                .expect("mix");
            let logits = gen_x(rows, 4, 0, 2, &mut rng);
            let immutable = [&streams, &mix, &wn, &inject];
            let before: Vec<Vec<u8>> = immutable.iter().map(|t| raw(gpu, t)).collect();
            // Fixed-capacity scratch; inactive rows in either slot stay poisoned.
            // [GUARD][gates MAX_ROWS*4][slot A 16*MAX_ROWS][slot B 16*MAX_ROWS][GUARD].
            let cap_words = MAX_ROWS * 4 + 2 * MAX_ROWS * 16;
            let mut host = vec![f32::from_bits(SENTINEL); GUARD + cap_words + GUARD];
            host[GUARD..GUARD + rows * 4].copy_from_slice(&logits);
            let scratch = gpu.upload_f32(&host, &[host.len()]).expect("scratch");
            let gates = scratch.sub_offset(GUARD, rows * 4);
            let q_slot = scratch.sub_offset(GUARD + MAX_ROWS * 4 + slot * 16 * MAX_ROWS, 16 * rows);
            let q_other =
                scratch.sub_offset(GUARD + MAX_ROWS * 4 + (1 - slot) * 16 * MAX_ROWS, 16 * rows);

            // Producer: write the streams and the quarters of the NEXT write's gate rows.
            let out1 = alloc_out(gpu, rows * HC_K);
            let out1v = out_view(&out1, rows * HC_K);
            let norm1 = alloc_out(gpu, rows * HC_K);
            let norm1v = out_view(&norm1, rows * HC_K);
            let next = HyperNextGates {
                norm_weight: &wn,
                inject: &inject,
                quarters: &q_slot,
            };
            hyper_write_norm(
                gpu,
                &HyperWrite {
                    input: &streams,
                    normalized: &norm1v,
                    mixed: &mix,
                    gates: &gates,
                    output: &out1v,
                    branches: 4,
                    hidden,
                    state_bf16: false,
                },
                &wn,
                &norm1v,
                None,
                Some(&next),
                None,
            )
            .expect("hyper_write_norm producer");
            let producer = gpu
                .last_launched_kernel()
                .expect("HC producer route")
                .to_owned();
            receipt(
                gpu,
                &producer,
                "hc-quarters-producer",
                rows,
                "branches=4 hidden=2560",
            );
            assert_weights(gpu, &immutable, &before);
            for output in [&out1, &norm1] {
                let bits = fetch(gpu, output);
                assert_eq!(
                    mismatches(&bits, &guarded(&bits[GUARD..GUARD + rows * HC_K])),
                    0,
                    "HC producer output guards"
                );
            }
            let normalized_before = raw(gpu, &norm1v);
            let scratch_bits = fetch(gpu, &scratch);
            let q_start = GUARD + MAX_ROWS * 4 + slot * 16 * MAX_ROWS;
            let q_dev: Vec<u32> = scratch_bits[q_start..q_start + 16 * rows].to_vec();
            // Canaries: guards, the other slot and the live gates are untouched; the slot is fully written.
            let mut canary_bad = 0usize;
            for (i, &w) in scratch_bits.iter().enumerate() {
                let inside_slot = (q_start..q_start + 16 * rows).contains(&i);
                let in_gates = (GUARD..GUARD + rows * 4).contains(&i);
                if in_gates {
                    canary_bad += usize::from(w != host[i].to_bits());
                } else if inside_slot {
                    canary_bad += usize::from(w == SENTINEL);
                } else {
                    canary_bad += usize::from(w != SENTINEL);
                }
            }
            report.record(&label, &format!("rows={rows} scratch canaries"), canary_bad);

            // The normalized rows are the k4 gate GEMV's input (next norm weight == norm weight).
            // Exact quarter dots against gemv_bf16_xf32 on the quarter slices.
            let mut quarter_bad = 0usize;
            for b in 0..4 {
                let yq = alloc_out(gpu, 4 * rows);
                for r in 0..rows {
                    gpu.gemv_bf16_xf32(
                        &qw[b],
                        &norm1v.sub_offset(r * HC_K + b * HC_PART, HC_PART),
                        &yq.sub_offset(GUARD + r * 4, 4),
                        4,
                        HC_PART,
                    )
                    .expect("quarter gemv");
                    receipt(
                        gpu,
                        "gemv_bf16_xf32",
                        "actual-singleton-quarter",
                        1,
                        &format!("reference_row={r} quarter={b} M=4 K={HC_PART}"),
                    );
                }
                let yq_bits = fetch(gpu, &yq);
                assert_eq!(
                    mismatches(&yq_bits, &guarded(&yq_bits[GUARD..GUARD + rows * 4])),
                    0,
                    "quarter singleton guards"
                );
                for r in 0..rows {
                    for g in 0..4 {
                        if q_dev[r * 16 + g * 4 + b] != yq_bits[GUARD + r * 4 + g] {
                            if quarter_bad == 0 {
                                eprintln!("HC quarters first mismatch R={rows} row={r} gate={g} quarter={b} got={:08x} expected={:08x}",
                                    q_dev[r * 16 + g * 4 + b], yq_bits[GUARD + r * 4 + g]);
                            }
                            quarter_bad += 1;
                        }
                    }
                }
                gpu.free_tensor(yq).expect("free yq");
            }
            report.record(
                &label,
                &format!("rows={rows} quarters_out vs per-quarter gemv"),
                quarter_bad,
            );

            // Folded gate: producer quarters, singleton k4, wide k4 (hc_act off).
            let fold: Vec<u32> = (0..rows * 4)
                .map(|i| {
                    let (r, g) = (i / 4, i % 4);
                    let q = |b: usize| f32::from_bits(q_dev[r * 16 + g * 4 + b]);
                    ((q(0) + q(1)) + (q(2) + q(3))).to_bits()
                })
                .collect();
            let single = ref_k4(gpu, &inject, &norm1v, 4, HC_K, None, rows);
            let wide_out = alloc_out(gpu, rows * 4);
            let wide_view = out_view(&wide_out, rows * 4);
            let norm_before = raw(gpu, &norm1v);
            let inject_before = raw(gpu, &inject);
            launch_k4(
                gpu,
                Arm::Direct,
                &inject,
                &norm1v,
                &wide_view,
                4,
                HC_K,
                None,
                rows,
                &mut keep,
            )
            .expect("tile k4 gates");
            assert_weights(gpu, &[&norm1v, &inject], &[norm_before, inject_before]);
            let wide_bits = fetch(gpu, &wide_out);
            report.record(
                &label,
                &format!("rows={rows} fold(quarters_out) vs singleton k4"),
                mismatches(&fold, &single),
            );
            report.record(
                &label,
                &format!("rows={rows} wide k4 gates vs fold(quarters_out)"),
                mismatches(&wide_bits, &guarded(&fold)),
            );

            // Consumer: gates from quarters_in vs the same launch fed the wide k4 result as gates.
            let poison = alloc_out(gpu, rows * 4);
            let poison_view = out_view(&poison, rows * 4);
            let gates_wide = gpu
                .upload_f32(
                    &wide_bits[GUARD..GUARD + rows * 4]
                        .iter()
                        .map(|&b| f32::from_bits(b))
                        .collect::<Vec<f32>>(),
                    &[rows * 4],
                )
                .expect("gates_wide");
            let (o_q, n_q) =
                hc_consume(gpu, &streams, &mix, &wn, &poison_view, Some(&q_slot), rows);
            let (o_g, n_g) = hc_consume(gpu, &streams, &mix, &wn, &gates_wide, None, rows);
            report.record(
                &label,
                &format!("rows={rows} consumer streams quarters_in vs wide gates"),
                mismatches(&o_q, &o_g),
            );
            report.record(
                &label,
                &format!("rows={rows} consumer normalized quarters_in vs wide gates"),
                mismatches(&n_q, &n_g),
            );
            let poison_after = fetch(gpu, &poison);
            report.record(
                &label,
                &format!("rows={rows} poisoned gates untouched"),
                mismatches(&poison_after, &vec![SENTINEL; poison_after.len()]),
            );
            if rows == 9 {
                // Negative control: the other (stale sentinel) slot as quarters_in must not agree.
                let (o_s, _) =
                    hc_consume(gpu, &streams, &mix, &wn, &poison_view, Some(&q_other), rows);
                if mismatches(&o_s, &o_g) == 0 {
                    report.fail("negative control: stale quarters slot agreed with the correct handoff (vacuous)");
                }
            }
            assert_weights(gpu, &immutable, &before);
            assert_eq!(
                raw(gpu, &norm1v),
                normalized_before,
                "quarter/k4 reference mutated normalized input"
            );
            assert_eq!(
                fetch(gpu, &scratch),
                scratch_bits,
                "HC consumer mutated scratch quarters/gates/guards"
            );

            for t in [
                streams, mix, scratch, out1, norm1, wide_out, poison, gates_wide,
            ] {
                gpu.free_tensor(t).expect("free handoff");
            }
            keep.clear();
        }
    }
    for t in qw {
        gpu.free_tensor(t).expect("free qw");
    }
    for t in [wn, inject] {
        gpu.free_tensor(t).expect("free");
    }
}

/// External model captures are an additional claim, never generated substitutes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedHc {
    arch: String,
    source: String,
    m: usize,
    k: usize,
    scale: Option<f32>,
    #[serde(default)]
    hc_act: Option<bool>,
    weights: PathBuf,
    weights_sha256: String,
    x: PathBuf,
    x_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn captured_bytes(base: &Path, path: &Path, expected: &str, bytes: usize) -> Vec<u8> {
    assert!(
        expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
        "capture file must have a SHA256"
    );
    let full = base.join(path);
    let data = std::fs::read(&full).unwrap_or_else(|e| panic!("capture {}: {e}", full.display()));
    assert_eq!(data.len(), bytes, "capture extent {}", full.display());
    assert_eq!(
        sha256(&data),
        expected.to_ascii_lowercase(),
        "capture hash {}",
        full.display()
    );
    data
}

struct CapturedOrdinary {
    name: String,
    form: String,
    ms: [usize; 4],
    k: usize,
    scale: Option<f32>,
    weights: [Vec<u8>; 4],
    x: Vec<f32>,
    symbol: String,
    module: String,
    object_sha: String,
    container_sha: Option<String>,
}

impl CapturedOrdinary {
    fn assert_identity(&self, gpu: &Gpu) {
        assert_eq!(
            gpu.last_launched_kernel(),
            Some(self.symbol.as_str()),
            "captured source must use its actual single/x4 C1 reference"
        );
        let (module, container_sha, sha) = gpu
            .lab_kernel_device_identity(&self.symbol)
            .expect("loaded captured C1 identity");
        assert_eq!(module, self.module, "captured C1 module mismatch");
        assert_eq!(sha, self.object_sha, "captured C1 raw object SHA mismatch");
        if let Some(expected) = &self.container_sha {
            assert_eq!(
                &container_sha, expected,
                "captured C1 container SHA mismatch"
            );
        }
    }
}

fn capture_sha(value: &serde_json::Value) -> &str {
    let pin = value.as_str().expect("capture needs SHA256 string");
    assert!(
        pin.len() == 64 && pin.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid capture SHA256"
    );
    pin
}

fn capture_relative(base: &Path, value: &serde_json::Value) -> PathBuf {
    let path = Path::new(value.as_str().expect("capture needs relative path"));
    assert!(
        !path.as_os_str().is_empty()
            && !path.is_absolute()
            && path
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "capture operand/provenance path must stay inside its fixture"
    );
    base.join(path)
}

/// Same streamed model/prompt pin policy as the canonical operands consumer.
fn capture_artifact(
    base: &Path,
    value: &serde_json::Value,
    verified: &mut BTreeMap<PathBuf, String>,
) {
    use std::io::Read as _;
    let raw = value["path"].as_str().expect("model/prompt path required");
    assert!(!raw.is_empty(), "model/prompt path required");
    let path = base.join(raw);
    let pin = capture_sha(&value["sha256"]).to_ascii_lowercase();
    if let Some(previous) = verified.get(&path) {
        assert_eq!(previous, &pin, "conflicting model/prompt pins");
        return;
    }
    let mut file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("capture artifact {}: {e}", path.display()));
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).expect("read capture artifact");
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        pin,
        "capture artifact SHA"
    );
    verified.insert(path, pin);
}

fn ordinary_extent(a: usize, b: usize, width: usize) -> usize {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(width))
        .filter(|&n| n <= i32::MAX as usize)
        .expect("captured ordinary extent exceeds device offset domain")
}

fn load_captured_ordinary(
    gpu: &Gpu,
    path: &Path,
    verified: &mut BTreeMap<PathBuf, String>,
) -> CapturedOrdinary {
    let manifest_bytes = std::fs::read(path).expect("read ordinary capture manifest");
    let c: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("parse ordinary capture manifest");
    assert_eq!(c["schema"], "qwen4-rowtile-operands-v1");
    assert_eq!(
        c["arch"].as_str(),
        Some(gpu.arch.as_str()),
        "capture architecture"
    );
    assert_eq!(c["rows"].as_u64(), Some(MAX_ROWS as u64));
    assert_eq!(c["family"], "bf16", "ordinary capture requires BF16");
    let name = c["name"]
        .as_str()
        .filter(|s| !s.is_empty())
        .expect("capture name");
    let form = c["form"].as_str().expect("capture form");
    assert!(
        matches!(form, "single" | "x4" | "hc_k4"),
        "only single/x4/HC captures"
    );
    let scale = if form == "hc_k4" {
        let active = c["hc_act"].as_bool().expect("HC activation flag");
        assert!(c.get("scale").is_some(), "HC scale required");
        if active {
            let value = c["scale"].as_f64().expect("activated HC scale");
            let scale = value as f32;
            assert!(value.is_finite() && scale.is_finite(), "finite HC scale");
            Some(scale)
        } else {
            assert!(c["scale"].is_null(), "raw HC scale must be null");
            None
        }
    } else {
        assert_eq!(
            c["hc_act"], false,
            "ordinary capture cannot be activated HC"
        );
        assert!(
            c.get("scale").is_some() && c["scale"].is_null(),
            "ordinary scale must be null"
        );
        None
    };
    let k = c["k"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .expect("capture K");
    let heights = c["ms"]
        .as_array()
        .filter(|a| a.len() == 4)
        .expect("four capture heights");
    let mut ms = [0usize; 4];
    for (m, value) in ms.iter_mut().zip(heights) {
        *m = value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n <= i32::MAX as usize)
            .expect("capture M");
        ordinary_extent(OUTPUT_ROWS, *m, 4);
    }
    let total_m = ms
        .iter()
        .try_fold(0usize, |n, &m| n.checked_add(m))
        .expect("capture grid extent");
    assert!(
        total_m > 0 && total_m <= i32::MAX as usize,
        "capture grid domain"
    );
    if matches!(form, "single" | "hc_k4") {
        assert!(
            ms[0] > 0 && ms[1..] == [0, 0, 0],
            "single source requires only slot0"
        );
    }
    if form == "hc_k4" {
        assert_eq!(k % 32, 0, "HC K alignment");
    }
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let source_path = capture_relative(base, &c["provenance"]["source"]);
    let source_bytes = std::fs::read(&source_path).expect("read ordinary source index");
    let source_sha = capture_sha(&c["provenance"]["source_sha256"]);
    assert_eq!(
        sha256(&source_bytes),
        source_sha.to_ascii_lowercase(),
        "source index SHA"
    );
    let source: serde_json::Value =
        serde_json::from_slice(&source_bytes).expect("parse source index");
    assert_eq!(source["schema"], "qwen4-rowtile-source-v1");
    assert_eq!(source["capture_mode"], "eager-liveprojection-operands-only");
    assert_eq!(source["retained_identity_claim"], false);
    for key in [
        "name", "family", "form", "ms", "k", "rows", "weights", "x", "hc_act", "scale",
    ] {
        assert!(c.get(key).is_some(), "missing manifest binding {key}");
        assert_eq!(source.get(key), c.get(key), "capture/source binding {key}");
    }
    assert_eq!(source["component_offset"], 0, "ordinary component offset");
    assert!(
        source.get("mixed").is_some() && source["mixed"].is_null(),
        "not a mixed caller"
    );
    let singleton = &source["singleton"];
    let symbol = singleton["symbol"]
        .as_str()
        .expect("source singleton symbol");
    assert_eq!(
        symbol,
        if form == "hc_k4" {
            "gemv_bf16_xf32_k4"
        } else if form == "single" {
            "gemv_bf16_xf32"
        } else {
            "gemv_bf16_xf32_x4"
        },
        "source form requires its actual single/x4 C1, not a labelled substitute"
    );
    let module = singleton["module"]
        .as_str()
        .filter(|s| !s.is_empty())
        .expect("source module");
    let object_sha = capture_sha(&singleton["object_sha256"]).to_ascii_lowercase();
    let container_sha = singleton
        .get("container_sha256")
        .map(|value| capture_sha(value).to_ascii_lowercase());
    if form == "hc_k4" {
        assert!(
            container_sha.is_some(),
            "canonical HC requires container SHA"
        );
    }
    let recipes = source["captures"]
        .as_array()
        .filter(|a| a.len() == MAX_ROWS)
        .expect("source needs 64 real AR recipes");
    let start = recipes[0]["position"].as_u64().expect("source position");
    let occurrence = recipes[0]["occurrence"]
        .as_u64()
        .expect("source occurrence");
    let grid = serde_json::json!([total_m, 1, 1]);
    let block = serde_json::json!([if form == "hc_k4" { 128 } else { 32 }, 1, 1]);
    for (row, recipe) in recipes.iter().enumerate() {
        assert_eq!(recipe["row"].as_u64(), Some(row as u64), "source row");
        let position = start
            .checked_add(row as u64)
            .expect("source position overflow");
        assert_eq!(
            recipe["position"].as_u64(),
            Some(position),
            "source sequential position"
        );
        assert_eq!(
            recipe["occurrence"].as_u64(),
            Some(occurrence),
            "source stable occurrence"
        );
        assert!(
            recipe["token"]
                .as_u64()
                .is_some_and(|n| n <= u32::MAX as u64),
            "source token"
        );
        assert_eq!(recipe["grid"], grid, "source actual singleton grid");
        assert_eq!(recipe["block"], block, "source actual singleton block");
        assert_eq!(
            recipe["shared_mem"].as_u64(),
            Some(0),
            "source singleton LDS"
        );
    }
    let source_base = source_path.parent().unwrap_or(base);
    capture_artifact(source_base, &source["model"], verified);
    capture_artifact(source_base, &source["prompt"], verified);
    let descriptors = c["weights"]
        .as_array()
        .filter(|a| a.len() == 4)
        .expect("four pinned W files");
    let bindings = source["weight_bindings"]
        .as_array()
        .filter(|a| a.len() == 4)
        .expect("four immutable W bindings");
    let mut weights: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
    for i in 0..4 {
        let bytes = ordinary_extent(ms[i], k, 2);
        assert_eq!(
            bindings[i]["bytes"].as_u64(),
            Some(bytes as u64),
            "W binding extent"
        );
        assert!(
            ["bytes", "byte_offset", "name", "layer"]
                .iter()
                .all(|key| bindings[i].get(*key).is_some())
                && bindings[i]["byte_offset"].as_u64().is_some()
                && (bindings[i]["name"].is_null() || bindings[i]["name"].is_string())
                && (bindings[i]["layer"].is_null() || bindings[i]["layer"].is_i64()),
            "immutable W binding metadata"
        );
        let w_path = capture_relative(base, &descriptors[i]["path"]);
        weights[i] = captured_bytes(
            Path::new(""),
            &w_path,
            capture_sha(&descriptors[i]["sha256"]),
            bytes,
        );
        assert!(
            weights[i].chunks_exact(2).all(|b| {
                f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).is_finite()
            }),
            "capture requires finite BF16 W"
        );
    }
    let x_path = capture_relative(base, &c["x"]["path"]);
    let xb = captured_bytes(
        Path::new(""),
        &x_path,
        capture_sha(&c["x"]["sha256"]),
        ordinary_extent(MAX_ROWS, k, 4),
    );
    let x: Vec<f32> = xb
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(
        x.iter().all(|v| v.is_finite()),
        "capture requires finite F32 X"
    );
    println!("G1 capture_manifest={} manifest_sha256={} source_sha256={source_sha} model_sha256={} prompt_sha256={} W_sha256={:?} X_sha256={} form={form} source_singleton={module}/{symbol} object_sha256={object_sha} container_sha256={container_sha:?} retained_identity_claim=false",
        path.display(), sha256(&manifest_bytes), source["model"]["sha256"], source["prompt"]["sha256"],
        descriptors.iter().map(|w| &w["sha256"]).collect::<Vec<_>>(), c["x"]["sha256"]);
    CapturedOrdinary {
        name: name.to_string(),
        form: form.to_string(),
        ms,
        k,
        scale,
        weights,
        x,
        symbol: symbol.to_string(),
        module: module.to_string(),
        object_sha,
        container_sha,
    }
}

fn run_captured_ordinary(gpu: &mut Gpu, report: &mut Report, c: &CapturedOrdinary) {
    assert!(
        matches!(c.form.as_str(), "single" | "x4"),
        "--captured-ordinary requires ordinary single/x4 form"
    );
    let mut unused_rng = Rng(1);
    let mats = make_mats(gpu, &c.ms, c.k, &mut unused_rng, Some(&c.weights));
    let x = gpu
        .upload_f32(&c.x, &[c.x.len()])
        .expect("upload captured ordinary X");
    // Preserve the source form, including x4 with only one nonzero slot.
    let reference = if c.form == "single" {
        vec![
            ref_single(gpu, &mats[0], &x, MAX_ROWS, Some(c)),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]
    } else {
        ref_x4(gpu, &mats, &x, MAX_ROWS, Some(c))
    };
    let mut keep = Vec::new();
    for arm in [Arm::Routed, Arm::Direct] {
        let label = format!(
            "captured-ordinary {} {} source={:?}",
            c.form,
            arm.name(),
            c.name
        );
        for rows in 1..=MAX_ROWS {
            let (got, sym) = run_x4(gpu, arm, &mats, &x, rows, None, &mut keep);
            report.symbol(sym.as_deref().expect("captured ordinary actual route"));
            report.record(
                &label,
                &format!("rows={rows} eager"),
                check_x4(&got, &reference, &mats, rows),
            );
            if let Arm::Direct = arm {
                let (got, _) = run_x4(gpu, arm, &mats, &x, rows, Some(&c.x), &mut keep);
                report.record(
                    &label,
                    &format!("rows={rows} graph"),
                    check_x4(&got, &reference, &mats, rows),
                );
            }
            keep.clear();
        }
    }
    assert_input(gpu, &x, &c.x);
    for (i, mat) in mats.into_iter().enumerate() {
        if mat.m != 0 {
            assert_eq!(
                raw(gpu, &mat.w),
                c.weights[i],
                "captured ordinary immutable W"
            );
        }
        gpu.free_tensor(mat.w).expect("free captured ordinary W");
    }
    gpu.free_tensor(x).expect("free captured ordinary X");
}

fn capture_paths() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut hc = Vec::new();
    let mut ordinary = Vec::new();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let paths = if arg == "--captured-hc" {
            &mut hc
        } else if arg == "--captured-ordinary" {
            &mut ordinary
        } else {
            panic!("unknown argument {arg:?}; expected --captured-hc or --captured-ordinary MANIFEST.json");
        };
        paths.push(PathBuf::from(
            args.next().expect("capture flag needs a manifest path"),
        ));
    }
    (hc, ordinary)
}

fn run_captured_hc(gpu: &mut Gpu, report: &mut Report, path: &Path) {
    let manifest = std::fs::read(path).expect("read captured HC manifest");
    let value: serde_json::Value =
        serde_json::from_slice(&manifest).expect("parse captured HC manifest");
    let mut canonical = if value.get("schema").is_some() {
        let loaded = load_captured_ordinary(gpu, path, &mut BTreeMap::new());
        assert_eq!(loaded.form, "hc_k4", "--captured-hc requires HC form");
        Some(loaded)
    } else {
        None
    };
    let c: CapturedHc = if let Some(source) = &canonical {
        CapturedHc {
            arch: gpu.arch.clone(),
            source: source.name.clone(),
            m: source.ms[0],
            k: source.k,
            scale: source.scale,
            hc_act: Some(source.scale.is_some()),
            weights: PathBuf::new(),
            weights_sha256: capture_sha(&value["weights"][0]["sha256"]).to_string(),
            x: PathBuf::new(),
            x_sha256: capture_sha(&value["x"]["sha256"]).to_string(),
        }
    } else {
        serde_json::from_value(value).expect("parse legacy captured HC manifest")
    };
    if let Some(active) = c.hc_act {
        assert_eq!(active, c.scale.is_some(), "HC activation/scale mismatch");
    }
    assert_eq!(c.arch, gpu.arch, "capture must match actual architecture");
    assert!(!c.source.trim().is_empty(), "capture provenance required");
    assert!(
        c.m > 0 && c.k > 0 && c.k % 32 == 0 && c.scale.is_none_or(f32::is_finite),
        "invalid captured HC domain"
    );
    // Device byte offsets remain 32-bit; do not silently truncate extents.
    let w_bytes =
        c.m.checked_mul(c.k)
            .and_then(|n| n.checked_mul(2))
            .expect("capture W extent");
    let x_bytes = MAX_ROWS
        .checked_mul(c.k)
        .and_then(|n| n.checked_mul(4))
        .expect("capture X extent");
    let y_bytes = OUTPUT_ROWS
        .checked_mul(c.m)
        .and_then(|n| n.checked_mul(4))
        .expect("capture Y extent");
    assert!(
        c.m <= i32::MAX as usize
            && c.k <= i32::MAX as usize
            && [w_bytes, x_bytes, y_bytes]
                .iter()
                .all(|&n| n <= i32::MAX as usize),
        "captured HC extent exceeds device offset domain"
    );
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let (wb, x_host) = if let Some(source) = &mut canonical {
        (
            std::mem::take(&mut source.weights[0]),
            std::mem::take(&mut source.x),
        )
    } else {
        let wb = captured_bytes(base, &c.weights, &c.weights_sha256, w_bytes);
        let xb = captured_bytes(base, &c.x, &c.x_sha256, x_bytes);
        let x = xb
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>();
        (wb, x)
    };
    let weights: Vec<u16> = wb
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert!(
        weights
            .iter()
            .all(|&w| f32::from_bits((w as u32) << 16).is_finite())
            && x_host.iter().all(|x| x.is_finite()),
        "capture claim requires finite model operands"
    );
    println!("G1 capture_manifest={} manifest_sha256={} source={:?} W_sha256={} X_sha256={} M={} K={} scale_bits={:?} hc_act={} capture_format={}",
        path.display(), sha256(&manifest), c.source, c.weights_sha256, c.x_sha256,
        c.m, c.k, c.scale.map(f32::to_bits), c.scale.is_some(),
        if canonical.is_some() { "qwen4-rowtile-operands-v1" } else { "legacy-hc" });
    let w = upload_bf16(gpu, &weights);
    let x = gpu
        .upload_f32(&x_host, &[x_host.len()])
        .expect("upload captured X");
    let mut keep = Vec::new();
    // Canonical qualification is exactly the captured mode; legacy fixtures retain
    // their historical extra raw-mode exercise, labelled separately.
    let modes = if canonical.is_none() && c.scale.is_some() {
        vec![c.scale, None]
    } else {
        vec![c.scale]
    };
    for act in modes {
        let reference = ref_k4_bound(gpu, &w, &x, c.m, c.k, act, MAX_ROWS, canonical.as_ref());
        for arm in [Arm::Routed, Arm::Direct] {
            let label = format!(
                "{} {} source={:?} act={act:?}",
                if act == c.scale {
                    "captured-HC"
                } else {
                    "legacy-extra-HC"
                },
                arm.name(),
                c.source
            );
            for rows in 1..=MAX_ROWS {
                let (got, sym) = run_k4(gpu, arm, &w, &x, c.m, c.k, act, rows, None, &mut keep);
                report.symbol(sym.as_deref().expect("captured HC actual route"));
                report.record(
                    &label,
                    &format!("rows={rows} eager"),
                    check_k4(&got, &reference, rows, c.m),
                );
                if let Arm::Direct = arm {
                    let (got, _) = run_k4(
                        gpu,
                        arm,
                        &w,
                        &x,
                        c.m,
                        c.k,
                        act,
                        rows,
                        Some(&x_host),
                        &mut keep,
                    );
                    report.record(
                        &label,
                        &format!("rows={rows} graph"),
                        check_k4(&got, &reference, rows, c.m),
                    );
                }
                keep.clear();
            }
        }
    }
    assert_eq!(raw(gpu, &w), wb, "capture W immutable");
    assert_input(gpu, &x, &x_host);
    gpu.free_tensor(w).expect("free captured W");
    gpu.free_tensor(x).expect("free captured X");
}

fn main() {
    let (captures, ordinary_paths) = capture_paths();
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    println!("bf16_rows_oracle: arch={arch}");
    if !matches!(arch.as_str(), "gfx1151" | "gfx1201") {
        eprintln!("bf16_rows_oracle: needs exact gfx1151 or gfx1201");
        std::process::exit(2);
    }
    if !WideTwinFamily::Bf16.enabled() {
        eprintln!(
            "bf16_rows_oracle: the routed arm needs {}=1 (wide twin switch is default off)",
            WideTwinFamily::Bf16.env_var()
        );
        std::process::exit(2);
    }
    // Only eager routed calls use the selector; direct tiles must also cover R=1.
    gpu.set_wide_verify_rows(MAX_ROWS);
    // Load exact-architecture fixed8 objects before graph capture.
    for hc in [false, true] {
        let func = tile_symbol(&gpu, hc);
        gpu.ensure_row_tile_pm_entry(&func)
            .unwrap_or_else(|e| panic!("load {func}: {e:?}"));
    }
    let mut report = Report::default();
    for seed in 0..2 {
        run_x4_cases(&mut gpu, &mut report, seed);
        run_k4_cases(&mut gpu, &mut report, seed);
    }
    run_hc_handoff(&mut gpu, &mut report);
    for path in &captures {
        run_captured_hc(&mut gpu, &mut report, path);
    }
    let mut verified_artifacts = BTreeMap::new();
    let mut single_captures = 0;
    let mut x4_captures = 0;
    for path in &ordinary_paths {
        let capture = load_captured_ordinary(&gpu, path, &mut verified_artifacts);
        run_captured_ordinary(&mut gpu, &mut report, &capture);
        if capture.form == "single" {
            single_captures += 1;
        } else {
            x4_captures += 1;
        }
    }
    if captures.is_empty() && ordinary_paths.is_empty() {
        println!("G1 capture_status=NOT_PROVIDED qualification=synthetic-components-only real-model-capture-gate=unqualified");
    } else {
        println!(
            "G1 capture_status=executed hc_manifests={} ordinary_single_manifests={single_captures} ordinary_x4_manifests={x4_captures} scope=provided-component-captures-only full-G1-qualification=withheld",
            captures.len()
        );
    }

    println!("bf16_rows_oracle: per-case summary (runs, mismatching words)");
    for (label, s) in &report.rows {
        println!("  {label:<84} runs={:<5} mismatch_words={}", s.runs, s.bad);
    }
    println!("bf16_rows_oracle: routed launches by symbol");
    for (sym, n) in &report.symbols {
        println!("  {sym:<44} {n}");
    }
    let runs: usize = report.rows.values().map(|s| s.runs).sum();
    let bad: usize = report.rows.values().map(|s| s.bad).sum();
    if report.failures != 0 || bad != 0 {
        println!(
            "bf16_rows_oracle: FAIL runs={runs} failures={} mismatch_words={bad}",
            report.failures,
        );
        std::process::exit(1);
    }
    println!("bf16_rows_oracle: PASS executed component arms runs={runs} mismatch_words=0");
}
