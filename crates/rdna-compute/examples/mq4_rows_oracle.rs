// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-input MQ4G256V2 raw-byte oracle: direct tile R=1..=64 versus the
//! actual production `Gpu::gemv_mq4g256v2` singleton, with loaded-object receipts.
//! No CPU, performance, or retained-certificate claim is made.
//!
//! Qualification requires `--capture DIR` (repeatable), in addition to all
//! three seeded synthetic sweeps. `--synthetic-only` and `--diagnostic-product`
//! are explicitly nonqualifying; only the latter runs product/graph diagnostics.
//! Unknown options, missing routes/metadata, mutation, and byte mismatches fail.
//!
//! DIR/meta.json schema:
//! `{"schema":"mq4_rows_oracle.capture.v1","arch":"gfx1201","M":3,"K":256,
//! "rows":64,"weights":{"path":"weights.bin","sha256":"<64 hex>"},
//! "activations":{"path":"x.bin","sha256":"<64 hex>"},
//! "provenance":{"source":"source-index.json",
//! "source_sha256":"<64 hex>"}}`.
//! Weights are exact packed `[M][K/256]` 136-byte groups; activations are exact
//! little-endian f32 `[rows][K]`. At least 64 genuine captured rows are required;
//! naturally repeated values/rows are legal. The first 64 are tested without
//! manufacturing additional captured rows. Candidate activation backing has
//! NaN prefix/suffix guards and at least eight poisoned tail rows, even at R=64.
//! All packed f16 headers and f32 activations must be finite. All three files
//! (including the source provenance artifact) are required and SHA256-verified.
//! Capturing code remains responsible for the stated source/operand relationship.
//!
//! Example: `cargo run --release -p rdna-compute --features lab
//! --example mq4_rows_oracle -- --capture /path/to/capture`.

use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

const GUARD: usize = 64;
const MAX_ROWS: usize = 64;
const NAN_BITS: u32 = 0x7fc0_0000;
/// Graphs are diagnostic only, never evidence for the direct tile.
const GRAPH_ROWS: [usize; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64];

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
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// Host packer: `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0),
/// hB (s1, z1), then 128 bytes of nibbles (element e at byte e / 2, low
/// nibble = even e).
fn pack_random_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 136);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            out.extend_from_slice(&random_scale(rng).to_le_bytes());
            out.extend_from_slice(&random_zero(rng).to_le_bytes());
        }
        for _ in 0..128 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

struct Mat {
    name: String,
    m: usize,
    k: usize,
    w: GpuTensor,
    /// Singleton reference, `[MAX_ROWS][m]` result bits; row r is
    /// `gemv_mq4g256v2` on activation row r. Rows `..R` are the reference of R rows.
    reference: Vec<u32>,
    reference_receipts: Vec<Value>,
}

const SHAPES: [(&str, usize, usize); 10] = [
    ("gdn.qkv", 10240, 2560),
    ("gdn.z", 6144, 2560),
    ("gdn.a", 48, 2560),
    ("gdn.b", 48, 2560),
    ("gdn.out", 2560, 6144),
    ("qsa.q", 12288, 2560),
    ("qsa.k", 512, 2560),
    ("qsa.v", 512, 2560),
    ("qsa.indexer", 640, 2560),
    ("qsa.o", 2560, 6144),
];

/// Declared projection groupings, not a claim of captured model census coverage.
const GROUPS: [(&str, &[usize]); 3] = [
    ("group gdn qkv|z|a|b K=2560", &[0, 1, 2, 3]),
    ("group qsa q|k|v|indexer K=2560", &[5, 6, 7, 8]),
    ("group out|o K=6144", &[4, 9]),
];

/// Synthetic single-matrix grid.
const SYN_M: [usize; 12] = [1, 2, 3, 7, 8, 9, 15, 17, 33, 63, 64, 65];
const SYN_K: [usize; 7] = [256, 512, 768, 1024, 1280, 2560, 6144];
/// Synthetic x4 groupings: M per slot (0 = zero-M part, any slot), at these K.
const SYN_GROUPS: [[usize; 4]; 8] = [
    [9, 0, 7, 1],
    [1, 1, 1, 1],
    [3, 0, 0, 13],
    [8, 9, 8, 7],
    [0, 5, 0, 2],
    [17, 16, 15, 33],
    [0, 0, 0, 9],
    [7, 3, 5, 11],
];
const SYN_GROUP_K: [usize; 7] = SYN_K;

type Slots<'a> = [Option<&'a Mat>; 4];

fn nan_bytes(words: usize) -> Vec<u8> {
    NAN_BITS.to_le_bytes().repeat(words)
}

fn fill_nan(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync before NaN fill");
    gpu.hip
        .memcpy_htod(&t.buf, &nan_bytes(t.numel()))
        .expect("NaN fill");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    let bytes = snapshot(gpu, t);
    assert_eq!(
        bytes.len(),
        t.numel() * 4,
        "fetch requires a complete f32 allocation"
    );
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("raw output word")))
        .collect()
}

/// Activation rows of every character: ordinary, wildly scaled, alternating
/// sign (cancellation), signed zeros, and subnormals. Finite for every weight
/// the packer draws (|w| <= ~5, K <= 6144).
fn host_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAX_ROWS * k);
    for r in 0..MAX_ROWS {
        let e = ((r * 7 + 3) % 13) as i32 * 10 - 60;
        for j in 0..k {
            let base = (rng.unit() + rng.unit() - 1.0) * 1.5;
            let v = match r % 8 {
                0 => base,
                3 => {
                    // Alternating signs of nearly equal magnitude: heavy cancellation.
                    let m = 0.75 + 0.015625 * (rng.unit() - 0.5);
                    if j % 2 == 0 {
                        m
                    } else {
                        -m
                    }
                }
                5 => match j % 3 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => base * 2f32.powi(-100),
                },
                6 => base * 2f32.powi(-135),
                _ => base * 2f32.powi(e),
            };
            x.push(v);
        }
    }
    x
}

fn upload_x(gpu: &mut Gpu, k: usize, rng: &mut Rng) -> GpuTensor {
    let x = host_x(k, rng);
    gpu.upload_f32(&x, &[x.len()]).expect("upload x")
}

fn valid_sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn identity(gpu: &Gpu, symbol: &str) -> Value {
    assert!(!symbol.is_empty(), "missing launched symbol");
    let (module, container_sha, sha) = gpu
        .lab_kernel_device_identity(symbol)
        .expect("actual loaded kernel identity");
    assert!(
        !module.is_empty() && valid_sha(&sha) && valid_sha(&container_sha),
        "missing/invalid loaded object metadata"
    );
    json!({"symbol": symbol, "module": module, "object_sha256": sha, "container_sha256": container_sha})
}

/// Snapshot the whole allocation, including bytes outside a tensor subview.
fn snapshot(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip
        .device_synchronize()
        .expect("sync before backing snapshot");
    let mut bytes = vec![0; t.buf.size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .expect("snapshot backing bytes");
    bytes
}

fn unchanged(gpu: &Gpu, t: &GpuTensor, before: &[u8], label: &str) {
    assert_eq!(snapshot(gpu, t), before, "mutated input backing: {label}");
}

/// The actual AR singleton, never a substitute PM C1 or CPU arithmetic.
fn singleton_recorded(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
) -> (Vec<u32>, Vec<Value>) {
    let n = 2 * GUARD + MAX_ROWS * m;
    let y = gpu
        .upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n])
        .expect("alloc ref");
    let wb = snapshot(gpu, w);
    let xb = snapshot(gpu, x);
    let mut routes = Vec::with_capacity(MAX_ROWS);
    let mut completed = Vec::with_capacity(MAX_ROWS * m);
    for r in 0..MAX_ROWS {
        gpu.gemv_mq4g256v2(
            w,
            &x.sub_offset(r * k, k),
            &y.sub_offset(GUARD + r * m, m),
            m,
            k,
        )
        .expect("production singleton gemv_mq4g256v2");
        let symbol = gpu
            .last_launched_kernel()
            .expect("singleton missing route")
            .to_owned();
        let route = identity(gpu, &symbol);
        let bits = fetch(gpu, &y);
        assert_eq!(
            &bits[GUARD..GUARD + r * m],
            completed.as_slice(),
            "singleton overwrote a preceding reference row"
        );
        completed.extend_from_slice(&bits[GUARD + r * m..GUARD + (r + 1) * m]);
        assert!(
            bits[..GUARD]
                .iter()
                .chain(&bits[GUARD + (r + 1) * m..])
                .all(|&b| b == NAN_BITS),
            "singleton wrote guards/inactive rows"
        );
        unchanged(gpu, w, &wb, "singleton weights");
        unchanged(gpu, x, &xb, "singleton x");
        routes.push(json!({"row": r, "M": m, "K": k, "route": route}));
    }
    gpu.free_tensor(y).expect("free ref");
    (completed, routes)
}

fn singleton_reference(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
) -> Vec<u32> {
    let (bits, routes) = singleton_recorded(gpu, w, x, m, k);
    println!(
        "{}",
        json!({"kind":"diagnostic_singleton_reference","routes":routes})
    );
    bits
}

fn make_mat(gpu: &mut Gpu, name: String, m: usize, k: usize, x: &GpuTensor, rng: &mut Rng) -> Mat {
    let bytes = pack_random_mq4g256v2(m, k, rng);
    let w = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("upload weights");
    let (reference, reference_receipts) = singleton_recorded(gpu, &w, x, m, k);
    Mat {
        name,
        m,
        k,
        w,
        reference,
        reference_receipts,
    }
}

/// Prefix/suffix guards AND every inactive capacity word must retain poison.
fn tally(got: &[u32], want: &[u32], m: usize) -> (usize, usize, usize) {
    assert!(got.len() >= want.len() + 2 * GUARD);
    let guard = got[..GUARD]
        .iter()
        .chain(&got[GUARD + want.len()..])
        .filter(|&&b| b != NAN_BITS)
        .count();
    let (mut even, mut odd) = (0, 0);
    for (i, (&g, &w)) in got[GUARD..GUARD + want.len()].iter().zip(want).enumerate() {
        if g != w {
            if (i % m) % 2 == 0 {
                even += 1;
            } else {
                odd += 1;
            }
        }
    }
    (even, odd, guard)
}

struct Launched {
    /// All four independently guarded allocations, including empty slots.
    bits: Vec<Vec<u32>>,
    route: Value,
    symbol: String,
    refusal: Option<String>,
}

fn present<'a>(slots: &Slots<'a>) -> Vec<&'a Mat> {
    slots.iter().flatten().copied().collect()
}
/// Copy only live operands. Inactive rows (including the next physical tile)
/// are poison, so an accidental overread cannot consume plausible finite data.
fn poisoned_x_host(source: &[f32], k: usize, rows: usize, subview_rows: usize) -> Vec<f32> {
    let live = rows * k;
    assert!(source.len() >= live, "short activation source");
    assert!(
        source[..live].iter().all(|v| v.is_finite()),
        "nonfinite live activation"
    );
    let prefix = GUARD + subview_rows * k;
    let mut out = vec![f32::from_bits(NAN_BITS); prefix + (MAX_ROWS + 8) * k + GUARD];
    out[prefix..prefix + live].copy_from_slice(&source[..live]);
    out
}

fn launch(
    gpu: &mut Gpu,
    slots: &Slots,
    x: &GpuTensor,
    rows: usize,
    graph: bool,
    diagnostic_product: bool,
    x_subview_rows: usize,
) -> Launched {
    assert!((1..=MAX_ROWS).contains(&rows));
    assert!(
        !graph || diagnostic_product,
        "graphs cannot qualify a direct tile"
    );
    let k = slots
        .iter()
        .flatten()
        .next()
        .map_or(x.numel() / MAX_ROWS, |m| m.k);
    assert!(k > 0 && k % 256 == 0);
    assert!(slots.iter().flatten().all(|m| m.k == k));
    let xb = snapshot(gpu, x);
    let source = gpu
        .download_f32(x)
        .expect("download live activation operands");
    let poison_host = poisoned_x_host(&source, k, rows, x_subview_rows);
    let poisoned = gpu
        .upload_f32(&poison_host, &[poison_host.len()])
        .expect("upload poisoned x backing");
    let candidate_x = poisoned.sub_offset(GUARD + x_subview_rows * k, rows * k);
    let poisoned_before = snapshot(gpu, &poisoned);
    let wb: Vec<Option<Vec<u8>>> = slots
        .iter()
        .map(|m| m.map(|m| snapshot(gpu, &m.w)))
        .collect();
    let dummy = gpu
        .upload_raw(&[0u8; 136], &[136])
        .expect("empty weight allocation");
    let dummy_before = snapshot(gpu, &dummy);
    let outs: Vec<GpuTensor> = slots
        .iter()
        .map(|slot| {
            let cap = slot.map_or(1, |m| MAX_ROWS * m.m);
            let n = 2 * GUARD + cap;
            gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n])
                .expect("alloc poisoned y")
        })
        .collect();
    let views: Vec<GpuTensor> = slots
        .iter()
        .zip(&outs)
        .map(|(slot, y)| y.sub_offset(GUARD, slot.map_or(0, |m| rows * m.m)))
        .collect();
    let parts: [(&GpuTensor, &GpuTensor, usize); 4] = std::array::from_fn(|s| match slots[s] {
        Some(m) => (&m.w, &views[s], m.m),
        None => (&dummy, &views[s], 0),
    });
    if slots.iter().all(Option::is_none) {
        let previous_symbol = gpu.last_launched_kernel().map(str::to_owned);
        let refusal = gpu
            .lab_gemv_mq4g256v2_rows_x4_t8(parts, &candidate_x, k, rows)
            .expect_err("all-empty invalid grid unexpectedly accepted");
        assert_eq!(
            gpu.last_launched_kernel(),
            previous_symbol.as_deref(),
            "invalid-grid refusal changed launched-route marker"
        );
        let bits = outs.iter().map(|y| fetch(gpu, y)).collect();
        unchanged(gpu, x, &xb, "all-empty x");
        unchanged(
            gpu,
            &poisoned,
            &poisoned_before,
            "all-empty poisoned activation backing",
        );
        unchanged(gpu, &dummy, &dummy_before, "all-empty weights");
        for y in outs {
            gpu.free_tensor(y).expect("free empty output");
        }
        gpu.free_tensor(dummy).expect("free empty weights");
        gpu.free_tensor(poisoned)
            .expect("free all-empty poisoned x");
        return Launched {
            bits,
            route: Value::Null,
            symbol: String::new(),
            refusal: Some(refusal.to_string()),
        };
    }
    let symbol;
    if graph {
        gpu.wide_verify_rows = 0;
        gpu.ensure_capture_stream().expect("capture stream");
        gpu.hip.device_synchronize().expect("sync");
        gpu.begin_stream_capture().expect("begin capture");
        let captured = gpu.gemv_mq4g256v2_rows_x4(parts, &candidate_x, k, rows);
        symbol = gpu.last_launched_kernel().map(str::to_owned);
        let g = gpu.end_stream_capture().expect("end capture");
        captured.expect("captured product launch");
        let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
        for y in &outs {
            fill_nan(gpu, y);
        }
        gpu.launch_graph(&exec).expect("graph replay");
        gpu.hip.device_synchronize().expect("sync");
        gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
        gpu.hip.graph_destroy(g).expect("destroy graph");
    } else {
        if diagnostic_product {
            gpu.wide_verify_rows = rows;
            let result = gpu.gemv_mq4g256v2_rows_x4(parts, &candidate_x, k, rows);
            gpu.wide_verify_rows = 0;
            result.expect("diagnostic product rows launch");
        } else {
            gpu.lab_gemv_mq4g256v2_rows_x4_t8(parts, &candidate_x, k, rows)
                .expect("required direct tile unsupported/missing");
        }
        symbol = gpu.last_launched_kernel().map(str::to_owned);
    }
    let symbol = symbol.expect("candidate missing actual launched route");
    let route = identity(gpu, &symbol);
    let bits = outs.iter().map(|y| fetch(gpu, y)).collect();
    unchanged(gpu, x, &xb, "tile x (whole backing)");
    unchanged(
        gpu,
        &poisoned,
        &poisoned_before,
        "poisoned live/tail/prefix activation backing",
    );
    for (m, before) in slots.iter().zip(&wb) {
        if let (Some(m), Some(before)) = (m, before) {
            unchanged(gpu, &m.w, before, &m.name);
        }
    }
    unchanged(gpu, &dummy, &dummy_before, "empty weights");
    for y in outs {
        gpu.free_tensor(y).expect("free y");
    }
    gpu.free_tensor(dummy).expect("free empty weights");
    gpu.free_tensor(poisoned).expect("free poisoned x backing");
    Launched {
        bits,
        route,
        symbol,
        refusal: None,
    }
}

fn expected_rows(slots: &Slots, ids: &[usize]) -> Vec<Vec<u32>> {
    slots
        .iter()
        .map(|slot| match slot {
            Some(m) => ids
                .iter()
                .flat_map(|&r| m.reference[r * m.m..(r + 1) * m.m].iter().copied())
                .collect(),
            None => Vec::new(),
        })
        .collect()
}

#[derive(Default)]
struct Tally {
    runs: usize,
    even: usize,
    odd: usize,
    guard: usize,
    symbol: usize,
}

struct Ctx {
    arch: String,
    diagnostic_product: bool,
    synthetic_only: bool,
    seed: u64,
    cases: usize,
    captured: usize,
    reached: BTreeSet<usize>,
    tallies: BTreeMap<String, Tally>,
}

impl Ctx {
    fn bad_total(&self) -> usize {
        self.tallies
            .values()
            .map(|t| t.even + t.odd + t.guard + t.symbol)
            .sum()
    }

    fn check(
        &mut self,
        gpu: &mut Gpu,
        label: &str,
        tag: &str,
        slots: &Slots,
        x: &GpuTensor,
        ids: &[usize],
        graph: bool,
    ) {
        let rows = ids.len();
        let empty_refusal = slots.iter().all(Option::is_none);
        let k = slots
            .iter()
            .flatten()
            .next()
            .map_or(x.numel() / MAX_ROWS, |m| m.k);
        let mask = slots.iter().enumerate().fold(0usize, |mask, (s, m)| {
            mask | (usize::from(m.is_some()) << s)
        });
        let x_subview_rows = if tag == "subview+3" { 3 } else { 0 };
        let launched = launch(
            gpu,
            slots,
            x,
            rows,
            graph,
            self.diagnostic_product,
            x_subview_rows,
        );
        let activation_backing = json!({"live_rows":rows,"capacity_rows":MAX_ROWS+8,
            "poison_tail_rows":MAX_ROWS+8-rows,"guard_words_each":GUARD,
            "subview_offset_rows":x_subview_rows,"prefix_words":GUARD+x_subview_rows*k,
            "original_and_candidate_backing_unchanged":true});
        let want = expected_rows(slots, ids);
        let (mut even, mut odd, mut guard) = (0, 0, 0);
        let mut first = Value::Null;
        let mut matrices = Vec::new();
        for s in 0..4 {
            let m = slots[s].map_or(0, |m| m.m);
            let got = &launched.bits[s];
            let (e, o, g) = tally(got, &want[s], m);
            even += e;
            odd += o;
            guard += g;
            for (i, &actual) in got.iter().enumerate() {
                let active = i >= GUARD && i < GUARD + want[s].len();
                let expected = if active { want[s][i - GUARD] } else { NAN_BITS };
                if actual != expected && first.is_null() {
                    let logical = i.saturating_sub(GUARD);
                    first = json!({"slot":s,"allocation_word":i,
                        "region":if active {"live"} else {"guard_or_inactive"},
                        "row":if active {Some(logical / m)} else {None},
                        "column":if active {Some(logical % m)} else {None},
                        "parity":if active {Some((logical % m) % 2)} else {None},
                        "actual_bits":format!("0x{actual:08x}"),
                        "expected_bits":format!("0x{expected:08x}")});
                }
            }
            matrices.push(match slots[s] {
                Some(mat) => json!({"slot":s,"name":mat.name,"M":m,"K":mat.k,
                    "live_words":rows*m,"capacity_words":MAX_ROWS*m,
                    "guard_words_each":GUARD,
                    "references":ids.iter().map(|&r| &mat.reference_receipts[r]).collect::<Vec<_>>()}),
                None => json!({"slot":s,"M":0,"live_words":0,"capacity_words":1,"guard_words_each":GUARD}),
            });
        }
        let expected = format!("gemv_mq4g256v2_rows_x4_t8_pm_{}", self.arch);
        let symbol_bad =
            usize::from(!empty_refusal && !self.diagnostic_product && launched.symbol != expected);
        let bad = even + odd + guard + symbol_bad;
        println!(
            "{}",
            json!({"kind":"cell","arch":self.arch,"seed":self.seed,
            "label":label,"tag":tag,"mode":if empty_refusal {"invalid_grid_refusal_control"} else if graph {"diagnostic_product_graph_last_route"} else if self.diagnostic_product {"diagnostic_product"} else {"direct_tile"},
            "expected_refusal":empty_refusal,"qualifies_component_cell":!empty_refusal && !self.diagnostic_product && !self.synthetic_only,
            "source_kind":if self.seed == 0 {"captured"} else {"seeded_synthetic"},
            "K":k,"live_slot_mask":mask,"refusal":launched.refusal,
            "symbol":launched.route["symbol"],"module":launched.route["module"],
            "object_sha256":launched.route["object_sha256"],
            "container_sha256":launched.route["container_sha256"],
            "rows":rows,"candidate":launched.route,"expected_symbol":if self.diagnostic_product || empty_refusal {None} else {Some(expected)},
            "activation_backing":activation_backing,
            "matrices":matrices,"first_mismatch":first,"even_mismatch_words":even,
            "odd_mismatch_words":odd,"guard_or_inactive_words":guard,"symbol_errors":symbol_bad,
            "raw_bytes_equal":bad == 0})
        );
        let entry = self.tallies.entry(label.to_string()).or_default();
        entry.runs += 1;
        entry.even += even;
        entry.odd += odd;
        entry.guard += guard;
        entry.symbol += symbol_bad;
        self.cases += 1;
        assert_eq!(
            bad, 0,
            "strict raw-byte/route mismatch: {label} rows={rows} {tag}"
        );
        if !graph && !empty_refusal {
            self.reached.insert(rows);
        }
    }

    fn run(&mut self, gpu: &mut Gpu, label: &str, slots: &Slots, x: &GpuTensor, views: bool) {
        let empty_refusal = slots.iter().all(Option::is_none);
        let mut reached = BTreeSet::new();
        for rows in 1..=MAX_ROWS {
            let ids: Vec<usize> = (0..rows).collect();
            self.check(gpu, label, "", slots, x, &ids, false);
            if !empty_refusal {
                reached.insert(rows);
            }
            if self.diagnostic_product && !empty_refusal && GRAPH_ROWS.contains(&rows) {
                self.check(gpu, label, "", slots, x, &ids, true);
            }
        }
        assert_eq!(
            reached.len(),
            if empty_refusal { 0 } else { MAX_ROWS },
            "unreached width in cell {label}"
        );
        if !views || slots.iter().all(Option::is_none) {
            return;
        }
        let k = present(slots)[0].k;
        let host = gpu.download_f32(x).expect("download x");
        for rows in [2usize, 9, 15, 16, 17, 31, 32, 33, 63, 64] {
            let mut permuted = host.clone();
            for i in 0..rows {
                permuted[i * k..(i + 1) * k]
                    .copy_from_slice(&host[(rows - 1 - i) * k..(rows - i) * k]);
            }
            let xp = gpu
                .upload_f32(&permuted, &[permuted.len()])
                .expect("upload permuted x");
            let ids: Vec<usize> = (0..rows).rev().collect();
            self.check(gpu, label, "permuted", slots, &xp, &ids, false);
            gpu.free_tensor(xp).expect("free permuted x");
            // Padding keeps offset-three probes legal even at R=63/64.
            let mut padded = vec![f32::from_bits(NAN_BITS); 3 * k];
            padded.extend_from_slice(&host);
            let backing = gpu
                .upload_f32(&padded, &[padded.len()])
                .expect("upload x subview backing");
            let xv = backing.sub_offset(3 * k, MAX_ROWS * k);
            let ids: Vec<usize> = (0..rows).collect();
            self.check(gpu, label, "subview+3", slots, &xv, &ids, false);
            gpu.free_tensor(backing).expect("free x subview backing");
        }
    }
}

/// Graph replay repeated after the activation is overwritten at the same
/// address: capture once, replay with x, overwrite x in place, replay again.
fn graph_changed_x(
    gpu: &mut Gpu,
    label: &str,
    slots: &Slots,
    rows: usize,
    seed: u64,
    ctx: &mut Ctx,
) {
    assert!(
        ctx.diagnostic_product,
        "changed-x graphs are diagnostic only"
    );
    let k = present(slots)[0].k;
    let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let first = host_x(k, &mut rng);
    let x = gpu
        .upload_f32(&first, &[first.len()])
        .expect("upload stable x");
    let initial_poison = poisoned_x_host(&first, k, rows, 0);
    let poisoned = gpu
        .upload_f32(&initial_poison, &[initial_poison.len()])
        .expect("graph poisoned x");
    let candidate_x = poisoned.sub_offset(GUARD, rows * k);
    let wb: Vec<Option<Vec<u8>>> = slots
        .iter()
        .map(|m| m.map(|m| snapshot(gpu, &m.w)))
        .collect();
    let dummy = gpu
        .upload_raw(&[0u8; 136], &[136])
        .expect("empty graph weights");
    let db = snapshot(gpu, &dummy);
    let outs: Vec<GpuTensor> = slots
        .iter()
        .map(|m| {
            let n = 2 * GUARD + m.map_or(1, |m| MAX_ROWS * m.m);
            gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n])
                .expect("graph y")
        })
        .collect();
    let views: Vec<GpuTensor> = slots
        .iter()
        .zip(&outs)
        .map(|(m, y)| y.sub_offset(GUARD, m.map_or(0, |m| rows * m.m)))
        .collect();
    let parts = std::array::from_fn(|s| match slots[s] {
        Some(m) => (&m.w, &views[s], m.m),
        None => (&dummy, &views[s], 0),
    });
    gpu.wide_verify_rows = 0;
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    let captured = gpu.gemv_mq4g256v2_rows_x4(parts, &candidate_x, k, rows);
    let symbol = gpu
        .last_launched_kernel()
        .expect("graph missing captured route")
        .to_owned();
    let g = gpu.end_stream_capture().expect("end capture");
    captured.expect("capture product launch");
    let route = identity(gpu, &symbol);
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    for round in 0..2 {
        if round == 1 {
            let second = host_x(k, &mut rng);
            gpu.hip
                .device_synchronize()
                .expect("sync before overwriting stable x");
            let bytes: Vec<u8> = second.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip
                .memcpy_htod(&x.buf, &bytes)
                .expect("overwrite stable x");
            let poison_host = poisoned_x_host(&second, k, rows, 0);
            let poison_bytes: Vec<u8> = poison_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip
                .memcpy_htod(&poisoned.buf, &poison_bytes)
                .expect("overwrite stable poisoned graph x");
        }
        let refs: Vec<Option<(Vec<u32>, Vec<Value>)>> = slots
            .iter()
            .map(|m| m.map(|m| singleton_recorded(gpu, &m.w, &x, m.m, m.k)))
            .collect();
        let xb = snapshot(gpu, &x);
        let poisoned_before = snapshot(gpu, &poisoned);
        for y in &outs {
            fill_nan(gpu, y);
        }
        gpu.launch_graph(&exec).expect("graph replay");
        let (mut even, mut odd, mut guard) = (0, 0, 0);
        for s in 0..4 {
            let got = fetch(gpu, &outs[s]);
            let m = slots[s].map_or(0, |m| m.m);
            let want = refs[s]
                .as_ref()
                .map_or(&[][..], |(bits, _)| &bits[..rows * m]);
            let (e, o, g) = tally(&got, want, m);
            even += e;
            odd += o;
            guard += g;
            if let (Some(mat), Some(before)) = (slots[s], &wb[s]) {
                unchanged(gpu, &mat.w, before, "changed-x graph weights");
            }
        }
        unchanged(gpu, &x, &xb, "changed-x graph x");
        unchanged(
            gpu,
            &poisoned,
            &poisoned_before,
            "changed-x poisoned graph activation backing",
        );
        unchanged(gpu, &dummy, &db, "changed-x graph empty weights");
        println!(
            "{}",
            json!({"kind":"diagnostic_changed_x","mode":"diagnostic_product_graph_last_route",
            "qualifies_component_cell":false,"arch":ctx.arch,"seed":seed,"label":label,
            "rows":rows,"round":round,"candidate":route,
            "activation_backing":{"live_rows":rows,"capacity_rows":MAX_ROWS+8,
                "poison_tail_rows":MAX_ROWS+8-rows,"guard_words_each":GUARD,
                "original_and_candidate_backing_unchanged":true},
            "references":refs.iter().map(|r|r.as_ref().map(|(_,routes)|&routes[..rows])).collect::<Vec<_>>(),
            "even_mismatch_words":even,"odd_mismatch_words":odd,"guard_or_inactive_words":guard})
        );
        let entry = ctx
            .tallies
            .entry(format!("{label} graph changed-x"))
            .or_default();
        entry.runs += 1;
        entry.even += even;
        entry.odd += odd;
        entry.guard += guard;
        ctx.cases += 1;
        assert_eq!(even + odd + guard, 0, "diagnostic changed-x byte mismatch");
    }
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
    for y in outs {
        gpu.free_tensor(y).expect("free graph y");
    }
    gpu.free_tensor(dummy).expect("free empty graph weights");
    gpu.free_tensor(x).expect("free stable graph x");
    gpu.free_tensor(poisoned)
        .expect("free stable poisoned graph x");
}

fn manifest_usize(meta: &Value, key: &str) -> usize {
    usize::try_from(
        meta[key]
            .as_u64()
            .unwrap_or_else(|| panic!("missing integer {key}")),
    )
    .expect("manifest dimension overflows usize")
}

fn capture_file(dir: &Path, entry: &Value) -> Vec<u8> {
    let relative = entry["path"].as_str().expect("capture file path");
    let path = Path::new(relative);
    assert!(
        !relative.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "capture path must be relative without traversal"
    );
    let root = dir.canonicalize().expect("capture directory");
    let file = dir.join(path).canonicalize().expect("capture file");
    assert!(file.starts_with(&root), "capture symlink escapes directory");
    let expected = entry["sha256"].as_str().expect("capture file SHA256");
    assert!(valid_sha(expected), "invalid capture SHA256");
    let bytes = std::fs::read(file).expect("read captured bytes");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        expected.to_ascii_lowercase(),
        "capture file SHA256 mismatch"
    );
    bytes
}

fn run_capture(gpu: &mut Gpu, ctx: &mut Ctx, dir: &Path) {
    let raw = std::fs::read(dir.join("meta.json")).expect("capture meta.json");
    let meta: Value = serde_json::from_slice(&raw).expect("parse capture metadata");
    assert_eq!(
        meta["schema"].as_str(),
        Some("mq4_rows_oracle.capture.v1"),
        "capture schema"
    );
    assert_eq!(
        meta["arch"].as_str(),
        Some(ctx.arch.as_str()),
        "capture arch mismatch"
    );
    let (m, k, rows) = (
        manifest_usize(&meta, "M"),
        manifest_usize(&meta, "K"),
        manifest_usize(&meta, "rows"),
    );
    assert!(
        m > 0 && k > 0 && k % 256 == 0 && rows >= MAX_ROWS,
        "capture extents"
    );
    let provenance = &meta["provenance"];
    let source = provenance["source"]
        .as_str()
        .expect("capture source artifact path");
    assert!(!source.is_empty(), "capture source provenance missing");
    let source_entry = json!({"path":source,"sha256":provenance["source_sha256"]});
    drop(capture_file(dir, &source_entry));
    let weights = capture_file(dir, &meta["weights"]);
    let activations = capture_file(dir, &meta["activations"]);
    let weight_bytes = m
        .checked_mul(k / 256)
        .and_then(|n| n.checked_mul(136))
        .expect("weight extent overflow");
    let x_bytes = rows
        .checked_mul(k)
        .and_then(|n| n.checked_mul(4))
        .expect("activation extent overflow");
    assert_eq!(weights.len(), weight_bytes, "packed capture weight extents");
    assert_eq!(activations.len(), x_bytes, "capture activation extents");
    for group in weights.chunks_exact(136) {
        for h in group[..8].chunks_exact(2) {
            assert_ne!(
                u16::from_le_bytes([h[0], h[1]]) & 0x7c00,
                0x7c00,
                "nonfinite packed f16"
            );
        }
    }
    let all_x: Vec<f32> = activations
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("f32 bytes")))
        .collect();
    assert!(
        all_x.iter().all(|v| v.is_finite()),
        "nonfinite captured activation"
    );
    // Equal genuinely captured rows are legal; provenance, not uniqueness, is required.
    println!(
        "{}",
        json!({"kind":"capture_source","arch":ctx.arch,"directory":dir,
        "manifest_sha256":format!("{:x}",Sha256::digest(&raw)),"manifest":meta,
        "captured_rows":rows,"tested_rows":MAX_ROWS,"finite_only":true})
    );
    let w = gpu
        .upload_raw(&weights, &[weights.len()])
        .expect("upload captured weights");
    let x = gpu
        .upload_f32(&all_x[..MAX_ROWS * k], &[MAX_ROWS * k])
        .expect("upload captured x");
    let (reference, reference_receipts) = singleton_recorded(gpu, &w, &x, m, k);
    let mat = Mat {
        name: format!("captured {}", dir.display()),
        m,
        k,
        w,
        reference,
        reference_receipts,
    };
    ctx.seed = 0;
    ctx.run(gpu, &mat.name, &[Some(&mat), None, None, None], &x, true);
    ctx.captured += 1;
    gpu.free_tensor(mat.w).expect("free captured weights");
    gpu.free_tensor(x).expect("free captured x");
}

fn main() {
    let mut diagnostic_product = false;
    let mut synthetic_only = false;
    let mut captures: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str().expect("CLI must be UTF-8") {
            "--diagnostic-product" => diagnostic_product = true,
            "--synthetic-only" => synthetic_only = true,
            "--capture" => {
                captures.push(PathBuf::from(args.next().expect("--capture requires DIR")))
            }
            other => panic!("unknown option: {other}"),
        }
    }
    assert!(
        synthetic_only || diagnostic_product || !captures.is_empty(),
        "qualification requires --capture DIR; use --synthetic-only for nonqualifying diagnostics"
    );
    assert!(
        !synthetic_only || captures.is_empty(),
        "--synthetic-only contradicts --capture"
    );
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    assert!(
        matches!(arch.as_str(), "gfx1151" | "gfx1201" | "gfx1100"),
        "unsupported architecture"
    );
    let env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .collect();
    println!(
        "{}",
        json!({"kind":"header","arch":arch,"environment":env,
        "mode":if diagnostic_product {"diagnostic_product"} else {"direct_tile"},
        "qualification_candidate":!diagnostic_product && !synthetic_only,
        "finite_only":true,"max_rows":MAX_ROWS,
        "shape_grid":"declared projection dimensions, not a captured model census",
        "no_cpu_or_performance_or_retained_certificate_claim":true})
    );
    let mut ctx = Ctx {
        arch: arch.clone(),
        diagnostic_product,
        synthetic_only,
        seed: 0,
        cases: 0,
        captured: 0,
        reached: BTreeSet::new(),
        tallies: BTreeMap::new(),
    };
    for dir in &captures {
        run_capture(&mut gpu, &mut ctx, dir);
    }

    for seed in 1..=3u64 {
        ctx.seed = seed;
        let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
        let mut xs: BTreeMap<usize, GpuTensor> = BTreeMap::new();
        for k in [256usize, 512, 768, 1024, 1280, 2560, 6144] {
            let x = upload_x(&mut gpu, k, &mut rng);
            xs.insert(k, x);
        }

        // (a) Declared projection dimensions; (b) their declared x4 groupings.
        let mut trunk: Vec<Mat> = Vec::new();
        for &(name, m, k) in &SHAPES {
            trunk.push(make_mat(
                &mut gpu,
                name.to_string(),
                m,
                k,
                &xs[&k],
                &mut rng,
            ));
        }
        for (i, mat) in trunk.iter().enumerate() {
            let label = format!("declared projection {} M={} K={}", mat.name, mat.m, mat.k);
            ctx.run(
                &mut gpu,
                &label,
                &[Some(mat), None, None, None],
                &xs[&mat.k],
                i == 0,
            );
        }
        for &(name, idx) in &GROUPS {
            let mut slots: Slots = [None; 4];
            for (s, &i) in idx.iter().enumerate() {
                slots[s] = Some(&trunk[i]);
            }
            let k = trunk[idx[0]].k;
            ctx.run(&mut gpu, name, &slots, &xs[&k], true);
            if diagnostic_product && seed == 1 {
                graph_changed_x(&mut gpu, name, &slots, 5, seed, &mut ctx);
                graph_changed_x(&mut gpu, name, &slots, 24, seed, &mut ctx);
            }
        }

        // Negative controls: wrong activation, swapped matrices, and parity.
        // These must disagree, including the new 33/63/64 boundaries.
        if seed == 1 {
            let slots: Slots = [
                Some(&trunk[0]),
                Some(&trunk[1]),
                Some(&trunk[2]),
                Some(&trunk[3]),
            ];
            for rows in [3usize, 20, 33, 63, 64] {
                let launched = launch(
                    &mut gpu,
                    &slots,
                    &xs[&2560],
                    rows,
                    false,
                    diagnostic_product,
                    0,
                );
                let got = &launched.bits;
                let mut host = gpu.download_f32(&xs[&2560]).expect("download x");
                host[0] += 1.0;
                let x_perturbed = gpu.upload_f32(&host, &[host.len()]).expect("upload x'");
                let perturbed_bad: Vec<usize> = slots
                    .iter()
                    .flatten()
                    .zip(got)
                    .map(|(mat, g)| {
                        let want =
                            &singleton_reference(&mut gpu, &mat.w, &x_perturbed, mat.m, mat.k)
                                [..rows * mat.m];
                        let (e, o, gd) = tally(g, want, mat.m);
                        e + o + gd
                    })
                    .collect();
                gpu.free_tensor(x_perturbed).expect("free x'");
                let (a, b) = (trunk[2].m, trunk[3].m);
                assert_eq!(
                    (trunk[2].name.as_str(), trunk[3].name.as_str(), a, b),
                    ("gdn.a", "gdn.b", 48, 48)
                );
                let (e, o, gd) = tally(&got[2], &trunk[3].reference[..rows * b], a);
                let swapped_bad = e + o + gd;
                // Even/odd column swap: column 2j of the launch against reference column 2j+1.
                let (mut parity_agree, mut parity_total) = (0usize, 0usize);
                let ga = &got[2][GUARD..GUARD + rows * a];
                for r in 0..rows {
                    for c in (0..a).step_by(2) {
                        parity_total += 1;
                        parity_agree +=
                            usize::from(ga[r * a + c] == trunk[2].reference[r * a + c + 1]);
                    }
                }
                println!(
                    "{}",
                    json!({"kind":"negative_control","arch":arch,"seed":seed,
                    "mode":if diagnostic_product {"diagnostic_product"} else {"direct_tile"},
                    "rows":rows,"candidate":launched.route,
                    "perturbed_x_mismatch_words":perturbed_bad,"swapped_mismatch_words":swapped_bad,
                    "parity_swapped_agree":parity_agree,"parity_total":parity_total})
                );
                if perturbed_bad.iter().any(|&n| n == 0)
                    || swapped_bad == 0
                    || parity_agree * 2 > parity_total
                {
                    eprintln!("negative control FAILED: a wrong reference agreed with the launch (vacuous oracle)");
                    std::process::exit(1);
                }
            }
        }
        for mat in trunk {
            gpu.free_tensor(mat.w).expect("free weights");
        }

        // (c) synthetic single matrices: odd / block-straddling M x every K tail.
        let mut syn = 0u64;
        for &k in &SYN_K {
            for &m in &SYN_M {
                syn += 1;
                let mat = make_mat(
                    &mut gpu,
                    format!("syn M={m} K={k}"),
                    m,
                    k,
                    &xs[&k],
                    &mut rng,
                );
                let label = format!("synthetic single M={m} K={k}");
                // Every R=1..=64; graph samples only in explicit diagnostic mode.
                ctx.run(
                    &mut gpu,
                    &label,
                    &[Some(&mat), None, None, None],
                    &xs[&k],
                    syn % 5 == 0,
                );
                gpu.free_tensor(mat.w).expect("free weights");
            }
        }

        // (d) synthetic x4 groupings: odd boundaries, zero-M parts in any slot.
        for &k in &SYN_GROUP_K {
            let mut groups = SYN_GROUPS.to_vec();
            groups.extend((0..16usize).map(|mask| {
                std::array::from_fn(|s| {
                    if mask & (1 << s) != 0 {
                        [1, 7, 33, 64][s]
                    } else {
                        0
                    }
                })
            }));
            for ms in &groups {
                let mats: Vec<Option<Mat>> = ms
                    .iter()
                    .enumerate()
                    .map(|(s, &m)| {
                        (m > 0).then(|| {
                            make_mat(
                                &mut gpu,
                                format!("slot{s} M={m} K={k}"),
                                m,
                                k,
                                &xs[&k],
                                &mut rng,
                            )
                        })
                    })
                    .collect();
                let slots: Slots = [
                    mats[0].as_ref(),
                    mats[1].as_ref(),
                    mats[2].as_ref(),
                    mats[3].as_ref(),
                ];
                let label = format!("synthetic x4 M={ms:?} K={k}");
                ctx.run(&mut gpu, &label, &slots, &xs[&k], true);
                if diagnostic_product && seed == 1 && k == 1280 && ms.iter().any(|&m| m > 0) {
                    graph_changed_x(&mut gpu, &label, &slots, 17, seed, &mut ctx);
                }
                for mat in mats.into_iter().flatten() {
                    gpu.free_tensor(mat.w).expect("free weights");
                }
            }
        }

        for (_, x) in xs {
            gpu.free_tensor(x).expect("free x");
        }
    }

    assert_eq!(ctx.bad_total(), 0, "strict mismatches");
    assert_eq!(
        ctx.reached,
        (1..=MAX_ROWS).collect(),
        "unreached direct/product widths"
    );
    let qualifies = !diagnostic_product && !synthetic_only;
    assert!(
        !qualifies || ctx.captured > 0,
        "missing real captured operands"
    );
    println!(
        "{}",
        json!({"kind":"summary","arch":arch,
        "result":if qualifies {"PASS"} else {"DIAGNOSTIC_COMPLETE"},
        "qualifies_G1_component":qualifies,"cases":ctx.cases,"captured_sources":ctx.captured,
        "seeds":[1,2,3],"reached_rows":ctx.reached,"mismatch_bytes":0,
        "all_empty_mask":"expected invalid-grid refusal, no claimed launched route",
        "finite_only":true,"no_cpu_or_performance_or_retained_certificate_claim":true})
    );
}
