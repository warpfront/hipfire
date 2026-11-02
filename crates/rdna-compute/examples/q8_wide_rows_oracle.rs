// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict gfx1151 Q8 fixed-eight-row PM byte gate, logical R=1..=64.
//!
//! Strict capture run (on Halo): `cargo run --release -p rdna-compute --example
//! q8_wide_rows_oracle -- --capture q8-capture.json --capture-sha256 <64-hex-pin>`.
//! `--seeded-only` is an exclusive diagnostic mode, never a captured/G1 PASS.
//! Missing, duplicate, conflicting and unknown CLI options are fatal.
//! Other architectures, missing kernels, image drift and missing identity receipts
//! are REFUSAL (nonzero), never skipped/PASS. Direct embedded PM launches bypass
//! product row-cap policy; this is not a product-dispatch or throughput certificate.
//! References are actual `Gpu::gemv_q8_0` / `Gpu::gemv_q8_0_k8` launches.
//!
//! Capture schema `q8_rowtile_capture.v1` has `schema`, `arch: "gfx1151"`,
//! `source_index: {path, sha256}`, and nonempty `cases`. Each case requires
//! `{name, kind: "single"|"x2"|"k8", k, ms: [M...], scale: null|f32,
//! weights: [{path, sha256}], x: {path, sha256},
//! singleton: {module, symbol, object_sha256, container_sha256?}}`. Object SHA is
//! the extracted per-architecture device ELF; container SHA is separate.
//! Weights are exact raw Q8_0 `M*(K/32)*34` bytes; X is `64*K` little-endian
//! finite F32 values. Manifest, source index and every payload are SHA-pinned
//! before any GPU upload. Names are unique, scales finite (null unless k8).
//! The historical `singleton` field identifies the actual captured caller,
//! including recognized Q8-bearing pair/mixed callers, not an invented primitive.
//! Its original C1 actually executes before auxiliary primitive comparisons.
//! Mixed BF16 operands come from the adjacent hash-pinned source-index subgroup;
//! both subgroups' complete guarded outputs and immutable inputs are checked.
//!
//! Every seeded/captured case exercises every R=1..64, reversed rows, guards,
//! immutable operands and actual singleton byte comparisons. The initial capture
//! baseline uses only captured weight/X bytes. Alternate weights and changed-X
//! negative controls are synthetic; stable-address replay uses generated changed-X
//! with the captured weights. Receipts distinguish these claim boundaries.
//! Inactive rows alone contain NaN poison. Seeded live operands are deterministic
//! finite signed-zero/subnormal/exponent stress across multiple seeds; HC includes
//! raw/activated outputs and each isolated K partition. Frozen seeded accounting
//! remains separate from capture counts. No universal NaN/Inf, CPU-emulation,
//! state, graph-replay or end-to-end model exactness claim is made.
//! PASS requires both the entire seeded matrix and every supplied captured case.

use hip_bridge::KernargBlob;
use rdna_compute::kernels::{
    GEMV_Q8_0_WIDE_SRC, QWEN4_GEMV_Q8_0_SRC, QWEN4_GEMV_Q8_0_WIDE_SRC, QWEN4_LANE_ROWS_PM_GFX1151,
    QWEN4_LANE_ROWS_PM_GFX1151_SHA256,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 64;
const GUARD: usize = 64;
const SENTINEL: u8 = 0xa5;
const OUTPUT_POISON: u32 = 0x7fc1_2345;
const INPUT_POISON: u32 = 0x7fc5_4321;
const AMPS: [f32; 8] = [
    1.0,
    1.0 / 1024.0,
    1.0 / 16.0,
    4.0,
    64.0,
    0.5,
    0.0,
    1.1529215e18,
];
const PM_MODULE: &str = "qwen4_lane_rows_pm";
const Q8_EMITTER_SRC: &str = include_str!("../../peacemaker-isa/src/kernels/qwen4_lane_rows_q8.rs");
const PM_BUNDLE_SRC: &str = include_str!("../../peacemaker-isa/src/kernels/qwen4_lane_rows.rs");

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for c in &mut chunks {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let tail = self.next().to_le_bytes();
        rest.copy_from_slice(&tail[..rest.len()]);
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// A finite f16: mostly `2^-lo..2^hi`, sometimes `+-0` or subnormal.
    fn f16(&mut self, lo_exp: u16, hi_exp: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => {
                sign | ((15 - lo_exp + self.below(u64::from(lo_exp + hi_exp + 1)) as u16) << 10)
                    | self.below(0x400) as u16
            }
        }
    }
}

/// Finite f16 scales and all signed int8 values, including explicit scale extremes.
fn gen_q8(m: usize, k: usize, wide_scales: bool, rng: &mut Rng) -> Vec<u8> {
    let (lo, hi) = if wide_scales { (14, 15) } else { (8, 0) };
    let mut bytes = vec![0u8; m * (k / 32) * 34];
    for (i, blk) in bytes.chunks_exact_mut(34).enumerate() {
        rng.fill(&mut blk[2..]);
        let scale = match i % 31 {
            0 => 0,
            1 => 0x8000,
            2 => 1,
            3 => 0x8001,
            4 if wide_scales => 0x7bff,
            5 if wide_scales => 0xfbff,
            _ => rng.f16(lo, hi),
        };
        // Keep M=1,K=32 controls sensitive even when the only scale is signed zero.
        let scale = if m * (k / 32) < 8 {
            rng.f16(8, 0) | 0x0400
        } else {
            scale
        };
        blk[..2].copy_from_slice(&scale.to_le_bytes());
    }
    bytes
}

/// `[MAXR][k]` activations, row r at magnitude `AMPS[r % 8]`, with +-0 and F32
/// subnormals; `keep` (if given) zeroes everything outside `[lo, hi)` of each row.
fn gen_x(k: usize, keep: Option<(usize, usize)>, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % 8];
        for i in 0..k {
            let v = match rng.below(64) {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                0 => 0.0,
                1 => -0.0,
                2..=5 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => rng.unit() * amp,
            };
            x.push(match keep {
                Some((lo, hi)) if i < lo || i >= hi => 0.0,
                _ => v,
            });
        }
    }
    x
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// One output allocation: `GUARD | MAXR * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    active: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + MAXR * m], DType::F32)
            .map_err(err)?;
        let active = full.sub_offset(GUARD, MAXR * m);
        Ok(Self { full, active, m })
    }
    fn view(&self) -> &GpuTensor {
        &self.active
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAXR * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu, rows: usize) -> Result<()> {
        let mut bytes = vec![SENTINEL; self.bytes()];
        for word in bytes[GUARD * 4..(GUARD + rows * self.m) * 4].chunks_exact_mut(4) {
            word.copy_from_slice(&OUTPUT_POISON.to_le_bytes());
        }
        gpu.hip.memcpy_htod(&self.full.buf, &bytes).map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.bytes());
        Ok(v)
    }
    /// Sentinel everywhere except `rows` rows, each `f(r)`'s row of `reference`.
    fn expected(&self, reference: &[u8], rows: usize, f: impl Fn(usize) -> usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let row = self.m * 4;
        for r in 0..rows {
            let src = GUARD * 4 + f(r) * row;
            let dst = GUARD * 4 + r * row;
            e[dst..dst + row].copy_from_slice(&reference[src..src + row]);
        }
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXR * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (row, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("row {row} col {} byte {}", rem / 4, rem % 4)
        }
    }
    fn integrity(&self, bytes: &[u8], rows: usize, label: &str) -> Result<()> {
        let start = GUARD * 4;
        let end = start + rows * self.m * 4;
        for (i, &v) in bytes.iter().enumerate() {
            if (i < start || i >= end) && v != SENTINEL {
                return Err(format!(
                    "{label}: output guard/inactive corruption at {}",
                    self.locate(i)
                ));
            }
        }
        for (i, word) in bytes[start..end].chunks_exact(4).enumerate() {
            if word == OUTPUT_POISON.to_le_bytes() {
                return Err(format!(
                    "{label}: active output left poisoned at {}",
                    self.locate(start + i * 4)
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Single,
    X2,
    Hc(Option<f32>),
}

/// Primary and alternate weights retain their exact original upload bytes.
struct Weight {
    primary: GpuTensor,
    alternate: GpuTensor,
    bytes: Vec<u8>,
    alternate_bytes: Vec<u8>,
}

/// BF16 operands of the same captured mixed C1 invocation.
struct MixedOperands {
    weights: [GpuTensor; 2],
    bytes: [Vec<u8>; 2],
    outs: [Out; 2],
}

struct Case {
    name: String,
    k: usize,
    kind: Kind,
    weights: Vec<Weight>,
    outs: Vec<Out>,
    x: GpuTensor,
    x_alt: GpuTensor,
    host_x: Vec<f32>,
    keep: Option<(usize, usize)>,
    capture: Option<CaptureCaller>,
    mixed: Option<MixedOperands>,
}

#[derive(Default)]
struct Tally {
    cases: usize,
    candidates: usize,
    singleton_rows: usize,
    controls: usize,
    caller_rows: usize,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha(pin: &str) -> bool {
    pin.len() == 64 && pin.bytes().all(|b| b.is_ascii_hexdigit())
}

struct Cli {
    capture: Option<(PathBuf, String)>,
}

impl Cli {
    fn parse() -> Result<Self> {
        let mut args = std::env::args_os().skip(1);
        let (mut capture, mut pin, mut seeded_only) = (None, None, false);
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--capture") if capture.is_none() => {
                    let value = args.next().ok_or("--capture requires a manifest path")?;
                    if value.is_empty() || value.to_string_lossy().starts_with("--") {
                        return Err("--capture requires a manifest path".into());
                    }
                    capture = Some(PathBuf::from(value));
                }
                Some("--capture-sha256") if pin.is_none() => {
                    let value = args
                        .next()
                        .ok_or("--capture-sha256 requires a SHA256 pin")?;
                    let value = value.into_string().map_err(|_| "non-UTF8 SHA256 pin")?;
                    if !valid_sha(&value) {
                        return Err(
                            "--capture-sha256 requires exactly 64 hexadecimal digits".into()
                        );
                    }
                    pin = Some(value.to_ascii_lowercase());
                }
                Some("--seeded-only") if !seeded_only => seeded_only = true,
                _ => {
                    return Err(format!(
                        "unknown or duplicate option: {}",
                        arg.to_string_lossy()
                    ))
                }
            }
        }
        match (capture, pin, seeded_only) {
            (Some(path), Some(pin), false) => Ok(Self { capture: Some((path, pin)) }),
            (None, None, true) => Ok(Self { capture: None }),
            _ => Err("require --capture <manifest.json> --capture-sha256 <hex>, or exclusive --seeded-only diagnostic mode".into()),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCaller {
    module: String,
    symbol: String,
    object_sha256: String,
    container_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSpec {
    name: String,
    kind: String,
    k: usize,
    ms: Vec<usize>,
    // Value makes even an omitted nullable scale a schema error.
    scale: serde_json::Value,
    weights: Vec<CaptureFile>,
    x: CaptureFile,
    singleton: CaptureCaller,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureManifest {
    schema: String,
    arch: String,
    source_index: CaptureFile,
    cases: Vec<CaptureSpec>,
}

struct CapturedCase {
    spec: CaptureSpec,
    kind: Kind,
    weights: Vec<Vec<u8>>,
    host_x: Vec<f32>,
    mixed: Option<([usize; 2], [Vec<u8>; 2])>,
}

struct Capture {
    arch: String,
    cases: Vec<CapturedCase>,
}

fn pinned_file(base: &Path, file: &CaptureFile, expected_len: Option<usize>) -> Result<Vec<u8>> {
    if file.path.as_os_str().is_empty() || file.path.is_absolute() || !valid_sha(&file.sha256) {
        return Err(format!(
            "invalid manifest-relative payload path/pin: {}",
            file.path.display()
        ));
    }
    let path = base.join(&file.path);
    if let Some(len) = expected_len {
        let actual = std::fs::metadata(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        if actual != len as u64 {
            return Err(format!("{}: byte length {actual} != {len}", path.display()));
        }
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if expected_len.is_some_and(|len| bytes.len() != len)
        || !sha(&bytes).eq_ignore_ascii_case(&file.sha256)
    {
        return Err(format!("{}: payload length/hash drift", path.display()));
    }
    Ok(bytes)
}

fn device_extent(factors: &[usize], label: &str) -> Result<usize> {
    let n = factors
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or_else(|| format!("{label}: byte extent overflow"))?;
    if n > u32::MAX as usize {
        return Err(format!("{label}: byte extent {n} must stay below 2^32"));
    }
    Ok(n)
}

fn capture_file(value: &serde_json::Value) -> Result<CaptureFile> {
    serde_json::from_value(value.clone()).map_err(err)
}

fn capture_usize(value: &serde_json::Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| "capture needs nonnegative usize".into())
}

/// Resolve only hash-pinned existing index entries; names never stand in for
/// a source, operand, module, or device-object identity.
fn source_bundle(
    base: &Path,
    entry: &serde_json::Value,
    index: &serde_json::Value,
) -> Result<(serde_json::Value, PathBuf)> {
    let source_pin = capture_file(&entry["source"])?;
    let manifest_pin = capture_file(&entry["manifest"])?;
    let source: serde_json::Value =
        serde_json::from_slice(&pinned_file(base, &source_pin, None)?).map_err(err)?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&pinned_file(base, &manifest_pin, None)?).map_err(err)?;
    let source_path = base.join(&source_pin.path);
    let source_base = source_path
        .parent()
        .ok_or("source has no parent")?
        .to_path_buf();
    let manifest_path = base.join(&manifest_pin.path);
    let manifest_base = manifest_path.parent().ok_or("manifest has no parent")?;
    let provenance_path = manifest["provenance"]["source"]
        .as_str()
        .ok_or("missing source provenance path")?;
    if source["schema"] != "qwen4-rowtile-source-v1"
        || manifest["schema"] != "qwen4-rowtile-operands-v1"
        || manifest["arch"] != index["arch"]
        || manifest["provenance"]["source_sha256"]
            .as_str()
            .is_none_or(|sha| !sha.eq_ignore_ascii_case(&source_pin.sha256))
        || std::fs::canonicalize(manifest_base.join(provenance_path)).map_err(err)?
            != std::fs::canonicalize(&source_path).map_err(err)?
        || source["capture_mode"] != index["capture_mode"]
        || source["retained_identity_claim"] != false
        || source["model"] != index["model"]
        || source["prompt"] != index["prompt"]
        || source["model"]["sha256"]
            .as_str()
            .is_none_or(|pin| !valid_sha(pin))
        || source["prompt"]["sha256"]
            .as_str()
            .is_none_or(|pin| !valid_sha(pin))
        || source["rows"] != MAXR
    {
        return Err("source/manifest/index provenance mismatch".into());
    }
    for key in ["name", "family", "form", "ms", "k"] {
        if source[key] != entry[key] || source[key] != manifest[key] {
            return Err(format!("source/manifest/index {key} mismatch"));
        }
    }
    for key in ["rows", "weights", "x", "scale", "hc_act"] {
        if source[key] != manifest[key] {
            return Err(format!("source/manifest {key} mismatch"));
        }
    }
    let records = source["captures"]
        .as_array()
        .filter(|v| v.len() == MAXR)
        .ok_or("source must have 64 actual row records")?;
    let positions = index["positions"]
        .as_array()
        .filter(|v| v.len() == MAXR)
        .ok_or("index must have 64 positions")?;
    let tokens = index["tokens"]
        .as_array()
        .filter(|v| v.len() == MAXR)
        .ok_or("index must have 64 tokens")?;
    for (row, record) in records.iter().enumerate() {
        if record["row"] != row
            || record["position"] != positions[row]
            || record["token"] != tokens[row]
            || positions[row].as_u64().is_none()
            || tokens[row].as_u64().is_none()
            || record["occurrence"] != records[0]["occurrence"]
            || record["occurrence"].as_u64().is_none()
            || record["grid"].as_array().is_none_or(|v| {
                v.len() != 3
                    || v.iter()
                        .any(|n| n.as_u64().is_none_or(|n| n == 0 || n > u32::MAX as u64))
            })
            || record["grid"] != records[0]["grid"]
            || record["block"] != serde_json::json!([32, 1, 1])
            || record["shared_mem"] != 0
        {
            return Err(format!("invalid actual source row {row}"));
        }
    }
    println!("capture_source name={} source_path={:?} source_sha256={} manifest_path={:?} manifest_sha256={} model_sha256={} prompt_sha256={} retained_identity_claim=false",
        source["name"], source_pin.path, source_pin.sha256, manifest_pin.path,
        manifest_pin.sha256, source["model"]["sha256"], source["prompt"]["sha256"]);
    Ok((source, source_base))
}

fn same_payload(a_base: &Path, a: &CaptureFile, b_base: &Path, b: &CaptureFile) -> Result<()> {
    if !valid_sha(&a.sha256)
        || !a.sha256.eq_ignore_ascii_case(&b.sha256)
        || std::fs::canonicalize(a_base.join(&a.path)).map_err(err)?
            != std::fs::canonicalize(b_base.join(&b.path)).map_err(err)?
    {
        return Err("root/source payload binding mismatch".into());
    }
    Ok(())
}

fn captured_source(
    index_base: &Path,
    base: &Path,
    index: &serde_json::Value,
    spec: &CaptureSpec,
) -> Result<Option<([usize; 2], [Vec<u8>; 2])>> {
    let entries = index["cases"].as_array().ok_or("index cases missing")?;
    let mut matching = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry["family"] == "q8" && entry["name"] == spec.name);
    let (entry_index, entry) = matching
        .next()
        .ok_or_else(|| format!("{}: missing Q8 source entry", spec.name))?;
    if matching.next().is_some() {
        return Err(format!("{}: ambiguous Q8 source entry", spec.name));
    }
    let (source, source_base) = source_bundle(index_base, entry, index)?;
    let singleton: CaptureCaller =
        serde_json::from_value(source["singleton"].clone()).map_err(err)?;
    if singleton.symbol != spec.singleton.symbol
        || singleton.module != spec.singleton.module
        || !singleton
            .object_sha256
            .eq_ignore_ascii_case(&spec.singleton.object_sha256)
        || match (
            &singleton.container_sha256,
            &spec.singleton.container_sha256,
        ) {
            (Some(a), Some(b)) => !a.eq_ignore_ascii_case(b),
            (None, None) => false,
            _ => true,
        }
        || source["k"] != spec.k
        || source["scale"] != spec.scale
        || source["hc_act"] != false
        || source["form"]
            != match spec.kind.as_str() {
                "k8" => "hc_k8",
                other => other,
            }
    {
        return Err(format!(
            "{}: captured source identity/domain differs",
            spec.name
        ));
    }
    let ms = source["ms"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("source Ms missing")?;
    let weights = source["weights"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("source weight descriptors missing")?;
    let bindings = source["weight_bindings"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("source weight bindings missing")?;
    let mut matrix = 0;
    for slot in 0..4 {
        let m = capture_usize(&ms[slot])?;
        if m == 0 {
            continue;
        }
        if spec.ms.get(matrix) != Some(&m) {
            return Err("source Q8 Ms/order differs".into());
        }
        let file = capture_file(&weights[slot])?;
        same_payload(base, &spec.weights[matrix], &source_base, &file)?;
        if capture_usize(&bindings[slot]["bytes"])?
            != device_extent(&[m, spec.k / 32, 34], "Q8 binding")?
        {
            return Err("source Q8 weight extent mismatch".into());
        }
        matrix += 1;
    }
    if matrix != spec.ms.len() {
        return Err("source Q8 matrix count differs".into());
    }
    same_payload(base, &spec.x, &source_base, &capture_file(&source["x"])?)?;
    let mixed_symbol = "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair";
    if singleton.symbol != mixed_symbol {
        if !source["mixed"].is_null() || source["component_offset"] != 0 {
            return Err("ordinary source has mixed component metadata".into());
        }
        return Ok(None);
    }
    if source["component_offset"] != 2
        || spec.kind != "x2"
        || spec.ms.len() != 2
        || spec.ms[0] != spec.ms[1]
        || spec.k != 2560
    {
        return Err("mixed Q8 component domain mismatch".into());
    }
    let bf_entry = entry_index
        .checked_sub(1)
        .and_then(|i| entries.get(i))
        .ok_or("mixed Q8 lacks preceding BF16 subgroup")?;
    let (bf, bf_base) = source_bundle(index_base, bf_entry, index)?;
    if bf["family"] != "bf16"
        || bf["form"] != "x2"
        || bf["k"] != spec.k
        || bf["component_offset"] != 0
        || bf["singleton"] != source["singleton"]
        || bf["mixed"] != source["mixed"]
        || source["mixed"]["symbol"] != mixed_symbol
        || bf["captures"] != source["captures"]
        || bf["x"]["sha256"] != source["x"]["sha256"]
        || bf["hc_act"] != false
        || !bf["scale"].is_null()
    {
        return Err("mixed BF16/Q8 adjacency/invocation binding mismatch".into());
    }
    let bf_ms = bf["ms"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("BF16 Ms missing")?;
    let ms = [capture_usize(&bf_ms[0])?, capture_usize(&bf_ms[1])?];
    if ms.contains(&0)
        || ms.iter().any(|&m| m > i32::MAX as usize)
        || bf_ms[2] != 0
        || bf_ms[3] != 0
        || source["mixed"]["bf16_ms"] != serde_json::json!(ms)
    {
        return Err("mixed BF16 Ms mismatch".into());
    }
    let mut bytes = [Vec::new(), Vec::new()];
    for i in 0..2 {
        let file = capture_file(&bf["weights"][i])?;
        if source["mixed"]["bf16_weight_sha256"][i]
            .as_str()
            .is_none_or(|sha| !sha.eq_ignore_ascii_case(&file.sha256))
        {
            return Err("mixed BF16 weight pin mismatch".into());
        }
        let len = device_extent(&[ms[i], spec.k, 2], "mixed BF16 weight")?;
        if capture_usize(&bf["weight_bindings"][i]["bytes"])? != len {
            return Err("mixed BF16 binding extent mismatch".into());
        }
        bytes[i] = pinned_file(&bf_base, &file, Some(len))?;
        if bytes[i]
            .chunks_exact(2)
            .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
        {
            return Err("mixed BF16 weight is nonfinite".into());
        }
    }
    // Verify the paired activation payload too, not merely its claimed pin.
    let bf_x = capture_file(&bf["x"])?;
    let x_bytes = pinned_file(
        &bf_base,
        &bf_x,
        Some(device_extent(&[MAXR, spec.k, 4], "mixed X")?),
    )?;
    if !sha(&x_bytes).eq_ignore_ascii_case(&spec.x.sha256) {
        return Err("mixed BF16/Q8 actual X differs".into());
    }
    Ok(Some((ms, bytes)))
}

impl CaptureSpec {
    fn domain(&self) -> Result<Kind> {
        if self.name.trim().is_empty() || self.k == 0 || self.k > i32::MAX as usize {
            return Err("capture name/K outside supported domain".into());
        }
        let kind = match self.kind.as_str() {
            "single" if self.k == 2560 || (self.k <= 1536 && self.k % 32 == 0) => Kind::Single,
            "x2" if self.k == 2560 => Kind::X2,
            "k8" if self.k % 256 == 0 => {
                let scale = if self.scale.is_null() {
                    None
                } else {
                    let value = self
                        .scale
                        .as_f64()
                        .ok_or("k8 scale must be null or finite F32")?;
                    let value = value as f32;
                    if !value.is_finite() {
                        return Err("k8 scale must be finite F32".into());
                    }
                    Some(value)
                };
                Kind::Hc(scale)
            }
            _ => {
                return Err(format!(
                    "{}: unsupported kind/K {}/{}",
                    self.name, self.kind, self.k
                ))
            }
        };
        let count = if matches!(kind, Kind::X2) { 2 } else { 1 };
        if self.ms.len() != count
            || self.weights.len() != count
            || (!matches!(kind, Kind::Hc(_)) && !self.scale.is_null())
        {
            return Err(format!("{}: matrix count/scale mismatch", self.name));
        }
        device_extent(&[MAXR, self.k, 4], "capture X")?;
        for &m in &self.ms {
            if m == 0 || m > i32::MAX as usize {
                return Err(format!("{}: M must be positive signed 32-bit", self.name));
            }
            device_extent(&[m, self.k / 32, 34], "capture weight")?;
            let output = device_extent(&[MAXR, m, 4], "capture output")?;
            output
                .checked_add(2 * GUARD * 4)
                .ok_or("capture guarded output overflow")?;
        }
        self.caller_source(kind)?;
        Ok(kind)
    }

    fn caller_source(&self, kind: Kind) -> Result<&'static str> {
        let caller = &self.singleton;
        let ordinary = matches!(kind, Kind::Single);
        let pair = matches!(kind, Kind::Single | Kind::X2) && self.k == 2560;
        let (module, src) = match caller.symbol.as_str() {
            "gemv_q8_0_k320_staged" if ordinary && self.k == 320 => {
                ("qwen4_gemv_q8_0", QWEN4_GEMV_Q8_0_SRC)
            }
            "gemv_q8_0_k2560_staged" if ordinary && self.k == 2560 => {
                ("qwen4_gemv_q8_0", QWEN4_GEMV_Q8_0_SRC)
            }
            "gemv_q8_0_k2560_staged_pair" | "gemv_q8_0_k2560_staged_rows_x2" if pair => {
                ("qwen4_gemv_q8_0", QWEN4_GEMV_Q8_0_SRC)
            }
            "gemv_q8_0_k8" if matches!(kind, Kind::Hc(_)) => {
                ("qwen4_gemv_q8_0", QWEN4_GEMV_Q8_0_SRC)
            }
            "gemv_q8_0_wide_k640_staged" if ordinary && self.k == 640 => {
                ("qwen4_gemv_q8_0_wide", QWEN4_GEMV_Q8_0_WIDE_SRC)
            }
            "gemv_q8_0_wide" if ordinary && self.k <= 1536 && self.k != 320 && self.k != 640 => {
                ("gemv_q8_0_wide", GEMV_Q8_0_WIDE_SRC)
            }
            "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair" if pair => (
                "qwen4_decode_head_fuse",
                rdna_compute::kernels::QWEN4_DECODE_HEAD_FUSE_SRC,
            ),
            _ => {
                return Err(format!(
                    "{}: unsupported captured caller/domain {}",
                    self.name, caller.symbol
                ))
            }
        };
        if caller.module != module
            || !valid_sha(&caller.object_sha256)
            || caller
                .container_sha256
                .as_ref()
                .is_some_and(|sha| !valid_sha(sha))
        {
            return Err(format!(
                "{}: captured caller module/object pin mismatch",
                self.name
            ));
        }
        Ok(src)
    }
}

impl Capture {
    fn read(path: &Path, expected_sha: &str) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let manifest_sha = sha(&bytes);
        if manifest_sha != expected_sha {
            return Err(format!(
                "{}: manifest SHA drift actual={manifest_sha} expected={expected_sha}",
                path.display()
            ));
        }
        let manifest: CaptureManifest = serde_json::from_slice(&bytes).map_err(err)?;
        if manifest.schema != "q8_rowtile_capture.v1"
            || manifest.arch != "gfx1151"
            || manifest.cases.is_empty()
        {
            return Err("capture schema/architecture mismatch or empty cases".into());
        }
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let index_bytes = pinned_file(base, &manifest.source_index, None)?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(err)?;
        if index["schema"] != "qwen4-rowtile-index-v1"
            || index["arch"] != manifest.arch
            || index["rows"] != MAXR
            || index["capture_mode"] != "eager-liveprojection-operands-only"
            || index["retained_identity_claim"] != false
            || index["cases"].as_array().is_none_or(|entries| {
                entries
                    .iter()
                    .filter(|entry| entry["family"] == "q8")
                    .count()
                    != manifest.cases.len()
            })
        {
            return Err("capture source index domain/provenance mismatch".into());
        }
        let index_path = base.join(&manifest.source_index.path);
        let index_base = index_path.parent().ok_or("source index has no parent")?;
        println!("capture_manifest path={:?} sha256={manifest_sha} schema={} arch={} source_index={:?} source_index_sha256={}",
            path, manifest.schema, manifest.arch, manifest.source_index.path, manifest.source_index.sha256);
        let mut names = HashSet::new();
        let mut cases = Vec::with_capacity(manifest.cases.len());
        for spec in manifest.cases {
            if !names.insert(spec.name.clone()) {
                return Err(format!("duplicate capture case name {:?}", spec.name));
            }
            let kind = spec.domain()?;
            let mixed = captured_source(index_base, base, &index, &spec)?;
            let mut weights = Vec::with_capacity(spec.weights.len());
            for (matrix, (file, &m)) in spec.weights.iter().zip(&spec.ms).enumerate() {
                let len = device_extent(&[m, spec.k / 32, 34], "capture weight")?;
                let bytes = pinned_file(base, file, Some(len))?;
                if bytes
                    .chunks_exact(34)
                    .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7c00 == 0x7c00)
                {
                    return Err(format!("{}: nonfinite Q8 scale matrix={matrix}", spec.name));
                }
                println!("capture_payload case={:?} kind={} matrix={matrix} M={m} K={} path={:?} sha256={} bytes={len}",
                    spec.name, spec.kind, spec.k, file.path, file.sha256);
                weights.push(bytes);
            }
            let len = device_extent(&[MAXR, spec.k, 4], "capture X")?;
            let bytes = pinned_file(base, &spec.x, Some(len))?;
            let mut host_x = Vec::with_capacity(len / 4);
            for word in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                if !value.is_finite() {
                    return Err(format!("{}: nonfinite live captured X", spec.name));
                }
                host_x.push(value);
            }
            println!("capture_payload case={:?} kind={} X_path={:?} X_sha256={} X_bytes={len} scale={} captured_caller_module={} captured_caller_symbol={} captured_caller_object_sha256={}",
                spec.name, spec.kind, spec.x.path, spec.x.sha256, spec.scale,
                spec.singleton.module, spec.singleton.symbol, spec.singleton.object_sha256);
            cases.push(CapturedCase {
                spec,
                kind,
                weights,
                host_x,
                mixed,
            });
        }
        Ok(Self {
            arch: manifest.arch,
            cases,
        })
    }
}

fn raw(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0u8; len];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).map_err(err)?;
    Ok(bytes)
}

fn equal_bytes(label: &str, got: &[u8], want: &[u8]) -> Result<()> {
    if got.len() != want.len() {
        return Err(format!(
            "{label}: byte extents {} != {}",
            got.len(),
            want.len()
        ));
    }
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        return Err(format!(
            "{label}: first mismatch byte={i} got={:02x} want={:02x} extent={}",
            got[i],
            want[i],
            got.len()
        ));
    }
    Ok(())
}

impl Case {
    fn symbol(&self) -> &'static str {
        match self.kind {
            Kind::X2 => "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1151",
            Kind::Hc(_) => "gemv_q8_0_k8_rows_t8_pm_gfx1151",
            Kind::Single => match self.k {
                320 => "gemv_q8_0_k320_rows_t8_pm_gfx1151",
                2560 => "gemv_q8_0_k2560_rows_t8_pm_gfx1151",
                _ => "gemv_q8_0_wide_rows_t8_pm_gfx1151",
            },
        }
    }

    fn singleton(&self) -> (&'static str, &'static str, &'static str) {
        if matches!(self.kind, Kind::Hc(_)) {
            return ("gemv_q8_0_k8", "qwen4_gemv_q8_0", QWEN4_GEMV_Q8_0_SRC);
        }
        match self.k {
            320 => (
                "gemv_q8_0_k320_staged",
                "qwen4_gemv_q8_0",
                QWEN4_GEMV_Q8_0_SRC,
            ),
            640 => (
                "gemv_q8_0_wide_k640_staged",
                "qwen4_gemv_q8_0_wide",
                QWEN4_GEMV_Q8_0_WIDE_SRC,
            ),
            2560 => (
                "gemv_q8_0_k2560_staged",
                "qwen4_gemv_q8_0",
                QWEN4_GEMV_Q8_0_SRC,
            ),
            _ => ("gemv_q8_0_wide", "gemv_q8_0_wide", GEMV_Q8_0_WIDE_SRC),
        }
    }

    fn block(&self) -> u32 {
        if matches!(self.kind, Kind::Hc(_)) {
            512
        } else {
            256
        }
    }

    fn input_snapshot(&self, gpu: &Gpu) -> Result<[Vec<u8>; 2]> {
        let len = MAXR * self.k * 4;
        Ok([raw(gpu, &self.x, len)?, raw(gpu, &self.x_alt, len)?])
    }

    fn immutable(&self, gpu: &Gpu, before: &[Vec<u8>; 2], label: &str) -> Result<()> {
        let after = self.input_snapshot(gpu)?;
        for i in 0..2 {
            equal_bytes(
                &format!("{} {label} input {i}", self.name),
                &after[i],
                &before[i],
            )?;
        }
        for (matrix, w) in self.weights.iter().enumerate() {
            for (alternate, tensor, bytes) in [
                (false, &w.primary, &w.bytes),
                (true, &w.alternate, &w.alternate_bytes),
            ] {
                equal_bytes(
                    &format!(
                        "{} {label} weight matrix={matrix} alternate={alternate}",
                        self.name
                    ),
                    &raw(gpu, tensor, bytes.len())?,
                    bytes,
                )?;
            }
        }
        if let Some(mixed) = &self.mixed {
            for i in 0..2 {
                equal_bytes(
                    &format!("{} {label} BF16 weight {i}", self.name),
                    &raw(gpu, &mixed.weights[i], mixed.bytes[i].len())?,
                    &mixed.bytes[i],
                )?;
            }
        }
        Ok(())
    }

    fn captured_row(&self, gpu: &mut Gpu, row: usize) -> Result<()> {
        let caller = self.capture.as_ref().ok_or("missing captured caller")?;
        let x = self.x.sub_offset(row * self.k, self.k);
        let q0 = self.outs[0]
            .view()
            .sub_offset(row * self.outs[0].m, self.outs[0].m);
        let q1 = self
            .outs
            .get(1)
            .map(|o| o.view().sub_offset(row * o.m, o.m));
        match caller.symbol.as_str() {
            "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair" => {
                let mixed = self.mixed.as_ref().ok_or("missing real BF16 subgroup")?;
                let y0 = mixed.outs[0]
                    .view()
                    .sub_offset(row * mixed.outs[0].m, mixed.outs[0].m);
                let y1 = mixed.outs[1]
                    .view()
                    .sub_offset(row * mixed.outs[1].m, mixed.outs[1].m);
                gpu.gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
                    [
                        (&mixed.weights[0], &y0, mixed.outs[0].m),
                        (&mixed.weights[1], &y1, mixed.outs[1].m),
                    ],
                    [
                        (&self.weights[0].primary, &q0),
                        (
                            &self.weights[1].primary,
                            q1.as_ref().ok_or("missing second Q8 output")?,
                        ),
                    ],
                    self.outs[0].m,
                    &x,
                    self.k,
                )
                .map_err(err)?;
            }
            "gemv_q8_0_k2560_staged_pair" => {
                if self.outs.len() != 2 || self.outs[0].m != self.outs[1].m {
                    return Err("captured pair requires two equal Ms".into());
                }
                gpu.gemv_q8_0_pair(
                    &self.weights[0].primary,
                    &self.weights[1].primary,
                    &x,
                    &q0,
                    q1.as_ref().ok_or("missing second Q8 output")?,
                    self.outs[0].m,
                    self.k,
                )
                .map_err(err)?;
            }
            "gemv_q8_0_k2560_staged_rows_x2" => {
                gpu.gemv_q8_0_k2560_staged_rows_x2(
                    &self.weights[0].primary,
                    &q0,
                    self.outs[0].m,
                    &self.weights[1].primary,
                    q1.as_ref().ok_or("missing second Q8 output")?,
                    self.outs[1].m,
                    &x,
                    1,
                )
                .map_err(err)?;
            }
            "gemv_q8_0_k8" => {
                let Kind::Hc(scale) = self.kind else {
                    return Err("captured HC domain mismatch".into());
                };
                gpu.gemv_q8_0_k8(
                    &self.weights[0].primary,
                    &x,
                    &q0,
                    self.outs[0].m,
                    self.k,
                    scale,
                )
                .map_err(err)?;
            }
            _ => {
                gpu.gemv_q8_0(&self.weights[0].primary, &x, &q0, self.outs[0].m, self.k)
                    .map_err(err)?;
            }
        }
        if gpu.last_launched_kernel() != Some(caller.symbol.as_str()) {
            return Err(format!(
                "{}: actual captured C1 route differs from {}",
                self.name, caller.symbol
            ));
        }
        let (module, container, object) = gpu
            .lab_kernel_device_identity(&caller.symbol)
            .map_err(err)?;
        if module != caller.module
            || !object.eq_ignore_ascii_case(&caller.object_sha256)
            || caller
                .container_sha256
                .as_ref()
                .is_some_and(|pin| !container.eq_ignore_ascii_case(pin))
        {
            return Err(format!(
                "{}: captured C1 module/raw/container mismatch",
                self.name
            ));
        }
        println!("captured_c1 case={:?} row={row} symbol={} module={module} object_sha256={object} container_sha256={container}",
            self.name, caller.symbol);
        Ok(())
    }

    fn launch(&self, gpu: &mut Gpu, x: &GpuTensor, rows: usize, swap: bool) -> Result<()> {
        let weight = |i: usize| {
            let w = &self.weights[i];
            if swap {
                &w.alternate
            } else {
                &w.primary
            }
        };
        let mut b = KernargBlob::new();
        b.push_ptr(weight(0).buf.as_ptr());
        if matches!(self.kind, Kind::X2) {
            b.push_ptr(weight(1).buf.as_ptr());
        }
        b.push_ptr(x.buf.as_ptr());
        for o in &self.outs {
            b.push_ptr(o.view().buf.as_ptr());
        }
        for o in &self.outs {
            b.push_i32(o.m as i32);
        }
        b.push_i32(self.k as i32);
        b.push_i32(rows as i32);
        if let Kind::Hc(scale) = self.kind {
            b.push_f32(scale.unwrap_or(1.0));
            b.push_i32(i32::from(scale.is_some()));
        }
        let expected = match self.kind {
            Kind::Single => 36,
            Kind::X2 => 56,
            Kind::Hc(_) => 44,
        };
        if b.as_mut_slice().len() != expected {
            return Err(format!(
                "REFUSAL {} ABI extent != {expected}",
                self.symbol()
            ));
        }
        let tile = if matches!(self.kind, Kind::Hc(_)) {
            2
        } else {
            8
        };
        let gx: usize = self.outs.iter().map(|o| o.m.div_ceil(tile)).sum();
        gpu.launch_kernel_blob(
            self.symbol(),
            [gx as u32, rows.div_ceil(8) as u32, 1],
            [self.block(), 1, 1],
            0, // All LDS is static, never dynamic.
            b.as_mut_slice(),
        )
        .map_err(err)
    }
}

fn loaded_identity(gpu: &Gpu, symbol: &str, module: &str, pin: Option<&str>) -> Result<()> {
    let (actual_module, container_sha, object_sha) =
        gpu.lab_kernel_device_identity(symbol).map_err(err)?;
    if actual_module != module || !valid_sha(&object_sha) || !valid_sha(&container_sha) {
        return Err(format!(
            "REFUSAL symbol={symbol} expected_module={module} actual_module={actual_module} object_sha={object_sha}"
        ));
    }
    if let Some(expected) = pin {
        if object_sha != expected {
            return Err(format!(
                "REFUSAL {symbol} loaded object SHA={object_sha} embedded pin={expected}"
            ));
        }
    }
    println!("identity symbol={symbol} module={actual_module} object_sha256={object_sha} container_sha256={container_sha}");
    Ok(())
}

/// Execute the original C1 first, including its BF16 subgroup, at every
/// sequential position and every prefix/reversal. Primitive Q8 is auxiliary.
fn captured_reference(
    gpu: &mut Gpu,
    case: &Case,
    tally: &mut Tally,
) -> Result<Option<Vec<Vec<u8>>>> {
    if case.capture.is_none() {
        return Ok(None);
    }
    let outputs = || {
        case.outs
            .iter()
            .chain(case.mixed.iter().flat_map(|m| m.outs.iter()))
    };
    for out in outputs() {
        out.reset(gpu, MAXR)?;
    }
    let before = case.input_snapshot(gpu)?;
    for row in 0..MAXR {
        case.captured_row(gpu, row)?;
        tally.caller_rows += 1;
    }
    case.immutable(gpu, &before, "captured C1 baseline")?;
    let mut baseline = Vec::with_capacity(case.outs.len() + case.mixed.as_ref().map_or(0, |_| 2));
    for out in outputs() {
        let bytes = out.download(gpu)?;
        out.integrity(&bytes, MAXR, "captured C1 baseline")?;
        baseline.push(bytes);
    }
    for rows in 1..=MAXR {
        for reverse in [false, true] {
            upload_rows(gpu, &case.x, case.k, &case.host_x, rows, reverse)?;
            let before = case.input_snapshot(gpu)?;
            for out in outputs() {
                out.reset(gpu, rows)?;
            }
            for row in 0..rows {
                case.captured_row(gpu, row)?;
                tally.caller_rows += 1;
            }
            case.immutable(gpu, &before, "captured C1 prefix")?;
            for (component, out) in outputs().enumerate() {
                let bytes = out.download(gpu)?;
                out.integrity(&bytes, rows, "captured C1 prefix")?;
                let expected = out.expected(&baseline[component], rows, |r| {
                    if reverse {
                        rows - 1 - r
                    } else {
                        r
                    }
                });
                equal_bytes(
                    &format!(
                        "{} original C1 R={rows} reverse={reverse} component={component}",
                        case.name
                    ),
                    &bytes,
                    &expected,
                )?;
            }
        }
    }
    upload_rows(gpu, &case.x, case.k, &case.host_x, MAXR, false)?;
    if let Some(mixed) = &case.mixed {
        let before = case.input_snapshot(gpu)?;
        for out in &mixed.outs {
            out.reset(gpu, MAXR)?;
        }
        for row in 0..MAXR {
            let x = case.x.sub_offset(row * case.k, case.k);
            for i in 0..2 {
                let out = &mixed.outs[i];
                let y = out.view().sub_offset(row * out.m, out.m);
                gpu.gemv_bf16_xf32(&mixed.weights[i], &x, &y, out.m, case.k)
                    .map_err(err)?;
                if gpu.last_launched_kernel() != Some("gemv_bf16_xf32") {
                    return Err("mixed auxiliary BF16 route mismatch".into());
                }
                loaded_identity(gpu, "gemv_bf16_xf32", "gemv_bf16_xf32", None)?;
            }
        }
        case.immutable(gpu, &before, "mixed BF16 auxiliary")?;
        for i in 0..2 {
            let bytes = mixed.outs[i].download(gpu)?;
            mixed.outs[i].integrity(&bytes, MAXR, "mixed BF16 auxiliary")?;
            equal_bytes(
                "actual mixed C1 versus BF16 component",
                &bytes,
                &baseline[case.outs.len() + i],
            )?;
        }
    }
    baseline.truncate(case.outs.len());
    Ok(Some(baseline))
}

fn run_reference(gpu: &mut Gpu, case: &Case, tally: &mut Tally) -> Result<Vec<Vec<u8>>> {
    let captured = captured_reference(gpu, case, tally)?;
    let before = case.input_snapshot(gpu)?;
    for o in &case.outs {
        o.reset(gpu, MAXR)?;
    }
    for row in 0..MAXR {
        let xr = case.x.sub_offset(row * case.k, case.k);
        for (matrix, (o, w)) in case.outs.iter().zip(&case.weights).enumerate() {
            let y = o.view().sub_offset(row * o.m, o.m);
            let result = match case.kind {
                Kind::Hc(scale) => gpu.gemv_q8_0_k8(&w.primary, &xr, &y, o.m, case.k, scale),
                _ => gpu.gemv_q8_0(&w.primary, &xr, &y, o.m, case.k),
            };
            result.map_err(|e| {
                format!(
                    "REFUSAL {} singleton matrix={matrix} row={row}: {e}",
                    case.name
                )
            })?;
            tally.singleton_rows += 1;
        }
    }
    case.immutable(gpu, &before, "singleton")?;
    let (symbol, module, src) = case.singleton();
    loaded_identity(gpu, symbol, module, None)?;
    println!(
        "singleton_source symbol={symbol} source_sha256={}",
        sha(src.as_bytes())
    );
    case.outs
        .iter()
        .enumerate()
        .map(|(matrix, o)| {
            let bytes = o.download(gpu)?;
            o.integrity(
                &bytes,
                MAXR,
                &format!("{} singleton matrix={matrix}", case.name),
            )?;
            if let Some(reference) = &captured {
                equal_bytes(
                    "actual captured C1 versus auxiliary Q8 component",
                    &bytes,
                    &reference[matrix],
                )?;
            }
            Ok(bytes)
        })
        .collect()
}

/// Inactive logical rows alone contain NaN poison; live rows are finite fixtures.
fn upload_rows(
    gpu: &Gpu,
    target: &GpuTensor,
    k: usize,
    src: &[f32],
    rows: usize,
    reverse: bool,
) -> Result<()> {
    let mut x = vec![f32::from_bits(INPUT_POISON); MAXR * k];
    for r in 0..rows {
        let from = if reverse { rows - 1 - r } else { r };
        x[r * k..(r + 1) * k].copy_from_slice(&src[from * k..(from + 1) * k]);
    }
    gpu.hip
        .memcpy_htod(&target.buf, &f32_bytes(&x))
        .map_err(err)
}

fn check(
    gpu: &mut Gpu,
    case: &Case,
    tally: &mut Tally,
    label: &str,
    refs: &[Vec<u8>],
    rows: usize,
    reverse: bool,
    alternate_x: bool,
) -> Result<()> {
    let before = case.input_snapshot(gpu)?;
    for o in &case.outs {
        o.reset(gpu, rows)?;
    }
    let x = if alternate_x { &case.x_alt } else { &case.x };
    case.launch(gpu, x, rows, false).map_err(|e| {
        format!(
            "REFUSAL {} {label} symbol={} R={rows}: {e}",
            case.name,
            case.symbol()
        )
    })?;
    tally.candidates += 1;
    case.immutable(gpu, &before, &format!("{label} R={rows}"))?;
    for (matrix, o) in case.outs.iter().enumerate() {
        let got = o.download(gpu)?;
        let context = format!(
            "{} {label} symbol={} R={rows} matrix={matrix}",
            case.name,
            case.symbol()
        );
        o.integrity(&got, rows, &context)?;
        let want = o.expected(
            &refs[matrix],
            rows,
            |r| if reverse { rows - 1 - r } else { r },
        );
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            return Err(format!(
                "{context}: first mismatch {} got={:02x} want={:02x} output_allocation_bytes={} active_output_bytes={} input_allocation_bytes={}",
                o.locate(i), got[i], want[i], o.bytes(), rows * o.m * 4, MAXR * case.k * 4
            ));
        }
        println!("cell case={} mode={label} R={rows} symbol={} matrix={matrix} X_active_bytes={} X_inactive_bytes={} Y_active_bytes={} Y_inactive_bytes={} Y_allocation_bytes={} status=equal",
            case.name, case.symbol(), rows * case.k * 4, (MAXR - rows) * case.k * 4,
            rows * o.m * 4, (MAXR - rows) * o.m * 4, o.bytes());
    }
    Ok(())
}

fn negative_controls(
    gpu: &mut Gpu,
    case: &Case,
    refs: &[Vec<u8>],
    rng: &mut Rng,
    tally: &mut Tally,
) -> Result<()> {
    let rows = MAXR;
    let changed = gen_x(case.k, case.keep, rng);
    upload_rows(gpu, &case.x, case.k, &case.host_x, rows, false)?;
    upload_rows(gpu, &case.x_alt, case.k, &changed, rows, false)?;
    for swap in [true, false] {
        let before = case.input_snapshot(gpu)?;
        for o in &case.outs {
            o.reset(gpu, rows)?;
        }
        let x = if swap { &case.x } else { &case.x_alt };
        case.launch(gpu, x, rows, swap)
            .map_err(|e| format!("REFUSAL {} negative R={rows} swap={swap}: {e}", case.name))?;
        tally.candidates += 1;
        tally.controls += 1;
        case.immutable(gpu, &before, &format!("negative R={rows} swap={swap}"))?;
        for (matrix, o) in case.outs.iter().enumerate() {
            let got = o.download(gpu)?;
            let context = format!(
                "{} negative R={rows} swap={swap} matrix={matrix}",
                case.name
            );
            o.integrity(&got, rows, &context)?;
            let lo = GUARD * 4;
            let hi = lo + rows * o.m * 4;
            if got[lo..hi] == refs[matrix][lo..hi] {
                return Err(format!("{context}: no active output byte changed"));
            }
        }
    }
    Ok(())
}

fn verify(gpu: &mut Gpu, case: &mut Case, rng: &mut Rng, tally: &mut Tally) -> Result<()> {
    gpu.ensure_lane_rows_pm_entry(case.symbol())
        .map_err(|e| format!("REFUSAL {}: {e}", case.symbol()))?;
    loaded_identity(
        gpu,
        case.symbol(),
        PM_MODULE,
        Some(QWEN4_LANE_ROWS_PM_GFX1151_SHA256),
    )?;
    let blocks = gpu
        .occupancy_max_active_blocks(case.symbol(), [case.block(), 1, 1], 0)
        .map_err(err)?;
    if blocks < 1 {
        return Err(format!(
            "REFUSAL {} has no resident workgroup",
            case.symbol()
        ));
    }
    let static_lds = if matches!(case.kind, Kind::Hc(_)) {
        16384
    } else {
        8192
    };
    println!("case={} K={} block={} resident_blocks={} static_LDS_bytes={static_lds} dynamic_LDS_bytes=0 X_bytes={} image_sha256={}",
        case.name, case.k, case.block(), blocks, MAXR * case.k * 4, QWEN4_LANE_ROWS_PM_GFX1151_SHA256);
    for (matrix, (w, o)) in case.weights.iter().zip(&case.outs).enumerate() {
        println!("extent matrix={matrix} M={} W_bytes={} W_alt_bytes={} Y_allocation_bytes={} Y_head_bytes={} Y_tail_bytes={} max_active_bytes={}",
            o.m, w.bytes.len(), w.alternate_bytes.len(), o.bytes(), GUARD * 4, GUARD * 4, MAXR * o.m * 4);
    }
    let refs = run_reference(gpu, case, tally)?;
    for rows in 1..=MAXR {
        upload_rows(gpu, &case.x, case.k, &case.host_x, rows, false)?;
        check(gpu, case, tally, "forward", &refs, rows, false, false)?;
        upload_rows(gpu, &case.x_alt, case.k, &case.host_x, rows, true)?;
        check(gpu, case, tally, "reversed", &refs, rows, true, true)?;
    }
    negative_controls(gpu, case, &refs, rng, tally)?;
    let stable_address = case.x.buf.as_ptr();
    let original_x = std::mem::replace(&mut case.host_x, gen_x(case.k, case.keep, rng));
    upload_rows(gpu, &case.x, case.k, &case.host_x, MAXR, false)?;
    let replay = run_reference(gpu, case, tally)?;
    if case.x.buf.as_ptr() != stable_address {
        return Err(format!("{} replay changed input address", case.name));
    }
    for rows in 1..=MAXR {
        upload_rows(gpu, &case.x, case.k, &case.host_x, rows, false)?;
        check(
            gpu,
            case,
            tally,
            "stable-address-replay",
            &replay,
            rows,
            false,
            false,
        )?;
    }
    case.host_x = original_x;
    tally.cases += 1;
    Ok(())
}

fn make_case(
    gpu: &mut Gpu,
    ms: &[usize],
    k: usize,
    kind: Kind,
    wide: bool,
    part: Option<usize>,
    seed: u64,
    rng: &mut Rng,
) -> Result<Case> {
    let keep = part.map(|p| (p * (k / 8), (p + 1) * (k / 8)));
    let mut weights = Vec::new();
    let mut outs = Vec::new();
    for &m in ms {
        let bytes = gen_q8(m, k, wide, rng);
        let alternate_bytes = gen_q8(m, k, wide, rng);
        let primary = gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?;
        let alternate = gpu
            .upload_raw(&alternate_bytes, &[alternate_bytes.len()])
            .map_err(err)?;
        weights.push(Weight {
            primary,
            alternate,
            bytes,
            alternate_bytes,
        });
        outs.push(Out::new(gpu, m)?);
    }
    let host_x = gen_x(k, keep, rng);
    let x = gpu.upload_f32(&host_x, &[host_x.len()]).map_err(err)?;
    let x_alt = gpu.upload_f32(&host_x, &[host_x.len()]).map_err(err)?;
    let arm = match kind {
        Kind::Single => "single".to_string(),
        Kind::X2 => "x2".to_string(),
        Kind::Hc(scale) => format!("hc scale={scale:?}"),
    };
    Ok(Case {
        name: format!(
            "{arm} M={ms:?} K={k} wide_scales={wide} partition={part:?} seed={seed:016x}"
        ),
        k,
        kind,
        weights,
        outs,
        x,
        x_alt,
        host_x,
        keep,
        capture: None,
        mixed: None,
    })
}

fn make_captured_case(gpu: &mut Gpu, captured: CapturedCase, rng: &mut Rng) -> Result<Case> {
    let CapturedCase {
        spec,
        kind,
        weights: captured_weights,
        host_x,
        mixed,
    } = captured;
    let mut weights = Vec::with_capacity(captured_weights.len());
    let mut outs = Vec::with_capacity(spec.ms.len());
    for (bytes, &m) in captured_weights.into_iter().zip(&spec.ms) {
        let alternate_bytes = gen_q8(m, spec.k, false, rng);
        let primary = gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?;
        let alternate = gpu
            .upload_raw(&alternate_bytes, &[alternate_bytes.len()])
            .map_err(err)?;
        weights.push(Weight {
            primary,
            alternate,
            bytes,
            alternate_bytes,
        });
        outs.push(Out::new(gpu, m)?);
    }
    let mixed = match mixed {
        Some((ms, bytes)) => {
            let weights = [
                gpu.upload_raw(&bytes[0], &[bytes[0].len()]).map_err(err)?,
                gpu.upload_raw(&bytes[1], &[bytes[1].len()]).map_err(err)?,
            ];
            let outs = [Out::new(gpu, ms[0])?, Out::new(gpu, ms[1])?];
            Some(MixedOperands {
                weights,
                bytes,
                outs,
            })
        }
        None => None,
    };
    let x = gpu.upload_f32(&host_x, &[host_x.len()]).map_err(err)?;
    let x_alt = gpu.upload_f32(&host_x, &[host_x.len()]).map_err(err)?;
    Ok(Case {
        name: spec.name,
        k: spec.k,
        kind,
        weights,
        outs,
        x,
        x_alt,
        host_x,
        keep: None,
        capture: Some(spec.singleton),
        mixed,
    })
}

fn run() -> Result<()> {
    let cli = Cli::parse()?;
    // Pin and validate every captured source/operand before initializing a GPU.
    let capture = cli
        .capture
        .as_ref()
        .map(|(path, pin)| Capture::read(path, pin))
        .transpose()?;
    let pin = QWEN4_LANE_ROWS_PM_GFX1151_SHA256;
    if QWEN4_LANE_ROWS_PM_GFX1151.is_empty()
        || !valid_sha(pin)
        || sha(QWEN4_LANE_ROWS_PM_GFX1151) != pin
    {
        return Err(format!(
            "REFUSAL embedded candidate empty/drift image_sha256={} pin={pin}",
            sha(QWEN4_LANE_ROWS_PM_GFX1151)
        ));
    }
    println!("candidate image_bytes={} image_sha256={pin} q8_emitter_source_sha256={} bundle_source_sha256={}",
        QWEN4_LANE_ROWS_PM_GFX1151.len(), sha(Q8_EMITTER_SRC.as_bytes()), sha(PM_BUNDLE_SRC.as_bytes()));
    let mut gpu = Gpu::init().map_err(|e| format!("REFUSAL GPU initialization: {e}"))?;
    if gpu.arch != "gfx1151" {
        return Err(format!(
            "REFUSAL unsupported architecture {}; requires exact gfx1151, no gfx1201/gfx1100 alias",
            gpu.arch
        ));
    }
    println!("hardware arch={} device={} uuid={} pci={} hip_runtime={:?} mode=direct-PM-versus-actual-Halo-singleton R=1..64",
        gpu.arch, gpu.device_id,
        gpu.hip.device_uuid(gpu.device_id).map_err(err)?,
        gpu.hip.device_pci_bus_id(gpu.device_id).map_err(err)?,
        gpu.hip.runtime_version().map_err(err)?);
    gpu.lane_rows_scope = false;
    let mut tally = Tally::default();
    for seed in [0x0858_c8c8_9f3a_7d11, 0xb64e_1a29_7703_5cd7] {
        let mut rng = Rng(seed);
        for (k, production, retained_m) in [(320, 512, 48), (640, 640, 48), (2560, 2560, 640)] {
            for m in [1, 2, 3, 7, 8, 9, 17, 33, production, retained_m] {
                for wide in [false, true] {
                    let mut case =
                        make_case(&mut gpu, &[m], k, Kind::Single, wide, None, seed, &mut rng)?;
                    verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
                }
            }
        }
        // All legal small-K block counts, including every four-chain remainder
        // and the upper boundary K1440/1472/1504/1536.
        for nblk in 1..=48 {
            let k = nblk * 32;
            if k == 320 || k == 640 {
                continue;
            } // Complete staged cases above.
            for m in [1, 2, 3, 7, 8, 9, 17] {
                for wide in [false, true] {
                    let mut case =
                        make_case(&mut gpu, &[m], k, Kind::Single, wide, None, seed, &mut rng)?;
                    verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
                }
            }
        }
        for (m0, m1) in [
            (1, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (7, 8),
            (8, 9),
            (9, 1),
            (9, 7),
            (17, 33),
            (640, 640),
            (640, 2560),
        ] {
            for wide in [false, true] {
                let mut case = make_case(
                    &mut gpu,
                    &[m0, m1],
                    2560,
                    Kind::X2,
                    wide,
                    None,
                    seed,
                    &mut rng,
                )?;
                verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
            }
        }
        // Raw/activated HC, partial partition tiles, production K/M and all
        // eight partition masks, preserving masks in changed-X and replay arms.
        for scale in [
            None,
            Some(0.25),
            Some(-0.25),
            Some(1.0 / 1_048_576.0),
            Some(1024.0),
        ] {
            for k in [256, 512, 768, 1024, 1536, 2048, 2304, 5120, 10240] {
                for m in [1, 2, 3, 4, 5, 7, 8, 9, 17, 320] {
                    let mut case = make_case(
                        &mut gpu,
                        &[m],
                        k,
                        Kind::Hc(scale),
                        false,
                        None,
                        seed,
                        &mut rng,
                    )?;
                    verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
                }
            }
            let mut extreme_weights = make_case(
                &mut gpu,
                &[9],
                10240,
                Kind::Hc(scale),
                true,
                None,
                seed,
                &mut rng,
            )?;
            verify(&mut gpu, &mut extreme_weights, &mut rng, &mut tally)?;
            for k in [256, 768, 1536, 2304, 10240] {
                for part in 0..8 {
                    let mut case = make_case(
                        &mut gpu,
                        &[9],
                        k,
                        Kind::Hc(scale),
                        false,
                        Some(part),
                        seed,
                        &mut rng,
                    )?;
                    verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
                }
            }
        }
    }
    // Frozen shape matrix: two seeds, staged/small-K/x2/HC cases; every
    // completed case owes 192 equality cells and two sensitive controls.
    const EXPECTED_CASES: usize = 2 * (3 * 10 * 2 + 46 * 7 * 2 + 11 * 2 + 5 * (9 * 10 + 1 + 5 * 8));
    const X2_CASES: usize = 2 * 11 * 2;
    if tally.cases != EXPECTED_CASES
        || tally.controls != tally.cases * 2
        || tally.candidates != tally.cases * (3 * MAXR + 2)
        || tally.singleton_rows != 2 * MAXR * (EXPECTED_CASES + X2_CASES)
    {
        return Err(format!("REFUSAL incomplete gate cases={} expected={EXPECTED_CASES} candidates={} singleton_rows={} controls={}",
            tally.cases, tally.candidates, tally.singleton_rows, tally.controls));
    }
    if let Some(capture) = capture {
        if capture.arch != gpu.arch {
            return Err("captured/device architecture mismatch".into());
        }
        let expected_cases = capture.cases.len() * 2;
        let expected_matrices = capture.cases.iter().map(|c| c.weights.len()).sum::<usize>() * 2;
        let mut captured_tally = Tally::default();
        for captured in capture.cases {
            let mut rng = Rng(0x0858_c8c8_9f3a_7d11);
            let mut case = make_captured_case(&mut gpu, captured, &mut rng)?;
            for seed in [0x0858_c8c8_9f3a_7d11, 0xb64e_1a29_7703_5cd7] {
                let mut rng = Rng(seed);
                upload_rows(&gpu, &case.x, case.k, &case.host_x, MAXR, false)?;
                upload_rows(&gpu, &case.x_alt, case.k, &case.host_x, MAXR, false)?;
                println!("captured_case case={:?} seed={seed:016x} operand_origin=eager-liveprojection capture_mode=eager-liveprojection-operands-only retained_identity_claim=false alternate_weights=synthetic_negative_control replay_X=synthetic_stress", case.name);
                verify(&mut gpu, &mut case, &mut rng, &mut captured_tally)?;
            }
        }
        let caller_rows_per_case = 2 * (MAXR + MAXR * (MAXR + 1));
        if captured_tally.cases != expected_cases
            || captured_tally.controls != expected_cases * 2
            || captured_tally.candidates != expected_cases * (3 * MAXR + 2)
            || captured_tally.singleton_rows != 2 * MAXR * expected_matrices
            || captured_tally.caller_rows != expected_cases * caller_rows_per_case
        {
            return Err("incomplete captured original-C1 gate".into());
        }
        println!("PASS q8_wide_rows_oracle captured_cases={} actual_caller_rows={} candidate_launches={} singleton_rows={} negative_controls={} seeded_cases={} R=1..64 finite-live-operands-only retained_identity_claim=false",
            captured_tally.cases, captured_tally.caller_rows, captured_tally.candidates,
            captured_tally.singleton_rows, captured_tally.controls, tally.cases);
    } else {
        println!("DIAGNOSTIC q8_wide_rows_oracle seeded_cases={} candidate_launches={} singleton_rows={} negative_controls={} captured_G1=false",
            tally.cases, tally.candidates, tally.singleton_rows, tally.controls);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL q8_wide_rows_oracle: {e}");
        std::process::exit(1);
    }
}
