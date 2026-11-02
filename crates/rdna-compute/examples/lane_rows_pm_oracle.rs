// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict gfx1151 lane-row PM byte gate, at every logical width 1..=64.
//!
//! `lane_rows_pm_oracle --singleton-pins pins.json [--seed N] [--bench N] [--capture manifest.json ...]`
//!
//! `pins.json` is a reviewed JSON map from actual runtime singleton symbols to
//! their SHA-256 code-object hashes. Missing pins, unavailable object identity,
//! unsupported devices, missing tile exports and launch errors are refusals.
//! The embedded PM image is independently checked against its registered pin.
//!
//! Each candidate uses one fixed-eight-row-tile launch, grid.y=ceil(R/8).
//! Its complete guarded output (including inactive rows) must equal independent
//! actual runtime singleton calls. Narrow PM calls, including PM C1, are an
//! additional comparison, never the singleton reference. Incumbent rows are
//! compared separately; every byte mismatch is fatal. Seeded finite inputs
//! include signed zeros, subnormals, disparate magnitudes and HC exp extremes.
//! This hardware gate makes no CPU-emulator or universal exceptional-value claim.
//!
//! `--bench N` runs the existing narrow diagnostic timing only after bytes pass;
//! it is not a full-window throughput qualification.
//!
//! Real-capture arm (`--capture manifest.json`, repeatable): each manifest is a
//! SHA-pinned `qwen4-rowtile-operands-v1` (arch gfx1151, rows 64) with
//! `provenance.source` pinning its `qwen4-rowtile-source-v1` source-index.
//! Source must carry `capture_mode="eager-liveprojection-operands-only"` and
//! `retained_identity_claim=false`, 64 sequential row recipes with stable
//! singleton occurrence/grid/block/shared_mem, four immutable weight bindings,
//! and model/prompt artifact pins (streaming SHA, conflicting pins refused).
//! Weights (`m*(k/256)*200` MQ6, `m*(k/32)*34` Q8) and X (`64*k*4` LE f32)
//! are SHA-pinned with exact byte extents before upload; inactive X rows are
//! NaN-poisoned per width and guarded outputs/immutability are checked every
//! arm. Admitted here: (`mq6`,`single`/`x4`), (`q8`,`single`/`x2`/`hc_k8`).
//! For captured `q8/x2` whose source singleton is the actual mixed caller
//! `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair` (`qwen4_decode_head_fuse`, ABI88,
//! grid `[M0+M1+2*MQ,1,1]`, block 32), the reference launches that actual mixed
//! symbol on the Q8 subgroup (`M0=M1=0`, proper ABI/dummy guarded arrays) and
//! compares its Q8 outputs; original mixed Ms/type/component_offset/symbol/
//! objectSHA are preserved and logged, never replaced by a generic primitive.
//! Q8 pair caller `gemv_q8_0_k2560_staged_pair` likewise launches its actual
//! pair arithmetic. Seeded gates stay separate; no fake/random captures and no
//! skip=PASS: missing/changed captures are refusals.

use hip_bridge::KernargBlob;
use rdna_compute::{kernels, DType, Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

/// Logical rows of every case; the physical PM tile remains eight.
const MAXB: usize = 64;
/// Largest narrow column count.
const NARROW_MAX: usize = 8;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
const SENTINEL: u8 = 0x7f;
const HC_SCALE: f32 = 0.25;
/// Column magnitudes: `0.0` is an all-zero (+-0) column, `1024` reaches the
/// k8 `hc_act` exp guards. Columns 0..8 are the narrow oracle's.
const AMPS: [f32; 32] = [
    1.0,
    1.0 / 1024.0,
    1.0 / 16.0,
    4.0,
    64.0,
    0.5,
    0.0,
    1024.0,
    2.0,
    1.0 / 256.0,
    8.0,
    1.0 / 4096.0,
    128.0,
    0.25,
    32.0,
    1.0 / 64.0,
    0.0,
    0.75,
    16.0,
    1024.0,
    1.0 / 32.0,
    256.0,
    3.0,
    1.0 / 8.0,
    1.0 / 2048.0,
    48.0,
    1.5,
    512.0,
    1.0 / 512.0,
    12.0,
    0.0,
    1024.0,
];

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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

/// MQ6G256V2 weights: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16)
/// and 192 packed payload bytes.
fn gen_mq6(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.chunks_exact_mut(200) {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.f16(9, 2), rng.f16(7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// Q8_0 weights: `[m][k/32]` blocks of an f16 scale and 32 int8.
fn gen_q8(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 32) * 34];
    for blk in bytes.chunks_exact_mut(34) {
        rng.fill(&mut blk[2..]);
        blk[..2].copy_from_slice(&rng.f16(8, 0).to_le_bytes());
    }
    bytes
}

/// `[MAXB][k]` activations with the per-column magnitudes of [`AMPS`].
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXB * k);
    for c in 0..MAXB {
        let amp = AMPS[c % AMPS.len()];
        for _ in 0..k {
            x.push(match rng.below(64) {
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
            });
        }
    }
    x
}

/// One output allocation: `GUARD | MAXB * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + MAXB * m], DType::F32)
            .map_err(err)?;
        Ok(Self { full, m })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXB * self.m)
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAXB * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.full.buf, i32::from(SENTINEL), self.bytes())
            .map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.bytes());
        Ok(v)
    }
    /// Expected bytes: sentinel everywhere, `reference`'s first `b` columns in place.
    fn expected(&self, reference: &[u8], b: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + b * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXB * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (col, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("column {col} row {} byte {}", rem / 4, rem % 4)
        }
    }
}

// ---------------------------------------------------------------------------
// Wide image: direct module / blob launches.
// ---------------------------------------------------------------------------

/// Frozen fixed-eight-row-tile exports (total logical R in the existing ABI).
fn wide_symbol(stem: &str) -> String {
    format!("{stem}_t8_pm_gfx1151")
}

const MQ6_STEM: &str = "gemv_mq6g256v2_rows";
const MQ6_X4_STEM: &str = "gemv_mq6g256v2_rows_x4";
const Q8_K8_STEM: &str = "gemv_q8_0_k8_rows";
const Q8_K320_STEM: &str = "gemv_q8_0_k320_rows";
const Q8_K2560_STEM: &str = "gemv_q8_0_k2560_rows";
const Q8_WIDE_STEM: &str = "gemv_q8_0_wide_rows";
const Q8_K2560_X2_STEM: &str = "gemv_q8_0_k2560_rows_x2";

const STEMS: [&str; 7] = [
    MQ6_STEM,
    MQ6_X4_STEM,
    Q8_K8_STEM,
    Q8_K320_STEM,
    Q8_K2560_STEM,
    Q8_WIDE_STEM,
    Q8_K2560_X2_STEM,
];

/// Private pinned module; resolve every required tile symbol before any case.
struct WideImage {
    _module: hip_bridge::Module,
    funcs: HashMap<String, hip_bridge::Function>,
    object_sha: String,
    container_sha: String,
}

impl WideImage {
    fn load(gpu: &Gpu) -> Result<Self> {
        let want = kernels::QWEN4_LANE_ROWS_PM_GFX1151_SHA256;
        let got = sha256_hex(kernels::QWEN4_LANE_ROWS_PM_GFX1151);
        if want.is_empty() {
            return Err("QWEN4_LANE_ROWS_PM_GFX1151_SHA256 is not pinned".into());
        }
        if got != want {
            return Err(format!(
                "embedded qwen4_lane_rows_pm image sha256 {got} is not the pinned {want}"
            ));
        }
        let device =
            rdna_compute::lab_extract_device_bytes(kernels::QWEN4_LANE_ROWS_PM_GFX1151, &gpu.arch)
                .ok_or("missing or ambiguous gfx1151 raw device image")?;
        let object_sha = sha256_hex(device);
        gpu.bind_thread().map_err(err)?;
        let module = gpu
            .hip
            .module_load_data(kernels::QWEN4_LANE_ROWS_PM_GFX1151)
            .map_err(err)?;
        let mut funcs = HashMap::new();
        let mut missing = Vec::new();
        for stem in STEMS {
            let symbol = wide_symbol(stem);
            match gpu.hip.module_get_function(&module, &symbol) {
                Ok(f) => {
                    funcs.insert(symbol, f);
                }
                Err(e) => missing.push(format!("{symbol} ({e})")),
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "embedded image {got} lacks wide symbols: {}",
                missing.join(", ")
            ));
        }
        Ok(Self {
            _module: module,
            funcs,
            object_sha,
            container_sha: got,
        })
    }

    /// Launch one total-R tile grid; ABI length is checked before submission.
    fn launch(
        &self,
        gpu: &Gpu,
        symbol: &str,
        grid: usize,
        rows: usize,
        block: u32,
        mut kb: KernargBlob,
        abi: usize,
    ) -> Result<()> {
        if kb.len() != abi {
            return Err(format!(
                "{symbol}: kernarg is {} bytes, ABI is {abi}",
                kb.len()
            ));
        }
        if grid == 0 || grid > u32::MAX as usize || !(1..=MAXB).contains(&rows) {
            return Err(format!("{symbol}: grid {grid} out of range"));
        }
        let func = self
            .funcs
            .get(symbol)
            .ok_or_else(|| format!("{symbol}: not loaded"))?;
        gpu.bind_thread().map_err(err)?;
        // The raw launch is not ordered against the Gpu's own stream.
        gpu.hip.device_synchronize().map_err(err)?;
        // SAFETY: the blob follows the symbol's ABI and addresses live allocations.
        unsafe {
            gpu.hip.launch_kernel_blob(
                func,
                [grid as u32, rows.div_ceil(8) as u32, 1],
                [block, 1, 1],
                0,
                None,
                kb.as_mut_slice(),
            )
        }
        .map_err(err)
    }
}

type Launch = Box<dyn Fn(&mut Gpu, usize) -> Result<()>>;
/// `wide(gpu, image, total_rows)`.
type WideLaunch = Box<dyn Fn(&mut Gpu, &WideImage, usize, usize) -> Result<()>>;

/// Real operands remain a distinct arm; provenance is not a synthetic fixture claim.
///
/// `mixed` preserves the original mixed caller for Q8 x2 captures whose source
/// singleton is `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair`: the two original
/// BF16 heights, its module/symbol/object SHA, and the homogeneous-group
/// component offsets (BF16 = 0, Q8 = 2) from the paired lab callbacks.
/// `None` for ordinary/pair captures.
struct MixedCaller {
    bf16_ms: [usize; 2],
    module: String,
    symbol: String,
    object_sha: String,
    container_sha: String,
}

struct Captured {
    name: String,
    family: String,
    form: String,
    ms: [usize; 4],
    k: usize,
    hc_act: bool,
    scale: Option<f32>,
    weights: [Vec<u8>; 4],
    x: Vec<f32>,
    singleton: (String, String, String, String),
    mixed: Option<MixedCaller>,
}

fn sha_pin(value: &serde_json::Value) -> Result<&str> {
    let pin = value.as_str().ok_or("missing SHA256 string")?;
    if pin.len() != 64 || !pin.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid SHA256 pin".into());
    }
    Ok(pin)
}

fn fixture_path(base: &Path, value: &serde_json::Value) -> Result<PathBuf> {
    let path = Path::new(value.as_str().ok_or("missing fixture path")?);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("operand/provenance path must stay relative to its fixture directory".into());
    }
    Ok(base.join(path))
}

fn pinned_blob(
    base: &Path,
    descriptor: &serde_json::Value,
    extent: Option<usize>,
) -> Result<Vec<u8>> {
    let path = fixture_path(base, &descriptor["path"])?;
    if let Some(bytes) = extent {
        if std::fs::metadata(&path).map_err(err)?.len() != bytes as u64 {
            return Err(format!("{}: byte extent != {bytes}", path.display()));
        }
    }
    let bytes = std::fs::read(&path).map_err(err)?;
    let pin = sha_pin(&descriptor["sha256"])?;
    if sha256_hex(&bytes) != pin {
        return Err(format!("{}: SHA256 does not match {pin}", path.display()));
    }
    Ok(bytes)
}

fn pin_artifact(
    base: &Path,
    value: &serde_json::Value,
    verified: &mut HashMap<PathBuf, String>,
) -> Result<()> {
    use std::io::Read as _;
    let raw = value["path"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing model/prompt path")?;
    let path = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        base.join(raw)
    };
    let pin = sha_pin(&value["sha256"])?;
    if let Some(previous) = verified.get(&path) {
        return if previous == pin {
            Ok(())
        } else {
            Err("conflicting artifact SHA pins".into())
        };
    }
    let mut file = std::fs::File::open(&path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(err)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if got != pin {
        return Err(format!(
            "{}: artifact SHA256 {got} != {pin}",
            path.display()
        ));
    }
    verified.insert(path, pin.to_string());
    Ok(())
}

fn load_capture(path: &Path, verified: &mut HashMap<PathBuf, String>) -> Result<Captured> {
    let manifest_bytes = std::fs::read(path).map_err(err)?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).map_err(err)?;
    if manifest["schema"] != "qwen4-rowtile-operands-v1"
        || manifest["arch"] != "gfx1151"
        || manifest["rows"] != MAXB as u64
    {
        return Err("capture requires qwen4-rowtile-operands-v1, exact gfx1151 and rows64".into());
    }
    let base = path.parent().unwrap_or(Path::new("."));
    let source_path = fixture_path(base, &manifest["provenance"]["source"])?;
    let source_bytes = std::fs::read(&source_path).map_err(err)?;
    let source_pin = sha_pin(&manifest["provenance"]["source_sha256"])?;
    if sha256_hex(&source_bytes) != source_pin {
        return Err("capture provenance SHA mismatch".into());
    }
    let source: serde_json::Value = serde_json::from_slice(&source_bytes).map_err(err)?;
    if source["schema"] != "qwen4-rowtile-source-v1"
        || source["capture_mode"] != "eager-liveprojection-operands-only"
        || source["retained_identity_claim"] != false
    {
        return Err("capture lacks actual eager live-projection provenance".into());
    }
    for key in [
        "name", "family", "form", "ms", "k", "rows", "weights", "x", "hc_act", "scale",
    ] {
        if manifest.get(key).is_none() || source.get(key) != manifest.get(key) {
            return Err(format!("capture/source binding differs at {key}"));
        }
    }
    let source_base = source_path.parent().unwrap_or(base);
    pin_artifact(source_base, &source["model"], verified)?;
    pin_artifact(source_base, &source["prompt"], verified)?;
    let name = manifest["name"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing capture name")?
        .to_string();
    let family = manifest["family"]
        .as_str()
        .ok_or("missing capture family")?
        .to_string();
    let form = manifest["form"]
        .as_str()
        .ok_or("missing capture form")?
        .to_string();
    let k = manifest["k"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0 && v <= i32::MAX as usize)
        .ok_or("invalid capture K")?;
    let heights = manifest["ms"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture ms needs four slots")?;
    let mut ms = [0usize; 4];
    for (m, value) in ms.iter_mut().zip(heights) {
        *m = value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v <= i32::MAX as usize)
            .ok_or("invalid capture M")?;
    }
    let hc_act = manifest["hc_act"]
        .as_bool()
        .ok_or("capture needs hc_act bool")?;
    let scale = if manifest["scale"].is_null() {
        None
    } else {
        let value = manifest["scale"].as_f64().ok_or("invalid HC scale")? as f32;
        if !value.is_finite() {
            return Err("nonfinite HC scale is not admitted".into());
        }
        Some(value)
    };
    if hc_act != scale.is_some() {
        return Err("HC activation/optional scale binding disagrees".into());
    }
    let single = ms[0] > 0 && ms[1..].iter().all(|&m| m == 0);
    let pair = ms[0] > 0 && ms[0] == ms[1] && ms[2..] == [0, 0];
    let admitted = match (family.as_str(), form.as_str()) {
        ("mq6", "single") => single && k.is_multiple_of(256),
        ("mq6", "x4") => ms.iter().any(|&m| m > 0) && k.is_multiple_of(256),
        ("q8", "single") => single && (k == 2560 || (k <= 1536 && k.is_multiple_of(32))),
        ("q8", "x2") => pair && k == 2560,
        ("q8", "hc_k8") => single && k.is_multiple_of(256),
        _ => false,
    };
    if !admitted || (form != "hc_k8" && (hc_act || scale.is_some())) {
        return Err(format!(
            "capture route not admitted here: {family}/{form} ms={ms:?} K={k}"
        ));
    }
    let captures = source["captures"]
        .as_array()
        .filter(|a| a.len() == MAXB)
        .ok_or("capture needs all64 row recipes")?;
    let start = captures[0]["position"]
        .as_u64()
        .ok_or("missing capture position")?;
    let occurrence = captures[0]["occurrence"]
        .as_u64()
        .ok_or("missing capture occurrence")?;
    for (row, recipe) in captures.iter().enumerate() {
        if recipe["row"].as_u64() != Some(row as u64)
            || recipe["position"].as_u64() != start.checked_add(row as u64)
            || recipe["occurrence"].as_u64() != Some(occurrence)
            || recipe["token"].as_u64().is_none_or(|v| v > u32::MAX as u64)
        {
            return Err(format!(
                "capture row {row} lacks stable sequential singleton binding"
            ));
        }
        for field in ["grid", "block"] {
            let dims = recipe[field]
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or("missing singleton launch geometry")?;
            if dims
                .iter()
                .any(|v| v.as_u64().is_none_or(|n| n == 0 || n > u32::MAX as u64))
                || recipe[field] != captures[0][field]
            {
                return Err("capture singleton launch geometry changed across rows".into());
            }
        }
        if recipe["shared_mem"].as_u64().is_none()
            || recipe["shared_mem"] != captures[0]["shared_mem"]
        {
            return Err("capture lacks stable singleton LDS recipe".into());
        }
    }
    let descriptors = manifest["weights"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture needs four weight descriptors")?;
    let bindings = source["weight_bindings"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture needs four immutable weight bindings")?;
    let mut weights: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
    for i in 0..4 {
        let bytes = ms[i]
            .checked_mul(k / if family == "mq6" { 256 } else { 32 })
            .and_then(|v| v.checked_mul(if family == "mq6" { 200 } else { 34 }))
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("capture weight byte extent overflow")?;
        if bindings[i]["bytes"].as_u64() != Some(bytes as u64)
            || bindings[i]["byte_offset"].as_u64().is_none()
            || !(bindings[i]["name"].is_null() || bindings[i]["name"].is_string())
            || !(bindings[i]["layer"].is_null() || bindings[i]["layer"].is_i64())
        {
            return Err("capture immutable weight binding/extent missing".into());
        }
        if ms[i] == 0 && bytes != 0 {
            return Err("capture zero slot must carry zero weight bytes".into());
        }
        weights[i] = pinned_blob(base, &descriptors[i], Some(bytes))?;
    }
    let x_bytes = k
        .checked_mul(MAXB)
        .and_then(|v| v.checked_mul(4))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("capture activation extent overflow")?;
    let bytes = pinned_blob(base, &manifest["x"], Some(x_bytes))?;
    let x = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte activation")))
        .collect();
    let singleton = &source["singleton"];
    let symbol = singleton["symbol"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing actual singleton symbol")?
        .to_string();
    let module = singleton["module"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing actual singleton module")?
        .to_string();
    let sha = sha_pin(&singleton["object_sha256"])?.to_string();
    let container_sha = sha_pin(&singleton["container_sha256"])?.to_string();
    // Original mixed caller metadata for captured Q8 x2: preserve the BF16
    // heights, module/symbol/object SHA, and component offsets (BF16 = 0,
    // Q8 = 2) instead of collapsing to a generic primitive. The ordinary
    // Q8-pair callers `gemv_q8_0_k2560_staged_pair` /
    // `gemv_q8_0_k2560_staged_rows_x2` need no extra mixed metadata.
    if family == "q8"
        && form == "x2"
        && symbol != "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair"
        && symbol != "gemv_q8_0_k2560_staged"
        && symbol != "gemv_q8_0_k2560_staged_pair"
        && symbol != "gemv_q8_0_k2560_staged_rows_x2"
        && symbol != "gemv_q8_0"
    {
        return Err(format!(
            "captured Q8 x2 source singleton {symbol} is not an admitted Q8-bearing caller"
        ));
    }
    let mixed = if family == "q8" && form == "x2" {
        let mixed_value = &source["mixed"];
        if symbol == "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair" {
            if module != "qwen4_decode_head_fuse" {
                return Err("mixed caller module must be qwen4_decode_head_fuse".into());
            }
            // Frozen producer convention (single, no alias): `mixed` is an
            // object {symbol, bf16_ms[2], bf16_weight_sha256[2]}; Q8 case
            // carries component_offset 2, BF16 subgroup 0. No bare `bf16_ms`.
            let _ = mixed_value.as_object().ok_or(
                "mixed caller needs frozen object mixed={symbol,bf16_ms,bf16_weight_sha256}",
            )?;
            if mixed_value["symbol"] != symbol {
                return Err("mixed.symbol must equal the actual mixed caller singleton".into());
            }
            let heights = mixed_value["bf16_ms"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("mixed caller needs original BF16 heights bf16_ms[2]")?;
            let mut bf16_ms = [0usize; 2];
            for (m, v) in bf16_ms.iter_mut().zip(heights.iter()) {
                *m = v
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n <= i32::MAX as usize)
                    .ok_or("invalid mixed BF16 height")?;
            }
            let weight_shas = mixed_value["bf16_weight_sha256"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("mixed caller needs BF16 weight SHAs bf16_weight_sha256[2]")?;
            for w in weight_shas {
                sha_pin(w)?;
            }
            if source["component_offset"] != 2 {
                return Err("mixed Q8 source-index component_offset must be 2".into());
            }
            let grid = captures[0]["grid"]
                .as_array()
                .ok_or("missing singleton launch grid")?;
            let grid0 = grid[0].as_u64().ok_or("invalid singleton launch grid")? as usize;
            if grid0 != bf16_ms[0] + bf16_ms[1] + 2 * ms[0] {
                return Err(format!(
                    "mixed caller grid M0+M1+2MQ mismatch: grid={grid0} M0={} M1={} MQ={}",
                    bf16_ms[0], bf16_ms[1], ms[0]
                ));
            }
            let block = captures[0]["block"]
                .as_array()
                .ok_or("missing singleton launch block")?;
            if block[0] != 32 || block[1] != 1 || block[2] != 1 {
                return Err("mixed caller block must be [32,1,1]".into());
            }
            println!("real-capture mixed_caller module={module} symbol={symbol} object_sha256={sha} container_sha256={container_sha} bf16_ms={bf16_ms:?} bf16_weight_sha256=[{},{}] component_offset_bf16=0 component_offset_q8=2 grid=[{grid0},1,1] block=[32,1,1]",
                weight_shas[0].as_str().unwrap_or("?"), weight_shas[1].as_str().unwrap_or("?"));
            Some(MixedCaller {
                bf16_ms,
                module: module.clone(),
                symbol: symbol.clone(),
                object_sha: sha.clone(),
                container_sha: container_sha.clone(),
            })
        } else {
            if !mixed_value.is_null() {
                return Err("ordinary Q8 x2 capture must not carry mixed caller metadata".into());
            }
            None
        }
    } else {
        None
    };
    println!("real-capture manifest={} manifest_sha256={} source_sha256={source_pin} model_sha256={} prompt_sha256={} source_singleton={module}/{symbol} object_sha256={sha} container_sha256={container_sha} retained_identity_claim=false",
        path.display(), sha256_hex(&manifest_bytes), source["model"]["sha256"], source["prompt"]["sha256"]);
    Ok(Captured {
        name,
        family,
        form,
        ms,
        k,
        hc_act,
        scale,
        weights,
        x,
        singleton: (symbol, module, sha, container_sha),
        mixed,
    })
}
fn captured_case(gpu: &mut Gpu, capture: &Captured) -> Result<Case> {
    let mut unused_rng = Rng(1);
    let mut case = match (capture.family.as_str(), capture.form.as_str()) {
        ("mq6", "single") => mq6_case_inputs(
            gpu,
            capture.ms[0],
            capture.k,
            &mut unused_rng,
            Some(capture),
        )?,
        ("mq6", "x4") => {
            mq6_x4_case_inputs(gpu, capture.ms, capture.k, &mut unused_rng, Some(capture))?
        }
        ("q8", "hc_k8") => q8_k8_case_inputs(
            gpu,
            capture.ms[0],
            capture.k,
            capture.hc_act,
            &mut unused_rng,
            Some(capture),
        )?,
        ("q8", "x2") => match &capture.mixed {
            Some(mixed) => q8_x2_mixed_case_inputs(
                gpu,
                mixed.bf16_ms,
                [capture.ms[0], capture.ms[1]],
                &mut unused_rng,
                Some(capture),
            )?,
            None if capture.singleton.0 == "gemv_q8_0_k2560_staged_pair"
                || capture.singleton.0 == "gemv_q8_0_k2560_staged_rows_x2" =>
            {
                q8_x2_pair_case_inputs(
                    gpu,
                    [capture.ms[0], capture.ms[1]],
                    &capture.singleton.0,
                    &mut unused_rng,
                    Some(capture),
                )?
            }
            None => q8_x2_case_inputs(
                gpu,
                [capture.ms[0], capture.ms[1]],
                &mut unused_rng,
                Some(capture),
            )?,
        },
        ("q8", "single") => {
            let kind = match capture.k {
                320 => Q8Kind::K320,
                2560 => Q8Kind::K2560,
                _ => Q8Kind::Wide,
            };
            q8_case_inputs(
                gpu,
                kind,
                capture.ms[0],
                capture.k,
                &mut unused_rng,
                Some(capture),
            )?
        }
        _ => return Err("unsupported captured family/form".into()),
    };
    case.name = format!("real-capture {} {}", capture.name, case.name);
    case.source_identity = Some(capture.singleton.clone());
    Ok(case)
}

/// Actual singleton, separate narrow PM/rows diagnostics and direct t8 candidate.
struct Case {
    name: String,
    outs: Vec<Out>,
    /// Packed weight bytes one launch reads.
    weight_bytes: usize,
    k: usize,
    /// Immutable inputs, checked bytewise after each launch arm.
    inputs: Vec<Rc<GpuTensor>>,
    /// Symbol the 1-row reference must launch (checked when known).
    ref_symbol: Option<&'static str>,
    tile_stem: &'static str,
    source_identity: Option<(String, String, String, String)>,
    single: Launch,
    rows: Option<Launch>,
    pm: Option<Launch>,
    wide: WideLaunch,
}

impl Case {
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.outs.iter().try_for_each(|o| o.reset(gpu))
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.outs.iter().map(|o| o.download(gpu)).collect()
    }
    fn input_bytes(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        gpu.hip.device_synchronize().map_err(err)?;
        self.inputs
            .iter()
            .map(|t| {
                let mut bytes = vec![0; t.buf.size()];
                gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).map_err(err)?;
                Ok(bytes)
            })
            .collect()
    }
    fn check_inputs(&self, gpu: &Gpu, expected: &[Vec<u8>]) -> Result<()> {
        if self.input_bytes(gpu)? != expected {
            return Err(format!(
                "{}: immutable weights/activations changed",
                self.name
            ));
        }
        Ok(())
    }
    fn poison_tail(&self, gpu: &Gpu, original: &[Vec<u8>], rows: usize) -> Result<Vec<Vec<u8>>> {
        let x = self.inputs.last().ok_or("missing activation")?;
        let mut poisoned = original.last().ok_or("missing activation bytes")?.clone();
        poisoned[rows * self.k * 4..MAXB * self.k * 4].fill(0xff);
        gpu.hip.memcpy_htod(&x.buf, &poisoned).map_err(err)?;
        let mut expected = original.to_vec();
        *expected.last_mut().ok_or("missing activation bytes")? = poisoned;
        Ok(expected)
    }
    fn restore_x(&self, gpu: &Gpu, original: &[Vec<u8>]) -> Result<()> {
        gpu.hip
            .memcpy_htod(
                &self.inputs.last().ok_or("missing activation")?.buf,
                original.last().ok_or("missing activation bytes")?,
            )
            .map_err(err)
    }
}

fn mq6_case(gpu: &mut Gpu, m: usize, k: usize, rng: &mut Rng) -> Result<Case> {
    mq6_case_inputs(gpu, m, k, rng, None)
}

fn mq6_case_inputs(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    let weights = capture.map_or_else(
        || Cow::Owned(gen_mq6(m, k, rng)),
        |c| Cow::Borrowed(c.weights[0].as_slice()),
    );
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(k, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    let (w3, x3, y3) = (w.clone(), x.clone(), y.clone());
    Ok(Case {
        name: format!("mq6 m={m} k={k}"),
        outs: vec![out],
        weight_bytes: weights.len(),
        k,
        inputs: vec![w.clone(), x.clone()],
        ref_symbol: None,
        tile_stem: MQ6_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            gpu.gemv_mq6g256v2(
                &w1,
                &x1.sub_offset(c * k, k),
                &y1.sub_offset(c * m, m),
                m,
                k,
            )
            .map_err(err)
        }),
        rows: Some(Box::new(move |gpu, b| {
            gpu.gemm_mq6g256v2_f32_rows(&w2, &x2, &y2, m, k, b)
                .map_err(err)
        })),
        pm: Some(Box::new(move |gpu, b| {
            gpu.gemv_mq6g256v2_xbatch_pm(&w, &x, &y, m, k, b)
                .map_err(err)
        })),
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(w3.buf.as_ptr());
            kb.push_ptr(x3.sub_offset(c * k, b * k).buf.as_ptr());
            kb.push_ptr(y3.sub_offset(c * m, b * m).buf.as_ptr());
            kb.push_i32(m as i32);
            kb.push_i32(k as i32);
            kb.push_i32(b as i32);
            img.launch(gpu, &wide_symbol(MQ6_STEM), m.div_ceil(8), b, 256, kb, 36)
        }),
    })
}

fn mq6_x4_case(gpu: &mut Gpu, ms: [usize; 4], k: usize, rng: &mut Rng) -> Result<Case> {
    mq6_x4_case_inputs(gpu, ms, k, rng, None)
}

fn mq6_x4_case_inputs(
    gpu: &mut Gpu,
    ms: [usize; 4],
    k: usize,
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    let mut ws = Vec::new();
    let mut outs = Vec::new();
    let mut weight_bytes = 0;
    for (i, &m) in ms.iter().enumerate() {
        // An unused slot still needs a valid pointer.
        let bytes = if m == 0 {
            Cow::Owned(vec![0u8; 256])
        } else {
            capture.map_or_else(
                || Cow::Owned(gen_mq6(m, k, rng)),
                |c| Cow::Borrowed(c.weights[i].as_slice()),
            )
        };
        weight_bytes += if m == 0 { 0 } else { bytes.len() };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
        outs.push(Out::new(gpu, m)?);
    }
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(k, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let ys: Vec<Rc<GpuTensor>> = outs.iter().map(|o| Rc::new(o.view())).collect();
    let (ws1, ys1, x1) = (ws.clone(), ys.clone(), x.clone());
    let (ws2, ys2, x2) = (ws.clone(), ys.clone(), x.clone());
    let (ws3, ys3, x3) = (ws.clone(), ys.clone(), x.clone());
    Ok(Case {
        name: format!("mq6 x4 ms={ms:?} k={k}"),
        outs,
        weight_bytes,
        k,
        inputs: ws.iter().cloned().chain([x.clone()]).collect(),
        ref_symbol: None,
        tile_stem: MQ6_X4_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            let row_ys: [GpuTensor; 4] =
                std::array::from_fn(|i| ys1[i].sub_offset(c * ms[i], ms[i]));
            let parts = std::array::from_fn(|i| (&*ws1[i], &row_ys[i], ms[i]));
            gpu.gemv_mq6g256v2_x4(parts, &x1.sub_offset(c * k, k), k)
                .map_err(err)
        }),
        rows: Some(Box::new(move |gpu, b| {
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*ws2[i], &*ys2[i], ms[i]));
            gpu.gemm_mq6g256v2_f32_rows_x4(parts, &x2, k, b)
                .map_err(err)
        })),
        pm: Some(Box::new(move |gpu, b| {
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*ws[i], &*ys[i], ms[i]));
            gpu.gemv_mq6g256v2_x4_xbatch_pm(parts, &x, k, b)
                .map_err(err)
        })),
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            for w in &ws3 {
                kb.push_ptr(w.buf.as_ptr());
            }
            kb.push_ptr(x3.sub_offset(c * k, b * k).buf.as_ptr());
            for (y, m) in ys3.iter().zip(ms) {
                kb.push_ptr(y.sub_offset(c * m, b * m).buf.as_ptr());
            }
            for m in ms {
                kb.push_i32(m as i32);
            }
            kb.push_i32(k as i32);
            kb.push_i32(b as i32);
            let grid: usize = ms.iter().map(|m| m.div_ceil(8)).sum();
            img.launch(gpu, &wide_symbol(MQ6_X4_STEM), grid, b, 256, kb, 96)
        }),
    })
}

/// `Gpu::gemv_q8_0_k8` is launched for the HC k8 projection at `K % 256 == 0`;
/// the narrow PM launcher (and the rows incumbent compared against it) is
/// admitted at `K % 2048 == 0` only.
fn q8_k8_case(gpu: &mut Gpu, m: usize, k: usize, hc_act: bool, rng: &mut Rng) -> Result<Case> {
    q8_k8_case_inputs(gpu, m, k, hc_act, rng, None)
}

fn q8_k8_case_inputs(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    hc_act: bool,
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    let weights = capture.map_or_else(
        || Cow::Owned(gen_q8(m, k, rng)),
        |c| Cow::Borrowed(c.weights[0].as_slice()),
    );
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(k, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let scale = capture.map_or(hc_act.then_some(HC_SCALE), |c| c.scale);
    let narrow = k.is_multiple_of(2048);
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    let (w3, x3, y3) = (w.clone(), x.clone(), y.clone());
    let rows: Option<Launch> = narrow.then(|| {
        Box::new(move |gpu: &mut Gpu, b: usize| {
            gpu.gemv_q8_0_k8_rows(&w2, &x2, &y2, m, k, scale, b)
                .map_err(err)
        }) as Launch
    });
    let pm: Option<Launch> = narrow.then(|| {
        Box::new(move |gpu: &mut Gpu, b: usize| {
            gpu.gemv_q8_0_k8_xbatch_pm(&w, &x, &y, m, k, scale, b)
                .map_err(err)
        }) as Launch
    });
    Ok(Case {
        name: format!("q8 k8 m={m} k={k} hc_act={}", u8::from(hc_act)),
        outs: vec![out],
        weight_bytes: weights.len(),
        k,
        inputs: vec![w1.clone(), x1.clone()],
        ref_symbol: Some("gemv_q8_0_k8"),
        tile_stem: Q8_K8_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            gpu.gemv_q8_0_k8(
                &w1,
                &x1.sub_offset(c * k, k),
                &y1.sub_offset(c * m, m),
                m,
                k,
                scale,
            )
            .map_err(err)
        }),
        rows,
        pm,
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(w3.buf.as_ptr());
            kb.push_ptr(x3.sub_offset(c * k, b * k).buf.as_ptr());
            kb.push_ptr(y3.sub_offset(c * m, b * m).buf.as_ptr());
            kb.push_i32(m as i32);
            kb.push_i32(k as i32);
            kb.push_i32(b as i32);
            kb.push_f32(scale.unwrap_or(1.0));
            kb.push_i32(i32::from(scale.is_some()));
            img.launch(gpu, &wide_symbol(Q8_K8_STEM), m.div_ceil(2), b, 512, kb, 44)
        }),
    })
}

/// Ordinary Q8_0 recipe of the wide symbol.
#[derive(Clone, Copy)]
enum Q8Kind {
    /// K = 320, ten blocks, one chain.
    K320,
    /// K = 2560, one chain.
    K2560,
    /// Small K (`<= 1536`, not 320): four chains, tail blocks to `a0..a2`.
    Wide,
}

impl Q8Kind {
    fn stem(self) -> &'static str {
        match self {
            Self::K320 => Q8_K320_STEM,
            Self::K2560 => Q8_K2560_STEM,
            Self::Wide => Q8_WIDE_STEM,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Self::K320 => "k320",
            Self::K2560 => "k2560",
            Self::Wide => "wide",
        }
    }
    fn admits(self, k: usize) -> bool {
        match self {
            Self::K320 => k == 320,
            Self::K2560 => k == 2560,
            Self::Wide => k != 320 && k.is_multiple_of(32) && (32..=1536).contains(&k),
        }
    }
}

/// The symbol `Gpu::gemv_q8_0` launches on gfx1151 at `k`.
fn q8_decode_symbol(k: usize) -> &'static str {
    match k {
        320 => "gemv_q8_0_k320_staged",
        640 => "gemv_q8_0_wide_k640_staged",
        2560 => "gemv_q8_0_k2560_staged",
        _ => "gemv_q8_0_wide",
    }
}

fn q8_case(gpu: &mut Gpu, kind: Q8Kind, m: usize, k: usize, rng: &mut Rng) -> Result<Case> {
    q8_case_inputs(gpu, kind, m, k, rng, None)
}

fn q8_case_inputs(
    gpu: &mut Gpu,
    kind: Q8Kind,
    m: usize,
    k: usize,
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    if !kind.admits(k) {
        return Err(format!("q8 {} fixture does not take k={k}", kind.tag()));
    }
    let weights = capture.map_or_else(
        || Cow::Owned(gen_q8(m, k, rng)),
        |c| Cow::Borrowed(c.weights[0].as_slice()),
    );
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(k, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    let (w3, x3, y3) = (w.clone(), x.clone(), y.clone());
    let (rows, pm): (Option<Launch>, Option<Launch>) = if matches!(kind, Q8Kind::K320) {
        (
            Some(Box::new(move |gpu, b| {
                gpu.gemv_q8_0_staged_rows(&w2, &x2, &y2, m, k, b)
                    .map_err(err)
            })),
            Some(Box::new(move |gpu, b| {
                gpu.gemv_q8_0_k320_xbatch_pm(&w, &x, &y, m, b).map_err(err)
            })),
        )
    } else {
        (None, None)
    };
    Ok(Case {
        name: format!("q8 {} m={m} k={k}", kind.tag()),
        outs: vec![out],
        weight_bytes: weights.len(),
        k,
        inputs: vec![w1.clone(), x1.clone()],
        ref_symbol: Some(q8_decode_symbol(k)),
        tile_stem: kind.stem(),
        source_identity: None,
        single: Box::new(move |gpu, c| {
            gpu.gemv_q8_0(
                &w1,
                &x1.sub_offset(c * k, k),
                &y1.sub_offset(c * m, m),
                m,
                k,
            )
            .map_err(err)
        }),
        rows,
        pm,
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(w3.buf.as_ptr());
            kb.push_ptr(x3.sub_offset(c * k, b * k).buf.as_ptr());
            kb.push_ptr(y3.sub_offset(c * m, b * m).buf.as_ptr());
            kb.push_i32(m as i32);
            kb.push_i32(k as i32);
            kb.push_i32(b as i32);
            img.launch(
                gpu,
                &wide_symbol(kind.stem()),
                m.div_ceil(8),
                b,
                256,
                kb,
                36,
            )
        }),
    })
}

/// Two K = 2560 Q8_0 matrices (shared gate/up) reading one `x`; an unused slot
/// carries `m = 0`.
fn q8_x2_case(gpu: &mut Gpu, ms: [usize; 2], rng: &mut Rng) -> Result<Case> {
    q8_x2_case_inputs(gpu, ms, rng, None)
}

fn q8_x2_case_inputs(
    gpu: &mut Gpu,
    ms: [usize; 2],
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    const K: usize = 2560;
    let mut ws = Vec::new();
    let mut outs = Vec::new();
    let mut weight_bytes = 0;
    for (i, &m) in ms.iter().enumerate() {
        let bytes = if m == 0 {
            Cow::Owned(vec![0u8; 256])
        } else {
            capture.map_or_else(
                || Cow::Owned(gen_q8(m, K, rng)),
                |c| Cow::Borrowed(c.weights[i].as_slice()),
            )
        };
        weight_bytes += if m == 0 { 0 } else { bytes.len() };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
        outs.push(Out::new(gpu, m)?);
    }
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(K, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let ys: Vec<Rc<GpuTensor>> = outs.iter().map(|o| Rc::new(o.view())).collect();
    let (ws1, ys1, x1) = (ws.clone(), ys.clone(), x.clone());
    Ok(Case {
        name: format!("q8 k2560 x2 ms={ms:?}"),
        outs,
        weight_bytes,
        k: K,
        inputs: ws.iter().cloned().chain([x.clone()]).collect(),
        ref_symbol: Some(q8_decode_symbol(K)),
        tile_stem: Q8_K2560_X2_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            for i in 0..2 {
                if ms[i] > 0 {
                    let (m, yv) = (ms[i], &ys1[i]);
                    gpu.gemv_q8_0(
                        &ws1[i],
                        &x1.sub_offset(c * K, K),
                        &yv.sub_offset(c * m, m),
                        m,
                        K,
                    )
                    .map_err(err)?;
                }
            }
            Ok(())
        }),
        rows: None,
        pm: None,
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(ws[0].buf.as_ptr());
            kb.push_ptr(ws[1].buf.as_ptr());
            kb.push_ptr(x.sub_offset(c * K, b * K).buf.as_ptr());
            kb.push_ptr(ys[0].sub_offset(c * ms[0], b * ms[0]).buf.as_ptr());
            kb.push_ptr(ys[1].sub_offset(c * ms[1], b * ms[1]).buf.as_ptr());
            kb.push_i32(ms[0] as i32);
            kb.push_i32(ms[1] as i32);
            kb.push_i32(K as i32);
            kb.push_i32(b as i32);
            let grid = ms[0].div_ceil(8) + ms[1].div_ceil(8);
            img.launch(gpu, &wide_symbol(Q8_K2560_X2_STEM), grid, b, 256, kb, 56)
        }),
    })
}
/// Captured mixed-decode Q8 subgroup: the reference launches the actual mixed
/// caller `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair` (module
/// `qwen4_decode_head_fuse`, ABI88: nine pointers@0..64, M0/M1/MQ/K@72/76/80/84;
/// grid `[M0+M1+2*MQ,1,1]`, block 32) on the real Q8 subgroup with the original
/// BF16 heights `M0/M1` (BF16 subgroup zeroed) and compares actual mixed
/// arithmetic on the two Q8 outputs -- not metadata identity plus a generic
/// primitive substitution. `bf16_ms` are the preserved original mixed heights;
/// `ms` are the two captured Q8 heights. Frozen single-launch convention: only
/// the Q8 subgroup launches (`M0 = M1 = 0` with proper ABI and dummy guarded
/// BF16 arrays); no mirrored `MQ = 0` BF16-half relaunch -- the BF16 half is
/// covered by its own bf16/x2 captured case (producer emits both sides with
/// backfilled `mixed`). Branch independence per
/// `qwen4_decode_head_fuse.hip:185-201` (M0+M1 rows run the BF16 body, the
/// next 2*MQ rows the Q8 staged body). Missing
/// `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair` lab wrapper is owned by slice G;
/// request it via the integration owner, do not add one.
fn q8_x2_mixed_case_inputs(
    gpu: &mut Gpu,
    bf16_ms: [usize; 2],
    ms: [usize; 2],
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    const K: usize = 2560;
    const MIXED: &str = "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair";
    let mut ws = Vec::new();
    let mut outs = Vec::new();
    let mut weight_bytes = 0;
    for (i, &m) in ms.iter().enumerate() {
        let bytes = if m == 0 {
            Cow::Owned(vec![0u8; 256])
        } else {
            capture.map_or_else(
                || Cow::Owned(gen_q8(m, K, rng)),
                |c| Cow::Borrowed(c.weights[i].as_slice()),
            )
        };
        weight_bytes += if m == 0 { 0 } else { bytes.len() };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
        outs.push(Out::new(gpu, m)?);
    }
    // Dummy guarded BF16 subgroup arrays: real allocations so the mixed ABI
    // stays intact while the Q8 subgroup runs with M0 = M1 = 0 (kernel writes
    // no BF16 rows, but pointers must be live). Dummy views are persistent
    // full-tensor guards so immutability checks see them.
    let mut bf16_ws = Vec::new();
    let mut bf16_ys = Vec::new();
    for &m in bf16_ms.iter() {
        let bytes = vec![0u8; m.max(1) * K * 2];
        let w = Rc::new(gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?);
        let y = Rc::new(gpu.zeros(&[m.max(1)], DType::F32).map_err(err)?);
        bf16_ws.push(w);
        bf16_ys.push(y);
    }
    if ms[0] != ms[1] || ms[0] == 0 {
        return Err(format!(
            "mixed Q8 subgroup needs equal nonzero pair heights, got {ms:?}"
        ));
    }
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(K, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let ys: Vec<Rc<GpuTensor>> = outs.iter().map(|o| Rc::new(o.view())).collect();
    let (ws1, ys1, x1) = (ws.clone(), ys.clone(), x.clone());
    let (bf16_ws1, bf16_ys1) = (bf16_ws.clone(), bf16_ys.clone());
    Ok(Case {
        name: format!("q8 k2560 mixed x2 bf16_ms={bf16_ms:?} q8_ms={ms:?}"),
        outs,
        weight_bytes,
        k: K,
        inputs: ws
            .iter()
            .cloned()
            .chain(bf16_ws.iter().cloned())
            .chain([x.clone()])
            .collect(),
        ref_symbol: Some(MIXED),
        tile_stem: Q8_K2560_X2_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            gpu.gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
                [
                    (&bf16_ws1[0], &bf16_ys1[0], 0),
                    (&bf16_ws1[1], &bf16_ys1[1], 0),
                ],
                [
                    (&ws1[0], &ys1[0].sub_offset(c * ms[0], ms[0])),
                    (&ws1[1], &ys1[1].sub_offset(c * ms[1], ms[1])),
                ],
                ms[0],
                &x1.sub_offset(c * K, K),
                K,
            )
            .map_err(err)?;
            Ok(())
        }),
        rows: None,
        pm: None,
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(ws[0].buf.as_ptr());
            kb.push_ptr(ws[1].buf.as_ptr());
            kb.push_ptr(x.sub_offset(c * K, b * K).buf.as_ptr());
            kb.push_ptr(ys[0].sub_offset(c * ms[0], b * ms[0]).buf.as_ptr());
            kb.push_ptr(ys[1].sub_offset(c * ms[1], b * ms[1]).buf.as_ptr());
            kb.push_i32(ms[0] as i32);
            kb.push_i32(ms[1] as i32);
            kb.push_i32(K as i32);
            kb.push_i32(b as i32);
            let grid = ms[0].div_ceil(8) + ms[1].div_ceil(8);
            img.launch(gpu, &wide_symbol(Q8_K2560_X2_STEM), grid, b, 256, kb, 56)
        }),
    })
}

/// Captured Q8-pair caller: the reference launches the actual pair arithmetic
/// (`gemv_q8_0_k2560_staged_pair` one-launch pair, or the per-matrix
/// `gemv_q8_0` fallback for other admitted Q8-bearing symbols), never a
/// metadata-only check. `caller` is the preserved source singleton symbol.
fn q8_x2_pair_case_inputs(
    gpu: &mut Gpu,
    ms: [usize; 2],
    caller: &str,
    rng: &mut Rng,
    capture: Option<&Captured>,
) -> Result<Case> {
    const K: usize = 2560;
    let mut ws = Vec::new();
    let mut outs = Vec::new();
    let mut weight_bytes = 0;
    for (i, &m) in ms.iter().enumerate() {
        let bytes = if m == 0 {
            Cow::Owned(vec![0u8; 256])
        } else {
            capture.map_or_else(
                || Cow::Owned(gen_q8(m, K, rng)),
                |c| Cow::Borrowed(c.weights[i].as_slice()),
            )
        };
        weight_bytes += if m == 0 { 0 } else { bytes.len() };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
        outs.push(Out::new(gpu, m)?);
    }
    let xs = capture.map_or_else(
        || Cow::Owned(gen_x(K, rng)),
        |c| Cow::Borrowed(c.x.as_slice()),
    );
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let ys: Vec<Rc<GpuTensor>> = outs.iter().map(|o| Rc::new(o.view())).collect();
    let (ws1, ys1, x1) = (ws.clone(), ys.clone(), x.clone());
    let pair_launch = caller == "gemv_q8_0_k2560_staged_pair";
    let ref_symbol: &'static str = if pair_launch {
        "gemv_q8_0_k2560_staged_pair"
    } else {
        "gemv_q8_0_k2560_staged"
    };
    Ok(Case {
        name: format!("q8 k2560 pair x2 caller={caller} ms={ms:?}"),
        outs,
        weight_bytes,
        k: K,
        inputs: ws.iter().cloned().chain([x.clone()]).collect(),
        ref_symbol: Some(ref_symbol),
        tile_stem: Q8_K2560_X2_STEM,
        source_identity: None,
        single: Box::new(move |gpu, c| {
            if pair_launch {
                if ms[0] == 0 || ms[1] == 0 {
                    return Err("actual Q8 pair launch needs two nonzero heights".into());
                }
                gpu.gemv_q8_0_pair(
                    &ws1[0],
                    &ws1[1],
                    &x1.sub_offset(c * K, K),
                    &ys1[0].sub_offset(c * ms[0], ms[0]),
                    &ys1[1].sub_offset(c * ms[1], ms[1]),
                    ms[0],
                    K,
                )
                .map_err(err)?;
            } else {
                for i in 0..2 {
                    if ms[i] > 0 {
                        let (m, yv) = (ms[i], &ys1[i]);
                        gpu.gemv_q8_0(
                            &ws1[i],
                            &x1.sub_offset(c * K, K),
                            &yv.sub_offset(c * m, m),
                            m,
                            K,
                        )
                        .map_err(err)?;
                    }
                }
            }
            Ok(())
        }),
        rows: None,
        pm: None,
        wide: Box::new(move |gpu, img, b, c| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(ws[0].buf.as_ptr());
            kb.push_ptr(ws[1].buf.as_ptr());
            kb.push_ptr(x.sub_offset(c * K, b * K).buf.as_ptr());
            kb.push_ptr(ys[0].sub_offset(c * ms[0], b * ms[0]).buf.as_ptr());
            kb.push_ptr(ys[1].sub_offset(c * ms[1], b * ms[1]).buf.as_ptr());
            kb.push_i32(ms[0] as i32);
            kb.push_i32(ms[1] as i32);
            kb.push_i32(K as i32);
            kb.push_i32(b as i32);
            let grid = ms[0].div_ceil(8) + ms[1].div_ceil(8);
            img.launch(gpu, &wide_symbol(Q8_K2560_X2_STEM), grid, b, 256, kb, 56)
        }),
    })
}

/// `(mismatching byte count, first mismatching index)`.
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

#[derive(Default)]
struct Tally {
    cases: usize,
    launches: usize,
    wide_launches: usize,
    pm_vs_single: usize,
    pm_vs_rows: usize,
    rows_vs_single: usize,
    wide_vs_single: usize,
    wide_vs_pm_c1: usize,
}

impl Tally {
    fn failed(&self) -> bool {
        self.pm_vs_single
            + self.pm_vs_rows
            + self.rows_vs_single
            + self.wide_vs_single
            + self.wide_vs_pm_c1
            > 0
    }
}

/// Diff every output of `got` against `want`; the first difference is described.
fn compare(case: &Case, got: &[Vec<u8>], want: &[Vec<u8>]) -> (usize, Option<String>) {
    let mut bad = 0;
    let mut first = None;
    for (i, ((g, w), out)) in got.iter().zip(want).zip(&case.outs).enumerate() {
        let (n, at) = diff(g, w);
        bad += n;
        if let (None, Some(at)) = (&first, at) {
            first = Some(format!(
                "matrix {i} {} (got {:#04x}, want {:#04x})",
                out.locate(at),
                g[at],
                w[at]
            ));
        }
    }
    (bad, first)
}

fn verify(
    gpu: &mut Gpu,
    img: &WideImage,
    pins: &HashMap<String, String>,
    case: &Case,
    tally: &mut Tally,
) -> Result<()> {
    let original_inputs = case.input_bytes(gpu)?;
    case.reset(gpu)?;
    let mut ref_symbol = String::from("<none>");
    for c in 0..MAXB {
        (case.single)(gpu, c)?;
        let symbol = gpu
            .last_launched_kernel()
            .ok_or("singleton route not recorded")?;
        let (module, container_sha, sha) = gpu.lab_kernel_device_identity(symbol).map_err(err)?;
        if let Some((want_symbol, want_module, want_sha, want_container)) = &case.source_identity {
            if symbol != want_symbol
                || &module != want_module
                || &sha != want_sha
                || &container_sha != want_container
            {
                return Err(format!("{}: actual singleton {module}/{symbol} object_sha256={sha} container_sha256={container_sha} differs from captured source {want_module}/{want_symbol} object_sha256={want_sha} container_sha256={want_container}", case.name));
            }
        }
        let pin = pins
            .get(symbol)
            .ok_or_else(|| format!("{}: singleton {symbol} has no reviewed SHA pin", case.name))?;
        if &sha != pin {
            return Err(format!(
                "{}: singleton {module}/{symbol} SHA {sha} != pinned {pin}",
                case.name
            ));
        }
        if c == 0 {
            ref_symbol = symbol.to_string();
            println!(
                "{}: actual singleton {module}/{symbol} object_sha256={sha} container_sha256={container_sha}",
                case.name
            );
        } else if symbol != ref_symbol {
            return Err(format!(
                "{}: singleton row {c} changed route to {symbol}",
                case.name
            ));
        }
    }
    if let Some(want) = case.ref_symbol {
        if ref_symbol != want {
            return Err(format!(
                "{}: the 1-row reference launched {ref_symbol}, not the decode object {want}",
                case.name
            ));
        }
    }
    let reference = case.download(gpu)?;
    let guarded_reference: Vec<Vec<u8>> = case
        .outs
        .iter()
        .zip(&reference)
        .map(|(out, bytes)| out.expected(bytes, MAXB))
        .collect();
    let (guard_bad, guard_first) = compare(case, &reference, &guarded_reference);
    if guard_bad != 0 {
        return Err(format!(
            "{}: actual singleton corrupted guards: {guard_first:?}",
            case.name
        ));
    }
    case.check_inputs(gpu, &original_inputs)?;
    // Independent PM total-R=1 calls are a separate hardware arm, not AR.
    case.reset(gpu)?;
    for c in 0..MAXB {
        (case.wide)(gpu, img, 1, c)?;
    }
    let pm_c1 = case.download(gpu)?;
    let (c1_bad, c1_first) = compare(case, &pm_c1, &reference);
    println!(
        "PM t8 C1 case={:?} symbol={} object_sha256={} container_sha256={} independent_rows={MAXB} mismatch_bytes={c1_bad}",
        case.name,
        wide_symbol(case.tile_stem),
        img.object_sha,
        img.container_sha
    );
    tally.pm_vs_single += c1_bad;
    tally.launches += MAXB;
    if let Some(first) = c1_first {
        eprintln!(
            "{}: PM t8 C1 differs from actual singleton at {first}",
            case.name
        );
    }
    case.check_inputs(gpu, &original_inputs)?;
    let mut line = format!("{:<44}", case.name);
    let mut bad_case = c1_bad;

    // Narrow PM (and the incumbent rows kernel) over B = 1..=8.
    if let Some(pm) = &case.pm {
        for b in 1..=NARROW_MAX {
            let expected: Vec<Vec<u8>> = case
                .outs
                .iter()
                .zip(&reference)
                .map(|(o, r)| o.expected(r, b))
                .collect();
            case.reset(gpu)?;
            pm(gpu, b)?;
            let got_pm = case.download(gpu)?;
            case.check_inputs(gpu, &original_inputs)?;
            let (bad, first) = compare(case, &got_pm, &expected);
            tally.pm_vs_single += bad;
            tally.launches += 1;
            let mut note = String::new();
            if let Some(first) = first {
                note.push_str(&format!(
                    " B={b}: PM differs from 1-row incumbent at {first};"
                ));
            }
            if let (true, Some(rows)) = (b >= 2, &case.rows) {
                case.reset(gpu)?;
                rows(gpu, b)?;
                let got_rows = case.download(gpu)?;
                case.check_inputs(gpu, &original_inputs)?;
                let (rows_bad, rows_first) = compare(case, &got_rows, &expected);
                let (pm_rows_bad, pm_rows_first) = compare(case, &got_pm, &got_rows);
                tally.rows_vs_single += rows_bad;
                tally.pm_vs_rows += pm_rows_bad;
                tally.launches += 1;
                if let Some(first) = rows_first {
                    note.push_str(&format!(
                        " B={b}: incumbent rows differs from 1-row incumbent at {first};"
                    ));
                }
                if let Some(first) = pm_rows_first {
                    note.push_str(&format!(
                        " B={b}: PM differs from incumbent rows at {first};"
                    ));
                }
                bad_case += rows_bad + pm_rows_bad;
            }
            bad_case += bad;
            line.push_str(&note);
        }
    }

    // Every total R, with NaN-poisoned inactive activations and guarded outputs.
    let mut wide_launches = 0;
    for b in 1..=MAXB {
        let expected: Vec<Vec<u8>> = case
            .outs
            .iter()
            .zip(&reference)
            .map(|(o, r)| o.expected(r, b))
            .collect();
        let poisoned_inputs = case.poison_tail(gpu, &original_inputs, b)?;
        case.reset(gpu)?;
        (case.wide)(gpu, img, b, 0)?;
        let got = case.download(gpu)?;
        case.check_inputs(gpu, &poisoned_inputs)?;
        case.restore_x(gpu, &original_inputs)?;
        let (bad, first) = compare(case, &got, &expected);
        let expected_c1: Vec<Vec<u8>> = case
            .outs
            .iter()
            .zip(&pm_c1)
            .map(|(out, bytes)| out.expected(bytes, b))
            .collect();
        let (pm_bad, pm_first) = compare(case, &got, &expected_c1);
        tally.wide_vs_pm_c1 += pm_bad;
        bad_case += pm_bad;
        if let Some(first) = pm_first {
            line.push_str(&format!(
                " B={b}: tile8 differs from separate PM C1 at {first};"
            ));
        }
        println!("rowtile case={:?} module=qwen4_lane_rows_pm symbol={} object_sha256={} container_sha256={} R={b} grid_y={} output_bytes={:?} mismatch_bytes={bad}",
            case.name, wide_symbol(case.tile_stem), img.object_sha, img.container_sha,
            b.div_ceil(8), case.outs.iter().map(Out::bytes).collect::<Vec<_>>());
        tally.wide_vs_single += bad;
        tally.launches += 1;
        tally.wide_launches += 1;
        wide_launches += 1;
        bad_case += bad;
        if let Some(first) = first {
            line.push_str(&format!(
                " B={b}: tile8 differs from actual singleton at {first};"
            ));
        }
    }
    tally.cases += 1;
    let mut scope = Vec::new();
    if case.pm.is_some() {
        scope.push(format!("narrow B=1..{NARROW_MAX}"));
    }
    scope.push(format!("wide {wide_launches} launches"));
    println!(
        "{line} [ref {ref_symbol}] {}",
        if bad_case == 0 {
            format!("ok ({})", scope.join(", "))
        } else {
            format!("MISMATCH ({bad_case} bytes)")
        }
    );
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_us(gpu: &mut Gpu, n: usize, f: &dyn Fn(&mut Gpu) -> Result<()>) -> Result<f64> {
    let (start, stop) = (
        gpu.hip.event_create().map_err(err)?,
        gpu.hip.event_create().map_err(err)?,
    );
    for _ in 0..5 {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        gpu.hip.event_record(&start, None).map_err(err)?;
        f(gpu)?;
        gpu.hip.event_record(&stop, None).map_err(err)?;
        gpu.hip.event_synchronize(&stop).map_err(err)?;
        samples.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1000.0);
    }
    gpu.hip.event_destroy(start).map_err(err)?;
    gpu.hip.event_destroy(stop).map_err(err)?;
    Ok(median(samples))
}

fn bench(gpu: &mut Gpu, case: &Case, n: usize) -> Result<()> {
    let (Some(pm_launch), Some(rows_launch)) = (&case.pm, &case.rows) else {
        return Err(format!(
            "{}: bench needs the narrow PM and rows routes",
            case.name
        ));
    };
    for b in [4, 8] {
        let pm = time_us(gpu, n, &|g| pm_launch(g, b))?;
        let rows = time_us(gpu, n, &|g| rows_launch(g, b))?;
        println!(
            "{:<44} B={b} PM {pm:>9.1} us  incumbent rows {rows:>9.1} us  x{:.2}  PM weights {:.0} GB/s",
            case.name,
            rows / pm,
            case.weight_bytes as f64 / (pm * 1e-6) / 1e9
        );
    }
    Ok(())
}

/// Every verification shape of the contract.
fn verify_cases(gpu: &mut Gpu, mut seed: u64) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    // MQ6: odd M, M = 48, K tails mod four groups (K/256 = 1, 2, 3, 5, 10, 24).
    for (m, k) in [
        (1, 256),
        (5, 2560),
        (13, 2560),
        (2560, 6144),
        (4099, 2560),
        (9, 768),
        (17, 1280),
        (33, 512),
        (48, 2560),
    ] {
        cases.push(mq6_case(gpu, m, k, &mut next_rng())?);
    }
    for m in [2, 3, 7, 8] {
        cases.push(mq6_case(gpu, m, 256, &mut next_rng())?);
    }
    for ms in [
        [10240, 6144, 48, 48],
        [12288, 640, 512, 512],
        [3, 0, 5, 0],
        [7, 9, 15, 33],
        [0, 17, 0, 1],
    ] {
        cases.push(mq6_x4_case(gpu, ms, 2560, &mut next_rng())?);
    }
    cases.push(mq6_x4_case(gpu, [17, 5, 48, 9], 1280, &mut next_rng())?);
    // Every nonempty x4 slot mask, differing heights and a ragged output tile.
    for mask in 1u32..16 {
        let ms = std::array::from_fn(|i| {
            if mask & (1 << i) != 0 {
                [1, 2, 7, 9][i]
            } else {
                0
            }
        });
        cases.push(mq6_x4_case(gpu, ms, 768, &mut next_rng())?);
    }
    // Q8 HC k8: K % 2048 == 0 (narrow PM too) and K/8 partitions with a short final tile.
    for (m, k) in [(320, 10240), (7, 2048), (7, 2304), (5, 2560), (13, 512)] {
        for hc_act in [false, true] {
            cases.push(q8_k8_case(gpu, m, k, hc_act, &mut next_rng())?);
        }
    }
    // Q8 K = 320 (narrow PM too).
    for m in [10240, 17, 1, 9, 33] {
        cases.push(q8_case(gpu, Q8Kind::K320, m, 320, &mut next_rng())?);
    }
    // Q8 K = 2560, single chain.
    for m in [1, 13, 17, 2560, 4099] {
        cases.push(q8_case(gpu, Q8Kind::K2560, m, 2560, &mut next_rng())?);
    }
    for ms in [[2560, 2560], [7, 9], [17, 1], [8, 9], [7, 0], [0, 3]] {
        cases.push(q8_x2_case(gpu, ms, &mut next_rng())?);
    }
    // Q8 four-chain small K: K = 640 (shared-expert down) and block tails.
    for m in [2560, 17, 1, 9] {
        cases.push(q8_case(gpu, Q8Kind::Wide, m, 640, &mut next_rng())?);
    }
    for k in [96, 160, 192, 224, 352, 480, 1248, 1344] {
        for m in [13, 9] {
            cases.push(q8_case(gpu, Q8Kind::Wide, m, k, &mut next_rng())?);
        }
    }
    for m in [1, 2, 3, 7, 8, 9] {
        for (kind, k) in [
            (Q8Kind::K320, 320),
            (Q8Kind::K2560, 2560),
            (Q8Kind::Wide, 96),
        ] {
            cases.push(q8_case(gpu, kind, m, k, &mut next_rng())?);
        }
    }
    Ok(cases)
}

fn run() -> Result<bool> {
    const USAGE: &str = "usage: lane_rows_pm_oracle --singleton-pins pins.json [--capture manifest.json ...] [--seed N] [--bench N]";
    let mut bench_n = None;
    let mut pins_path = None;
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut capture_paths = Vec::<PathBuf>::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--bench" => {
                bench_n = Some(
                    args.next()
                        .ok_or("--bench needs N")?
                        .parse::<usize>()
                        .map_err(err)?,
                )
            }
            "--singleton-pins" => {
                pins_path = Some(args.next().ok_or("--singleton-pins needs a JSON file")?);
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or("--seed needs N")?
                    .parse::<u64>()
                    .map_err(err)?;
            }
            "--capture" => {
                capture_paths.push(PathBuf::from(
                    args.next().ok_or("--capture needs manifest.json")?,
                ));
            }
            other => return Err(format!("unknown argument {other} ({USAGE})")),
        }
    }
    let path = pins_path.ok_or(USAGE)?;
    let pin_bytes = std::fs::read(&path).map_err(err)?;
    let pins: HashMap<String, String> = serde_json::from_slice(&pin_bytes).map_err(err)?;
    if pins.is_empty()
        || pins
            .values()
            .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("singleton pins must be a nonempty symbol-to-SHA256 JSON map".into());
    }
    println!("singleton pins: {path} sha256={}", sha256_hex(&pin_bytes));
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch.as_str() != "gfx1151" {
        return Err(format!("needs exact gfx1151, found {}", gpu.arch.as_str()));
    }
    gpu.lane_rows_scope = false;
    if gpu.lane_rows_pm_enabled() {
        return Err("the incumbents must run unrouted (lane_rows_scope is off)".into());
    }
    let img = WideImage::load(&gpu)?;
    println!(
        "tile8 image: {} symbols, logical rows 1..={MAXB}, object_sha256={} container_sha256={}",
        img.funcs.len(),
        img.object_sha,
        img.container_sha
    );
    let mut tally = Tally::default();
    println!("finite hardware fixtures: seed={seed}");
    for case in verify_cases(&mut gpu, seed)? {
        verify(&mut gpu, &img, &pins, &case, &mut tally)?;
    }
    let mut verified_artifacts = HashMap::new();
    for path in &capture_paths {
        let capture = load_capture(path, &mut verified_artifacts)?;
        let case = captured_case(&mut gpu, &capture)?;
        verify(&mut gpu, &img, &pins, &case, &mut tally)?;
    }
    println!(
        "{} cases, {} launches ({} wide): wide vs 1-row incumbent {} bad bytes, PM vs 1-row incumbent {} bad bytes, PM vs incumbent rows {} bad bytes, incumbent rows vs 1-row {} bad bytes",
        tally.cases,
        tally.launches,
        tally.wide_launches,
        tally.wide_vs_single,
        tally.pm_vs_single,
        tally.pm_vs_rows,
        tally.rows_vs_single
    );
    println!("tile8 vs separate PM C1: {} bad bytes", tally.wide_vs_pm_c1);
    if tally.failed() {
        return Ok(false);
    }
    if let Some(n) = bench_n.filter(|&n| n > 0) {
        let mut rng = Rng(0x00c8_0c8c_8c8c_8c8d);
        // The C8 lane-step shapes (DAG.md section 5), one case alive at a time.
        let case = mq6_x4_case(&mut gpu, [12288, 640, 512, 512], 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_x4_case(&mut gpu, [10240, 6144, 48, 48], 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_case(&mut gpu, 248_320, 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_case(&mut gpu, 2560, 6144, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = q8_k8_case(&mut gpu, 320, 10240, true, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = q8_case(&mut gpu, Q8Kind::K320, 10240, 320, &mut rng)?;
        bench(&mut gpu, &case, n)?;
    }
    Ok(!tally.failed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("lane_rows_pm_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("lane_rows_pm_oracle: {e}");
            std::process::exit(2);
        }
    }
}
