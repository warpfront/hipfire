// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict MQ6 G1 hardware byte gate, not a CPU proof or a throughput receipt.
//!
//! `mq6_wide_rows_oracle [--seed INTEGER] --captured MANIFEST [--captured MANIFEST ...]`
//! always runs two deterministic seeds, every R=1..64, and original, reversed,
//! and stable-address changed-input sweeps. The reference is the actual runtime
//! singleton, with grouped singleton checked against per-matrix singleton.
//! Direct PM fixed-eight-row tiles run at every width; public rows launchers
//! run at R=2..64 and must reach the tile above eight. No diagnostic failures,
//! partial-width success, guessed object identities, or architecture aliases.
//!
//! X has a nonzero view offset, NaN prefix/suffix and NaN inactive rows at EVERY
//! width. Output guards/inactive rows and immutable X/weight bytes are checked.
//! Finite signed-zero, subnormal, cancellation and exponent operands are covered;
//! explicit extreme-overflow fixtures are separate, not an arbitrary NaN-domain
//! claim. All-zero x4 masks are guard-only direct-launch refusal cases, not
//! arithmetic/route coverage. Receipts identify actual loaded module/object SHA.
//! At least one SHA-pinned `qwen4-rowtile-operands-v1` MQ6 capture is required;
//! generated-only runs cannot complete G1. Capture source-index, shape, blobs,
//! 64 eager-liveprojection row records and loaded singleton identity must agree.
//! This is component-only eager operand provenance, never a retained/full-state
//! claim. Original captured weights/X are used unchanged; alternate weights
//! and replay X are explicitly synthetic negative-control/stress operands.
//! Exit 1 means byte/reference/negative-control failure; exit 2 means missing
//! device, architecture, scope, symbol, identity, or other runtime prerequisite.

use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::rc::Rc;

#[derive(Debug)]
struct Failure {
    code: i32,
    message: String,
    identity: Option<Identity>,
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self {
            code: 2,
            message,
            identity: None,
        }
    }
}

type Result<T, E = Failure> = std::result::Result<T, E>;

fn mismatch(message: String) -> Failure {
    Failure {
        code: 1,
        message,
        identity: None,
    }
}

const MAXR: usize = 64;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
/// Output sentinel; live NaN results are still compared by their exact bytes.
const SENTINEL: u8 = 0xff;
/// NaN activation rows before and after the `[MAXR, K]` X view.
const PRE_ROWS: usize = 3;
/// Per-row magnitudes cycle; `0.0` is the all-zero (+-0) row.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

fn err<E: std::fmt::Display>(e: E) -> Failure {
    e.to_string().into()
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
    /// [`Rng::f16`], or in an extreme fixture sometimes a legal extreme header.
    fn header(&mut self, extreme: bool, lo_exp: u16, hi_exp: u16) -> u16 {
        const SPECIAL: [u16; 8] = [
            0x7bff, 0xfbff, 0x0001, 0x8001, 0x03ff, 0x0400, 0x0000, 0x8000,
        ];
        if extreme && self.below(4) == 0 {
            return SPECIAL[self.below(8) as usize];
        }
        self.f16(lo_exp, hi_exp)
    }
}

/// MQ6G256V2 weights: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16,
/// independent per half) and 192 random packed payload bytes (every six-bit
/// field including the q5 word-boundary crossing).
fn gen_mq6(m: usize, k: usize, extreme: bool, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.chunks_exact_mut(200) {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.header(extreme, 9, 2), rng.header(extreme, 7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// `[MAXR][k]` activations: row magnitudes cycle through [`AMPS`]; row kinds
/// add alternating signs (`r % 11 == 3`), +-2^20 cancellation with a small
/// residue (`5`), pure subnormals/zeros (`7`), and in an extreme fixture
/// `1e30`-scale rows (`r % 13 == 9`) that overflow against extreme headers.
fn gen_x(k: usize, extreme: bool, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % AMPS.len()];
        for i in 0..k {
            let v = match r % 11 {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                3 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                5 => match i % 4 {
                    0 => 1_048_576.0,
                    1 => -1_048_576.0,
                    _ => rng.unit() * 1e-3,
                },
                7 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => match rng.below(64) {
                    0 => 0.0,
                    1 => -0.0,
                    2..=5 => {
                        f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32)
                    }
                    _ if extreme && r % 13 == 9 => rng.unit() * 1e30,
                    _ => rng.unit() * amp,
                },
            };
            x.push(v);
        }
    }
    x
}

/// One output allocation: `GUARD | MAXR * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + MAXR * m], DType::F32)
            .map_err(err)?;
        Ok(Self { full, m })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXR * self.m)
    }
    fn bytes(&self) -> usize {
        self.full.buf.size()
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
        Ok(v)
    }
    /// Expected bytes: sentinel everywhere, `reference`'s first `rows` rows in place.
    fn expected(&self, reference: &[u8], rows: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + rows * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
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
            format!("row {row} column {} byte {}", rem / 4, rem % 4)
        }
    }
}

#[derive(Deserialize)]
struct BlobPin {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct CaptureManifest {
    schema: String,
    family: String,
    name: String,
    arch: String,
    form: String,
    ms: [usize; 4],
    k: usize,
    rows: usize,
    weights: [BlobPin; 4],
    x: BlobPin,
    scale: serde_json::Value,
    provenance: SourcePin,
    hc_act: Option<bool>,
}

#[derive(Deserialize)]
struct SourcePin {
    source: String,
    source_sha256: String,
}

#[derive(Deserialize)]
struct CapturedSingleton {
    symbol: String,
    module: String,
    object_sha256: String,
    container_sha256: Option<String>,
}

#[derive(Deserialize)]
struct WeightBinding {
    #[serde(rename = "name")]
    _name: String,
    layer: serde_json::Value,
    byte_offset: usize,
    bytes: usize,
}

#[derive(Deserialize)]
struct CapturedRow {
    row: usize,
    position: usize,
    token: u32,
    occurrence: usize,
}

#[derive(Deserialize)]
struct SourceIndex {
    schema: String,
    name: String,
    family: String,
    form: String,
    ms: [usize; 4],
    k: usize,
    rows: usize,
    weights: [BlobPin; 4],
    x: BlobPin,
    singleton: CapturedSingleton,
    weight_bindings: [WeightBinding; 4],
    captures: Vec<CapturedRow>,
    model: BlobPin,
    prompt: BlobPin,
    capture_mode: String,
    retained_identity_claim: bool,
    hc_act: Option<bool>,
}

struct CaptureProvenance {
    manifest_path: PathBuf,
    manifest_sha256: String,
    source_path: PathBuf,
    source_sha256: String,
    singleton: CapturedSingleton,
}

struct LoadedCapture {
    manifest: CaptureManifest,
    weights: Vec<Vec<u8>>,
    x_host: Vec<f32>,
    provenance: CaptureProvenance,
}

fn valid_sha(sha: &str) -> bool {
    sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())
}

fn relative_file(parent: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    if name.is_empty() || relative.is_absolute() {
        return Err(format!("capture path must be nonempty and relative: {name:?}").into());
    }
    Ok(parent.join(relative))
}

fn verify_pin(bytes: &[u8], sha: &str, label: &str) -> Result<()> {
    if !valid_sha(sha) {
        return Err(format!("{label} has invalid SHA256 {sha:?}").into());
    }
    let actual = format!("{:x}", Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(sha) {
        return Err(format!("{label} SHA256 mismatch: actual={actual} pinned={sha}").into());
    }
    Ok(())
}

fn pinned_blob(parent: &Path, pin: &BlobPin, size: usize) -> Result<Vec<u8>> {
    let path = relative_file(parent, &pin.path)?;
    if !valid_sha(&pin.sha256) {
        return Err(format!("{} has invalid SHA256", path.display()).into());
    }
    if std::fs::metadata(&path).map_err(err)?.len() != size as u64 {
        return Err(format!("{} must contain exactly {size} bytes", path.display()).into());
    }
    let bytes = std::fs::read(&path).map_err(err)?;
    if bytes.len() != size {
        return Err(format!(
            "{} extent changed while reading; expected={size} actual={}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    verify_pin(&bytes, &pin.sha256, &path.display().to_string())?;
    Ok(bytes)
}

fn capture_extent(factors: &[usize], label: &str) -> Result<usize> {
    let bytes = factors
        .iter()
        .try_fold(1usize, |n, &factor| n.checked_mul(factor))
        .ok_or_else(|| Failure::from(format!("{label} overflows host extent")))?;
    if bytes > u32::MAX as usize {
        return Err(format!("{label} exceeds device 32-bit byte offsets: {bytes}").into());
    }
    Ok(bytes)
}

fn matching_pin(source: &BlobPin, manifest: &BlobPin, label: &str) -> Result<()> {
    if source.path != manifest.path
        || !valid_sha(&source.sha256)
        || !source.sha256.eq_ignore_ascii_case(&manifest.sha256)
    {
        return Err(format!("source/manifest {label} path/SHA linkage differs").into());
    }
    Ok(())
}

impl LoadedCapture {
    fn load(path: &Path, arch: &str) -> Result<Self> {
        let manifest_bytes = std::fs::read(path).map_err(err)?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
        let manifest: CaptureManifest = serde_json::from_slice(&manifest_bytes).map_err(err)?;
        if manifest.schema != "qwen4-rowtile-operands-v1"
            || manifest.family != "mq6"
            || manifest.name.trim().is_empty()
            || manifest.arch != arch
            || manifest.rows != MAXR
            || !manifest.scale.is_null()
            || manifest.hc_act == Some(true)
        {
            return Err(
                "capture schema/family/name/architecture/rows/scale mismatch"
                    .to_string()
                    .into(),
            );
        }
        let ms = manifest.ms;
        let k = manifest.k;
        if k == 0
            || k > i32::MAX as usize
            || k % 256 != 0
            || ms.iter().any(|&m| m > i32::MAX as usize)
            || match manifest.form.as_str() {
                "single" => ms[0] == 0 || ms[1..].iter().any(|&m| m != 0),
                "x4" => ms.iter().all(|&m| m == 0),
                _ => true,
            }
        {
            return Err("invalid captured MQ6 form/dimensions".to_string().into());
        }
        let x_bytes = capture_extent(&[MAXR, k, 4], "captured X")?;
        capture_extent(&[MAXR + 2 * PRE_ROWS, k, 4], "guarded X allocation")?;
        let mut weight_bytes = [0usize; 4];
        let mut blocks = 0usize;
        for (i, &m) in ms.iter().enumerate() {
            weight_bytes[i] = capture_extent(&[m, k / 256, 200], "MQ6 weight")?;
            let y = capture_extent(&[MAXR, m, 4], "captured Y")?;
            let guarded_y = y
                .checked_add(2 * GUARD * 4)
                .ok_or_else(|| Failure::from("guarded Y allocation overflows".to_string()))?;
            capture_extent(&[guarded_y], "guarded Y allocation")?;
            blocks = blocks
                .checked_add(m.div_ceil(8))
                .ok_or_else(|| Failure::from("captured grid overflows".to_string()))?;
        }
        if blocks == 0 || blocks > u32::MAX as usize {
            return Err("invalid captured MQ6 grid".to_string().into());
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let source_path = relative_file(parent, &manifest.provenance.source)?;
        let source_bytes = std::fs::read(&source_path).map_err(err)?;
        verify_pin(
            &source_bytes,
            &manifest.provenance.source_sha256,
            "source-index",
        )?;
        let source: SourceIndex = serde_json::from_slice(&source_bytes).map_err(err)?;
        if source.schema != "qwen4-rowtile-source-v1"
            || source.name != manifest.name
            || source.family != manifest.family
            || source.form != manifest.form
            || source.ms != ms
            || source.k != k
            || source.rows != MAXR
            || source.capture_mode != "eager-liveprojection-operands-only"
            || source.retained_identity_claim
            || source.hc_act == Some(true)
        {
            return Err("source-index schema/common shape differs from manifest"
                .to_string()
                .into());
        }
        let expected_symbol = if manifest.form == "x4" {
            "gemv_mq6g256v2_x4"
        } else {
            "gemv_mq6g256v2"
        };
        if source.singleton.symbol != expected_symbol
            || source.singleton.module.trim().is_empty()
            || !valid_sha(&source.singleton.object_sha256)
            || source
                .singleton
                .container_sha256
                .as_ref()
                .is_some_and(|sha| !valid_sha(sha))
        {
            return Err("invalid source-index actual singleton identity"
                .to_string()
                .into());
        }
        for pin in [&source.model, &source.prompt] {
            // These are authoritative recorded artifact pins, not files to copy.
            if pin.path.is_empty() || !valid_sha(&pin.sha256) {
                return Err("invalid source-index model/prompt pin".to_string().into());
            }
        }
        for i in 0..4 {
            matching_pin(&source.weights[i], &manifest.weights[i], "weight")?;
            let binding = &source.weight_bindings[i];
            if !(binding.layer.is_null()
                || binding.layer.as_u64().is_some()
                || binding.layer.as_i64().is_some())
                || binding.bytes != weight_bytes[i]
                || binding.byte_offset.checked_add(binding.bytes).is_none()
            {
                return Err(format!("invalid source-index weight binding matrix {i}").into());
            }
        }
        matching_pin(&source.x, &manifest.x, "X")?;
        if source.captures.len() != MAXR {
            return Err(format!("source-index requires exactly {MAXR} captured rows").into());
        }
        let start = source.captures[0].position;
        for (r, row) in source.captures.iter().enumerate() {
            if row.row != r || start.checked_add(r) != Some(row.position) {
                return Err(format!("invalid source-index captured row {r}").into());
            }
            // Token/occurrence are recorded integer provenance, not a new
            // semantic restriction on which real projection captures qualify.
            let _ = (row.token, row.occurrence);
        }
        let weights = manifest
            .weights
            .iter()
            .zip(weight_bytes)
            .map(|(pin, size)| pinned_blob(parent, pin, size))
            .collect::<Result<Vec<_>>>()?;
        let raw_x = pinned_blob(parent, &manifest.x, x_bytes)?;
        let mut x_host = Vec::with_capacity(MAXR * k);
        for (i, bytes) in raw_x.chunks_exact(4).enumerate() {
            let value =
                f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            if !value.is_finite() {
                return Err(
                    format!("captured X is nonfinite at row {} column {}", i / k, i % k).into(),
                );
            }
            x_host.push(value);
        }
        let provenance = CaptureProvenance {
            manifest_path: path.to_path_buf(),
            manifest_sha256,
            source_path,
            source_sha256: manifest.provenance.source_sha256.clone(),
            singleton: source.singleton,
        };
        Ok(Self {
            manifest,
            weights,
            x_host,
            provenance,
        })
    }
}

/// One shape: one matrix (`ms = [m, 0, 0, 0]`) or the x4 group `ms`, shared K.
struct Fixture {
    name: String,
    x4: bool,
    ms: [usize; 4],
    k: usize,
    ws: Vec<Rc<GpuTensor>>,
    /// Same shapes, different bytes: the swapped-weights negative control.
    ws_alt: Vec<Rc<GpuTensor>>,
    outs: Vec<Out>,
    /// `[PRE_ROWS NaN rows | MAXR rows | PRE_ROWS NaN rows] x k`.
    x_full: GpuTensor,
    x_host: Vec<f32>,
    provenance: Option<CaptureProvenance>,
}

/// Pool-backed ownership, with every logical byte initialized before use.
fn upload_weight_bytes(gpu: &mut Gpu, bytes: &[u8]) -> Result<GpuTensor> {
    let tensor = gpu.alloc_tensor(&[bytes.len()], DType::Raw).map_err(err)?;
    gpu.hip.memcpy_htod(&tensor.buf, bytes).map_err(err)?;
    Ok(tensor)
}

fn upload_weights(
    gpu: &mut Gpu,
    ms: [usize; 4],
    k: usize,
    extreme: bool,
    rng: &mut Rng,
) -> Result<Vec<Rc<GpuTensor>>> {
    let mut ws = Vec::new();
    for &m in &ms {
        // An unused slot still needs a valid pointer.
        let bytes = if m == 0 {
            vec![0u8; 256]
        } else {
            gen_mq6(m, k, extreme, rng)
        };
        ws.push(Rc::new(upload_weight_bytes(gpu, &bytes)?));
    }
    Ok(ws)
}

impl Fixture {
    /// Consume only allocation owners, after verification's last synchronized check.
    /// Views remain local to launches/checks; no Rc aliases may outlive verification.
    fn release(self, gpu: &mut Gpu) -> Result<()> {
        for weight in self.ws.into_iter().chain(self.ws_alt) {
            let owner = Rc::try_unwrap(weight).map_err(|_| {
                Failure::from("weight alias survived fixture verification".to_string())
            })?;
            gpu.free_tensor(owner).map_err(err)?;
        }
        for out in self.outs {
            gpu.free_tensor(out.full).map_err(err)?;
        }
        gpu.free_tensor(self.x_full).map_err(err)
    }

    fn new(
        gpu: &mut Gpu,
        ms: [usize; 4],
        x4: bool,
        k: usize,
        extreme: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        let ws = upload_weights(gpu, ms, k, extreme, rng)?;
        let ws_alt = upload_weights(gpu, ms, k, extreme, rng)?;
        let mut outs = Vec::new();
        for &m in &ms {
            outs.push(Out::new(gpu, m)?);
        }
        let x_host = gen_x(k, extreme, rng);
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
        let name = if x4 {
            format!(
                "mq6 x4 ms={ms:?} k={k}{}",
                if extreme { " extreme" } else { "" }
            )
        } else {
            format!(
                "mq6 m={} k={k}{}",
                ms[0],
                if extreme { " extreme" } else { "" }
            )
        };
        let fx = Self {
            name,
            x4,
            ms,
            k,
            ws,
            ws_alt,
            outs,
            x_full,
            x_host,
            provenance: None,
        };
        fx.prepare_x(gpu, &fx.x_host, MAXR)?;
        Ok(fx)
    }

    /// Upload the pinned raw operands directly, never generate then replace
    /// production-sized original weights or captured activations.
    fn from_capture(gpu: &mut Gpu, capture: LoadedCapture, seed: u64) -> Result<Self> {
        let LoadedCapture {
            manifest,
            weights,
            x_host,
            provenance,
        } = capture;
        let ms = manifest.ms;
        let k = manifest.k;
        let x4 = manifest.form == "x4";
        let mut ws = Vec::with_capacity(4);
        for bytes in weights {
            let tensor = if bytes.is_empty() {
                // Empty matrices own no logical bytes but still need valid pointers.
                upload_weight_bytes(gpu, &[0u8; 256])?
            } else {
                upload_weight_bytes(gpu, &bytes)?
            };
            ws.push(Rc::new(tensor));
        }
        let mut rng = Rng((seed ^ 0x6d71_3663_6170_7475) | 1);
        let ws_alt = upload_weights(gpu, ms, k, false, &mut rng)?;
        let outs = ms
            .iter()
            .map(|&m| Out::new(gpu, m))
            .collect::<Result<Vec<_>>>()?;
        let x_full = gpu
            .zeros(&[(MAXR + 2 * PRE_ROWS) * k], DType::F32)
            .map_err(err)?;
        let fx = Self {
            name: format!(
                "captured mq6 {} form={} ms={ms:?} k={k}",
                manifest.name, manifest.form
            ),
            x4,
            ms,
            k,
            ws,
            ws_alt,
            outs,
            x_full,
            x_host,
            provenance: Some(provenance),
        };
        fx.prepare_x(gpu, &fx.x_host, MAXR)?;
        Ok(fx)
    }

    fn reference_identity(&self, gpu: &Gpu, symbol: &str) -> Result<Identity> {
        let identity = launched_identity(gpu, symbol)?;
        if let Some(capture) = &self.provenance {
            if symbol == capture.singleton.symbol
                && (identity.module != capture.singleton.module
                    || !identity
                        .sha
                        .eq_ignore_ascii_case(&capture.singleton.object_sha256)
                    || capture
                        .singleton
                        .container_sha256
                        .as_ref()
                        .is_some_and(|sha| !identity.container_sha.eq_ignore_ascii_case(sha)))
            {
                let failure = format!(
                    "captured singleton object differs: actual module={} sha={} captured module={} sha={}",
                    identity.module, identity.sha, capture.singleton.module, capture.singleton.object_sha256
                ).into();
                return checked_identity(identity, Err(failure));
            }
        }
        Ok(identity)
    }

    /// Poison all nonlive rows, including R..MAXR, before each width.
    fn prepare_x(&self, gpu: &Gpu, input: &[f32], rows: usize) -> Result<Vec<u8>> {
        let nan = f32::NAN.to_bits().to_le_bytes();
        let mut bytes = nan.repeat(self.x_full.buf.size() / 4);
        let start = PRE_ROWS * self.k * 4;
        for (dst, value) in bytes[start..start + rows * self.k * 4]
            .chunks_exact_mut(4)
            .zip(input)
        {
            dst.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        gpu.hip.memcpy_htod(&self.x_full.buf, &bytes).map_err(err)?;
        Ok(bytes)
    }

    /// Exact-size view of the first `rows` activation rows.
    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }

    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.outs.iter().try_for_each(|o| o.reset(gpu))
    }

    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.outs.iter().map(|o| o.download(gpu)).collect()
    }

    /// Independent actual singleton rows; grouped singleton must match each prefix.
    fn reference(
        &self,
        gpu: &mut Gpu,
        input: &[f32],
        seed: u64,
        mode: &str,
        weights: &WeightSnapshot,
    ) -> Result<Vec<Vec<u8>>> {
        self.reset(gpu)?;
        for r in 0..MAXR {
            let rows = r + 1;
            let x_bytes = self.prepare_x(gpu, input, rows)?;
            let x = self.x_view(rows).sub_offset(r * self.k, self.k);
            for i in 0..4 {
                let m = self.ms[i];
                if m == 0 {
                    continue;
                }
                let y = self.outs[i].view().sub_offset(r * m, m);
                let result = (|| {
                    gpu.gemv_mq6g256v2(&self.ws[i], &x, &y, m, self.k)
                        .map_err(err)?;
                    let identity = self.reference_identity(gpu, "gemv_mq6g256v2")?;
                    checked_identity(
                        identity,
                        check_tensor(
                            gpu,
                            &self.x_full,
                            &x_bytes,
                            "X",
                            self.k * 4,
                            4,
                            -(PRE_ROWS as isize),
                        ),
                    )
                })();
                // Later matrices have not written this row yet, but no output
                // allocation may have changed guards or inactive rows.
                let result = result
                    .and_then(|identity| checked_identity(identity, self.check_guards(gpu, rows)));
                finish_receipt(
                    gpu,
                    self,
                    seed,
                    mode,
                    &format!("singleton/matrix{i}"),
                    rows,
                    result,
                )?;
            }
        }
        let by_single = self.download(gpu)?;
        weights.check_arm(gpu, self, seed, mode, "singleton")?;
        if !self.x4 {
            return Ok(by_single);
        }
        self.reset(gpu)?;
        for r in 0..MAXR {
            let rows = r + 1;
            let x_bytes = self.prepare_x(gpu, input, rows)?;
            let x = self.x_view(rows).sub_offset(r * self.k, self.k);
            let ys: [GpuTensor; 4] =
                std::array::from_fn(|i| self.outs[i].view().sub_offset(r * self.ms[i], self.ms[i]));
            let parts = std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
            let result = (|| {
                gpu.gemv_mq6g256v2_x4(parts, &x, self.k).map_err(err)?;
                let identity = self.reference_identity(gpu, "gemv_mq6g256v2_x4")?;
                let checks = (|| {
                    self.require_equal(gpu, &by_single, rows)?;
                    check_tensor(
                        gpu,
                        &self.x_full,
                        &x_bytes,
                        "X",
                        self.k * 4,
                        4,
                        -(PRE_ROWS as isize),
                    )
                })();
                checked_identity(identity, checks)
            })();
            finish_receipt(gpu, self, seed, mode, "singleton_x4", rows, result)?;
        }
        weights.check_arm(gpu, self, seed, mode, "singleton_x4")?;
        self.download(gpu)
    }

    /// Public rows launcher (`rows >= 2`).
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize) -> Result<()> {
        let x = self.x_view(rows);
        if self.x4 {
            let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
            gpu.gemm_mq6g256v2_f32_rows_x4(parts, &x, self.k, rows)
                .map_err(err)
        } else {
            let y = self.outs[0].view();
            gpu.gemm_mq6g256v2_f32_rows(&self.ws[0], &x, &y, self.ms[0], self.k, rows)
                .map_err(err)
        }
    }

    fn symbol(&self, arch: &str) -> String {
        format!(
            "gemv_mq6g256v2_rows{}_t8_pm_{arch}",
            if self.x4 { "_x4" } else { "" }
        )
    }

    /// The frozen runtime hook owns geometry and ABI validation.
    fn run_direct(&self, gpu: &mut Gpu, rows: usize, alt: bool) -> Result<()> {
        let ws = if alt { &self.ws_alt } else { &self.ws };
        let x = self.x_view(rows);
        let ys: [GpuTensor; 4] = std::array::from_fn(|i| self.outs[i].view());
        if self.x4 {
            let parts = std::array::from_fn(|i| (&*ws[i], &ys[i], self.ms[i]));
            gpu.lab_gemv_mq6g256v2_rows_x4_t8(parts, &x, self.k, rows)
                .map_err(err)
        } else {
            gpu.lab_gemv_mq6g256v2_rows_t8(&ws[0], &x, &ys[0], self.ms[0], self.k, rows)
                .map_err(err)
        }
    }

    /// Diff every output of `got` against `want`; the first difference is described.
    fn compare(&self, got: &[Vec<u8>], want: &[Vec<u8>]) -> (usize, Option<String>) {
        let mut bad = 0;
        let mut first = None;
        for (i, ((g, w), out)) in got.iter().zip(want).zip(&self.outs).enumerate() {
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

    /// Download and compare against the first `rows` rows of `reference`.
    fn check(
        &self,
        gpu: &Gpu,
        reference: &[Vec<u8>],
        rows: usize,
    ) -> Result<(usize, Option<String>)> {
        let expected: Vec<Vec<u8>> = self
            .outs
            .iter()
            .zip(reference)
            .map(|(o, r)| o.expected(r, rows))
            .collect();
        let got = self.download(gpu)?;
        Ok(self.compare(&got, &expected))
    }

    fn require_equal(&self, gpu: &Gpu, reference: &[Vec<u8>], rows: usize) -> Result<()> {
        let (bad, first) = self.check(gpu, reference, rows)?;
        if bad != 0 {
            return Err(mismatch(format!(
                "{bad} output bytes differ: {}; logical_y_bytes={:?} allocation_y_bytes={:?}",
                first.unwrap_or_else(|| "length mismatch".to_string()),
                self.ms.map(|m| rows * m * 4),
                self.outs.iter().map(Out::bytes).collect::<Vec<_>>()
            )));
        }
        Ok(())
    }

    fn check_guards(&self, gpu: &Gpu, rows: usize) -> Result<()> {
        for (i, out) in self.outs.iter().enumerate() {
            let bytes = out.download(gpu)?;
            let lo = GUARD * 4;
            let hi = lo + rows * out.m * 4;
            if let Some((at, value)) = bytes
                .iter()
                .enumerate()
                .find(|(at, value)| (*at < lo || *at >= hi) && **value != SENTINEL)
            {
                return Err(mismatch(format!(
                    "matrix {i} {} changed to {value:#04x}; R={rows} logical_bytes={} allocation_bytes={}",
                    out.locate(at), rows * out.m * 4, out.bytes()
                )));
            }
        }
        Ok(())
    }
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

/// Metadata is queried from the actual launch, never inferred from architecture.
#[derive(Debug)]
struct Identity {
    module: String,
    symbol: String,
    sha: String,
    container_sha: String,
}

fn actual_identity(gpu: &Gpu) -> Result<Identity> {
    let symbol = gpu
        .last_launched_kernel()
        .ok_or_else(|| Failure::from("missing last_launched_kernel".to_string()))?;
    let (module, container_sha, sha) = gpu.lab_kernel_device_identity(symbol).map_err(err)?;
    if module.trim().is_empty() || !valid_sha(&sha) || !valid_sha(&container_sha) {
        return Err(format!(
            "invalid loaded identity: module={module:?} symbol={symbol} sha={sha:?}"
        )
        .into());
    }
    Ok(Identity {
        module,
        symbol: symbol.to_string(),
        sha,
        container_sha,
    })
}

fn launched_identity(gpu: &Gpu, expected: &str) -> Result<Identity> {
    // Call immediately after each real runtime launch, including singletons.
    let identity = actual_identity(gpu)?;
    if identity.symbol != expected {
        let failure = format!(
            "route refusal: launched {} from {} sha={}, expected {expected}",
            identity.symbol, identity.module, identity.sha
        )
        .into();
        return checked_identity(identity, Err(failure));
    }
    Ok(identity)
}

fn checked_identity(identity: Identity, result: Result<()>) -> Result<Identity> {
    match result {
        Ok(()) => Ok(identity),
        Err(mut failure) => {
            failure.identity = Some(identity);
            Err(failure)
        }
    }
}

fn receipt(
    gpu: &Gpu,
    fx: &Fixture,
    seed: u64,
    mode: &str,
    arm: &str,
    rows: usize,
    evidence: (Option<&Identity>, Option<&Failure>),
) {
    let (identity, failure) = evidence;
    let unavailable = if mode == "no_work_refusal" {
        "<no-launch>"
    } else {
        "<missing>"
    };
    let (module, symbol, sha, container_sha) =
        identity.map_or((unavailable, unavailable, unavailable, unavailable), |id| {
            (
                id.module.as_str(),
                id.symbol.as_str(),
                id.sha.as_str(),
                id.container_sha.as_str(),
            )
        });
    let capture = fx.provenance.as_ref();
    let origin = if capture.is_some() {
        "captured"
    } else {
        "generated"
    };
    let capture_mode = if capture.is_some() {
        "eager-liveprojection-operands-only"
    } else {
        "generated"
    };
    let x_origin = match mode {
        "reversed" => "permuted_base_operands",
        "changed_stable_address" => "synthetic_stress",
        "negative_control" if arm == "changed_input_negative" => "synthetic_negative_control",
        _ => origin,
    };
    let weight_origin = if arm == "swapped_weights_negative" {
        "synthetic_negative_control"
    } else {
        origin
    };
    println!(
        "mq6_g1 arch={} fixture={:?} seed={seed} mode={mode} arm={arm} R={rows} module={module:?} symbol={symbol} object_sha256={sha} container_sha256={container_sha} x_offset_bytes={} x_logical_bytes={} x_allocation_bytes={} y_logical_bytes={:?} y_allocation_bytes={:?} weight_logical_bytes={:?} weight_allocation_bytes={:?} alternate_weight_allocation_bytes={:?} operand_origin={origin} x_origin={x_origin} weight_origin={weight_origin} capture_mode={capture_mode} retained_identity_claim=false manifest={:?} manifest_sha256={:?} source={:?} source_sha256={:?} alternate_weights=synthetic_negative_control status={} detail={:?}",
        gpu.arch, fx.name, PRE_ROWS * fx.k * 4, rows * fx.k * 4,
        fx.x_full.buf.size(), fx.ms.map(|m| rows * m * 4),
        fx.outs.iter().map(Out::bytes).collect::<Vec<_>>(),
        fx.ms.map(|m| m * (fx.k / 256) * 200),
        fx.ws.iter().map(|w| w.buf.size()).collect::<Vec<_>>(),
        fx.ws_alt.iter().map(|w| w.buf.size()).collect::<Vec<_>>(),
        capture.map(|p| &p.manifest_path), capture.map(|p| &p.manifest_sha256),
        capture.map(|p| &p.source_path), capture.map(|p| &p.source_sha256),
        if failure.is_some() { "FAIL" } else { "PASS" },
        failure.map(|e| e.message.as_str())
    );
}

fn finish_receipt(
    gpu: &Gpu,
    fx: &Fixture,
    seed: u64,
    mode: &str,
    arm: &str,
    rows: usize,
    result: Result<Identity>,
) -> Result<()> {
    match result {
        Ok(identity) => {
            receipt(gpu, fx, seed, mode, arm, rows, (Some(&identity), None));
            Ok(())
        }
        Err(mut failure) => {
            // Never relabel a stale last-launch record after a failed launch.
            receipt(
                gpu,
                fx,
                seed,
                mode,
                arm,
                rows,
                (failure.identity.as_ref(), Some(&failure)),
            );
            failure.message = format!("mode={mode} arm={arm} R={rows}: {}", failure.message);
            Err(failure)
        }
    }
}

fn tensor_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0u8; tensor.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).map_err(err)?;
    Ok(bytes)
}

fn check_tensor(
    gpu: &Gpu,
    tensor: &GpuTensor,
    expected: &[u8],
    label: &str,
    row_bytes: usize,
    element_bytes: usize,
    row_origin: isize,
) -> Result<()> {
    let got = tensor_bytes(gpu, tensor)?;
    let (bad, first) = diff(&got, expected);
    if bad != 0 {
        let at = first.unwrap_or(got.len().min(expected.len()));
        return Err(mismatch(format!(
            "{label} mutated at row {} column {} byte {} (allocation byte {at}); bad_bytes={bad} got_extent={} expected_extent={} got={:?} expected={:?}",
            (at / row_bytes) as isize + row_origin, (at % row_bytes) / element_bytes,
            at % element_bytes, got.len(), expected.len(), got.get(at), expected.get(at)
        )));
    }
    Ok(())
}

/// Both real and negative-control weight allocations are immutable. Batch their
/// exact downloads at arm boundaries rather than rereading production weights
/// after every logical width.
struct WeightSnapshot(Vec<Vec<u8>>);

impl WeightSnapshot {
    fn new(gpu: &Gpu, fx: &Fixture) -> Result<Self> {
        fx.ws
            .iter()
            .chain(&fx.ws_alt)
            .map(|w| tensor_bytes(gpu, w))
            .collect::<Result<Vec<_>>>()
            .map(Self)
    }

    fn check(&self, gpu: &Gpu, fx: &Fixture) -> Result<()> {
        for (i, (w, expected)) in fx.ws.iter().chain(&fx.ws_alt).zip(&self.0).enumerate() {
            check_tensor(
                gpu,
                w,
                expected,
                &format!("weight matrix {} alternate={} MQ6-group", i % 4, i >= 4),
                (fx.k / 256) * 200,
                200,
                0,
            )?;
        }
        Ok(())
    }

    fn check_arm(&self, gpu: &Gpu, fx: &Fixture, seed: u64, mode: &str, arm: &str) -> Result<()> {
        if let Err(failure) = self.check(gpu, fx) {
            let result = checked_identity(actual_identity(gpu)?, Err(failure));
            return finish_receipt(gpu, fx, seed, mode, arm, MAXR, result);
        }
        Ok(())
    }
}

fn public_symbol(fx: &Fixture, arch: &str, rows: usize) -> String {
    if rows > 8 {
        fx.symbol(arch)
    } else {
        // lane_rows_scope=false intentionally selects these actual incumbents
        // on both qualified architectures; arbitrary "narrow" symbols are not allowed.
        format!(
            "gemm_mq6g256v2_f32_rows{}_r{rows}",
            if fx.x4 { "_x4" } else { "" }
        )
    }
}

fn sweep(
    gpu: &mut Gpu,
    fx: &Fixture,
    input: &[f32],
    reference: &[Vec<u8>],
    seed: u64,
    mode: &str,
    weights: &WeightSnapshot,
) -> Result<()> {
    for direct in [true, false] {
        let arm = if direct { "direct_t8" } else { "public_rows" };
        let first = if direct { 1 } else { 2 };
        let mut reached = [false; MAXR];
        for rows in first..=MAXR {
            let result = (|| {
                let x_bytes = fx.prepare_x(gpu, input, rows)?;
                fx.reset(gpu)?;
                let expected = if direct {
                    fx.symbol(&gpu.arch)
                } else {
                    public_symbol(fx, &gpu.arch, rows)
                };
                if direct {
                    fx.run_direct(gpu, rows, false)?;
                } else {
                    fx.run_launcher(gpu, rows)?;
                }
                let identity = launched_identity(gpu, &expected)?;
                let checks = (|| {
                    fx.require_equal(gpu, reference, rows)?;
                    check_tensor(
                        gpu,
                        &fx.x_full,
                        &x_bytes,
                        "X",
                        fx.k * 4,
                        4,
                        -(PRE_ROWS as isize),
                    )
                })();
                checked_identity(identity, checks)
            })();
            finish_receipt(gpu, fx, seed, mode, arm, rows, result)?;
            reached[rows - 1] = true;
        }
        weights.check_arm(gpu, fx, seed, mode, arm)?;
        if reached[first - 1..].iter().any(|&seen| !seen) {
            return Err(format!("incomplete {arm} width coverage in {mode}").into());
        }
        println!(
            "mq6_g1_coverage arch={} fixture={:?} seed={seed} mode={mode} arm={arm} reached={first}..={MAXR} count={} status=PASS",
            gpu.arch, fx.name, reached.iter().filter(|&&seen| seen).count()
        );
    }
    Ok(())
}

fn require_changed(fx: &Fixture, got: &[Vec<u8>], original: &[Vec<u8>], label: &str) -> Result<()> {
    for (i, &m) in fx.ms.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let lo = GUARD * 4;
        let hi = lo + MAXR * m * 4;
        if got[i][lo..hi] == original[i][lo..hi] {
            return Err(mismatch(format!(
                "blind {label} negative control: matrix {i} live bytes [{lo},{hi}) unchanged; allocation_bytes={}",
                fx.outs[i].bytes()
            )));
        }
    }
    Ok(())
}

fn negative_control(
    gpu: &mut Gpu,
    fx: &Fixture,
    input: &[f32],
    original: &[Vec<u8>],
    seed: u64,
    alternate_weights: bool,
    weights: &WeightSnapshot,
) -> Result<()> {
    let arm = if alternate_weights {
        "swapped_weights_negative"
    } else {
        "changed_input_negative"
    };
    let result = (|| {
        let x_bytes = fx.prepare_x(gpu, input, MAXR)?;
        fx.reset(gpu)?;
        let expected = fx.symbol(&gpu.arch);
        fx.run_direct(gpu, MAXR, alternate_weights)?;
        let identity = launched_identity(gpu, &expected)?;
        let checks = (|| {
            fx.check_guards(gpu, MAXR)?;
            require_changed(fx, &fx.download(gpu)?, original, arm)?;
            check_tensor(
                gpu,
                &fx.x_full,
                &x_bytes,
                "X",
                fx.k * 4,
                4,
                -(PRE_ROWS as isize),
            )?;
            weights.check(gpu, fx)
        })();
        checked_identity(identity, checks)
    })();
    finish_receipt(gpu, fx, seed, "negative_control", arm, MAXR, result)
}

fn verify_no_work(gpu: &mut Gpu, fx: &Fixture, seed: u64, weights: &WeightSnapshot) -> Result<()> {
    for rows in 1..=MAXR {
        let result = (|| {
            let x_bytes = fx.prepare_x(gpu, &fx.x_host, rows)?;
            fx.reset(gpu)?;
            let before = gpu.last_launched_kernel().map(str::to_string);
            if fx.run_direct(gpu, rows, false).is_ok() {
                return Err(Failure::from(
                    "allzero direct hook must refuse a zero grid".to_string(),
                ));
            }
            if gpu.last_launched_kernel().map(str::to_string) != before {
                return Err(Failure::from(
                    "allzero refusal changed the launch route".to_string(),
                ));
            }
            fx.check_guards(gpu, 0)?;
            check_tensor(
                gpu,
                &fx.x_full,
                &x_bytes,
                "X",
                fx.k * 4,
                4,
                -(PRE_ROWS as isize),
            )?;
            weights.check(gpu, fx)?;
            Ok(())
        })();
        receipt(
            gpu,
            fx,
            seed,
            "no_work_refusal",
            "direct_no_work",
            rows,
            (None, result.as_ref().err()),
        );
        result.map_err(|mut e| {
            e.message = format!("no_work_refusal R={rows}: {}", e.message);
            e
        })?;
    }
    println!(
        "mq6_g1_coverage arch={} fixture={:?} seed={seed} mode=no_work_refusal reached=1..=64 route_reached=false status=PASS",
        gpu.arch, fx.name
    );
    Ok(())
}

fn verify(gpu: &mut Gpu, fx: &Fixture, seed: u64) -> Result<()> {
    let weights = WeightSnapshot::new(gpu, fx)?;
    if fx.ms.iter().all(|&m| m == 0) {
        return verify_no_work(gpu, fx, seed, &weights);
    }
    let original = fx.reference(gpu, &fx.x_host, seed, "original", &weights)?;
    sweep(gpu, fx, &fx.x_host, &original, seed, "original", &weights)?;

    let mut reversed = Vec::with_capacity(fx.x_host.len());
    for row in fx.x_host.chunks_exact(fx.k).rev() {
        reversed.extend_from_slice(row);
    }
    let reference_rev = fx.reference(gpu, &reversed, seed, "reversed", &weights)?;
    sweep(
        gpu,
        fx,
        &reversed,
        &reference_rev,
        seed,
        "reversed",
        &weights,
    )?;
    drop(reference_rev);

    // Same allocations/pointers, changed contents; never a cached stale reference.
    let mut rng = Rng((seed ^ 0x00ba_d5ee_d00d_f00d ^ ((fx.ms[0] as u64) << 8) ^ fx.k as u64) | 1);
    let changed = gen_x(fx.k, false, &mut rng);
    let reference_new = fx.reference(gpu, &changed, seed, "changed_stable_address", &weights)?;
    sweep(
        gpu,
        fx,
        &changed,
        &reference_new,
        seed,
        "changed_stable_address",
        &weights,
    )?;
    drop(reference_new);
    negative_control(gpu, fx, &changed, &original, seed, false, &weights)?;
    negative_control(gpu, fx, &fx.x_host, &original, seed, true, &weights)?;
    Ok(())
}

/// Host-only shape descriptions; construct/release GPU fixtures sequentially.
fn fixtures() -> Vec<([usize; 4], bool, usize, bool)> {
    let mut cases = Vec::new();
    for groups in 1..=24 {
        for m in [1, 2, 3, 7, 8, 9, 17] {
            cases.push(([m, 0, 0, 0], false, groups * 256, false));
        }
        cases.push(([1, 2, 9, 17], true, groups * 256, false));
    }
    for (m, k) in [
        (15, 1792),
        (16, 2048),
        (48, 2560),
        (101, 6144),
        (64, 10240),
        (2560, 6144),
        (4099, 2560),
    ] {
        cases.push(([m, 0, 0, 0], false, k, false));
    }
    for (ms, k) in [
        ([10240, 6144, 48, 48], 2560),
        ([12288, 640, 512, 512], 2560),
        ([3, 0, 5, 0], 2560),
        ([1, 2, 3, 7], 2560),
        ([9, 0, 0, 17], 2560),
        ([0, 0, 0, 1], 2560),
        ([0, 5, 0, 0], 2560),
        ([7, 8, 9, 15], 768),
        ([48, 48, 1, 1], 1280),
        ([4099, 640, 512, 9], 2560),
        ([1, 7, 9, 17], 10240),
    ] {
        cases.push((ms, true, k, false));
    }
    for mask in 0..16 {
        let ms = std::array::from_fn(|i| {
            if mask & (1 << i) != 0 {
                [1, 2, 9, 17][i]
            } else {
                0
            }
        });
        cases.push((ms, true, 2560, false));
    }
    cases.push(([48, 0, 0, 0], false, 2560, true));
    cases.push(([13, 0, 0, 0], false, 768, true));
    cases.push(([5, 13, 0, 9], true, 2560, true));
    cases
}

fn run() -> Result<()> {
    let mut seed = 0x6d71_3677_6964_6531u64;
    let mut args = std::env::args().skip(1);
    let mut seed_seen = false;
    let mut captured = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" if !seed_seen => {
                seed = args.next()
                    .ok_or_else(|| Failure::from("--seed requires an unsigned integer".to_string()))?
                    .parse::<u64>().map_err(err)?;
                seed_seen = true;
            }
            "--captured" => {
                let path = args.next()
                    .ok_or_else(|| Failure::from("--captured requires a manifest path".to_string()))?;
                captured.push(PathBuf::from(path));
            }
            _ => return Err(format!(
                "unknown/repeated argument {arg}; usage: mq6_wide_rows_oracle [--seed INTEGER] --captured MANIFEST [--captured MANIFEST ...]"
            ).into()),
        }
    }
    if captured.is_empty() {
        return Err("at least one --captured MQ6 manifest is required; synthetic-only G1 completion is forbidden".to_string().into());
    }
    // The family selector is LazyLock-backed, so set it before Gpu initialization.
    // SAFETY: main has not started any threads or initialized HIP yet.
    unsafe {
        std::env::set_var("HIPFIRE_QWEN4_WIDE_TWIN_MQ6", "1");
    }
    rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(Some(MAXR));
    let mut gpu =
        Gpu::init().map_err(|e| Failure::from(format!("fixture=<setup> seed={seed}: {e}")))?;
    if !matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") {
        return Err(format!("fixture=<setup> seed={seed}: unsupported MQ6 rowtile architecture {}; no gfx1100 alias", gpu.arch).into());
    }
    gpu.lane_rows_scope = false;
    gpu.set_wide_verify_rows(MAXR);
    gpu.set_qwen4_verify_reference(Some(
        rdna_compute::prefill_invariant::Qwen4VerifyReference::Eager,
    ));
    if gpu.few_row_limit() < MAXR || gpu.prefill_invariant_scope {
        return Err(format!("fixture=<setup> seed={seed}: eager rows64 scope unavailable: few_row_limit={} prefill_invariant={}", gpu.few_row_limit(), gpu.prefill_invariant_scope).into());
    }
    for form in ["", "_x4"] {
        let name = format!("gemv_mq6g256v2_rows{form}_t8_pm_{}", gpu.arch);
        gpu.ensure_row_tile_pm_entry(&name).map_err(|e| {
            Failure::from(format!("fixture=<setup> seed={seed} symbol={name}: {e}"))
        })?;
    }
    let seeds = [seed, seed ^ 0x9e37_79b9_7f4a_7c15];
    let mut count = 0;
    for seed in seeds {
        for (index, (ms, x4, k, extreme)) in fixtures().into_iter().enumerate() {
            let context =
                format!("fixture={index} ms={ms:?} x4={x4} k={k} extreme={extreme} seed={seed}");
            let result = (|| {
                let mut rng =
                    Rng(seed.wrapping_add((index as u64).wrapping_mul(0xd134_2543_de82_ef95)) | 1);
                let fx = Fixture::new(&mut gpu, ms, x4, k, extreme, &mut rng)?;
                verify(&mut gpu, &fx, seed)?;
                fx.release(&mut gpu)
            })();
            result.map_err(|mut e| {
                e.message = format!("{context}: {}", e.message);
                e
            })?;
            count += 1;
        }
    }
    let mut captured_count = 0;
    for seed in seeds {
        for path in &captured {
            let context = format!("captured_fixture={} seed={seed}", path.display());
            let result = (|| {
                let capture = LoadedCapture::load(path, &gpu.arch)?;
                let fx = Fixture::from_capture(&mut gpu, capture, seed)?;
                verify(&mut gpu, &fx, seed)?;
                fx.release(&mut gpu)
            })();
            result.map_err(|mut e| {
                e.message = format!("{context}: {}", e.message);
                e
            })?;
            captured_count += 1;
        }
    }
    println!("mq6_g1_complete arch={} seeds={seeds:?} generated_fixtures={count} captured_fixtures={captured_count} direct_widths=1..=64 public_widths=2..=64 modes=original,reversed,changed_stable_address capture_mode=eager-liveprojection-operands-only retained_identity_claim=false status=PASS", gpu.arch);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("mq6_wide_rows_oracle: {}", e.message);
        std::process::exit(e.code);
    }
}
