// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `moe` subcommand: the MoE module oracle of the Raon arch crate.
//!
//! Runs the PRODUCTION operators (`hipfire_arch_raon::moe` descriptors over
//! `execute_grouped_moe` / `execute_gated_ffn`; never a duplicate kernel) on
//! fixture cases and on deterministic synthetic cases, and compares them with
//! independent CPU references:
//!
//! * routing against the fixture (exact indices, `1e-5` / `1e-6` weights) or a
//!   CPU router reference with the documented tie policy (lower id first);
//! * routed / shared / out against a CPU module reference computed from the
//!   SAME dequantized qt44 (`MQ4G256V2`) weights the oracle itself quantized
//!   and uploaded (f64 dots, the GPU's routing fed to the expert math so a
//!   routing mismatch cannot cascade);
//! * informationally, against the fixture outputs of the original weights
//!   (`quant_error_vs_original_weights`; only a catastrophic `> 0.5` fails).
//!
//! Weight rows are rotated (FWHT-256, the activation convention) before they
//! are quantized, so `dot(W_rot, rotate(x)) == dot(W, x)`.
//!
//! Options (all optional): `case` (substring filter), `protocols`
//! (`indexed,grouped`), `expert_limit`, `awq_real`, `allow_refusal`,
//! `synthetic` (default 1), `negative` (refused: this module has none).
use hipfire_arch_raon::moe::{
    forward_dense, raon_moe_op, RaonDenseRefs, RaonDenseScratch, RaonMoeDims, RaonMoeLayerRefs,
    RaonMoeScratch,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_gated_ffn, execute_grouped_moe, CastPolicy, FfnActivation, GatedFfnOp,
    GroupedMoeProtocol, GroupedMoeShared, GroupedMoeSharedScratch, RouterKind, SharedExpertPolicy,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

// ───────────────────────────── numeric helpers ─────────────────────────────

/// f32 -> IEEE binary16 bits, round-to-nearest-even (overflow -> inf).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x0200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut h = half;
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (man << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// BF16 round-to-nearest-even of an f32 (NaN stays NaN).
fn bf16r(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = [0f64; 4];
    let mut i = 0;
    while i + 4 <= n {
        for l in 0..4 {
            s[l] += a[i + l] as f64 * b[i + l] as f64;
        }
        i += 4;
    }
    while i < n {
        s[0] += a[i] as f64 * b[i] as f64;
        i += 1;
    }
    s[0] + s[1] + s[2] + s[3]
}

/// Deterministic generator for the synthetic cases (64-bit LCG + Box-Muller).
struct Lcg {
    state: u64,
    spare: Option<f64>,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let mut g = Self {
            state: z,
            spare: None,
        };
        g.next_u64();
        g
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s as f32;
        }
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        self.spare = Some(r * th.sin());
        (r * th.cos()) as f32
    }
}

fn mix_seed(parts: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &p in parts {
        h ^= p;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

fn str_seed(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn normal_vec(g: &mut Lcg, n: usize, std: f32, bf16: bool) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = g.normal() * std;
            if bf16 {
                bf16r(v)
            } else {
                v
            }
        })
        .collect()
}

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(16)
}

/// Split `out` (`nrows * stride` elements) into row chunks and run
/// `f(first_row, end_row, chunk)` on up to `threads` scoped threads.
fn par_chunks<T: Send>(
    threads: usize,
    nrows: usize,
    stride: usize,
    out: &mut [T],
    f: &(dyn Fn(usize, usize, &mut [T]) -> Result<(), String> + Sync),
) -> Result<(), String> {
    if nrows.checked_mul(stride) != Some(out.len()) {
        return Err(format!(
            "par_chunks: {} elements != {nrows} rows x {stride}",
            out.len()
        ));
    }
    if nrows == 0 || stride == 0 {
        return Ok(());
    }
    let threads = threads.max(1).min(nrows);
    if threads == 1 {
        return f(0, nrows, out);
    }
    let per = nrows.div_ceil(threads);
    std::thread::scope(|s| {
        let mut hs = Vec::new();
        for (ci, chunk) in out.chunks_mut(per * stride).enumerate() {
            let r0 = ci * per;
            let r1 = (r0 + per).min(nrows);
            hs.push(s.spawn(move || f(r0, r1, chunk)));
        }
        let mut res = Ok(());
        for h in hs {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if res.is_ok() {
                        res = Err(e);
                    }
                }
                Err(_) => {
                    if res.is_ok() {
                        res = Err("worker thread panicked".to_string());
                    }
                }
            }
        }
        res
    })
}

// ───────────────────────────── metrics and bands ───────────────────────────

#[derive(Clone, Copy, Debug)]
struct Metrics {
    rel_l2: f64,
    cos: f64,
    max_abs: f64,
    ref_max: f64,
    nonfinite: usize,
}

fn compare(got: &[f32], want: &[f32]) -> Metrics {
    let n = got.len().min(want.len());
    let (mut dd, mut ww, mut gg, mut gw) = (0f64, 0f64, 0f64, 0f64);
    let (mut max_abs, mut ref_max) = (0f64, 0f64);
    let mut nonfinite = 0usize;
    for j in 0..n {
        let (a, b) = (got[j] as f64, want[j] as f64);
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = a - b;
        dd += d * d;
        ww += b * b;
        gg += a * a;
        gw += a * b;
        max_abs = max_abs.max(d.abs());
        ref_max = ref_max.max(b.abs());
    }
    if got.len() != want.len() {
        nonfinite += got.len().max(want.len()) - n;
    }
    let (rel_l2, cos) = if nonfinite > 0 {
        (f64::INFINITY, 0.0)
    } else {
        let rel = if ww > 0.0 {
            (dd / ww).sqrt()
        } else {
            dd.sqrt()
        };
        let cos = if gg > 0.0 && ww > 0.0 {
            gw / (gg.sqrt() * ww.sqrt())
        } else if gg == 0.0 && ww == 0.0 {
            1.0
        } else {
            0.0
        };
        (rel, cos)
    };
    Metrics {
        rel_l2,
        cos,
        max_abs,
        ref_max,
        nonfinite,
    }
}

/// `(pass, band label)` of one comparison. BF16 mode: `rel_l2 <= 1e-2 &&
/// cos >= 0.9999`. F32 mode: `rel_l2 <= 1e-4 && max_abs <= 2e-5 + 2e-4*max|ref|`;
/// only for the grouped core (activations pass through fp16 in the WMMA core)
/// a miss is retried against `rel_l2 <= 2e-3`, labelled `fp16-wmma-core`.
fn band_check(m: &Metrics, bf16: bool, grouped: bool) -> (bool, &'static str) {
    if m.nonfinite > 0 {
        return (false, if bf16 { "bf16" } else { "f32" });
    }
    if bf16 {
        return (m.rel_l2 <= 1e-2 && m.cos >= 0.9999, "bf16");
    }
    if m.rel_l2 <= 1e-4 && m.max_abs <= 2e-5 + 2e-4 * m.ref_max {
        return (true, "f32");
    }
    if grouped && m.rel_l2 <= 2e-3 {
        return (true, "fp16-wmma-core");
    }
    (false, "f32")
}

fn jf(x: f64) -> Value {
    match serde_json::Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None => Value::String(format!("{x}")),
    }
}

fn metrics_json(m: &Metrics, band: &str, pass: bool) -> Value {
    json!({
        "rel_l2": jf(m.rel_l2),
        "cos": jf(m.cos),
        "max_abs": jf(m.max_abs),
        "nonfinite": m.nonfinite,
        "band": band,
        "pass": pass,
    })
}

fn metrics_info_json(m: &Metrics) -> Value {
    json!({"rel_l2": jf(m.rel_l2), "cos": jf(m.cos), "max_abs": jf(m.max_abs), "nonfinite": m.nonfinite})
}

// ───────────────────────────── safetensors reader ──────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum StDtype {
    F32,
    Bf16,
    F16,
    I32,
    I64,
    U8,
}

impl StDtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F32" => Self::F32,
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "U8" => Self::U8,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

struct StTensor {
    dtype: StDtype,
    shape: Vec<usize>,
    /// Absolute byte offsets inside the file.
    start: u64,
    end: u64,
}

struct StFile {
    file: std::fs::File,
    path: PathBuf,
    tensors: BTreeMap<String, StTensor>,
    /// Names rooted at a `moe.` path component -> full tensor names.
    moe_suffix: HashMap<String, Vec<String>>,
}

impl StFile {
    fn open(path: &Path) -> Result<Self, String> {
        let ctx = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut file = std::fs::File::open(path).map_err(ctx)?;
        let len = file.metadata().map_err(ctx)?.len();
        if len < 8 {
            return Err(format!(
                "{}: file shorter than a safetensors header",
                path.display()
            ));
        }
        let mut hl = [0u8; 8];
        file.read_exact(&mut hl).map_err(ctx)?;
        let hlen = u64::from_le_bytes(hl);
        if hlen > len - 8 || hlen > (1u64 << 28) {
            return Err(format!(
                "{}: header length {hlen} is invalid",
                path.display()
            ));
        }
        let mut hb = vec![0u8; hlen as usize];
        file.read_exact(&mut hb).map_err(ctx)?;
        let header: Value = serde_json::from_slice(&hb)
            .map_err(|e| format!("{}: header json: {e}", path.display()))?;
        let obj = header
            .as_object()
            .ok_or_else(|| format!("{}: header is not an object", path.display()))?;
        let data_len = len - 8 - hlen;
        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let bad = |what: &str| format!("{}: tensor `{name}`: {what}", path.display());
            let dts = v
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("no dtype"))?;
            let dtype =
                StDtype::parse(dts).ok_or_else(|| bad(&format!("unsupported dtype {dts}")))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("no shape"))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad("bad shape"))?;
            let off = v
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("no data_offsets"))?;
            if off.len() != 2 {
                return Err(bad("data_offsets must have two entries"));
            }
            let a = off[0].as_u64().ok_or_else(|| bad("bad data_offsets"))?;
            let b = off[1].as_u64().ok_or_else(|| bad("bad data_offsets"))?;
            if a > b || b > data_len {
                return Err(bad(&format!(
                    "data_offsets [{a}, {b}] outside the {data_len}-byte data area"
                )));
            }
            let numel = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| bad("shape overflows"))?;
            let want = numel
                .checked_mul(dtype.size())
                .ok_or_else(|| bad("size overflows"))? as u64;
            if b - a != want {
                return Err(bad(&format!(
                    "{} bytes of data for shape {shape:?} {dts} (expected {want})",
                    b - a
                )));
            }
            tensors.insert(
                name.clone(),
                StTensor {
                    dtype,
                    shape,
                    start: 8 + hlen + a,
                    end: 8 + hlen + b,
                },
            );
        }
        let mut moe_suffix: HashMap<String, Vec<String>> = HashMap::new();
        for name in tensors.keys() {
            let bytes = name.as_bytes();
            for (p, _) in name.char_indices() {
                if name[p..].starts_with("moe.") && (p == 0 || bytes[p - 1] == b'.') {
                    moe_suffix
                        .entry(name[p..].to_string())
                        .or_default()
                        .push(name.clone());
                }
            }
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            tensors,
            moe_suffix,
        })
    }

    /// Resolve a checkpoint tensor by its `moe.`-rooted suffix; the prefix
    /// before `moe.` may vary, but the match must be unique.
    fn lookup(&self, suffix: &str) -> Result<String, String> {
        match self.moe_suffix.get(suffix) {
            Some(v) if v.len() == 1 => Ok(v[0].clone()),
            Some(v) => Err(format!(
                "{}: tensor suffix `{suffix}` is ambiguous ({} matches, e.g. `{}`)",
                self.path.display(),
                v.len(),
                v[0]
            )),
            None => Err(format!(
                "{}: missing tensor `{suffix}`",
                self.path.display()
            )),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("{}: missing tensor `{name}`", self.path.display()))?;
        let mut buf = vec![0u8; (t.end - t.start) as usize];
        let mut f = &self.file;
        f.seek(SeekFrom::Start(t.start))
            .and_then(|_| f.read_exact(&mut buf))
            .map_err(|e| format!("{}: read `{name}`: {e}", self.path.display()))?;
        let data: Vec<f32> = match t.dtype {
            StDtype::F32 => buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            StDtype::Bf16 => buf
                .chunks_exact(2)
                .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            StDtype::F16 => buf
                .chunks_exact(2)
                .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            StDtype::I32 => buf
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            StDtype::I64 => buf
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            StDtype::U8 => buf.iter().map(|&b| b as f32).collect(),
        };
        Ok((t.shape.clone(), data))
    }

    /// Read `name` and require exactly `shape`.
    fn read_shaped(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (s, d) = self.read_f32(name)?;
        if s != shape {
            return Err(format!(
                "tensor `{name}` has shape {s:?}, expected {shape:?}"
            ));
        }
        Ok(d)
    }
}

// ───────────────────────────── rotation and qt44 ───────────────────────────

const FWHT_N: usize = 256;
const QT44_GROUP_BYTES: usize = 136;

static SIGNS: LazyLock<(Vec<f32>, Vec<f32>)> = LazyLock::new(|| {
    (
        rdna_compute::gen_fwht_signs(42, FWHT_N),
        rdna_compute::gen_fwht_signs(1042, FWHT_N),
    )
});

fn fwht_signs() -> &'static (Vec<f32>, Vec<f32>) {
    &SIGNS
}

fn butterfly(g: &mut [f32; FWHT_N]) {
    let mut stride = 1;
    while stride < FWHT_N {
        let mut j = 0;
        while j < FWHT_N {
            for k in 0..stride {
                let a = g[j + k];
                let b = g[j + k + stride];
                g[j + k] = a + b;
                g[j + k + stride] = a - b;
            }
            j += stride * 2;
        }
        stride <<= 1;
    }
}

/// Forward activation rotation of one 256 group: `x*signs1`, unnormalized
/// butterfly, `*0.0625*signs2`.
fn rotate256(g: &mut [f32; FWHT_N]) {
    let (s1, s2) = fwht_signs();
    for i in 0..FWHT_N {
        g[i] *= s1[i];
    }
    butterfly(g);
    for i in 0..FWHT_N {
        g[i] *= 0.0625 * s2[i];
    }
}

/// Transpose (= inverse) of [`rotate256`].
#[cfg(test)]
fn unrotate256(g: &mut [f32; FWHT_N]) {
    let (s1, s2) = fwht_signs();
    for i in 0..FWHT_N {
        g[i] *= s2[i];
    }
    butterfly(g);
    for i in 0..FWHT_N {
        g[i] *= 0.0625 * s1[i];
    }
}

fn rotate_vec(x: &mut [f32]) {
    for block in x.chunks_exact_mut(FWHT_N) {
        if let Ok(g) = <&mut [f32; FWHT_N]>::try_from(block) {
            rotate256(g);
        }
    }
}

fn row_bytes(k: usize) -> usize {
    k / FWHT_N * QT44_GROUP_BYTES
}

/// Quantize one natural-domain weight row (rotated per 256 group first) into
/// qt44 bytes; the symmetric 4-bit scheme of the module spec.
fn quantize_row(row: &[f32], out: &mut [u8]) -> Result<(), String> {
    let groups = row.len() / FWHT_N;
    if row.len() % FWHT_N != 0 || out.len() != groups * QT44_GROUP_BYTES {
        return Err("quantize_row: bad row geometry".to_string());
    }
    for g in 0..groups {
        let mut r = [0f32; FWHT_N];
        r.copy_from_slice(&row[g * FWHT_N..(g + 1) * FWHT_N]);
        if r.iter().any(|v| !v.is_finite()) {
            return Err("non-finite weight".to_string());
        }
        rotate256(&mut r);
        let ob = &mut out[g * QT44_GROUP_BYTES..(g + 1) * QT44_GROUP_BYTES];
        ob.fill(0);
        for h in 0..2 {
            let half = &r[h * 128..(h + 1) * 128];
            let maxabs = half.iter().fold(0f32, |a, &v| a.max(v.abs()));
            if maxabs == 0.0 {
                continue;
            }
            let d = maxabs / 8.0;
            let s16 = f32_to_f16_bits(d);
            let z16 = f32_to_f16_bits(-8.0 * d);
            if (s16 & 0x7c00) == 0x7c00 || (z16 & 0x7c00) == 0x7c00 {
                return Err(format!(
                    "weight magnitude {maxabs} overflows the f16 scale/zero"
                ));
            }
            ob[4 * h..4 * h + 2].copy_from_slice(&s16.to_le_bytes());
            ob[4 * h + 2..4 * h + 4].copy_from_slice(&z16.to_le_bytes());
            for (j, &w) in half.iter().enumerate() {
                let q = ((w / d).round_ties_even() + 8.0).clamp(0.0, 15.0) as u8;
                let col = h * 128 + j;
                let byte = &mut ob[8 + col / 2];
                if col % 2 == 0 {
                    *byte |= q;
                } else {
                    *byte |= q << 4;
                }
            }
        }
    }
    Ok(())
}

/// Decode one qt44 row (rotated domain) with the f16-rounded scale/zero.
fn dequant_row(bytes: &[u8], k: usize, out: &mut [f32]) {
    for g in 0..k / FWHT_N {
        let ob = &bytes[g * QT44_GROUP_BYTES..(g + 1) * QT44_GROUP_BYTES];
        let o = &mut out[g * FWHT_N..(g + 1) * FWHT_N];
        for h in 0..2 {
            let s = f16_bits_to_f32(u16::from_le_bytes([ob[4 * h], ob[4 * h + 1]]));
            let z = f16_bits_to_f32(u16::from_le_bytes([ob[4 * h + 2], ob[4 * h + 3]]));
            for i in 0..64 {
                let b = ob[8 + h * 64 + i];
                o[h * 128 + 2 * i] = s * (b & 15) as f32 + z;
                o[h * 128 + 2 * i + 1] = s * (b >> 4) as f32 + z;
            }
        }
    }
}

fn quantize_matrix(w: &[f32], m: usize, k: usize, threads: usize) -> Result<Vec<u8>, String> {
    if k == 0 || k % FWHT_N != 0 || w.len() != m * k {
        return Err(format!(
            "quantize_matrix: {} values for [{m}, {k}]",
            w.len()
        ));
    }
    let rb = row_bytes(k);
    let mut out = vec![0u8; m * rb];
    par_chunks(
        threads,
        m,
        rb,
        &mut out,
        &|r0: usize, r1: usize, chunk: &mut [u8]| {
            for r in r0..r1 {
                quantize_row(
                    &w[r * k..(r + 1) * k],
                    &mut chunk[(r - r0) * rb..(r - r0 + 1) * rb],
                )
                .map_err(|e| format!("row {r}: {e}"))?;
            }
            Ok(())
        },
    )?;
    Ok(out)
}

// ───────────────────────────── geometry and CPU references ─────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
struct Geom {
    e: usize,
    k: usize,
    c: usize,
    i: usize,
    ipad: usize,
}

const SMALL: Geom = Geom {
    e: 16,
    k: 3,
    c: 256,
    i: 192,
    ipad: 256,
};
const REAL: Geom = Geom {
    e: 128,
    k: 6,
    c: 2560,
    i: 1088,
    ipad: 1280,
};
const DENSE_MI: usize = 12032;

#[derive(Clone, Copy, Debug, PartialEq)]
enum RouterSel {
    SigmoidBias,
    Softmax,
}

const ROUTER_EPS: f32 = 1e-20;
const ROUTER_SCALE: f32 = 1.0;

fn act_ref(act: FfnActivation, g: f32, u: f32) -> Result<f32, String> {
    let (g, u) = (g as f64, u as f64);
    let sig = 1.0 / (1.0 + (-g).exp());
    match act {
        FfnActivation::Situ {
            gate_beta,
            value_beta,
        } => {
            let (gb, vb) = (gate_beta as f64, value_beta as f64);
            Ok((((gb * (g / gb).tanh()) * sig) * (vb * (u / vb).tanh())) as f32)
        }
        FfnActivation::SwiGlu => Ok((g * sig * u) as f32),
        other => Err(format!("CPU reference has no activation {other:?}")),
    }
}

/// Host qt44 bytes of one gated FFN (routed expert, shared expert or dense
/// layer): gate `[mi, c]`, up `[mi, c]`, down `[c, down_k]`.
struct GatedQ {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

/// Natural-domain f32 weights of one gated FFN: gate/up `[i, c]`, down `[c, i]`.
struct Nat {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// Scale (AWQ), zero-extend the down K from `i` to `ipad` with `pad_fill`,
/// rotate and quantize one gated FFN.
#[allow(clippy::too_many_arguments)]
fn build_gated_q(
    nat: Nat,
    c: usize,
    i: usize,
    ipad: usize,
    pad_fill: f32,
    awq_in: Option<&[f32]>,
    awq_down: Option<&[f32]>,
    threads: usize,
) -> Result<GatedQ, String> {
    if nat.gate.len() != i * c || nat.up.len() != i * c || nat.down.len() != c * i {
        return Err(format!(
            "gated weights have {}/{}/{} values for gate/up/down [{i},{c}]/[{c},{i}]",
            nat.gate.len(),
            nat.up.len(),
            nat.down.len()
        ));
    }
    let mut gate = nat.gate;
    let mut up = nat.up;
    if let Some(s) = awq_in {
        if s.len() != c {
            return Err(format!(
                "awq gate/up vector has {} entries, hidden is {c}",
                s.len()
            ));
        }
        for m in [&mut gate, &mut up] {
            for row in m.chunks_mut(c) {
                for (v, sv) in row.iter_mut().zip(s) {
                    *v *= *sv;
                }
            }
        }
    }
    if let Some(s) = awq_down {
        if s.len() != ipad {
            return Err(format!(
                "awq down vector has {} entries, down K is {ipad}",
                s.len()
            ));
        }
    }
    let mut dpad = vec![pad_fill; c * ipad];
    for r in 0..c {
        let dst = &mut dpad[r * ipad..(r + 1) * ipad];
        for j in 0..i {
            dst[j] = nat.down[r * i + j];
        }
        if let Some(s) = awq_down {
            for j in 0..ipad {
                dst[j] *= s[j];
            }
        }
    }
    Ok(GatedQ {
        gate: quantize_matrix(&gate, i, c, threads)?,
        up: quantize_matrix(&up, i, c, threads)?,
        down: quantize_matrix(&dpad, c, ipad, threads)?,
    })
}

struct GatedRef<'a> {
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
    awq_down: Option<&'a [f32]>,
}

/// CPU gated FFN of `t` tokens from their rotated (AWQ-divided) inputs:
/// `down(rotate(pad(act(bf16(gate), bf16(up)))))`, f64 dots, per-mode
/// rounding. Output is token-major `[t, c]`.
#[allow(clippy::too_many_arguments)]
fn ffn_batch(
    w: &GatedRef<'_>,
    xrot: &[f32],
    t: usize,
    c: usize,
    i: usize,
    ipad: usize,
    act: FfnActivation,
    bf16: bool,
    pad_act: f32,
    threads: usize,
) -> Result<Vec<f32>, String> {
    let rbc = row_bytes(c);
    let rbd = row_bytes(ipad);
    if xrot.len() != t * c
        || w.gate.len() != i * rbc
        || w.up.len() != i * rbc
        || w.down.len() != c * rbd
    {
        return Err("ffn_batch: operand sizes disagree with the geometry".to_string());
    }
    act_ref(act, 0.0, 0.0)?;
    let mut gu = vec![0f32; i * 2 * t];
    par_chunks(
        threads,
        i,
        2 * t,
        &mut gu,
        &|r0: usize, r1: usize, chunk: &mut [f32]| {
            let mut row = vec![0f32; c];
            for r in r0..r1 {
                let o = &mut chunk[(r - r0) * 2 * t..(r - r0 + 1) * 2 * t];
                dequant_row(&w.gate[r * rbc..(r + 1) * rbc], c, &mut row);
                for tk in 0..t {
                    o[tk] = dot_f64(&row, &xrot[tk * c..(tk + 1) * c]) as f32;
                }
                dequant_row(&w.up[r * rbc..(r + 1) * rbc], c, &mut row);
                for tk in 0..t {
                    o[t + tk] = dot_f64(&row, &xrot[tk * c..(tk + 1) * c]) as f32;
                }
            }
            Ok(())
        },
    )?;
    let mut rot = vec![0f32; t * ipad];
    for tk in 0..t {
        let dst = &mut rot[tk * ipad..(tk + 1) * ipad];
        for r in 0..i {
            let mut gv = gu[r * 2 * t + tk];
            let mut uv = gu[r * 2 * t + t + tk];
            if bf16 {
                gv = bf16r(gv);
                uv = bf16r(uv);
            }
            let mut a = act_ref(act, gv, uv)?;
            if bf16 {
                a = bf16r(a);
            }
            dst[r] = a;
        }
        for v in dst.iter_mut().take(ipad).skip(i) {
            *v = pad_act;
        }
        if let Some(s) = w.awq_down {
            if s.len() != ipad {
                return Err("ffn_batch: down awq length".to_string());
            }
            for j in 0..ipad {
                dst[j] /= s[j];
            }
        }
        rotate_vec(dst);
    }
    let mut yr = vec![0f32; c * t];
    par_chunks(
        threads,
        c,
        t,
        &mut yr,
        &|r0: usize, r1: usize, chunk: &mut [f32]| {
            let mut row = vec![0f32; ipad];
            for r in r0..r1 {
                dequant_row(&w.down[r * rbd..(r + 1) * rbd], ipad, &mut row);
                for tk in 0..t {
                    let mut v = dot_f64(&row, &rot[tk * ipad..(tk + 1) * ipad]) as f32;
                    if bf16 {
                        v = bf16r(v);
                    }
                    chunk[(r - r0) * t + tk] = v;
                }
            }
            Ok(())
        },
    )?;
    let mut y = vec![0f32; t * c];
    for r in 0..c {
        for tk in 0..t {
            y[tk * c + r] = yr[r * t + tk];
        }
    }
    Ok(y)
}

/// Rotated, AWQ-divided inputs of every token.
fn rotated_inputs(x: &[f32], n: usize, c: usize, awq: Option<&[f32]>) -> Vec<f32> {
    let mut xr = x[..n * c].to_vec();
    for row in xr.chunks_mut(c) {
        if let Some(s) = awq {
            for (v, sv) in row.iter_mut().zip(s) {
                *v /= *sv;
            }
        }
        rotate_vec(row);
    }
    xr
}

/// Router reference: scores (sigmoid or softmax probabilities, f32), biased
/// selection scores, top-k ids (descending biased score, tie -> lower id) and
/// weights.
struct RouteRef {
    scores: Vec<f32>,
    biased: Vec<f32>,
    idx: Vec<i32>,
}

/// Weights of the ids `ids` selected from one row of `scores`:
/// sigmoid -> `s/(sum s + eps)*scale`, softmax -> `p/sum p` (f32, rank order).
fn route_weights(kind: RouterSel, scores: &[f32], ids: &[i32]) -> Vec<f32> {
    let pick = |id: i32| scores.get(id.max(0) as usize).copied().unwrap_or(f32::NAN);
    let mut sum = 0f32;
    for &id in ids {
        sum += pick(id);
    }
    ids.iter()
        .map(|&id| match kind {
            RouterSel::SigmoidBias => pick(id) / (sum + ROUTER_EPS) * ROUTER_SCALE,
            RouterSel::Softmax => pick(id) / sum,
        })
        .collect()
}

fn route_ref(
    kind: RouterSel,
    x: &[f32],
    router_w: &[f32],
    bias: &[f32],
    n: usize,
    g: &Geom,
) -> RouteRef {
    let (e, c, k) = (g.e, g.c, g.k);
    let mut scores = vec![0f32; n * e];
    let mut biased = vec![0f32; n * e];
    let mut idx = vec![0i32; n * k];
    for t in 0..n {
        let xr = &x[t * c..(t + 1) * c];
        let logits: Vec<f64> = (0..e)
            .map(|j| dot_f64(&router_w[j * c..(j + 1) * c], xr))
            .collect();
        let srow = &mut scores[t * e..(t + 1) * e];
        match kind {
            RouterSel::SigmoidBias => {
                for j in 0..e {
                    srow[j] = (1.0 / (1.0 + (-logits[j]).exp())) as f32;
                }
            }
            RouterSel::Softmax => {
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = logits.iter().map(|l| (l - m).exp()).sum();
                for j in 0..e {
                    srow[j] = ((logits[j] - m).exp() / z) as f32;
                }
            }
        }
        let brow = &mut biased[t * e..(t + 1) * e];
        for j in 0..e {
            brow[j] = match kind {
                RouterSel::SigmoidBias => srow[j] + bias[j],
                RouterSel::Softmax => srow[j],
            };
        }
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| brow[b].total_cmp(&brow[a]).then(a.cmp(&b)));
        let ids: Vec<i32> = order[..k].iter().map(|&j| j as i32).collect();
        idx[t * k..(t + 1) * k].copy_from_slice(&ids);
    }
    RouteRef {
        scores,
        biased,
        idx,
    }
}

struct RefOut {
    routed: Vec<f32>,
    shared: Option<Vec<f32>>,
    out: Vec<f32>,
}

// ───────────────────────────── weight sets ─────────────────────────────────

/// Host side of one uploaded weight set (what the CPU reference decodes).
struct WeightSet {
    geom: Geom,
    /// Experts `< experts.len()` are quantized; the rest are unquantized
    /// (`expert_limit` debugging only).
    experts: Vec<GatedQ>,
    shared: Option<GatedQ>,
    awq_gu: Option<Vec<f32>>,
    awq_down: Vec<Vec<f32>>,
    awq_sh_down: Option<Vec<f32>>,
    dev: Option<WeightDev>,
}

impl WeightSet {
    fn partial(&self) -> bool {
        self.experts.len() < self.geom.e
    }
}

/// Device tensors of a gated FFN.
struct GatedDev {
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
    /// F32 `[hidden]` gate/up AWQ divide owned by this FFN (dense layer).
    awq_in: Option<GpuTensor>,
    /// F32 `[down_k]` AWQ divide of the down projection.
    awq_down: Option<GpuTensor>,
}

struct WeightDev {
    gate_up: Vec<GpuTensor>,
    down: Vec<GpuTensor>,
    gate_up_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    awq_gu: Option<GpuTensor>,
    awq_down: Vec<GpuTensor>,
    awq_down_ptrs: Option<GpuTensor>,
    shared: Option<GatedDev>,
}

fn free_all(gpu: &mut Gpu, ts: impl IntoIterator<Item = GpuTensor>) {
    for t in ts {
        let _ = gpu.free_tensor(t);
    }
}

impl GatedDev {
    fn free(self, gpu: &mut Gpu) {
        free_all(
            gpu,
            [self.gate, self.up, self.down]
                .into_iter()
                .chain(self.awq_in)
                .chain(self.awq_down),
        );
    }
}

impl WeightDev {
    fn free(self, gpu: &mut Gpu) {
        free_all(gpu, self.gate_up);
        free_all(gpu, self.down);
        free_all(gpu, [self.gate_up_ptrs, self.down_ptrs]);
        free_all(gpu, self.awq_gu);
        free_all(gpu, self.awq_down);
        free_all(gpu, self.awq_down_ptrs);
        if let Some(s) = self.shared {
            s.free(gpu);
        }
    }
}

impl WeightSet {
    fn free(mut self, gpu: &mut Gpu) {
        if let Some(d) = self.dev.take() {
            d.free(gpu);
        }
    }
}

fn up_f32(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, &[data.len()])
        .map_err(|e| format!("upload f32 [{}]: {e}", data.len()))
}

fn up_bytes(gpu: &Gpu, data: &[u8]) -> Result<GpuTensor, String> {
    gpu.upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload {} bytes: {e}", data.len()))
}

fn up_ptr_table(gpu: &Gpu, ts: &[&GpuTensor]) -> Result<GpuTensor, String> {
    let mut b = Vec::with_capacity(ts.len() * 8);
    for t in ts {
        b.extend_from_slice(&(t.buf.as_ptr() as usize as u64).to_le_bytes());
    }
    up_bytes(gpu, &b)
}

fn alloc_f32(gpu: &mut Gpu, words: usize) -> Result<GpuTensor, String> {
    gpu.alloc_tensor(&[words.max(1)], DType::F32)
        .map_err(|e| format!("alloc {words} words: {e}"))
}

fn down_f32(gpu: &Gpu, t: &GpuTensor, words: usize) -> Result<Vec<f32>, String> {
    let mut v = gpu.download_f32(t).map_err(|e| format!("download: {e}"))?;
    if v.len() < words {
        return Err(format!("download returned {} words, need {words}", v.len()));
    }
    v.truncate(words);
    Ok(v)
}

/// Overwrite a tensor with quiet NaN words (an unwritten output stays NaN).
fn poison(gpu: &Gpu, t: &GpuTensor) -> Result<(), String> {
    gpu.bind_thread().map_err(|e| format!("bind: {e}"))?;
    let n = t.numel() * 4;
    let mut b = Vec::with_capacity(n);
    for _ in 0..t.numel() {
        b.extend_from_slice(&0x7fc0_0000u32.to_le_bytes());
    }
    gpu.hip
        .memcpy_htod(&t.buf, &b)
        .map_err(|e| format!("poison: {e}"))
}

fn upload_gated(
    gpu: &mut Gpu,
    q: &GatedQ,
    awq_in: Option<&[f32]>,
    awq_down: Option<&[f32]>,
) -> Result<GatedDev, String> {
    Ok(GatedDev {
        gate: up_bytes(gpu, &q.gate)?,
        up: up_bytes(gpu, &q.up)?,
        down: up_bytes(gpu, &q.down)?,
        awq_in: match awq_in {
            Some(s) => Some(up_f32(gpu, s)?),
            None => None,
        },
        awq_down: match awq_down {
            Some(s) => Some(up_f32(gpu, s)?),
            None => None,
        },
    })
}

/// Deterministic positive AWQ divide: `0.5 + U[0,1)`; entries `>= live` are 1.
fn awq_vec(seed: u64, tag: u64, live: usize, total: usize) -> Vec<f32> {
    let mut g = Lcg::new(mix_seed(&[seed, 0xa3c9, tag]));
    (0..total)
        .map(|j| {
            if j < live {
                0.5 + g.uniform() as f32
            } else {
                1.0
            }
        })
        .collect()
}

#[derive(Clone)]
enum WSrc {
    File(PathBuf),
    Synth { seed: u64 },
}

#[derive(Clone)]
struct WSpec {
    src: WSrc,
    geom: Geom,
    /// Value written into the pad columns of the down projections.
    pad_fill: f32,
    shared: bool,
    awq: bool,
    expert_limit: Option<usize>,
}

enum Reader {
    File(StFile),
    Synth(u64),
}

/// Natural weights of a synthetic gated FFN: gate `2.5/sqrt(c)`, up
/// `10/sqrt(c)`, down `1/sqrt(i)` normals (SiTU lands in its nonlinear range).
fn synth_gated(seed: u64, tag: u64, c: usize, i: usize) -> Nat {
    let mut g = Lcg::new(mix_seed(&[seed, tag, 1]));
    let gate = normal_vec(&mut g, i * c, 2.5 / (c as f32).sqrt(), false);
    let mut g = Lcg::new(mix_seed(&[seed, tag, 2]));
    let up = normal_vec(&mut g, i * c, 10.0 / (c as f32).sqrt(), false);
    let mut g = Lcg::new(mix_seed(&[seed, tag, 3]));
    let down = normal_vec(&mut g, c * i, 1.0 / (i as f32).sqrt(), false);
    Nat { gate, up, down }
}

const SHARED_TAG: u64 = 0xffff_0001;
const DENSE_TAG: u64 = 0xffff_0002;

impl Reader {
    fn open(src: &WSrc) -> Result<Self, String> {
        Ok(match src {
            WSrc::File(p) => Self::File(StFile::open(p)?),
            WSrc::Synth { seed } => Self::Synth(*seed),
        })
    }

    fn gated_from_file(st: &StFile, base: &str, c: usize, i: usize) -> Result<Nat, String> {
        let get = |what: &str, shape: [usize; 2]| -> Result<Vec<f32>, String> {
            let name = st.lookup(&format!("{base}.{what}_proj.weight"))?;
            st.read_shaped(&name, &shape)
        };
        Ok(Nat {
            gate: get("gate", [i, c])?,
            up: get("up", [i, c])?,
            down: get("down", [c, i])?,
        })
    }

    fn expert(&self, e: usize, g: &Geom) -> Result<Nat, String> {
        match self {
            Self::File(st) => Self::gated_from_file(st, &format!("moe.experts.{e}"), g.c, g.i),
            Self::Synth(seed) => Ok(synth_gated(*seed, e as u64, g.c, g.i)),
        }
    }

    fn shared(&self, g: &Geom) -> Result<Nat, String> {
        match self {
            Self::File(st) => Self::gated_from_file(st, "moe.shared_experts", g.c, g.i),
            Self::Synth(seed) => Ok(synth_gated(*seed, SHARED_TAG, g.c, g.i)),
        }
    }
}

/// Quantize (threaded over experts) and upload one weight set.
fn build_wset(gpu: &mut Gpu, spec: &WSpec) -> Result<WeightSet, String> {
    let g = spec.geom;
    if g.c % FWHT_N != 0 || g.ipad % FWHT_N != 0 || g.ipad < g.i {
        return Err(format!(
            "geometry {g:?} is not a multiple-of-256 qt44 geometry"
        ));
    }
    let seed = match &spec.src {
        WSrc::Synth { seed } => *seed,
        WSrc::File(p) => str_seed(&p.display().to_string()),
    };
    let awq_gu = spec.awq.then(|| awq_vec(seed, 0, g.c, g.c));
    let awq_down: Vec<Vec<f32>> = if spec.awq {
        (0..g.e)
            .map(|e| awq_vec(seed, 1 + e as u64, g.i, g.ipad))
            .collect()
    } else {
        Vec::new()
    };
    let awq_sh_down = (spec.awq && spec.shared).then(|| awq_vec(seed, SHARED_TAG, g.i, g.ipad));
    let n_q = spec.expert_limit.unwrap_or(g.e).min(g.e);
    if n_q == 0 {
        return Err("expert_limit 0: nothing to quantize".to_string());
    }

    let workers = worker_count().min(n_q);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let results: Vec<Result<Vec<(usize, GatedQ)>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<Vec<(usize, GatedQ)>, String> {
                    let rd = Reader::open(&spec.src)
                        .inspect_err(|_| failed.store(true, Ordering::Relaxed))?;
                    let mut v = Vec::new();
                    loop {
                        if failed.load(Ordering::Relaxed) {
                            break;
                        }
                        let e = next.fetch_add(1, Ordering::Relaxed);
                        if e >= n_q {
                            break;
                        }
                        let q = (|| -> Result<GatedQ, String> {
                            let nat = rd.expert(e, &g)?;
                            build_gated_q(
                                nat,
                                g.c,
                                g.i,
                                g.ipad,
                                spec.pad_fill,
                                awq_gu.as_deref(),
                                awq_down.get(e).map(|v| v.as_slice()),
                                1,
                            )
                        })()
                        .map_err(|er: String| {
                            failed.store(true, Ordering::Relaxed);
                            format!("expert {e}: {er}")
                        })?;
                        v.push((e, q));
                    }
                    Ok(v)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("quantize worker panicked".to_string()))
            })
            .collect()
    });
    let mut slots: Vec<Option<GatedQ>> = (0..n_q).map(|_| None).collect();
    for r in results {
        for (e, q) in r? {
            slots[e] = Some(q);
        }
    }
    let experts: Vec<GatedQ> = slots
        .into_iter()
        .enumerate()
        .map(|(e, q)| q.ok_or_else(|| format!("expert {e} was not quantized")))
        .collect::<Result<_, _>>()?;

    let shared = if spec.shared {
        let rd = Reader::open(&spec.src)?;
        let nat = rd.shared(&g).map_err(|e| format!("shared expert: {e}"))?;
        Some(
            build_gated_q(
                nat,
                g.c,
                g.i,
                g.ipad,
                spec.pad_fill,
                awq_gu.as_deref(),
                awq_sh_down.as_deref(),
                worker_count(),
            )
            .map_err(|e| format!("shared expert: {e}"))?,
        )
    } else {
        None
    };

    // Upload. Experts `>= n_q` alias expert 0 so every table entry is a valid
    // pointer (routing is rejected before it could select them).
    let mut ws = WeightSet {
        geom: g,
        experts,
        shared,
        awq_gu,
        awq_down,
        awq_sh_down,
        dev: None,
    };
    let up = (|| -> Result<WeightDev, String> {
        let mut gate_up = Vec::new();
        let mut down = Vec::new();
        let mut awq_dev = Vec::new();
        for (e, q) in ws.experts.iter().enumerate() {
            let mut cat = Vec::with_capacity(q.gate.len() + q.up.len());
            cat.extend_from_slice(&q.gate);
            cat.extend_from_slice(&q.up);
            gate_up.push(up_bytes(gpu, &cat)?);
            down.push(up_bytes(gpu, &q.down)?);
            if spec.awq {
                awq_dev.push(up_f32(gpu, &ws.awq_down[e])?);
            }
        }
        let table_of = |gpu: &Gpu, v: &[GpuTensor]| -> Result<GpuTensor, String> {
            let refs: Vec<&GpuTensor> = (0..g.e).map(|e| &v[e.min(n_q - 1)]).collect();
            up_ptr_table(gpu, &refs)
        };
        let gate_up_ptrs = table_of(gpu, &gate_up)?;
        let down_ptrs = table_of(gpu, &down)?;
        let (awq_gu, awq_down_ptrs) = if spec.awq {
            let gu = ws.awq_gu.as_deref().map(|s| up_f32(gpu, s)).transpose()?;
            let tbl = table_of(gpu, &awq_dev)?;
            (gu, Some(tbl))
        } else {
            (None, None)
        };
        let shared = match &ws.shared {
            Some(q) => Some(upload_gated(gpu, q, None, ws.awq_sh_down.as_deref())?),
            None => None,
        };
        Ok(WeightDev {
            gate_up,
            down,
            gate_up_ptrs,
            down_ptrs,
            awq_gu,
            awq_down: awq_dev,
            awq_down_ptrs,
            shared,
        })
    })()?;
    ws.dev = Some(up);
    Ok(ws)
}

// ───────────────────────────── CPU module reference ────────────────────────

/// CPU MoE reference from the dequantized qt44 weights, fed with the routing
/// the GPU produced. Expert math is threaded over experts.
fn cpu_moe_ref(
    c: &CaseData,
    ws: &WeightSet,
    idx: &[i32],
    w: &[f32],
    pad_act: f32,
) -> Result<RefOut, String> {
    let g = c.geom;
    let (n, k, cc) = (c.n, g.k, g.c);
    if idx.len() != n * k || w.len() != n * k {
        return Err(format!(
            "routing tables hold {}/{} entries, expected {}",
            idx.len(),
            w.len(),
            n * k
        ));
    }
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); g.e];
    for slot in 0..n * k {
        let e = idx[slot];
        if e < 0 || e as usize >= g.e {
            return Err(format!(
                "routing selected invalid expert {e} (token {}, rank {})",
                slot / k,
                slot % k
            ));
        }
        let e = e as usize;
        if e >= ws.experts.len() {
            return Err(format!(
                "routing selected expert {e}, which `expert_limit` left unquantized (token {}, rank {})",
                slot / k,
                slot % k
            ));
        }
        buckets[e].push(slot);
    }
    for t in 0..n {
        let mut ids: Vec<i32> = idx[t * k..(t + 1) * k].to_vec();
        ids.sort_unstable();
        if ids.windows(2).any(|p| p[0] == p[1]) {
            return Err(format!(
                "routing selected one expert twice for token {t}: {:?}",
                &idx[t * k..(t + 1) * k]
            ));
        }
    }
    let xrot = rotated_inputs(&c.x, n, cc, ws.awq_gu.as_deref());
    let active: Vec<usize> = (0..g.e).filter(|&e| !buckets[e].is_empty()).collect();
    let next = AtomicUsize::new(0);
    let workers = worker_count().min(active.len().max(1));
    let want_shared = c.shared;
    let experts: &[GatedQ] = &ws.experts;
    let awq_downs: &[Vec<f32>] = &ws.awq_down;
    let shared_q: Option<&GatedQ> = ws.shared.as_ref();
    let awq_sh_down: Option<&[f32]> = ws.awq_sh_down.as_deref();
    let (expert_res, shared_res) = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<Vec<(usize, Vec<f32>)>, String> {
                    let mut v = Vec::new();
                    loop {
                        let a = next.fetch_add(1, Ordering::Relaxed);
                        if a >= active.len() {
                            break;
                        }
                        let e = active[a];
                        let slots = &buckets[e];
                        let mut xin = Vec::with_capacity(slots.len() * cc);
                        for &sl in slots {
                            let t = sl / k;
                            xin.extend_from_slice(&xrot[t * cc..(t + 1) * cc]);
                        }
                        let q = &experts[e];
                        let r = GatedRef {
                            gate: &q.gate,
                            up: &q.up,
                            down: &q.down,
                            awq_down: awq_downs.get(e).map(|v| v.as_slice()),
                        };
                        let y = ffn_batch(
                            &r,
                            &xin,
                            slots.len(),
                            cc,
                            g.i,
                            g.ipad,
                            c.act,
                            c.bf16,
                            pad_act,
                            1,
                        )
                        .map_err(|er| format!("expert {e}: {er}"))?;
                        for (j, &sl) in slots.iter().enumerate() {
                            v.push((sl, y[j * cc..(j + 1) * cc].to_vec()));
                        }
                    }
                    Ok(v)
                })
            })
            .collect();
        let sh = want_shared.then(|| {
            s.spawn(|| -> Result<Vec<f32>, String> {
                let q = shared_q.ok_or("weight set has no shared expert")?;
                let r = GatedRef {
                    gate: &q.gate,
                    up: &q.up,
                    down: &q.down,
                    awq_down: awq_sh_down,
                };
                ffn_batch(&r, &xrot, n, cc, g.i, g.ipad, c.act, c.bf16, pad_act, 4)
            })
        });
        let er: Vec<_> = hs
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("reference worker panicked".to_string()))
            })
            .collect();
        let sr = sh.map(|h| {
            h.join()
                .unwrap_or_else(|_| Err("shared reference panicked".to_string()))
        });
        (er, sr)
    });
    let mut ys = vec![0f32; n * k * cc];
    for r in expert_res {
        for (sl, y) in r? {
            ys[sl * cc..(sl + 1) * cc].copy_from_slice(&y);
        }
    }
    let shared = match shared_res {
        Some(r) => Some(r?),
        None => None,
    };
    let mut routed = vec![0f32; n * cc];
    let mut out = vec![0f32; n * cc];
    for t in 0..n {
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&r| idx[t * k + r]);
        let acc = &mut routed[t * cc..(t + 1) * cc];
        for &r in &order {
            let sl = t * k + r;
            let wt = w[sl];
            let y = &ys[sl * cc..(sl + 1) * cc];
            for j in 0..cc {
                let prod = y[j] * wt;
                acc[j] += prod;
            }
        }
        if c.bf16 {
            for v in acc.iter_mut() {
                *v = bf16r(*v);
            }
        }
        for j in 0..cc {
            let mut v = acc[j];
            if let Some(sh) = &shared {
                v += sh[t * cc + j];
            }
            out[t * cc + j] = if c.bf16 { bf16r(v) } else { v };
        }
    }
    Ok(RefOut {
        routed,
        shared,
        out,
    })
}

// ───────────────────────────── cases ───────────────────────────────────────

#[derive(Clone, Debug)]
enum Extra {
    /// Every token's selection contains these experts.
    Coverage(Vec<usize>),
    /// Every token selects exactly this set.
    SameSet(Vec<usize>),
    /// Every token's ordered selection equals this list (tie policy).
    Order(Vec<i32>),
    /// Weights sum to 1 and differ by more than 10x within every token.
    Unequal,
    /// Down pad columns are poisoned: the output must not depend on them.
    PadGuard,
}

struct Expected {
    scores: Option<Vec<f32>>,
    idx: Vec<i32>,
    w: Vec<f32>,
    routed: Vec<f32>,
    shared: Option<Vec<f32>>,
    out: Vec<f32>,
}

struct CaseData {
    name: String,
    geom: Geom,
    n: usize,
    bf16: bool,
    act: FfnActivation,
    router: RouterSel,
    shared: bool,
    x: Vec<f32>,
    router_w: Vec<f32>,
    e_bias: Vec<f32>,
    expected: Option<Expected>,
    /// Biased-score distance under which a different selection counts as a
    /// near-tie (synthetic cases only; fixtures and tie tests are exact).
    tie_tol: f32,
    extras: Vec<Extra>,
}

fn act_betas(a: FfnActivation) -> (f32, f32) {
    match a {
        FfnActivation::Situ {
            gate_beta,
            value_beta,
        } => (gate_beta, value_beta),
        _ => (4.0, 25.0),
    }
}

fn dims_of(c: &CaseData) -> RaonMoeDims {
    let (gate_beta, value_beta) = act_betas(c.act);
    RaonMoeDims {
        hidden: c.geom.c,
        n_experts: c.geom.e,
        top_k: c.geom.k,
        logical_mi: c.geom.i,
        down_k: c.geom.ipad,
        dense_mi: DENSE_MI,
        gate_beta,
        value_beta,
        router_eps: ROUTER_EPS,
        router_scale: ROUTER_SCALE,
        tiers: hipfire_arch_raon::precision::RaonSiteTiers::ALL_F16,
    }
}

fn proto_label(p: GroupedMoeProtocol) -> &'static str {
    match p {
        GroupedMoeProtocol::Indexed => "indexed",
        GroupedMoeProtocol::Grouped => "grouped",
        GroupedMoeProtocol::Auto => "auto",
    }
}

fn mode_label(bf16: bool) -> &'static str {
    if bf16 {
        "bf16"
    } else {
        "f32"
    }
}

// ───────────────────────────── GPU execution ───────────────────────────────

fn qt44_ref<'a>(t: &'a GpuTensor, m: usize, k: usize, awq: Option<&'a GpuTensor>) -> WeightRef<'a> {
    WeightRef {
        buf: t,
        dtype: DType::MQ4G256V2,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: awq,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

/// Projections of a gated FFN; gate/up carry `awq_in`, the down its own divide.
fn gated_refs<'a>(
    d: &'a GatedDev,
    awq_in: Option<&'a GpuTensor>,
    c: usize,
    mi: usize,
    down_k: usize,
) -> GroupedMoeShared<'a> {
    GroupedMoeShared::new(
        qt44_ref(&d.gate, mi, c, awq_in),
        qt44_ref(&d.up, mi, c, awq_in),
        qt44_ref(&d.down, c, down_k, d.awq_down.as_ref()),
    )
}

struct DevIn {
    x: GpuTensor,
    router_w: GpuTensor,
    bias: GpuTensor,
}

impl DevIn {
    fn new(gpu: &mut Gpu, c: &CaseData) -> Result<Self, String> {
        let x = up_f32(gpu, &c.x)?;
        // The production loader path: BF16 when every value is BF16-exact (the
        // generator rounds, as the checkpoint is BF16), else F32.
        let router_w = hipfire_arch_raon::upload_rows_weight(
            gpu,
            &c.router_w,
            c.geom.e,
            c.geom.c,
            "router_w",
        )?
        .buf;
        let bias = up_f32(gpu, &c.e_bias)?;
        Ok(Self { x, router_w, bias })
    }
    fn free(self, gpu: &mut Gpu) {
        free_all(gpu, [self.x, self.router_w, self.bias]);
    }
}

struct GpuRun {
    out: Vec<f32>,
    routed: Vec<f32>,
    shared: Option<Vec<f32>>,
    idx: Vec<i32>,
    w: Vec<f32>,
    ms: f64,
}

enum RunErr {
    /// A named capability refusal of the production op.
    Refusal(String),
    Fail(String),
}

fn classify(e: DispatchError) -> RunErr {
    let s = e.to_string();
    if matches!(e, DispatchError::UnsupportedVariant { .. }) {
        RunErr::Refusal(s)
    } else {
        RunErr::Fail(s)
    }
}

fn fail<E: std::fmt::Display>(e: E) -> RunErr {
    RunErr::Fail(e.to_string())
}

fn gpu_moe_run(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    c: &CaseData,
    wd: &WeightDev,
    dev: &DevIn,
    scratch: &RaonMoeScratch,
    protocol: GroupedMoeProtocol,
) -> Result<GpuRun, RunErr> {
    let words = c.n * c.geom.c;
    let out_t = up_f32(gpu, &vec![f32::NAN; words]).map_err(RunErr::Fail)?;
    let routed_t = match up_f32(gpu, &vec![f32::NAN; words]) {
        Ok(t) => t,
        Err(e) => {
            free_all(gpu, [out_t]);
            return Err(RunErr::Fail(e));
        }
    };
    let r = gpu_moe_run_inner(gpu, ctx, c, wd, dev, scratch, protocol, &out_t, &routed_t);
    free_all(gpu, [out_t, routed_t]);
    r
}

#[allow(clippy::too_many_arguments)]
fn gpu_moe_run_inner(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    c: &CaseData,
    wd: &WeightDev,
    dev: &DevIn,
    scratch: &RaonMoeScratch,
    protocol: GroupedMoeProtocol,
    out_t: &GpuTensor,
    routed_t: &GpuTensor,
) -> Result<GpuRun, RunErr> {
    let g = c.geom;
    let dims = dims_of(c);
    dims.validate().map_err(RunErr::Fail)?;
    if c.shared && wd.shared.is_none() {
        return Err(RunErr::Fail("weight set has no shared expert".to_string()));
    }
    let shared_refs = match &wd.shared {
        Some(d) => gated_refs(d, wd.awq_gu.as_ref(), g.c, g.i, g.ipad),
        None => GroupedMoeShared::new(
            qt44_ref(&wd.gate_up_ptrs, g.i, g.c, None),
            qt44_ref(&wd.gate_up_ptrs, g.i, g.c, None),
            qt44_ref(&wd.down_ptrs, g.c, g.ipad, None),
        ),
    };
    let layer = RaonMoeLayerRefs {
        router_weight: &dev.router_w,
        correction_bias: &dev.bias,
        gate_up_ptrs: &wd.gate_up_ptrs,
        down_ptrs: &wd.down_ptrs,
        gate_up_awq: wd.awq_gu.as_ref(),
        down_awq_ptrs: wd.awq_down_ptrs.as_ref(),
        shared: shared_refs,
        iu4_symmetric: false,
    };
    let mut op = raon_moe_op(
        ctx,
        &dims,
        &layer,
        protocol,
        c.n,
        &dev.x,
        scratch,
        out_t,
        Some(routed_t),
    );
    op.policy.router = match c.router {
        RouterSel::SigmoidBias => RouterKind::SigmoidBiasRenorm {
            eps: ROUTER_EPS,
            scale: ROUTER_SCALE,
        },
        RouterSel::Softmax => RouterKind::SoftmaxRenorm,
    };
    op.policy.activation = c.act;
    op.policy.cast = if c.bf16 {
        CastPolicy::Bf16Boundaries
    } else {
        CastPolicy::Exact32
    };
    if c.router == RouterSel::Softmax {
        op.router.correction_bias = None;
    }
    if !c.shared {
        op.policy.shared = SharedExpertPolicy::Absent;
        op.shared = None;
        op.geometry.shared_mi = 0;
        op.geometry.shared_down_k = 0;
        op.scratch.shared = None;
    }
    poison(gpu, &scratch.topk_indices).map_err(RunErr::Fail)?;
    poison(gpu, &scratch.topk_weights).map_err(RunErr::Fail)?;
    if c.shared {
        poison(gpu, &scratch.shared_out).map_err(RunErr::Fail)?;
    }
    let t0 = Instant::now();
    execute_grouped_moe(gpu, &op).map_err(classify)?;
    gpu.hip.device_synchronize().map_err(fail)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let words = c.n * g.c;
    let slots = c.n * g.k;
    let out = down_f32(gpu, out_t, words).map_err(RunErr::Fail)?;
    let routed = down_f32(gpu, routed_t, words).map_err(RunErr::Fail)?;
    let shared = if c.shared {
        Some(down_f32(gpu, &scratch.shared_out, words).map_err(RunErr::Fail)?)
    } else {
        None
    };
    let idx = down_f32(gpu, &scratch.topk_indices, slots)
        .map_err(RunErr::Fail)?
        .into_iter()
        .map(|v| v.to_bits() as i32)
        .collect();
    let w = down_f32(gpu, &scratch.topk_weights, slots).map_err(RunErr::Fail)?;
    Ok(GpuRun {
        out,
        routed,
        shared,
        idx,
        w,
        ms,
    })
}

// ───────────────────────────── evaluation ──────────────────────────────────

struct Eval {
    entry: Map<String, Value>,
    fails: Vec<String>,
}

impl Eval {
    fn new(name: &str) -> Self {
        let mut entry = Map::new();
        entry.insert("name".into(), json!(name));
        Self {
            entry,
            fails: Vec::new(),
        }
    }

    fn put(&mut self, k: &str, v: Value) {
        self.entry.insert(k.to_string(), v);
    }

    fn fail(&mut self, msg: String) {
        self.fails.push(msg);
    }

    fn section(&mut self, name: &str, got: &[f32], want: &[f32], bf16: bool, grouped: bool) {
        let m = compare(got, want);
        let (ok, band) = band_check(&m, bf16, grouped);
        self.put(name, metrics_json(&m, band, ok));
        if !ok {
            self.fail(format!(
                "{name} outside the {band} band: rel_l2={:.3e} cos={:.7} max_abs={:.3e} nonfinite={}",
                m.rel_l2, m.cos, m.max_abs, m.nonfinite
            ));
        }
    }

    fn finish(mut self) -> Value {
        let pass = self.fails.is_empty();
        let detail = self
            .fails
            .first()
            .cloned()
            .unwrap_or_else(|| "ok".to_string());
        self.entry.insert("pass".into(), json!(pass));
        self.entry.insert("detail".into(), json!(detail));
        if self.fails.len() > 1 {
            self.entry.insert("all_failures".into(), json!(self.fails));
        }
        Value::Object(self.entry)
    }
}

fn pass_of(v: &Value) -> bool {
    v.get("pass").and_then(Value::as_bool).unwrap_or(false)
}

/// Routing comparison; returns (json, failures).
fn check_routing(c: &CaseData, rr: &RouteRef, gidx: &[i32], gw: &[f32]) -> (Value, Vec<String>) {
    let g = c.geom;
    let (n, k, e) = (c.n, g.k, g.e);
    let mut fails = Vec::new();
    let (exp_idx, strict) = match &c.expected {
        Some(x) => (x.idx.as_slice(), true),
        None => (rr.idx.as_slice(), false),
    };
    let tol = if strict { 0.0 } else { c.tie_tol };
    let mut first_mismatch: Option<Value> = None;
    let mut near_tie = 0usize;
    for t in 0..n {
        for r in 0..k {
            let (got, want) = (gidx[t * k + r], exp_idx[t * k + r]);
            if got == want {
                continue;
            }
            let near = !strict
                && tol > 0.0
                && got >= 0
                && (got as usize) < e
                && want >= 0
                && (want as usize) < e
                && (rr.biased[t * e + got as usize] - rr.biased[t * e + want as usize]).abs()
                    <= tol;
            if near {
                near_tie += 1;
                continue;
            }
            if first_mismatch.is_none() {
                first_mismatch = Some(json!({"token": t, "rank": r, "got": got, "expected": want}));
                fails.push(format!(
                    "routing mismatch at token {t} rank {r}: got expert {got}, expected {want}"
                ));
            }
        }
    }
    let exp_w: Vec<f32> = match &c.expected {
        Some(x) => x.w.clone(),
        None => (0..n)
            .flat_map(|t| {
                let ids: Vec<i32> = gidx[t * k..(t + 1) * k].to_vec();
                route_weights(c.router, &rr.scores[t * e..(t + 1) * e], &ids)
            })
            .collect(),
    };
    let wm = compare(gw, &exp_w);
    let wok = wm.nonfinite == 0 && wm.rel_l2 <= 1e-5 && wm.max_abs <= 1e-6;
    if !wok {
        fails.push(format!(
            "topk_w outside rel_l2<=1e-5 / max_abs<=1e-6: rel_l2={:.3e} max_abs={:.3e} nonfinite={}",
            wm.rel_l2, wm.max_abs, wm.nonfinite
        ));
    }
    let mut j = json!({
        "pass": fails.is_empty(),
        "reference": if strict { "fixture" } else { "cpu-router" },
        "topk_w": metrics_json(&wm, "routing", wok),
        "routing_mismatch": first_mismatch.clone().unwrap_or(Value::Null),
        "near_tie_accepted": near_tie,
    });
    if let (Some(x), Some(sc)) = (&c.expected, Some(&rr.scores)) {
        if let Some(fs) = &x.scores {
            let m = compare(sc, fs);
            j["scores_vs_fixture_informational"] = metrics_info_json(&m);
        }
    }
    (j, fails)
}

fn check_extras(c: &CaseData, gidx: &[i32], gw: &[f32]) -> Vec<String> {
    let k = c.geom.k;
    let mut fails = Vec::new();
    for ex in &c.extras {
        for t in 0..c.n {
            let row = &gidx[t * k..(t + 1) * k];
            let wr = &gw[t * k..(t + 1) * k];
            match ex {
                Extra::Coverage(ids) => {
                    if let Some(m) = ids.iter().find(|&&id| !row.contains(&(id as i32))) {
                        fails.push(format!(
                            "coverage: token {t} did not select expert {m}: {row:?}"
                        ));
                        break;
                    }
                }
                Extra::SameSet(ids) => {
                    let mut a: Vec<i32> = row.to_vec();
                    a.sort_unstable();
                    let mut b: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
                    b.sort_unstable();
                    if a != b {
                        fails.push(format!(
                            "skew: token {t} selected {row:?}, expected the set {b:?}"
                        ));
                        break;
                    }
                }
                Extra::Order(order) => {
                    if row != order.as_slice() {
                        fails.push(format!("tie policy: token {t} selected {row:?}, expected {order:?} (lower id first)"));
                        break;
                    }
                }
                Extra::Unequal => {
                    let sum: f32 = wr.iter().sum();
                    let (mx, mn) = wr
                        .iter()
                        .fold((f32::MIN, f32::MAX), |(a, b), &v| (a.max(v), b.min(v)));
                    if !((sum - 1.0).abs() <= 1e-6) || !(mn > 0.0 && mx / mn > 10.0) {
                        fails.push(format!(
                            "unequal weights: token {t} sum={sum} max/min={} (need sum==1 within 1e-6, ratio > 10): {wr:?}",
                            mx / mn
                        ));
                        break;
                    }
                }
                Extra::PadGuard => {}
            }
        }
    }
    fails
}

/// Full evaluation of one protocol run of a case.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    name: &str,
    c: &CaseData,
    ws: &WeightSet,
    awq: bool,
    awq_rot_ok: bool,
    protocol: GroupedMoeProtocol,
    run: &GpuRun,
    rr: &RouteRef,
    refo: &Result<&RefOut, &String>,
    poison_ref: Option<&RefOut>,
) -> Eval {
    let grouped = protocol == GroupedMoeProtocol::Grouped;
    let mut ev = Eval::new(name);
    ev.put("case", json!(c.name));
    ev.put("protocol", json!(proto_label(protocol)));
    ev.put("awq", json!(awq));
    ev.put("mode", json!(mode_label(c.bf16)));
    ev.put("partial", json!(ws.partial()));
    ev.put("n", json!(c.n));
    ev.put(
        "geometry",
        json!({"E": c.geom.e, "K": c.geom.k, "C": c.geom.c, "I": c.geom.i, "Ipad": c.geom.ipad}),
    );
    ev.put("ms", jf(run.ms));
    if awq && !awq_rot_ok {
        ev.fail(
            "awq rotation kernel disagrees with the CPU divide-then-rotate reference".to_string(),
        );
    }
    let (rj, rfails) = check_routing(c, rr, &run.idx, &run.w);
    ev.put(
        "routing_mismatch",
        rj.get("routing_mismatch").cloned().unwrap_or(Value::Null),
    );
    ev.put("routing", rj);
    for f in rfails {
        ev.fail(f);
    }
    for f in check_extras(c, &run.idx, &run.w) {
        ev.fail(f);
    }
    match refo {
        Err(e) => ev.fail(format!("CPU reference: {e}")),
        Ok(r) => {
            ev.section("routed", &run.routed, &r.routed, c.bf16, grouped);
            if let (Some(gs), Some(rs)) = (&run.shared, &r.shared) {
                ev.section("shared", gs, rs, c.bf16, grouped);
            }
            ev.section("out", &run.out, &r.out, c.bf16, grouped);
            if let Some(p) = poison_ref {
                let m = compare(&run.out, &p.out);
                let sensitive = m.rel_l2 > 0.1;
                ev.put(
                    "pad_guard",
                    json!({
                        "poisoned_reference_rel_l2": jf(m.rel_l2),
                        "sensitive": sensitive,
                        "note": "reference with the down pad activation set to 1.0; the module must stay far from it",
                    }),
                );
                if !sensitive {
                    ev.fail(format!(
                        "pad guard is not sensitive: output is within rel_l2={:.3e} of the poisoned-pad reference",
                        m.rel_l2
                    ));
                }
            }
        }
    }
    if let Some(x) = &c.expected {
        let mo = compare(&run.out, &x.out);
        let mr = compare(&run.routed, &x.routed);
        let mut q = Map::new();
        q.insert("out".into(), metrics_info_json(&mo));
        q.insert("routed".into(), metrics_info_json(&mr));
        if let (Some(gs), Some(xs)) = (&run.shared, &x.shared) {
            q.insert("shared".into(), metrics_info_json(&compare(gs, xs)));
        }
        q.insert(
            "note".into(),
            json!("informational: qt44 4-bit weights vs the original-weight fixture"),
        );
        ev.put("quant_error_vs_original_weights", Value::Object(q));
        if !(mo.rel_l2 <= 0.5) {
            ev.fail(format!(
                "catastrophic quantization error vs the original-weight fixture: out rel_l2={:.3e}",
                mo.rel_l2
            ));
        }
    }
    ev
}

/// Run every requested protocol of `c` against one weight set.
#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    opts: &Opts,
    c: &CaseData,
    ws: &WeightSet,
    awq: bool,
    awq_rot_ok: bool,
    results: &mut Vec<Value>,
) {
    let awq_sfx = if awq { "/awq" } else { "" };
    let name_of = |p: GroupedMoeProtocol| format!("{}/{}{}", c.name, proto_label(p), awq_sfx);
    let setup_fail = |results: &mut Vec<Value>, msg: String| {
        for &p in &opts.protocols {
            let mut ev = Eval::new(&name_of(p));
            ev.put("case", json!(c.name));
            ev.put("protocol", json!(proto_label(p)));
            ev.put("awq", json!(awq));
            ev.fail(msg.clone());
            emit(results, ev.finish());
        }
    };
    let Some(wd) = ws.dev.as_ref() else {
        setup_fail(results, "weight set has no device tensors".to_string());
        return;
    };
    let dev = match DevIn::new(gpu, c) {
        Ok(d) => d,
        Err(e) => {
            setup_fail(results, format!("input upload: {e}"));
            return;
        }
    };
    let scratch = match RaonMoeScratch::new(gpu, &dims_of(c), c.n) {
        Ok(s) => s,
        Err(e) => {
            dev.free(gpu);
            setup_fail(results, format!("scratch: {e}"));
            return;
        }
    };
    let rr = route_ref(c.router, &c.x, &c.router_w, &c.e_bias, c.n, &c.geom);
    let want_pad_guard = c.extras.iter().any(|e| matches!(e, Extra::PadGuard));
    type Cached = (Vec<i32>, Vec<u32>, Result<RefOut, String>, Option<RefOut>);
    let mut cache: Option<Cached> = None;
    let mut first_ok: Option<(GroupedMoeProtocol, GpuRun)> = None;
    let mut cross_done = false;
    for &p in &opts.protocols {
        let name = name_of(p);
        match gpu_moe_run(gpu, ctx, c, wd, &dev, &scratch, p) {
            Err(RunErr::Refusal(text)) => {
                let mut ev = Eval::new(&name);
                ev.put("case", json!(c.name));
                ev.put("protocol", json!(proto_label(p)));
                ev.put("awq", json!(awq));
                ev.put("mode", json!(mode_label(c.bf16)));
                ev.put("refused", json!(true));
                ev.put("refusal", json!(text));
                if opts.allow_refusal {
                    ev.put("allowed_refusal", json!(true));
                } else {
                    ev.fail(format!("protocol refused: {text}"));
                }
                let v = ev.finish();
                emit(results, v);
            }
            Err(RunErr::Fail(text)) => {
                let mut ev = Eval::new(&name);
                ev.put("case", json!(c.name));
                ev.put("protocol", json!(proto_label(p)));
                ev.put("awq", json!(awq));
                ev.fail(format!("run failed: {text}"));
                let v = ev.finish();
                emit(results, v);
            }
            Ok(run) => {
                let key_w: Vec<u32> = run.w.iter().map(|f| f.to_bits()).collect();
                let stale = !matches!(&cache, Some((i, w, _, _)) if *i == run.idx && *w == key_w);
                if stale {
                    let r = cpu_moe_ref(c, ws, &run.idx, &run.w, 0.0);
                    let pr = if want_pad_guard && r.is_ok() {
                        cpu_moe_ref(c, ws, &run.idx, &run.w, 1.0).ok()
                    } else {
                        None
                    };
                    cache = Some((run.idx.clone(), key_w, r, pr));
                }
                let Some((_, _, r, pr)) = cache.as_ref() else {
                    continue;
                };
                let refo = r.as_ref();
                let ev = evaluate(
                    &name,
                    c,
                    ws,
                    awq,
                    awq_rot_ok,
                    p,
                    &run,
                    &rr,
                    &refo,
                    pr.as_ref(),
                );
                let v = ev.finish();
                emit(results, v);
                if first_ok.is_none() {
                    first_ok = Some((p, run));
                } else if !cross_done {
                    if let Some((p0, run0)) = first_ok.as_ref() {
                        let v = cross_check(c, awq, *p0, run0, p, &run);
                        emit(results, v);
                        cross_done = true;
                    }
                }
            }
        }
    }
    scratch.free(gpu);
    dev.free(gpu);
}

/// indexed-vs-grouped agreement of two protocol runs of one case.
fn cross_check(
    c: &CaseData,
    awq: bool,
    pa: GroupedMoeProtocol,
    a: &GpuRun,
    pb: GroupedMoeProtocol,
    b: &GpuRun,
) -> Value {
    let name = format!(
        "{}/{}-vs-{}{}",
        c.name,
        proto_label(pa),
        proto_label(pb),
        if awq { "/awq" } else { "" }
    );
    let mut ev = Eval::new(&name);
    ev.put("case", json!(c.name));
    ev.put("awq", json!(awq));
    ev.put("mode", json!(mode_label(c.bf16)));
    ev.put(
        "label",
        json!(
            "indexed-vs-grouped (core rounding differs: grouped = fp16 WMMA, indexed = fp32 GEMV)"
        ),
    );
    ev.put("routing_identical", json!(a.idx == b.idx));
    if a.idx != b.idx {
        ev.fail("the two protocols selected different experts".to_string());
    }
    ev.section("routed", &a.routed, &b.routed, c.bf16, true);
    if let (Some(x), Some(y)) = (&a.shared, &b.shared) {
        ev.section("shared", x, y, c.bf16, true);
    }
    ev.section("out", &a.out, &b.out, c.bf16, true);
    ev.finish()
}

// ───────────────────────────── gated FFN runs (shared-only, dense) ─────────

fn emit(results: &mut Vec<Value>, v: Value) {
    eprintln!(
        "moe: {}: {} {}",
        v.get("name").and_then(Value::as_str).unwrap_or("?"),
        if pass_of(&v) { "PASS" } else { "FAIL" },
        v.get("detail").and_then(Value::as_str).unwrap_or("")
    );
    results.push(v);
}

/// CPU gated FFN of `n` tokens from the natural-domain inputs.
#[allow(clippy::too_many_arguments)]
fn cpu_gated_ref(
    q: &GatedQ,
    awq_in: Option<&[f32]>,
    awq_down: Option<&[f32]>,
    x: &[f32],
    n: usize,
    c: usize,
    i: usize,
    ipad: usize,
    act: FfnActivation,
    bf16: bool,
) -> Result<Vec<f32>, String> {
    let xr = rotated_inputs(x, n, c, awq_in);
    let r = GatedRef {
        gate: &q.gate,
        up: &q.up,
        down: &q.down,
        awq_down,
    };
    ffn_batch(&r, &xr, n, c, i, ipad, act, bf16, 0.0, worker_count())
}

fn gated_entry(
    name: &str,
    case: &str,
    kind: &str,
    awq: bool,
    awq_rot_ok: bool,
    bf16: bool,
    n: usize,
    section: &str,
    got: Result<(Vec<f32>, f64), RunErr>,
    want: Result<Vec<f32>, String>,
    opts: &Opts,
) -> Value {
    let mut ev = Eval::new(name);
    ev.put("case", json!(case));
    ev.put("protocol", json!(kind));
    ev.put("awq", json!(awq));
    ev.put("mode", json!(mode_label(bf16)));
    ev.put("n", json!(n));
    if awq && !awq_rot_ok {
        ev.fail(
            "awq rotation kernel disagrees with the CPU divide-then-rotate reference".to_string(),
        );
    }
    match got {
        Err(RunErr::Refusal(t)) => {
            ev.put("refused", json!(true));
            ev.put("refusal", json!(t));
            if opts.allow_refusal {
                ev.put("allowed_refusal", json!(true));
            } else {
                ev.fail(format!("refused: {t}"));
            }
        }
        Err(RunErr::Fail(t)) => ev.fail(format!("run failed: {t}")),
        Ok((v, ms)) => {
            ev.put("ms", jf(ms));
            match want {
                Err(e) => ev.fail(format!("CPU reference: {e}")),
                Ok(w) => ev.section(section, &v, &w, bf16, false),
            }
        }
    }
    ev.finish()
}

/// Shared expert alone through `execute_gated_ffn` (mi = I, down K = Ipad).
fn gpu_shared_only(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    c: &CaseData,
    wd: &WeightDev,
    x: &GpuTensor,
) -> Result<(Vec<f32>, f64), RunErr> {
    let g = c.geom;
    let n = c.n;
    let mut ts: Vec<GpuTensor> = Vec::new();
    for words in [
        n * g.c,
        n * g.i,
        n * g.i,
        n * g.ipad,
        n * g.ipad,
        n * g.c,
        rdna_compute::ffn_activation::single_expert_table_words(n),
    ] {
        match alloc_f32(gpu, words) {
            Ok(t) => ts.push(t),
            Err(e) => {
                free_all(gpu, ts);
                return Err(RunErr::Fail(e));
            }
        }
    }
    let r = (|| -> Result<(Vec<f32>, f64), RunErr> {
        let sh = wd
            .shared
            .as_ref()
            .ok_or_else(|| RunErr::Fail("weight set has no shared expert".to_string()))?;
        poison(gpu, &ts[5]).map_err(RunErr::Fail)?;
        let op = GatedFfnOp {
            ctx,
            activation: c.act,
            cast: if c.bf16 {
                CastPolicy::Bf16Boundaries
            } else {
                CastPolicy::Exact32
            },
            rows: n,
            hidden: g.c,
            mi: g.i,
            down_k: g.ipad,
            x,
            input_awq: wd.awq_gu.as_ref(),
            weights: gated_refs(sh, wd.awq_gu.as_ref(), g.c, g.i, g.ipad),
            x_rot: &ts[0],
            scratch: GroupedMoeSharedScratch {
                gate_out: &ts[1],
                up_out: &ts[2],
                act: &ts[3],
                rot: &ts[4],
                out: &ts[5],
                tables: &ts[6],
            },
            out: &ts[5],
        };
        let t0 = Instant::now();
        execute_gated_ffn(gpu, &op).map_err(classify)?;
        gpu.hip.device_synchronize().map_err(fail)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok((down_f32(gpu, &ts[5], n * g.c).map_err(RunErr::Fail)?, ms))
    })();
    free_all(gpu, ts);
    r
}

fn run_shared_only(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    opts: &Opts,
    c: &CaseData,
    ws: &WeightSet,
    awq: bool,
    awq_rot_ok: bool,
    results: &mut Vec<Value>,
) {
    let name = format!("{}/gated-ffn{}", c.name, if awq { "/awq" } else { "" });
    let g = c.geom;
    let got = match (ws.dev.as_ref(), up_f32(gpu, &c.x)) {
        (Some(wd), Ok(x)) => {
            let r = gpu_shared_only(gpu, ctx, c, wd, &x);
            free_all(gpu, [x]);
            r
        }
        (None, _) => Err(RunErr::Fail("weight set has no device tensors".to_string())),
        (_, Err(e)) => Err(RunErr::Fail(e)),
    };
    let want = match &ws.shared {
        Some(q) => cpu_gated_ref(
            q,
            ws.awq_gu.as_deref(),
            ws.awq_sh_down.as_deref(),
            &c.x,
            c.n,
            g.c,
            g.i,
            g.ipad,
            c.act,
            c.bf16,
        ),
        None => Err("weight set has no shared expert".to_string()),
    };
    let v = gated_entry(
        &name,
        &c.name,
        "gated-ffn",
        awq,
        awq_rot_ok,
        c.bf16,
        c.n,
        "shared",
        got,
        want,
        opts,
    );
    emit(results, v);
}

/// S7: dense layer-0 FFN (hidden 2560, mi = down K = 12032) through the
/// production `forward_dense`, SiTU + BF16 boundaries, AWQ off and on.
fn run_dense(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    opts: &Opts,
    awq_rot_ok: bool,
    results: &mut Vec<Value>,
) {
    let dims = RaonMoeDims::raon();
    let (c, mi) = (dims.hidden, dims.dense_mi);
    let act = dims.activation();
    let ns = [1usize, 4, 33];
    let names: Vec<(bool, usize, String)> = [false, true]
        .into_iter()
        .flat_map(|awq| {
            ns.iter().map(move |&n| {
                (
                    awq,
                    n,
                    format!(
                        "S7-dense-N{n}-bf16/gated-ffn{}",
                        if awq { "/awq" } else { "" }
                    ),
                )
            })
        })
        .filter(|(_, _, nm)| opts.case.as_ref().is_none_or(|f| nm.contains(f.as_str())))
        .collect();
    for awq in [false, true] {
        let todo: Vec<&(bool, usize, String)> =
            names.iter().filter(|(a, _, _)| *a == awq).collect();
        if todo.is_empty() {
            continue;
        }
        let seed = 0x5a77u64;
        let awq_in = awq.then(|| awq_vec(seed, 10, c, c));
        let awq_down = awq.then(|| awq_vec(seed, 11, mi, mi));
        let fail_all = |results: &mut Vec<Value>, msg: String| {
            for (_, n, nm) in &todo {
                let mut ev = Eval::new(nm);
                ev.put("case", json!(format!("S7-dense-N{n}-bf16")));
                ev.put("awq", json!(awq));
                ev.fail(msg.clone());
                emit(results, ev.finish());
            }
        };
        let q = match build_gated_q(
            synth_gated(seed, DENSE_TAG, c, mi),
            c,
            mi,
            mi,
            0.0,
            awq_in.as_deref(),
            awq_down.as_deref(),
            worker_count(),
        ) {
            Ok(q) => q,
            Err(e) => {
                fail_all(results, format!("dense weights: {e}"));
                continue;
            }
        };
        let dev = match upload_gated(gpu, &q, awq_in.as_deref(), awq_down.as_deref()) {
            Ok(d) => d,
            Err(e) => {
                fail_all(results, format!("dense upload: {e}"));
                continue;
            }
        };
        let max_n = todo.iter().map(|(_, n, _)| *n).max().unwrap_or(1);
        let scratch = match RaonDenseScratch::new(gpu, &dims, max_n) {
            Ok(s) => s,
            Err(e) => {
                dev.free(gpu);
                fail_all(results, format!("dense scratch: {e}"));
                continue;
            }
        };
        for (_, n, nm) in &todo {
            let n = *n;
            let mut g = Lcg::new(mix_seed(&[seed, n as u64, awq as u64]));
            let x = normal_vec(&mut g, n * c, 1.0, true);
            let got = (|| -> Result<(Vec<f32>, f64), RunErr> {
                let xd = up_f32(gpu, &x).map_err(RunErr::Fail)?;
                let od = match up_f32(gpu, &vec![f32::NAN; n * c]) {
                    Ok(t) => t,
                    Err(e) => {
                        free_all(gpu, [xd]);
                        return Err(RunErr::Fail(e));
                    }
                };
                let r = (|| -> Result<(Vec<f32>, f64), RunErr> {
                    let refs = RaonDenseRefs {
                        weights: gated_refs(&dev, dev.awq_in.as_ref(), c, mi, mi),
                        input_awq: dev.awq_in.as_ref(),
                    };
                    let t0 = Instant::now();
                    forward_dense(gpu, ctx, &dims, &refs, n, &xd, &scratch, &od)
                        .map_err(RunErr::Fail)?;
                    gpu.hip.device_synchronize().map_err(fail)?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    Ok((down_f32(gpu, &od, n * c).map_err(RunErr::Fail)?, ms))
                })();
                free_all(gpu, [xd, od]);
                r
            })();
            let want = cpu_gated_ref(
                &q,
                awq_in.as_deref(),
                awq_down.as_deref(),
                &x,
                n,
                c,
                mi,
                mi,
                act,
                true,
            );
            let v = gated_entry(
                nm,
                &format!("S7-dense-N{n}-bf16"),
                "gated-ffn",
                awq,
                awq_rot_ok,
                true,
                n,
                "out",
                got,
                want,
                opts,
            );
            emit(results, v);
        }
        scratch.free(gpu);
        dev.free(gpu);
    }
}

// ───────────────────────────── synthetic cases ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum SKind {
    Coverage,
    Skew,
    Tie(u8),
    Unequal,
    PadGuard,
    RoutedOnly,
    SharedOnly,
    NonRaon,
}

#[derive(Clone, Copy, Debug)]
struct SynthCase {
    kind: SKind,
    geo: &'static str,
    n: usize,
    bf16: bool,
    k: usize,
    xscale: f32,
}

impl SynthCase {
    fn name(&self) -> String {
        let mode = mode_label(self.bf16);
        let x8 = if self.xscale != 1.0 { "-x8" } else { "" };
        let (g, n) = (self.geo, self.n);
        match self.kind {
            SKind::Coverage => format!("S1-coverage-{g}-N{n}{x8}-{mode}"),
            SKind::Skew => format!("S2-skew-{g}-N{n}-{mode}"),
            SKind::Tie(v) => format!("S3-tie{}-N{n}-{mode}", if v == 0 { "a" } else { "b" }),
            SKind::Unequal => format!("S4-unequal-N{n}-{mode}"),
            SKind::PadGuard => format!("S5-padguard-N{n}-{mode}"),
            SKind::RoutedOnly => format!("S6-routed-only-N{n}-{mode}"),
            SKind::SharedOnly => format!("S6-shared-only-N{n}-{mode}"),
            SKind::NonRaon => format!("S8-nonraon-K{}-N{n}-{mode}", self.k),
        }
    }
}

struct SynthGroup {
    geom: Geom,
    seed: u64,
    pad_fill: f32,
    cases: Vec<SynthCase>,
}

fn synth_groups() -> Vec<SynthGroup> {
    let sc = |kind, geo, n, k, xscale| SynthCase {
        kind,
        geo,
        n,
        bf16: true,
        k,
        xscale,
    };
    let ns_all = [1usize, 2, 3, 4, 31, 64, 65, 129];
    let mut a = Vec::new();
    for &n in &ns_all {
        a.push(sc(SKind::Coverage, "small", n, SMALL.k, 1.0));
    }
    a.push(sc(SKind::Coverage, "small", 64, SMALL.k, 8.0));
    for n in [1usize, 3, 64, 129] {
        a.push(sc(SKind::Skew, "small", n, SMALL.k, 1.0));
    }
    for v in 0..2u8 {
        for n in [3usize, 65] {
            a.push(sc(SKind::Tie(v), "small", n, SMALL.k, 1.0));
        }
    }
    for n in [1usize, 3, 65] {
        a.push(sc(SKind::Unequal, "small", n, SMALL.k, 1.0));
        a.push(sc(SKind::RoutedOnly, "small", n, SMALL.k, 1.0));
        a.push(sc(SKind::SharedOnly, "small", n, SMALL.k, 1.0));
    }
    for k in [1usize, 3, 8] {
        for n in [1usize, 3, 64] {
            a.push(SynthCase {
                kind: SKind::NonRaon,
                geo: "small",
                n,
                bf16: false,
                k,
                xscale: 1.0,
            });
        }
    }
    let b: Vec<SynthCase> = [1usize, 3, 65]
        .iter()
        .map(|&n| sc(SKind::PadGuard, "small", n, SMALL.k, 1.0))
        .collect();
    let mut c = Vec::new();
    for &n in &ns_all {
        c.push(sc(SKind::Coverage, "real", n, REAL.k, 1.0));
    }
    for n in [1usize, 3, 64, 129] {
        c.push(sc(SKind::Skew, "real", n, REAL.k, 1.0));
    }
    vec![
        SynthGroup {
            geom: SMALL,
            seed: 0x5a11,
            pad_fill: 0.0,
            cases: a,
        },
        SynthGroup {
            geom: SMALL,
            seed: 0x5a12,
            pad_fill: 1e3,
            cases: b,
        },
        SynthGroup {
            geom: REAL,
            seed: 0x5a13,
            pad_fill: 0.0,
            cases: c,
        },
    ]
}

fn build_synth(sc: &SynthCase, grp: &SynthGroup) -> CaseData {
    let mut g = grp.geom;
    if sc.kind == SKind::NonRaon {
        g.k = sc.k;
    }
    let name = sc.name();
    let mut rng = Lcg::new(str_seed(&name));
    let (e, c, n) = (g.e, g.c, sc.n);
    let nonraon = sc.kind == SKind::NonRaon;
    let act = if nonraon {
        FfnActivation::SwiGlu
    } else {
        FfnActivation::Situ {
            gate_beta: 4.0,
            value_beta: 25.0,
        }
    };
    let router = if nonraon {
        RouterSel::Softmax
    } else {
        RouterSel::SigmoidBias
    };
    let rstd = 1.5 / (c as f32).sqrt();
    let mut router_w = normal_vec(&mut rng, e * c, rstd, true);
    let mut bias: Vec<f32> = if nonraon {
        vec![0.0; e]
    } else {
        (0..e)
            .map(|_| bf16r((rng.uniform() as f32 - 0.5) * 0.04))
            .collect()
    };
    let u = normal_vec(&mut rng, c, 1.0, true);
    let unorm2: f32 = u.iter().map(|v| v * v).sum();
    let tokens = |rng: &mut Lcg, a: f32, b: f32| -> Vec<f32> {
        let mut x = Vec::with_capacity(n * c);
        for _ in 0..n {
            for &uj in &u {
                x.push(bf16r(a * uj + b * rng.normal()));
            }
        }
        x
    };
    let set_row = |rw: &mut Vec<f32>, row: usize, scale: f32| {
        for j in 0..c {
            rw[row * c + j] = bf16r(scale * u[j]);
        }
    };
    let mut extras = Vec::new();
    let mut tie_tol = 2e-6f32;
    let x: Vec<f32>;
    match sc.kind {
        SKind::Coverage => {
            x = tokens(&mut rng, 0.6, 0.8);
            let s = 6.0 / (unorm2 * 0.6);
            set_row(&mut router_w, 0, s);
            set_row(&mut router_w, e - 1, s);
            bias[0] = 3.0;
            bias[e - 1] = 2.5;
            extras.push(Extra::Coverage(vec![0, e - 1]));
            // the x*8 batch scales the whole input
        }
        SKind::Skew => {
            x = normal_vec(&mut rng, n * c, 1.0, true);
            let ids: Vec<usize> = if e == SMALL.e {
                vec![2, 5, 11]
            } else {
                vec![3, 17, 42, 64, 99, 126]
            };
            for (r, &id) in ids.iter().enumerate() {
                bias[id] = 10.0 + 2.0 * (ids.len() - 1 - r) as f32;
            }
            extras.push(Extra::SameSet(ids));
        }
        SKind::Tie(v) => {
            x = tokens(&mut rng, 1.0, 0.05);
            let s = 30.0 / unorm2;
            set_row(&mut router_w, 3, s);
            set_row(&mut router_w, 9, s);
            bias[0] = 10.0;
            bias[3] = 5.0;
            bias[9] = 5.0;
            let order = if v == 0 {
                vec![0, 3, 9]
            } else {
                bias[1] = 9.0;
                vec![0, 1, 3]
            };
            extras.push(Extra::Order(order));
            tie_tol = 0.0;
        }
        SKind::Unequal => {
            x = tokens(&mut rng, 1.0, 0.05);
            let ids = [1usize, 7, 12];
            let logits = [-5.0f32, 0.5, 3.0];
            for (r, &id) in ids.iter().enumerate() {
                set_row(&mut router_w, id, logits[r] / unorm2);
                bias[id] = 20.0 - 4.0 * r as f32;
            }
            extras.push(Extra::Order(ids.iter().map(|&i| i as i32).collect()));
            extras.push(Extra::Unequal);
        }
        SKind::PadGuard => {
            x = normal_vec(&mut rng, n * c, 1.0, true);
            extras.push(Extra::PadGuard);
        }
        SKind::RoutedOnly | SKind::SharedOnly | SKind::NonRaon => {
            x = normal_vec(&mut rng, n * c, 1.0, true);
        }
    }
    let x = if sc.xscale != 1.0 {
        x.iter().map(|v| bf16r(v * sc.xscale)).collect()
    } else {
        x
    };
    CaseData {
        name,
        geom: g,
        n,
        bf16: sc.bf16,
        act,
        router,
        shared: !matches!(sc.kind, SKind::RoutedOnly | SKind::NonRaon),
        x,
        router_w,
        e_bias: bias,
        expected: None,
        tie_tol,
        extras,
    }
}

// ───────────────────────────── options and fixtures ────────────────────────

struct Opts {
    case: Option<String>,
    protocols: Vec<GroupedMoeProtocol>,
    expert_limit: Option<usize>,
    awq_real: bool,
    allow_refusal: bool,
    synthetic: bool,
}

fn flag(o: &BTreeMap<String, String>, key: &str, default: bool) -> Result<bool, String> {
    match o.get(key).map(|s| s.trim()) {
        None => Ok(default),
        Some("1") | Some("true") => Ok(true),
        Some("0") | Some("false") => Ok(false),
        Some(other) => Err(format!("--{key} `{other}`: expected 0 or 1")),
    }
}

impl Opts {
    fn parse(o: &BTreeMap<String, String>) -> Result<Self, String> {
        let protos = o
            .get("protocols")
            .map(String::as_str)
            .unwrap_or("indexed,grouped");
        let mut protocols = Vec::new();
        for p in protos.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let p = match p {
                "indexed" => GroupedMoeProtocol::Indexed,
                "grouped" => GroupedMoeProtocol::Grouped,
                other => {
                    return Err(format!(
                        "--protocols `{other}`: expected indexed and/or grouped"
                    ))
                }
            };
            if !protocols.contains(&p) {
                protocols.push(p);
            }
        }
        if protocols.is_empty() {
            return Err("--protocols selects no protocol".to_string());
        }
        let expert_limit = match o.get("expert_limit") {
            Some(s) => Some(
                s.trim()
                    .parse::<usize>()
                    .map_err(|e| format!("--expert_limit `{s}`: {e}"))?,
            ),
            None => None,
        };
        Ok(Self {
            case: o.get("case").cloned(),
            protocols,
            expert_limit,
            awq_real: flag(o, "awq_real", false)?,
            allow_refusal: flag(o, "allow_refusal", false)?,
            synthetic: flag(o, "synthetic", true)?,
        })
    }

    fn wants(&self, name: &str) -> bool {
        self.case.as_ref().is_none_or(|f| name.contains(f.as_str()))
    }
}

fn get_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("manifest dims.{key} is missing or not an unsigned integer"))
}

fn ids_from_f32(v: &[f32], e: usize, what: &str) -> Result<Vec<i32>, String> {
    v.iter()
        .enumerate()
        .map(|(j, &f)| {
            if f.is_finite() && f == f.round() && f >= 0.0 && (f as usize) < e {
                Ok(f as i32)
            } else {
                Err(format!("`{what}`[{j}] = {f} is not an expert id below {e}"))
            }
        })
        .collect()
}

/// Parse one manifest entry and read its case file.
fn load_fixture_case(dir: &Path, v: &Value) -> Result<(CaseData, WSrc), String> {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .ok_or("manifest case has no `name`")?
        .to_string();
    let file = v
        .get("file")
        .and_then(Value::as_str)
        .ok_or("manifest case has no `file`")?;
    let dims = v.get("dims").ok_or("manifest case has no `dims`")?;
    let geom = Geom {
        e: get_usize(dims, "E")?,
        k: get_usize(dims, "K")?,
        c: get_usize(dims, "C")?,
        i: get_usize(dims, "I")?,
        ipad: get_usize(dims, "Ipad")?,
    };
    let n = get_usize(dims, "N")?;
    let params = v.get("params").cloned().unwrap_or(Value::Null);
    let pstr = |k: &str| params.get(k).and_then(Value::as_str);
    let mode = pstr("mode")
        .or_else(|| v.get("mode").and_then(Value::as_str))
        .unwrap_or("bf16");
    let bf16 = match mode {
        "bf16" => true,
        "f32" => false,
        other => return Err(format!("unknown mode `{other}`")),
    };
    let beta = |k: &str, d: f32| {
        params
            .get(k)
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .unwrap_or(d)
    };
    let act = match pstr("activation").unwrap_or("situ") {
        "situ" => FfnActivation::Situ {
            gate_beta: beta("gate_beta", 4.0),
            value_beta: beta("value_beta", 25.0),
        },
        "swiglu" => FfnActivation::SwiGlu,
        other => return Err(format!("unknown activation `{other}`")),
    };
    let router = match pstr("router").unwrap_or("sigmoid_bias") {
        "sigmoid_bias" => RouterSel::SigmoidBias,
        "softmax" => RouterSel::Softmax,
        other => return Err(format!("unknown router `{other}`")),
    };
    let shared = params
        .get("shared")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if n == 0
        || n > (1 << 20)
        || geom.k == 0
        || geom.k > geom.e
        || geom.e > 1024
        || geom.c % FWHT_N != 0
        || geom.c > (1 << 16)
        || geom.ipad % FWHT_N != 0
        || geom.ipad > (1 << 16)
        || geom.ipad < geom.i
        || geom.i == 0
    {
        return Err(format!("unsupported geometry N={n} {geom:?}"));
    }
    let weights = v
        .get("weights")
        .and_then(Value::as_str)
        .ok_or("manifest case has no `weights`")?;
    let st = StFile::open(&dir.join(file))?;
    let x = st.read_shaped("x", &[n, geom.c])?;
    let router_w = st.read_shaped("router_w", &[geom.e, geom.c])?;
    let e_bias = if router == RouterSel::SigmoidBias {
        st.read_shaped("e_bias", &[geom.e])?
    } else {
        vec![0.0; geom.e]
    };
    let scores = if router == RouterSel::SigmoidBias && st.has("scores") {
        Some(st.read_shaped("scores", &[n, geom.e])?)
    } else if router == RouterSel::SigmoidBias {
        return Err("fixture lacks tensor `scores`".to_string());
    } else {
        None
    };
    let idx = ids_from_f32(
        &st.read_shaped("topk_idx", &[n, geom.k])?,
        geom.e,
        "topk_idx",
    )?;
    let w = st.read_shaped("topk_w", &[n, geom.k])?;
    let routed = st.read_shaped("routed", &[n, geom.c])?;
    let out = st.read_shaped("out", &[n, geom.c])?;
    let shared_t = if shared {
        Some(st.read_shaped("shared", &[n, geom.c])?)
    } else {
        None
    };
    let case = CaseData {
        name,
        geom,
        n,
        bf16,
        act,
        router,
        shared,
        x,
        router_w,
        e_bias,
        expected: Some(Expected {
            scores,
            idx,
            w,
            routed,
            shared: shared_t,
            out,
        }),
        tie_tol: 0.0,
        extras: Vec::new(),
    };
    Ok((case, WSrc::File(dir.join(weights))))
}

// ───────────────────────────── self checks and entry ───────────────────────

struct RotCheck {
    plain: Value,
    awq_ok: bool,
    awq: Value,
}

/// The CPU rotation must equal the production GPU rotation (plain and AWQ);
/// a plain-rotation mismatch voids every case, so it fails the whole run.
fn rotation_check(gpu: &mut Gpu) -> Result<RotCheck, String> {
    let (k, rows) = (512usize, 3usize);
    let mut g = Lcg::new(0x707a);
    let x = normal_vec(&mut g, k * rows, 1.0, false);
    let s = awq_vec(7, 0, k, k);
    let xd = up_f32(gpu, &x)?;
    let sd = up_f32(gpu, &s)?;
    let a = alloc_f32(gpu, k * rows)?;
    let b = alloc_f32(gpu, k * rows)?;
    let r = (|| -> Result<(Metrics, Metrics), String> {
        gpu.rotate_x_mq_batched(&xd, &a, k, rows)
            .map_err(|e| format!("rotate_x_mq_batched: {e}"))?;
        let got = down_f32(gpu, &a, k * rows)?;
        let mut want = x.clone();
        for row in want.chunks_mut(k) {
            rotate_vec(row);
        }
        let plain = compare(&got, &want);
        gpu.rotate_x_mq_awq_batched(&xd, &sd, &b, k, rows)
            .map_err(|e| format!("rotate_x_mq_awq_batched: {e}"))?;
        let got = down_f32(gpu, &b, k * rows)?;
        let want = rotated_inputs(&x, rows, k, Some(&s));
        Ok((plain, compare(&got, &want)))
    })();
    free_all(gpu, [xd, sd, a, b]);
    let (plain, awq) = r?;
    let ok = |m: &Metrics| m.nonfinite == 0 && m.rel_l2 <= 1e-5;
    if !ok(&plain) {
        return Err(format!(
            "rotation self-check FAILED: the CPU FWHT-256 rotation differs from the GPU rotate_x_mq_batched (rel_l2={:.3e}, max_abs={:.3e}); every case would be void",
            plain.rel_l2, plain.max_abs
        ));
    }
    Ok(RotCheck {
        plain: metrics_info_json(&plain),
        awq_ok: ok(&awq),
        awq: metrics_info_json(&awq),
    })
}

fn fixture_fail_entries(opts: &Opts, name: &str, awq: bool, msg: &str, results: &mut Vec<Value>) {
    for &p in &opts.protocols {
        let mut ev = Eval::new(&format!(
            "{name}/{}{}",
            proto_label(p),
            if awq { "/awq" } else { "" }
        ));
        ev.put("case", json!(name));
        ev.put("protocol", json!(proto_label(p)));
        ev.put("awq", json!(awq));
        ev.fail(msg.to_string());
        emit(results, ev.finish());
    }
}

pub fn run(fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    if opts.contains_key("negative") {
        return Err("moe has no negative controls".to_string());
    }
    let o = Opts::parse(opts)?;
    let dir = fixtures.join("moe");
    let mpath = dir.join("manifest.json");
    let text = std::fs::read_to_string(&mpath)
        .map_err(|e| format!("moe manifest {}: {e}", mpath.display()))?;
    let manifest: Value = serde_json::from_str(&text)
        .map_err(|e| format!("moe manifest {}: {e}", mpath.display()))?;
    let cases = manifest
        .get("cases")
        .and_then(Value::as_array)
        .filter(|c| !c.is_empty())
        .ok_or_else(|| format!("moe manifest {} declares no cases", mpath.display()))?;

    let started = Instant::now();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
    let ctx = DispatchCtx::new(&gpu);
    let rot = rotation_check(&mut gpu)?;
    let mut results: Vec<Value> = Vec::new();

    let mut fixture_selected = 0usize;
    for (ci, cv) in cases.iter().enumerate() {
        let nm = cv.get("name").and_then(Value::as_str).map(str::to_string);
        let label = nm
            .clone()
            .unwrap_or_else(|| format!("<manifest case #{ci}>"));
        if !o.wants(&label) {
            continue;
        }
        fixture_selected += 1;
        let (c, wsrc) = match load_fixture_case(&dir, cv) {
            Ok(x) => x,
            Err(e) => {
                let mut ev = Eval::new(&label);
                ev.put("case", json!(label));
                ev.fail(format!("fixture: {e}"));
                emit(&mut results, ev.finish());
                continue;
            }
        };
        let awqs: &[bool] = if c.geom.c <= 512 || o.awq_real {
            &[false, true]
        } else {
            &[false]
        };
        for &awq in awqs {
            let spec = WSpec {
                src: wsrc.clone(),
                geom: c.geom,
                pad_fill: 0.0,
                shared: c.shared,
                awq,
                expert_limit: o.expert_limit,
            };
            match build_wset(&mut gpu, &spec) {
                Ok(ws) => {
                    run_case(&mut gpu, &ctx, &o, &c, &ws, awq, rot.awq_ok, &mut results);
                    ws.free(&mut gpu);
                }
                Err(e) => {
                    fixture_fail_entries(&o, &c.name, awq, &format!("weights: {e}"), &mut results)
                }
            }
        }
    }

    let mut synthetic_selected = 0usize;
    if o.synthetic {
        for grp in synth_groups() {
            let todo: Vec<&SynthCase> = grp.cases.iter().filter(|s| o.wants(&s.name())).collect();
            if todo.is_empty() {
                continue;
            }
            synthetic_selected += todo.len();
            let awqs: &[bool] = if grp.geom.c <= 512 || o.awq_real {
                &[false, true]
            } else {
                &[false]
            };
            for &awq in awqs {
                let spec = WSpec {
                    src: WSrc::Synth { seed: grp.seed },
                    geom: grp.geom,
                    pad_fill: grp.pad_fill,
                    shared: true,
                    awq,
                    expert_limit: o.expert_limit,
                };
                let ws = match build_wset(&mut gpu, &spec) {
                    Ok(ws) => ws,
                    Err(e) => {
                        for sc in &todo {
                            fixture_fail_entries(
                                &o,
                                &sc.name(),
                                awq,
                                &format!("synthetic weights: {e}"),
                                &mut results,
                            );
                        }
                        continue;
                    }
                };
                for sc in &todo {
                    let c = build_synth(sc, &grp);
                    if sc.kind == SKind::SharedOnly {
                        run_shared_only(&mut gpu, &ctx, &o, &c, &ws, awq, rot.awq_ok, &mut results);
                    } else {
                        run_case(&mut gpu, &ctx, &o, &c, &ws, awq, rot.awq_ok, &mut results);
                    }
                }
                ws.free(&mut gpu);
            }
        }
        let before = results.len();
        run_dense(&mut gpu, &ctx, &o, rot.awq_ok, &mut results);
        synthetic_selected += results.len() - before;
    }

    if results.is_empty() {
        return Err(format!(
            "no moe case matches the filter {:?} ({} manifest cases at {})",
            o.case,
            cases.len(),
            mpath.display()
        ));
    }
    let passed = results.iter().filter(|r| pass_of(r)).count();
    let refused = results
        .iter()
        .filter(|r| r.get("refused").and_then(Value::as_bool) == Some(true))
        .count();
    let partial = results
        .iter()
        .filter(|r| r.get("partial").and_then(Value::as_bool) == Some(true))
        .count();
    let all_pass = passed == results.len();
    Ok(json!({
        "module": "moe",
        "declared_cases": results.len(),
        "pass": all_pass,
        "cases": results,
        "summary": {
            "total": passed + (results.len() - passed),
            "passed": passed,
            "failed": results.len() - passed,
            "refused": refused,
            "partial": partial,
            "manifest_cases": cases.len(),
            "fixture_cases_selected": fixture_selected,
            "synthetic_selected": synthetic_selected,
            "protocols": o.protocols.iter().map(|p| proto_label(*p)).collect::<Vec<_>>(),
            "rotation_check": rot.plain,
            "rotation_check_awq": {"pass": rot.awq_ok, "metrics": rot.awq},
            "elapsed_s": started.elapsed().as_secs_f64(),
        },
    }))
}

// ───────────────────────────── CPU unit tests ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir() -> PathBuf {
        let base = std::env::var_os("CARGO_TARGET_TMPDIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join("raon_moe_oracle_tmp")))
            })
            .unwrap_or_else(|| PathBuf::from("raon_moe_oracle_tmp"));
        std::fs::create_dir_all(&base).expect("create test scratch dir");
        base
    }

    #[test]
    fn f16_known_values_and_rounding() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        // 65520 is the halfway point between 65504 and the next (unrepresentable) value.
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(1e9), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(2f32.powi(-24)), 1);
        assert_eq!(f32_to_f16_bits(2f32.powi(-25)), 0); // tie -> even (0)
        assert_eq!(f32_to_f16_bits(1.5 * 2f32.powi(-25)), 1);
        assert_eq!(f32_to_f16_bits(2f32.powi(-26)), 0);
        // 1 + 2^-11 is the tie between 1.0 and 1 + 2^-10: even mantissa wins.
        assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_roundtrips_every_finite_half() {
        for h in 0..=u16::MAX {
            if (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 {
                continue;
            }
            assert_eq!(f32_to_f16_bits(f16_bits_to_f32(h)), h, "half {h:#06x}");
        }
    }

    #[test]
    fn bf16_round_to_nearest_even_edges() {
        assert_eq!(bf16r(1.0), 1.0);
        // exact ties (bit 15 set, lower bits zero) round to the even neighbour
        assert_eq!(bf16r(1.0 + 2f32.powi(-8)), 1.0);
        assert_eq!(bf16r(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2.0 * 2f32.powi(-7));
        // just above a tie rounds up
        assert_eq!(
            bf16r(f32::from_bits(0x3f80_8001)),
            f32::from_bits(0x3f81_0000)
        );
        // just below a tie rounds down
        assert_eq!(bf16r(f32::from_bits(0x3f80_7fff)), 1.0);
        assert_eq!(bf16r(f32::MAX), f32::INFINITY);
        assert_eq!(bf16r(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(bf16r(f32::NAN).is_nan());
        assert_eq!(bf16r(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(bf16r(f32::from_bits(1)), 0.0); // smallest subnormal rounds to zero
    }

    fn random_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut g = Lcg::new(seed);
        normal_vec(&mut g, n, 1.0, false)
    }

    #[test]
    fn rotation_matches_naive_hadamard_and_inverts() {
        let (s1, s2) = fwht_signs();
        let x = random_vec(3, FWHT_N);
        let mut got = [0f32; FWHT_N];
        got.copy_from_slice(&x);
        rotate256(&mut got);
        for i in 0..FWHT_N {
            let mut acc = 0f64;
            for j in 0..FWHT_N {
                let sign = if (i & j).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                acc += sign * (x[j] * s1[j]) as f64;
            }
            let want = (acc * 0.0625) as f32 * s2[i];
            assert!(
                (got[i] - want).abs() <= 1e-4,
                "row {i}: {} vs {want}",
                got[i]
            );
        }
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        assert!((norm(&got) - norm(&x)).abs() < 1e-4 * norm(&x));
        unrotate256(&mut got);
        for i in 0..FWHT_N {
            assert!(
                (got[i] - x[i]).abs() < 1e-5,
                "restore {i}: {} vs {}",
                got[i],
                x[i]
            );
        }
    }

    #[test]
    fn quantize_dequantize_error_is_bounded() {
        let k = 512;
        let m = 5;
        let w = random_vec(11, m * k);
        let q = quantize_matrix(&w, m, k, 3).expect("quantize");
        assert_eq!(q.len(), m * row_bytes(k));
        let mut dq = vec![0f32; k];
        for r in 0..m {
            let mut want = w[r * k..(r + 1) * k].to_vec();
            rotate_vec(&mut want);
            dequant_row(&q[r * row_bytes(k)..(r + 1) * row_bytes(k)], k, &mut dq);
            for h in 0..k / 128 {
                let span = &want[h * 128..(h + 1) * 128];
                let d = span.iter().fold(0f32, |a, &v| a.max(v.abs())) / 8.0;
                for j in 0..128 {
                    let err = (dq[h * 128 + j] - span[j]).abs();
                    // q is a round-to-nearest of w/d; the extreme positive value clamps (<= d off);
                    // f16 rounding of scale/zero adds <= 2^-10 relative of 8d.
                    assert!(
                        err <= d * 1.02 + 1e-6,
                        "row {r} half {h} col {j}: err {err} d {d}"
                    );
                }
            }
        }
        // threaded and serial quantization agree bit-for-bit
        assert_eq!(q, quantize_matrix(&w, m, k, 1).expect("serial"));
    }

    #[test]
    fn quantize_all_zero_half_is_all_zero_bytes_and_decodes_to_zero() {
        let mut w = vec![0f32; 256];
        w[200] = 1.0; // only the second half is non-zero in the natural domain, but rotation mixes both
        let mut z = vec![0f32; 256];
        let mut out = vec![0u8; QT44_GROUP_BYTES];
        quantize_row(&z, &mut out).expect("zero row");
        assert!(out.iter().all(|&b| b == 0));
        let mut dq = vec![1f32; 256];
        dequant_row(&out, 256, &mut dq);
        assert!(dq.iter().all(|&v| v == 0.0));
        z[0] = f32::NAN;
        assert!(quantize_row(&z, &mut out).is_err());
        z[0] = 1e9;
        assert!(
            quantize_row(&z, &mut out).is_err(),
            "f16 overflow must be reported"
        );
        quantize_row(&w, &mut out).expect("nonzero row");
        assert!(out.iter().any(|&b| b != 0));
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = Map::new();
        let mut off = 0usize;
        for (name, dtype, shape, data) in tensors {
            header.insert(
                name.to_string(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + data.len()]}),
            );
            off += data.len();
        }
        let hb = serde_json::to_vec(&Value::Object(header)).expect("header");
        let mut f = Vec::new();
        f.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        f.extend_from_slice(&hb);
        for (_, _, _, d) in tensors {
            f.extend_from_slice(d);
        }
        std::fs::write(path, f).expect("write safetensors");
    }

    #[test]
    fn safetensors_reader_reads_dtypes_and_resolves_suffixes() {
        let dir = scratch_dir();
        let p = dir.join("reader_ok.safetensors");
        let f32s: Vec<u8> = [1.5f32, -2.0, 3.25, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf: Vec<u8> = [0x3fc0u16, 0xc000, 0x4040]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let i32s: Vec<u8> = [3i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
        write_safetensors(
            &p,
            &[
                (
                    "model.layers.5.moe.experts.2.gate_proj.weight",
                    "F32",
                    vec![2, 2],
                    f32s,
                ),
                (
                    "model.layers.5.moe.shared_experts.up_proj.weight",
                    "BF16",
                    vec![3],
                    bf,
                ),
                ("idx", "I32", vec![2], i32s),
            ],
        );
        let st = StFile::open(&p).expect("open");
        let name = st.lookup("moe.experts.2.gate_proj.weight").expect("suffix");
        assert_eq!(
            st.read_shaped(&name, &[2, 2]).expect("f32"),
            vec![1.5, -2.0, 3.25, 0.0]
        );
        assert!(
            st.read_shaped(&name, &[4]).is_err(),
            "shape mismatch must be an error"
        );
        let n2 = st
            .lookup("moe.shared_experts.up_proj.weight")
            .expect("suffix 2");
        assert_eq!(st.read_f32(&n2).expect("bf16").1, vec![1.5, -2.0, 3.0]);
        assert_eq!(st.read_f32("idx").expect("i32").1, vec![3.0, -1.0]);
        assert!(st.lookup("moe.experts.3.gate_proj.weight").is_err());
        assert!(st.read_f32("nope").is_err());
    }

    #[test]
    fn safetensors_reader_rejects_malformed_files() {
        let dir = scratch_dir();
        let p = dir.join("reader_bad.safetensors");
        std::fs::write(&p, [1u8, 2, 3]).expect("write");
        assert!(StFile::open(&p).is_err(), "too short");
        std::fs::write(&p, u64::MAX.to_le_bytes()).expect("write");
        assert!(StFile::open(&p).is_err(), "huge header length");
        // data_offsets beyond the file
        let hb = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        f.extend_from_slice(hb);
        f.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &f).expect("write");
        assert!(StFile::open(&p).is_err(), "offsets past the data area");
        // byte count disagrees with shape x dtype
        let hb = br#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        f.extend_from_slice(hb);
        f.extend_from_slice(&[0u8; 16]);
        std::fs::write(&p, &f).expect("write");
        assert!(StFile::open(&p).is_err(), "size/shape disagreement");
        let hb = br#"{"a":{"dtype":"F8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        f.extend_from_slice(hb);
        f.push(0);
        std::fs::write(&p, &f).expect("write");
        assert!(StFile::open(&p).is_err(), "unsupported dtype");
    }

    #[test]
    fn router_reference_breaks_ties_toward_the_lower_id() {
        let g = Geom {
            e: 6,
            k: 3,
            c: 256,
            i: 64,
            ipad: 256,
        };
        // Experts 1 and 4 share an identical row and an identical bias.
        let mut rw = random_vec(5, g.e * g.c);
        let row1: Vec<f32> = rw[g.c..2 * g.c].to_vec();
        rw[4 * g.c..5 * g.c].copy_from_slice(&row1);
        let bias = [0.0f32, 5.0, 0.0, 0.0, 5.0, 0.0];
        let x = random_vec(6, g.c);
        let r = route_ref(RouterSel::SigmoidBias, &x, &rw, &bias, 1, &g);
        assert_eq!(&r.idx[..2], &[1, 4], "tied pair: lower id first");
        // only one slot left for the tied pair: the lower id wins
        let g2 = Geom { k: 1, ..g };
        let r = route_ref(RouterSel::SigmoidBias, &x, &rw, &bias, 1, &g2);
        assert_eq!(r.idx, vec![1]);
        let sum: f32 = ref_weights(&r, RouterSel::SigmoidBias, &g2).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn router_reference_weights_are_renormalized() {
        let g = Geom {
            e: 8,
            k: 3,
            c: 256,
            i: 64,
            ipad: 256,
        };
        let rw = random_vec(8, g.e * g.c);
        let x = random_vec(9, 2 * g.c);
        let bias = vec![0.0f32; g.e];
        for kind in [RouterSel::SigmoidBias, RouterSel::Softmax] {
            let r = route_ref(kind, &x, &rw, &bias, 2, &g);
            for t in 0..2 {
                let wv = ref_weights(&r, kind, &g);
                let w = &wv[t * g.k..(t + 1) * g.k];
                assert!(
                    (w.iter().sum::<f32>() - 1.0).abs() < 1e-5,
                    "{kind:?} token {t}: {w:?}"
                );
                let ids = &r.idx[t * g.k..(t + 1) * g.k];
                for p in ids.windows(2) {
                    let s = &r.biased[t * g.e..(t + 1) * g.e];
                    assert!(s[p[0] as usize] >= s[p[1] as usize], "descending order");
                }
            }
        }
        // sigmoid scores are logistic of the logits
        let r = route_ref(RouterSel::SigmoidBias, &x, &rw, &bias, 2, &g);
        let l0 = dot_f64(&rw[..g.c], &x[..g.c]);
        assert!((r.scores[0] as f64 - 1.0 / (1.0 + (-l0).exp())).abs() < 1e-6);
    }

    fn small_case(n: usize, bf16: bool, shared: bool) -> (CaseData, WeightSet) {
        let g = Geom {
            e: 4,
            k: 2,
            c: 256,
            i: 64,
            ipad: 256,
        };
        let mut rng = Lcg::new(99);
        let x = normal_vec(&mut rng, n * g.c, 1.0, true);
        let experts: Vec<GatedQ> = (0..g.e)
            .map(|e| {
                build_gated_q(
                    synth_gated(7, e as u64, g.c, g.i),
                    g.c,
                    g.i,
                    g.ipad,
                    0.0,
                    None,
                    None,
                    1,
                )
                .expect("expert")
            })
            .collect();
        let sh = build_gated_q(
            synth_gated(7, SHARED_TAG, g.c, g.i),
            g.c,
            g.i,
            g.ipad,
            0.0,
            None,
            None,
            1,
        )
        .expect("shared");
        let ws = WeightSet {
            geom: g,
            experts,
            shared: shared.then_some(sh),
            awq_gu: None,
            awq_down: Vec::new(),
            awq_sh_down: None,
            dev: None,
        };
        let c = CaseData {
            name: "t".into(),
            geom: g,
            n,
            bf16,
            act: FfnActivation::Situ {
                gate_beta: 4.0,
                value_beta: 25.0,
            },
            router: RouterSel::SigmoidBias,
            shared,
            x,
            router_w: Vec::new(),
            e_bias: Vec::new(),
            expected: None,
            tie_tol: 0.0,
            extras: Vec::new(),
        };
        (c, ws)
    }

    #[test]
    fn module_reference_matches_a_token_by_token_recomputation() {
        let n = 5;
        for (bf16, shared) in [(false, true), (true, true), (true, false)] {
            let (c, ws) = small_case(n, bf16, shared);
            let g = c.geom;
            // ranks deliberately not in ascending id order
            let idx: Vec<i32> = vec![3, 0, 1, 2, 2, 1, 0, 3, 1, 3];
            let w: Vec<f32> = vec![0.7, 0.3, 0.25, 0.75, 0.5, 0.5, 0.9, 0.1, 0.4, 0.6];
            let r = cpu_moe_ref(&c, &ws, &idx, &w, 0.0).expect("reference");
            for t in 0..n {
                let xt = &c.x[t * g.c..(t + 1) * g.c];
                let mut ids = [(idx[t * 2], w[t * 2]), (idx[t * 2 + 1], w[t * 2 + 1])];
                ids.sort_by_key(|p| p.0);
                let mut acc = vec![0f32; g.c];
                for (e, wt) in ids {
                    let q = &ws.experts[e as usize];
                    let y = cpu_gated_ref(q, None, None, xt, 1, g.c, g.i, g.ipad, c.act, bf16)
                        .expect("expert");
                    for j in 0..g.c {
                        acc[j] += y[j] * wt;
                    }
                }
                if bf16 {
                    acc.iter_mut().for_each(|v| *v = bf16r(*v));
                }
                let sh = ws.shared.as_ref().map(|q| {
                    cpu_gated_ref(q, None, None, xt, 1, g.c, g.i, g.ipad, c.act, bf16)
                        .expect("shared")
                });
                for j in 0..g.c {
                    assert_eq!(
                        r.routed[t * g.c + j],
                        acc[j],
                        "routed t{t} j{j} bf16={bf16}"
                    );
                    let mut v = acc[j];
                    if let Some(s) = &sh {
                        v += s[j];
                    }
                    let v = if bf16 { bf16r(v) } else { v };
                    assert_eq!(
                        r.out[t * g.c + j],
                        v,
                        "out t{t} j{j} bf16={bf16} shared={shared}"
                    );
                }
            }
            assert_eq!(r.shared.is_some(), shared);
        }
    }

    #[test]
    fn module_reference_rejects_bad_routing() {
        let (c, ws) = small_case(1, false, false);
        assert!(
            cpu_moe_ref(&c, &ws, &[0, 0], &[0.5, 0.5], 0.0).is_err(),
            "duplicate expert"
        );
        assert!(
            cpu_moe_ref(&c, &ws, &[0, 9], &[0.5, 0.5], 0.0).is_err(),
            "expert out of range"
        );
        assert!(
            cpu_moe_ref(&c, &ws, &[0, -1], &[0.5, 0.5], 0.0).is_err(),
            "negative expert"
        );
        assert!(
            cpu_moe_ref(&c, &ws, &[0], &[0.5], 0.0).is_err(),
            "short table"
        );
    }

    #[test]
    fn quantized_ffn_tracks_the_natural_weights_chain() {
        let (c, i, ipad) = (256usize, 64usize, 256usize);
        let nat = synth_gated(21, 3, c, i);
        let x = random_vec(22, c);
        // natural-domain f64 chain
        let act = FfnActivation::Situ {
            gate_beta: 4.0,
            value_beta: 25.0,
        };
        let mut a = vec![0f32; ipad];
        for r in 0..i {
            let g = dot_f64(&nat.gate[r * c..(r + 1) * c], &x) as f32;
            let u = dot_f64(&nat.up[r * c..(r + 1) * c], &x) as f32;
            a[r] = act_ref(act, g, u).expect("act");
        }
        let want: Vec<f32> = (0..c)
            .map(|r| dot_f64(&nat.down[r * i..(r + 1) * i], &a[..i]) as f32)
            .collect();
        let q = build_gated_q(nat, c, i, ipad, 0.0, None, None, 1).expect("quantize");
        let got = cpu_gated_ref(&q, None, None, &x, 1, c, i, ipad, act, false).expect("ref");
        let m = compare(&got, &want);
        assert!(
            m.rel_l2 < 0.3 && m.cos > 0.95,
            "4-bit chain drifted too far: {m:?}"
        );
    }

    #[test]
    fn awq_reference_is_consistent_with_unscaled_math() {
        // (W*s) quantized and x/s rotated approximates the unscaled product.
        let (c, i, ipad) = (256usize, 64usize, 256usize);
        let s = awq_vec(1, 0, c, c);
        let sd = awq_vec(1, 1, i, ipad);
        let x = random_vec(31, c);
        let act = FfnActivation::SwiGlu;
        let plain =
            build_gated_q(synth_gated(5, 5, c, i), c, i, ipad, 0.0, None, None, 1).expect("plain");
        let scaled = build_gated_q(
            synth_gated(5, 5, c, i),
            c,
            i,
            ipad,
            0.0,
            Some(&s),
            Some(&sd),
            1,
        )
        .expect("awq");
        let a =
            cpu_gated_ref(&plain, None, None, &x, 1, c, i, ipad, act, false).expect("plain ref");
        let b = cpu_gated_ref(&scaled, Some(&s), Some(&sd), &x, 1, c, i, ipad, act, false)
            .expect("awq ref");
        let m = compare(&b, &a);
        assert!(m.cos > 0.9, "awq and plain chains disagree: {m:?}");
    }

    #[test]
    fn bands_and_metrics_behave() {
        let want = vec![1.0f32, 2.0, 3.0, 4.0];
        let exact = compare(&want, &want);
        assert_eq!(band_check(&exact, true, false), (true, "bf16"));
        assert_eq!(band_check(&exact, false, false), (true, "f32"));
        let off: Vec<f32> = want.iter().map(|v| v * 1.0005).collect();
        let m = compare(&off, &want);
        assert_eq!(band_check(&m, true, false).0, true);
        assert_eq!(
            band_check(&m, false, false).0,
            false,
            "5e-4 is outside the strict f32 band"
        );
        assert_eq!(
            band_check(&m, false, true),
            (true, "fp16-wmma-core"),
            "grouped widening is labelled"
        );
        let nan = compare(&[f32::NAN, 1.0, 1.0, 1.0], &want);
        assert!(!band_check(&nan, true, true).0 && !band_check(&nan, false, true).0);
        let short = compare(&want[..3], &want);
        assert!(short.nonfinite > 0, "length mismatch must not pass");
    }

    #[test]
    fn synthetic_case_names_are_unique_and_builders_are_deterministic() {
        let mut names = std::collections::BTreeSet::new();
        for grp in synth_groups() {
            for sc in &grp.cases {
                assert!(
                    names.insert((grp.seed, sc.name())),
                    "duplicate name {}",
                    sc.name()
                );
            }
        }
        let grp = &synth_groups()[0];
        let sc = grp
            .cases
            .iter()
            .find(|s| s.kind == SKind::Tie(0))
            .expect("tie case");
        let a = build_synth(sc, grp);
        let b = build_synth(sc, grp);
        assert_eq!(a.x, b.x);
        assert_eq!(a.router_w, b.router_w);
        // the tied rows are identical and tied biases are equal
        let c = a.geom.c;
        assert_eq!(a.router_w[3 * c..4 * c], a.router_w[9 * c..10 * c]);
        assert_eq!(a.e_bias[3], a.e_bias[9]);
        let r = route_ref(a.router, &a.x, &a.router_w, &a.e_bias, a.n, &a.geom);
        for t in 0..a.n {
            assert_eq!(
                &r.idx[t * 3..(t + 1) * 3],
                &[0, 3, 9],
                "tie reference token {t}"
            );
        }
    }

    /// Reference weights of the reference selection, token-major `[n, k]`.
    fn ref_weights(r: &RouteRef, kind: RouterSel, g: &Geom) -> Vec<f32> {
        let n = r.idx.len() / g.k;
        (0..n)
            .flat_map(|t| {
                route_weights(
                    kind,
                    &r.scores[t * g.e..(t + 1) * g.e],
                    &r.idx[t * g.k..(t + 1) * g.k],
                )
            })
            .collect()
    }

    #[test]
    fn synthetic_routing_designs_hold_on_the_reference_router() {
        let groups = synth_groups();
        let check = |sc: &SynthCase, grp: &SynthGroup| {
            let c = build_synth(sc, grp);
            let r = route_ref(c.router, &c.x, &c.router_w, &c.e_bias, c.n, &c.geom);
            let w = ref_weights(&r, c.router, &c.geom);
            let issues = check_extras(&c, &r.idx, &w);
            assert!(issues.is_empty(), "{}: {issues:?}", c.name);
        };
        for grp in &groups {
            for sc in &grp.cases {
                if matches!(
                    sc.kind,
                    SKind::Coverage | SKind::Skew | SKind::Tie(_) | SKind::Unequal
                ) {
                    check(sc, grp);
                }
            }
        }
    }

    #[test]
    fn options_parse_and_reject_garbage() {
        let mut m = BTreeMap::new();
        let o = Opts::parse(&m).expect("defaults");
        assert_eq!(
            o.protocols,
            vec![GroupedMoeProtocol::Indexed, GroupedMoeProtocol::Grouped]
        );
        assert!(o.synthetic && !o.awq_real && !o.allow_refusal && o.expert_limit.is_none());
        m.insert("protocols".into(), "grouped".into());
        m.insert("synthetic".into(), "0".into());
        m.insert("expert_limit".into(), "8".into());
        let o = Opts::parse(&m).expect("explicit");
        assert_eq!(o.protocols, vec![GroupedMoeProtocol::Grouped]);
        assert!(!o.synthetic);
        assert_eq!(o.expert_limit, Some(8));
        m.insert("protocols".into(), "auto".into());
        assert!(Opts::parse(&m).is_err());
        m.insert("protocols".into(), "indexed".into());
        m.insert("awq_real".into(), "maybe".into());
        assert!(Opts::parse(&m).is_err());
    }

    #[test]
    fn negative_flag_is_refused_before_touching_anything() {
        let mut m = BTreeMap::new();
        m.insert("negative".to_string(), "1".to_string());
        let e = run(Path::new("/nonexistent"), &m).expect_err("negative must error");
        assert!(e.contains("no negative controls"), "{e}");
    }

    #[test]
    fn missing_or_empty_manifest_is_an_error_with_the_path() {
        let dir = scratch_dir().join("empty_fixtures");
        std::fs::create_dir_all(dir.join("moe")).expect("mkdir");
        let e = run(&dir, &BTreeMap::new()).expect_err("missing manifest");
        assert!(e.contains("manifest.json"), "{e}");
        std::fs::write(dir.join("moe").join("manifest.json"), r#"{"cases":[]}"#).expect("write");
        let e = run(&dir, &BTreeMap::new()).expect_err("empty manifest");
        assert!(e.contains("no cases") && e.contains("manifest.json"), "{e}");
    }

    #[test]
    fn fixture_loader_reads_a_tiny_case_and_reports_the_offending_name() {
        let dir = scratch_dir().join("tiny_fixtures");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let (n, e, k, c) = (2usize, 4usize, 2usize, 256usize);
        let f = |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let x = random_vec(1, n * c);
        let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
            ("x", "F32", vec![n, c], f(&x)),
            ("router_w", "F32", vec![e, c], f(&random_vec(2, e * c))),
            ("e_bias", "F32", vec![e], f(&[0.0; 4])),
            ("scores", "F32", vec![n, e], f(&[0.5; 8])),
            ("topk_idx", "F32", vec![n, k], f(&[0.0, 1.0, 2.0, 3.0])),
            ("topk_w", "F32", vec![n, k], f(&[0.5; 4])),
            ("routed", "F32", vec![n, c], f(&vec![0.0; n * c])),
            ("shared", "F32", vec![n, c], f(&vec![0.0; n * c])),
            ("out", "F32", vec![n, c], f(&vec![0.0; n * c])),
        ];
        write_safetensors(&dir.join("case.safetensors"), &tensors);
        let case = json!({
            "name": "tiny", "file": "case.safetensors", "mode": "bf16",
            "dims": {"N": n, "E": e, "K": k, "C": c, "I": 64, "Ipad": 256},
            "params": {"activation": "situ", "router": "sigmoid_bias", "shared": true, "gate_beta": 4.0, "value_beta": 25.0},
            "weights": "w.safetensors"
        });
        let (cd, src) = load_fixture_case(&dir, &case).expect("load");
        assert_eq!((cd.n, cd.geom.e, cd.shared, cd.bf16), (n, e, true, true));
        assert!(matches!(src, WSrc::File(p) if p.ends_with("w.safetensors")));
        assert_eq!(
            cd.expected.as_ref().map(|x| x.idx.clone()),
            Some(vec![0, 1, 2, 3])
        );
        // a shape error names the offending tensor
        let mut bad = case.clone();
        bad["dims"]["N"] = json!(3);
        let err = load_fixture_case(&dir, &bad).err().expect("shape mismatch");
        assert!(err.contains("`x`"), "{err}");
        // unknown activation is a case failure, not a panic
        let mut bad = case.clone();
        bad["params"]["activation"] = json!("gelu");
        assert!(load_fixture_case(&dir, &bad).is_err());
        // an expert id outside [0, E) is rejected
        let mut t2 = tensors;
        t2[4] = ("topk_idx", "F32", vec![n, k], f(&[0.0, 1.0, 2.0, 9.0]));
        write_safetensors(&dir.join("case.safetensors"), &t2);
        let err = load_fixture_case(&dir, &case).err().expect("bad id");
        assert!(err.contains("topk_idx"), "{err}");
    }
}
