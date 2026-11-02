// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared helpers for the Qwen4 Tier C cases: Qwen4 geometry, deterministic
//! inputs, BF16 bit helpers, upload/download shims and comparators that report
//! max-abs / max-rel / max-ulp and the first offending element.

use rdna_compute::{DType, Gpu, GpuTensor};

// ── Qwen4 geometry (crates/hipfire-arch-qwen4/src/config.rs, Flash-Next pin) ──
pub const HIDDEN: usize = 2560;
pub const HC_STREAMS: usize = 4;
pub const HC_WIDE: usize = HIDDEN * HC_STREAMS; // 10,240
pub const GDN_KEY_HEADS: usize = 16;
pub const GDN_VALUE_HEADS: usize = 48;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_CONV_KERNEL: usize = 4;
pub const QSA_Q_HEADS: usize = 24;
pub const QSA_KV_HEADS: usize = 2;
pub const QSA_HEAD_DIM: usize = 256;
pub const QSA_INDEX_HEADS: usize = 4;
pub const QSA_INDEX_DIM: usize = 128;
pub const QSA_COMPRESS: usize = 4;
pub const QSA_BUDGET: usize = 2048;
pub const MOE_EXPERTS: usize = 512;
pub const MOE_TOP_K: usize = 10;
pub const MOE_INTERMEDIATE: usize = 640;
pub const VOCAB: usize = 248_320;

/// Row counts every row-batched case runs at: decode (1), small prefill, and a
/// count that is not a multiple of any power-of-two tile (wave64 / tile tails).
pub const ROW_COUNTS: [usize; 3] = [1, 7, 33];

// ── Tolerances: the Qwen4 format's frozen ones ─────────────────────────────────
// `crates/hipfire-arch-qwen4/reference_oracle/upstream.py:275-276`
// (also `generate_fixtures.py:150-151`). Never loosen.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub name: &'static str,
    pub atol: f32,
    pub rtol: f32,
}

pub const F32_ACCUM: Tol = Tol {
    name: "f32_accumulation",
    atol: 1.0e-5,
    rtol: 1.0e-5,
};
pub const BF16_INPUT_F32_ACCUM: Tol = Tol {
    name: "bf16_input_f32_accumulation",
    atol: 0.015625,
    rtol: 0.03125,
};

// ── Deterministic inputs ──────────────────────────────────────────────────────
/// SplitMix64: deterministic, seedable, no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
    /// `n` values already representable in BF16 (so a BF16 upload is exact).
    pub fn vec_bf16_exact(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| bf16_round(self.range(lo, hi))).collect()
    }
}

// ── BF16 bit helpers ──────────────────────────────────────────────────────────
/// RNE F32 → BF16 bits (same rule as the kernels' `__float2bfloat16` / the
/// frozen contract: `(u + 0x7FFF + ((u >> 16) & 1)) >> 16`; NaN quieted).
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let u = value.to_bits();
    if value.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
pub fn bf16_round(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(value))
}

// ── Upload / download shims (errors as String) ───────────────────────────────
pub fn e<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

pub fn up_f32(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, &[data.len()]).map_err(e)
}
/// Upload F32 values as a BF16 tensor (RNE), `DType::BF16`, shape `[n]`.
pub fn up_bf16(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
        .collect();
    let mut t = gpu.upload_raw(&bytes, &[data.len()]).map_err(e)?;
    t.dtype = DType::BF16;
    Ok(t)
}
/// 4-byte integer tensors are carried as `DType::F32` bit containers (there is
/// no integer DType), shape `[n]`, so `byte_size()` is `4n` and downloads are
/// complete. Kernels read the bits as u32/i32.
pub fn up_u32(gpu: &mut Gpu, data: &[u32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[data.len()]).map_err(e)?;
    t.dtype = DType::F32;
    Ok(t)
}
pub fn up_i32(gpu: &mut Gpu, data: &[i32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[data.len()]).map_err(e)?;
    t.dtype = DType::F32;
    Ok(t)
}
pub fn zeros_f32(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[n], DType::F32).map_err(e)
}
/// F32 tensor pre-filled with a NaN sentinel so unwritten outputs fail loudly.
pub fn sentinel_f32(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    up_f32(gpu, &vec![f32::from_bits(0x7FC0_DEAD); n])
}
pub fn down_f32(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>, String> {
    gpu.download_f32(t).map_err(e)
}
pub fn down_bytes(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>, String> {
    gpu.download_raw_bytes(t).map_err(e)
}
pub fn down_bf16(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u16>, String> {
    Ok(down_bytes(gpu, t)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
pub fn down_u32(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u32>, String> {
    Ok(down_bytes(gpu, t)?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub fn down_i32(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<i32>, String> {
    Ok(down_u32(gpu, t)?.into_iter().map(|v| v as i32).collect())
}
pub fn free_all(gpu: &mut Gpu, tensors: Vec<GpuTensor>) -> Result<(), String> {
    for t in tensors {
        gpu.free_tensor(t).map_err(e)?;
    }
    Ok(())
}
pub fn sync(gpu: &Gpu) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(e)
}

// ── Comparators ───────────────────────────────────────────────────────────────
/// Distance in units in the last place between two finite F32 values
/// (sign-magnitude → ordered integer).
pub fn ulp_f32(a: f32, b: f32) -> u64 {
    fn ordered(x: f32) -> i64 {
        let u = x.to_bits() as i32 as i64;
        if u < 0 {
            i32::MIN as i64 - u
        } else {
            u
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub n: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub max_ulp: u64,
    pub worst: usize,
}

impl Stats {
    pub fn summary(&self) -> String {
        format!(
            "n={} max_abs={:.3e} max_rel={:.3e} max_ulp={} worst@{}",
            self.n, self.max_abs, self.max_rel, self.max_ulp, self.worst
        )
    }
}

fn stats_f32(got: &[f32], want: &[f32]) -> Stats {
    let mut s = Stats {
        n: got.len(),
        ..Stats::default()
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        let rel = if w != 0.0 { abs / w.abs() } else { abs };
        if abs > s.max_abs || abs.is_nan() {
            s.max_abs = if abs.is_nan() { f32::INFINITY } else { abs };
            s.worst = i;
        }
        s.max_rel = s.max_rel.max(rel);
        if g.is_finite() && w.is_finite() {
            s.max_ulp = s.max_ulp.max(ulp_f32(g, w));
        }
    }
    s
}

/// Bit-exact F32 compare (−0 vs +0 counts as a mismatch).
pub fn exact_f32(label: &str, got: &[f32], want: &[f32]) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let s = stats_f32(got, want);
    let bad: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&i) = bad.first() {
        return Err(format!(
            "{label}: {} / {} elements differ bitwise; first @{i}: got {:?} ({:#010x}) want {:?} ({:#010x}); {}",
            bad.len(),
            got.len(),
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits(),
            s.summary()
        ));
    }
    Ok(format!("{label}: bit-exact n={}", got.len()))
}

/// Element-wise `|got − want| ≤ atol + rtol·|want|` with the given frozen tol;
/// NaN/Inf in `got` where `want` is finite always fails.
pub fn close_f32(label: &str, got: &[f32], want: &[f32], tol: Tol) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let s = stats_f32(got, want);
    let mut n_bad = 0usize;
    let mut first = None;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ok = if w.is_finite() {
            g.is_finite() && (g - w).abs() <= tol.atol + tol.rtol * w.abs()
        } else {
            g.to_bits() == w.to_bits()
        };
        if !ok {
            n_bad += 1;
            first.get_or_insert(i);
        }
    }
    if let Some(i) = first {
        return Err(format!(
            "{label}: {n_bad} / {} outside {} (atol {:e} rtol {:e}); first @{i}: got {:?} want {:?}; {}",
            got.len(),
            tol.name,
            tol.atol,
            tol.rtol,
            got[i],
            want[i],
            s.summary()
        ));
    }
    Ok(format!("{label}: within {} {}", tol.name, s.summary()))
}

/// Bit-exact compare of BF16 bit patterns.
pub fn exact_bf16(label: &str, got: &[u16], want: &[u16]) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&i) = bad.first() {
        let max_ulp = bad
            .iter()
            .map(|&j| (got[j] as i32 - want[j] as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        return Err(format!(
            "{label}: {} / {} BF16 differ; first @{i}: got {:#06x} ({}) want {:#06x} ({}); max raw-bit delta {max_ulp}",
            bad.len(),
            got.len(),
            got[i],
            bf16_bits_to_f32(got[i]),
            want[i],
            bf16_bits_to_f32(want[i])
        ));
    }
    Ok(format!("{label}: BF16 bit-exact n={}", got.len()))
}

/// Exact compare for integer outputs (ids, indices, counts, permutations).
pub fn exact_ints<T: PartialEq + std::fmt::Debug>(
    label: &str,
    got: &[T],
    want: &[T],
) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&i) = bad.first() {
        return Err(format!(
            "{label}: {} / {} differ; first @{i}: got {:?} want {:?}",
            bad.len(),
            got.len(),
            got[i],
            want[i]
        ));
    }
    Ok(format!("{label}: exact n={}", got.len()))
}

/// Join per-check summaries. Any `Err` fails the case; the failure message
/// carries EVERY failing check (with its data) and the count of passing ones.
pub fn all(results: Vec<Result<String, String>>) -> Result<String, String> {
    let total = results.len();
    let mut passed = Vec::with_capacity(total);
    let mut failed = Vec::new();
    for r in results {
        match r {
            Ok(msg) => passed.push(msg),
            Err(msg) => failed.push(msg),
        }
    }
    if failed.is_empty() {
        Ok(passed.join("; "))
    } else {
        Err(format!(
            "{} / {total} checks failed: {}",
            failed.len(),
            failed.join(" | ")
        ))
    }
}
