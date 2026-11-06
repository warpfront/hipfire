// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 8-bit HC mixer weights (`W[320, 10240]` down, `W[10240, 320]` up) read by the gfx1201 PeaceMaker module
//! `hc8_wmma_pm_gfx1201`: host quantizers, device weight, launcher.
//!
//! Weight formats (per-row f32 scale `s[m]`, applied once in the kernel epilogue):
//! * `e4m3`: `s = 2^ceil(log2(rowamax / 448))`, `q = OCP E4M3 RNE(w / s)` (subnormals kept, `|q| <= 448`).
//! * `i8`: `s = rowamax / 127`, `q = clamp(rint(w / s), -127, 127)`.
//!
//! An all-zero row gets `s = 1` (any finite value is exact; the kernel multiplies by it).
//! The 8-bit bytes are pre-swizzled into 512-B WMMA tiles by `peacemaker_isa::kernels::hc8_wmma_gfx1201::swizzle_hc8`.
//! Kernel entries `hc8_gemm_{fmt}_mw{MW}_kw{KW}_n{NT}_{epi}`; kernarg 48 B: `q@0 s@8 x@16 y@24` pointers,
//! `M u32@32 K u32@36 rows u32@40 act_scale f32@44`; block `32*MW*KW`, grid x `M / (16*MW)`.

use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
pub use peacemaker_isa::kernels::hc8_wmma_gfx1201::swizzle_hc8;
use std::sync::LazyLock;

pub const MODULE: &str = "hc8_wmma_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco");
pub const IMAGE_SHA256: &str = include_str!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco.sha256");
pub const KERNARG_BYTES: usize = 48;
/// The `(mw, kw)` geometries the module exports.
pub const GEOMS: [Hc8Geom; 5] = [
    Hc8Geom { mw: 1, kw: 8 },
    Hc8Geom { mw: 1, kw: 16 },
    Hc8Geom { mw: 4, kw: 1 },
    Hc8Geom { mw: 8, kw: 1 },
    Hc8Geom { mw: 2, kw: 4 },
];
pub const MAX_ROWS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hc8Fmt {
    E4m3,
    I8,
}

impl Hc8Fmt {
    pub fn name(self) -> &'static str {
        match self {
            Hc8Fmt::E4m3 => "e4m3",
            Hc8Fmt::I8 => "i8",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hc8Geom {
    pub mw: usize,
    pub kw: usize,
}

/// Starting geometry table (the screen decides the final one): long K splits across 16 waves, short K
/// takes four M-tiles per single-wave group.
pub fn default_geom(_m: usize, k: usize, _rows: usize) -> Hc8Geom {
    if k >= 4096 {
        Hc8Geom { mw: 1, kw: 16 }
    } else {
        Hc8Geom { mw: 4, kw: 1 }
    }
}

/// 8-bit weight on the device: swizzled bytes `q` (`m * k`) and f32 row scales `s` (`m`).
pub struct Hc8Weight {
    pub q: GpuTensor,
    pub s: GpuTensor,
    pub m: usize,
    pub k: usize,
    pub fmt: Hc8Fmt,
}

impl Hc8Weight {
    /// Weight bytes a call reads: `q` plus `s`.
    pub fn bytes(&self) -> usize {
        self.m * self.k + self.m * 4
    }

    pub fn upload(
        gpu: &mut Gpu,
        fmt: Hc8Fmt,
        swizzled: &[u8],
        scales: &[f32],
        m: usize,
        k: usize,
    ) -> HipResult<Self> {
        if swizzled.len() != m * k || scales.len() != m {
            return Err(err(format!(
                "upload: {} q bytes / {} scales for [{m}, {k}]",
                swizzled.len(),
                scales.len()
            )));
        }
        let q = gpu.upload_raw(swizzled, &[swizzled.len()])?;
        let s = gpu.upload_f32(scales, &[m])?;
        Ok(Self { q, s, m, k, fmt })
    }

    /// Quantize BF16 bits `[m, k]` row-major and upload.
    pub fn from_bf16(gpu: &mut Gpu, fmt: Hc8Fmt, bf16: &[u16], m: usize, k: usize) -> HipResult<Self> {
        let (q, s) = match fmt {
            Hc8Fmt::E4m3 => quantize_e4m3_rows(bf16, m, k),
            Hc8Fmt::I8 => quantize_i8_rows(bf16, m, k),
        };
        Self::upload(gpu, fmt, &q, &s, m, k)
    }
}

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("hc8: {}", message.as_ref()))
}

// ---- host quantizers ------------------------------------------------------------------------

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Value of the non-negative OCP E4M3 code `c` (`0..=0x7e`).
fn e4m3_mag(c: u8) -> f32 {
    let (e, m) = (i32::from(c >> 3) & 15, f32::from(c & 7));
    if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Decode an OCP E4M3 byte (`0x7f` / `0xff` are NaN).
pub fn decode_e4m3(b: u8) -> f32 {
    if b & 0x7f == 0x7f {
        return f32::NAN;
    }
    let v = e4m3_mag(b & 0x7f);
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// Round-to-nearest-even encode of `v` (`|v| <= 448`) to an OCP E4M3 byte; subnormals kept.
pub fn encode_e4m3(v: f32) -> u8 {
    let a = v.abs();
    // Largest code whose magnitude is <= a.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if e4m3_mag(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let code = if lo == 0x7e {
        lo
    } else {
        let (d_lo, d_hi) = (a - e4m3_mag(lo), e4m3_mag(lo + 1) - a);
        if d_lo < d_hi || (d_lo == d_hi && lo % 2 == 0) {
            lo
        } else {
            lo + 1
        }
    };
    if v.is_sign_negative() {
        code | 0x80
    } else {
        code
    }
}

fn row_amax(bf16: &[u16], k: usize, r: usize) -> f32 {
    bf16[r * k..(r + 1) * k]
        .iter()
        .fold(0.0f32, |a, &h| a.max(bf16_to_f32(h).abs()))
}

/// Row-major E4M3 bytes `[m, k]` and f32 row scales.
pub fn quantize_e4m3_rows_rowmajor(bf16: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(bf16.len(), m * k);
    let mut q = vec![0u8; m * k];
    let mut s = vec![1.0f32; m];
    for r in 0..m {
        let amax = row_amax(bf16, k, r);
        if amax > 0.0 {
            // 2^ceil(log2(amax / 448)), exact via the exponent field (amax / 448 is a normal f32 here).
            let ratio = amax / 448.0;
            let bits = ratio.to_bits();
            let e = ((bits >> 23) & 0xff) as i32 - 127;
            let sc = if bits & 0x7f_ffff == 0 { e } else { e + 1 };
            s[r] = 2f32.powi(sc);
        }
        for c in 0..k {
            q[r * k + c] = encode_e4m3(bf16_to_f32(bf16[r * k + c]) / s[r]);
        }
    }
    (q, s)
}

/// Row-major int8 bytes `[m, k]` (two's complement) and f32 row scales.
pub fn quantize_i8_rows_rowmajor(bf16: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(bf16.len(), m * k);
    let mut q = vec![0u8; m * k];
    let mut s = vec![1.0f32; m];
    for r in 0..m {
        let amax = row_amax(bf16, k, r);
        if amax > 0.0 {
            s[r] = amax / 127.0;
        }
        for c in 0..k {
            let v = (bf16_to_f32(bf16[r * k + c]) / s[r])
                .round_ties_even()
                .clamp(-127.0, 127.0);
            q[r * k + c] = (v as i8) as u8;
        }
    }
    (q, s)
}

/// E4M3 quantizer: swizzled weight bytes and row scales.
pub fn quantize_e4m3_rows(bf16: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (q, s) = quantize_e4m3_rows_rowmajor(bf16, m, k);
    (swizzle_hc8(&q, m, k), s)
}

/// int8 quantizer: swizzled weight bytes and row scales.
pub fn quantize_i8_rows(bf16: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (q, s) = quantize_i8_rows_rowmajor(bf16, m, k);
    (swizzle_hc8(&q, m, k), s)
}

/// Dequantized `[m, k]` f32 (`q * s`) from row-major bytes.
pub fn dequant_rowmajor(fmt: Hc8Fmt, q: &[u8], s: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| {
            let v = match fmt {
                Hc8Fmt::E4m3 => decode_e4m3(q[i]),
                Hc8Fmt::I8 => f32::from(q[i] as i8),
            };
            v * s[i / k]
        })
        .collect()
}

// ---- launcher -------------------------------------------------------------------------------

static IMAGE_VERIFIED: LazyLock<Result<(), String>> = LazyLock::new(|| {
    use sha2::{Digest, Sha256};
    let expected = IMAGE_SHA256.trim();
    if expected.is_empty() {
        return Err("hc8 object SHA256 is not pinned".into());
    }
    let actual = format!("{:x}", Sha256::digest(IMAGE));
    if actual != expected {
        return Err(format!(
            "hc8 object SHA256 {actual} differs from pinned {expected}"
        ));
    }
    Ok(())
});

/// The pinned-SHA256 check of the embedded image, shared with `crate::hc8_weight`'s dequant entries.
pub(crate) fn verify_image() -> Result<(), String> {
    IMAGE_VERIFIED.clone()
}

/// Entry name for `fmt`, geometry, column tiles and epilogue.
pub fn entry_name(fmt: Hc8Fmt, g: Hc8Geom, nt: usize, act: bool) -> String {
    format!(
        "hc8_gemm_{}_mw{}_kw{}_n{}_{}",
        fmt.name(),
        g.mw,
        g.kw,
        nt,
        if act { "act" } else { "set" }
    )
}

/// `y[rows, m] = epi(x[rows, k] . dequant(W)^T)` with `epi` = `hc_act(acc * s, scale)` for `act = Some(scale)`
/// else `acc * s`. `x` is F32 `[rows, k]` (BF16-exact values), `y` F32 `[rows, m]`; `rows` in `1..=64`.
/// Returns the launched entry name.
pub fn hc8_gemm(
    gpu: &mut Gpu,
    w: &Hc8Weight,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    act: Option<f32>,
    geom: Hc8Geom,
) -> HipResult<String> {
    if gpu.arch != "gfx1201" {
        return Err(err("requires exact gfx1201; no HIP fallback"));
    }
    if !GEOMS.contains(&geom) {
        return Err(err(format!("geometry {geom:?} is not exported")));
    }
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(err(format!("rows {rows} not in 1..={MAX_ROWS}")));
    }
    let (m, k) = (w.m, w.k);
    if k == 0 || k % (32 * geom.kw) != 0 {
        return Err(err(format!("k {k} % (32 * kw {}) != 0", geom.kw)));
    }
    if m == 0 || m % (16 * geom.mw) != 0 {
        return Err(err(format!("m {m} % (16 * mw {}) != 0", geom.mw)));
    }
    for (t, dtype, bytes, name) in [
        (x, DType::F32, rows * k * 4, "x"),
        (y, DType::F32, rows * m * 4, "y"),
    ] {
        if t.dtype != dtype || t.buf.size() < bytes {
            return Err(err(format!("{name} must be F32 with >= {bytes} bytes")));
        }
    }
    if w.q.buf.size() < m * k || w.s.buf.size() < m * 4 {
        return Err(err("weight tensors smaller than [m, k]"));
    }
    IMAGE_VERIFIED.clone().map_err(err)?;
    let nt = rows.div_ceil(16);
    let name = entry_name(w.fmt, geom, nt, act.is_some());
    gpu.bind_thread()?;
    gpu.ensure_embedded_kernel(MODULE, IMAGE, &name)?;
    let mut args = [0u8; KERNARG_BYTES];
    for (i, p) in [w.q.buf.as_ptr(), w.s.buf.as_ptr(), x.buf.as_ptr(), y.buf.as_ptr()]
        .into_iter()
        .enumerate()
    {
        args[i * 8..i * 8 + 8].copy_from_slice(&(p as u64).to_le_bytes());
    }
    args[32..36].copy_from_slice(&(m as u32).to_le_bytes());
    args[36..40].copy_from_slice(&(k as u32).to_le_bytes());
    args[40..44].copy_from_slice(&(rows as u32).to_le_bytes());
    args[44..48].copy_from_slice(&act.unwrap_or(0.0).to_le_bytes());
    gpu.launch_blob_recorded(
        &name,
        [(m / (16 * geom.mw)) as u32, 1, 1],
        [(32 * geom.mw * geom.kw) as u32, 1, 1],
        0,
        &mut args,
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    Ok(name)
}

/// Fused HC read-up `hc8_read_up_e4m3_n{1..4}`: `mixed[rows, hidden]` = the up projection of `low`
/// on the HC8 weight `w` (`[4 * hidden, low_rank]`) folded with `normalized` through the product
/// `hyper_read_projected_f32` DAG (four branches), word-equal to [`hc8_gemm`] (`set`) followed by
/// `hyper_read_projected`. `low` is F32 `[rows, low_rank]` (BF16-exact), `normalized` F32
/// `[rows, 4 * hidden]`, `mixed` F32 `[rows, hidden]`; `rows` in `1..=64`, `hidden % 16 == 0`,
/// `low_rank % 32 == 0`. Returns the launched entry name.
pub fn hc8_read_up(
    gpu: &mut Gpu,
    w: &Hc8Weight,
    low: &GpuTensor,
    normalized: &GpuTensor,
    mixed: &GpuTensor,
    rows: usize,
    hidden: usize,
) -> HipResult<String> {
    use peacemaker_isa::kernels::hc8_wmma_gfx1201::{read_up_symbol, READ_UP_BLOCK, READ_UP_KERNARG_BYTES};
    if gpu.arch != "gfx1201" {
        return Err(err("requires exact gfx1201; no HIP fallback"));
    }
    if w.fmt != Hc8Fmt::E4m3 {
        return Err(err("read_up: the fused read exists for e4m3 only"));
    }
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(err(format!("rows {rows} not in 1..={MAX_ROWS}")));
    }
    let k = w.k;
    if hidden == 0 || hidden % 16 != 0 || w.m != 4 * hidden || k == 0 || k % 32 != 0 {
        return Err(err(format!("read_up: weight [{}, {k}] is not [4 * hidden {hidden}, K % 32 == 0]", w.m)));
    }
    for (t, bytes, name) in [
        (low, rows * k * 4, "low"),
        (normalized, rows * 4 * hidden * 4, "normalized"),
        (mixed, rows * hidden * 4, "mixed"),
    ] {
        if t.dtype != DType::F32 || t.buf.size() < bytes {
            return Err(err(format!("read_up: {name} must be F32 with >= {bytes} bytes")));
        }
    }
    IMAGE_VERIFIED.clone().map_err(err)?;
    let name = read_up_symbol(rows.div_ceil(16) as u32);
    gpu.bind_thread()?;
    gpu.ensure_embedded_kernel(MODULE, IMAGE, &name)?;
    let mut args = [0u8; READ_UP_KERNARG_BYTES as usize];
    for (i, p) in [
        w.q.buf.as_ptr(),
        w.s.buf.as_ptr(),
        low.buf.as_ptr(),
        normalized.buf.as_ptr(),
        mixed.buf.as_ptr(),
    ]
    .into_iter()
    .enumerate()
    {
        args[i * 8..i * 8 + 8].copy_from_slice(&(p as u64).to_le_bytes());
    }
    args[40..44].copy_from_slice(&(hidden as u32).to_le_bytes());
    args[44..48].copy_from_slice(&(k as u32).to_le_bytes());
    args[48..52].copy_from_slice(&(rows as u32).to_le_bytes());
    gpu.launch_blob_recorded(
        &name,
        [(hidden / 16) as u32, 1, 1],
        [READ_UP_BLOCK, 1, 1],
        0,
        &mut args,
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn e4m3_codec_roundtrip_and_extremes() {
        for c in 0..=0x7eu8 {
            assert_eq!(encode_e4m3(e4m3_mag(c)), c);
            assert_eq!(encode_e4m3(-e4m3_mag(c)), c | 0x80);
        }
        assert_eq!(e4m3_mag(0x7e), 448.0);
        assert_eq!(e4m3_mag(1), 2f32.powi(-9));
        assert_eq!(decode_e4m3(0x7f).is_nan(), true);
        assert_eq!(encode_e4m3(448.0), 0x7e);
    }

    #[test]
    fn e4m3_rounds_to_nearest_even() {
        // 1.0625 is the tie between 1.0 (0x38, even) and 1.125 (0x39).
        assert_eq!(encode_e4m3(1.0625), 0x38);
        // 1.1875 is the tie between 1.125 (0x39) and 1.25 (0x3a, even).
        assert_eq!(encode_e4m3(1.1875), 0x3a);
        // Subnormal tie: 1.5 * 2^-9 between 2^-9 (code 1) and 2^-8 (code 2, even).
        assert_eq!(encode_e4m3(1.5 * 2f32.powi(-9)), 2);
        assert_eq!(encode_e4m3(0.25 * 2f32.powi(-9)), 0);
        assert_eq!(encode_e4m3(-0.0), 0x80);
    }

    #[test]
    fn e4m3_scale_is_power_of_two_covering_amax() {
        let (m, k) = (3, 8);
        let mut w = vec![bf(0.0); m * k];
        w[0] = bf(448.0); // row 0: s = 1
        w[k] = bf(450.0); // row 1: first bf16 above 448 (449 rounds to 448) -> s = 2
        w[2 * k + 3] = bf(-0.01); // row 2
        let (q, s) = quantize_e4m3_rows_rowmajor(&w, m, k);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 2.0);
        assert!(s[2].log2().fract() == 0.0 && 0.01 / s[2] <= 448.0 && 0.01 / (s[2] / 2.0) > 448.0);
        assert_eq!(q[0], 0x7e);
        for r in 0..m {
            assert!(q[r * k..(r + 1) * k]
                .iter()
                .all(|&b| decode_e4m3(b).abs() <= 448.0));
        }
        // Dequant error bounded by half an E4M3 ulp of the row max.
        let dq = dequant_rowmajor(Hc8Fmt::E4m3, &q, &s, m, k);
        assert!((dq[k] - 450.0).abs() <= 16.0 * s[1]);
    }

    #[test]
    fn i8_scale_and_rounding() {
        let (m, k) = (2, 4);
        let w = vec![
            bf(127.0), bf(-127.0), bf(0.5), bf(1.5), // row 0: s = 1: 0.5 -> 0, 1.5 -> 2
            bf(0.0), bf(0.0), bf(0.0), bf(0.0), // zero row
        ];
        let (q, s) = quantize_i8_rows_rowmajor(&w, m, k);
        assert_eq!(s, vec![1.0, 1.0]);
        assert_eq!(&q[..4], &[127u8, (-127i8) as u8, 0, 2]);
        assert_eq!(&q[4..], &[0u8; 4]);
        let w2 = vec![bf(2.0), bf(-1.0), bf(0.0), bf(1.0)];
        let (q2, s2) = quantize_i8_rows_rowmajor(&w2, 1, 4);
        assert_eq!(s2[0], 2.0 / 127.0);
        assert_eq!(q2[0], 127);
        assert_eq!(q2[1] as i8, -64); // -63.5 ties to even -64
    }

    #[test]
    fn entry_names_and_default_geom() {
        assert_eq!(
            entry_name(Hc8Fmt::E4m3, Hc8Geom { mw: 1, kw: 16 }, 3, true),
            "hc8_gemm_e4m3_mw1_kw16_n3_act"
        );
        assert_eq!(default_geom(320, 10240, 1), Hc8Geom { mw: 1, kw: 16 });
        assert_eq!(default_geom(10240, 320, 1), Hc8Geom { mw: 4, kw: 1 });
    }
}
