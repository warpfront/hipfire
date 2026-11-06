// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Qwen4 HC8 sidecar weight on the device (`DType::HC8E4M3Row`, qt=57): one buffer holding the
//! swizzled E4M3 bytes `q` (`m * k`, [`crate::hc8::swizzle_hc8`] order) followed by the f32 row
//! scales `s` (`m`), so a resident weight is a single tensor like every other dtype.
//!
//! * decode / verify (`rows <= `[`crate::hc8::MAX_ROWS`]): [`Gpu::hc8_gemm_tensor`] views the
//!   buffer as an [`Hc8Weight`] (non-owning) and launches the PeaceMaker `hc8_gemm` kernel;
//! * prefill and every BF16-only consumer: [`Gpu::hc8_dequant_bf16`] / [`Gpu::hc8_bf16_scratch`] expand
//!   the weight exactly (`q * s` is a BF16) into a reused per-GPU scratch, and the F16 shadow of
//!   [`Gpu::ensure_bf16_f16_shadow`] is built from the E4M3 buffer itself (keyed on its pointer),
//!   never from the reused scratch.

use crate::hc8::{self, Hc8Fmt, Hc8Weight};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;


fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("hc8 weight: {}", message.as_ref()))
}

/// Exact payload bytes of an `[m, k]` HC8 weight: `m * k` E4M3 bytes plus `4 * m` scale bytes.
/// `None` unless `m % 16 == 0`, `k % 32 == 0` (the swizzle tile) and nothing overflows.
pub fn hc8_payload_bytes(m: usize, k: usize) -> Option<usize> {
    if m == 0 || k == 0 || m % 16 != 0 || k % 32 != 0 {
        return None;
    }
    m.checked_mul(k)?.checked_add(m.checked_mul(4)?)
}

/// Whether `[m, k]` has a launchable decode geometry ([`hc8::default_geom`] divides it).
pub fn hc8_decode_shape_supported(m: usize, k: usize) -> bool {
    let g = hc8::default_geom(m, k, 1);
    hc8_payload_bytes(m, k).is_some() && k % (32 * g.kw) == 0 && m % (16 * g.mw) == 0
}

/// Swizzled byte offset of element `(r, c)` of an `[_, k]` HC8 weight: the closed form the
/// `hc8_dequant` kernels evaluate (the inverse of [`hc8::swizzle_hc8`]'s `for_each_slot`).
pub fn hc8_swizzled_offset(r: usize, c: usize, k: usize) -> usize {
    let (mt, rr, kt, kk) = (r >> 4, r & 15, c >> 5, c & 31);
    let (step, rem) = (kk >> 4, kk & 15);
    let h = (rem >> 2) & 1;
    let j = (rem & 3) + 4 * (rem >> 3);
    (mt * (k >> 5) + kt) * 512 + (h * 16 + rr) * 16 + step * 8 + j
}

impl Gpu {
    fn hc8_check(&self, w: &GpuTensor, m: usize, k: usize) -> HipResult<usize> {
        if self.arch != "gfx1201" {
            return Err(err("requires exact gfx1201"));
        }
        if w.dtype != DType::HC8E4M3Row {
            return Err(err(format!("weight is {:?}, not HC8E4M3Row", w.dtype)));
        }
        let bytes = hc8_payload_bytes(m, k).ok_or_else(|| err(format!("[{m}, {k}] is not an HC8 shape")))?;
        if w.buf.size() < bytes {
            return Err(err(format!(
                "weight holds {} bytes, an [{m}, {k}] payload is {bytes}",
                w.buf.size()
            )));
        }
        Ok(bytes)
    }

    /// Non-owning [`Hc8Weight`] over a resident `HC8E4M3Row` tensor.
    pub fn hc8_weight_view(&self, w: &GpuTensor, m: usize, k: usize) -> HipResult<Hc8Weight> {
        let bytes = self.hc8_check(w, m, k)?;
        debug_assert_eq!(bytes, m * k + 4 * m);
        Ok(Hc8Weight {
            q: GpuTensor {
                buf: w.buf.byte_view(0, m * k),
                shape: vec![m * k],
                dtype: DType::Raw,
            },
            s: GpuTensor {
                buf: w.buf.byte_view(m * k, 4 * m),
                shape: vec![m],
                dtype: DType::F32,
            },
            m,
            k,
            fmt: Hc8Fmt::E4m3,
        })
    }

    /// `y[rows, m] = epi(x[rows, k] . W^T)` on the `hc8_gemm` PeaceMaker kernel, `rows` in
    /// `1..=`[`hc8::MAX_ROWS`]; `act = Some(scale)` applies the HC activation, `None` is the plain
    /// scaled product. `x` / `y` are F32 (`x` BF16-exact). Returns the launched entry name.
    #[allow(clippy::too_many_arguments)]
    pub fn hc8_gemm_tensor(
        &mut self,
        w: &GpuTensor,
        m: usize,
        k: usize,
        x: &GpuTensor,
        y: &GpuTensor,
        rows: usize,
        act: Option<f32>,
    ) -> HipResult<String> {
        let view = self.hc8_weight_view(w, m, k)?;
        let geom = hc8::default_geom(m, k, rows);
        hc8::hc8_gemm(self, &view, x, y, rows, act, geom)
    }

    /// Fused HC read-up ([`hc8::hc8_read_up`]) on a resident `HC8E4M3Row` up weight `[4 * hidden, k]`:
    /// `mixed` from `low` and `normalized`, word-equal to `hc8_gemm_tensor` (no act) +
    /// `hyper_read_projected`.
    #[allow(clippy::too_many_arguments)]
    pub fn hc8_read_up_tensor(
        &mut self,
        w: &GpuTensor,
        m: usize,
        k: usize,
        low: &GpuTensor,
        normalized: &GpuTensor,
        mixed: &GpuTensor,
        rows: usize,
        hidden: usize,
    ) -> HipResult<String> {
        let view = self.hc8_weight_view(w, m, k)?;
        hc8::hc8_read_up(self, &view, low, normalized, mixed, rows, hidden)
    }

    /// Exact expansion of an HC8 weight into `dst` (BF16 or F16, `m * k` elements, row-major).
    /// `dst` must not alias `w`.
    pub fn hc8_dequant(&mut self, w: &GpuTensor, m: usize, k: usize, dst: &GpuTensor) -> HipResult<()> {
        self.hc8_check(w, m, k)?;
        let entry = match dst.dtype {
            DType::BF16 => "hc8_dequant_bf16",
            DType::F16 => "hc8_dequant_f16",
            other => return Err(err(format!("dequant target is {other:?}, not BF16 / F16"))),
        };
        if dst.buf.size() < m * k * 2 {
            return Err(err(format!("dequant target holds {} bytes, needs {}", dst.buf.size(), m * k * 2)));
        }
        self.hc8_dequant_raw(w.buf.as_ptr(), m, k, dst.buf.as_ptr(), entry)
    }

    /// [`Gpu::hc8_dequant`] to a raw F16 destination (the shadow cache's allocation).
    pub(crate) fn hc8_dequant_f16_ptr(&mut self, w: &GpuTensor, m: usize, k: usize, dst: *mut c_void) -> HipResult<()> {
        self.hc8_check(w, m, k)?;
        self.hc8_dequant_raw(w.buf.as_ptr(), m, k, dst, "hc8_dequant_f16")
    }

    /// [`Gpu::hc8_dequant_f16_ptr`] over a raw payload pointer (the borrowed-shadow rebuild keeps
    /// only addresses). The caller guarantees `src` holds the `m * k + 4 * m` byte payload.
    pub(crate) fn hc8_dequant_f16_at(&mut self, src: *mut c_void, m: usize, k: usize, dst: *mut c_void) -> HipResult<()> {
        if hc8_payload_bytes(m, k).is_none() {
            return Err(err(format!("[{m}, {k}] is not an HC8 shape")));
        }
        self.hc8_dequant_raw(src, m, k, dst, "hc8_dequant_f16")
    }

    fn hc8_dequant_raw(&mut self, src: *mut c_void, m: usize, k: usize, dst: *mut c_void, entry: &str) -> HipResult<()> {
        let n = m * k;
        let n_u32 = u32::try_from(n).ok().filter(|&n| n < (1 << 31)).ok_or_else(|| err("weight too large"))?;
        hc8::verify_image().map_err(err)?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(hc8::MODULE, hc8::IMAGE, entry)?;
        // Kernarg (32 B): q u64@0, s u64@8 (right after q), out u64@16, m u32@24, k u32@28.
        let mut args = [0u8; 32];
        let qp = src as u64;
        // SAFETY: the callers proved the buffer holds `m * k + 4 * m` bytes.
        let sp = unsafe { (src as *mut u8).add(m * k) } as u64;
        args[0..8].copy_from_slice(&qp.to_le_bytes());
        args[8..16].copy_from_slice(&sp.to_le_bytes());
        args[16..24].copy_from_slice(&(dst as u64).to_le_bytes());
        args[24..28].copy_from_slice(&(m as u32).to_le_bytes());
        args[28..32].copy_from_slice(&(k as u32).to_le_bytes());
        self.launch_blob_recorded(
            entry,
            [n_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }

    /// Per-GPU reused BF16 scratch of `m * k` elements (`slot` 0 = HC down, 1 = HC up), filled with the
    /// exact BF16 expansion of `w`. The contents are valid until the next call for the same slot;
    /// caches keyed on the scratch pointer are NOT valid (the F16 shadow keys on the E4M3 weight).
    /// Allocated once per slot and never regrown for the HC shapes; a different size errors.
    pub fn hc8_bf16_scratch(&mut self, slot: usize, w: &GpuTensor, m: usize, k: usize) -> HipResult<GpuTensor> {
        self.hc8_check(w, m, k)?;
        if slot >= self.scratch.hc8_bf16.len() {
            return Err(err(format!("scratch slot {slot} out of range")));
        }
        let bytes = m * k * 2;
        if self.scratch.hc8_bf16[slot].is_none() {
            if self.replay.is_recording() || self.graphs.capture_mode {
                return Err(err("the BF16 scratch cannot be first allocated during recording / capture"));
            }
            self.invalidate_for_scratch_growth();
            let scratch = self.alloc_tensor(&[m * k], DType::BF16)?;
            self.scratch.hc8_bf16[slot] = Some(scratch);
        }
        let scratch = self.scratch.hc8_bf16[slot].as_ref().expect("allocated above");
        if scratch.buf.size() != bytes {
            return Err(err(format!(
                "scratch slot {slot} is {} bytes, [{m}, {k}] needs {bytes}",
                scratch.buf.size()
            )));
        }
        let view = GpuTensor {
            buf: scratch.buf.byte_view(0, bytes),
            shape: vec![m, k],
            dtype: DType::BF16,
        };
        self.hc8_dequant(w, m, k, &view)?;
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_bytes_are_q_then_scales() {
        assert_eq!(hc8_payload_bytes(320, 10240), Some(320 * 10240 + 4 * 320));
        assert_eq!(hc8_payload_bytes(10240, 320), Some(10240 * 320 + 4 * 10240));
        assert_eq!(hc8_payload_bytes(0, 320), None);
        assert_eq!(hc8_payload_bytes(20, 320), None);
        assert_eq!(hc8_payload_bytes(32, 100), None);
        assert!(hc8_decode_shape_supported(320, 10240));
        assert!(hc8_decode_shape_supported(10240, 320));
    }

    /// The kernel's closed-form inverse addresses exactly the byte `swizzle_hc8` put each element at.
    #[test]
    fn dequant_offsets_invert_the_swizzle() {
        for (m, k) in [(32usize, 64usize), (48, 96), (64, 320)] {
            let rowmajor: Vec<u8> = (0..m * k).map(|i| (i.wrapping_mul(131) >> 3) as u8).collect();
            let swizzled = hc8::swizzle_hc8(&rowmajor, m, k);
            for r in 0..m {
                for c in 0..k {
                    assert_eq!(swizzled[hc8_swizzled_offset(r, c, k)], rowmajor[r * k + c], "({r}, {c}) of [{m}, {k}]");
                }
            }
        }
    }
}
