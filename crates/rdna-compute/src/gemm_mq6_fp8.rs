// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native-FP8 prefill GEMM for MQ6G256V2 weights on exact gfx1201.
//!
//! The MQ6 codes are decoded in-register into E4M3 WMMA operands inside the
//! GEMM (`kernels/src/gemm_mq6g256v2_wmma_fp8.gfx1201.hip`): no weight
//! conversion pass and no resident FP8 weight copy. Activations come from the
//! existing MQ4v2 FP8 producer ([`Gpu::rotate_x_mq_fp8_gfx12_batched`]) as an
//! [`Mq4v2Fp8Prepared`]. Two arms: `Split` (exact hi/lo E4M3 planes, default)
//! and `Round` (one lossy `E4M3_RNE(q - 32)` plane); output F32 or BF16 bits
//! (RNE, the incumbent's formula), overwrite.
//!
//! Opt-in research route (default off): on Flash-Next gfx1201 it lost to the
//! F16 PM kernel at pp16384 (exact `Split` -6.5%; lossy `Round` +2.7% but
//! failed the FN KLD bar); evidence in `release-0.4.3/fp8-lean/RESULTS.md`.
//! Env (developer): `HIPFIRE_GFX12_MQ6_FP8=1` on (any scope), anything else
//! off; `HIPFIRE_GFX12_MQ6_FP8_ARM` (`split` | `round`);
//! `HIPFIRE_GFX12_MQ6_FP8_DB=1` double-buffered kernel variant.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use crate::scratch::Mq4v2Fp8Prepared;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Weight-plane arm of the MQ6 FP8 GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mq6Fp8Arm {
    /// Exact: `q - 32 = 8*hi + lo` as two E4M3 planes, two WMMAs per tile.
    #[default]
    Split,
    /// Lossy: one `E4M3_RNE(q - 32)` plane, one WMMA per tile.
    Round,
}

/// MQ6G256V2 group bytes (8 B header + 192 B 6-bit payload).
const MQ6_GROUP_BYTES: usize = 200;
/// Weight rows / tokens per block (fixed 128 x 128 tile).
const MQ6_FP8_TILE: usize = 128;

/// Scope decision of the MQ6 FP8 route past the device/shape preconditions:
/// on only when `HIPFIRE_GFX12_MQ6_FP8=1`.
fn mq6_fp8_scope_admits(flag: Option<bool>) -> bool {
    flag == Some(true)
}

impl Gpu {
    /// Move the shared MQ6 F32 rotation scratch (grown to at least `elems`
    /// values, captured graphs invalidated first when it will grow) out of the
    /// scratch state: the FP8 producer and the INT8 route
    /// ([`Gpu::mq6_i8_prepare_x`]) take `&mut self` while they use it. Hand it
    /// back with [`Gpu::put_mq6_x_rot`] whatever the outcome.
    pub(crate) fn take_mq6_x_rot(&mut self, elems: usize) -> HipResult<GpuTensor> {
        if self
            .scratch
            .mq6_fp8_x_rot
            .as_ref()
            .map_or(true, |t| t.buf.size() < elems * 4)
        {
            self.invalidate_for_scratch_growth();
        }
        self.scratch
            .ensure_mq6_fp8_x_rot(&self.hip, self.device_id, elems)?;
        self.scratch
            .mq6_fp8_x_rot
            .take()
            .ok_or_else(|| HipError::new(0, "mq6 x_rot scratch missing"))
    }

    /// Return the scratch taken by [`Gpu::take_mq6_x_rot`].
    pub(crate) fn put_mq6_x_rot(&mut self, x_rot: GpuTensor) {
        self.scratch.mq6_fp8_x_rot = Some(x_rot);
    }

    /// Whether the MQ6 FP8 route admits one launch of `m` weight rows, `k` and
    /// `batch_size` tokens: exact gfx1201, eager (no replay recording / graph
    /// capture), `k % 256 == 0`, non-empty dims, and the opt-in
    /// `HIPFIRE_GFX12_MQ6_FP8=1` (any scope).
    pub fn gemm_mq6g256v2_fp8_applies(&self, m: usize, k: usize, batch_size: usize) -> bool {
        self.arch == "gfx1201"
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && k > 0
            && k % 256 == 0
            && m > 0
            && batch_size > 0
            && mq6_fp8_scope_admits(self.flags.gfx12_mq6_fp8)
    }

    /// Rotate F32 `x` (`[batch_size x k]`) with the MQ FWHT (same signs as
    /// [`Gpu::rotate_x_mq_batched_f16`]) into the FP8 prepared planes (E4M3
    /// bytes, per-row pow2 scale, per-128-half decoded sums). The producer's
    /// F32 `x_out` operand is an internal grow-on-demand scratch.
    ///
    /// Pointers in the result belong to this `Gpu` and stay valid only until
    /// the next FP8 producer call.
    pub fn mq6_fp8_prepare_x(
        &mut self,
        x: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<Mq4v2Fp8Prepared> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "mq6_fp8_prepare_x: eager-only"));
        }
        if x.dtype != DType::F32 {
            return Err(HipError::new(0, "mq6_fp8_prepare_x: x must be F32"));
        }
        if self.arch != "gfx1201" || k == 0 || k % 256 != 0 || batch_size == 0 {
            return Err(HipError::new(
                0,
                "mq6_fp8_prepare_x: exact gfx1201, K%256==0, N>0 required",
            ));
        }
        let elems = batch_size
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "mq6_fp8_prepare_x: batch*k overflow"))?;
        if x.numel() < elems {
            return Err(HipError::new(0, "mq6_fp8_prepare_x: undersized x"));
        }
        // The in-register producer (K <= 17408, the default) never writes its
        // F32 `x_out` operand, only bounds-checks it: hand it `x` itself so the
        // prefill route allocates no extra VRAM. Otherwise it writes the rotated
        // row, which must not clobber `x`: use the grow-on-demand scratch.
        if crate::gemv::fp8_prod_inreg(k) {
            return self.rotate_x_mq_fp8_gfx12_batched(x, None, None, x, k, batch_size);
        }
        let x_out = self.take_mq6_x_rot(elems)?;
        let prepared = self.rotate_x_mq_fp8_gfx12_batched(x, None, None, &x_out, k, batch_size);
        self.put_mq6_x_rot(x_out);
        prepared
    }

    /// Overwrite `y` (`[batch_size x m]`, F32, or BF16 bits when
    /// `y.dtype == DType::BF16`) with `W . X` for MQ6G256V2 weights `a_raw`
    /// (`m` rows x `k/256` groups x 200 B) and the FP8-prepared activations.
    /// Arm from `HIPFIRE_GFX12_MQ6_FP8_ARM` (`Split` default). Exact gfx1201,
    /// eager only; validates `prepared` (`n`, `k`, `scale_mode == 1`).
    pub fn gemm_mq6g256v2_fp8_prepared(
        &mut self,
        a_raw: &GpuTensor,
        prepared: &Mq4v2Fp8Prepared,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" || self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_fp8: exact gfx1201 eager-only",
            ));
        }
        if m == 0 || batch_size == 0 || k == 0 || k % 256 != 0 {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_fp8: need M>0, N>0, K>0, K%256==0",
            ));
        }
        if prepared.n != batch_size || prepared.k != k || prepared.scale_mode != 1 {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_fp8: prepared (n,k,scale_mode) mismatch",
            ));
        }
        let bf16 = match y.dtype {
            DType::F32 => false,
            DType::BF16 => true,
            _ => {
                return Err(HipError::new(
                    0,
                    "gemm_mq6g256v2_fp8: y must be F32 or BF16",
                ))
            }
        };
        let weight_bytes = m
            .checked_mul(k / 256)
            .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_fp8: weight bytes overflow"))?;
        let y_elems = batch_size
            .checked_mul(m)
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_fp8: batch*m overflow"))?;
        let out_bytes = y_elems * if bf16 { 2 } else { 4 };
        if a_raw.buf.size() < weight_bytes || y.buf.size() < out_bytes {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_fp8: undersized weight or output tensor",
            ));
        }

        let (func_name, ksrc): (&'static str, &'static str) = match (
            self.flags.gfx12_mq6_fp8_arm,
            bf16,
            self.flags.gfx12_mq6_fp8_db,
        ) {
            (Mq6Fp8Arm::Split, false, false) => (
                "gemm_mq6g256v2_wmma_fp8_split_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Split, true, false) => (
                "gemm_mq6g256v2_wmma_fp8_split_bf16_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_BF16_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Round, false, false) => (
                "gemm_mq6g256v2_wmma_fp8_round_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Round, true, false) => (
                "gemm_mq6g256v2_wmma_fp8_round_bf16_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_BF16_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Split, false, true) => (
                "gemm_mq6g256v2_wmma_fp8_split_db_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_DB_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Split, true, true) => (
                "gemm_mq6g256v2_wmma_fp8_split_bf16_db_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_BF16_DB_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Round, false, true) => (
                "gemm_mq6g256v2_wmma_fp8_round_db_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_DB_GFX1201_SRC,
            ),
            (Mq6Fp8Arm::Round, true, true) => (
                "gemm_mq6g256v2_wmma_fp8_round_bf16_db_gfx1201",
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_BF16_DB_GFX1201_SRC,
            ),
        };
        self.ensure_kernel(func_name, ksrc, func_name)?;
        let mut aw = a_raw.buf.as_ptr();
        let mut xf = prepared.x_fp8;
        let mut hs = prepared.half_sums;
        let mut rsc = prepared.row_scales;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut aw as *mut _ as *mut c_void,
            &mut xf as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut rsc as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(MQ6_FP8_TILE);
        let batch_tiles = batch_size.div_ceil(MQ6_FP8_TILE);
        let bytes = weight_bytes + batch_size * k + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aw);
                b.push_ptr(xf);
                b.push_ptr(hs);
                b.push_ptr(rsc);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8_e4m3::e4m3fn_to_f32;

    /// The two 8-entry `v_perm` tables of the kernel (little-endian bytes of two
    /// dwords each). One definition for the host tests; the kernel source is
    /// pinned to these literals below.
    const CL0: u32 = 0x4440_3800;
    const CL1: u32 = 0x4E4C_4A48;
    const CH0: u32 = 0xD0D8_DCE0;
    const CH1: u32 = 0x5C58_5000;

    fn table_byte(t0: u32, t1: u32, i: usize) -> u8 {
        let w = if i < 4 { t0 } else { t1 };
        (w >> (8 * (i & 3))) as u8
    }

    fn dec(b: u8) -> f32 {
        e4m3fn_to_f32(b).expect("table byte is a finite E4M3")
    }

    /// `v_perm_b32(s0, s1, sel)` for selector bytes 0..7: byte `i` of the result
    /// = {s1 bytes 0-3, s0 bytes 4-7}[sel_i].
    fn perm(s0: u32, s1: u32, sel: u32) -> u32 {
        let mut out = 0u32;
        for i in 0..4 {
            let s = ((sel >> (8 * i)) & 0xFF) as usize;
            assert!(s < 8, "selector byte {s} outside the table range");
            let b = if s < 4 {
                (s1 >> (8 * s)) as u8
            } else {
                (s0 >> (8 * (s - 4))) as u8
            };
            out |= (b as u32) << (8 * i);
        }
        out
    }

    fn sp4(t: u32) -> u32 {
        (t & 0x3F) | ((t << 2) & 0x3F00) | ((t << 4) & 0x3F_0000) | ((t << 6) & 0x3F00_0000)
    }

    /// Pack 16 six-bit codes (code i = bits [6i, 6i+6)) into three dwords.
    fn pack16(codes: &[u32; 16]) -> (u32, u32, u32) {
        let mut v = 0u128;
        for (i, &c) in codes.iter().enumerate() {
            assert!(c < 64);
            v |= (c as u128) << (6 * i);
        }
        (v as u32, (v >> 32) as u32, (v >> 64) as u32)
    }

    /// Host model of the kernel's SPLIT decode: per code, `(hi byte, lo byte)`.
    fn split_decode(codes: &[u32; 16]) -> [(u8, u8); 16] {
        let (w0, w1, w2) = pack16(codes);
        let f = [w0, (w0 >> 24) | (w1 << 8), (w1 >> 16) | (w2 << 16), w2 >> 8];
        let mut out = [(0u8, 0u8); 16];
        for (j, &fj) in f.iter().enumerate() {
            let b = sp4(fj) & 0x3F3F_3F3F;
            let lo = perm(CL1, CL0, b & 0x0707_0707);
            let hi = perm(CH1, CH0, (b >> 3) & 0x0707_0707);
            for i in 0..4 {
                out[4 * j + i] = ((hi >> (8 * i)) as u8, (lo >> (8 * i)) as u8);
            }
        }
        out
    }

    #[test]
    fn kernel_source_pins_the_table_literals_and_symbols() {
        let src = include_str!("../../../kernels/src/gemm_mq6g256v2_wmma_fp8.gfx1201.hip");
        for (name, v) in [("CL0", CL0), ("CL1", CL1), ("CH0", CH0), ("CH1", CH1)] {
            let line = format!("#define HIPFIRE_MQ6_FP8_{name} 0x{v:08X}u");
            assert!(src.contains(&line), "kernel source lacks `{line}`");
        }
        // Only the part of this file above the test module counts as the launcher.
        let launcher = include_str!("gemm_mq6_fp8.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for (konst, sym, split, out, db) in [
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_split_gfx1201",
                1,
                0,
                false,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_BF16_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_split_bf16_gfx1201",
                1,
                1,
                false,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_round_gfx1201",
                0,
                0,
                false,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_BF16_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_round_bf16_gfx1201",
                0,
                1,
                false,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_DB_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_split_db_gfx1201",
                1,
                0,
                true,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_SPLIT_BF16_DB_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_split_bf16_db_gfx1201",
                1,
                1,
                true,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_DB_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_round_db_gfx1201",
                0,
                0,
                true,
            ),
            (
                kernels::GEMM_MQ6G256V2_WMMA_FP8_ROUND_BF16_DB_GFX1201_SRC,
                "gemm_mq6g256v2_wmma_fp8_round_bf16_db_gfx1201",
                0,
                1,
                true,
            ),
        ] {
            assert!(
                konst.contains(src),
                "{sym}: source must embed the kernel file"
            );
            // The macro prefix prepended to the kernel file selects the variant;
            // only the DB consts define HIPFIRE_MQ6_FP8_DB (the file defaults it to 0).
            let prefix = konst.strip_suffix(src).expect("prefix + kernel file");
            assert!(prefix.contains(&format!("#define HIPFIRE_MQ6_FP8_SPLIT {split}\n")));
            assert!(prefix.contains(&format!("#define HIPFIRE_MQ6_FP8_OUT_BF16 {out}\n")));
            assert!(prefix.contains(&format!("#define HIPFIRE_MQ6_FP8_KERNEL {sym}\n")));
            assert_eq!(
                prefix.contains("#define HIPFIRE_MQ6_FP8_DB 1\n"),
                db,
                "{sym}: DB define"
            );
            assert!(
                launcher.contains(&format!("\"{sym}\"")),
                "launcher does not select {sym}"
            );
        }
    }

    #[test]
    fn split_tables_sum_to_q_minus_32_for_every_code() {
        for q in 0u32..64 {
            let lo = (q & 7) as usize;
            let hi = (q >> 3) as usize;
            let lo_v = dec(table_byte(CL0, CL1, lo));
            let hi_v = dec(table_byte(CH0, CH1, hi));
            assert_eq!(lo_v, lo as f32, "lo plane q={q}");
            assert_eq!(hi_v, 8.0 * (hi as f32 - 4.0), "hi plane q={q}");
            assert_eq!(hi_v + lo_v, q as f32 - 32.0, "q={q}");
        }
    }

    #[test]
    fn split_decode_model_is_exact_for_every_code_at_every_position() {
        // Every code value at every one of the 16 positions (and so every
        // dword-straddling field), against pseudo-random neighbours.
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 26) & 63
        };
        for pos in 0..16 {
            for q in 0u32..64 {
                let mut codes = [0u32; 16];
                for c in codes.iter_mut() {
                    *c = next();
                }
                codes[pos] = q;
                let dec_all = split_decode(&codes);
                for (i, &(hi_b, lo_b)) in dec_all.iter().enumerate() {
                    let v = dec(hi_b) + dec(lo_b);
                    assert_eq!(v, codes[i] as f32 - 32.0, "pos={pos} q={q} i={i}");
                }
            }
        }
    }

    /// Nearest representable finite E4M3 value to `v`, ties to the even code
    /// byte (RNE: an even mantissa LSB is the even neighbour).
    fn rne_e4m3_reference(v: f32) -> f32 {
        let sign = v.is_sign_negative();
        let mag = v.abs();
        let mut best: Option<(f32, u8)> = None;
        for b in 0u8..=0x7E {
            let cand = dec(b);
            let better = match best {
                None => true,
                Some((bv, bb)) => {
                    let (d, bd) = ((cand - mag).abs(), (bv - mag).abs());
                    d < bd || (d == bd && b % 2 == 0 && bb % 2 == 1)
                }
            };
            if better {
                best = Some((cand, b));
            }
        }
        let m = best.unwrap().0;
        if sign {
            -m
        } else {
            m
        }
    }

    #[test]
    fn round_arm_reference_matches_nearest_even_e4m3_of_q_minus_32() {
        for q in 0u32..64 {
            let v = q as f32 - 32.0;
            let r = rne_e4m3_reference(v);
            if v.abs() <= 16.0 {
                assert_eq!(r, v, "|q-32| <= 16 is exact in E4M3 (q={q})");
            } else {
                // Spacing is 2 on [16, 32): even integers are exact, odd ones
                // tie and round to the even-mantissa neighbour.
                let a = v.abs() as i32;
                let expect = if a % 2 == 0 {
                    a
                } else if (a / 2) % 2 == 0 {
                    // a = 2n+1, neighbours 2n (mantissa n-8) and 2n+2.
                    a - 1
                } else {
                    a + 1
                };
                assert_eq!(r.abs() as i32, expect, "q={q}");
                assert_eq!(r.is_sign_negative(), v < 0.0, "sign q={q}");
            }
        }
    }

    #[test]
    fn scope_admission_is_opt_in() {
        assert!(!mq6_fp8_scope_admits(None));
        assert!(!mq6_fp8_scope_admits(Some(false)));
        assert!(mq6_fp8_scope_admits(Some(true)));
        assert_eq!(Mq6Fp8Arm::default(), Mq6Fp8Arm::Split);
    }
}
