// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF-native weight route glue (`HIPFIRE_GGUF_NATIVE=1`): the persistent
//! scratch the ggml kernels share, the weight-buffer layout contract and the
//! decode / prefill entry points built from the shim launchers
//! ([`crate::ggml_mmvq`], [`crate::ggml_convert`]).
//!
//! Numeric contract of the decode route: `ggml.a8q81.<type>` (the activation
//! row is quantised per 32 to `q8_1`, int32 block dots, f32 scale fold, f32
//! output). Prefill dequantises each layer's weights into a persistent
//! scratch (F16 for the shipped route, BF16-rounded F32 for the codec-only
//! quality tier) and runs the engine's ordinary GEMMs on it.
//!
//! **Weight buffer layout.** A `DType::Ggml*` weight is the tensor's ggml
//! block bytes verbatim (`m` rows of whole blocks), optionally followed by
//! a *K-chunk map*: the `out_proj` weight of a checkpoint whose value heads
//! are stored in a different order than the engine's keeps its bytes (a row
//! permutation is a memcpy, a column permutation of 256-element blocks is
//! not) and records the head permutation of its K axis instead. The tail
//! starts at the next 256-byte boundary after the weight bytes and holds
//! `k/32` int32 q8_1-block source indices (the activation gather of the
//! decode producer) followed by `k/128` int32 destination head indices (the
//! column map of the prefill dequantisation).

use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};

/// Elements per head chunk of the K-chunk map (the Qwen3.5 value head dim).
pub const GGML_KMAP_CHUNK: usize = 128;
/// Tail alignment inside a weight buffer.
const TAIL_ALIGN: usize = 256;
/// Activation scratch for the q8_1 producer: K up to 29127 (one row).
const Q81_SCRATCH_BYTES: usize = 1 << 20;

/// Persistent scratch of the ggml route, held in [`crate::scratch::ScratchState`].
#[derive(Default)]
pub struct GgmlScratch {
    /// One q8_1-quantised activation row (decode producer output).
    pub q81: Option<GpuTensor>,
    /// Prefill dequantisation region (one layer's worth of weights).
    pub mat: Option<GpuTensor>,
    /// Bytes of the `mat` region (sized by the loader for the largest weight).
    pub mat_bytes: usize,
    /// Activation copy for the A8 emulation (`n * k` F32), grown on demand.
    pub a8: Option<GpuTensor>,
    pub a8_bytes: usize,
    /// Codec-only activation split (`hi`, `lo` F32 `n * k` each, then the
    /// second GEMM's `n * m` F32 output), grown on demand.
    pub split: Option<GpuTensor>,
    pub split_bytes: usize,
    /// Prefill numeric mode, set by the loader from the environment.
    pub prefill: GgmlPrefill,
}

/// How prefill treats a ggml weight (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GgmlPrefill {
    /// Codec-only quality tier: weights dequantised to BF16-rounded values,
    /// multiplied on the engine's F16 WMMA GEMM against F32 activations
    /// carried as an exact hi (BF16-rounded) + lo (residual) pair, so the
    /// activation keeps about 20 mantissa bits and the accumulation is F32.
    pub codec_only: bool,
    /// Emulate the `ggml.a8q81` activation quantisation in prefill: the GEMM
    /// input is rounded through q8_1 (per 32, f16 scale) before the multiply.
    pub a8_emulation: bool,
}

/// Bytes of `m` rows of `k` elements of a ggml dtype (`None` for another
/// dtype, `k` not a whole number of blocks, or overflow).
pub fn ggml_weight_bytes(dtype: DType, m: usize, k: usize) -> Option<usize> {
    let (qk, bytes) = dtype.ggml_block()?;
    if k == 0 || k % qk != 0 {
        return None;
    }
    (k / qk).checked_mul(bytes)?.checked_mul(m)
}

/// Byte offset where a K-chunk map starts in a weight buffer of `m` x `k`.
pub fn ggml_kmap_offset(dtype: DType, m: usize, k: usize) -> Option<usize> {
    let w = ggml_weight_bytes(dtype, m, k)?;
    w.checked_add(TAIL_ALIGN - 1).map(|v| v & !(TAIL_ALIGN - 1))
}

/// Bytes of the K-chunk map of a `k`-wide weight.
pub fn ggml_kmap_bytes(k: usize) -> usize {
    4 * (k / 32 + k / GGML_KMAP_CHUNK)
}

/// Total buffer bytes of a weight with a K-chunk map.
pub fn ggml_weight_buffer_bytes(
    dtype: DType,
    m: usize,
    k: usize,
    with_kmap: bool,
) -> Option<usize> {
    if with_kmap {
        ggml_kmap_offset(dtype, m, k)?.checked_add(ggml_kmap_bytes(k))
    } else {
        ggml_weight_bytes(dtype, m, k)
    }
}

/// Whether a buffer of `buf_bytes` carries the K-chunk map of an `m` x `k`
/// weight (the size is the only discriminator: a plain weight is exactly
/// `ggml_weight_bytes`).
pub fn ggml_has_kmap(buf_bytes: usize, dtype: DType, m: usize, k: usize) -> bool {
    k % GGML_KMAP_CHUNK == 0
        && ggml_weight_buffer_bytes(dtype, m, k, true).is_some_and(|n| n == buf_bytes)
}

/// Head permutation of a value-head axis that the exporter re-ordered from
/// grouped-by-key-head (`h = kh * r + j`) to tiled (`g = j * nk + kh`):
/// entry `g` is the grouped head `h` stored at tiled position `g`.
pub fn ggml_tiled_to_grouped_heads(nk: usize, r: usize) -> Vec<i32> {
    (0..nk * r)
        .map(|g| ((g % nk) * r + g / nk) as i32)
        .collect()
}

/// The K-chunk map bytes (`k/32` q8_1 source blocks then `k/128` head
/// destinations) for a weight whose K axis is `heads` chunks of 128 stored in
/// the order `head_of_stored[g]` (see [`ggml_tiled_to_grouped_heads`]).
pub fn ggml_build_kmap(head_of_stored: &[i32]) -> Vec<u8> {
    let per_head = GGML_KMAP_CHUNK / 32;
    let mut out = Vec::with_capacity(4 * head_of_stored.len() * (per_head + 1));
    for &h in head_of_stored {
        for j in 0..per_head as i32 {
            out.extend_from_slice(&(h * per_head as i32 + j).to_le_bytes());
        }
    }
    for &h in head_of_stored {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn raw_view(buf: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: buf.buf.byte_view(offset, len),
        shape: vec![len],
        dtype: DType::Raw,
    }
}

/// Non-owning views of a weight's K-chunk map: `(q8_1 source blocks,
/// destination head chunks)`.
pub fn ggml_kmap_views(
    w: &GpuTensor,
    dtype: DType,
    m: usize,
    k: usize,
) -> Option<(GpuTensor, GpuTensor)> {
    if !ggml_has_kmap(w.buf.size(), dtype, m, k) {
        return None;
    }
    let off = ggml_kmap_offset(dtype, m, k)?;
    let blk = 4 * (k / 32);
    Some((
        raw_view(w, off, blk),
        raw_view(w, off + blk, 4 * (k / GGML_KMAP_CHUNK)),
    ))
}

impl Gpu {
    /// Allocate the persistent q8_1 activation scratch (idempotent). Called by
    /// the loader so no allocation happens inside a decode step or a capture.
    pub fn ggml_ensure_scratch(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.scratch.ggml.q81.is_none() {
            let buf = self.hip.malloc(Q81_SCRATCH_BYTES)?;
            self.scratch.ggml.q81 = Some(GpuTensor {
                buf,
                shape: vec![Q81_SCRATCH_BYTES],
                dtype: DType::Raw,
            });
        }
        Ok(())
    }

    /// Allocate (or grow) the prefill dequantisation region to hold `bytes`.
    /// The loader sizes it for the largest layer before KV admission so the
    /// region exists before the first prefill.
    pub fn ggml_ensure_materialize(&mut self, bytes: usize) -> HipResult<()> {
        self.bind_thread()?;
        if self.scratch.ggml.mat_bytes >= bytes {
            return Ok(());
        }
        self.invalidate_for_scratch_growth();
        if let Some(old) = self.scratch.ggml.mat.take() {
            self.hip.device_synchronize()?;
            let _ = self.hip.free(old.buf);
        }
        let buf = self.hip.malloc(bytes)?;
        self.scratch.ggml.mat = Some(GpuTensor {
            buf,
            shape: vec![bytes],
            dtype: DType::Raw,
        });
        self.scratch.ggml.mat_bytes = bytes;
        Ok(())
    }

    /// Set the prefill numeric mode (loader, from `HIPFIRE_GGUF_CODEC_ONLY` /
    /// `HIPFIRE_GGUF_A8_EMU`).
    pub fn ggml_set_prefill_mode(&mut self, mode: GgmlPrefill) {
        self.scratch.ggml.prefill = mode;
    }

    /// Prefill: `y[n, m] = x[n, k] · W^T` for a ggml weight `w` (layout above).
    /// The weight is dequantised into the persistent region (F16, or BF16-rounded
    /// F32 in the codec-only tier; the K-chunk map restores the engine's K
    /// order) and multiplied by the engine's GEMM. With the A8 emulation the
    /// activation is first copied and rounded through q8_1 (the copy keeps a
    /// shared producer buffer intact).
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_prefill_gemm(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        m: usize,
        k: usize,
        x: &GpuTensor,
        y: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        let mode = self.scratch.ggml.prefill;
        // Codec-only holds the F32 dequantisation (4 B) and its F16 copy (2 B).
        let elem = if mode.codec_only { 6 } else { 2 };
        let need = m
            .checked_mul(k)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| err("ggml prefill: weight extent overflows"))?;
        let mat = self.ggml_materialize_view()?;
        if need > mat.buf.size() {
            return Err(err(format!(
                "ggml prefill: weight {m}x{k} needs {need} dequantisation bytes, the region holds {}",
                mat.buf.size()
            )));
        }
        if n == 0 || m == 0 {
            return Ok(());
        }
        let input = if mode.a8_emulation {
            let bytes = n
                .checked_mul(k)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| err("ggml prefill: activation extent overflows"))?;
            if x.buf.size() < bytes {
                return Err(err("ggml prefill: activation buffer is smaller than n x k"));
            }
            if self.scratch.ggml.a8_bytes < bytes {
                self.invalidate_for_scratch_growth();
                if let Some(old) = self.scratch.ggml.a8.take() {
                    self.hip.device_synchronize()?;
                    let _ = self.hip.free(old.buf);
                }
                let buf = self.hip.malloc(bytes)?;
                self.scratch.ggml.a8 = Some(GpuTensor {
                    buf,
                    shape: vec![n * k],
                    dtype: DType::F32,
                });
                self.scratch.ggml.a8_bytes = bytes;
            }
            let tmp = self
                .scratch
                .ggml
                .a8
                .as_ref()
                .expect("allocated")
                .shallow_clone();
            self.copy_d2d(x, &tmp, bytes)?;
            self.ggml_q81_roundtrip(&tmp, n, k)?;
            tmp
        } else {
            x.shallow_clone()
        };
        if mode.codec_only {
            // Weights: dequantise to BF16-rounded F32, then copy into F16 (a
            // BF16-rounded value is exact in F16 over the weights' range).
            let wf = GpuTensor {
                buf: mat.buf.byte_view(0, m * k * 4),
                shape: vec![m * k],
                dtype: DType::F32,
            };
            let wh = GpuTensor {
                buf: mat.buf.byte_view(m * k * 4, m * k * 2),
                shape: vec![m, k],
                dtype: DType::F16,
            };
            self.ggml_dequant_weight(dtype, w, m, k, &wf, Some(true))?;
            self.cast_f32_to_f16(&wf, &wh)?;
            // Activations: x = hi + lo, hi = bf16(x), lo = x - hi (exact in
            // F32). Both parts go through the F16 WMMA GEMM (F32 accumulate)
            // and the two products are summed in F32.
            let (nk, nm) = (n * k, n * m);
            let bytes = (2 * nk + nm) * 4;
            if self.scratch.ggml.split_bytes < bytes {
                self.invalidate_for_scratch_growth();
                if let Some(old) = self.scratch.ggml.split.take() {
                    self.hip.device_synchronize()?;
                    let _ = self.hip.free(old.buf);
                }
                let buf = self.hip.malloc(bytes)?;
                self.scratch.ggml.split = Some(GpuTensor {
                    buf,
                    shape: vec![bytes / 4],
                    dtype: DType::F32,
                });
                self.scratch.ggml.split_bytes = bytes;
            }
            let sp = self
                .scratch
                .ggml
                .split
                .as_ref()
                .expect("allocated")
                .shallow_clone();
            let hi = GpuTensor {
                buf: sp.buf.byte_view(0, nk * 4),
                shape: vec![nk],
                dtype: DType::F32,
            };
            let lo = GpuTensor {
                buf: sp.buf.byte_view(nk * 4, nk * 4),
                shape: vec![nk],
                dtype: DType::F32,
            };
            let y2 = GpuTensor {
                buf: sp.buf.byte_view(2 * nk * 4, nm * 4),
                shape: vec![nm],
                dtype: DType::F32,
            };
            self.copy_d2d(&input, &hi, nk * 4)?;
            self.bf16_round_trip_f32(&hi)?;
            self.copy_d2d(&hi, &lo, nk * 4)?;
            crate::tensor_ops::scale_f32(
                self,
                &crate::tensor_ops::ScaleF32 {
                    values: &lo,
                    scale: -1.0,
                },
            )?;
            self.add_inplace_f32_raw(lo.buf.as_ptr(), input.buf.as_ptr(), nk)?;
            self.gemm_f16_wmma_mb8(&wh, &hi, y, m, k, n)?;
            self.gemm_f16_wmma_mb8(&wh, &lo, &y2, m, k, n)?;
            self.add_inplace_f32_raw(y.buf.as_ptr(), y2.buf.as_ptr(), nm)
        } else {
            let wv = GpuTensor {
                buf: mat.buf.byte_view(0, need),
                shape: vec![m, k],
                dtype: DType::F16,
            };
            self.ggml_dequant_weight(dtype, w, m, k, &wv, None)?;
            self.gemm_f16_wmma_mb8(&wv, &input, y, m, k, n)
        }
    }

    /// Release the ggml scratch (model unload).
    pub fn ggml_free_scratch(&mut self) {
        for t in [
            self.scratch.ggml.q81.take(),
            self.scratch.ggml.mat.take(),
            self.scratch.ggml.a8.take(),
            self.scratch.ggml.split.take(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = self.hip.device_synchronize();
            let _ = self.hip.free(t.buf);
        }
        self.scratch.ggml.mat_bytes = 0;
        self.scratch.ggml.a8_bytes = 0;
        self.scratch.ggml.split_bytes = 0;
    }

    /// Non-owning view of the q8_1 scratch.
    fn ggml_q81_view(&self, bytes: usize) -> HipResult<GpuTensor> {
        let q = self.scratch.ggml.q81.as_ref().ok_or_else(|| {
            err("ggml route: q8_1 scratch was not allocated (ggml_ensure_scratch)")
        })?;
        if bytes > q.buf.size() {
            return Err(err(format!(
                "ggml route: activation row needs {bytes} q8_1 bytes, scratch holds {}",
                q.buf.size()
            )));
        }
        Ok(q.shallow_clone())
    }

    /// Non-owning view of the prefill dequantisation region.
    pub fn ggml_materialize_view(&self) -> HipResult<GpuTensor> {
        self.scratch
            .ggml
            .mat
            .as_ref()
            .map(GpuTensor::shallow_clone)
            .ok_or_else(|| {
                err("ggml route: prefill dequantisation scratch was not allocated (loader sizes it)")
            })
    }

    /// Quantise one activation row `x` (F32 `[k]`) to q8_1 in the shared
    /// scratch and return a view of it. `kmap` (a weight's K-chunk map
    /// block-source view) gathers the 32-blocks in the weight's K order.
    pub fn ggml_quantize_activation(
        &mut self,
        x: &GpuTensor,
        k: usize,
        kmap: Option<&GpuTensor>,
    ) -> HipResult<GpuTensor> {
        if k % 32 != 0 || k == 0 {
            return Err(err(format!(
                "ggml route: activation length {k} is not a multiple of 32"
            )));
        }
        if x.buf.size() < k * 4 {
            return Err(err(format!(
                "ggml route: activation buffer holds {} bytes, {} needed",
                x.buf.size(),
                k * 4
            )));
        }
        let q81 = self.ggml_q81_view(Gpu::ggml_q81_row_bytes(k))?;
        match kmap {
            Some(map) => self.ggml_q8_1_quantize_gather(x, &q81, map, k)?,
            None => self.ggml_q8_1_quantize(x, &q81, k, 1)?,
        }
        Ok(q81)
    }

    /// `y = W x` (or `y += W x`) for a ggml-format weight `w` (`m` x `k`,
    /// layout above) and an F32 activation row `x`: q8_1 producer, then the
    /// `ggml_mmvq` export of the weight's type.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_gemv(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        m: usize,
        k: usize,
        x: &GpuTensor,
        y: &GpuTensor,
        accumulate: bool,
    ) -> HipResult<()> {
        let kmap = ggml_kmap_views(w, dtype, m, k);
        let q81 = self.ggml_quantize_activation(x, k, kmap.as_ref().map(|(blk, _)| blk))?;
        self.ggml_mmvq(dtype, w, &q81, y, m, k, accumulate)
    }

    /// The mmvq half of [`Gpu::ggml_gemv`] for an activation row already
    /// quantised by [`Gpu::ggml_quantize_activation`] (projections that share
    /// one normalised input share one producer launch). A weight with a
    /// K-chunk map must be fed a gathered row, never a shared one.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_gemv_quantized(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        m: usize,
        k: usize,
        q81: &GpuTensor,
        y: &GpuTensor,
        accumulate: bool,
    ) -> HipResult<()> {
        if ggml_has_kmap(w.buf.size(), dtype, m, k) {
            return Err(err(
                "ggml route: a weight with a K-chunk map needs its own gathered activation row",
            ));
        }
        self.ggml_mmvq(dtype, w, q81, y, m, k, accumulate)
    }

    /// Prefill: dequantise a ggml weight (`m` x `k`) into `out` as F16
    /// (`bf16_round == None`) or as BF16-rounded F32 (`Some(true)`); the
    /// weight's K-chunk map (if any) restores the engine's K order.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_dequant_weight(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        m: usize,
        k: usize,
        out: &GpuTensor,
        to_f32_bf16_round: Option<bool>,
    ) -> HipResult<()> {
        let colmap = ggml_kmap_views(w, dtype, m, k).map(|(_, head)| head);
        match to_f32_bf16_round {
            None => self.ggml_dequant_f16(dtype, w, out, m, k, colmap.as_ref()),
            Some(round) => self.ggml_dequant_f32(dtype, w, out, m, k, round, colmap.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmap_layout_is_stable() {
        // out_proj of the 27B: m = 5120, k = 6144 (48 heads of 128).
        let (m, k) = (5120usize, 6144usize);
        for (dt, row) in [
            (DType::GgmlIQ4XS, 136 * (k / 256)),
            (DType::GgmlQ6K, 210 * (k / 256)),
            (DType::GgmlQ8_0, 34 * (k / 32)),
        ] {
            assert_eq!(ggml_weight_bytes(dt, m, k), Some(m * row));
            let off = ggml_kmap_offset(dt, m, k).unwrap();
            assert_eq!(off % 256, 0);
            assert!(off >= m * row && off < m * row + 256);
            let total = ggml_weight_buffer_bytes(dt, m, k, true).unwrap();
            assert_eq!(total, off + 4 * (k / 32 + k / 128));
            assert!(ggml_has_kmap(total, dt, m, k));
            assert!(!ggml_has_kmap(m * row, dt, m, k));
        }
        assert_eq!(ggml_weight_bytes(DType::F16, 1, 256), None);
        assert_eq!(ggml_weight_bytes(DType::GgmlQ4K, 4, 255), None);
    }

    #[test]
    fn tiled_head_map_inverts_the_exporter_reorder() {
        // 16 key heads x 3 value heads per key head (the 27B): grouped head
        // h = kh * 3 + j is stored at tiled position g = j * 16 + kh.
        let (nk, r) = (16usize, 3usize);
        let map = ggml_tiled_to_grouped_heads(nk, r);
        assert_eq!(map.len(), 48);
        let mut seen = vec![false; 48];
        for (g, &h) in map.iter().enumerate() {
            let (kh, j) = (h as usize / r, h as usize % r);
            assert_eq!(g, j * nk + kh);
            assert!(!seen[h as usize]);
            seen[h as usize] = true;
        }
        let bytes = ggml_build_kmap(&map);
        assert_eq!(bytes.len(), ggml_kmap_bytes(48 * 128));
        let word = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        // block part: 4 q8_1 blocks per head, consecutive from the head's first block.
        for g in 0..48 {
            for j in 0..4 {
                assert_eq!(word(4 * g + j), map[g] * 4 + j as i32);
            }
        }
        // head part follows.
        for g in 0..48 {
            assert_eq!(word(48 * 4 + g), map[g]);
        }
    }
}
