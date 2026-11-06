// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab-only: a byte-exact scratch twin of a lane's head state, so the batched
//! head-round oracle can run the serial and the batched round from the same
//! state without touching the lane's own.

use super::*;

impl MtpGpuState {
    /// A new head state on this one's backend and capacity holding a copy of
    /// its context rows (full K/V, raw index keys, pooled keys), selection,
    /// selection length word, wide hidden and marks (`position`, the
    /// lengths, `step_index`). The twin owns fresh device storage (free it
    /// with `free_gpu`); its snapshot arena is idle. The copies are stream
    /// ordered after the source's pending work.
    pub(crate) fn lab_twin(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
    ) -> Result<Self, MtpGpuError> {
        let mut twin = Self::new_with_backend(gpu, config, self.full_capacity, self.backend)?;
        let copied = (|| -> Result<(), MtpGpuError> {
            let needed = self.full_len.max(self.raw_len).max(1);
            twin.ensure_mapped_capacity(gpu, needed)?;
            let rows = |source: &GpuTensor, capacity: usize, len: usize| -> usize {
                len * source.byte_size().checked_div(capacity).unwrap_or(0)
            };
            for (source, target, bytes) in [
                (
                    &self.full_keys,
                    &twin.full_keys,
                    rows(&self.full_keys, self.full_capacity, self.full_len),
                ),
                (
                    &self.full_values,
                    &twin.full_values,
                    rows(&self.full_values, self.full_capacity, self.full_len),
                ),
                (
                    &self.raw_index_keys,
                    &twin.raw_index_keys,
                    rows(&self.raw_index_keys, self.raw_capacity, self.raw_len),
                ),
                (
                    &self.pooled_keys,
                    &twin.pooled_keys,
                    rows(&self.pooled_keys, self.pooled_capacity, self.pooled_len),
                ),
                (
                    &self.selected_indices,
                    &twin.selected_indices,
                    self.selected_indices.byte_size(),
                ),
                (
                    &self.selected_len_out,
                    &twin.selected_len_out,
                    self.selected_len_out.byte_size(),
                ),
                (
                    &self.wide_hidden,
                    &twin.wide_hidden,
                    self.wide_hidden.byte_size(),
                ),
            ] {
                if bytes > 0 {
                    gpu.copy_d2d(source, target, bytes)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = copied {
            let _ = twin.free_gpu(gpu);
            return Err(error);
        }
        twin.position = self.position;
        twin.full_len = self.full_len;
        twin.raw_len = self.raw_len;
        twin.pooled_len = self.pooled_len;
        twin.selected_len = self.selected_len;
        twin.step_index = self.step_index;
        Ok(twin)
    }

    /// Settling test of the draft-history mask (PARITY.md 4.3): overwrite
    /// every mapped context row of the four head arenas (full K, full V, raw
    /// index keys, pooled keys) beyond this state's valid lengths with `byte`
    /// repeated (`0xFF`: F32 NaN; `0x7F`: a huge finite sentinel). A head step
    /// that reads only rows below the lengths is bit-identical with or
    /// without the fill; rows at or above a length are rewritten by the step
    /// that appends them, so a correct run is unchanged.
    pub(crate) fn lab_fill_stale(&self, gpu: &Gpu, byte: u8) -> Result<(), MtpGpuError> {
        let mapped = self.mapped_tokens;
        let pooled_mapped = mapped.div_ceil(self.compress).min(self.pooled_capacity);
        for (tensor, capacity, len, rows_mapped) in [
            (
                &self.full_keys,
                self.full_capacity,
                self.full_len,
                mapped.min(self.full_capacity),
            ),
            (
                &self.full_values,
                self.full_capacity,
                self.full_len,
                mapped.min(self.full_capacity),
            ),
            (
                &self.raw_index_keys,
                self.raw_capacity,
                self.raw_len,
                mapped.min(self.raw_capacity),
            ),
            (
                &self.pooled_keys,
                self.pooled_capacity,
                self.pooled_len,
                pooled_mapped,
            ),
        ] {
            let row_elems = tensor.numel().checked_div(capacity).unwrap_or(0);
            if row_elems == 0 || rows_mapped <= len {
                continue;
            }
            let view = tensor.sub_offset(len * row_elems, (rows_mapped - len) * row_elems);
            gpu.hip
                .memset(&view.buf, i32::from(byte), view.buf.size())
                .map_err(MtpGpuError::Hip)?;
        }
        Ok(())
    }
}
