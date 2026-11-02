// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launcher for the sparse sampled-row compaction of the DFlash
//! sampled-head draft (`kernels/src/sampled_rows_compact.hip`).
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::sampled_rows_compact_into_f32`].
pub const SAMPLED_ROWS_COMPACT_SRC: &str =
    include_str!("../../../kernels/src/sampled_rows_compact.hip");
/// Module key.
pub const SAMPLED_ROWS_COMPACT_MODULE: &str = "sampled_rows_compact";
/// Device symbol.
pub const SAMPLED_ROWS_COMPACT_SYMBOL: &str = "sampled_rows_compact_f32";
/// Threads per block (one block per row); `SRC_THREADS` in the kernel.
const BLOCK: u32 = 256;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(
        0,
        &format!("sampled_rows_compact_into_f32: {}", msg.as_ref()),
    )
}

impl Gpu {
    /// Compact the entries `p >= tau[r]` of each row `r` of `probs` (`[rows, vocab]`
    /// F32) into `idx` / `val` in ascending column order, one block per row.
    ///
    /// * `tau` (`rows` F32, device): per-row threshold.
    /// * `counts` (`rows` x 4-byte i32): the TRUE number of kept columns per
    ///   row, even when it exceeds `capacity` and for `tau[r] <= 0` (which keeps
    ///   every non-NaN column). NaN never passes `p >= tau`.
    /// * `idx` (`rows * capacity` x i32): kept column ids; ranks `>= capacity`
    ///   are counted but not stored; slots past `min(count, capacity)` are
    ///   left untouched.
    /// * `val` (`rows * capacity` F32): the raw f32 bits of the kept probs.
    ///
    /// `counts`/`idx` have
    /// no integer `DType`, so only byte sizes are checked for them. Output
    /// buffers must not overlap `probs`.
    #[allow(clippy::too_many_arguments)]
    pub fn sampled_rows_compact_into_f32(
        &mut self,
        probs: &GpuTensor,
        tau: &GpuTensor,
        counts: &GpuTensor,
        idx: &GpuTensor,
        val: &GpuTensor,
        vocab: usize,
        rows: usize,
        capacity: usize,
    ) -> HipResult<()> {
        if rows == 0 || vocab == 0 || capacity == 0 {
            return Err(err(format!(
                "rows {rows}, vocab {vocab}, capacity {capacity} must all be > 0"
            )));
        }
        if vocab > i32::MAX as usize
            || capacity > i32::MAX as usize
            || rows > u32::MAX as usize
            || rows
                .checked_mul(vocab.max(capacity))
                .is_none_or(|n| n.checked_mul(4).is_none())
        {
            return Err(err("rows * vocab / capacity exceeds the addressable range"));
        }
        if probs.dtype != DType::F32 || val.dtype != DType::F32 || tau.dtype != DType::F32 {
            return Err(err(format!(
                "probs, tau and val must be F32 (got {:?}, {:?}, {:?})",
                probs.dtype, tau.dtype, val.dtype
            )));
        }
        let probs_need = rows * vocab * 4;
        let slot_need = rows * capacity * 4;
        let counts_need = rows * 4;
        for (name, t, need) in [
            ("probs", probs, probs_need),
            ("tau", tau, counts_need),
            ("counts", counts, counts_need),
            ("idx", idx, slot_need),
            ("val", val, slot_need),
        ] {
            if t.buf.size() < need {
                return Err(err(format!(
                    "{name} buffer too small (have {} need {need} bytes)",
                    t.buf.size()
                )));
            }
        }
        self.bind_thread()?;
        self.ensure_kernel(
            SAMPLED_ROWS_COMPACT_MODULE,
            SAMPLED_ROWS_COMPACT_SRC,
            SAMPLED_ROWS_COMPACT_SYMBOL,
        )?;
        let pp = probs.buf.as_ptr();
        let tp = tau.buf.as_ptr();
        let cp = counts.buf.as_ptr();
        let ip = idx.buf.as_ptr();
        let vp = val.buf.as_ptr();
        let vocab_i = vocab as i32;
        let cap_i = capacity as i32;
        let mut params: [*mut c_void; 7] = [
            &pp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &vocab_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
        ];
        let bytes = probs_need + counts_need + slot_need * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "sampled_rows_compact",
            SAMPLED_ROWS_COMPACT_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            SAMPLED_ROWS_COMPACT_SYMBOL,
            [rows as u32, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(tp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(vp);
                b.push_i32(vocab_i);
                b.push_i32(cap_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
