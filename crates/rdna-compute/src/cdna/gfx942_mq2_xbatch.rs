// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! x-batched MQ2G256V2 GEMV of the MTP draft head's ranking on exact gfx942.
//!
//! One weight pass for up to eight rotated activation rows
//! (`kernels/src/qwen4_g942_gemv_mq2g256v2_xbatch.gfx942.hip`), every output element
//! bit-equal to the one-row `gemv_mq2g256v2` of that row (kernel oracle:
//! `examples/qwen4_g942_mq2_xbatch_oracle.rs`). Covers K = 2048 and K = 2560 (the two hidden
//! widths of the Qwen4 family: the x values of a row group stay register resident, so the
//! kernel is instantiated per K); every other shape keeps the hipcc x-batched kernel.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use super::Gfx942Device;
use crate::{Gpu, GpuTensor};

pub(super) const QWEN4_G942_MQ2_XBATCH_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_gemv_mq2g256v2_xbatch.gfx942.hip");
/// Module key of [`QWEN4_G942_MQ2_XBATCH_SRC`].
pub const QWEN4_G942_MQ2_XBATCH_MODULE: &str = "qwen4_g942_gemv_mq2g256v2_xbatch";

/// Rows (output rows of `W`) one 256-thread block covers: two row slots of
/// `ROWS_PER_SLOT` (`MQ2X_ROWS_PER_SLOT` in the kernel file) consecutive rows.
const ROWS_PER_BLOCK: usize = 32;
/// Most activation rows one launch takes (the kernel's widest instance).
pub const QWEN4_G942_MQ2_XBATCH_MAX_ROWS: usize = 8;

/// The entry of the `(rows, k)` instance: `rows <= 4` runs the four-row body, up to eight the
/// eight-row body; `None` outside the covered widths.
pub fn qwen4_g942_mq2_xbatch_entry(rows: usize, k: usize) -> Option<&'static str> {
    if rows == 0 || rows > QWEN4_G942_MQ2_XBATCH_MAX_ROWS {
        return None;
    }
    Some(match (rows <= 4, k) {
        (true, 2048) => "qwen4_g942_gemv_mq2g256v2_xbatch4_k2048",
        (true, 2560) => "qwen4_g942_gemv_mq2g256v2_xbatch4_k2560",
        (false, 2048) => "qwen4_g942_gemv_mq2g256v2_xbatch8_k2048",
        (false, 2560) => "qwen4_g942_gemv_mq2g256v2_xbatch8_k2560",
        _ => return None,
    })
}

/// Whether [`Gfx942Device::mq2g256v2_xbatch_head`] covers `m` rows of width `k`: `K` in
/// {2048, 2560} and every weight byte offset inside the kernel's 32-bit row index.
pub fn qwen4_g942_mq2_xbatch_covers(m: usize, k: usize) -> bool {
    m > 0
        && matches!(k, 2048 | 2560)
        && (k / 256)
            .checked_mul(72)
            .and_then(|row| row.checked_mul(m))
            .is_some_and(|bytes| bytes <= i32::MAX as usize)
        && m.checked_mul(8)
            .is_some_and(|bytes| bytes <= i32::MAX as usize)
}

/// Bind `kernel` of the module: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_MQ2_XBATCH_IMAGE=/absolute/device.elf` names one, that explicitly supplied
/// code object (the Route-A twin proof: same module, same entry names, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_mq2_xbatch_ensure(gpu: &mut Gpu, kernel: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_MQ2_XBATCH_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_MQ2_XBATCH_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_G942_MQ2_XBATCH_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                kernel,
            )
        }
        _ => gpu.ensure_kernel(
            QWEN4_G942_MQ2_XBATCH_MODULE,
            QWEN4_G942_MQ2_XBATCH_SRC,
            kernel,
        ),
    }
}

impl Gfx942Device<'_> {
    /// `Y[b, row] = W[row] . X[b]` for `batch` (1..=8) rows of an MQ2G256V2 weight, one weight
    /// pass; each element equals `Gpu::gemv_mq2g256v2` on that row bit for bit. `x` holds the
    /// rows already in the weight's FWHT basis (`[batch, k]` F32), `y` is `[batch, m]` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2g256v2_xbatch_head(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        let entry = qwen4_g942_mq2_xbatch_entry(batch, k)
            .filter(|_| qwen4_g942_mq2_xbatch_covers(m, k))
            .ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("gfx942 MQ2 x-batch: M={m} K={k} B={batch} outside the module"),
                )
            })?;
        self.gpu.bind_thread()?;
        g942_mq2_xbatch_ensure(self.gpu, entry)?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let (m_val, k_val, b_val) = (m as i32, k as i32, batch as i32);
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let bytes = m * (k / 256) * 72 + batch * k * 4 + batch * m * 4;
        let timer = crate::profile::begin_timer(&self.gpu.hip, "gemv", entry, bytes);
        let result = self.gpu.launch_maybe_blob(
            entry,
            [m.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }
}
