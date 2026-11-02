// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder (peacemaker) multi-column MQ4G256 V2 GEMVs for exact gfx1201.
//!
//! One launch computes up to [`PM_XBATCH_MAX`] output columns, each
//! byte-identical to its single-column incumbent:
//!
//! - [`Gpu::gemv_mq4g256v2_xbatch_pm`]: `y[b] = W · x[b]`, per column equal
//!   to the hipcc `gemv_mq4g256v2_xbatch` at B = 1 (PmPlainX oracle,
//!   `kernels/pm-decode/gfx1201/gemv_hfq4g256_xbatch_mq4v2.acceptance.json`).
//! - [`Gpu::gemv_mq4g256v2_residual_xbatch`]: `y[b] += W · x[b]`, per column
//!   equal to the hipcc `gemv_mq4g256v2_residual` (PmResidualX oracle,
//!   `gemv_hfq4g256_residual_xbatch_mq4v2.oracle.json`).
//! - [`Gpu::gemv_mq4g256v2_multirow_r2_xbatch_pm`]: `y[b] = W · x[b]`, per
//!   column equal to the singleton `gemv_mq4g256v2_multirow_r2` (the lm_head
//!   route; PmMultirow oracle,
//!   `gemv_hfq4g256_multirow_xbatch_mq4v2.acceptance.json`).
//!
//! The embedded objects are the oracle-accepted ones regenerated for 0.4.3 with the emitter's
//! code-end tail fill (`s_code_end` words appended to `.text`): kernel bodies, descriptors,
//! symbols and metadata are unchanged, and only the added tail, the ELF layout fields derived
//! from it and the `_DYNAMIC` symbol value differ. The oracles above
//! ran on the pre-fill objects (GPU hardware measurements) and were not re-run. They are pinned by SHA-256 and
//! verified once before first load (fails closed). They load only on exact
//! `gfx1201` with `kernel.pm_decode` on — the same gate as
//! [`crate::pm_decode_twins`]; everywhere else the launchers run the hipcc
//! incumbent (the x-batch GEMV, or the singleton residual per column), which
//! is the oracle reference itself, so results are identical either way.
//!
//! ABI (both): `A, x, y` pointers at 0/8/16, `M, K, B` i32 at 24/28/32;
//! block 32; x is `[B][K]`, y is `[B][M]` (f32, row-major).

use crate::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::sync::LazyLock;

/// Largest column count of one launch.
pub const PM_XBATCH_MAX: usize = 8;

struct PmObject {
    module: &'static str,
    symbol: &'static str,
    image: &'static [u8],
    sha256: &'static str,
    /// Output rows one workgroup owns; grid = `ceil(M / rows)`. The plain
    /// object was accepted at grid M (surplus workgroups own no rows).
    rows_per_workgroup: Option<usize>,
}

impl PmObject {
    fn verify(&self) -> Result<(), String> {
        let got: String = Sha256::digest(self.image)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if got != self.sha256 {
            return Err(format!(
                "{}: embedded object sha256 {got} is not the accepted {}",
                self.module, self.sha256
            ));
        }
        Ok(())
    }

    fn grid(&self, m: usize) -> u32 {
        match self.rows_per_workgroup {
            Some(rows) => m.div_ceil(rows) as u32,
            None => m as u32,
        }
    }
}

/// PmPlainX (`gemv_hfq4g256_xbatch_mq4v2.acceptance.json`: `.hxaco`
/// accepted, 84 cases x B 1..8, zero mismatch bytes; launched at grid M).
static PLAIN: PmObject = PmObject {
    module: "gemv_hfq4g256_xbatch_mq4v2_pm",
    symbol: "gemv_mq4g256v2_xbatch_pm",
    image: include_bytes!("../../../kernels/pm-decode/gfx1201/gemv_hfq4g256_xbatch_mq4v2.hxaco"),
    sha256: "97c2872246c123a03bbb4751fed75b2f7235654482c96b007eb69c2ffbb41b59",
    rows_per_workgroup: None,
};

/// PmResidualX (`gemv_hfq4g256_residual_xbatch_mq4v2.oracle.json`: the
/// `.co` is the accepted candidate object; 4 rows per workgroup).
static RESIDUAL: PmObject = PmObject {
    module: "gemv_hfq4g256_residual_xbatch_mq4v2_pm",
    symbol: "gemv_mq4g256v2_residual_xbatch",
    image: include_bytes!(
        "../../../kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_xbatch_mq4v2.co"
    ),
    sha256: "7ad33f79e552f550a46a96c0863760bb3ac31ac8d7223e256a59ade20eb4ade1",
    rows_per_workgroup: Some(4),
};

static PLAIN_VERIFIED: LazyLock<Result<(), String>> = LazyLock::new(|| PLAIN.verify());
static RESIDUAL_VERIFIED: LazyLock<Result<(), String>> = LazyLock::new(|| RESIDUAL.verify());

/// PmMultirow (`gemv_hfq4g256_multirow_xbatch_mq4v2.acceptance.json`: the
/// `.co` ELF is the oracled object; 2 rows per workgroup).
static MULTIROW: PmObject = PmObject {
    module: "gemv_hfq4g256_multirow_xbatch_mq4v2_pm",
    symbol: "gemv_mq4g256v2_multirow_r2_xbatch_pm",
    image: include_bytes!(
        "../../../kernels/pm-decode/gfx1201/gemv_hfq4g256_multirow_xbatch_mq4v2.co"
    ),
    sha256: "fadb665f00c35bea787fda7c6956d9af61f5f6d1e60f759c2c7f2c23a1ddd9bf",
    rows_per_workgroup: Some(2),
};

static MULTIROW_VERIFIED: LazyLock<Result<(), String>> = LazyLock::new(|| MULTIROW.verify());

/// The embedded objects as closed-route plan entries: `(module, image,
/// symbol)`, preloaded at model load so the VMM exact route never loads a
/// module after the route is sealed. Their digests are checked again at
/// the first launch.
pub(crate) fn route_objects() -> [(&'static str, &'static [u8], &'static [&'static str]); 3] {
    const PLAIN_SYMBOLS: &[&str] = &["gemv_mq4g256v2_xbatch_pm"];
    const RESIDUAL_SYMBOLS: &[&str] = &["gemv_mq4g256v2_residual_xbatch"];
    const MULTIROW_SYMBOLS: &[&str] = &["gemv_mq4g256v2_multirow_r2_xbatch_pm"];
    [
        (PLAIN.module, PLAIN.image, PLAIN_SYMBOLS),
        (RESIDUAL.module, RESIDUAL.image, RESIDUAL_SYMBOLS),
        (MULTIROW.module, MULTIROW.image, MULTIROW_SYMBOLS),
    ]
}

impl Gpu {
    /// The accepted PM objects replace the hipcc incumbents here.
    fn pm_xbatch_enabled(&self) -> bool {
        self.scratch.pm_decode && self.arch == "gfx1201"
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_pm_xbatch(
        &mut self,
        obj: &'static PmObject,
        verified: &Result<(), String>,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        verified
            .clone()
            .map_err(|reason| HipError::new(0, &reason))?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(obj.module, obj.image, obj.symbol)?;
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
        self.launch_maybe_blob(
            obj.symbol,
            [obj.grid(m), 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(b_val);
                b
            },
        )
    }

    fn check_xbatch_shape(name: &str, m: usize, k: usize, batch: usize) -> HipResult<()> {
        if !(1..=PM_XBATCH_MAX).contains(&batch)
            || m == 0
            || k == 0
            || !k.is_multiple_of(256)
            || m > i32::MAX as usize
            || k > i32::MAX as usize
        {
            return Err(HipError::new(
                0,
                &format!("{name}: batch {batch} (1..={PM_XBATCH_MAX}), m {m} > 0, k {k} % 256 == 0 required"),
            ));
        }
        Ok(())
    }

    /// `y[b] = W · x[b]` for `b < batch` (<= [`PM_XBATCH_MAX`]), each column
    /// byte-identical to the singleton MQ4G256 V2 x-batch GEMV at B = 1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq4g256v2_xbatch_pm(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        Self::check_xbatch_shape("gemv_mq4g256v2_xbatch_pm", m, k, batch)?;
        if self.pm_xbatch_enabled() {
            return self.launch_pm_xbatch(&PLAIN, &PLAIN_VERIFIED, a_raw, x, y, m, k, batch);
        }
        self.gemv_mq4g256v2_xbatch(a_raw, x, y, m, k, batch)
    }

    /// `y[b] = W · x[b]` for `b < batch` (<= [`PM_XBATCH_MAX`]), each column
    /// byte-identical to the singleton `gemv_mq4g256v2_multirow_r2` (the
    /// lm_head GEMV). Byte offsets of W, x and y must fit 32 bits.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq4g256v2_multirow_r2_xbatch_pm(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        Self::check_xbatch_shape("gemv_mq4g256v2_multirow_r2_xbatch_pm", m, k, batch)?;
        let fits = |n: Option<usize>| n.is_some_and(|n| n <= u32::MAX as usize);
        if !fits(m.checked_mul(batch * 4))
            || !fits(k.checked_mul(batch * 4))
            || !fits((k / 256).checked_mul(m * 136))
        {
            return Err(HipError::new(
                0,
                "gemv_mq4g256v2_multirow_r2_xbatch_pm: 32-bit byte offsets required",
            ));
        }
        if self.pm_xbatch_enabled() {
            return self.launch_pm_xbatch(&MULTIROW, &MULTIROW_VERIFIED, a_raw, x, y, m, k, batch);
        }
        for b in 0..batch {
            let xb = x.sub_offset(b * k, k);
            let yb = y.sub_offset(b * m, m);
            self.gemv_mq4g256v2_multirow(a_raw, &xb, &yb, m, k)?;
        }
        Ok(())
    }

    /// `y[b] += W · x[b]` for `b < batch` (<= [`PM_XBATCH_MAX`]), each column
    /// byte-identical to the singleton `gemv_mq4g256v2_residual` on
    /// (`x[b]`, `y[b]`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq4g256v2_residual_xbatch(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        Self::check_xbatch_shape("gemv_mq4g256v2_residual_xbatch", m, k, batch)?;
        if self.pm_xbatch_enabled() {
            return self.launch_pm_xbatch(&RESIDUAL, &RESIDUAL_VERIFIED, a_raw, x, y, m, k, batch);
        }
        for b in 0..batch {
            let xb = x.sub_offset(b * k, k);
            let yb = y.sub_offset(b * m, m);
            self.gemv_hfq4g256_residual_mq4v2(a_raw, &xb, &yb, m, k)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_objects_match_their_accepted_digests() {
        PLAIN.verify().unwrap();
        RESIDUAL.verify().unwrap();
        MULTIROW.verify().unwrap();
    }
}
