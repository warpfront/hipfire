// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact HFQ4G256-v1 GEMM for Qwen35 MTP prompt fill on gfx1201.
//! Four output rows share each activation load; column tiles are grid-y.
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::LazyLock};

const MODULE: &str = "gemm_hfq4g256_rows_pm";
const IMAGE: &[u8] =
    include_bytes!("../../../kernels/pm-decode/gfx1201/gemm_hfq4g256_rows_pm.hxaco");
const SHA256: &str = "657eb7c1828fdb80f078c239f17527b4b6e249023824d7864178840bfe2129cb";
static VERIFIED: LazyLock<bool> = LazyLock::new(|| {
    let got: String = Sha256::digest(IMAGE)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    got == SHA256
});

pub(crate) fn route_object() -> (&'static str, &'static [u8], &'static [&'static str]) {
    (MODULE, IMAGE, &[MODULE])
}

impl Gpu {
    /// Shape/architecture admission. All byte offsets must fit the kernel's
    /// 32-bit raw-buffer addressing; grid-y uses the 16-bit hardware id.
    pub fn supports_gemm_hfq4g256_rows_pm(&self, m: usize, k: usize, n: usize) -> bool {
        let fits = |v: Option<usize>| v.is_some_and(|v| v <= i32::MAX as usize);
        self.arch == "gfx1201"
            && m > 0
            && k > 0
            && n > 0
            && k.is_multiple_of(256)
            && n.div_ceil(8) <= u16::MAX as usize
            && fits(m.checked_mul(4).and_then(|v| v.checked_mul(n)))
            && fits(
                k.checked_mul(4)
                    .and_then(|v| v.checked_mul(n.div_ceil(8) * 8)),
            )
            && fits(m.checked_mul(k / 256).and_then(|v| v.checked_mul(136)))
    }

    /// Overwrite y[N][M] with HFQ4G256-v1 A[M][K] times x[N][K].
    /// MQ4 callers must rotate x first. No fallback: the caller owns admission.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_rows_pm(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if !self.supports_gemm_hfq4g256_rows_pm(m, k, n)
            || x.dtype != DType::F32
            || y.dtype != DType::F32
        {
            return Err(HipError::new(0, "gemm_hfq4g256_rows_pm: gfx1201, positive dimensions, K % 256 == 0, f32 x/y and bounded byte offsets required"));
        }
        if a.buf.size() < m * (k / 256) * 136
            || x.buf.size() < n * k * 4
            || y.buf.size() < n * m * 4
        {
            return Err(HipError::new(0, "gemm_hfq4g256_rows_pm: undersized tensor"));
        }
        if !*VERIFIED {
            return Err(HipError::new(
                0,
                "gemm_hfq4g256_rows_pm: embedded image hash mismatch",
            ));
        }
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, MODULE)?;
        let (ap, xp, yp) = (a.buf.as_ptr(), x.buf.as_ptr(), y.buf.as_ptr());
        let (mv, kv, nv) = (m as i32, k as i32, n as i32);
        let mut params: [*mut c_void; 6] = [
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            MODULE,
            [m.div_ceil(4) as u32, n.div_ceil(8) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b
            },
        )
    }
}
