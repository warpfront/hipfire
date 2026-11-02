// SPDX-License-Identifier: Apache-2.0
//! GPU-only Bf16Direct glue. No allocation, NPU open or scheduler policy.
use crate::{kernels, DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::sync::LazyLock;
static PM_VERIFIED: LazyLock<bool> = LazyLock::new(|| {
    let digest = Sha256::digest(kernels::NPU_TRUNK_PM_GFX1151);
    format!("{digest:x}") == kernels::NPU_TRUNK_PM_GFX1151_SHA256
});

/// G80 panel geometry. All panels use the same padded row count.
/// `n512 = false`: the N1280 program (256-row waves, 4 A segments, 1280-column native-tile panels; the GPU consume and
/// emulate GEMM read this A). `n512 = true`: the N512 program (512-row waves, 8 A segments, 512-column panels the NPU
/// writes row-major itself). Both A streams are `rows * K * 9 / 8` bytes of identical 4608-byte K64 half chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NpuBf16Layout {
    pub padded_rows: usize,
    pub k: usize,
    pub panels: usize,
    pub n512: bool,
}
impl NpuBf16Layout {
    /// Rows of one wave of the program (the M padding unit).
    pub fn wave_rows(self) -> usize {
        if self.n512 {
            512
        } else {
            256
        }
    }
    /// Columns of one panel.
    pub fn panel_n(self) -> usize {
        if self.n512 {
            512
        } else {
            1280
        }
    }
    pub fn validate(self) -> HipResult<()> {
        // K is bounded by the N512 program's resident B panel (99 K64 chunks); the N1280 NPU program itself holds
        // K <= 2560, but its A layout also feeds the K-generic emulate GEMM (27B, K = 5120).
        if self.padded_rows == 0
            || self.padded_rows > 65536
            || self.padded_rows % self.wave_rows() != 0
            || self.k == 0
            || self.k > 6336
            || self.k % 64 != 0
            || self.panels == 0
            || self.panels > 51
        {
            return Err(HipError::new(1, "npu trunk: invalid G80 geometry"));
        }
        if self.c_bytes() > u32::MAX as usize {
            return Err(HipError::new(
                1,
                "npu trunk: C exceeds PM 32-bit buffer offsets",
            ));
        }
        Ok(())
    }
    /// N1280 only: native tiles and the emulate GEMM's A addressing.
    fn n1280(self) -> HipResult<()> {
        self.validate()?;
        if self.n512 {
            return Err(HipError::new(1, "npu trunk: N1280 layout required"));
        }
        Ok(())
    }
    pub fn a_bytes(self) -> usize {
        self.padded_rows * (self.k / 64) * 72
    }
    pub fn c_bytes(self) -> usize {
        self.padded_rows * self.padded_cols() * 2
    }
    pub fn padded_cols(self) -> usize {
        self.panels * self.panel_n()
    }
}
/// One native projection destination. Dtype is taken from the tensor (F32/BF16).
#[derive(Clone, Copy)]
pub struct NpuTrunkOutput<'a> {
    pub col0: usize,
    pub cols: usize,
    pub tensor: &'a GpuTensor,
}
fn need(t: &GpuTensor, n: usize) -> HipResult<()> {
    if t.buf.size() < n {
        Err(HipError::new(1, "npu trunk: undersized tensor"))
    } else {
        Ok(())
    }
}
fn outputs(layout: NpuBf16Layout, rows: usize, outs: &[NpuTrunkOutput<'_>]) -> HipResult<()> {
    layout.validate()?;
    if rows == 0 || rows > layout.padded_rows {
        return Err(HipError::new(1, "npu trunk: invalid live rows"));
    }
    for o in outs {
        if o.cols == 0
            || o.col0
                .checked_add(o.cols)
                .is_none_or(|n| n > layout.padded_cols())
            || !matches!(o.tensor.dtype, DType::F32 | DType::BF16)
        {
            return Err(HipError::new(1, "npu trunk: invalid output range/dtype"));
        }
        need(o.tensor, rows * o.cols * o.tensor.dtype.size())?;
    }
    Ok(())
}
impl Gpu {
    fn npu_pm(&mut self, symbol: &str) -> HipResult<()> {
        if !self.arch_caps.is_gfx1151() {
            return Err(HipError::new(1, "npu trunk PM requires gfx1151"));
        }
        if !*PM_VERIFIED {
            return Err(HipError::new(
                1,
                "npu trunk: embedded PM image differs from certified pin",
            ));
        }
        self.bind_thread()?;
        self.ensure_embedded_kernel(
            "npu_trunk_pm_gfx1151",
            kernels::NPU_TRUNK_PM_GFX1151,
            symbol,
        )
    }
    fn npu_launch(
        &mut self,
        symbol: &str,
        grid: [u32; 3],
        threads: u32,
        mut args: KernargBlob,
    ) -> HipResult<()> {
        self.launch_blob_recorded(
            symbol,
            grid,
            [threads, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }
    /// Quantize finite F16 [rows,k] into the G80 A stream and compact [rows]
    /// sidecars, each LE u16 Dm followed by LE i16 De. All padding is written zero.
    pub fn npu_trunk_pack_a_bf16(
        &mut self,
        x: &GpuTensor,
        rows: usize,
        layout: NpuBf16Layout,
        a: &GpuTensor,
        d: &GpuTensor,
    ) -> HipResult<()> {
        layout.validate()?;
        need(a, layout.a_bytes())?;
        unsafe { self.npu_trunk_pack_a_bf16_ptr(x, rows, layout, a.buf.as_ptr(), d) }
    }
    /// Lab-only incumbent producer for old/new byte and timing comparisons.
    #[cfg(feature = "lab")]
    pub fn npu_trunk_pack_a_bf16_old(
        &mut self,
        x: &GpuTensor,
        rows: usize,
        layout: NpuBf16Layout,
        a: &GpuTensor,
        d: &GpuTensor,
    ) -> HipResult<()> {
        layout.validate()?;
        need(a, layout.a_bytes())?;
        unsafe { self.npu_trunk_pack_a_impl(x, rows, layout, a.buf.as_ptr(), d, false) }
    }
    /// Same producer into a device-accessible host-registered NPU arena.
    /// # Safety
    /// `a` holds layout.a_bytes() writable bytes and stays alive through stream completion.
    pub unsafe fn npu_trunk_pack_a_bf16_ptr(
        &mut self,
        x: &GpuTensor,
        rows: usize,
        layout: NpuBf16Layout,
        a: *mut c_void,
        d: &GpuTensor,
    ) -> HipResult<()> {
        let fast = matches!(layout.k, 2560 | 5120) && self.arch_caps.is_gfx1151();
        unsafe { self.npu_trunk_pack_a_impl(x, rows, layout, a, d, fast) }
    }
    unsafe fn npu_trunk_pack_a_impl(
        &mut self,
        x: &GpuTensor,
        rows: usize,
        layout: NpuBf16Layout,
        a: *mut c_void,
        d: &GpuTensor,
        fast: bool,
    ) -> HipResult<()> {
        layout.validate()?;
        if x.dtype != DType::F16 || rows == 0 || rows > layout.padded_rows || a.is_null() {
            return Err(HipError::new(1, "npu trunk: invalid activation"));
        }
        need(x, rows * layout.k * 2)?;
        need(d, rows * 4)?;
        let symbol = match (fast, layout.k) {
            (false, _) => "npu_trunk_pack_a_bf16_old",
            (true, 2560) => "npu_trunk_pack_a_bf16",
            _ => "npu_trunk_pack_a_bf16_k5120",
        };
        self.ensure_kernel_public("npu_trunk", kernels::NPU_TRUNK_SRC, symbol)?;
        let mut b = KernargBlob::new();
        b.push_ptr(x.buf.as_ptr());
        b.push_ptr(a);
        b.push_ptr(d.buf.as_ptr());
        for n in [rows, layout.k, layout.padded_rows, layout.wave_rows()] {
            b.push_i32(n as i32);
        }
        self.npu_launch(
            symbol,
            [
                if fast {
                    layout.padded_rows as u32 / 8
                } else {
                    layout.padded_rows as u32
                },
                1,
                1,
            ],
            256,
            b,
        )
    }
    /// Scatter panel-major native BF16 tiles, dropping padding. F32 stores widen bits exactly.
    pub fn npu_trunk_consume_bf16(
        &mut self,
        c: &GpuTensor,
        layout: NpuBf16Layout,
        outs: &[NpuTrunkOutput<'_>],
        rows: usize,
    ) -> HipResult<()> {
        layout.n1280()?;
        need(c, layout.c_bytes())?;
        unsafe { self.npu_trunk_consume_bf16_ptr(c.buf.as_ptr(), layout, outs, rows) }
    }
    /// # Safety
    /// `c` holds layout.c_bytes() readable bytes, visible to the GPU, until stream completion.
    pub unsafe fn npu_trunk_consume_bf16_ptr(
        &mut self,
        c: *mut c_void,
        layout: NpuBf16Layout,
        outs: &[NpuTrunkOutput<'_>],
        rows: usize,
    ) -> HipResult<()> {
        layout.n1280()?;
        outputs(layout, rows, outs)?;
        if c.is_null() {
            return Err(HipError::new(1, "npu trunk: null C arena"));
        }
        for o in outs {
            let symbol = if o.tensor.dtype == DType::BF16 {
                "npu_trunk_consume_bf16_pm"
            } else {
                "npu_trunk_consume_f32_pm"
            };
            self.npu_pm(symbol)?;
            let mut b = KernargBlob::new();
            b.push_ptr(c);
            b.push_ptr(o.tensor.buf.as_ptr());
            for n in [layout.padded_rows, o.col0, o.cols] {
                b.push_i32(n as i32);
            }
            self.npu_launch(symbol, [o.cols.div_ceil(32) as u32, rows as u32, 1], 32, b)?;
        }
        Ok(())
    }
    /// Vector/LDS retile of native G80 BF16 into projection-owned row-major tensors.
    /// The old PM consume remains the byte oracle; this bandwidth path uses HIP compilation.
    pub fn npu_trunk_consume_bf16_fast(
        &mut self,
        c: &GpuTensor,
        layout: NpuBf16Layout,
        outs: &[NpuTrunkOutput<'_>],
        rows: usize,
    ) -> HipResult<()> {
        layout.n1280()?;
        need(c, layout.c_bytes())?;
        unsafe { self.npu_trunk_consume_bf16_fast_ptr(c.buf.as_ptr(), layout, outs, rows) }
    }
    /// # Safety
    /// `c` holds layout.c_bytes() readable GPU-visible bytes until stream completion,
    /// and does not alias any output tensor.
    pub unsafe fn npu_trunk_consume_bf16_fast_ptr(
        &mut self,
        c: *mut c_void,
        layout: NpuBf16Layout,
        outs: &[NpuTrunkOutput<'_>],
        rows: usize,
    ) -> HipResult<()> {
        layout.n1280()?;
        outputs(layout, rows, outs)?;
        if c.is_null() {
            return Err(HipError::new(1, "npu trunk: null C arena"));
        }
        self.ensure_kernel_public(
            "npu_trunk",
            kernels::NPU_TRUNK_SRC,
            "npu_trunk_consume_bf16_fast",
        )?;
        for o in outs {
            let mut b = KernargBlob::new();
            b.push_ptr(c);
            b.push_ptr(o.tensor.buf.as_ptr());
            for n in [
                layout.padded_rows,
                rows,
                o.col0,
                o.cols,
                usize::from(o.tensor.dtype == DType::BF16),
            ] {
                b.push_i32(n as i32);
            }
            self.npu_launch(
                "npu_trunk_consume_bf16_fast",
                [
                    (o.col0 % 80 + o.cols).div_ceil(80) as u32,
                    rows.div_ceil(128) as u32,
                    1,
                ],
                256,
                b,
            )?;
        }
        Ok(())
    }
    /// Exact signed-IU8 WMMA dot products from the producer's packed A.
    /// `w` is row-major [n,k] i8; `sums` is caller-owned [rows,n] I32 scratch.
    pub fn npu_trunk_emulate_i32(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        sums: &GpuTensor,
        layout: NpuBf16Layout,
        rows: usize,
        n: usize,
    ) -> HipResult<()> {
        layout.n1280()?;
        if rows == 0
            || rows > layout.padded_rows
            || n == 0
            || n > layout.padded_cols()
            || rows * n * 4 > u32::MAX as usize
        {
            return Err(HipError::new(1, "npu trunk: invalid GEMM shape"));
        }
        need(a, layout.a_bytes())?;
        need(w, n * layout.k)?;
        need(sums, rows * n * 4)?;
        self.npu_pm("npu_trunk_i8_gemm_pm")?;
        let mut b = KernargBlob::new();
        for t in [a, w, sums] {
            b.push_ptr(t.buf.as_ptr());
        }
        for n in [layout.k, rows, n, layout.padded_rows] {
            b.push_i32(n as i32);
        }
        self.npu_launch(
            "npu_trunk_i8_gemm_pm",
            [n.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
            32,
            b,
        )
    }
    /// Integer BF16-direct epilogue shared by the emulate oracle. Scale arrays
    /// are compact LE u16 mantissa/i16 exponent pairs, as in the producer.
    pub fn npu_trunk_emulate_epilogue(
        &mut self,
        sums: &GpuTensor,
        sm_se: &GpuTensor,
        dm_de: &GpuTensor,
        layout: NpuBf16Layout,
        rows: usize,
        n: usize,
        outs: &[NpuTrunkOutput<'_>],
    ) -> HipResult<()> {
        outputs(layout, rows, outs)?;
        if n == 0 || n > layout.padded_cols() || outs.iter().any(|o| o.col0 + o.cols > n) {
            return Err(HipError::new(1, "npu trunk: invalid epilogue width"));
        }
        need(sums, rows * n * 4)?;
        need(sm_se, n * 4)?;
        need(dm_de, rows * 4)?;
        self.ensure_kernel_public("npu_trunk", kernels::NPU_TRUNK_SRC, "npu_trunk_epi_bf16")?;
        for o in outs {
            let mut b = KernargBlob::new();
            for t in [sums, sm_se, dm_de, o.tensor] {
                b.push_ptr(t.buf.as_ptr());
            }
            for n in [
                rows,
                n,
                o.col0,
                o.cols,
                usize::from(o.tensor.dtype == DType::BF16),
            ] {
                b.push_i32(n as i32);
            }
            self.npu_launch(
                "npu_trunk_epi_bf16",
                [o.cols.div_ceil(256) as u32, rows as u32, 1],
                256,
                b,
            )?;
        }
        Ok(())
    }
    /// GPU-only quality/parity twin; scratch belongs to the caller, never allocated on launch.
    pub fn npu_trunk_emulate_bf16(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        sm_se: &GpuTensor,
        dm_de: &GpuTensor,
        sums: &GpuTensor,
        layout: NpuBf16Layout,
        rows: usize,
        n: usize,
        outs: &[NpuTrunkOutput<'_>],
    ) -> HipResult<()> {
        outputs(layout, rows, outs)?;
        self.npu_trunk_emulate_i32(a, w, sums, layout, rows, n)?;
        self.npu_trunk_emulate_epilogue(sums, sm_se, dm_de, layout, rows, n, outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn g80_arena_geometry_and_ragged_padding() {
        let l = NpuBf16Layout {
            padded_rows: 4864,
            k: 2560,
            panels: 13,
            n512: false,
        };
        l.validate().unwrap();
        assert_eq!(l.a_bytes(), 4 * 19 * 40 * 4608);
        assert_eq!(l.c_bytes(), 4864 * 16640 * 2);
        assert_eq!(l.padded_cols(), 16640);
        assert!(NpuBf16Layout {
            padded_rows: 4647,
            ..l
        }
        .validate()
        .is_err());
        assert!(NpuBf16Layout { k: 2561, ..l }.validate().is_err());
        assert!(NpuBf16Layout { panels: 0, ..l }.validate().is_err());
        assert!(NpuBf16Layout {
            padded_rows: 65536,
            panels: 51,
            ..l
        }
        .validate()
        .is_err());
        // 27B dense in-projection on the N512 program: 512-row waves, 512-column panels, K5120.
        let dense = NpuBf16Layout {
            padded_rows: 8192,
            k: 5120,
            panels: 33,
            n512: true,
        };
        dense.validate().unwrap();
        use peacemaker_npu::kernels::{gemm_core, gemm_g80_n512::design_g80_n512};
        let design = design_g80_n512(
            8192,
            512,
            5120,
            gemm_core::Epilogue::Bf16,
            gemm_core::Control::Fast,
        );
        assert_eq!(dense.a_bytes(), design.args[0].bytes);
        assert_eq!(dense.padded_cols(), 16896);
        assert!(NpuBf16Layout {
            padded_rows: 4864,
            ..dense
        }
        .validate()
        .is_err());
        assert!(NpuBf16Layout { k: 6400, ..dense }.validate().is_err());
        // The N1280 layout at K5120 is the emulate GEMM's input (no N1280 NPU program reaches it).
        let emulate = NpuBf16Layout {
            n512: false,
            panels: 13,
            ..dense
        };
        emulate.validate().unwrap();
        assert_eq!(emulate.a_bytes(), dense.a_bytes());
        assert!(dense.n1280().is_err() && emulate.n1280().is_ok());
    }
    #[test]
    fn fast_uint4_lds_reorder_matches_g80_byte_offsets() {
        use peacemaker_npu::kernels::gemm_g80::bf16_output_offset;
        for padded_rows in [4864usize, 8192] {
            let waves = padded_rows / 256;
            for panel in [0usize, 1, 12] {
                for tn in 0..16 {
                    for mt in [0usize, 1, 2, padded_rows / 128 - 1] {
                        let base = panel * padded_rows * 1280
                            + ((tn / 2) * waves * 4 + (mt % 2) * waves * 2 + (mt / 2) * 2 + tn % 2)
                                * 10240;
                        for i in 0..1280 {
                            let micro = i / 8;
                            let r = (micro / 10) * 8 + i % 8;
                            let j = (micro % 10) * 8;
                            for k in 0..8 {
                                let expected = panel * padded_rows * 1280 * 2
                                    + bf16_output_offset(
                                        padded_rows,
                                        mt * 128 + r,
                                        tn * 80 + j + k,
                                    );
                                assert_eq!((base + i * 8 + k) * 2, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}
