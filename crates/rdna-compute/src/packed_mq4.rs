// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Default-off PR #616 packed layout, with native Q8-group32 activation precision.
//! This is not the MQ4V2 INT8 sidecar (136 bytes/block): it uses 144-byte DS4.
use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

const SOURCE: &str = include_str!("../../../kernels/src/gemm_mq4_packed.gfx1100.hip");
const MODULE: &str = "gemm_mq4_packed_gfx1100";

fn admitted(arch: &str, enabled: bool, recording: bool, m: usize, k: usize, n: usize) -> bool {
    enabled
        && arch == "gfx1100"
        && !recording
        && n > 0
        && n % 256 == 0
        && matches!(
            (m, k),
            (17408, 5120)
                | (5120, 17408)
                | (5120, 6144)
                | (6144, 5120)
                | (10240, 5120)
                | (12288, 5120)
        )
}

impl Gpu {
    /// Caller additionally checks the weight dtype and producer/consumer contract.
    pub fn packed_mq4_admitted(&self, m: usize, k: usize, n: usize) -> bool {
        admitted(
            &self.arch,
            self.flags.packed_mq4_prefill,
            self.graphs.capture_mode || self.replay.is_recording(),
            m,
            k,
            n,
        )
    }

    /// F32 input/output entry. No pointer-keyed activation reuse: every call packs
    /// its input afresh. The allocation follows existing DS4 scratch lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4_packed(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.gemm_packed_impl(w, x, y, m, k, n, add, false)
    }

    /// Uniform qt44 only; Lloyd codebooks must not enter this entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4v2_packed(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.gemm_packed_impl(w, x, y, m, k, n, add, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_packed_impl(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
        v2: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.packed_mq4_admitted(m, k, n) {
            return Err(HipError::new(
                0,
                "packed MQ4 requires admitted gfx1100 prefill shape, outside capture",
            ));
        }
        // One opt-in route marker per format/epilogue, so a fallback cannot
        // accidentally be reported as validation of this experimental kernel.
        static SEEN_MQ4: AtomicBool = AtomicBool::new(false);
        static SEEN_V2: AtomicBool = AtomicBool::new(false);
        static SEEN_MQ4_ADD: AtomicBool = AtomicBool::new(false);
        static SEEN_V2_ADD: AtomicBool = AtomicBool::new(false);
        let seen = match (v2, add) {
            (false, false) => &SEEN_MQ4,
            (true, false) => &SEEN_V2,
            (false, true) => &SEEN_MQ4_ADD,
            (true, true) => &SEEN_V2_ADD,
        };
        if !seen.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[packed-mq4] admitted format={} m={m} k={k} n={n} add={add}",
                if v2 { "mq4v2" } else { "mq4" }
            );
        }
        let needed = crate::scratch::q8_1_mmq_x_needed(k, n);
        if crate::scratch::scratch_will_grow(
            self.scratch.q8_1_mmq_x_scratch_bytes,
            self.scratch.q8_1_mmq_x_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let mut xp = x.buf.as_ptr();
        let mut qp = self.scratch.reserve_packed_mq4(&self.hip, k, n)?;
        let mut ki = k as i32;
        let mut ni = n as i32;
        const QUANT: &str = "quantize_mq4_packed_group128";
        self.ensure_kernel(MODULE, SOURCE, QUANT)?;
        let mut params = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            QUANT,
            [k.div_ceil(1024) as u32, n as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(qp);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        )?;
        let kernel = match (v2, add) {
            (false, false) => "gemm_mq4_packed_gfx1100_full_set",
            (false, true) => "gemm_mq4_packed_gfx1100_full_add",
            (true, false) => "gemm_mq4v2_packed_gfx1100_full_set",
            (true, true) => "gemm_mq4v2_packed_gfx1100_full_add",
        };
        if v2 {
            static V2: OnceLock<String> = OnceLock::new();
            let src = V2.get_or_init(|| {
                format!(
                    "#define PACKED_MQ4_V2 1\n{}",
                    SOURCE
                        .replace("gemm_mq4_packed_gfx1100", "gemm_mq4v2_packed_gfx1100")
                        .replace(
                            "quantize_mq4_packed_group128",
                            "quantize_mq4v2_packed_group128"
                        )
                )
            });
            self.ensure_kernel("gemm_mq4v2_packed_gfx1100", src, kernel)?;
        } else {
            self.ensure_kernel(MODULE, SOURCE, kernel)?;
        }
        let mut wp = w.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ai = i32::from(add);
        let mut params = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ai as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            kernel,
            [(m / 64) as u32, (n / 256) as u32, 1],
            [32, 8, 1],
            ((256 * 36 + 64 * 76) * 4 + 256 * 4) as u32,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(qp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b.push_i32(ai);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::admitted;
    #[test]
    fn gate_is_opt_in_exact_arch_shape_and_full_tile_only() {
        assert!(admitted("gfx1100", true, false, 17408, 5120, 2048));
        assert!(!admitted("gfx1100", false, false, 17408, 5120, 2048));
        for arch in ["gfx1101", "gfx1151", "gfx1201"] {
            assert!(!admitted(arch, true, false, 17408, 5120, 2048));
        }
        for n in [0, 1, 16, 255, 257] {
            assert!(!admitted("gfx1100", true, false, 17408, 5120, n));
        }
        assert!(!admitted("gfx1100", true, true, 17408, 5120, 2048));
        assert!(!admitted("gfx1100", true, false, 4096, 4096, 2048));
    }
}
