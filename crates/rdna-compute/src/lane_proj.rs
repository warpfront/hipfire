// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-batched multi-weight MQ4G256V2 projection GEMV for the A3B
//! (`qwen3_5_moe`, `.mq4rs`) exact VMM batch route, exact gfx1201, K = 2048.
//!
//! [`Gpu::gemv_mq4g256v2_lane_proj`] replaces the per-weight
//! [`Gpu::gemv_mq4g256v2_xbatch_pm`] loops of a projection group with one launch
//! over up to four weight segments (DeltaNet QKVZA, FullAttention QKV, the MoE
//! router / selector / shared gate / shared up quartet) and up to
//! [`LANE_PROJ_MAX_LANES`] activation columns. Per (row, column) the bytes are
//! the singleton `fused_qkvza_mq4g256v2` / `fused_qkv_mq4g256v2` launch's (and
//! so the x-batch GEMV's): see `kernels/src/gemv_mq4g256v2_lane_proj_k2048.gfx1201.hip`
//! and the lab gate `tests/lane_proj.rs`.
//!
//! Layout: `x` is `[lanes][2048]` f32, segment `s`'s output is
//! `[lanes][rows_s]` f32 (row-major, lane `b` at element offset `b * rows_s`).

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

/// Most weight segments one launch covers.
pub const LANE_PROJ_MAX_SEGMENTS: usize = 4;
/// Most activation columns one launch covers.
pub const LANE_PROJ_MAX_LANES: usize = 8;
/// Reduction length the kernel is compiled for.
pub const LANE_PROJ_K: usize = 2048;

/// Production module name; the entries are `{module}_n{lanes}` for
/// `lanes` in `1..=8` (the lane count is a compile-time parameter of the entry).
pub const LANE_PROJ_MODULE: &str = "gemv_mq4g256v2_lane_proj_k2048_gfx1201";

/// Entry symbols of a module named `module`, one per lane count.
pub fn lane_proj_symbols(module: &str) -> Vec<String> {
    (1..=LANE_PROJ_MAX_LANES)
        .map(|n| format!("{module}_n{n}"))
        .collect()
}

/// Rows per half-wave (a wave64 runs two independent rows per instruction).
pub const LANE_PROJ_ROWS_PER_WAVE: usize = 3;
/// wave64s per workgroup.
pub const LANE_PROJ_WAVES: usize = 8;
/// Resident workgroups per WGP the grid is sized to (measured: the 64 KB LDS
/// staging of 8 lanes lets three workgroups of 8 wave64 share a WGP).
pub const LANE_PROJ_WGS_PER_WGP: usize = 3;

/// One weight segment: `rows` MQ4G256V2 rows (K = 2048) of `w` and the f32
/// `[lanes][rows]` output `y`.
#[derive(Clone, Copy)]
pub struct LaneProjSegment<'a> {
    pub w: &'a GpuTensor,
    pub rows: usize,
    pub y: &'a GpuTensor,
}

/// Tuning variant of the kernel (lab only; the product runs
/// [`LaneProjVariant::PRODUCT`]). Any variant computes the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneProjVariant {
    /// Rows per half-wave (compile-time).
    pub rows_per_wave: usize,
    /// wave64s per workgroup (compile-time).
    pub waves: usize,
    /// Grid cap in workgroups per WGP.
    pub wgs_per_wgp: usize,
}

impl LaneProjVariant {
    pub const PRODUCT: Self = Self {
        rows_per_wave: LANE_PROJ_ROWS_PER_WAVE,
        waves: LANE_PROJ_WAVES,
        wgs_per_wgp: LANE_PROJ_WGS_PER_WGP,
    };

    fn module(&self) -> String {
        if self.rows_per_wave == Self::PRODUCT.rows_per_wave && self.waves == Self::PRODUCT.waves {
            LANE_PROJ_MODULE.to_string()
        } else {
            format!("{LANE_PROJ_MODULE}_r{}w{}", self.rows_per_wave, self.waves)
        }
    }

    fn source(&self) -> String {
        if self.module() == LANE_PROJ_MODULE {
            return kernels::GEMV_MQ4G256V2_LANE_PROJ_K2048_GFX1201_SRC.to_string();
        }
        format!(
            "#define HIPFIRE_LANE_PROJ_ROWS {}\n#define HIPFIRE_LANE_PROJ_WAVES {}\n#define HIPFIRE_LANE_PROJ_KERNEL {}\n{}",
            self.rows_per_wave,
            self.waves,
            self.module(),
            kernels::GEMV_MQ4G256V2_LANE_PROJ_K2048_GFX1201_SRC
        )
    }
}

fn proj_err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

/// Production module source (also used by the kernel registry).
pub fn lane_proj_module_source() -> String {
    LaneProjVariant::PRODUCT.source()
}

/// WGP count of device `device_id` (HIP reports WGPs on gfx12), cached.
fn wgp_count(gpu: &Gpu) -> usize {
    static CACHE: Mutex<Option<HashMap<i32, usize>>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    *map.entry(gpu.device_id).or_insert_with(|| {
        gpu.hip
            .get_device_attribute(
                crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                gpu.device_id,
            )
            .ok()
            .filter(|&v| v > 0)
            .map_or(32, |v| v as usize)
    })
}

impl Gpu {
    /// Whether [`Self::gemv_mq4g256v2_lane_proj`] is admitted: exact gfx1201,
    /// `lanes` in `1..=8`, `k == 2048`, 1..=4 segments of at most `i32::MAX`
    /// rows (a pure function of shapes; the caller guarantees the weights are
    /// MQ4G256V2 without AWQ / ParoQuant, as for the x-batch GEMV it replaces).
    pub fn lane_proj_admitted(&self, rows: &[usize], k: usize, lanes: usize) -> bool {
        let total: usize = rows.iter().sum();
        self.arch_caps.is_gfx1201()
            && k == LANE_PROJ_K
            && (1..=LANE_PROJ_MAX_LANES).contains(&lanes)
            && (1..=LANE_PROJ_MAX_SEGMENTS).contains(&rows.len())
            && rows.iter().all(|&m| m <= i32::MAX as usize)
            && (1..=i32::MAX as usize).contains(&total)
    }

    /// `y_s[b] = W_s · x[b]` for every segment `s` and lane `b < lanes`; lane
    /// `b` of segment `s` is byte-identical to the singleton one-row launch
    /// (and the x-batch GEMV) on `x[b]` alone.
    pub fn gemv_mq4g256v2_lane_proj(
        &mut self,
        segs: &[LaneProjSegment<'_>],
        x: &GpuTensor,
        lanes: usize,
    ) -> HipResult<()> {
        self.gemv_mq4g256v2_lane_proj_variant(segs, x, lanes, LaneProjVariant::PRODUCT)
    }

    #[doc(hidden)]
    pub fn gemv_mq4g256v2_lane_proj_variant(
        &mut self,
        segs: &[LaneProjSegment<'_>],
        x: &GpuTensor,
        lanes: usize,
        variant: LaneProjVariant,
    ) -> HipResult<()> {
        let rows: Vec<usize> = segs.iter().map(|s| s.rows).collect();
        if !self.lane_proj_admitted(&rows, LANE_PROJ_K, lanes) {
            return Err(proj_err(format!(
                "gemv_mq4g256v2_lane_proj: not admitted (exact gfx1201, K = 2048, 1..=8 lanes, 1..=4 segments): rows {rows:?}, lanes {lanes}"
            )));
        }
        let x_span = lanes * LANE_PROJ_K * 4;
        if x.buf.size() < x_span {
            return Err(proj_err(format!(
                "gemv_mq4g256v2_lane_proj: x holds {} B, needs {x_span} B",
                x.buf.size()
            )));
        }
        for (i, s) in segs.iter().enumerate() {
            if s.w.buf.size() < s.rows * (LANE_PROJ_K / 256) * 136 {
                return Err(proj_err(format!(
                    "gemv_mq4g256v2_lane_proj: segment {i} weight holds {} B, needs {} B",
                    s.w.buf.size(),
                    s.rows * (LANE_PROJ_K / 256) * 136
                )));
            }
            if s.y.buf.size() < lanes * s.rows * 4 {
                return Err(proj_err(format!(
                    "gemv_mq4g256v2_lane_proj: segment {i} output holds {} B, needs {} B",
                    s.y.buf.size(),
                    lanes * s.rows * 4
                )));
            }
        }
        self.bind_thread()?;
        let module = variant.module();
        let symbol = format!("{module}_n{lanes}");
        if !self.functions.contains_key(symbol.as_str()) {
            let src = variant.source();
            self.ensure_kernel(&module, &src, &symbol)?;
        }

        let mut a = [segs[0].w.buf.as_ptr(); LANE_PROJ_MAX_SEGMENTS];
        let mut y = [segs[0].y.buf.as_ptr(); LANE_PROJ_MAX_SEGMENTS];
        let mut m = [0i32; LANE_PROJ_MAX_SEGMENTS];
        for (i, s) in segs.iter().enumerate() {
            a[i] = s.w.buf.as_ptr();
            y[i] = s.y.buf.as_ptr();
            m[i] = s.rows as i32;
        }
        let x_ptr = x.buf.as_ptr();
        let total: usize = rows.iter().sum();
        // Every half-wave (two per wave64) walks a contiguous run of rows in
        // batches of `rows_per_wave`; the grid is sized to what is resident.
        let grid = total
            .div_ceil(variant.waves * 2 * variant.rows_per_wave)
            .clamp(1, (wgp_count(self) * variant.wgs_per_wgp).max(1)) as u32;
        let shared = (lanes * LANE_PROJ_K * 4) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &a[0] as *const _ as *mut c_void,
            &a[1] as *const _ as *mut c_void,
            &a[2] as *const _ as *mut c_void,
            &a[3] as *const _ as *mut c_void,
            &y[0] as *const _ as *mut c_void,
            &y[1] as *const _ as *mut c_void,
            &y[2] as *const _ as *mut c_void,
            &y[3] as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &m[0] as *const _ as *mut c_void,
            &m[1] as *const _ as *mut c_void,
            &m[2] as *const _ as *mut c_void,
            &m[3] as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            &symbol,
            [grid, 1, 1],
            [(variant.waves * 64) as u32, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in a {
                    b.push_ptr(p);
                }
                for p in y {
                    b.push_ptr(p);
                }
                b.push_ptr(x_ptr);
                for v in m {
                    b.push_i32(v);
                }
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_source_is_the_registered_source() {
        assert_eq!(
            lane_proj_module_source(),
            kernels::GEMV_MQ4G256V2_LANE_PROJ_K2048_GFX1201_SRC
        );
        assert!(kernels::GEMV_MQ4G256V2_LANE_PROJ_K2048_GFX1201_SRC
            .starts_with("// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n"));
    }

    #[test]
    fn variants_get_distinct_modules_and_defines() {
        let v = LaneProjVariant {
            rows_per_wave: 4,
            waves: 16,
            wgs_per_wgp: 2,
        };
        assert_ne!(v.module(), LaneProjVariant::PRODUCT.module());
        assert!(v.source().starts_with("#define HIPFIRE_LANE_PROJ_ROWS 4\n"));
        // the grid cap alone does not change the module
        let g = LaneProjVariant {
            wgs_per_wgp: 7,
            ..LaneProjVariant::PRODUCT
        };
        assert_eq!(g.module(), LANE_PROJ_MODULE);
    }

    #[test]
    fn registry_entry_exports_every_lane_entry() {
        let entries = crate::kernel_registry::entries("gfx1201", "").expect("registry");
        let e = entries
            .iter()
            .find(|e| e.module == LANE_PROJ_MODULE)
            .expect("lane_proj module registered for gfx1201");
        let want = lane_proj_symbols(LANE_PROJ_MODULE);
        assert_eq!(e.symbols.len(), want.len());
        for sym in &want {
            assert!(e.symbols.contains(&sym.as_str()), "missing {sym}");
        }
        assert_eq!(e.source(), lane_proj_module_source());
        // gfx1100 / gfx1151 must not carry the exact-gfx1201 module
        for arch in ["gfx1100", "gfx1151"] {
            let other = crate::kernel_registry::entries(arch, "").expect("registry");
            assert!(other.iter().all(|e| e.module != LANE_PROJ_MODULE), "{arch}");
        }
    }

    #[test]
    fn symbols_cover_one_to_eight_lanes() {
        let s = lane_proj_symbols(LANE_PROJ_MODULE);
        assert_eq!(s.len(), 8);
        assert_eq!(s[0], format!("{LANE_PROJ_MODULE}_n1"));
        assert_eq!(s[7], format!("{LANE_PROJ_MODULE}_n8"));
    }
}
