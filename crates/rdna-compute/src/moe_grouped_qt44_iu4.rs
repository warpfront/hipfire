// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raon grouped W4A4 routed-expert GEMM for gfx1201 (`kernels/src/moe_grouped_qt44_iu4_sym.gfx12.hip`).
//!
//! One kernel family serves the gate/up (`M = 2 * 1088`, `K = 2560`) and the down
//! (`M = 2560`, `K = 1280`) projection of a symmetric qt44 (MQ4G256V2, `zp == -8 * sc`)
//! expert set against the `block_i4_128` activation sidecar of `Gpu::ensure_int4_mmq_x`
//! (epoch-major `[K / 128, x_src_rows]`). Output is F32 `[m_total, M]` in grouped order,
//! a drop-in for `gemm_mq4g256v2_moe_grouped_wmma_k2`'s output. See the kernel header for
//! the arithmetic and the leader/tile-run scheme.

use crate::dispatch::{Gpu, GpuTensor};
use crate::gemm::moe_expert_table_len;
use hip_bridge::HipResult;
use std::ffi::c_void;

pub const SRC: &str = include_str!("../../../kernels/src/moe_grouped_qt44_iu4_sym.gfx12.hip");
pub const MODULE: &str = "moe_grouped_qt44_iu4_sym_gfx12";

/// Row fragments per wave and tile-run widths built into the module (`nt`, `rf`).
pub const ENTRIES: [(usize, usize); 5] = [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)];

pub const A8_SRC: &str = include_str!("../../../kernels/src/moe_grouped_qt44_a8_sym.gfx12.hip");
pub const A8_MODULE: &str = "moe_grouped_qt44_a8_sym_gfx12";
/// Entries of the W4A8 module (`nt4r1` spills the VGPR file and is not built).
pub const A8_ENTRIES: [(usize, usize); 4] = [(1, 1), (2, 1), (1, 2), (2, 2)];

/// gfx11 (gfx1100 / gfx1151) twin of the W4A8 module: RDNA3 K16 iu8 operand form, same fold, so its
/// output is bit-identical to the gfx1201 module's for equal inputs.
pub const A8_GFX11_SRC: &str =
    include_str!("../../../kernels/src/moe_grouped_qt44_a8_sym.gfx11.hip");
pub const A8_GFX11_MODULE: &str = "moe_grouped_qt44_a8_sym_gfx11";
/// Entries of the gfx11 W4A8 module (`nt4r2` spills and is not built).
pub const A8_GFX11_ENTRIES: [(usize, usize); 5] = [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)];

/// Whether `arch` has a W4A8 module (gfx1201, or the gfx11 twin on gfx1100 / gfx1151).
pub fn a8_supported(arch: &str) -> bool {
    matches!(arch, "gfx1201" | "gfx1100" | "gfx1151")
}

/// Whether `arch` has the W4A4 and FP8 modules (gfx1201 only).
pub fn a4_fp8_supported(arch: &str) -> bool {
    arch == "gfx1201"
}

/// Built W4A8 `(nt, rf)` entries of `arch`'s module (empty when unsupported).
pub fn a8_entries(arch: &str) -> &'static [(usize, usize)] {
    match arch {
        "gfx1201" => &A8_ENTRIES,
        "gfx1100" | "gfx1151" => &A8_GFX11_ENTRIES,
        _ => &[],
    }
}

/// Symbol of the gfx11 W4A8 `(nt, rf)` entry.
pub fn symbol_a8_gfx11(nt: usize, rf: usize) -> Option<&'static str> {
    Some(match (nt, rf) {
        (1, 1) => "moe_grouped_qt44_a8_sym_gfx11_nt1r1",
        (2, 1) => "moe_grouped_qt44_a8_sym_gfx11_nt2r1",
        (4, 1) => "moe_grouped_qt44_a8_sym_gfx11_nt4r1",
        (1, 2) => "moe_grouped_qt44_a8_sym_gfx11_nt1r2",
        (2, 2) => "moe_grouped_qt44_a8_sym_gfx11_nt2r2",
        _ => return None,
    })
}

/// Default gfx11 W4A8 `(nt, rf)` from the node sweeps and the end-to-end A/B. gfx1100 (7900 XTX):
/// `nt2r2` at >= 12288 grouped rows (4096 tokens: gate/up 4.02 ms, down 2.54 ms vs F16 5.64 / 3.60),
/// below it `nt2r1` (on the XTX's 512-row prefill chunks `nt2r1` beat `nt1r1` end to end). gfx1151
/// (Strix Halo, bandwidth bound): `nt4r1` below 12288 grouped rows (512 tokens: gate/up 2.12 ms, down
/// 1.32 ms), `nt2r2` above (4096 tokens: 8.00 / 5.01 ms). `rf = 2` needs `m % 128 == 0`.
pub fn select_entry_a8_gfx11(arch: &str, m_total: usize, m: usize) -> (usize, usize) {
    if m_total >= 12_288 && m % 128 == 0 {
        (2, 2)
    } else if arch == "gfx1151" {
        (4, 1)
    } else {
        (2, 1)
    }
}

/// Symbol of the W4A8 `(nt, rf)` entry.
pub fn symbol_a8(nt: usize, rf: usize) -> Option<&'static str> {
    Some(match (nt, rf) {
        (1, 1) => "moe_grouped_qt44_a8_sym_gfx1201_nt1r1",
        (2, 1) => "moe_grouped_qt44_a8_sym_gfx1201_nt2r1",
        (1, 2) => "moe_grouped_qt44_a8_sym_gfx1201_nt1r2",
        (2, 2) => "moe_grouped_qt44_a8_sym_gfx1201_nt2r2",
        _ => return None,
    })
}

/// Default W4A8 `(nt, rf)` from the node sweep (`bench_moe_grouped_qt44_iu4`, R9700, 64..8192 tokens,
/// uniform and skewed routing, `raon/pfx/ev/fp8/a8sel/`): from 4096 grouped rows the two-tile x
/// two-fragment entry wins (routed gate/up 0.95x of `nt2r1` at 640 tokens, 0.83x at 1024, 0.73x at
/// 4096; one-expert shared/dense at 4096 rows 0.89-0.94x); below it (weight-stream bound) `nt2r1`
/// ties or wins. Every entry gives bit-identical output.
pub fn select_entry_a8(m_total: usize, m: usize) -> (usize, usize) {
    if m_total >= 4096 && m % 128 == 0 {
        (2, 2)
    } else {
        (2, 1)
    }
}

pub const FP8_SRC: &str = include_str!("../../../kernels/src/moe_grouped_qt44_fp8_sym.gfx12.hip");
pub const FP8_MODULE: &str = "moe_grouped_qt44_fp8_sym_gfx12";
/// Entries of the W4FP8 module (the a8 set; `nt4r1` spills the VGPR file and is not built).
pub const FP8_ENTRIES: [(usize, usize); 4] = [(1, 1), (2, 1), (1, 2), (2, 2)];
/// e4m3 activation sidecar producer (`Gpu::ensure_fp8_mmq_x`, kernel `quantize_fp8_mmq_ds128`).
pub const FP8_X_SRC: &str = include_str!("../../../kernels/src/quantize_fp8_mmq_ds128.gfx12.hip");
pub const FP8_X_MODULE: &str = "quantize_fp8_mmq_ds128_gfx12";

/// Symbol of the W4FP8 `(nt, rf)` entry.
pub fn symbol_fp8(nt: usize, rf: usize) -> Option<&'static str> {
    Some(match (nt, rf) {
        (1, 1) => "moe_grouped_qt44_fp8_sym_gfx1201_nt1r1",
        (2, 1) => "moe_grouped_qt44_fp8_sym_gfx1201_nt2r1",
        (1, 2) => "moe_grouped_qt44_fp8_sym_gfx1201_nt1r2",
        (2, 2) => "moe_grouped_qt44_fp8_sym_gfx1201_nt2r2",
        _ => return None,
    })
}

/// Default W4FP8 `(nt, rf)` from the node sweep (`bench_moe_grouped_qt44_iu4 --sites routed,shared,dense`,
/// R9700, uniform routing, 512..4096 tokens): the two-tile x two-fragment entry (needs `m % 128 == 0`)
/// wins from about 4096 grouped rows (routed 1024 tokens, 7152 rows: 863 vs 1017 us gate/up; 4096-row
/// single-expert down/dense 6-11%); below it `nt2r1` ties or wins (routed 512 tokens, 4048 rows, is a
/// tie; dense gate/up at <= 2048 rows is 16% faster with `nt2r1`).
pub fn select_entry_fp8(m_total: usize, m: usize) -> (usize, usize) {
    if m_total >= 4096 && m % 128 == 0 {
        (2, 2)
    } else {
        (2, 1)
    }
}

/// Activation sidecar family of a grouped symmetric launch.
#[derive(Clone, Copy)]
enum SymAct {
    A4,
    A8,
    Fp8,
}

/// Symbol of the `(nt, rf)` entry.
pub fn symbol(nt: usize, rf: usize) -> Option<&'static str> {
    Some(match (nt, rf) {
        (1, 1) => "moe_grouped_qt44_iu4_sym_gfx1201_nt1r1",
        (2, 1) => "moe_grouped_qt44_iu4_sym_gfx1201_nt2r1",
        (4, 1) => "moe_grouped_qt44_iu4_sym_gfx1201_nt4r1",
        (1, 2) => "moe_grouped_qt44_iu4_sym_gfx1201_nt1r2",
        (2, 2) => "moe_grouped_qt44_iu4_sym_gfx1201_nt2r2",
        _ => return None,
    })
}

/// Default `(nt, rf)` entry of a launch with `m_total` grouped rows and `m` output rows, from the
/// node bench (`bench_moe_grouped_qt44_iu4`, R9700): at >= 12288 grouped rows (about 2048 tokens at
/// top-6) the two-tile x two-fragment entry is fastest (136 TOPS at 4096 tokens), below it the
/// four-tile x one-fragment entry (weight-stream bound at 512 tokens). `rf = 2` needs `m % 128 == 0`.
pub fn select_entry(m_total: usize, m: usize) -> (usize, usize) {
    if m_total >= 12288 && m % 128 == 0 {
        (2, 2)
    } else {
        (4, 1)
    }
}

impl Gpu {
    /// Grouped symmetric qt44 expert GEMM over a `block_i4_128` sidecar (`xq`, as returned by
    /// [`Gpu::ensure_int4_mmq_x`]); `y_grouped` is F32 `[m_total, m]`. `m % (64 * rf) == 0`,
    /// `k % 256 == 0`, `m_total % 16 == 0`. `expert_offsets` is the scatter's `[E + 1]` slot
    /// prefix (16-aligned). gfx1201 only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g256v2_moe_grouped_iu4_sym(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        expert_offsets: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        xq: *mut c_void,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
        nt: usize,
        rf: usize,
    ) -> HipResult<()> {
        self.grouped_sym_launch(
            SymAct::A4,
            expert_weight_ptrs,
            expert_tile_ids,
            expert_offsets,
            sorted_slot_index,
            xq,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
            nt,
            rf,
        )
    }

    /// [`Gpu::gemm_mq4g256v2_moe_grouped_iu4_sym`] over the int8 `block_i8_128` sidecar of
    /// [`Gpu::ensure_int8_mmq_x`] (W4A8, `wmma_i32_16x16x16_iu8`): same ABI, shapes and F32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g256v2_moe_grouped_a8_sym(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        expert_offsets: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        xq: *mut c_void,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
        nt: usize,
        rf: usize,
    ) -> HipResult<()> {
        self.grouped_sym_launch(
            SymAct::A8,
            expert_weight_ptrs,
            expert_tile_ids,
            expert_offsets,
            sorted_slot_index,
            xq,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
            nt,
            rf,
        )
    }

    /// [`Gpu::gemm_mq4g256v2_moe_grouped_iu4_sym`] over the e4m3 `block_fp8_128` sidecar of
    /// [`Gpu::ensure_fp8_mmq_x`] (W4FP8, `wmma_f32_16x16x16_fp8_fp8`): same ABI, shapes and F32
    /// output as [`Gpu::gemm_mq4g256v2_moe_grouped_a8_sym`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g256v2_moe_grouped_fp8_sym(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        expert_offsets: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        xq: *mut c_void,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
        nt: usize,
        rf: usize,
    ) -> HipResult<()> {
        self.grouped_sym_launch(
            SymAct::Fp8,
            expert_weight_ptrs,
            expert_tile_ids,
            expert_offsets,
            sorted_slot_index,
            xq,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
            nt,
            rf,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_sym_launch(
        &mut self,
        act: SymAct,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        expert_offsets: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        xq: *mut c_void,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
        nt: usize,
        rf: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let err = |what: &str| hip_bridge::HipError::new(0, what);
        let gfx11 = matches!(self.arch.as_str(), "gfx1100" | "gfx1151");
        let supported = match act {
            SymAct::A8 => a8_supported(&self.arch),
            SymAct::A4 | SymAct::Fp8 => a4_fp8_supported(&self.arch),
        };
        if !supported {
            return Err(err(
                "grouped qt44 symmetric GEMM: W4A8 on gfx1201/gfx1100/gfx1151, W4A4/FP8 on gfx1201 only",
            ));
        }
        let (func, module, src) = match act {
            SymAct::A4 => (symbol(nt, rf), MODULE, SRC),
            SymAct::A8 if gfx11 => (symbol_a8_gfx11(nt, rf), A8_GFX11_MODULE, A8_GFX11_SRC),
            SymAct::A8 => (symbol_a8(nt, rf), A8_MODULE, A8_SRC),
            SymAct::Fp8 => (symbol_fp8(nt, rf), FP8_MODULE, FP8_SRC),
        };
        let func = func.ok_or_else(|| err("grouped qt44 symmetric GEMM: unbuilt (nt, rf)"))?;
        if m == 0
            || m % (64 * rf) != 0
            || k == 0
            || k % 256 != 0
            || m_total % 16 != 0
            || x_row_div == 0
        {
            return Err(err(
                "grouped qt44 symmetric GEMM: shape (M % 64*rf, K % 256, m_total % 16)",
            ));
        }
        if m_total.max(x_src_rows).max(m) > (i32::MAX as usize) / 2 {
            return Err(err(
                "grouped qt44 symmetric GEMM: extent exceeds 32-bit slot indices",
            ));
        }
        self.ensure_kernel(module, src, func)?;
        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let yp = y_grouped.buf.as_ptr();
        let (mv, kv, dv, mt, xs) = (
            m as i32,
            k as i32,
            x_row_div as i32,
            m_total as i32,
            x_src_rows as i32,
        );
        let nev = moe_expert_table_len(expert_weight_ptrs);
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
            &mt as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        let grid = [(m / (64 * rf)) as u32, (m_total / 16) as u32, 1];
        let bytes = m_total * (k / 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func, bytes);
        let result = self.launch_maybe_blob(func, grid, [128, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ep);
            b.push_ptr(tp);
            b.push_ptr(op);
            b.push_ptr(sp);
            b.push_ptr(xq);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b.push_i32(dv);
            b.push_i32(mt);
            b.push_i32(xs);
            b.push_i32(nev);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_have_symbols_and_the_default_is_built() {
        for (nt, rf) in ENTRIES {
            assert!(symbol(nt, rf).is_some(), "nt{nt}r{rf}");
        }
        assert!(symbol(8, 1).is_none());
        for (m_total, m) in [(4048, 2176), (25552, 2176), (25552, 2560), (12288, 64)] {
            let (nt, rf) = select_entry(m_total, m);
            assert!(ENTRIES.contains(&(nt, rf)));
            assert_eq!(m % (64 * rf), 0);
        }
        assert_eq!(select_entry(25552, 2176), (2, 2));
        assert_eq!(select_entry(4048, 2176), (4, 1));
        // rf = 2 needs m % 128 == 0
        assert_eq!(select_entry(25552, 192), (4, 1));
    }

    #[test]
    fn fp8_entries_have_symbols_and_the_selector_is_built() {
        for (nt, rf) in FP8_ENTRIES {
            assert!(symbol_fp8(nt, rf).is_some(), "nt{nt}r{rf}");
        }
        assert!(symbol_fp8(4, 1).is_none());
        for (m_total, m) in [
            (512, 1088),
            (4048, 2176),
            (7152, 2176),
            (25552, 2560),
            (4096, 12032),
        ] {
            let (nt, rf) = select_entry_fp8(m_total, m);
            assert!(FP8_ENTRIES.contains(&(nt, rf)));
            assert_eq!(m % (64 * rf), 0);
        }
        assert_eq!(select_entry_fp8(7152, 2176), (2, 2));
        assert_eq!(select_entry_fp8(4048, 2176), (2, 1));
        assert_eq!(select_entry_fp8(4096, 1088), (2, 1));
    }

    #[test]
    fn gfx11_a8_entries_have_symbols_and_the_default_is_built() {
        for arch in ["gfx1100", "gfx1151"] {
            assert!(a8_supported(arch) && !a4_fp8_supported(arch));
            for &(nt, rf) in a8_entries(arch) {
                assert!(symbol_a8_gfx11(nt, rf).is_some(), "{arch} nt{nt}r{rf}");
            }
            for (m_total, m) in [(4048, 2176), (25552, 2176), (25552, 2560), (12288, 64)] {
                let (nt, rf) = select_entry_a8_gfx11(arch, m_total, m);
                assert!(a8_entries(arch).contains(&(nt, rf)), "{arch} {m_total} {m}");
                assert_eq!(m % (64 * rf), 0);
            }
        }
        assert_eq!(select_entry_a8_gfx11("gfx1100", 4048, 2176), (2, 1));
        assert_eq!(select_entry_a8_gfx11("gfx1151", 4048, 2176), (4, 1));
        assert_eq!(select_entry_a8_gfx11("gfx1100", 25552, 2560), (2, 2));
        assert!(!a8_supported("gfx1030") && a8_entries("gfx942").is_empty());
    }

    #[test]
    fn a8_selector_switches_to_nt2r2_from_4096_grouped_rows() {
        for (m_total, m) in [
            (64, 2560),
            (4048, 2176),
            (4096, 12032),
            (7152, 2176),
            (25552, 2560),
        ] {
            let (nt, rf) = select_entry_a8(m_total, m);
            assert!(A8_ENTRIES.contains(&(nt, rf)));
            assert_eq!(m % (64 * rf), 0);
        }
        assert_eq!(select_entry_a8(4048, 2176), (2, 1));
        assert_eq!(select_entry_a8(7152, 2176), (2, 2));
        assert_eq!(select_entry_a8(4096, 2560), (2, 2));
        assert_eq!(select_entry_a8(4096, 1088), (2, 1));
    }
}
