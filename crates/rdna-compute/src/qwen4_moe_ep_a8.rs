// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A8 (int8 per-128 activation, symmetric-IU4 weight) variant of the Qwen4 Flash-Next EP owner-routed
//! symmetric MoE prefill chain on exact gfx1201.
//!
//! The A4 chain ([`Gpu::qwen4_moe_rotate256_i4`] -> [`Gpu::gemm_qwen4_moe_gate_up_silu_iu4_sym`] ->
//! [`Gpu::qwen4_moe_rotate128_i4_gp`] -> [`Gpu::gemm_qwen4_moe_down_iu4_sym`]) quantizes both MoE
//! activations to `block_i4_128`; this module is the same chain with `block_i8_128` activations
//! (`d = RN(amax / 127)`, `q = clamp(rint(RN(x / d)), -127, 127)`) and signed-IU8 WMMA. Routing, grouping,
//! tile-id / sorted-slot contracts and the BF16 grouped outputs are those of the A4 launchers, so the
//! owner-routed combine reads the same bytes:
//!
//! * gate producer: [`Gpu::qwen4_moe_rotate256_i8`] = FWHT256 + A8 emit (`mq_rotate_x_i8_gfx12`, the
//!   `mq_rotate_x_i4.hip` body), sidecar `[K/128, rows]` in the shared int8 scratch.
//! * gate/up: [`Gpu::gemm_qwen4_moe_gate_up_silu_a8_sym`], `Y [grouped_rows, M/2]` BF16 bits of
//!   `rt(silu(rt(g)) * rt(u))`, padding slots +0, sentinel tiles untouched.
//! * down producer: [`Gpu::qwen4_moe_rotate128_i8`], grouped BF16 SwiGLU rows -> FWHT128 -> A8, flat-slot
//!   sidecar `[K/128, total_slots]`.
//! * down: [`Gpu::gemm_qwen4_moe_down_a8_sym`], `Y [grouped_rows, M]` BF16 bits of `rt(sum)`.
//!
//! Per output and ascending K128 epoch the GEMMs fold the exact int32 chain `C_h` as
//! `acc = fmaf(RN(f32(sc_h) * d_h), (float)C_h, acc)` from +0. Both weight layouts are served: the shipped
//! qt44 / qt53 groups and the F1 drop-zero planar `MQ4G256V2P` / `MQ4G128V2P` ([`Gpu::mq4_experts_planar`]).
//!
//! Developer env `HIPFIRE_QWEN4_MOE_EP_A8` (default off, `=1` on; exact gfx1201 only): when the chain runs,
//! [`Gpu::qwen4_moe_ep_a8_requested`] routes it through these kernels, any row count the chain serves.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemm::moe_expert_table_len;
use crate::kernels;
use crate::scratch::Int8MmqPrepared;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

/// Developer switch of the A8 EP chain (default off; `=1` on).
pub const QWEN4_MOE_EP_A8_FLAG: &str = "HIPFIRE_QWEN4_MOE_EP_A8";

/// Module of the gate/up and down GEMM entries.
pub const EP_A8_GEMM_MODULE: &str = "qwen4_moe_ep_a8_gfx12";
/// Module and entry of the down A8 producer.
pub const EP_A8_ROTATE128_MODULE: &str = "qwen4_moe_rotate128_i8_gfx12";
pub const EP_A8_ROTATE128_SYMBOL: &str = "qwen4_moe_rotate128_i8";

/// `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_I8_128_BYTES: usize = 136;
/// Gate/up rows per CTA: 4 waves x 16 gate rows (the up rows ride along).
const ROWS_PER_CTA: usize = 64;

static EP_A8: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool(QWEN4_MOE_EP_A8_FLAG, false));

/// Tile-run widths built for gate/up (16-slot tiles per CTA).
pub const EP_A8_GATE_UP_NTS: [usize; 2] = [1, 2];
/// Tile-run widths built for down (`nt4` spills the VGPR file and is not built).
pub const EP_A8_DOWN_NTS: [usize; 2] = [1, 2];

/// Symbol of the gate/up entry (`planar`: F1 drop-zero planar experts).
pub fn ep_a8_gate_up_symbol(planar: bool, nt: usize) -> Option<&'static str> {
    Some(match (planar, nt) {
        (false, 1) => "qwen4_moe_ep_a8_gate_up_nt1",
        (false, 2) => "qwen4_moe_ep_a8_gate_up_nt2",
        (true, 1) => "qwen4_moe_ep_a8_gate_up_planar_nt1",
        (true, 2) => "qwen4_moe_ep_a8_gate_up_planar_nt2",
        _ => return None,
    })
}

/// Symbol of the down entry (`planar`: F1 drop-zero planar experts).
pub fn ep_a8_down_symbol(planar: bool, nt: usize) -> Option<&'static str> {
    Some(match (planar, nt) {
        (false, 1) => "qwen4_moe_ep_a8_down_nt1",
        (false, 2) => "qwen4_moe_ep_a8_down_nt2",
        (true, 1) => "qwen4_moe_ep_a8_down_planar_nt1",
        (true, 2) => "qwen4_moe_ep_a8_down_planar_nt2",
        _ => return None,
    })
}

/// Tile-run width of a launch of `grouped_rows` grouped rows: one 16-slot tile per CTA below two tiles'
/// worth of work, two tiles per CTA (the weight stream is read once per pair) from there.
pub fn ep_a8_select_nt(_down: bool, grouped_rows: usize) -> usize {
    if grouped_rows >= 32 {
        2
    } else {
        1
    }
}

/// `k` the down sidecar is reserved with: the shared int8 scratch reserves whole K256 multiples, the down
/// width (`k % 128 == 0`) rounds up (the sidecar uses `k / 128` of the `k_reserved / 128` epochs).
pub fn ep_a8_down_reserve_k(k: usize) -> usize {
    k.next_multiple_of(256)
}

/// Pure launch-shape admission of the gate/up GEMM: `m` splits into gate and up halves of whole 64-row
/// CTAs, `k % 256 == 0` (qt44 groups of 256), whole 16-slot tiles.
pub fn ep_a8_gate_up_shape_ok(m: usize, k: usize, grouped_rows: usize, x_row_div: usize) -> bool {
    m != 0
        && m % (2 * ROWS_PER_CTA) == 0
        && k != 0
        && k % 256 == 0
        && grouped_rows != 0
        && grouped_rows % 16 == 0
        && x_row_div != 0
}

/// Pure launch-shape admission of the down GEMM: `m % 64 == 0`, `k % 128 == 0` (one K128 epoch per
/// qt53 group), whole 16-slot tiles.
pub fn ep_a8_down_shape_ok(m: usize, k: usize, grouped_rows: usize, x_row_div: usize) -> bool {
    m != 0
        && m % ROWS_PER_CTA == 0
        && k != 0
        && k % 128 == 0
        && grouped_rows != 0
        && grouped_rows % 16 == 0
        && x_row_div != 0
}

/// Pure form of [`Gpu::qwen4_moe_ep_a8_requested`].
fn ep_a8_requested(arch: &str, flag: bool) -> bool {
    arch == "gfx1201" && flag
}

fn err(what: &str) -> HipError {
    HipError::new(0, what)
}

fn first_launch_receipt(stage: usize, line: impl FnOnce() -> String) {
    static SEEN: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
    if !SEEN[stage].swap(true, Ordering::Relaxed) {
        eprintln!("[INFO] qwen4 moe EP A8: first launch {}", line());
    }
}

impl Gpu {
    /// Whether the EP owner-routed symmetric MoE prefill chain runs its A8 kernels
    /// (`HIPFIRE_QWEN4_MOE_EP_A8=1`, exact gfx1201, eager: never while a graph is captured or a replay
    /// recorded, where a lazy int8-scratch growth would invalidate the capture and the A4 chain's
    /// pre-reserved scratch is the one in force). Flag read once per process.
    pub fn qwen4_moe_ep_a8_requested(&self) -> bool {
        ep_a8_requested(&self.arch, *EP_A8) && !self.graphs.capture_mode && !self.replay.is_recording()
    }

    fn qwen4_moe_ep_a8_require(&self, what: &str) -> HipResult<()> {
        if self.arch != "gfx1201" {
            return Err(err(&format!("{what}: the EP A8 chain is exact-gfx1201 only")));
        }
        Ok(())
    }

    /// Gate/up A8 producer of the EP chain: [`Gpu::qwen4_moe_rotate256_i4`]'s FWHT256 with the
    /// `block_i8_128` emit, epoch-major `[k / 128, rows]` in the shared int8 scratch. F32 input only.
    pub fn qwen4_moe_rotate256_i8(
        &mut self,
        x: &GpuTensor,
        k: usize,
        rows: usize,
    ) -> HipResult<Int8MmqPrepared> {
        self.qwen4_moe_ep_a8_require("qwen4_moe_rotate256_i8")?;
        if x.dtype != DType::F32 {
            return Err(err("qwen4_moe_rotate256_i8: F32 input only"));
        }
        crate::moe::require_span(x, &[rows, k, 4], "qwen4_moe_rotate256_i8 activations")?;
        let reservation = self.reserve_int8_mmq(k, rows)?;
        self.rotate_x_mq_i8_batched(x, None, reservation, k, rows)
    }

    /// Down A8 producer of the EP chain: grouped BF16 SwiGLU rows `h [grouped_rows, k]` -> FWHT128 -> A8,
    /// the compact flat-slot sidecar `Xq[group * total_slots + sorted[p]]` ([`Gpu::qwen4_moe_rotate128_i4`]
    /// layout with `block_i8_128` records). Rows with `sorted[p] < 0` emit nothing.
    pub fn qwen4_moe_rotate128_i8(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<Int8MmqPrepared> {
        self.qwen4_moe_ep_a8_require("qwen4_moe_rotate128_i8")?;
        if k == 0 || k % 128 != 0 || grouped_rows == 0 || total_slots == 0 {
            return Err(err(
                "qwen4_moe_rotate128_i8: need k % 128 == 0 and nonzero rows",
            ));
        }
        crate::moe::require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i8 grouped BF16 rows",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i8 sorted slots",
        )?;
        if grouped_rows.max(total_slots).max(k) > (i32::MAX as usize) / 2 {
            return Err(err("qwen4_moe_rotate128_i8: extent exceeds 32-bit indices"));
        }
        let grid_x = u32::try_from((k / 128) * grouped_rows)
            .map_err(|_| err("qwen4_moe_rotate128_i8: grid exceeds u32"))?;
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_kernel(
            EP_A8_ROTATE128_MODULE,
            kernels::QWEN4_MOE_ROTATE128_I8_SRC,
            EP_A8_ROTATE128_SYMBOL,
        )?;
        let reservation = self.reserve_int8_mmq(ep_a8_down_reserve_k(k), total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let (kv, pv, lv) = (k as i32, grouped_rows as i32, total_slots as i32);
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            EP_A8_ROTATE128_SYMBOL,
            grouped_rows * (k * 2 + 4) + k / 128 * total_slots * BLOCK_I8_128_BYTES,
        );
        let result = self.launch_maybe_blob(
            EP_A8_ROTATE128_SYMBOL,
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b.push_i32(pv);
                b.push_i32(lv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(Int8MmqPrepared::from_reservation(reservation))
    }

    /// Grouped A8 gate/up with the SwiGLU epilogue: `y` `[grouped_rows, m/2]` BF16 bits of
    /// `rt(silu(rt(g)) * rt(u))`, padding rows +0, sentinel tiles untouched. `xq` is the
    /// [`Gpu::qwen4_moe_rotate256_i8`] handle of `x_src_rows` tokens; `x_row_div` = top-k. Weights are the
    /// shipped qt44 groups, or the planar `MQ4G256V2P` layout when [`Gpu::mq4_experts_planar`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qwen4_moe_gate_up_silu_a8_sym(
        &mut self,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xq: &Int8MmqPrepared,
        y: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if !ep_a8_gate_up_shape_ok(m, k, grouped_rows, x_row_div) {
            return Err(err(
                "gemm_qwen4_moe_gate_up_silu_a8_sym: need m % 128 == 0, k % 256 == 0, grouped_rows % 16 == 0",
            ));
        }
        crate::moe::require_span(
            y,
            &[grouped_rows, m / 2, 2],
            "gemm_qwen4_moe_gate_up_silu_a8_sym BF16 grouped rows",
        )?;
        let (generation, live) = self.scratch.int8_mmq_live();
        let xp = xq.checked_ptr(generation, live, k, x_src_rows)?;
        self.qwen4_moe_ep_a8_gemm(false, ptrs, tiles, sorted, xp, y, [m, k, x_row_div, grouped_rows, x_src_rows])
    }

    /// Grouped A8 down: `y` `[grouped_rows, m]` BF16 bits of `rt(sum)` per grouped row (no residual, no
    /// combine), padding rows +0, sentinel tiles untouched. `xq` is the [`Gpu::qwen4_moe_rotate128_i8`] handle
    /// of `x_src_rows` flat slots; `x_row_div` = 1. Weights are the shipped qt53 groups, or the planar
    /// `MQ4G128V2P` layout when [`Gpu::mq4_experts_planar`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qwen4_moe_down_a8_sym(
        &mut self,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xq: &Int8MmqPrepared,
        y: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if !ep_a8_down_shape_ok(m, k, grouped_rows, x_row_div) {
            return Err(err(
                "gemm_qwen4_moe_down_a8_sym: need m % 64 == 0, k % 128 == 0, grouped_rows % 16 == 0",
            ));
        }
        crate::moe::require_span(
            y,
            &[grouped_rows, m, 2],
            "gemm_qwen4_moe_down_a8_sym BF16 grouped rows",
        )?;
        let (generation, live) = self.scratch.int8_mmq_live();
        let xp = xq.checked_ptr(generation, live, ep_a8_down_reserve_k(k), x_src_rows)?;
        self.qwen4_moe_ep_a8_gemm(true, ptrs, tiles, sorted, xp, y, [m, k, x_row_div, grouped_rows, x_src_rows])
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen4_moe_ep_a8_gemm(
        &mut self,
        down: bool,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xp: *mut c_void,
        y: &GpuTensor,
        [m, k, x_row_div, grouped_rows, x_src_rows]: [usize; 5],
    ) -> HipResult<()> {
        let what = if down {
            "gemm_qwen4_moe_down_a8_sym"
        } else {
            "gemm_qwen4_moe_gate_up_silu_a8_sym"
        };
        self.qwen4_moe_ep_a8_require(what)?;
        self.bind_thread()?;
        crate::moe::require_span(sorted, &[grouped_rows, 4], what)?;
        crate::moe::require_span(tiles, &[grouped_rows / 16, 4], what)?;
        if grouped_rows.max(x_src_rows).max(m).max(k) > (i32::MAX as usize) / 2 {
            return Err(err(&format!("{what}: extent exceeds 32-bit slot indices")));
        }
        let planar = self.mq4_experts_planar;
        let nt = ep_a8_select_nt(down, grouped_rows);
        let func = if down {
            ep_a8_down_symbol(planar, nt)
        } else {
            ep_a8_gate_up_symbol(planar, nt)
        }
        .ok_or_else(|| err(&format!("{what}: unbuilt tile-run width")))?;
        self.ensure_kernel(EP_A8_GEMM_MODULE, kernels::QWEN4_MOE_EP_A8_SRC, func)?;
        let ep = ptrs.buf.as_ptr();
        let tp = tiles.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let (mv, kv, dv, mt, xs) = (
            m as i32,
            k as i32,
            x_row_div as i32,
            grouped_rows as i32,
            x_src_rows as i32,
        );
        let nev = moe_expert_table_len(ptrs);
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
            &mt as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        let rows = if down { m } else { m / 2 };
        let grid = [(rows / ROWS_PER_CTA) as u32, (grouped_rows / 16) as u32, 1];
        first_launch_receipt(usize::from(down), || {
            format!(
                "stage={} entry={func} planar={planar} m={m} k={k} grouped_rows={grouped_rows} x_src_rows={x_src_rows}",
                if down { "down" } else { "gate_up" }
            )
        });
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            func,
            grouped_rows * (k / 2 + m) + grouped_rows * k / 128 * BLOCK_I8_128_BYTES,
        );
        let result = self.launch_maybe_blob(func, grid, [128, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ep);
            b.push_ptr(tp);
            b.push_ptr(sp);
            b.push_ptr(xp);
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
    fn entries_have_symbols_and_the_selector_picks_built_widths() {
        for planar in [false, true] {
            for nt in EP_A8_GATE_UP_NTS {
                let s = ep_a8_gate_up_symbol(planar, nt).unwrap();
                assert!(kernels::QWEN4_MOE_EP_A8_SRC.contains(s), "{s}");
                assert_eq!(s.contains("planar"), planar);
            }
            for nt in EP_A8_DOWN_NTS {
                let s = ep_a8_down_symbol(planar, nt).unwrap();
                assert!(kernels::QWEN4_MOE_EP_A8_SRC.contains(s), "{s}");
            }
            assert!(ep_a8_gate_up_symbol(planar, 4).is_none());
            assert!(ep_a8_down_symbol(planar, 4).is_none());
        }
        for down in [false, true] {
            for rows in [16, 32, 48, 64, 4096, 20480] {
                let nt = ep_a8_select_nt(down, rows);
                let built = if down { EP_A8_DOWN_NTS } else { EP_A8_GATE_UP_NTS };
                assert!(built.contains(&nt), "{down} {rows}");
            }
        }
        assert_eq!(ep_a8_select_nt(false, 16), 1);
        assert_eq!(ep_a8_select_nt(false, 32), 2);
    }

    #[test]
    fn producer_source_has_its_entry() {
        assert!(kernels::QWEN4_MOE_ROTATE128_I8_SRC.contains(EP_A8_ROTATE128_SYMBOL));
        assert!(kernels::QWEN4_MOE_ROTATE128_I8_SRC.contains("quantize_block_i8_128_wave"));
    }

    #[test]
    fn down_sidecar_reserve_rounds_to_k256_and_covers_the_epochs() {
        for k in [128usize, 640, 1280, 2560] {
            let kr = ep_a8_down_reserve_k(k);
            assert_eq!(kr % 256, 0);
            assert!(kr >= k && kr - k < 256);
            assert!(kr / 128 >= k / 128);
        }
        assert_eq!(ep_a8_down_reserve_k(640), 768);
    }

    #[test]
    fn shape_admission_matches_the_flash_next_geometry() {
        // Flash-Next EP: gate/up M = 2 * 640, K = 2560; down M = 2560, K = 640.
        assert!(ep_a8_gate_up_shape_ok(1280, 2560, 4096, 10));
        assert!(ep_a8_down_shape_ok(2560, 640, 4096, 1));
        assert!(!ep_a8_gate_up_shape_ok(1088, 2560, 4096, 10)); // M/2 not a whole 64-row CTA
        assert!(!ep_a8_gate_up_shape_ok(1280, 2688, 4096, 10)); // K % 256
        assert!(!ep_a8_gate_up_shape_ok(1280, 2560, 4090, 10)); // grouped_rows % 16
        assert!(!ep_a8_down_shape_ok(2560, 640, 0, 1));
        assert!(!ep_a8_down_shape_ok(2560, 600, 4096, 1));
        assert!(!ep_a8_down_shape_ok(2560, 640, 4096, 0));
    }

    #[test]
    fn requested_needs_exact_gfx1201_and_the_flag() {
        assert!(ep_a8_requested("gfx1201", true));
        assert!(!ep_a8_requested("gfx1201", false));
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", ""] {
            assert!(!ep_a8_requested(arch, true), "{arch}");
        }
        assert_eq!(QWEN4_MOE_EP_A8_FLAG, "HIPFIRE_QWEN4_MOE_EP_A8");
    }
}
