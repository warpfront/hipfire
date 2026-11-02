// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 shared-expert INT8 prefill route on exact gfx1201 (`HIPFIRE_QWEN4_SHARED_INT8`, default on).
//!
//! The Flash-Next MoE shared expert (`hidden` = 2560 -> gate/up `intermediate` = 640 ->
//! SwiGLU -> down 640 -> 2560, BF16 weights) runs on the F16 WMMA route
//! (`qwen4_wmma_lds_*`, BF16 weights through F16 shadows). This route replaces its three
//! GEMMs and the activation with:
//!
//! 1. A model-lifetime INT8 shadow of the BF16 weights, built on first use by
//!    [`Gpu::qwen4_shared_i8_shadow`]: every weight row is rotated by the activation's FWHT
//!    (gate/up: `mq_rotate_x`, 256-groups; down: `mq_rotate_x_128_v2`) and quantized to the
//!    A8 `block_i8_128` format (`kernels/src/qwen4_shared_i8_quant.gfx12.hip`). The gate and
//!    up rows are interleaved in groups of 4 (plane rows `8q..8q+3` = gate channels
//!    `4q..4q+3`, `8q+4..8q+7` = up channels) so the GEMM epilogue holds both operands of the
//!    SwiGLU in one lane.
//! 2. [`Gpu::qwen4_shared_i8_gate_up`]: the existing one-pass rotate + A8 producer of the
//!    normalized activations ([`Gpu::mq6_i8_prepare_x`]) and the int8 x int8 GEMM
//!    (`qwen4_shared_i8_gemm.gfx12.hip`) with the BF16-recipe SwiGLU in its epilogue
//!    (`bf16(silu(bf16(g)) * bf16(u))`, the `shared_expert_activation_bf16_f32` expressions),
//!    storing the BF16 hidden `[tokens, intermediate]`.
//! 3. [`Gpu::qwen4_shared_i8_hidden`]: FWHT-128 + A8 quantization of that hidden (and the
//!    selector's sigmoid in place, as the fused F16 activation does).
//! 4. [`Gpu::qwen4_shared_i8_down`]: the int8 GEMM into F32 rows, which the existing BF16
//!    scaled add / HC row fold consumes unchanged.
//!
//! Numeric (KLD-gated), not byte-identical to the F16 route: int8 weights/activations with
//! per-128 scales instead of F16 operands.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// A8 `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_BYTES: usize = 136;
/// Dynamic LDS of the GEMM entries (`QWEN4_SHARED_I8_LDS_BYTES` of the kernel header).
const GEMM_LDS_BYTES: u32 = 36864;
/// CTA tile (128 rows x 128 tokens, 256 threads).
const TILE: usize = 128;

const GEMM_MODULE: &str = "qwen4_shared_i8_gemm_gfx12";
const GEMM_ENTRY_GATE_UP: &str = "qwen4_shared_i8_gate_up_silu_bf16";
const GEMM_ENTRY_DOWN: &str = "qwen4_shared_i8_down_set_f32";
const QUANT_MODULE: &str = "qwen4_shared_i8_quant_gfx12";
const QUANT_ENTRY_ROT256: &str = "qwen4_shared_rot256_i8_w";
const QUANT_ENTRY_ROT128: &str = "qwen4_shared_rot128_i8";

/// `HIPFIRE_QWEN4_SHARED_INT8` (default on, `0` restores the F16 WMMA route): the shared
/// expert's gate/up/down on the INT8 route (exact gfx1201, eager,
/// `route_rows >= QWEN4_F16_WMMA_MIN_TOKENS`).
static QWEN4_SHARED_INT8: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_SHARED_INT8", true));

/// The model-lifetime INT8 shadows of one layer's shared expert: A8 planes
/// `[K/128][rows]` of 136-byte blocks (stored as F32 words: 34 per block).
pub(crate) struct SharedI8Shadow {
    /// Interleaved gate/up plane, `[hidden/128][2 * intermediate]`.
    pub gate_up: GpuTensor,
    /// Down plane, `[intermediate/128][down_rows]`.
    pub down: GpuTensor,
    pub intermediate: usize,
    pub hidden: usize,
    pub down_rows: usize,
}

/// The BF16 shared-expert weights one call reads (`gate`/`up` `[intermediate, hidden]`,
/// `down` `[down_rows, intermediate]`).
#[derive(Clone, Copy)]
pub struct SharedI8Weights<'a> {
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub down: &'a GpuTensor,
    pub intermediate: usize,
    pub hidden: usize,
    pub down_rows: usize,
}

/// Device bytes of one layer's INT8 shadows (`gate_up` + `down` planes) for a shared expert of
/// `intermediate` x `hidden` gate/up and `down_rows` x `intermediate` down; `None` when the
/// route cannot take the shape (`hidden % 256`, `intermediate % 128`) or the bytes overflow.
pub fn shared_i8_shadow_layer_bytes(
    intermediate: usize,
    hidden: usize,
    down_rows: usize,
) -> Option<u64> {
    let (gate_up, down) = shared_i8_shadow_plane_bytes(intermediate, hidden, down_rows)?;
    gate_up.checked_add(down)
}

/// `(gate_up plane bytes, down plane bytes)` of one layer's INT8 shadows, with the same
/// admissibility checks as [`shared_i8_shadow_layer_bytes`].
pub fn shared_i8_shadow_plane_bytes(
    intermediate: usize,
    hidden: usize,
    down_rows: usize,
) -> Option<(u64, u64)> {
    if hidden == 0 || hidden % 256 != 0 || intermediate == 0 || intermediate % 128 != 0 {
        return None;
    }
    let gate_up =
        u64::try_from(plane_words(hidden, intermediate.checked_mul(2)?).checked_mul(4)?).ok()?;
    let down = u64::try_from(plane_words(intermediate, down_rows).checked_mul(4)?).ok()?;
    Some((gate_up, down))
}

/// The flag half of the route's admission, shared with the Qwen4 VRAM census: the INT8 route
/// exists only on exact gfx1201 with `HIPFIRE_QWEN4_SHARED_INT8` not `0`.
pub fn shared_i8_route_enabled(flag: bool, arch: &str) -> bool {
    flag && arch == "gfx1201"
}

fn plane_words(k: usize, rows: usize) -> usize {
    (k / 128) * rows * (BLOCK_BYTES / 4)
}

impl Gpu {
    /// Whether this device and process can ever run the shared-expert INT8 route
    /// ([`shared_i8_route_enabled`] on `HIPFIRE_QWEN4_SHARED_INT8`): the census charge's gate.
    pub fn qwen4_shared_i8_enabled(&self) -> bool {
        shared_i8_route_enabled(*QWEN4_SHARED_INT8, &self.arch)
    }

    /// Whether the shared-expert INT8 route admits a call of `batch_size` rows on weights
    /// `w`: the flag (`HIPFIRE_QWEN4_SHARED_INT8`), exact gfx1201, eager, BF16 weights with
    /// 16-byte aligned bases, `hidden % 256 == 0`, `intermediate % 128 == 0` (the gate/up
    /// plane's `2 * intermediate` rows tile by 128), and the F16 WMMA route's row floor.
    pub fn qwen4_shared_i8_applies(&self, w: &SharedI8Weights<'_>, batch_size: usize) -> bool {
        let (intermediate, hidden) = (w.intermediate, w.hidden);
        *QWEN4_SHARED_INT8
            && self.arch == "gfx1201"
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch_size > 0
            && hidden > 0
            && hidden % 256 == 0
            && intermediate > 0
            && intermediate % 128 == 0
            && w.down_rows > 0
            && Self::shared_i8_check_weights(w).is_ok()
            && [w.gate, w.up, w.down]
                .iter()
                .all(|t| (t.buf.as_ptr() as usize) % 16 == 0)
            && (2 * intermediate).max(w.down_rows) * (hidden.max(intermediate) / 128) * BLOCK_BYTES
                < (1usize << 31)
            && self.route_rows(batch_size) >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
    }

    fn shared_i8_check_weights(w: &SharedI8Weights<'_>) -> HipResult<()> {
        let err = |m: &str| Err(HipError::new(0, &format!("qwen4_shared_i8: {m}")));
        if w.gate.dtype != DType::BF16 || w.up.dtype != DType::BF16 || w.down.dtype != DType::BF16 {
            return err("shared gate/up/down must be BF16");
        }
        if w.gate.numel() < w.intermediate * w.hidden
            || w.up.numel() < w.intermediate * w.hidden
            || w.down.numel() < w.down_rows * w.intermediate
        {
            return err("undersized shared weights");
        }
        if w.hidden % 256 != 0 || w.intermediate % 128 != 0 || w.intermediate == 0 {
            return err("hidden % 256 and intermediate % 128 required");
        }
        Ok(())
    }

    /// Build (first use) and return the INT8 shadows of one layer's shared expert, keyed on
    /// the gate weight's device pointer (weights are immutable after load; released by
    /// [`Gpu::invalidate_weight_caches`]).
    pub(crate) fn qwen4_shared_i8_shadow(
        &mut self,
        w: &SharedI8Weights<'_>,
    ) -> HipResult<(*mut c_void, *mut c_void)> {
        self.bind_thread()?;
        Self::shared_i8_check_weights(w)?;
        let key = w.gate.buf.as_ptr() as usize;
        if let Some(state) = self.shadow_borrow.as_ref() {
            let (gu, dn) = state
                .i8_lookup(key, w.intermediate, w.hidden, w.down_rows)
                .map_err(|m| HipError::new(0, &format!("qwen4_shared_i8: {m}")))?;
            return Ok((gu as *mut c_void, dn as *mut c_void));
        }
        if let Some(s) = self.shared_i8_shadows.get(&key) {
            if s.intermediate != w.intermediate
                || s.hidden != w.hidden
                || s.down_rows != w.down_rows
            {
                return Err(HipError::new(
                    0,
                    "qwen4_shared_i8: cached shadow has a different shape",
                ));
            }
            return Ok((s.gate_up.buf.as_ptr(), s.down.buf.as_ptr()));
        }
        let (i, k, m_down) = (w.intermediate, w.hidden, w.down_rows);
        let gate_up = self.alloc_tensor(&[plane_words(k, 2 * i)], DType::F32)?;
        let down = self.alloc_tensor(&[plane_words(i, m_down)], DType::F32)?;
        let gu_ptr = gate_up.buf.as_ptr();
        let dn_ptr = down.buf.as_ptr();
        self.shared_i8_build_planes(
            w.gate.buf.as_ptr(),
            w.up.buf.as_ptr(),
            w.down.buf.as_ptr(),
            i,
            k,
            m_down,
            gu_ptr,
            dn_ptr,
        )?;
        self.shared_i8_shadows.insert(
            key,
            SharedI8Shadow {
                gate_up,
                down,
                intermediate: i,
                hidden: k,
                down_rows: m_down,
            },
        );
        Ok((gu_ptr, dn_ptr))
    }

    /// Quantize one layer's BF16 shared-expert weights into the INT8 planes at
    /// `gu_dst` / `dn_dst` (caller-owned; sized by [`shared_i8_shadow_plane_bytes`]) on the
    /// current stream. Shared by the lazy builder and the borrowed-region rebuild.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shared_i8_build_planes(
        &mut self,
        gate: *mut c_void,
        up: *mut c_void,
        down: *mut c_void,
        intermediate: usize,
        hidden: usize,
        down_rows: usize,
        gu_dst: *mut c_void,
        dn_dst: *mut c_void,
    ) -> HipResult<()> {
        let (i, k, m_down) = (intermediate, hidden, down_rows);
        self.ensure_kernel(
            QUANT_MODULE,
            kernels::QWEN4_SHARED_I8_QUANT_GFX12_SRC,
            QUANT_ENTRY_ROT256,
        )?;
        self.ensure_kernel(
            QUANT_MODULE,
            kernels::QWEN4_SHARED_I8_QUANT_GFX12_SRC,
            QUANT_ENTRY_ROT128,
        )?;
        self.ensure_mq_signs()?;
        self.ensure_mq_signs_128()?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        for (weight, mode) in [(gate, 1i32), (up, 2i32)] {
            self.shared_i8_rot256_launch(weight, gu_dst, s1, s2, k, i, 2 * i, mode)?;
        }
        let t1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let t2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        self.shared_i8_rot128_launch(down, dn_dst, t1, t2, std::ptr::null_mut(), i, m_down)
    }

    #[allow(clippy::too_many_arguments)]
    fn shared_i8_rot256_launch(
        &mut self,
        wp: *mut c_void,
        yp: *mut c_void,
        s1: *mut c_void,
        s2: *mut c_void,
        k: usize,
        rows: usize,
        dst_rows: usize,
        mode: i32,
    ) -> HipResult<()> {
        let (kv, rv, dv) = (k as i32, rows as i32, dst_rows as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
            &mode as *const _ as *mut c_void,
        ];
        let bytes = rows * k * 2 + (k / 128) * rows * BLOCK_BYTES;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", QUANT_ENTRY_ROT256, bytes);
        let result = self.launch_maybe_blob(
            QUANT_ENTRY_ROT256,
            [((k / 256) * rows) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(yp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b.push_i32(rv);
                b.push_i32(dv);
                b.push_i32(mode);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn shared_i8_rot128_launch(
        &mut self,
        hp: *mut c_void,
        yp: *mut c_void,
        s1: *mut c_void,
        s2: *mut c_void,
        selector: *mut c_void,
        k: usize,
        rows: usize,
    ) -> HipResult<()> {
        let (kv, rv) = (k as i32, rows as i32);
        let mut params: Vec<*mut c_void> = vec![
            &hp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &selector as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
        ];
        let bytes = rows * k * 2 + (k / 128) * rows * BLOCK_BYTES;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", QUANT_ENTRY_ROT128, bytes);
        let result = self.launch_maybe_blob(
            QUANT_ENTRY_ROT128,
            [((k / 128) * rows) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(yp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(selector);
                b.push_i32(kv);
                b.push_i32(rv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn shared_i8_gemm_launch(
        &mut self,
        entry: &'static str,
        wq: *mut c_void,
        xq: *mut c_void,
        yp: *mut c_void,
        m: usize,
        k: usize,
        n: usize,
        out_bytes: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(GEMM_MODULE, kernels::QWEN4_SHARED_I8_GEMM_GFX12_SRC, entry)?;
        let (mv, kv, nv) = (m as i32, k as i32, n as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wq as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
        ];
        let bytes = (k / 128) * (m + n) * BLOCK_BYTES + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", entry, bytes);
        let result = self.launch_maybe_blob(
            entry,
            [m.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32, 1],
            [256, 1, 1],
            GEMM_LDS_BYTES,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wq);
                b.push_ptr(xq);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(yp);
        result
    }

    /// Normalized-activation producer and gate/up GEMM: `x` (F32 `[batch_size, hidden]`, the
    /// shared expert's normalized input) -> `hidden_bf16` (`[batch_size, intermediate]` BF16
    /// bits, `bf16(silu(bf16(gate)) * bf16(up))`). `hidden_bf16` must be 8-byte aligned and
    /// hold `batch_size * intermediate * 2` bytes. Gate with
    /// [`Gpu::qwen4_shared_i8_applies`].
    pub fn qwen4_shared_i8_gate_up(
        &mut self,
        w: &SharedI8Weights<'_>,
        x: &GpuTensor,
        hidden_bf16: &GpuTensor,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.qwen4_shared_i8_applies(w, batch_size) {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_gate_up: route does not apply",
            ));
        }
        let xq = self.mq6_i8_prepare_x(x, w.hidden, batch_size)?;
        self.qwen4_shared_i8_gate_up_prequant(w, xq, hidden_bf16, batch_size)
    }

    /// [`Gpu::qwen4_shared_i8_gate_up`] on the A8 blocks `xq` (`[hidden/128][batch_size]`) a
    /// producer already left in the INT8 activation scratch: [`Gpu::mq6_i8_prepare_x`]'s
    /// pointer, or the fused MoE front's ([`Gpu::qwen4_moe_front_producer_i8`], through
    /// [`Gpu::int8_mmq_prepared_ptr`]).
    pub fn qwen4_shared_i8_gate_up_prequant(
        &mut self,
        w: &SharedI8Weights<'_>,
        xq: *mut c_void,
        hidden_bf16: &GpuTensor,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.qwen4_shared_i8_applies(w, batch_size) || xq.is_null() {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_gate_up_prequant: route does not apply",
            ));
        }
        let out_bytes = batch_size * w.intermediate * 2;
        if hidden_bf16.buf.size() < out_bytes || (hidden_bf16.buf.as_ptr() as usize) % 8 != 0 {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_gate_up: undersized or unaligned hidden",
            ));
        }
        let (gu, _) = self.qwen4_shared_i8_shadow(w)?;
        self.shared_i8_gemm_launch(
            GEMM_ENTRY_GATE_UP,
            gu,
            xq,
            hidden_bf16.buf.as_ptr(),
            2 * w.intermediate,
            w.hidden,
            batch_size,
            out_bytes,
        )
    }

    /// FWHT-128 + A8 quantization of the BF16 hidden of [`Gpu::qwen4_shared_i8_gate_up`]
    /// into `blocks` (`[intermediate/128][batch_size]` blocks, `batch_size * intermediate /
    /// 128 * 136` bytes), and `selector[0..batch_size]` replaced by `sigmoid(bf16(selector))`
    /// in place (the fused F16 activation's selector handling).
    pub fn qwen4_shared_i8_hidden(
        &mut self,
        w: &SharedI8Weights<'_>,
        hidden_bf16: &GpuTensor,
        blocks: &GpuTensor,
        selector: &GpuTensor,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.qwen4_shared_i8_applies(w, batch_size) {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_hidden: route does not apply",
            ));
        }
        let i = w.intermediate;
        if hidden_bf16.buf.size() < batch_size * i * 2
            || blocks.buf.size() < plane_words(i, batch_size) * 4
            || selector.dtype != DType::F32
            || selector.numel() < batch_size
            || (hidden_bf16.buf.as_ptr() as usize) % 8 != 0
        {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_hidden: undersized or unaligned operand",
            ));
        }
        self.ensure_kernel(
            QUANT_MODULE,
            kernels::QWEN4_SHARED_I8_QUANT_GFX12_SRC,
            QUANT_ENTRY_ROT128,
        )?;
        self.ensure_mq_signs_128()?;
        let t1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let t2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let bp = blocks.buf.as_ptr();
        self.shared_i8_rot128_launch(
            hidden_bf16.buf.as_ptr(),
            bp,
            t1,
            t2,
            selector.buf.as_ptr(),
            i,
            batch_size,
        )?;
        self.invalidate_x_caches_for(bp);
        self.invalidate_x_caches_for(selector.buf.as_ptr());
        Ok(())
    }

    /// Down GEMM: `blocks` of [`Gpu::qwen4_shared_i8_hidden`] -> `out` (F32
    /// `[batch_size, down_rows]`, overwritten).
    pub fn qwen4_shared_i8_down(
        &mut self,
        w: &SharedI8Weights<'_>,
        blocks: &GpuTensor,
        out: &GpuTensor,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.qwen4_shared_i8_applies(w, batch_size) {
            return Err(HipError::new(
                0,
                "qwen4_shared_i8_down: route does not apply",
            ));
        }
        if out.dtype != DType::F32
            || out.numel() < batch_size * w.down_rows
            || blocks.buf.size() < plane_words(w.intermediate, batch_size) * 4
        {
            return Err(HipError::new(0, "qwen4_shared_i8_down: undersized operand"));
        }
        let (_, dn) = self.qwen4_shared_i8_shadow(w)?;
        self.shared_i8_gemm_launch(
            GEMM_ENTRY_DOWN,
            dn,
            blocks.buf.as_ptr(),
            out.buf.as_ptr(),
            w.down_rows,
            w.intermediate,
            batch_size,
            batch_size * w.down_rows * 4,
        )
    }

    /// Diagnostic: the INT8 shadows of `w` (built if absent) as host bytes
    /// `(gate_up plane, down plane)`, for the byte oracle (`examples/shared_i8_oracle.rs`).
    pub fn qwen4_shared_i8_shadow_host(
        &mut self,
        w: &SharedI8Weights<'_>,
    ) -> HipResult<(Vec<u8>, Vec<u8>)> {
        self.qwen4_shared_i8_shadow(w)?;
        self.hip.device_synchronize()?;
        let s = &self.shared_i8_shadows[&(w.gate.buf.as_ptr() as usize)];
        let (mut gu, mut dn) = (
            vec![0u8; plane_words(w.hidden, 2 * w.intermediate) * 4],
            vec![0u8; plane_words(w.intermediate, w.down_rows) * 4],
        );
        self.hip.memcpy_dtoh(&mut gu, &s.gate_up.buf)?;
        self.hip.memcpy_dtoh(&mut dn, &s.down.buf)?;
        Ok((gu, dn))
    }

    /// Release every shared-expert INT8 shadow (called with the other weight caches).
    pub(crate) fn drain_shared_i8_shadows(&mut self) {
        let shadows: Vec<SharedI8Shadow> = self.shared_i8_shadows.drain().map(|(_, s)| s).collect();
        for s in shadows {
            let _ = self.free_tensor(s.gate_up);
            let _ = self.free_tensor(s.down);
        }
    }
}
