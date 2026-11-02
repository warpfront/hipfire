// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ8G256 (qt14) prefill GEMM for gfx12 (gfx1200/gfx1201) and gfx11 (gfx1100/gfx1151).
//!
//! `Y[n, m] = sum_e fmaf(RN(sw[m, e/2] * dx[n, e/2]), (float)dot128_e(m, n), acc)` on int8
//! WMMA/MMQ tiles (`kernels/src/gemm_mq8g256_mmq_i8.gfx12.hip`, `.gfx11.hip`), with a batched
//! FWHT256 activation producer (`kernels/src/mq8_rotate_quantize_x_batched.hip`).
//!
//! # Layouts
//!
//! * **Weights, layout A** ([`Mq8WLayout::A`]): the artifact's qt=14 rows read in place.
//!   `m` rows of `k/256` groups of 258 B `{f16 scale, int8 q[256]}` (row stride `(k/256)*258`),
//!   [`mq8g256_a_bytes`] bytes, 2-byte aligned base.
//! * **Weights, layout B** ([`Mq8WLayout::B`]): the aligned repack ([`mq8g256_repack_b`]), the
//!   same VRAM: an f16 scale plane `[m, k/256]` (2 B each, row-major), zero padding to the next
//!   256 B boundary ([`mq8g256_b_code_offset`]), then the int8 code plane `[m, k]` row-major;
//!   [`mq8g256_b_bytes`] bytes, 16-byte aligned base.
//! * **Activations**: [`Gpu::mq8_quantize_x`] writes int8 codes `[n, k]` row-major and F32 scales
//!   `[k/256, n]` (`Xs[group * n + token]`) into the Gpu-owned `mq8_x_scratch`; the byte layout
//!   is `scratch::mq8_x_layout(k, n)` = `(scales_off, total)` with
//!   `scales_off = round_up(n*k, 256)`, `total = scales_off + n*(k/256)*4`. The scratch
//!   generation is bumped by every producer launch, so [`Gpu::mq8_acts_current`] tells a prepared
//!   [`Mq8Activations`] whether a later quantizer overwrote it; the GEMM refuses stale ones.
//! * **Output**: F32 `[n, m]`; the `_bf16` variants round every output to BF16 (RN-even,
//!   non-finite passthrough) and widen it back to F32.
//!
//! # Numerics (the CPU reference contract)
//!
//! Per output `(n, m)`: `acc = 0.0f32; for e in 0..k/128 (ascending): dot = sum_{i<128}
//! (i32)W8[m, 128e+i] * (i32)X8[n, 128e+i]; scale = RN(sw[m, e/2] * dx[n, e/2]);
//! acc = fmaf(scale, (float)dot, acc)`, `sw` = f16 -> f32 of the group scale, `dx` =
//! `Xs[(e/2)*n + token]`. `|dot128| <= 2^21`, so the integer dot is exact. Every tile and both
//! layouts are bit-identical (the tile menu is a performance choice only). The producer matches
//! `mq8_rotate_quantize_x` (sign-FWHT256 with the seed 42 / 1042 tables, amax/127, `rintf`,
//! clamp [-128, 127]) per token row; a zero group has scale 1.0 and codes 0.
//!
//! # Admission
//!
//! gfx12: exact RDNA4 (`arch_caps.is_rdna4()`); gfx11: arch gfx1100 or gfx1151. `k > 0`,
//! `k % 256 == 0`, `m, n > 0`, F32 `x` / `y`, producer `n*(k/256) < 2^31`, grid-y `ceil(n / BN) <= 65535`. See
//! [`mq8_admits`]. No environment variable. The weight tensor's dtype is not checked, like the
//! `gemv_mq8g256` launchers (a qt14 tensor, or a raw byte tensor holding layout B).
//!
//! # Scratch sizes
//!
//! `mq8_x_scratch` holds `n*k + pad + n*(k/256)*4` bytes (e.g. 4096 x 5120: 20 971 520 +
//! 327 680 = 21 299 200 B); it only grows, and growth invalidates captured graphs first.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemm_q8_a8::{q8_a8_pick_tile_pure, Q8A8Tile};
use crate::gemm_q8_mmq_gfx11::{q8_mmq_pick_tile, Q8MmqTile};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const PRODUCER_MODULE: &str = "mq8_rotate_quantize_x_batched";
const PRODUCER_SYMBOL: &str = "mq8_rotate_quantize_x_batched";
const GFX12_MODULE: &str = "gemm_mq8g256_mmq_i8_gfx12";
const GFX11_MODULE: &str = "gemm_mq8g256_mmq_i8_gfx11";
/// Bytes of one qt14 group: f16 scale + 256 int8 codes.
const GROUP_BYTES: usize = 258;
/// Weights per group.
const GROUP: usize = 256;
/// Grid y dimension limit of a launch.
const MAX_GRID_Y: usize = 65535;

/// On-GPU weight layout of an MQ8G256 tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mq8WLayout {
    /// qt14 258 B groups read in place.
    A,
    /// Aligned repack: f16 scale plane, then the row-major int8 code plane.
    B,
}

/// Output tile of the MQ8G256 GEMMs (`BM x BN` = weight rows x tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mq8Tile {
    T128x128,
    /// gfx12 only.
    T128x64,
    /// gfx11 only.
    T64x128,
    T64x64,
}

/// Kernel family of an admitted arch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mq8Family {
    Gfx12,
    Gfx11,
}

fn mq8_family(arch: &str) -> Option<Mq8Family> {
    match arch {
        "gfx1200" | "gfx1201" => Some(Mq8Family::Gfx12),
        "gfx1100" | "gfx1151" => Some(Mq8Family::Gfx11),
        _ => None,
    }
}

impl Mq8Tile {
    /// `(BM, BN)`.
    pub const fn dims(self) -> (usize, usize) {
        match self {
            Mq8Tile::T128x128 => (128, 128),
            Mq8Tile::T128x64 => (128, 64),
            Mq8Tile::T64x128 => (64, 128),
            Mq8Tile::T64x64 => (64, 64),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mq8Tile::T128x128 => "128x128",
            Mq8Tile::T128x64 => "128x64",
            Mq8Tile::T64x128 => "64x128",
            Mq8Tile::T64x64 => "64x64",
        }
    }

    /// Whether the kernel family of `arch` has this tile (`false` for an arch without MQ8 kernels).
    pub fn valid_on(self, arch: &str) -> bool {
        match (mq8_family(arch), self) {
            (Some(_), Mq8Tile::T128x128 | Mq8Tile::T64x64) => true,
            (Some(Mq8Family::Gfx12), Mq8Tile::T128x64) => true,
            (Some(Mq8Family::Gfx11), Mq8Tile::T64x128) => true,
            _ => false,
        }
    }

    /// Entry symbol of `arch` (`None` when the tile does not exist there).
    pub fn symbol(self, arch: &str, layout: Mq8WLayout, bf16_out: bool) -> Option<&'static str> {
        use Mq8WLayout::{A, B};
        let fam = mq8_family(arch)?;
        if !self.valid_on(arch) {
            return None;
        }
        Some(match (fam, self, layout, bf16_out) {
            (Mq8Family::Gfx12, Mq8Tile::T128x128, A, false) => "gemm_mq8g256_mmq_i8_t128x128_wa",
            (Mq8Family::Gfx12, Mq8Tile::T128x128, A, true) => {
                "gemm_mq8g256_mmq_i8_t128x128_wa_bf16"
            }
            (Mq8Family::Gfx12, Mq8Tile::T128x128, B, false) => "gemm_mq8g256_mmq_i8_t128x128_wb",
            (Mq8Family::Gfx12, Mq8Tile::T128x128, B, true) => {
                "gemm_mq8g256_mmq_i8_t128x128_wb_bf16"
            }
            (Mq8Family::Gfx12, Mq8Tile::T128x64, A, false) => "gemm_mq8g256_mmq_i8_t128x64_wa",
            (Mq8Family::Gfx12, Mq8Tile::T128x64, A, true) => "gemm_mq8g256_mmq_i8_t128x64_wa_bf16",
            (Mq8Family::Gfx12, Mq8Tile::T128x64, B, false) => "gemm_mq8g256_mmq_i8_t128x64_wb",
            (Mq8Family::Gfx12, Mq8Tile::T128x64, B, true) => "gemm_mq8g256_mmq_i8_t128x64_wb_bf16",
            (Mq8Family::Gfx12, Mq8Tile::T64x64, A, false) => "gemm_mq8g256_mmq_i8_t64x64_wa",
            (Mq8Family::Gfx12, Mq8Tile::T64x64, A, true) => "gemm_mq8g256_mmq_i8_t64x64_wa_bf16",
            (Mq8Family::Gfx12, Mq8Tile::T64x64, B, false) => "gemm_mq8g256_mmq_i8_t64x64_wb",
            (Mq8Family::Gfx12, Mq8Tile::T64x64, B, true) => "gemm_mq8g256_mmq_i8_t64x64_wb_bf16",
            (Mq8Family::Gfx11, Mq8Tile::T128x128, A, false) => {
                "gemm_mq8g256_mmq_i8_gfx11_t128x128_wa"
            }
            (Mq8Family::Gfx11, Mq8Tile::T128x128, A, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t128x128_wa_bf16"
            }
            (Mq8Family::Gfx11, Mq8Tile::T128x128, B, false) => {
                "gemm_mq8g256_mmq_i8_gfx11_t128x128_wb"
            }
            (Mq8Family::Gfx11, Mq8Tile::T128x128, B, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t128x128_wb_bf16"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x128, A, false) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x128_wa"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x128, A, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x128_wa_bf16"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x128, B, false) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x128_wb"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x128, B, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x128_wb_bf16"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x64, A, false) => "gemm_mq8g256_mmq_i8_gfx11_t64x64_wa",
            (Mq8Family::Gfx11, Mq8Tile::T64x64, A, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x64_wa_bf16"
            }
            (Mq8Family::Gfx11, Mq8Tile::T64x64, B, false) => "gemm_mq8g256_mmq_i8_gfx11_t64x64_wb",
            (Mq8Family::Gfx11, Mq8Tile::T64x64, B, true) => {
                "gemm_mq8g256_mmq_i8_gfx11_t64x64_wb_bf16"
            }
            // `valid_on` returned true above, so these combinations are unreachable.
            (Mq8Family::Gfx12, Mq8Tile::T64x128, ..) | (Mq8Family::Gfx11, Mq8Tile::T128x64, ..) => {
                return None
            }
        })
    }

    /// The tile of the same name in the gfx12 Q8_0 W8A8 menu.
    fn from_q8_a8(tile: Q8A8Tile) -> Self {
        match tile {
            Q8A8Tile::T128x128 => Mq8Tile::T128x128,
            Q8A8Tile::T128x64 => Mq8Tile::T128x64,
            Q8A8Tile::T64x64 => Mq8Tile::T64x64,
        }
    }

    /// The tile of the same name in the gfx11 Q8_0 W8A8 menu.
    fn from_q8_mmq(tile: Q8MmqTile) -> Self {
        match tile {
            Q8MmqTile::T128x128 => Mq8Tile::T128x128,
            Q8MmqTile::T64x128 => Mq8Tile::T64x128,
            Q8MmqTile::T64x64 => Mq8Tile::T64x64,
        }
    }
}

/// Shape/arch half of the MQ8G256 prefill admission (pure): gfx12 (gfx1200/gfx1201) or gfx11
/// (gfx1100/gfx1151), `k > 0 && k % 256 == 0`. [`Gpu::mq8_prefill_supported`] is this on the
/// device's arch.
pub fn mq8_admits(arch: &str, k: usize) -> bool {
    mq8_family(arch).is_some() && k > 0 && k % GROUP == 0
}

/// Layout A bytes of an `[m, k]` weight: `m * (k/256) * 258`.
pub fn mq8g256_a_bytes(m: usize, k: usize) -> usize {
    m * (k / GROUP) * GROUP_BYTES
}

/// Layout B byte offset of the int8 code plane: `round_up(m * (k/256) * 2, 256)`.
pub fn mq8g256_b_code_offset(m: usize, k: usize) -> usize {
    (m * (k / GROUP) * 2).next_multiple_of(256)
}

/// Layout B bytes of an `[m, k]` weight: `code_off + m * k`.
pub fn mq8g256_b_bytes(m: usize, k: usize) -> usize {
    mq8g256_b_code_offset(m, k) + m * k
}

/// CPU repack of an `[m, k]` layout A weight (qt14 258 B groups) into layout B: the f16 scale
/// bits are copied verbatim into the scale plane `[m, k/256]`, the codes into the row-major code
/// plane `[m, k]` at [`mq8g256_b_code_offset`], padding zero. Panics unless `k % 256 == 0` and
/// `a.len() >= mq8g256_a_bytes(m, k)`.
pub fn mq8g256_repack_b(a: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(
        k % GROUP == 0,
        "mq8g256_repack_b: k {k} is not a multiple of 256"
    );
    assert!(
        a.len() >= mq8g256_a_bytes(m, k),
        "mq8g256_repack_b: {} source bytes < m*k/256*258 = {}",
        a.len(),
        mq8g256_a_bytes(m, k)
    );
    let groups_per_row = k / GROUP;
    let code_off = mq8g256_b_code_offset(m, k);
    let mut out = vec![0u8; mq8g256_b_bytes(m, k)];
    for r in 0..m {
        for g in 0..groups_per_row {
            let gi = r * groups_per_row + g;
            let src = &a[gi * GROUP_BYTES..(gi + 1) * GROUP_BYTES];
            out[gi * 2..gi * 2 + 2].copy_from_slice(&src[..2]);
            let dst = code_off + r * k + g * GROUP;
            out[dst..dst + GROUP].copy_from_slice(&src[2..]);
        }
    }
    out
}

/// Tile for a `(m, n)` call on `arch` (pure; performance only, all tiles are bit-identical).
/// gfx12 follows the Q8_0 W8A8 picker (`Gpu::q8_a8_pick_tile`), gfx11 [`q8_mmq_pick_tile`].
/// An arch without MQ8 kernels gets the gfx12 rule (the caller's admission refuses it).
pub fn mq8_pick_tile(arch: &str, m: usize, n: usize) -> Mq8Tile {
    match mq8_family(arch) {
        Some(Mq8Family::Gfx11) => Mq8Tile::from_q8_mmq(q8_mmq_pick_tile(m, n)),
        _ => Mq8Tile::from_q8_a8(q8_a8_pick_tile_pure(m, n)),
    }
}

/// MQ8G256-quantized activations: the int8 codes `[n, k]` and the F32 scales `[k/256, n]` of the batched
/// producer. Either in the Gpu's `mq8_x_scratch` ([`Gpu::mq8_quantize_x`]; the pointers stay valid until the
/// next quantizer launch or a scratch growth, and [`Gpu::mq8_acts_current`] tells whether that happened) or in
/// caller-owned planes ([`Gpu::mq8_quantize_x_into`]; never stale). The prefill GEMMs, the decode GEMV and
/// the verify rows-exact GEMVs all consume this one type.
#[derive(Clone, Copy, Debug)]
pub struct Mq8Activations {
    pub(crate) xq: *mut c_void,
    pub(crate) xs: *mut c_void,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) generation: u64,
}

/// `generation` of activations that live in caller planes (no scratch to overwrite).
const MQ8_ACTS_EXTERNAL: u64 = u64::MAX;

impl Mq8Activations {
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }
    /// Device pointer of the int8 codes `[n, k]` (row-major).
    pub fn xq_ptr(&self) -> *mut c_void {
        self.xq
    }
    /// Device pointer of the F32 scales `[k/256, n]` (`Xs[group * n + token]`).
    pub fn xs_ptr(&self) -> *mut c_void {
        self.xs
    }
}

impl Gpu {
    /// Whether the MQ8G256 prefill route can run for this device and `k` (see [`mq8_admits`]).
    pub fn mq8_prefill_supported(&self, k: usize) -> bool {
        let arch_ok = self.arch_caps.is_rdna4() || mq8_family(&self.arch) == Some(Mq8Family::Gfx11);
        arch_ok && k > 0 && k % GROUP == 0
    }

    /// Whether `acts` still describes its planes (no scratch quantizer ran since; caller planes never go stale).
    /// Only scratch planes ([`Self::mq8_quantize_x`]) carry a generation; planes from
    /// [`Self::mq8_quantize_x_into`] / [`Self::mq8_acts_from_planes`] are `MQ8_ACTS_EXTERNAL` and always
    /// count as current, so their owner alone guarantees they still hold the intended `x` (never hand
    /// such planes to a consumer that relies on this check to detect staleness).
    pub fn mq8_acts_current(&self, acts: &Mq8Activations) -> bool {
        acts.generation == MQ8_ACTS_EXTERNAL || acts.generation == self.scratch.mq8_x_generation
    }

    /// Grow the MQ8 activation scratch to hold `(k, n)` now (eager, no launch), so a later
    /// [`Self::mq8_quantize_x`] of at most that size inside a graph capture or Redline recording
    /// never has to grow it (which [`Self::mq8_quantize_x`] refuses there). Call at load time with
    /// the largest captured row count and K. Advances the scratch generation (any prepared plane is
    /// stale afterwards).
    pub fn mq8_reserve_x(&mut self, k: usize, n: usize) -> HipResult<()> {
        if n == 0 || k == 0 || k % GROUP != 0 {
            return Err(HipError::new(1, "mq8_reserve_x: n > 0 and k % 256 == 0"));
        }
        self.bind_thread()?;
        let (_, needed) = crate::scratch::mq8_x_layout(k, n);
        if crate::scratch::scratch_will_grow(
            self.scratch.mq8_x_scratch_bytes,
            self.scratch.mq8_x_scratch.is_some(),
            needed,
        ) {
            if self.graphs.capture_mode || self.replay.is_recording() {
                return Err(HipError::new(
                    1,
                    "mq8_reserve_x: cannot grow the MQ8 scratch during capture/recording",
                ));
            }
            self.invalidate_for_scratch_growth();
            self.scratch.reserve_mq8_x(&self.hip, k, n)?;
        }
        Ok(())
    }

    /// Kernel family module + source of this device.
    fn mq8_gemm_module(&self) -> (&'static str, &'static str) {
        if mq8_family(&self.arch) == Some(Mq8Family::Gfx11) {
            (GFX11_MODULE, kernels::GEMM_MQ8G256_MMQ_I8_GFX11_SRC)
        } else {
            (GFX12_MODULE, kernels::GEMM_MQ8G256_MMQ_I8_GFX12_SRC)
        }
    }

    /// Rotate (sign-FWHT256) and quantize F32 `x [n, k]` (row-major) into the MQ8 scratch: int8
    /// codes `[n, k]` + F32 scales `[k/256, n]`. Ensures the sign tables. One launch for all n rows.
    pub fn mq8_quantize_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Mq8Activations> {
        let grid = self.mq8_producer_prepare("mq8_quantize_x", x, n, k)?;
        let (_, needed) = crate::scratch::mq8_x_layout(k, n);
        if crate::scratch::scratch_will_grow(
            self.scratch.mq8_x_scratch_bytes,
            self.scratch.mq8_x_scratch.is_some(),
            needed,
        ) {
            // Growing replaces the buffer that earlier captured/recorded nodes already reference:
            // refuse instead of corrupting the graph (pre-size with `mq8_reserve_x`).
            if self.graphs.capture_mode || self.replay.is_recording() {
                return Err(HipError::new(
                    1,
                    "mq8_quantize_x: the MQ8 activation scratch would grow during graph capture or \
                     Redline recording; pre-size it with Gpu::mq8_reserve_x at load",
                ));
            }
            self.invalidate_for_scratch_growth();
        }
        let (xq, xs) = self.scratch.reserve_mq8_x(&self.hip, k, n)?;
        self.mq8_producer_launch(x, xq, xs, n, k, grid)?;
        Ok(Mq8Activations {
            xq,
            xs,
            n,
            k,
            generation: self.scratch.mq8_x_generation,
        })
    }

    /// [`Self::mq8_quantize_x`] into caller-owned planes: `xq` holds at least `n * k` bytes, `xs` at least
    /// `n * (k/256)` F32 (F32 tensors; the byte plane may be any tensor with enough bytes). The returned
    /// activations reference those planes (never stale), so two X can be live at once.
    pub fn mq8_quantize_x_into(
        &mut self,
        x: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Mq8Activations> {
        let grid = self.mq8_producer_prepare("mq8_quantize_x_into", x, n, k)?;
        if xq.buf.size() < n * k
            || (xq.buf.as_ptr() as usize) % 16 != 0
            || xs.dtype != DType::F32
            || xs.buf.size() < n * (k / GROUP) * 4
        {
            return Err(HipError::new(
                1,
                "mq8_quantize_x_into: xq needs n*k bytes (16-byte aligned), xs n*(k/256) F32",
            ));
        }
        let (xq_p, xs_p) = (xq.buf.as_ptr(), xs.buf.as_ptr());
        self.mq8_producer_launch(x, xq_p, xs_p, n, k, grid)?;
        Ok(Mq8Activations {
            xq: xq_p,
            xs: xs_p,
            n,
            k,
            generation: MQ8_ACTS_EXTERNAL,
        })
    }

    /// Wrap activation planes that something other than the producer wrote (tests, replay tools) as
    /// [`Mq8Activations`]: `xq` int8 `[n, k]` (16-byte aligned, at least `n * k` bytes) and `xs` F32
    /// `[k/256, n]` (`xs[g * n + row]`). The planes are never stale.
    pub fn mq8_acts_from_planes(
        &self,
        xq: &GpuTensor,
        xs: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Mq8Activations> {
        if n == 0
            || k == 0
            || k % GROUP != 0
            || xq.buf.size() < n * k
            || (xq.buf.as_ptr() as usize) % 16 != 0
            || xs.dtype != DType::F32
            || xs.buf.size() < n * (k / GROUP) * 4
        {
            return Err(HipError::new(
                1,
                "mq8_acts_from_planes: n > 0, k % 256 == 0, xq >= n*k bytes (16-byte aligned), xs >= n*(k/256) F32",
            ));
        }
        Ok(Mq8Activations {
            xq: xq.buf.as_ptr(),
            xs: xs.buf.as_ptr(),
            n,
            k,
            generation: MQ8_ACTS_EXTERNAL,
        })
    }

    /// Shape/alignment admission of the producer; loads the producer module and the sign tables. Returns
    /// the launch grid.
    fn mq8_producer_prepare(
        &mut self,
        what: &str,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<[u32; 3]> {
        self.bind_thread()?;
        if !self.mq8_prefill_supported(k) || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "{what}: needs gfx12 or gfx1100/gfx1151, k % 256 == 0, n > 0 (arch {}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        let x_elems = n
            .checked_mul(k)
            .ok_or_else(|| HipError::new(1, &format!("{what}: n*k overflows")))?;
        if x.dtype != DType::F32
            || x.buf.size() < x_elems * 4
            || (x.buf.as_ptr() as usize) % 16 != 0
        {
            return Err(HipError::new(
                1,
                &format!("{what}: x must be 16-byte aligned F32 with at least n*k elements"),
            ));
        }
        if i32::try_from(k).is_err() || i32::try_from(n).is_err() {
            return Err(HipError::new(1, &format!("{what}: k or n exceeds i32")));
        }
        let groups = n * (k / GROUP);
        // The producer indexes (token, group) waves with 32-bit group arithmetic.
        if groups > i32::MAX as usize {
            return Err(HipError::new(
                1,
                &format!("{what}: n*(k/256) must be < 2^31 (32-bit group index of the producer)"),
            ));
        }
        let grid = [
            u32::try_from(groups.div_ceil(8)).map_err(|_| {
                HipError::new(1, &format!("{what}: n*k/256 too large for a 1-D launch"))
            })?,
            1,
            1,
        ];
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            PRODUCER_MODULE,
            kernels::MQ8_ROTATE_QUANTIZE_X_BATCHED_SRC,
            PRODUCER_SYMBOL,
        )?;
        Ok(grid)
    }

    /// The producer launch (plain; graph-capture safe).
    fn mq8_producer_launch(
        &mut self,
        x: &GpuTensor,
        xq: *mut c_void,
        xs: *mut c_void,
        n: usize,
        k: usize,
        grid: [u32; 3],
    ) -> HipResult<()> {
        let xp = x.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "quantize",
            PRODUCER_SYMBOL,
            n * k * 5 + n * (k / GROUP) * 4,
        );
        let result =
            self.launch_maybe_blob(PRODUCER_SYMBOL, grid, [256, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xq);
                b.push_ptr(xs);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ8G256 GEMM on already-quantized activations, F32 output, tile chosen by
    /// [`mq8_pick_tile`]. `Y[n, m] = X[n, k] @ W[m, k]^T`.
    pub fn gemm_mq8g256_gemm(
        &mut self,
        w: &GpuTensor,
        layout: Mq8WLayout,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = mq8_pick_tile(&self.arch, m, acts.n);
        self.gemm_mq8g256_gemm_tile_ex(w, layout, acts, y, m, tile, false)
    }

    /// [`Self::gemm_mq8g256_gemm`] whose epilogue rounds every output to BF16 (F32 plane holding
    /// BF16 values).
    pub fn gemm_mq8g256_gemm_bf16(
        &mut self,
        w: &GpuTensor,
        layout: Mq8WLayout,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = mq8_pick_tile(&self.arch, m, acts.n);
        self.gemm_mq8g256_gemm_tile_ex(w, layout, acts, y, m, tile, true)
    }

    /// Explicit tile and epilogue (`bf16_out`: round outputs to BF16); bench / tests.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq8g256_gemm_tile_ex(
        &mut self,
        w: &GpuTensor,
        layout: Mq8WLayout,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
        tile: Mq8Tile,
        bf16_out: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (n, k) = (acts.n, acts.k);
        if !self.mq8_acts_current(acts) {
            return Err(HipError::new(
                1,
                "gemm_mq8g256_gemm: activations were overwritten by a later MQ8 quantizer",
            ));
        }
        if !self.mq8_prefill_supported(k) || m == 0 || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "gemm_mq8g256_gemm: needs gfx12 or gfx1100/gfx1151, k % 256 == 0, m,n > 0 (arch {}, m {m}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        let symbol = tile.symbol(&self.arch, layout, bf16_out).ok_or_else(|| {
            HipError::new(
                1,
                &format!(
                    "gemm_mq8g256_gemm: tile {} does not exist on {}",
                    tile.label(),
                    self.arch
                ),
            )
        })?;
        let (w_bytes, w_align) = match layout {
            Mq8WLayout::A => (mq8g256_a_bytes(m, k), 2),
            Mq8WLayout::B => (mq8g256_b_bytes(m, k), 16),
        };
        let wp = w.buf.as_ptr();
        if w.buf.size() < w_bytes || (wp as usize) % w_align != 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "gemm_mq8g256_gemm: layout {layout:?} weight needs >= {w_bytes} bytes at a {w_align}-byte aligned base (have {} bytes at {:p})",
                    w.buf.size(),
                    wp
                ),
            ));
        }
        let y_bytes = n
            .checked_mul(m)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| HipError::new(1, "gemm_mq8g256_gemm: n*m overflows"))?;
        if y.dtype != DType::F32 || y.buf.size() < y_bytes {
            return Err(HipError::new(1, "gemm_mq8g256_gemm: y is not F32 [n, m]"));
        }
        if i32::try_from(m).is_err() {
            return Err(HipError::new(1, "gemm_mq8g256_gemm: m exceeds i32"));
        }
        let (bm, bn) = tile.dims();
        let grid_x = m.div_ceil(bm);
        let grid_y = n.div_ceil(bn);
        if grid_y > MAX_GRID_Y || grid_x > u32::MAX as usize {
            return Err(HipError::new(
                1,
                "gemm_mq8g256_gemm: shape too large for the launch grid (grid y <= 65535)",
            ));
        }
        let (module, src) = self.mq8_gemm_module();
        self.ensure_kernel(module, src, symbol)?;

        let xq = acts.xq;
        let xs = acts.xs;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let grid = [grid_x as u32, grid_y as u32, 1];
        let bytes = w_bytes + n * k + n * (k / GROUP) * 4 + y_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", symbol, bytes);
        let result = self.launch_maybe_blob(
            symbol,
            grid,
            [(2 * bm) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(xq);
                b.push_ptr(xs);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Producer + GEMM, F32 out: `Y[n, m] = X[n, k] @ W[m, k]^T` with the activation rotated
    /// and quantized to int8 per token per 256-group.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq8g256_batched(
        &mut self,
        w: &GpuTensor,
        layout: Mq8WLayout,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let acts = self.mq8_quantize_x(x, n, k)?;
        self.gemm_mq8g256_gemm(w, layout, &acts, y, m)
    }

    /// [`Self::gemm_mq8g256_batched`] with BF16-rounded outputs (F32 plane holding BF16 values).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq8g256_batched_bf16(
        &mut self,
        w: &GpuTensor,
        layout: Mq8WLayout,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let acts = self.mq8_quantize_x(x, n, k)?;
        self.gemm_mq8g256_gemm_bf16(w, layout, &acts, y, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCHS: [&str; 4] = ["gfx1200", "gfx1201", "gfx1100", "gfx1151"];
    /// Raon attention-class `(m, k)` shapes (SPEC.md).
    const RAON_SHAPES: [(usize, usize); 4] =
        [(5120, 2560), (2560, 5120), (576, 2560), (10240, 512)];
    const KS: [usize; 4] = [512, 768, 2560, 5120];

    #[test]
    fn sizes_and_offsets() {
        assert_eq!(mq8g256_a_bytes(1, 256), 258);
        assert_eq!(mq8g256_a_bytes(5120, 2560), 5120 * 10 * 258);
        // m*(k/256)*2 = 1*1*2 -> padded to 256.
        assert_eq!(mq8g256_b_code_offset(1, 256), 256);
        assert_eq!(mq8g256_b_bytes(1, 256), 256 + 256);
        // 128 groups of 2 B = exactly 256.
        assert_eq!(mq8g256_b_code_offset(128, 256), 256);
        assert_eq!(mq8g256_b_code_offset(129, 256), 512);
        for (m, k) in RAON_SHAPES {
            let off = mq8g256_b_code_offset(m, k);
            assert_eq!(off % 256, 0);
            assert!(off >= m * (k / 256) * 2 && off < m * (k / 256) * 2 + 256);
            assert_eq!(mq8g256_b_bytes(m, k), off + m * k);
            // Layout B never needs more than a few pad bytes over A's VRAM + the 0-length plane gap.
            assert!(mq8g256_b_bytes(m, k) <= mq8g256_a_bytes(m, k) + 256);
        }
        assert_eq!(mq8g256_b_bytes(5120, 2560), 5120 * 10 * 2 + 5120 * 2560);
    }

    #[test]
    fn x_scratch_layout() {
        let (off, total) = crate::scratch::mq8_x_layout(5120, 4096);
        assert_eq!(off, 4096 * 5120);
        assert_eq!(total, off + 4096 * 20 * 4);
        let (off, total) = crate::scratch::mq8_x_layout(256, 3);
        assert_eq!(off, 768);
        assert_eq!(total, 768 + 3 * 4);
        let (off, _) = crate::scratch::mq8_x_layout(256, 1);
        assert_eq!(off, 256);
        for n in [1usize, 3, 7, 512, 4095] {
            for k in KS {
                let (off, total) = crate::scratch::mq8_x_layout(k, n);
                assert_eq!(off % 256, 0);
                assert!(off >= n * k);
                assert_eq!(total, off + n * (k / 256) * 4);
            }
        }
    }

    /// Deterministic byte soup (xorshift) so every repack byte is distinguishable.
    fn pseudo_bytes(len: usize, mut s: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    fn unpack_b(b: &[u8], m: usize, k: usize) -> Vec<u8> {
        let kg = k / 256;
        let off = mq8g256_b_code_offset(m, k);
        let mut a = Vec::with_capacity(mq8g256_a_bytes(m, k));
        for r in 0..m {
            for g in 0..kg {
                let gi = r * kg + g;
                a.extend_from_slice(&b[gi * 2..gi * 2 + 2]);
                a.extend_from_slice(&b[off + r * k + g * 256..off + r * k + (g + 1) * 256]);
            }
        }
        a
    }

    #[test]
    fn repack_round_trip() {
        for k in KS {
            for m in [1usize, 2, 5, 127, 128, 129, 300] {
                let a = pseudo_bytes(mq8g256_a_bytes(m, k), (m * 31 + k) as u64 | 1);
                let b = repack_checked(&a, m, k);
                assert_eq!(b.len(), mq8g256_b_bytes(m, k));
                assert_eq!(unpack_b(&b, m, k), a, "round trip m {m} k {k}");
                // Padding between the planes is zero.
                let off = mq8g256_b_code_offset(m, k);
                assert!(b[m * (k / 256) * 2..off].iter().all(|&v| v == 0));
            }
        }
    }

    fn repack_checked(a: &[u8], m: usize, k: usize) -> Vec<u8> {
        mq8g256_repack_b(a, m, k)
    }

    #[test]
    fn repack_places_scale_bits_verbatim() {
        // Two rows, two groups: scale bytes are arbitrary f16 bit patterns (incl. NaN/denormal).
        let (m, k) = (2usize, 512usize);
        let mut a = vec![0u8; mq8g256_a_bytes(m, k)];
        let scales: [[u8; 2]; 4] = [[0x00, 0x3C], [0xFF, 0x7F], [0x01, 0x00], [0x34, 0xA2]];
        for (gi, s) in scales.iter().enumerate() {
            a[gi * 258..gi * 258 + 2].copy_from_slice(s);
            for i in 0..256 {
                a[gi * 258 + 2 + i] = (gi * 7 + i) as u8;
            }
        }
        let b = mq8g256_repack_b(&a, m, k);
        for (gi, s) in scales.iter().enumerate() {
            assert_eq!(&b[gi * 2..gi * 2 + 2], s);
        }
        let off = mq8g256_b_code_offset(m, k);
        assert_eq!(off, 256);
        // Row 1, group 0 codes land at row 1's code row, group 0.
        assert_eq!(b[off + k], (2 * 7) as u8);
        assert_eq!(b[off + k + 255], (2 * 7 + 255) as u8);
        // Row 0, group 1 (source group 1) follows group 0 in row 0's code row.
        assert_eq!(b[off + 256], 7u8);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn repack_rejects_unaligned_k() {
        let _ = mq8g256_repack_b(&[0u8; 1024], 1, 255);
    }

    #[test]
    #[should_panic(expected = "source bytes")]
    fn repack_rejects_short_source() {
        let _ = mq8g256_repack_b(&[0u8; 257], 1, 256);
    }

    #[test]
    fn pick_tile_matches_production_pickers() {
        for (m, _) in RAON_SHAPES {
            for n in [1usize, 64, 512, 768, 1024, 2048, 4096] {
                // gfx12: the Q8 W8A8 rule, by name.
                let q8 = q8_a8_pick_tile_pure(m, n);
                for arch in ["gfx1200", "gfx1201"] {
                    let t = mq8_pick_tile(arch, m, n);
                    assert_eq!(t.label(), q8.label(), "{arch} m {m} n {n}");
                    assert_eq!(t.dims(), q8.dims());
                    assert!(t.valid_on(arch));
                }
                // gfx11: q8_mmq_pick_tile, by name.
                let mmq = q8_mmq_pick_tile(m, n);
                for arch in ["gfx1100", "gfx1151"] {
                    let t = mq8_pick_tile(arch, m, n);
                    assert_eq!(t.label(), mmq.label(), "{arch} m {m} n {n}");
                    assert_eq!(t.dims(), mmq.dims());
                    assert!(t.valid_on(arch));
                }
            }
        }
    }

    #[test]
    fn pick_tile_raon_shapes() {
        // gfx12: 128x128 on every wide shape, 64x64 on the narrow M = 576, N = 512 class.
        assert_eq!(mq8_pick_tile("gfx1201", 5120, 4096), Mq8Tile::T128x128);
        assert_eq!(mq8_pick_tile("gfx1201", 5120, 512), Mq8Tile::T128x128);
        assert_eq!(mq8_pick_tile("gfx1201", 10240, 512), Mq8Tile::T128x128);
        assert_eq!(mq8_pick_tile("gfx1201", 576, 512), Mq8Tile::T64x64);
        // gfx11 (receipts in gemm_q8_mmq_gfx11.rs).
        assert_eq!(mq8_pick_tile("gfx1100", 5120, 512), Mq8Tile::T64x128);
        assert_eq!(mq8_pick_tile("gfx1100", 2560, 512), Mq8Tile::T64x64);
        assert_eq!(mq8_pick_tile("gfx1100", 5120, 4096), Mq8Tile::T128x128);
        assert_eq!(mq8_pick_tile("gfx1100", 576, 4096), Mq8Tile::T64x128);
    }

    #[test]
    fn tile_validity_per_arch() {
        for arch in ["gfx1200", "gfx1201"] {
            assert!(Mq8Tile::T128x128.valid_on(arch));
            assert!(Mq8Tile::T128x64.valid_on(arch));
            assert!(Mq8Tile::T64x64.valid_on(arch));
            assert!(!Mq8Tile::T64x128.valid_on(arch));
        }
        for arch in ["gfx1100", "gfx1151"] {
            assert!(Mq8Tile::T128x128.valid_on(arch));
            assert!(Mq8Tile::T64x128.valid_on(arch));
            assert!(Mq8Tile::T64x64.valid_on(arch));
            assert!(!Mq8Tile::T128x64.valid_on(arch));
        }
        for arch in ["gfx1030", "gfx1101", "gfx942", ""] {
            for t in [
                Mq8Tile::T128x128,
                Mq8Tile::T128x64,
                Mq8Tile::T64x128,
                Mq8Tile::T64x64,
            ] {
                assert!(!t.valid_on(arch));
                assert!(t.symbol(arch, Mq8WLayout::A, false).is_none());
            }
        }
    }

    #[test]
    fn symbol_naming() {
        let tiles = [
            Mq8Tile::T128x128,
            Mq8Tile::T128x64,
            Mq8Tile::T64x128,
            Mq8Tile::T64x64,
        ];
        for arch in ARCHS {
            let gfx11 = matches!(arch, "gfx1100" | "gfx1151");
            let (prefix, src) = if gfx11 {
                (
                    "gemm_mq8g256_mmq_i8_gfx11_t",
                    kernels::GEMM_MQ8G256_MMQ_I8_GFX11_SRC,
                )
            } else {
                (
                    "gemm_mq8g256_mmq_i8_t",
                    kernels::GEMM_MQ8G256_MMQ_I8_GFX12_SRC,
                )
            };
            let mut seen = std::collections::BTreeSet::new();
            for tile in tiles {
                for layout in [Mq8WLayout::A, Mq8WLayout::B] {
                    for bf16 in [false, true] {
                        let sym = tile.symbol(arch, layout, bf16);
                        assert_eq!(sym.is_some(), tile.valid_on(arch));
                        let Some(sym) = sym else { continue };
                        let wl = if layout == Mq8WLayout::A { "wa" } else { "wb" };
                        let want = format!(
                            "{prefix}{}_{wl}{}",
                            tile.label(),
                            if bf16 { "_bf16" } else { "" }
                        );
                        assert_eq!(sym, want);
                        assert!(seen.insert(sym), "duplicate symbol {sym}");
                        // The source instantiates it through the NAME##WL(##_bf16) macros.
                        let frag = format!(
                            "({prefix}{}_##WL{}",
                            tile.label(),
                            if bf16 { "##_bf16" } else { "," }
                        );
                        assert!(src.contains(&frag), "{sym}: source lacks {frag}");
                        assert!(
                            src.contains(&format!("TILES({wl},")),
                            "{sym}: no {wl} tile set"
                        );
                    }
                }
            }
            assert_eq!(seen.len(), 12, "{arch}");
        }
        assert!(kernels::MQ8_ROTATE_QUANTIZE_X_BATCHED_SRC
            .contains("void mq8_rotate_quantize_x_batched("));
    }

    #[test]
    fn admission() {
        for arch in ARCHS {
            assert!(mq8_admits(arch, 256));
            assert!(mq8_admits(arch, 2560));
            assert!(mq8_admits(arch, 512));
            assert!(!mq8_admits(arch, 0));
            assert!(!mq8_admits(arch, 128));
            assert!(!mq8_admits(arch, 576)); // q_b stays Q8F16
            assert!(!mq8_admits(arch, 2560 + 128));
        }
        for arch in [
            "gfx1030", "gfx1101", "gfx1102", "gfx1150", "gfx942", "gfx906", "",
        ] {
            assert!(!mq8_admits(arch, 2560), "{arch}");
        }
    }
}
