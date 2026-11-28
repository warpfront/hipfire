// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 blocked-prefill F32 projections on gfx12 WMMA (hi/lo F16).
//!
//! The loop controllers (`down [64, 4608]`, `gain`/`forget [2304, 64]`), the
//! attention sinks (`[24, 2304]`) and the dynamic conv taps (`[32, 2304]`) are
//! F32 keep-list weights. Their singleton GEMV per row reads the weight once
//! per eight rows; over a block of rows these kernels run them as tiled GEMMs.
//!
//! Both operands enter the WMMA as hi/lo F16 pairs, `x = hi + lo / 2048` with
//! `hi = f16(x)` and `lo = f16((x - hi) * 2048)` (the lo plane is pre-scaled so
//! it stays clear of the F16 subnormals). Each 16x16x16 step accumulates
//! `hi.hi` into one F32 accumulator and `hi.lo + lo.hi` into a second; the
//! output is `acc + corr / 2048`. F16 x F16 products are exact in F32, so the
//! result keeps ~22 mantissa bits per product (the dropped `lo.lo` term is
//! below 2^-22 relative): F32-class, not the GEMV's summation order.
//!
//! K is split over up to eight waves of a workgroup; their partials are summed
//! in wave order through LDS, so the result is deterministic run to run.
//!
//! * [`Gpu::inkling2_gemm_hilo_f32`]: `y_z [n, m] = x_z [n, k] . w_z [m, k]^T`
//!   for up to [`INKLING2_HILO_MAX_PROBLEMS`] same-shape problems (grid z).
//! * [`Gpu::inkling2_control_up_blend_hilo`]: the controller's gain and forget
//!   projections of the same `g` plus the blend epilogue
//!   `out = cur - tanh(F.g) cur + tanh(G.g) msg` (no gain/forget round trip).
//!
//! gfx12 wave32 WMMA only; callers keep the GEMV for the bitwise oracle mode.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};

/// Runtime module name of [`INKLING2_HILO_HIP_SRC`].
pub const INKLING2_HILO_MODULE: &str = "inkling2_hilo_gemm";

/// Problems one launch carries (grid z).
pub const INKLING2_HILO_MAX_PROBLEMS: usize = 4;

/// The two kernels behind [`Gpu::inkling2_gemm_hilo_f32`] and
/// [`Gpu::inkling2_control_up_blend_hilo`].
pub const INKLING2_HILO_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef float __attribute__((ext_vector_type(4))) f32x4_t;

#define HL_MAXP 4
#define HL_LO_SCALE 2048.0f
#define HL_LO_INV (1.0f / 2048.0f)

struct HlPtrs {
    const float* p[HL_MAXP];
};

// v = hi + lo / 2048, hi = f16(v), lo = f16((v - hi) * 2048).
__device__ __forceinline__ void hl_split(f32x4_t a, f32x4_t b, half8_t& hi, half8_t& lo)
{
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const _Float16 h = (_Float16)v[j];
        hi[j] = h;
        lo[j] = (_Float16)((v[j] - (float)h) * HL_LO_SCALE);
    }
}

// One wave's (16 TM) m x (16 TN) n tile of NW weights against the same x,
// over its K slice (blockDim.y waves split K into contiguous 16-wide step
// ranges); wave 0 returns the wave-ordered sum. x is the WMMA A operand and w
// the B operand (C = A B^T is [n, m]), so with the gfx12 layout (lane & 15 the
// operand row, lane >> 4 the K half) out[q][tm][tn][j] is element
// (n = 16 tn + 8 (lane >> 4) + j, m = 16 tm + (lane & 15)): sixteen lanes hold
// sixteen consecutive m of one output row (coalesced stores). U steps of
// operands are loaded ahead of their WMMAs. Rows past M / N are read from
// row 0 and zeroed (no divergent loads).
template <int TM, int TN, int NW, int U>
__device__ __forceinline__ void hl_tile(
    const float* const* w, const float* x, int M, int K, int N, int row0, int col0,
    float8_t (&out)[NW][TM][TN], float* red)
{
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int ks = blockDim.y;
    const int lm = lane & 15;
    const int kg = lane >> 4;
    const int steps = K >> 4;
    const int chunk = (steps + ks - 1) / ks;
    const int s0 = min(steps, wave * chunk);
    const int s1 = min(steps, s0 + chunk);

    const float* wp[NW][TM];
    bool win[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int r = row0 + tm * 16 + lm;
        win[tm] = r < M;
#pragma unroll
        for (int q = 0; q < NW; ++q) wp[q][tm] = w[q] + (size_t)(win[tm] ? r : 0) * K + kg * 8;
    }
    const float* xp[TN];
    bool xin[TN];
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        const int c = col0 + tn * 16 + lm;
        xin[tn] = c < N;
        xp[tn] = x + (size_t)(xin[tn] ? c : 0) * K + kg * 8;
    }

    float8_t hh[NW][TM][TN], cr[NW][TM][TN];
#pragma unroll
    for (int q = 0; q < NW; ++q)
#pragma unroll
        for (int tm = 0; tm < TM; ++tm)
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                hh[q][tm][tn] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                cr[q][tm][tn] = hh[q][tm][tn];
            }

    const f32x4_t z4 = {0.f, 0.f, 0.f, 0.f};
    for (int s = s0; s < s1; s += U) {
        f32x4_t xv[U][TN][2], wv[U][NW][TM][2];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int ko = (s + u) << 4;
            const bool live = s + u < s1;
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                xv[u][tn][0] = live ? *(const f32x4_t*)(xp[tn] + ko) : z4;
                xv[u][tn][1] = live ? *(const f32x4_t*)(xp[tn] + ko + 4) : z4;
            }
#pragma unroll
            for (int q = 0; q < NW; ++q)
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    wv[u][q][tm][0] = live ? *(const f32x4_t*)(wp[q][tm] + ko) : z4;
                    wv[u][q][tm][1] = live ? *(const f32x4_t*)(wp[q][tm] + ko + 4) : z4;
                }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (s + u >= s1) break;
            half8_t xh[TN], xl[TN];
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                if (!xin[tn]) { xv[u][tn][0] = z4; xv[u][tn][1] = z4; }
                hl_split(xv[u][tn][0], xv[u][tn][1], xh[tn], xl[tn]);
            }
#pragma unroll
            for (int q = 0; q < NW; ++q)
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    if (!win[tm]) { wv[u][q][tm][0] = z4; wv[u][q][tm][1] = z4; }
                    half8_t wh, wl;
                    hl_split(wv[u][q][tm][0], wv[u][q][tm][1], wh, wl);
#pragma unroll
                    for (int tn = 0; tn < TN; ++tn) {
                        hh[q][tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(xh[tn], wh, hh[q][tm][tn]);
                        cr[q][tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(xh[tn], wl, cr[q][tm][tn]);
                        cr[q][tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(xl[tn], wh, cr[q][tm][tn]);
                    }
                }
        }
    }

#pragma unroll
    for (int q = 0; q < NW; ++q)
#pragma unroll
        for (int tm = 0; tm < TM; ++tm)
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out[q][tm][tn][j] = hh[q][tm][tn][j] + cr[q][tm][tn][j] * HL_LO_INV;

    if (ks > 1) {
        constexpr int NA = NW * TM * TN * 8;
        if (wave > 0) {
            float* dst = red + (size_t)(wave - 1) * NA * 32;
            int i = 0;
#pragma unroll
            for (int q = 0; q < NW; ++q)
#pragma unroll
                for (int tm = 0; tm < TM; ++tm)
#pragma unroll
                    for (int tn = 0; tn < TN; ++tn)
#pragma unroll
                        for (int j = 0; j < 8; ++j) dst[(i++) * 32 + lane] = out[q][tm][tn][j];
        }
        __syncthreads();
        if (wave == 0) {
            for (int v = 1; v < ks; ++v) {
                const float* src = red + (size_t)(v - 1) * NA * 32;
                int i = 0;
#pragma unroll
                for (int q = 0; q < NW; ++q)
#pragma unroll
                    for (int tm = 0; tm < TM; ++tm)
#pragma unroll
                        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
                            for (int j = 0; j < 8; ++j) out[q][tm][tn][j] += src[(i++) * 32 + lane];
            }
        }
    }
}

// Y_z [N, M] = X_z [N, K] . W_z [M, K]^T, z = blockIdx.z, one (16 TM m) x (16 TN n)
// tile per workgroup.
template <int TM, int TN, int U>
__device__ __forceinline__ void hl_gemm(const HlPtrs& W, const HlPtrs& X, const HlPtrs& Y, int M, int K, int N, float* red)
{
    const int z = blockIdx.z;
    const int row0 = blockIdx.x * 16 * TM;
    const int col0 = blockIdx.y * 16 * TN;
    const float* w[1] = {W.p[z]};
    float8_t acc[1][TM][TN];
    hl_tile<TM, TN, 1, U>(w, X.p[z], M, K, N, row0, col0, acc, red);
    if (threadIdx.y != 0) return;
    float* y = (float*)Y.p[z];
    const int lm = threadIdx.x & 15;
    const int kg = threadIdx.x >> 4;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int m = row0 + tm * 16 + lm;
        if (m >= M) continue;
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int n = col0 + tn * 16 + 8 * kg + j;
                if (n < N) y[(size_t)n * M + m] = acc[0][tm][tn][j];
            }
    }
}

// Grid (ceil(M / 32), ceil(N / 16), problems), block (32, split-K waves): the
// few-output shapes (sinks, dynamic taps), where waves are scarce.
__launch_bounds__(256)
extern "C" __global__ void inkling2_gemm_hilo_f32(HlPtrs W, HlPtrs X, HlPtrs Y, int M, int K, int N)
{
    extern __shared__ float red[];
    hl_gemm<2, 1, 4>(W, X, Y, M, K, N, red);
}

// Grid (ceil(M / 32), ceil(N / 32), problems), block (32, split-K waves): half
// the operand traffic per output (controller down projections).
__launch_bounds__(256)
extern "C" __global__ void inkling2_gemm_hilo_f32_n32(HlPtrs W, HlPtrs X, HlPtrs Y, int M, int K, int N)
{
    extern __shared__ float red[];
    hl_gemm<2, 2, 4>(W, X, Y, M, K, N, red);
}

// One activation row (N == 1, decode). Grid (ceil(M / 16), 1, problems),
// block (32, split-K waves): one 16-wide m tile per workgroup, twice the
// workgroups of the 32-row tiles, and none of their dead column tiles. Every
// output element sees the same WMMA chain over the same K slices and the same
// wave-ordered sum as in the tiled kernels, so the row is bitwise theirs.
__launch_bounds__(256)
extern "C" __global__ void inkling2_gemm_hilo_f32_r1(HlPtrs W, HlPtrs X, HlPtrs Y, int M, int K, int N)
{
    extern __shared__ float red[];
    hl_gemm<1, 1, 4>(W, X, Y, M, K, N, red);
}

// Controller tail over rows: a = G_z . g_z, f = F_z . g_z (both [N, M]), then
// out_z = cur_z - tanh(f) * cur_z + tanh(a) * msg_z (inkling2_control_blend_f32).
// Grid (ceil(M / 16), ceil(N / 16), problems), block (32, split-K waves).
__launch_bounds__(256)
extern "C" __global__ void inkling2_control_up_blend_hilo(
    HlPtrs G, HlPtrs F, HlPtrs X, HlPtrs CUR, HlPtrs MSG, HlPtrs OUT, int M, int K, int N)
{
    extern __shared__ float red[];
    const int z = blockIdx.z;
    const int row0 = blockIdx.x * 16;
    const int col0 = blockIdx.y * 16;
    const float* w[2] = {G.p[z], F.p[z]};
    float8_t acc[2][1][1];
    hl_tile<1, 1, 2, 4>(w, X.p[z], M, K, N, row0, col0, acc, red);
    if (threadIdx.y != 0) return;
    const int m = row0 + (threadIdx.x & 15);
    if (m >= M) return;
    const float* cur = CUR.p[z];
    const float* msg = MSG.p[z];
    float* out = (float*)OUT.p[z];
    const int kg = threadIdx.x >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n = col0 + 8 * kg + j;
        if (n >= N) continue;
        const size_t i = (size_t)n * M + m;
        const float c = cur[i];
        const float f = tanhf(acc[1][0][0][j]);
        const float a = tanhf(acc[0][0][0][j]);
        out[i] = c - f * c + a * msg[i];
    }
}
"#;

/// One problem of [`Gpu::inkling2_gemm_hilo_f32`]: `y [n, m] = x [n, k] . w [m, k]^T`.
pub struct Inkling2HiloGemm<'a> {
    /// F32 `[m, k]` weight.
    pub w: &'a GpuTensor,
    /// F32 `[n, k]` activation rows.
    pub x: &'a GpuTensor,
    /// F32 `[n, m]` output rows.
    pub y: &'a GpuTensor,
}

/// One controller of [`Gpu::inkling2_control_up_blend_hilo`].
pub struct Inkling2HiloBlend<'a> {
    /// F32 `[m, k]` gain weight `G`.
    pub gain_w: &'a GpuTensor,
    /// F32 `[m, k]` forget weight `F`.
    pub forget_w: &'a GpuTensor,
    /// F32 `[n, k]` down-projected rows `g`.
    pub g: &'a GpuTensor,
    /// F32 `[n, m]` controlled input rows.
    pub cur: &'a GpuTensor,
    /// F32 `[n, m]` message rows.
    pub msg: &'a GpuTensor,
    /// F32 `[n, m]` output rows (may not alias `cur` / `msg`).
    pub out: &'a GpuTensor,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HlPtrs([*const c_void; INKLING2_HILO_MAX_PROBLEMS]);

impl HlPtrs {
    fn of<'a>(ts: impl Iterator<Item = &'a GpuTensor>) -> Self {
        let mut p = [std::ptr::null(); INKLING2_HILO_MAX_PROBLEMS];
        for (slot, t) in p.iter_mut().zip(ts) {
            *slot = t.buf.as_ptr() as *const c_void;
        }
        Self(p)
    }
}

/// Split-K waves for a `k`-deep reduction: >= 16 WMMA steps per wave, <= 8 waves.
fn split_k(k: usize) -> usize {
    (k / 256).clamp(1, 8)
}

/// Bytes of LDS the split-K reduction over `ks` waves needs for `accs`
/// accumulator floats per lane (`NW * TM * TN * 8` of the kernel's tile).
fn red_bytes(ks: usize, accs: usize) -> u32 {
    ((ks - 1) * accs * 32 * 4) as u32
}

impl Gpu {
    fn inkling2_hilo_check(
        &self,
        what: &str,
        problems: usize,
        m: usize,
        k: usize,
        n: usize,
        f32_extents: &[(&GpuTensor, usize, bool)],
    ) -> HipResult<()> {
        let bad = |msg: String| Err(HipError::new(0, &format!("{what}: {msg}")));
        if !self.arch_caps.has_wmma_w32_gfx12() {
            return bad(format!("needs gfx12 wave32 WMMA, not {}", self.arch));
        }
        if !(1..=INKLING2_HILO_MAX_PROBLEMS).contains(&problems) {
            return bad(format!(
                "{problems} problems (1..={INKLING2_HILO_MAX_PROBLEMS})"
            ));
        }
        if m == 0 || n == 0 || k == 0 || k % 16 != 0 {
            return bad(format!("shape m={m} k={k} n={n} (k % 16 == 0, all > 0)"));
        }
        if i32::try_from(m.max(n).max(k)).is_err()
            || u32::try_from(n.div_ceil(16)).is_err()
            || m.checked_mul(k).is_none()
            || n.checked_mul(k.max(m)).is_none()
        {
            return bad(format!("shape m={m} k={k} n={n} exceeds i32"));
        }
        for &(t, floats, aligned) in f32_extents {
            if t.buf.size() < floats * 4 {
                return bad(format!(
                    "operand of {} bytes < {} F32 elements",
                    t.buf.size(),
                    floats
                ));
            }
            if aligned && (t.buf.as_ptr() as usize) % 16 != 0 {
                return bad("WMMA operand not 16-byte aligned".into());
            }
        }
        Ok(())
    }

    /// `y_z [n, m] = x_z [n, k] . w_z [m, k]^T` for every problem `z` (F32 in
    /// and out, hi/lo F16 WMMA; see the module docs). gfx12 only; `k % 16 == 0`,
    /// 16-byte aligned `w` / `x`, contiguous rows.
    pub fn inkling2_gemm_hilo_f32(
        &mut self,
        problems: &[Inkling2HiloGemm<'_>],
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let mut ext = Vec::with_capacity(problems.len() * 3);
        for p in problems {
            ext.extend([(p.w, m * k, true), (p.x, n * k, true), (p.y, n * m, false)]);
        }
        self.inkling2_hilo_check("inkling2_gemm_hilo_f32", problems.len(), m, k, n, &ext)?;
        self.bind_thread()?;
        // One row (decode): 16-row m tiles, the only source of waves left
        // (bitwise the tiled kernels' row). M <= 32 is one workgroup row, so
        // column tiles are the only source of waves (sinks, dynamic taps):
        // 16-row tiles. Wider M (controller down) takes 32-row tiles, half the
        // operand traffic per output.
        let (sym, m_tile, n_tile, accs) = if n == 1 {
            ("inkling2_gemm_hilo_f32_r1", 16, 16, 8)
        } else if m >= 64 {
            ("inkling2_gemm_hilo_f32_n32", 32, 32, 32)
        } else {
            ("inkling2_gemm_hilo_f32", 32, 16, 16)
        };
        self.ensure_kernel(INKLING2_HILO_MODULE, INKLING2_HILO_HIP_SRC, sym)?;
        let w = HlPtrs::of(problems.iter().map(|p| p.w));
        let x = HlPtrs::of(problems.iter().map(|p| p.x));
        let y = HlPtrs::of(problems.iter().map(|p| p.y));
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let ks = split_k(k);
        let mut params: Vec<*mut c_void> = vec![
            &w as *const _ as *mut c_void,
            &x as *const _ as *mut c_void,
            &y as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            sym,
            [
                m.div_ceil(m_tile) as u32,
                n.div_ceil(n_tile) as u32,
                problems.len() as u32,
            ],
            [32, ks as u32, 1],
            red_bytes(ks, accs),
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for set in [&w, &x, &y] {
                    for p in set.0 {
                        b.push_ptr(p);
                    }
                }
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        )
    }

    /// Controller tail for every problem `z` over `n` rows: `a = G g`,
    /// `f = F g` (`G`, `F` F32 `[m, k]`, `g` F32 `[n, k]`), then
    /// `out = cur - tanh(f) cur + tanh(a) msg` (`[n, m]`). hi/lo F16 WMMA (see
    /// the module docs); gfx12 only, `k % 16 == 0`, 16-byte aligned weights
    /// and `g`, contiguous rows.
    pub fn inkling2_control_up_blend_hilo(
        &mut self,
        problems: &[Inkling2HiloBlend<'_>],
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let mut ext = Vec::with_capacity(problems.len() * 6);
        for p in problems {
            ext.extend([
                (p.gain_w, m * k, true),
                (p.forget_w, m * k, true),
                (p.g, n * k, true),
                (p.cur, n * m, false),
                (p.msg, n * m, false),
                (p.out, n * m, false),
            ]);
        }
        self.inkling2_hilo_check(
            "inkling2_control_up_blend_hilo",
            problems.len(),
            m,
            k,
            n,
            &ext,
        )?;
        self.bind_thread()?;
        self.ensure_kernel(
            INKLING2_HILO_MODULE,
            INKLING2_HILO_HIP_SRC,
            "inkling2_control_up_blend_hilo",
        )?;
        let gw = HlPtrs::of(problems.iter().map(|p| p.gain_w));
        let fw = HlPtrs::of(problems.iter().map(|p| p.forget_w));
        let g = HlPtrs::of(problems.iter().map(|p| p.g));
        let cur = HlPtrs::of(problems.iter().map(|p| p.cur));
        let msg = HlPtrs::of(problems.iter().map(|p| p.msg));
        let out = HlPtrs::of(problems.iter().map(|p| p.out));
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let ks = split_k(k);
        let mut params: Vec<*mut c_void> = vec![
            &gw as *const _ as *mut c_void,
            &fw as *const _ as *mut c_void,
            &g as *const _ as *mut c_void,
            &cur as *const _ as *mut c_void,
            &msg as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_control_up_blend_hilo",
            [
                m.div_ceil(16) as u32,
                n.div_ceil(16) as u32,
                problems.len() as u32,
            ],
            [32, ks as u32, 1],
            red_bytes(ks, 16),
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for set in [&gw, &fw, &g, &cur, &msg, &out] {
                    for p in set.0 {
                        b.push_ptr(p);
                    }
                }
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_k_keeps_sixteen_steps_per_wave() {
        assert_eq!(split_k(64), 1);
        assert_eq!(split_k(2304), 8);
        assert_eq!(split_k(4608), 8);
        assert_eq!(split_k(512), 2);
        for k in (16..=8192).step_by(16) {
            let ks = split_k(k);
            assert!((1..=8).contains(&ks));
            assert!(ks == 1 || k / 16 / ks >= 16);
        }
    }

    #[test]
    fn reduction_fits_lds() {
        assert_eq!(red_bytes(1, 32), 0);
        assert!(red_bytes(8, 32) <= 64 * 1024);
    }
}
