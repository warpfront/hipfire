// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// im2col for the FLUX VAE decoder 3x3 stride-1 pad-1 convolution, feeding
// the WMMA f16 GEMM route: input `[c_in][h][w]` f32 channel-major, output
// `[rows*w][c_in*9]` f16 rows where the column order `ci*9 + ky*3 + kx`
// matches the `[c_out][c_in*9]` weight layout the loader flattens to
// `[c_out][c_in*9]`. Pad positions are zeroed here so the GEMM needs no
// boundary logic.
//
// All three entries take the same arguments and cover the same output band:
// output row `r` corresponds to input row `y0 + r` for `r` in `[0, rows)`,
// so a caller can stage the conv over horizontal bands of the image and keep
// the (large) column matrix bounded. `y0 = 0, rows = h` is the whole image.
//
// Three lane mappings, A/B-selectable via `HIPFIRE_VAE_IM2COL_MAP`:
//   `lds` (default) — LDS-tiled: one workgroup stages a `[c_tile][(th+2) x
//                     (tw+2)]` halo patch of the input into LDS with
//                     coalesced f32 reads, then streams the `[th*tw]
//                     [c_tile*9]` output rows back out as long contiguous
//                     runs. Each input element is read from DRAM once
//                     instead of nine times.
//   `c`             — channel-fastest scalar gather (the previous default).
//   `p`             — pixel-fastest scalar gather (measured ~6x worse: the
//                     18-byte strided writes dominate).

#include <hip/hip_runtime.h>

// ─── LDS-tiled (default) ────────────────────────────────────────────────
//
// Grid: (ceil(w/tw), ceil(rows/th), ceil(c_in/c_tile)). Block: 1-D, at least
// `(th+2)*(tw+2)` threads recommended so the staging pass is a single sweep.
// Dynamic shared memory: `c_tile*(th+2)*(tw+2)*sizeof(_Float16)` bytes.
//
// Why the halves land in LDS rather than the floats: the output is f16, so
// casting once on the way in makes the patch half the size (a bigger
// `c_tile` fits, which is what lengthens the contiguous store runs) and
// removes eight of the nine casts per input element.
extern "C" __global__ void vae_im2col_f16_lds(
    const float* __restrict__ x,
    _Float16* __restrict__ out,
    int c_in,
    int h,
    int w,
    int y0,
    int rows,
    int c_tile,
    int th,
    int tw
) {
    extern __shared__ _Float16 patch[];

    const int pw = tw + 2;
    const int ph = th + 2;
    const int patch_elems = ph * pw;
    const long hw = (long)h * (long)w;

    const int ci0 = blockIdx.z * c_tile;
    const int ox0 = blockIdx.x * tw;
    const int ly0 = blockIdx.y * th;          // band-local first output row
    if (ci0 >= c_in || ox0 >= w || ly0 >= rows) return;
    const int c_here = min(c_tile, c_in - ci0);

    // ── stage: [c_here][ph][pw] halo patch, f32 global -> f16 LDS. Lanes run
    // fastest along x, so each row of the patch is one contiguous read.
    for (int r = threadIdx.x; r < patch_elems; r += blockDim.x) {
        const int ly = r / pw;
        const int lx = r - ly * pw;
        const int iy = y0 + ly0 + ly - 1;
        const int ix = ox0 + lx - 1;
        const bool inb = (iy >= 0 && iy < h && ix >= 0 && ix < w);
        const long base = (long)iy * (long)w + (long)ix;
        for (int cl = 0; cl < c_here; ++cl) {
            const float v = inb ? x[(long)(ci0 + cl) * hw + base] : 0.0f;
            patch[(long)cl * (long)patch_elems + r] = (_Float16)v;
        }
    }
    __syncthreads();

    // ── stream out: one contiguous `c_here*9`-half run per output pixel.
    // Every divisor below is a literal, so these are multiply-shifts, not
    // integer divides.
    const long row_stride = (long)c_in * 9;
    const int run = c_here * 9;
    for (int ly = 0; ly < th; ++ly) {
        const int oy = ly0 + ly;
        if (oy >= rows) break;
        for (int lx = 0; lx < tw; ++lx) {
            const int ox = ox0 + lx;
            if (ox >= w) break;
            _Float16* dst = out + ((long)oy * (long)w + (long)ox) * row_stride
                          + (long)ci0 * 9;
            for (int e = threadIdx.x; e < run; e += blockDim.x) {
                const int cl = e / 9;
                const int kk = e - cl * 9;
                const int ky = kk / 3;
                const int kx = kk - ky * 3;
                dst[e] = patch[(long)cl * (long)patch_elems
                             + (long)((ly + ky) * pw + (lx + kx))];
            }
        }
    }
}

// ─── channel-fastest scalar gather (previous default, kept for A/B) ──────
extern "C" __global__ void vae_im2col_f16_cfast(
    const float* __restrict__ x,
    _Float16* __restrict__ out,
    int c_in,
    int h,
    int w,
    int y0,
    int rows,
    int c_tile,
    int th,
    int tw
) {
    (void)c_tile; (void)th; (void)tw;
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)c_in * (long)rows * (long)w;
    if (t >= total) return;
    int p = (int)(t / c_in);
    int ci = (int)(t % c_in);
    int oy = p / w;
    int ox = p - oy * w;
    const float* xci = x + (long)ci * (long)h * (long)w;
    _Float16* row = out + (long)p * (long)c_in * 9 + (long)ci * 9;
    for (int ky = 0; ky < 3; ++ky) {
        int iy = y0 + oy + ky - 1;
        for (int kx = 0; kx < 3; ++kx) {
            int ix = ox + kx - 1;
            float v = (iy < 0 || iy >= h || ix < 0 || ix >= w)
                          ? 0.0f
                          : xci[(long)iy * (long)w + ix];
            row[ky * 3 + kx] = (_Float16)v;
        }
    }
}

// ─── pixel-fastest scalar gather (contiguous 32-float reads within one
// channel plane per tap, strided 18-byte-pair writes) ───────────────────
extern "C" __global__ void vae_im2col_f16_pfast(
    const float* __restrict__ x,
    _Float16* __restrict__ out,
    int c_in,
    int h,
    int w,
    int y0,
    int rows,
    int c_tile,
    int th,
    int tw
) {
    (void)c_tile; (void)th; (void)tw;
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long band = (long)rows * (long)w;
    long total = (long)c_in * band;
    if (t >= total) return;
    int ci = (int)(t / band);
    int p = (int)(t % band);
    int oy = p / w;
    int ox = p - oy * w;
    const float* xci = x + (long)ci * (long)h * (long)w;
    _Float16* row = out + (long)p * (long)c_in * 9 + (long)ci * 9;
    for (int ky = 0; ky < 3; ++ky) {
        int iy = y0 + oy + ky - 1;
        for (int kx = 0; kx < 3; ++kx) {
            int ix = ox + kx - 1;
            float v = (iy < 0 || iy >= h || ix < 0 || ix >= w)
                          ? 0.0f
                          : xci[(long)iy * (long)w + ix];
            row[ky * 3 + kx] = (_Float16)v;
        }
    }
}
