// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VCN JPEG → `pixel_values` preprocess, experiment `experiment/vcn-jpeg` only.
//!
//! The VCN decoder leaves NV12 (4:2:0) or interleaved full-res chroma (4:4:4)
//! in VRAM via dma-buf import. These two kernels finish the vision
//! preprocess on-device so `pixel_values` never leaves VRAM:
//!
//! 1. `vl_nv12_to_rgb_norm` — YUV → RGB (full-range BT.601, the JFIF
//!    convention the `image`-crate CPU path also uses) with separable
//!    CatmullRom (Keys a=0.5) resampling to the `smart_resize` target, then
//!    `/127.5 - 1` normalisation into CHW f32. Chroma is resampled in its
//!    native (possibly subsampled) grid, which is the closest GPU analogue
//!    of the CPU path's decode-then-RGB-resize order.
//! 2. `vl_extract_patches` — exact device port of
//!    `hipfire_arch_qwen35_vl::image::extract_patches` (C,T,ph,pw) per-patch
//!    layout, 2x2 spatial-merge-grouped order, T duplication.
//!
//! Bit-exactness vs the CPU path is NOT expected (VCN IDCT/upsampling differ
//! from libjpeg-turbo; documented in the parity example). Determinism: both
//! kernels are atomic-free, one thread per output element.

#include <hip/hip_runtime.h>

// Keys cubic with a = 0.5 (Catmull-Rom: Mitchell B=0, C=0.5). Term-identical
// to image-0.25.10 `bc_cubic_spline(x, 0.0, 0.5)` (verified term by term;
// stage-D host-port residual 0.35 LSB mean).
__device__ __forceinline__ float vl_cubic(float x) {
    x = fabsf(x);
    const float x2 = x * x;
    const float x3 = x2 * x;
    if (x <= 1.0f) return 1.5f * x3 - 2.5f * x2 + 1.0f;
    if (x <= 2.0f) return -0.5f * x3 + 2.5f * x2 - 4.0f * x + 2.0f;
    return 0.0f;
}

__device__ __forceinline__ int vl_clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Separable CatmullRom sample of one single-byte plane.
__device__ float vl_sample_plane(const unsigned char* __restrict__ base, int pitch, int w, int h,
                                 float fx, float fy) {
    const int x0 = (int)floorf(fx) - 1;
    const int y0 = (int)floorf(fy) - 1;
    float acc = 0.0f;
    float wsum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int yy = vl_clampi(y0 + j, 0, h - 1);
        const float wy = vl_cubic(fy - (float)(y0 + j));
        const unsigned char* row = base + (size_t)yy * (size_t)pitch;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int xx = vl_clampi(x0 + i, 0, w - 1);
            const float wx = vl_cubic(fx - (float)(x0 + i));
            const float wgt = wx * wy;
            acc += wgt * (float)row[xx];
            wsum += wgt;
        }
    }
    return wsum != 0.0f ? acc / wsum : acc;
}

// Chroma sample from either an interleaved plane (NV12/NV24: u_base = plane,
// v_base = plane + 1, step 2) or two planar planes (444P/422P: step 1).
__device__ __forceinline__ void vl_sample_chroma(const unsigned char* __restrict__ u_base,
                                                 const unsigned char* __restrict__ v_base,
                                                 int pitch, int step, int w2, int h2, float fx,
                                                 float fy, float* u, float* v) {
    const int x0 = (int)floorf(fx) - 1;
    const int y0 = (int)floorf(fy) - 1;
    float au = 0.0f, av = 0.0f, wsum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int yy = vl_clampi(y0 + j, 0, h2 - 1);
        const float wy = vl_cubic(fy - (float)(y0 + j));
        const unsigned char* row_u = u_base + (size_t)yy * (size_t)pitch;
        const unsigned char* row_v = v_base + (size_t)yy * (size_t)pitch;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int xx = vl_clampi(x0 + i, 0, w2 - 1);
            const float wx = vl_cubic(fx - (float)(x0 + i));
            const float wgt = wx * wy;
            au += wgt * (float)row_u[step * xx];
            av += wgt * (float)row_v[step * xx];
            wsum += wgt;
        }
    }
    if (wsum != 0.0f) {
        au /= wsum;
        av /= wsum;
    }
    *u = au;
    *v = av;
}

extern "C" __global__ void vl_nv12_to_rgb_norm(
    const unsigned char* __restrict__ surf_base,  // imported dma-buf base
    unsigned int y_pitch, unsigned int uv_pitch,
    // chroma plane bases relative to surf_base and the byte step between
    // horizontally adjacent samples: NV12 -> (uv_off, uv_off + 1, 2);
    // planar 444P/422P -> (u_off, v_off, 1).
    unsigned int u_offset, unsigned int v_offset, unsigned int chroma_step, unsigned int src_w,
    unsigned int src_h,
    // chroma subsampling shifts: 420 -> (1,1); 444 -> (0,0)
    unsigned int chroma_shift_x, unsigned int chroma_shift_y,
    unsigned int dst_w, unsigned int dst_h,
    float* __restrict__ dst_chw  // [3][dst_h][dst_w] f32, normalised
) {
    const unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // dst-centre mapping, same convention as image::imageops resize.
    const float sx = ((float)dx + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    const float sy = ((float)dy + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;

    const float yv =
        vl_sample_plane(surf_base, (int)y_pitch, (int)src_w, (int)src_h, sx, sy);

    const unsigned char* u_base = surf_base + u_offset;
    const unsigned char* v_base = surf_base + v_offset;
    const int cw = (int)((src_w + (1u << chroma_shift_x) - 1u) >> chroma_shift_x);
    const int ch = (int)((src_h + (1u << chroma_shift_y) - 1u) >> chroma_shift_y);
    // Centered chroma siting (JFIF: chroma[i] is the 2x2-block average,
    // nominally at luma S*i + (S-1)/2), matching libjpeg's fancy-upsample
    // phase. (Was off by half a chroma tap: +0.25 instead of -0.25.)
    const float cx = (sx - 0.5f * (float)((1u << chroma_shift_x) - 1u)) / (float)(1u << chroma_shift_x);
    const float cy = (sy - 0.5f * (float)((1u << chroma_shift_y) - 1u)) / (float)(1u << chroma_shift_y);
    float cb, cr;
    vl_sample_chroma(u_base, v_base, (int)uv_pitch, (int)chroma_step, cw, ch, cx, cy, &cb, &cr);

    // Full-range BT.601 (JFIF), matching the image-crate CPU conversion.
    float r = yv + 1.402f * (cr - 128.0f);
    float g = yv - 0.344136f * (cb - 128.0f) - 0.714136f * (cr - 128.0f);
    float b = yv + 1.772f * (cb - 128.0f);
    r = fminf(fmaxf(r, 0.0f), 255.0f);
    g = fminf(fmaxf(g, 0.0f), 255.0f);
    b = fminf(fmaxf(b, 0.0f), 255.0f);

    const size_t plane = (size_t)dst_w * dst_h;
    const size_t idx = (size_t)dy * dst_w + dx;
    dst_chw[idx] = r / 127.5f - 1.0f;
    dst_chw[plane + idx] = g / 127.5f - 1.0f;
    dst_chw[2 * plane + idx] = b / 127.5f - 1.0f;
}

// Exact device port of hipfire_arch_qwen35_vl::image::extract_patches.
// One thread per output element; inverts the CPU enumeration:
//
//   patch_out_idx = ((gy*gw + gx)*S + sy)*S + sx          (S = sms)
//   per-patch     = c*(T*P*P) + t*(P*P) + dy*P + dx
extern "C" __global__ void vl_extract_patches(const float* __restrict__ chw, unsigned int h,
                                              unsigned int w, unsigned int patch,
                                              unsigned int temporal, unsigned int sms,
                                              float* __restrict__ out) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t P = patch;
    const size_t patch_elems = (size_t)temporal * 3 * P * P;
    const size_t ph = h / P;
    const size_t pw = w / P;
    const size_t n_patches = ph * pw;
    if (idx >= n_patches * patch_elems) return;

    const size_t patch_out_idx = idx / patch_elems;
    const size_t e = idx % patch_elems;
    const size_t tpp = (size_t)temporal * P * P;
    const size_t c = e / tpp;
    const size_t r1 = e % tpp;
    const size_t t = r1 / (P * P);
    (void)t;  // same frame duplicated across temporal slots, as on CPU
    const size_t r2 = r1 % (P * P);
    const size_t dy = r2 / P;
    const size_t dx = r2 % P;

    const size_t S = sms;
    const size_t gw = pw / S;
    size_t tmp = patch_out_idx;
    const size_t isx = tmp % S;
    tmp /= S;
    const size_t isy = tmp % S;
    tmp /= S;
    const size_t gx = tmp % gw;
    const size_t gy = tmp / gw;
    const size_t py = gy * S + isy;
    const size_t px = gx * S + isx;

    const size_t y = py * P + dy;
    const size_t x = px * P + dx;
    out[idx] = chw[c * (size_t)h * w + y * w + x];
}
