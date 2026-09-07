//! Rust-native port of `kernels/src/vl_yuv_preprocess.hip` (experiment only).
//!
//! Same arithmetic order, clamping, and constants as the HIP source of truth.
//! Kernel symbols and kernarg layout are byte-identical to the hipcc build.

#![no_std]
#![feature(abi_gpu_kernel, link_llvm_intrinsics, asm_experimental_arch, core_intrinsics)]
#![allow(internal_features)]

use core::intrinsics::{fabs, floorf32};
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn tid_x() -> u32;
    #[link_name = "llvm.amdgcn.workitem.id.y"]
    fn tid_y() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn wg_x() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.y"]
    fn wg_y() -> u32;
}

/// Workgroup shape is part of the kernel contract, fixed in the harness
/// launch config — NOT read from the dispatch packet.
pub const WG_RGB: (u32, u32) = (16, 16);
/// Workgroup shape is part of the kernel contract, fixed in the harness
/// launch config — NOT read from the dispatch packet.
pub const WG_PATCH: u32 = 256;

// Keys cubic with a = 0.5 (Catmull-Rom: Mitchell B=0, C=0.5). Term-identical
// to image-0.25.10 `bc_cubic_spline(x, 0.0, 0.5)`.
unsafe fn vl_cubic(x: f32) -> f32 {
    let ax = fabs(x);
    let x2 = ax * ax;
    let x3 = x2 * ax;
    if ax <= 1.0 {
        return 1.5 * x3 - 2.5 * x2 + 1.0;
    }
    if ax <= 2.0 {
        return -0.5 * x3 + 2.5 * x2 - 4.0 * ax + 2.0;
    }
    0.0
}

#[inline(always)]
fn vl_clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

// Separable CatmullRom sample of one single-byte plane.
#[inline(always)]
unsafe fn vl_sample_plane(base: *const u8, pitch: i32, w: i32, h: i32, fx: f32, fy: f32) -> f32 {
    let x0 = floorf32(fx) as i32 - 1;
    let y0 = floorf32(fy) as i32 - 1;
    let mut acc = 0.0f32;
    let mut wsum = 0.0f32;
    let mut j = 0;
    while j < 4 {
        let yy = vl_clampi(y0 + j, 0, h - 1);
        let wy = vl_cubic(fy - (y0 + j) as f32);
        let row = base.add(yy as usize * pitch as usize);
        let mut i = 0;
        while i < 4 {
            let xx = vl_clampi(x0 + i, 0, w - 1);
            let wx = vl_cubic(fx - (x0 + i) as f32);
            let wgt = wx * wy;
            acc += wgt * *row.add(xx as usize) as f32;
            wsum += wgt;
            i += 1;
        }
        j += 1;
    }
    if wsum != 0.0 { acc / wsum } else { acc }
}

// Chroma sample from either an interleaved plane (NV12/NV24: u_base = plane,
// v_base = plane + 1, step 2) or two planar planes (444P/422P: step 1).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn vl_sample_chroma(
    u_base: *const u8,
    v_base: *const u8,
    pitch: i32,
    step: i32,
    w2: i32,
    h2: i32,
    fx: f32,
    fy: f32,
) -> (f32, f32) {
    let x0 = floorf32(fx) as i32 - 1;
    let y0 = floorf32(fy) as i32 - 1;
    let mut au = 0.0f32;
    let mut av = 0.0f32;
    let mut wsum = 0.0f32;
    let mut j = 0;
    while j < 4 {
        let yy = vl_clampi(y0 + j, 0, h2 - 1);
        let wy = vl_cubic(fy - (y0 + j) as f32);
        let row_u = u_base.add(yy as usize * pitch as usize);
        let row_v = v_base.add(yy as usize * pitch as usize);
        let mut i = 0;
        while i < 4 {
            let xx = vl_clampi(x0 + i, 0, w2 - 1);
            let wx = vl_cubic(fx - (x0 + i) as f32);
            let wgt = wx * wy;
            au += wgt * *row_u.add((step * xx) as usize) as f32;
            av += wgt * *row_v.add((step * xx) as usize) as f32;
            wsum += wgt;
            i += 1;
        }
        j += 1;
    }
    if wsum != 0.0 {
        au /= wsum;
        av /= wsum;
    }
    (au, av)
}

#[no_mangle]
pub unsafe extern "gpu-kernel" fn vl_nv12_to_rgb_norm(
    surf_base: *const u8,
    y_pitch: u32,
    uv_pitch: u32,
    u_offset: u32,
    v_offset: u32,
    chroma_step: u32,
    src_w: u32,
    src_h: u32,
    chroma_shift_x: u32,
    chroma_shift_y: u32,
    dst_w: u32,
    dst_h: u32,
    dst_chw: *mut f32,
) {
    let dx = wg_x() * WG_RGB.0 + tid_x();
    let dy = wg_y() * WG_RGB.1 + tid_y();
    if dx >= dst_w || dy >= dst_h {
        return;
    }

    // dst-centre mapping, same convention as image::imageops resize.
    let sx = (dx as f32 + 0.5) * (src_w as f32 / dst_w as f32) - 0.5;
    let sy = (dy as f32 + 0.5) * (src_h as f32 / dst_h as f32) - 0.5;

    let yv = vl_sample_plane(surf_base, y_pitch as i32, src_w as i32, src_h as i32, sx, sy);

    let u_base = surf_base.add(u_offset as usize);
    let v_base = surf_base.add(v_offset as usize);
    let cw = ((src_w + (1u32 << chroma_shift_x) - 1) >> chroma_shift_x) as i32;
    let ch = ((src_h + (1u32 << chroma_shift_y) - 1) >> chroma_shift_y) as i32;
    // Centered chroma siting (JFIF: chroma[i] is the 2x2-block average,
    // nominally at luma S*i + (S-1)/2), matching libjpeg's fancy-upsample
    // phase.
    let cx =
        (sx - 0.5 * ((1u32 << chroma_shift_x) - 1) as f32) / (1u32 << chroma_shift_x) as f32;
    let cy =
        (sy - 0.5 * ((1u32 << chroma_shift_y) - 1) as f32) / (1u32 << chroma_shift_y) as f32;
    let (cb, cr) = vl_sample_chroma(
        u_base,
        v_base,
        uv_pitch as i32,
        chroma_step as i32,
        cw,
        ch,
        cx,
        cy,
    );

    // Full-range BT.601 (JFIF), matching the image-crate CPU conversion.
    let mut r = yv + 1.402 * (cr - 128.0);
    let mut g = yv - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0);
    let mut b = yv + 1.772 * (cb - 128.0);
    // f32::max/min are llvm.maxnum/minnum: the same lowering as HIP's fmaxf/fminf.
    r = r.max(0.0).min(255.0);
    g = g.max(0.0).min(255.0);
    b = b.max(0.0).min(255.0);

    let plane = dst_w as usize * dst_h as usize;
    let idx = dy as usize * dst_w as usize + dx as usize;
    *dst_chw.add(idx) = r / 127.5 - 1.0;
    *dst_chw.add(plane + idx) = g / 127.5 - 1.0;
    *dst_chw.add(2 * plane + idx) = b / 127.5 - 1.0;
}

// Exact device port of hipfire_arch_qwen35_vl::image::extract_patches.
// One thread per output element; inverts the CPU enumeration:
//
//   patch_out_idx = ((gy*gw + gx)*S + sy)*S + sx          (S = sms)
//   per-patch     = c*(T*P*P) + t*(P*P) + dy*P + dx
#[no_mangle]
pub unsafe extern "gpu-kernel" fn vl_extract_patches(
    chw: *const f32,
    h: u32,
    w: u32,
    patch: u32,
    temporal: u32,
    sms: u32,
    out: *mut f32,
) {
    let idx = (wg_x() * WG_PATCH + tid_x()) as u64;
    let p = patch as u64;
    let patch_elems = temporal as u64 * 3 * p * p;
    let ph = h as u64 / p;
    let pw = w as u64 / p;
    let n_patches = ph * pw;
    if idx >= n_patches * patch_elems {
        return;
    }

    let patch_out_idx = idx / patch_elems;
    let e = idx % patch_elems;
    let tpp = temporal as u64 * p * p;
    let c = e / tpp;
    let r1 = e % tpp;
    let t = r1 / (p * p);
    let _ = t; // same frame duplicated across temporal slots, as on CPU
    let r2 = r1 % (p * p);
    let dy = r2 / p;
    let dx = r2 % p;

    let s = sms as u64;
    let gw = pw / s;
    let mut tmp = patch_out_idx;
    let isx = tmp % s;
    tmp /= s;
    let isy = tmp % s;
    tmp /= s;
    let gx = tmp % gw;
    let gy = tmp / gw;
    let py = gy * s + isy;
    let px = gx * s + isx;

    let y = py * p + dy;
    let x = px * p + dx;
    *out.add(idx as usize) = *chw.add((c * h as u64 * w as u64 + y * w as u64 + x) as usize);
}
