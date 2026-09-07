// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vl_vcn_debug`: staged bisection of the VCN parity gap (3.5e-1 on general_qa).
//!
//! Experiment `experiment/vcn-jpeg` only; requires `--features vcn-jpeg`.
//! Run: `flock -w 60 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/vl_vcn_debug [fixture]`
//!
//! Stages (each reports its own rel-L1 / LSB stats, isolating one suspect):
//! - E: `vl_extract_patches` on oracle CHW vs CPU `extract_patches` (exact; no VCN).
//! - A: VCN Y plane (D2H) vs turbo-decoded luma BT.601 full-range + linear fit
//!   (detects pitch/offset/range errors).
//! - B: VCN UV plane (D2H) vs turbo box-decimated Cb/Cr in both orders
//!   (detects U/V swap, subsampling shift).
//! - D: CPU-only host port of the kernel's separable CatmullRom on turbo RGB
//!   vs `image::imageops::resize` (detects filter mismatch, no GPU/VCN).

use hipfire_arch_qwen35_vl::image::{extract_patches, load_and_preprocess_from_bytes};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;

const PATCH: usize = 16;
const TEMPORAL: usize = 2;
const SMS: usize = 2;
const FACTOR: usize = PATCH * SMS;
const MIN_PX: usize = 65_536;
const MAX_PX: usize = 2_000_000;

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).abs();
        den += (*x as f64).abs();
    }
    num / den.max(1e-30)
}

fn byte_stats(name: &str, got: &[u8], refr: &[f32]) {
    assert_eq!(got.len(), refr.len());
    let n = got.len() as f64;
    let (mut sad, mut max) = (0.0f64, 0.0f32);
    let (mut sg, mut sr, mut srr, mut sgr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, r) in got.iter().zip(refr.iter()) {
        let d = (*g as f32 - *r).abs();
        sad += d as f64;
        max = max.max(d);
        let (gf, rf) = (*g as f64, *r as f64);
        sg += gf;
        sr += rf;
        srr += rf * rf;
        sgr += gf * rf;
    }
    // Least-squares fit got ~= a*ref + b (detects range scaling/offset).
    let var = srr - sr * sr / n;
    let a = if var.abs() < 1e-9 {
        0.0
    } else {
        (sgr - sg * sr / n) / var
    };
    let b = sg / n - a * sr / n;
    println!(
        "[stage] {name}: mean_abs={:.3} LSB max_abs={:.1} fit_a={:.4} fit_b={:.2} (n={})",
        sad / n,
        max,
        a,
        b,
        got.len()
    );
}

fn plane_stats(name: &str, bytes: &[u8]) {
    let n = bytes.len() as f64;
    let (mut sum, mut min, mut max) = (0u64, 255u8, 0u8);
    let mut hist = [0u64; 16];
    for b in bytes {
        sum += *b as u64;
        min = min.min(*b);
        max = max.max(*b);
        hist[(*b as usize) / 16] += 1;
    }
    print!(
        "[stage] {name}: mean={:.1} min={min} max={max} hist16=[",
        sum as f64 / n
    );
    for (i, h) in hist.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{}", (*h as f64 / n * 100.0) as u32);
    }
    println!("]% first32={}", hex32(bytes));
}

fn hex32(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(32)
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join("")
}
// ——— host port of the kernel's separable CatmullRom (stage D) ———
fn cubic(x: f32) -> f32 {
    let x = x.abs();
    let (x2, x3) = (x * x, x * x * x);
    if x <= 1.0 {
        1.5 * x3 - 2.5 * x2 + 1.0
    } else if x <= 2.0 {
        -0.5 * x3 + 2.5 * x2 - 4.0 * x + 2.0
    } else {
        0.0
    }
}

fn sample_plane(img: &[u8], stride: usize, w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let x0 = fx.floor() as i32 - 1;
    let y0 = fy.floor() as i32 - 1;
    let (mut acc, mut wsum) = (0.0f32, 0.0f32);
    for j in 0..4 {
        let yy = (y0 + j).clamp(0, h as i32 - 1) as usize;
        let wy = cubic(fy - (y0 + j) as f32);
        for i in 0..4 {
            let xx = (x0 + i).clamp(0, w as i32 - 1) as usize;
            let wx = cubic(fx - (x0 + i) as f32);
            acc += wx * wy * img[yy * stride + xx] as f32;
            wsum += wx * wy;
        }
    }
    acc / wsum
}

fn main() {
    let fixture = std::env::args().nth(1).unwrap_or("general_qa.jpg".to_string());
    let img_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/vision/images");
    let path: PathBuf = if fixture.contains('/') {
        PathBuf::from(&fixture)
    } else {
        img_dir.join(&fixture)
    };
    let bytes = std::fs::read(&path).expect("fixture read");
    let (pixels, img_h, img_w) =
        load_and_preprocess_from_bytes(&bytes, PATCH, SMS).expect("cpu preprocess");
    let cpu_patches = extract_patches(&pixels, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
    let turbo = libjpeg_turbo_rs::decompress_to(&bytes, libjpeg_turbo_rs::PixelFormat::Rgb)
        .expect("turbo decode");
    let (tw, th) = (turbo.width, turbo.height);
    println!("[debug] {fixture}: turbo {tw}x{th}, target {img_w}x{img_h}");

    // ——— Stage D (CPU-only): kernel-filter port vs image-crate resize ———
    {
        let rgb = &turbo.data;
        let resized = image::load_from_memory(&bytes)
            .expect("image decode")
            .resize_exact(img_w as u32, img_h as u32, image::imageops::FilterType::CatmullRom)
            .to_rgb8();
        let mut host = vec![0u8; 3 * img_w * img_h];
        for c in 0..3 {
            let plane: Vec<u8> = rgb.iter().skip(c).step_by(3).copied().collect();
            for dy in 0..img_h {
                let sy = (dy as f32 + 0.5) * (th as f32 / img_h as f32) - 0.5;
                for dx in 0..img_w {
                    let sx = (dx as f32 + 0.5) * (tw as f32 / img_w as f32) - 0.5;
                    host[(dy * img_w + dx) * 3 + c] =
                        sample_plane(&plane, tw, tw, th, sx, sy).clamp(0.0, 255.0) as u8;
                }
            }
        }
        let raw = resized.as_raw();
        let (mut sad, mut max) = (0u64, 0u8);
        for (a, b) in host.iter().zip(raw.iter()) {
            let d = a.abs_diff(*b);
            sad += d as u64;
            max = max.max(d);
        }
        println!(
            "[stage] D filter-port vs image-resize: mean_abs={:.3} LSB max_abs={max} (n={})",
            sad as f64 / raw.len() as f64,
            raw.len()
        );
    }

    // ——— HIP + kernels ———
    let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    let stream = hip.stream_create().expect("stream_create");
    let ksrc = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/src/vl_yuv_preprocess.hip");
    let tmp = std::env::temp_dir().join("vcn_debug");
    std::fs::create_dir_all(&tmp).unwrap();
    let hsaco = tmp.join("vl_yuv_preprocess.hsaco");
    let st = Command::new("hipcc")
        .args([
            "--genco",
            "-o",
            hsaco.to_str().unwrap(),
            &format!("--offload-arch={arch}"),
            ksrc.to_str().unwrap(),
        ])
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed");
    let image = std::fs::read(&hsaco).unwrap();
    let module = hip.module_load_data(&image).expect("module_load_data");
    let k_patch = hip
        .module_get_function(&module, "vl_extract_patches")
        .expect("vl_extract_patches");

    // ——— Stage E: patch kernel on oracle CHW (exact) ———
    {
        let d_chw = hip.malloc(pixels.len() * 4).expect("malloc chw");
        let chw_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        hip.memcpy_htod(&d_chw, chw_bytes).expect("htod");
        let n_elem = cpu_patches.len();
        let d_out = hip.malloc(n_elem * 4).expect("malloc patches");
        let mut chw_b = d_chw.as_ptr() as u64;
        let mut h_v = img_h as u32;
        let mut w_v = img_w as u32;
        let mut p_v = PATCH as u32;
        let mut t_v = TEMPORAL as u32;
        let mut s_v = SMS as u32;
        let mut out_b = d_out.as_ptr() as u64;
        let mut p: Vec<*mut c_void> = vec![
            (&mut chw_b as *mut u64).cast(),
            (&mut h_v as *mut u32).cast(),
            (&mut w_v as *mut u32).cast(),
            (&mut p_v as *mut u32).cast(),
            (&mut t_v as *mut u32).cast(),
            (&mut s_v as *mut u32).cast(),
            (&mut out_b as *mut u64).cast(),
        ];
        unsafe {
            hip.launch_kernel(
                &k_patch,
                [(n_elem as u32 + 255) / 256, 1, 1],
                [256, 1, 1],
                0,
                Some(&stream),
                &mut p,
            )
            .expect("launch patches");
        }
        hip.stream_synchronize(&stream).expect("sync");
        let mut raw = vec![0u8; n_elem * 4];
        hip.memcpy_dtoh(&mut raw, &d_out).expect("dtoh");
        let gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_elem) };
        let (mut maxd, mut first) = (0.0f32, usize::MAX);
        for (i, (a, b)) in cpu_patches.iter().zip(gpu.iter()).enumerate() {
            let d = (a - b).abs();
            if d > maxd {
                maxd = d;
                first = i;
            }
        }
        println!(
            "[stage] E patch-kernel vs cpu: rel-L1={:.3e} max_abs={:.3e} first_mismatch={} (n={})",
            rel_l1(&cpu_patches, gpu),
            maxd,
            if maxd > 0.0 { first.to_string() } else { "-".to_string() },
            n_elem
        );
        hip.free(d_chw).unwrap();
        hip.free(d_out).unwrap();
    }

    // ——— VCN decode + D2H of raw planes ———
    let mut sess = va_bridge::VaSession::open().expect("VA open");
    let frame = match sess.decode_jpeg(&bytes).expect("vcn decode") {
        va_bridge::DecodeOutcome::Decoded(f) => f,
        va_bridge::DecodeOutcome::Unsupported(reason) => panic!("fixture unsupported: {reason}"),
    };
    assert_eq!(frame.fourcc, va_bridge::VA_FOURCC_NV12);
    let (sw, sh) = (frame.width as usize, frame.height as usize);
    let (yp, uvo, uvp) = (
        frame.y_pitch() as usize,
        frame.uv_offset() as usize,
        frame.uv_pitch() as usize,
    );
    println!("[debug] surface {sw}x{sh} y_pitch={yp} uv_off={uvo} uv_pitch={uvp}");
    let y_raw = frame.copy_to_host(0, yp * sh).expect("y dtoh");
    let y_plane: Vec<u8> = (0..sh)
        .flat_map(|r| y_raw[r * yp..r * yp + sw].iter().copied())
        .collect();
    plane_stats("A y-plane", &y_plane);
    // ——— Stage A2: turbo reference stats + viewable dumps ———
    let y8: Vec<u8> = turbo
        .data
        .chunks_exact(3)
        .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).clamp(0.0, 255.0) as u8)
        .collect();
    plane_stats("A turbo-luma", &y8);
    image::GrayImage::from_raw(sw as u32, sh as u32, y_plane.clone())
        .expect("gray vcn")
        .save("/tmp/vcn_y.png")
        .expect("save vcn");
    image::GrayImage::from_raw(sw as u32, sh as u32, y8)
        .expect("gray turbo")
        .save("/tmp/turbo_y.png")
        .expect("save turbo");
    println!("[debug] wrote /tmp/vcn_y.png /tmp/turbo_y.png");

    // ——— Stage A: Y vs turbo luma ———
    {
        let mut y_ref = vec![0f32; sw * sh];
        for i in 0..sw * sh {
            let (r, g, b) = (
                turbo.data[3 * i] as f32,
                turbo.data[3 * i + 1] as f32,
                turbo.data[3 * i + 2] as f32,
            );
            y_ref[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        byte_stats("A y-vs-turbo-luma", &y_plane, &y_ref);
    }

    // ——— Stage B: UV vs turbo box-decimated chroma, both orders ———
    {
        let (cw, chh) = (sw / 2, sh / 2);
        let uv_raw = frame.copy_to_host(uvo, uvp * chh).expect("uv dtoh");
        let (mut b0, mut b1) = (vec![0u8; cw * chh], vec![0u8; cw * chh]);
        for r in 0..chh {
            for c in 0..cw {
                b0[r * cw + c] = uv_raw[r * uvp + 2 * c];
                b1[r * cw + c] = uv_raw[r * uvp + 2 * c + 1];
            }
        }
        let mut cb_ref = vec![0f32; cw * chh];
        let mut cr_ref = vec![0f32; cw * chh];
        for r in 0..chh {
            for c in 0..cw {
                let (mut cb, mut cr) = (0.0f32, 0.0f32);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let (yy, xx) = (2 * r + dy, 2 * c + dx);
                    let i = yy * sw + xx;
                    let (rf, gf, bf) = (
                        turbo.data[3 * i] as f32,
                        turbo.data[3 * i + 1] as f32,
                        turbo.data[3 * i + 2] as f32,
                    );
                    cb += -0.168736 * rf - 0.331264 * gf + 0.5 * bf + 128.0;
                    cr += 0.5 * rf - 0.418688 * gf - 0.081312 * bf + 128.0;
                }
                cb_ref[r * cw + c] = cb / 4.0;
                cr_ref[r * cw + c] = cr / 4.0;
            }
        }
        byte_stats("B b0-vs-Cb", &b0, &cb_ref);
        byte_stats("B b1-vs-Cr", &b1, &cr_ref);
        byte_stats("B b0-vs-Cr(SWAP)", &b0, &cr_ref);
        plane_stats("B uv-b0", &b0);
        plane_stats("B uv-b1", &b1);
    }
    // ——— Stage F: rgb kernel on synthetic NV12 (no VCN, no resize) ———
    {
        let k_rgb = hip
            .module_get_function(&module, "vl_nv12_to_rgb_norm")
            .expect("vl_nv12_to_rgb_norm");
        const W: usize = 64;
        const H: usize = 64;
        const CW: usize = W / 2;
        const CH: usize = H / 2;
        // Y gradient 16..240 down rows; chroma constant Cb=100, Cr=150.
        let mut nv12 = vec![0u8; W * H + CW * 2 * CH];
        for r in 0..H {
            let y = (16.0 + 224.0 * r as f32 / (H - 1) as f32) as u8;
            for c in 0..W {
                nv12[r * W + c] = y;
            }
        }
        for r in 0..CH {
            for c in 0..CW {
                nv12[W * H + r * W + 2 * c] = 100;
                nv12[W * H + r * W + 2 * c + 1] = 150;
            }
        }
        let d_surf = hip.malloc(nv12.len()).expect("malloc surf");
        hip.memcpy_htod(&d_surf, &nv12).expect("htod surf");
        let d_chw = hip.malloc(3 * W * H * 4).expect("malloc chw");
        let mut surf_a = d_surf.as_ptr() as u64;
        let mut chw_a = d_chw.as_ptr() as u64;
        let mut p0 = W as u32;
        let mut p1 = W as u32;
        let mut off = (W * H) as u32;
        let mut voff = (W * H) as u32 + 1;
        let mut cstep = 2u32;
        let mut sw_ = W as u32;
        let mut sh_ = H as u32;
        let mut shx = 1u32;
        let mut shy = 1u32;
        let mut dw = W as u32;
        let mut dh = H as u32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut surf_a as *mut u64).cast(),
            (&mut p0 as *mut u32).cast(),
            (&mut p1 as *mut u32).cast(),
            (&mut off as *mut u32).cast(),
            (&mut voff as *mut u32).cast(),
            (&mut cstep as *mut u32).cast(),
            (&mut sw_ as *mut u32).cast(),
            (&mut sh_ as *mut u32).cast(),
            (&mut shx as *mut u32).cast(),
            (&mut shy as *mut u32).cast(),
            (&mut dw as *mut u32).cast(),
            (&mut dh as *mut u32).cast(),
            (&mut chw_a as *mut u64).cast(),
        ];
        unsafe {
            hip.launch_kernel(&k_rgb, [(W as u32 + 15) / 16, (H as u32 + 15) / 16, 1], [16, 16, 1], 0, Some(&stream), &mut args)
                .expect("launch rgb");
        }
        hip.stream_synchronize(&stream).expect("sync");
        let mut raw = vec![0u8; 3 * W * H * 4];
        hip.memcpy_dtoh(&mut raw, &d_chw).expect("dtoh");
        let gpu: &[f32] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, 3 * W * H) };
        // Expected: constant chroma, Y varies per row; check interior rows.
        let (mut sad, mut maxd, mut n) = (0.0f64, 0.0f32, 0usize);
        for r in 2..H - 2 {
            let y = 16.0 + 224.0 * r as f32 / (H - 1) as f32;
            let rr = (y + 1.402 * (150.0 - 128.0)).clamp(0.0, 255.0) / 127.5 - 1.0;
            let gg = (y - 0.344136 * (100.0 - 128.0) - 0.714136 * (150.0 - 128.0)).clamp(0.0, 255.0) / 127.5 - 1.0;
            let bb = (y + 1.772 * (100.0 - 128.0)).clamp(0.0, 255.0) / 127.5 - 1.0;
            for c in 2..W - 2 {
                let i = r * W + c;
                for (ch, exp) in [(0, rr), (1, gg), (2, bb)] {
                    let d = (gpu[ch * W * H + i] - exp).abs();
                    sad += d as f64;
                    maxd = maxd.max(d);
                    n += 1;
                }
            }
        }
        println!(
            "[stage] F rgb-kernel synthetic: mean_abs={:.3e} max_abs={:.3e} (n={n})",
            sad / n as f64,
            maxd
        );
        hip.free(d_surf).unwrap();
        hip.free(d_chw).unwrap();
    }
    // ——— Stage G: planes-oracle readback vs turbo ———
    {
        let now = std::time::Instant::now();
        let d = sess.decode_jpeg_planes(&bytes).expect("planes decode");
        assert_eq!(d.planes.len(), 2, "stage G expects NV12");
        let (dy, duv) = (&d.planes[0], &d.planes[1]);
        let ms = now.elapsed().as_secs_f64() * 1e3;
        println!(
            "[debug] planes {}x{} fourcc=0x{:08x} in {:.2}ms",
            d.width, d.height, d.fourcc, ms
        );
        plane_stats("G derived-y", dy);
        let mut y_ref = vec![0f32; sw * sh];
        for i in 0..sw * sh {
            y_ref[i] = 0.299 * turbo.data[3 * i] as f32
                + 0.587 * turbo.data[3 * i + 1] as f32
                + 0.114 * turbo.data[3 * i + 2] as f32;
        }
        byte_stats("G derived-y-vs-turbo-luma", dy, &y_ref);
        let (cw, chh) = (sw / 2, sh / 2);
        let (mut b0, mut b1) = (vec![0u8; cw * chh], vec![0u8; cw * chh]);
        for r in 0..chh {
            for c in 0..cw {
                b0[r * cw + c] = duv[r * cw * 2 + c * 2];
                b1[r * cw + c] = duv[r * cw * 2 + c * 2 + 1];
            }
        }
        let mut cb_ref = vec![0f32; cw * chh];
        let mut cr_ref = vec![0f32; cw * chh];
        for r in 0..chh {
            for c in 0..cw {
                let (mut cb, mut cr) = (0.0f32, 0.0f32);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let i = (2 * r + dy) * sw + (2 * c + dx);
                    cb += -0.168736 * turbo.data[3 * i] as f32 - 0.331264 * turbo.data[3 * i + 1] as f32
                        + 0.5 * turbo.data[3 * i + 2] as f32
                        + 128.0;
                    cr += 0.5 * turbo.data[3 * i] as f32 - 0.418688 * turbo.data[3 * i + 1] as f32
                        - 0.081312 * turbo.data[3 * i + 2] as f32
                        + 128.0;
                }
                cb_ref[r * cw + c] = cb / 4.0;
                cr_ref[r * cw + c] = cr / 4.0;
            }
        }
        byte_stats("G b0-vs-Cb", &b0, &cb_ref);
        byte_stats("G b1-vs-Cr", &b1, &cr_ref);
        image::GrayImage::from_raw(d.width, d.height, dy.clone())
            .expect("gray derived")
            .save("/tmp/vcn_derived_y.png")
            .expect("save derived");
        println!("[debug] wrote /tmp/vcn_derived_y.png");
    }
    drop(frame);
    println!("[debug] done");
}


