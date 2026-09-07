// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vl_vcn_video_parity`: VCN H.264 → pixels parity vs the ffmpeg CPU oracle.
//!
//! Experiment `experiment/vcn-video` only; requires `--features vcn-video`
//! (default-off, no default-build-graph change).
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! flock -w 300 /tmp/hipfire-gpu.lock cargo run --release -p saddle-lab \
//!   --features vcn-video --example vl_vcn_video_parity
//! ```
//!
//! Input: the `-bf 0` Annex-B fixture + its source mp4. The CPU reference is
//! produced in-process by shelling out to ffmpeg
//! (`-f rawvideo -pix_fmt yuv420p`, ffmpeg's own H.264 decoder). The demuxer
//! is out of scope — Annex-B comes from
//! `ffmpeg -c:v copy -bsf:v h264_mp4toannexb` (see the va-bridge `h264`
//! module docs).
//!
//! Gates:
//! * decode: per-frame Y/U/V mean|max|d| (LSB) of VCN-derived NV12 vs
//!   ffmpeg yuv420p, all 60 frames. YUV-space comparison keeps the NV12→RGB
//!   conversion out of the decode gate (deliberate: ffmpeg's swscale RGB
//!   matrix is a fixed-point 601 variant ~1% off textbook BT.601, verified
//!   by least-squares fit with 0.7 LSB residual — comparing RGB would gate
//!   the conversion, not the decode).
//! * patches: rel-L1 of video-pair patch sets (consecutive frames fill the
//!   T=2 slots, Qwen3.5-VL video semantics) with both sides through the
//!   identical BT.601 conversion, isolating decode differences;
//! * timing: fps of VCN decode (+GPU kernel leg) vs ffmpeg CPU decode +
//!   CPU preprocess.
//!
//! The GPU `vl_nv12_to_rgb_norm` kernel leg is timed but NOT gated: that
//! kernel's math is tuned for JPEG full-range input (see the vcn-jpeg lane),
//! so a range/matrix gap vs limited-range video is expected and reported as
//! a diagnostic, not a failure.

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const PATCH: usize = 16;
const TEMPORAL: usize = 2; // consecutive frames fill the T slots (video)
const SMS: usize = 2;
const W: usize = 640;
const H: usize = 480;
const NFRAMES: usize = 60;

/// Textbook limited-range BT.601 NV12→RGB. Used identically on both sides of
/// the patches gate so the conversion cancels and only decode differs.
fn bt601_nv12_to_rgb(y: &[u8], uv: &[u8], w: usize, h: usize) -> Vec<u8> {
    assert_eq!(y.len(), w * h);
    let (cw, chh) = ((w + 1) / 2, (h + 1) / 2);
    assert_eq!(uv.len(), cw * chh * 2);
    let mut out = vec![0u8; w * h * 3];
    for r in 0..h {
        for c in 0..w {
            let yy = y[r * w + c] as f64;
            // NV12 interleaves U then V.
            let u = uv[(r / 2) * cw * 2 + (c / 2) * 2] as f64;
            let v = uv[(r / 2) * cw * 2 + (c / 2) * 2 + 1] as f64;
            let yv = (yy - 16.0) * (255.0 / 219.0);
            let du = u - 128.0;
            let dv = v - 128.0;
            out[(r * w + c) * 3] = (yv + 1.402 * dv).round().clamp(0.0, 255.0) as u8;
            out[(r * w + c) * 3 + 1] = (yv - 0.344_136 * du - 0.714_136 * dv)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[(r * w + c) * 3 + 2] = (yv + 1.772 * du).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn rgb_to_chw_norm(rgb: &[u8]) -> Vec<f32> {
    // rgb is HWC; produce CHW normalized like the repo CPU path.
    let plane = W * H;
    assert_eq!(rgb.len(), 3 * plane);
    let mut chw = vec![0f32; 3 * plane];
    for i in 0..plane {
        chw[i] = rgb[i * 3] as f32 / 127.5 - 1.0;
        chw[plane + i] = rgb[i * 3 + 1] as f32 / 127.5 - 1.0;
        chw[2 * plane + i] = rgb[i * 3 + 2] as f32 / 127.5 - 1.0;
    }
    chw
}

/// Video-pair patches mirroring `hipfire_arch_qwen35_vl::image::extract_patches`
/// layout (`[N, T*C*P*P]`, channel-outer temporal-inner, SMS-grouped), except
/// the T slots hold consecutive frames instead of duplicating one frame.
#[allow(clippy::too_many_arguments)]
fn extract_patches_video(
    chw0: &[f32],
    chw1: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    temporal: usize,
    sms: usize,
) -> Vec<f32> {
    assert_eq!(temporal, 2);
    let frames = [chw0, chw1];
    let (ph, pw) = (height / patch_size, width / patch_size);
    let patch_elems = temporal * channels * patch_size * patch_size;
    let mut patches = vec![0.0f32; ph * pw * patch_elems];
    let gw = pw / sms;
    for py in 0..ph {
        for px in 0..pw {
            let (gy, gx) = (py / sms, px / sms);
            let (sy, sx) = (py % sms, px % sms);
            let idx = ((gy * gw + gx) * sms + sy) * sms + sx;
            let out_base = idx * patch_elems;
            for c in 0..channels {
                for t in 0..temporal {
                    let chw = frames[t];
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let (y, x) = (py * patch_size + dy, px * patch_size + dx);
                            patches[out_base
                                + c * temporal * patch_size * patch_size
                                + t * patch_size * patch_size
                                + dy * patch_size
                                + dx] = chw[c * height * width + y * width + x];
                        }
                    }
                }
            }
        }
    }
    patches
}

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).abs();
        den += (*x as f64).abs();
    }
    num / den.max(1e-30)
}

fn plane_err(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let (mut sum, mut max) = (0u64, 0u8);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x.abs_diff(*y);
        sum += d as u64;
        max = max.max(d);
    }
    (sum as f64 / a.len() as f64, max)
}

/// Split ffmpeg yuv420p frame bytes into (Y, interleaved-UV) like DerivedFrame.
fn ref_nv12(frame: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (W, H);
    let (cw, chh) = (w / 2, h / 2);
    let y = frame[..w * h].to_vec();
    let u = &frame[w * h..w * h + cw * chh];
    let v = &frame[w * h + cw * chh..];
    let mut uv = vec![0u8; cw * chh * 2];
    for i in 0..cw * chh {
        uv[2 * i] = u[i];
        uv[2 * i + 1] = v[i];
    }
    (y, uv)
}

fn main() {
    let data_dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".hipfire/datasets/video");
    let annexb_path = data_dir.join("testsrc2_640x480_30fps_2s_bf0.h264");
    let mp4_path = data_dir.join("testsrc2_640x480_30fps_2s_bf0.mp4");
    let annexb = std::fs::read(&annexb_path).expect("annexb read");
    println!(
        "[vparity] annexb {}B mp4 {}B",
        annexb.len(),
        std::fs::metadata(&mp4_path).expect("mp4 stat").len()
    );

    // ── CPU reference: ffmpeg software decode to yuv420p ──
    let t0 = Instant::now();
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            mp4_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .output()
        .expect("ffmpeg spawn");
    assert!(out.status.success(), "ffmpeg decode failed");
    let cpu_dec_ms = t0.elapsed().as_secs_f64() * 1e3;
    let frame_bytes = W * H * 3 / 2;
    assert_eq!(out.stdout.len(), NFRAMES * frame_bytes, "reference bytes");
    println!("[vparity] ffmpeg CPU decode {cpu_dec_ms:.1}ms for {NFRAMES} frames");

    // ── VCN decode (derived NV12) ──
    let sess = va_bridge::VaSession::open_video(va_bridge::VA_PROFILE_H264_HIGH)
        .expect("open_video(H264High)");
    println!("[vparity] VA vendor: {}", sess.vendor());
    let t0 = Instant::now();
    let frames = sess.decode_h264_annexb(&annexb).expect("vcn decode");
    let vcn_dec_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(frames.len(), NFRAMES, "decoded frame count");
    // Eyeball rule: adjacent frames must differ (single-token attractor check).
    let y0: u64 = frames[0].y.iter().map(|&b| b as u64).sum();
    let y1: u64 = frames[1].y.iter().map(|&b| b as u64).sum();
    assert_ne!(y0, y1, "first two frames identical — suspect");

    // ── decode gate: Y/U/V planes vs ffmpeg yuv420p ──
    let t0 = Instant::now();
    let (mut worst_mean, mut worst_max) = (0f64, 0u8);
    println!(
        "{:>6} {:>8} {:>5} {:>8} {:>5} {:>8} {:>5}",
        "frame", "Y-mean", "Y-max", "U-mean", "U-max", "V-mean", "V-max"
    );
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width as usize, f.height as usize), (W, H));
        let (ry, ruv) = ref_nv12(&out.stdout[i * frame_bytes..(i + 1) * frame_bytes]);
        let (ym, yx) = plane_err(&f.y, &ry);
        // DerivedFrame.uv packs full rows; ref interleave matches by construction.
        assert_eq!(f.uv.len(), ruv.len());
        let (cw, chh) = (W / 2, H / 2);
        let (mut us, mut vs, mut um, mut vm) = (0u64, 0u64, 0u8, 0u8);
        for r in 0..chh {
            for c in 0..cw {
                let du = f.uv[r * cw * 2 + c * 2].abs_diff(ruv[r * cw * 2 + c * 2]);
                let dv = f.uv[r * cw * 2 + c * 2 + 1].abs_diff(ruv[r * cw * 2 + c * 2 + 1]);
                us += du as u64;
                vs += dv as u64;
                um = um.max(du);
                vm = vm.max(dv);
            }
        }
        let n = (cw * chh) as f64;
        let (umean, vmean) = (us as f64 / n, vs as f64 / n);
        worst_mean = worst_mean.max(ym).max(umean).max(vmean);
        worst_max = worst_max.max(yx).max(um).max(vm);
        if i < 5 || i >= NFRAMES - 2 {
            println!("{i:>6} {ym:>8.3} {yx:>5} {umean:>8.3} {um:>5} {vmean:>8.3} {vm:>5}");
        }
    }
    println!("[vparity] decode gate worst: mean={worst_mean:.3} max={worst_max}");

    // ── patches gate: video pairs, identical conversion both sides ──
    let t1 = Instant::now();
    let mut worst = 0f64;
    // Cache reference-side RGB (conversion dominates CPU time, not patches).
    let mut ref_rgb_all = Vec::with_capacity(NFRAMES);
    for k in 0..NFRAMES {
        let (ry, ruv) = ref_nv12(&out.stdout[k * frame_bytes..(k + 1) * frame_bytes]);
        ref_rgb_all.push(bt601_nv12_to_rgb(&ry, &ruv, W, H));
    }
    let mut vcn_rgb_all = Vec::with_capacity(NFRAMES);
    for f in &frames {
        vcn_rgb_all.push(bt601_nv12_to_rgb(&f.y, &f.uv, W, H));
    }
    for k in 0..NFRAMES / 2 {
        let a0 = rgb_to_chw_norm(&vcn_rgb_all[2 * k]);
        let a1 = rgb_to_chw_norm(&vcn_rgb_all[2 * k + 1]);
        let b0 = rgb_to_chw_norm(&ref_rgb_all[2 * k]);
        let b1 = rgb_to_chw_norm(&ref_rgb_all[2 * k + 1]);
        let pa = extract_patches_video(&a0, &a1, 3, H, W, PATCH, TEMPORAL, SMS);
        let pb = extract_patches_video(&b0, &b1, 3, H, W, PATCH, TEMPORAL, SMS);
        if k == 0 {
            assert_eq!(
                pa.len(),
                (H / PATCH) * (W / PATCH) * TEMPORAL * 3 * PATCH * PATCH
            );
        }
        let r = rel_l1(&pa, &pb);
        worst = worst.max(r);
        if k < 3 || k == NFRAMES / 2 - 1 {
            println!("[vparity] pair{k:02} rel-L1={r:.3e}");
        }
    }
    let cpu_patch_ms = t1.elapsed().as_secs_f64() * 1e3;
    let cpu_conv_ms = t0.elapsed().as_secs_f64() * 1e3 - cpu_patch_ms;
    println!("[vparity] patches gate worst rel-L1 = {worst:.3e}");

    // ── GPU kernel leg (timed, diagnostic): derived NV12 upload →
    // vl_nv12_to_rgb_norm + vl_extract_patches per frame ──
    let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    let stream = hip.stream_create().expect("stream_create");
    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/vl_yuv_preprocess.hip");
    let tmp = std::env::temp_dir().join("vcn_video_parity");
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
    let module = hip
        .module_load_data(&std::fs::read(&hsaco).unwrap())
        .expect("module_load_data");
    let k_rgb = hip
        .module_get_function(&module, "vl_nv12_to_rgb_norm")
        .expect("vl_nv12_to_rgb_norm");
    let k_patch = hip
        .module_get_function(&module, "vl_extract_patches")
        .expect("vl_extract_patches");
    // 640x480 is already smart-resize-stable (32-px aligned, in range).
    let n_chw = 3 * H * W;
    let n_elem = (H / PATCH) * (W / PATCH) * TEMPORAL * 3 * PATCH * PATCH;
    let t0 = Instant::now();
    let mut gpu_first: Vec<f32> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let mut nv12 = Vec::with_capacity(f.y.len() + f.uv.len());
        nv12.extend_from_slice(&f.y);
        nv12.extend_from_slice(&f.uv);
        let d_surf = hip.malloc(nv12.len()).expect("malloc surf");
        hip.memcpy_htod(&d_surf, &nv12).expect("htod");
        let d_chw = hip.malloc(n_chw * 4).expect("malloc chw");
        let d_out = hip.malloc(n_elem * 4).expect("malloc patches");
        let mut surf_a = d_surf.as_ptr() as u64;
        let mut chw_a = d_chw.as_ptr() as u64;
        let (mut y_pitch, mut uv_pitch) = (W as u32, W as u32);
        let (mut u_off, mut v_off, mut step) = ((W * H) as u32, (W * H) as u32 + 1, 2u32);
        let (mut src_w, mut src_h) = (W as u32, H as u32);
        let (mut sh_v, mut sv_v) = (1u32, 1u32);
        let (mut dst_w, mut dst_h) = (W as u32, H as u32);
        let mut p1: Vec<*mut c_void> = vec![
            (&mut surf_a as *mut u64).cast(),
            (&mut y_pitch as *mut u32).cast(),
            (&mut uv_pitch as *mut u32).cast(),
            (&mut u_off as *mut u32).cast(),
            (&mut v_off as *mut u32).cast(),
            (&mut step as *mut u32).cast(),
            (&mut src_w as *mut u32).cast(),
            (&mut src_h as *mut u32).cast(),
            (&mut sh_v as *mut u32).cast(),
            (&mut sv_v as *mut u32).cast(),
            (&mut dst_w as *mut u32).cast(),
            (&mut dst_h as *mut u32).cast(),
            (&mut chw_a as *mut u64).cast(),
        ];
        unsafe {
            hip.launch_kernel(
                &k_rgb,
                [(W as u32 + 15) / 16, (H as u32 + 15) / 16, 1],
                [16, 16, 1],
                0,
                Some(&stream),
                &mut p1,
            )
            .expect("launch rgb");
        }
        let mut chw_b = chw_a;
        let (mut h_v, mut w_v) = (H as u32, W as u32);
        let (mut p_v, mut t_v, mut s_v) = (PATCH as u32, TEMPORAL as u32, SMS as u32);
        let mut out_b = d_out.as_ptr() as u64;
        let mut p2: Vec<*mut c_void> = vec![
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
                &mut p2,
            )
            .expect("launch patches");
        }
        hip.stream_synchronize(&stream).expect("sync");
        let mut raw = vec![0u8; n_elem * 4];
        hip.memcpy_dtoh(&mut raw, &d_out).expect("dtoh");
        let v: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_elem) }.to_vec();
        if i == 0 {
            gpu_first = v;
        }
        hip.free(d_surf).unwrap();
        hip.free(d_chw).unwrap();
        hip.free(d_out).unwrap();
    }
    let gpu_kern_ms = t0.elapsed().as_secs_f64() * 1e3;
    // Diagnostic only: kernel math is full-range/JPEG-tuned; compare against
    // the CPU single-frame-duplicated path built from the same VCN frame.
    let c0 = rgb_to_chw_norm(&vcn_rgb_all[0]);
    let cpu_dup = extract_patches_video(&c0, &c0, 3, H, W, PATCH, TEMPORAL, SMS);
    println!(
        "[vparity] gpu-kernel diagnostic rel-L1 vs cpu-dup = {:.3e} (expected nonzero: range math differs)",
        rel_l1(&gpu_first, &cpu_dup)
    );

    // ── fps table ──
    let cpu_total_ms = cpu_dec_ms + cpu_conv_ms + cpu_patch_ms;
    let gpu_total_ms = vcn_dec_ms + gpu_kern_ms;
    println!("{:=<72}", "");
    println!("[vparity] fps table ({NFRAMES} frames 640x480):");
    println!(
        "[vparity]   ffmpeg CPU decode:      {cpu_dec_ms:8.1}ms  {:7.1} fps",
        1e3 / (cpu_dec_ms / NFRAMES as f64)
    );
    println!(
        "[vparity]   VCN decode (derived):   {vcn_dec_ms:8.1}ms  {:7.1} fps",
        1e3 / (vcn_dec_ms / NFRAMES as f64)
    );
    println!(
        "[vparity]   CPU BT.601+patches:     {:8.1}ms  {:7.1} fps",
        cpu_conv_ms + cpu_patch_ms,
        1e3 / ((cpu_conv_ms + cpu_patch_ms) / NFRAMES as f64)
    );
    println!(
        "[vparity]   GPU kernels (up+2k):    {gpu_kern_ms:8.1}ms  {:7.1} fps",
        1e3 / (gpu_kern_ms / NFRAMES as f64)
    );
    println!(
        "[vparity]   end-to-end CPU: {:8.1}ms  {:7.1} fps",
        cpu_total_ms,
        1e3 / (cpu_total_ms / NFRAMES as f64)
    );
    println!(
        "[vparity]   end-to-end VCN+GPU: {:8.1}ms  {:7.1} fps",
        gpu_total_ms,
        1e3 / (gpu_total_ms / NFRAMES as f64)
    );
    println!(
        "[vparity] decode gate worst mean={worst_mean:.3} worst max={worst_max} patches worst rel-L1={worst:.3e}"
    );
}
