// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vl_vcn_parity`: VCN JPEG → `pixel_values` parity vs the CPU oracles.
//!
//! Experiment `experiment/vcn-jpeg` only; requires `--features vcn-jpeg`
//! (default-off, no default-build-graph change).
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! source scripts/gpu-lock.sh && gpu_acquire vcn-jpeg \
//!   && cargo run --release -p saddle-lab --features vcn-jpeg --example vl_vcn_parity; gpu_release
//! ```
//! (The gpu-lock.sh `exec`-fd idiom fails under the omp sandbox shell;
//! there `flock -w 60 /tmp/hipfire-gpu.lock <cmd>` holds the same mutex.)
//!
//! Per fixture (benchmarks/vision/images): repo CPU path
//! (`load_and_preprocess_from_bytes` + `extract_patches`, timed) vs VCN path
//! (va-bridge decode → dma-buf import → `vl_yuv_preprocess` kernels → D2H,
//! timed), plus a `libjpeg-turbo-rs`-vs-`image` decode diagnostic. Prints
//! rel-L1 of `pixel_values` and a timing table.
//!
//! Acceptance (per experiment contract): per-fixture rel-L1 within ~1e-2
//! (the zune-equivalence argument: zune floor ≈ 4.6e-3 on doge,
//! `docs/VALIDATION.md` VL parity row expects ≈ 5e-3 on 4:2:0).

use hipfire_arch_qwen35_vl::image::{
    extract_patches, load_and_preprocess_from_bytes, smart_resize,
};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const PATCH: usize = 16;
const TEMPORAL: usize = 2; // vision_config_from_hfq default (qwen35_vl.rs:52-55)
const SMS: usize = 2;
const FACTOR: usize = PATCH * SMS;
const MIN_PX: usize = 65_536;
const MAX_PX: usize = 2_000_000; // VISION_MAX_PIXELS default (image.rs:67)

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).abs();
        den += (*x as f64).abs();
    }
    num / den.max(1e-30)
}

fn md5hex(data: &[u8]) -> String {
    // Minimal MD5 (RFC 1321) so prompts stay byte-pinned without a new dep.
    let mut s = Md5::new();
    s.update(data);
    hex(&s.finish())
}

// ——— tiny MD5 + hex (no new deps) ———
struct Md5 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    buf: Vec<u8>,
}
impl Md5 {
    fn new() -> Self {
        Self {
            a: 0x67452301,
            b: 0xefcdab89,
            c: 0x98badcfe,
            d: 0x10325476,
            buf: Vec::new(),
        }
    }
    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
    fn finish(mut self) -> [u8; 16] {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
            0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
            0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
            0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
            0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
            0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
            0xeb86d391,
        ];
        let bit_len = (self.buf.len() as u64).wrapping_mul(8);
        self.buf.push(0x80);
        while self.buf.len() % 64 != 56 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(&bit_len.to_le_bytes());
        for chunk in self.buf.chunks(64) {
            let mut m = [0u32; 16];
            for (i, w) in m.iter_mut().enumerate() {
                *w = u32::from_le_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            let (mut aa, mut bb, mut cc, mut dd) = (self.a, self.b, self.c, self.d);
            for i in 0..64 {
                let (f, g) = match i {
                    0..=15 => ((bb & cc) | (!bb & dd), i),
                    16..=31 => ((dd & bb) | (!dd & cc), (5 * i + 1) % 16),
                    32..=47 => (bb ^ cc ^ dd, (3 * i + 5) % 16),
                    _ => (cc ^ (bb | !dd), (7 * i) % 16),
                };
                let t = dd;
                dd = cc;
                cc = bb;
                bb = bb.wrapping_add(
                    aa.wrapping_add(f)
                        .wrapping_add(K[i])
                        .wrapping_add(m[g])
                        .rotate_left(S[i]),
                );
                aa = t;
            }
            self.a = self.a.wrapping_add(aa);
            self.b = self.b.wrapping_add(bb);
            self.c = self.c.wrapping_add(cc);
            self.d = self.d.wrapping_add(dd);
        }
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.a.to_le_bytes());
        out[4..8].copy_from_slice(&self.b.to_le_bytes());
        out[8..12].copy_from_slice(&self.c.to_le_bytes());
        out[12..16].copy_from_slice(&self.d.to_le_bytes());
        out
    }
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu_only = args.iter().any(|a| a == "--cpu-only");
    let only: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let fixtures = [
        "general_qa.jpg",
        "barney_cigar.jpg",
        "scene_1.jpg",
        "scene_2.jpg",
        "doge.jpeg",
    ];
    let img_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/vision/images");

    // ——— HIP + kernels ———
    let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!("[parity] hip arch={arch}");
    let stream = hip.stream_create().expect("stream_create");

    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/vl_yuv_preprocess.hip");
    let tmp = std::env::temp_dir().join("vcn_parity");
    std::fs::create_dir_all(&tmp).unwrap();
    let hsaco = tmp.join("vl_yuv_preprocess.hsaco");
    println!("[parity] hipcc {} ...", ksrc.display());
    let t_cc = Instant::now();
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
    println!("[parity] hipcc {:.2}s", t_cc.elapsed().as_secs_f32());
    let image = std::fs::read(&hsaco).unwrap();
    let module = hip.module_load_data(&image).expect("module_load_data");
    let k_rgb = hip
        .module_get_function(&module, "vl_nv12_to_rgb_norm")
        .expect("vl_nv12_to_rgb_norm");
    let k_patch = hip
        .module_get_function(&module, "vl_extract_patches")
        .expect("vl_extract_patches");

    // ——— VA session (None ⇒ CPU-fallback mode: oracles only) ———
    let session = match va_bridge::VaSession::open() {
        Ok(s) => {
            println!("[parity] VA vendor: {}", s.vendor());
            Some(s)
        }
        Err(e) => {
            println!("[parity] VA unavailable ({e}) — oracle-only mode");
            None
        }
    };
    println!(
        "{:>16} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10} {:>12}",
        "fixture", "dims", "cpu_ms", "vcn_dec", "vcn_kern", "vcn_tot", "speedup", "rel-L1", "path"
    );
    let mut worst = 0.0f64;
    for name in fixtures {
        if !only.is_empty() && !only.iter().any(|o| o.as_str() == name) {
            continue;
        }
        let bytes = std::fs::read(img_dir.join(name)).expect("fixture read");
        println!(
            "[parity] --- {name} md5={} ({}B)",
            md5hex(&bytes),
            bytes.len()
        );

        // Oracle A (acceptance): the repo CPU path.
        let t0 = Instant::now();
        let (pixels, img_h, img_w) =
            load_and_preprocess_from_bytes(&bytes, PATCH, SMS).expect("cpu preprocess");
        let cpu_pre_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let cpu_patches = extract_patches(&pixels, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
        let cpu_patch_ms = t0.elapsed().as_secs_f64() * 1e3;
        let cpu_ms = cpu_pre_ms + cpu_patch_ms;

        // Oracle B (diagnostic): turbo vs image-crate decode, native resolution.
        let turbo = libjpeg_turbo_rs::decompress_to(&bytes, libjpeg_turbo_rs::PixelFormat::Rgb)
            .expect("turbo decode");
        let via_image = image::load_from_memory(&bytes)
            .expect("image decode")
            .to_rgb8();
        let (tw, th) = (turbo.width, turbo.height);
        assert_eq!(
            (tw, th),
            (via_image.width() as usize, via_image.height() as usize)
        );
        let raw = via_image.as_raw();
        let mut diff_n = 0usize;
        let mut diff_max = 0u8;
        for (a, b) in turbo.data.iter().zip(raw.iter()) {
            let d = a.abs_diff(*b);
            if d > 0 {
                diff_n += 1;
                diff_max = diff_max.max(d);
            }
        }
        println!(
            "[parity] decode-xcheck turbo({tw}x{th}) vs image-crate: differ={diff_n}/{} max_abs={diff_max}",
            raw.len()
        );
        // VCN path (driver-resolved pixels) with turbo CPU fallback, or a
        // --cpu-only timing row for the kill-gate A/B.
        if cpu_only {
            println!(
                "{:>16} {:>9} {:>10.2} {:>10} {:>10} {:>10} {:>8} {:>10} {:>12}",
                name,
                format!("{img_w}x{img_h}"),
                cpu_ms,
                "-",
                "-",
                "-",
                "-",
                "-",
                "cpu-only"
            );
            println!(
                "[parity] {name}: cpu-only cpu_pre={cpu_pre_ms:.2}ms cpu_patch={cpu_patch_ms:.2}ms"
            );
            continue;
        }
        let Some(sess) = session.as_ref() else {
            println!("[parity] {name}: VA unavailable, skipping VCN path");
            continue;
        };
        let t0 = Instant::now();
        let derived = sess.decode_jpeg_derived(&bytes);
        let dec_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (path, got, kern_ms): (&str, Vec<f32>, f64) = match derived {
            Err(va_bridge::VaError::Unsupported(reason)) => {
                // Blocker-2 product behavior: non-4:2:0 falls back to turbo.
                println!("[parity] {name}: VCN unsupported ({reason}) — turbo CPU fallback");
                let t1 = Instant::now();
                let rgb = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(tw as u32, th as u32, turbo.data.clone())
                        .expect("turbo rgb"),
                );
                let rgb = rgb
                    .resize_exact(
                        img_w as u32,
                        img_h as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                    .to_rgb8();
                let plane = img_h * img_w;
                let mut chw = vec![0f32; 3 * plane];
                for (i, px) in rgb.pixels().enumerate() {
                    chw[i] = px[0] as f32 / 127.5 - 1.0;
                    chw[plane + i] = px[1] as f32 / 127.5 - 1.0;
                    chw[2 * plane + i] = px[2] as f32 / 127.5 - 1.0;
                }
                let fb = extract_patches(&chw, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
                ("cpu-fallback", fb, t1.elapsed().as_secs_f64() * 1e3)
            }
            Err(e) => panic!("vcn decode of {name}: {e}"),
            Ok(d) => {
                println!(
                    "[parity] derived: {}x{} fourcc=0x{:08x} y_pitch={} uv_pitch={}",
                    d.width, d.height, d.fourcc, d.y_pitch, d.uv_pitch
                );
                // Target dims: same smart_resize the repo path used.
                let (t_h, t_w) =
                    smart_resize(d.height as usize, d.width as usize, FACTOR, MIN_PX, MAX_PX);
                assert_eq!(
                    (t_h, t_w),
                    (img_h, img_w),
                    "smart_resize mismatch vs repo path (HIPFIRE_VL_MAX_PIXELS?)"
                );
                let (sw, sh) = (d.width as usize, d.height as usize);
                assert_eq!(d.y.len(), sw * sh);
                let cw = (sw + 1) / 2;
                assert_eq!(d.uv.len(), cw * ((sh + 1) / 2) * 2);
                // Packed NV12 upload (packed pitches: y=w, uv=2*cw).
                let mut nv12 = Vec::with_capacity(d.y.len() + d.uv.len());
                nv12.extend_from_slice(&d.y);
                nv12.extend_from_slice(&d.uv);
                let n_chw = 3 * t_h * t_w;
                let n_elem = (t_h / PATCH) * (t_w / PATCH) * TEMPORAL * 3 * PATCH * PATCH;
                assert_eq!(n_elem, cpu_patches.len());
                // Allocations outside the timed window (a product path pools them).
                let d_surf = hip.malloc(nv12.len()).expect("malloc surf");
                let d_chw = hip.malloc(n_chw * 4).expect("malloc chw");
                let d_out = hip.malloc(n_elem * 4).expect("malloc patches");
                let mut surf_a = d_surf.as_ptr() as u64;
                let mut chw_a = d_chw.as_ptr() as u64;
                let mut y_pitch = sw as u32;
                let mut uv_pitch = (2 * cw) as u32;
                let mut uv_offset = (sw * sh) as u32;
                let mut src_w = sw as u32;
                let mut src_h = sh as u32;
                // Derived NV12 is always 4:2:0-layout chroma.
                let mut sh_v = 1u32;
                let mut sv_v = 1u32;
                let t1 = Instant::now();
                hip.memcpy_htod(&d_surf, &nv12).expect("htod surf");
                let mut dst_w = t_w as u32;
                let mut dst_h = t_h as u32;
                let mut p1: Vec<*mut c_void> = vec![
                    (&mut surf_a as *mut u64).cast(),
                    (&mut y_pitch as *mut u32).cast(),
                    (&mut uv_pitch as *mut u32).cast(),
                    (&mut uv_offset as *mut u32).cast(),
                    (&mut src_w as *mut u32).cast(),
                    (&mut src_h as *mut u32).cast(),
                    (&mut sh_v as *mut u32).cast(),
                    (&mut sv_v as *mut u32).cast(),
                    (&mut dst_w as *mut u32).cast(),
                    (&mut dst_h as *mut u32).cast(),
                    (&mut chw_a as *mut u64).cast(),
                ];
                // SAFETY: module loaded, args are valid device pointers / values.
                unsafe {
                    hip.launch_kernel(
                        &k_rgb,
                        [(t_w as u32 + 15) / 16, (t_h as u32 + 15) / 16, 1],
                        [16, 16, 1],
                        0,
                        Some(&stream),
                        &mut p1,
                    )
                    .expect("launch rgb");
                }
                let mut chw_b = chw_a;
                let mut h_v = t_h as u32;
                let mut w_v = t_w as u32;
                let mut p_v = PATCH as u32;
                let mut t_v = TEMPORAL as u32;
                let mut s_v = SMS as u32;
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
                // SAFETY: same.
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
                let kern_ms = t1.elapsed().as_secs_f64() * 1e3;
                let mut raw = vec![0u8; n_elem * 4];
                hip.memcpy_dtoh(&mut raw, &d_out).expect("dtoh");
                // SAFETY: kernel wrote f32 elements; length checked above.
                let v: Vec<f32> =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_elem) }
                        .to_vec();
                hip.free(d_surf).unwrap();
                hip.free(d_chw).unwrap();
                hip.free(d_out).unwrap();
                ("vcn", v, kern_ms)
            }
        };
        let r = rel_l1(&cpu_patches, &got);
        worst = worst.max(r);
        let tot = dec_ms + kern_ms;
        println!(
            "{:>16} {:>9} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>8.2}x {:>10.3e} {:>12}",
            name,
            format!("{img_w}x{img_h}"),
            cpu_ms,
            dec_ms,
            kern_ms,
            tot,
            cpu_ms / tot.max(1e-9),
            r,
            path
        );
    }
    println!("[parity] worst rel-L1 = {worst:.3e} (equivalence bound 1e-2)");
}
