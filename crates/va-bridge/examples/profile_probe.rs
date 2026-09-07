// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T0 profile probe (`experiment/vcn-video`): which decode profiles does the
//! driver advertise, and does `vaCreateConfig` succeed for H.264 High (7),
//! HEVC Main (17), AV1 Profile 0 (32)?
//!
//! Run: `flock -w 60 /tmp/hipfire-gpu.lock cargo run -p va-bridge --example profile_probe`
//! (`HIPFIRE_VCN_DRM_NODE` overrides the render node, same as `VaSession`.)

use libloading::Library;
use std::ffi::{c_void, CStr};

type VaDisplay = *mut c_void;

fn profile_name(p: i32) -> &'static str {
    match p {
        -1 => "None",
        0 => "MPEG2Simple",
        1 => "MPEG2Main",
        2 => "MPEG4Simple",
        3 => "MPEG4AdvancedSimple",
        4 => "MPEG4Main",
        5 => "H264Baseline",
        6 => "H264Main",
        7 => "H264High",
        8 => "VC1Simple",
        9 => "VC1Main",
        10 => "VC1Advanced",
        11 => "H263Baseline",
        12 => "JPEGBaseline",
        13 => "H264ConstrainedBaseline",
        14 => "VP8Version0_3",
        15 => "H264MultiviewHigh",
        16 => "H264StereoHigh",
        17 => "HEVCMain",
        18 => "HEVCMain10",
        19 => "VP9Profile0",
        20 => "VP9Profile1",
        21 => "VP9Profile2",
        22 => "VP9Profile3",
        23 => "HEVCMain12",
        24 => "HEVCMain422_10",
        25 => "HEVCMain422_12",
        26 => "HEVCMain444",
        27 => "HEVCMain444_10",
        28 => "HEVCMain444_12",
        29 => "HEVCSccMain",
        30 => "HEVCSccMain10",
        31 => "HEVCSccMain444",
        32 => "AV1Profile0",
        33 => "AV1Profile1",
        _ => "?",
    }
}

fn entrypoint_name(e: i32) -> &'static str {
    match e {
        1 => "VLD",
        2 => "IZZ",
        3 => "IDCT",
        4 => "MoComp",
        5 => "EncSlice",
        6 => "EncPicture",
        7 => "EncSliceLP",
        8 => "VideoProc",
        9 => "FEI",
        10 => "Stats",
        _ => "?",
    }
}
macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let s: libloading::Symbol<$ty> = unsafe { $lib.get($name) }
            .unwrap_or_else(|_| panic!("missing symbol {}", String::from_utf8_lossy($name)));
        *s
    }};
}

fn main() {
    let node =
        std::env::var("HIPFIRE_VCN_DRM_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let va: Library = unsafe { Library::new("libva.so.2").expect("libva.so.2") };
    let va_drm: Library = unsafe { Library::new("libva-drm.so.2").expect("libva-drm.so.2") };
    unsafe {
        let get_dpy: extern "C" fn(i32) -> VaDisplay =
            sym!(va_drm, b"vaGetDisplayDRM", extern "C" fn(i32) -> VaDisplay);
        let init: extern "C" fn(VaDisplay, *mut i32, *mut i32) -> i32 = sym!(
            va,
            b"vaInitialize",
            extern "C" fn(VaDisplay, *mut i32, *mut i32) -> i32
        );
        let vendor: extern "C" fn(VaDisplay) -> *const i8 = sym!(
            va,
            b"vaQueryVendorString",
            extern "C" fn(VaDisplay) -> *const i8
        );
        let maxp: extern "C" fn(VaDisplay) -> i32 =
            sym!(va, b"vaMaxNumProfiles", extern "C" fn(VaDisplay) -> i32);
        let qprof: extern "C" fn(VaDisplay, *mut i32, *mut i32) -> i32 = sym!(
            va,
            b"vaQueryConfigProfiles",
            extern "C" fn(VaDisplay, *mut i32, *mut i32) -> i32
        );
        let qentry: extern "C" fn(VaDisplay, i32, *mut i32, *mut i32) -> i32 = sym!(
            va,
            b"vaQueryConfigEntrypoints",
            extern "C" fn(VaDisplay, i32, *mut i32, *mut i32) -> i32
        );
        let mkcfg: extern "C" fn(VaDisplay, i32, i32, *mut c_void, i32, *mut u32) -> i32 = sym!(
            va,
            b"vaCreateConfig",
            extern "C" fn(VaDisplay, i32, i32, *mut c_void, i32, *mut u32) -> i32
        );
        let errstr: extern "C" fn(i32) -> *const i8 =
            sym!(va, b"vaErrorStr", extern "C" fn(i32) -> *const i8);
        let es = |c: i32| CStr::from_ptr(errstr(c)).to_string_lossy().into_owned();

        let fd: std::os::fd::OwnedFd = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&node)
            .expect("open render node")
            .into();
        let dpy = get_dpy(std::os::fd::AsRawFd::as_raw_fd(&fd));
        assert!(!dpy.is_null());
        let (mut mj, mut mn) = (0, 0);
        assert_eq!(init(dpy, &mut mj, &mut mn), 0);
        println!(
            "node={node} vendor={} va={mj}.{mn}",
            CStr::from_ptr(vendor(dpy)).to_string_lossy()
        );
        let n = maxp(dpy).max(1).min(64) as usize;
        let mut profs = vec![0i32; n];
        let mut m = n as i32;
        assert_eq!(qprof(dpy, profs.as_mut_ptr(), &mut m), 0);
        println!("advertised profiles ({m}):");
        for p in &profs[..m.max(0) as usize] {
            let mut es_ = vec![0i32; 16];
            let mut k = 16i32;
            let st = qentry(dpy, *p, es_.as_mut_ptr(), &mut k);
            let eps = if st == 0 {
                es_[..k.max(0) as usize]
                    .iter()
                    .map(|e| format!("{e}={}", entrypoint_name(*e)))
                    .collect::<Vec<_>>()
                    .join(",")
            } else {
                format!("query-failed: {}", es(st))
            };
            println!("  profile {p}={} entrypoints: {eps}", profile_name(*p));
        }
        for target in [7i32, 17, 32] {
            let mut cfg = 0u32;
            let st = mkcfg(
                dpy,
                target,
                1, /* VLD */
                std::ptr::null_mut(),
                0,
                &mut cfg,
            );
            println!(
                "vaCreateConfig({target}={}/VLD): {}",
                profile_name(target),
                if st == 0 {
                    format!("OK config={cfg}")
                } else {
                    format!("FAIL {}", es(st))
                }
            );
        }
    }
}
