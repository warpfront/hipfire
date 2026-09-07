// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Smoke: decode the bf0 Annex-B fixture through VCN, print per-frame
//! geometry + Y-plane checksum. Not an acceptance gate (see the
//! `vl_vcn_video_parity` saddle-lab example); just a fast bring-up probe.
//!
//! Run: `flock -w 120 /tmp/hipfire-gpu.lock cargo run -q -p va-bridge
//! --example h264_smoke -- <file.h264> [--dump <prefix>]`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .expect("usage: h264_smoke <file.h264> [--dump p]");
    let data = std::fs::read(path).expect("fixture read");
    println!("[smoke] {} ({} bytes)", path, data.len());
    let mut sess = va_bridge::VaSession::open_video(va_bridge::VA_PROFILE_H264_HIGH)
        .expect("open_video(H264High)");
    println!("[smoke] VA vendor: {}", sess.vendor());
    let t0 = std::time::Instant::now();
    let frames = sess.decode_h264_annexb(&data).expect("decode");
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("[smoke] decoded {} frames in {ms:.1}ms", frames.len());
    let dump = args
        .windows(2)
        .find(|w| w[0] == "--dump")
        .map(|w| w[1].clone());
    for (i, f) in frames.iter().enumerate() {
        let sum: u64 = f.y.iter().map(|&b| b as u64).sum();
        let uvsum: u64 = f.uv.iter().map(|&b| b as u64).sum();
        println!(
            "[smoke] f{i:02} {}x{} y_sum={sum} uv_sum={uvsum}",
            f.width, f.height
        );
        if let Some(p) = &dump {
            std::fs::write(
                format!("{p}_f{i:02}.nv12"),
                [f.y.clone(), f.uv.clone()].concat(),
            )
            .expect("dump write");
        }
    }
    if args.iter().any(|a| a == "--zc") {
        let t0 = std::time::Instant::now();
        let zc = sess.decode_h264_annexb_zc(&data).expect("zc decode");
        println!(
            "[smoke] zc decoded {} frames in {:.1}ms; f00 fourcc=0x{:08x} ptr={:?}",
            zc.len(),
            t0.elapsed().as_secs_f64() * 1e3,
            zc[0].fourcc,
            zc[0].device_ptr()
        );
    }
}
