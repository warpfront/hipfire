// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Slice gate for JpegPacketAndState (experiment `experiment/vcn-ring`):
//! doge.jpeg on gfx1201, terminal JPEG fence wait, valid-plane readback.
//! Requires no timeout/reset, unchanged canaries, and ZERO differing bytes
//! versus `VaSession::decode_jpeg_planes`, plus a packet dump (dword list)
//! of the submitted IB.
//!
//! Run under the GPU lock (NOT gpu_acquire):
//!   exec 9>/tmp/hipfire-gpu.lock; flock -w 300 9 \
//!     cargo test -p redline --features vcn-jpeg --test vcn_jpeg_gate -- --nocapture
//! Skips (does not fail) when no GPU render node opens.

#![cfg(feature = "vcn-jpeg")]

use redline::device::Device;
use redline::queue::ComputeQueue;
use redline::vcn_jpeg::{JpegNativeFormat, VcnJpegDecoder};

const TIMEOUT_NS: u64 = 10_000_000_000;

fn load_doge() -> Vec<u8> {
    let path = "../../benchmarks/vision/images/doge.jpeg";
    std::fs::read(path).unwrap_or_else(|e| panic!("gate fixture missing ({path}): {e}"))
}

#[test]
fn doge_decode_parity() {
    let jpeg = load_doge();
    eprintln!("[gate] doge.jpeg: {} bytes", jpeg.len());

    let dev = match Device::open(None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[gate] SKIP: no GPU device: {e}");
            return;
        }
    };
    eprintln!("[gate] device: {}", dev.info.gfx_arch);
    let queue = ComputeQueue::new(&dev).expect("queue");
    let mut dec = VcnJpegDecoder::new(&dev).expect("decoder new");

    let pend = dec.submit(&dev, &queue, &jpeg).expect("submit");
    let layout = pend.layout();
    eprintln!(
        "[gate] layout: {:?} {}x{} pitch={} offsets={:?} rows={:?} planes={}",
        layout.format,
        layout.width,
        layout.height,
        layout.pitch,
        layout.plane_offsets,
        layout.plane_rows,
        layout.plane_count,
    );
    // Packet dump (dword list) of the submitted IB.
    let words = pend.ib_dwords();
    eprintln!(
        "[gate] IB: {} dwords ({} bytes)",
        words.len(),
        words.len() * 4
    );
    for (i, chunk) in words.chunks(8).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|w| format!("{w:#010x}")).collect();
        eprintln!("[gate] IB[{i:02}]: {}", hex.join(" "));
    }

    let ready = pend
        .wait_decode(&dev, &queue, TIMEOUT_NS)
        .expect("wait_decode");
    assert_eq!(ready.layout(), layout);
    // Surface readback while the flight is still borrowed.
    let surf = ready.surface();
    let mut host = vec![0u8; surf.size as usize];
    dev.download(surf, &mut host).expect("download");
    // Drop to Idle first: the canary bands persist in the BO, so checking
    // after the borrow releases is still exact evidence.
    drop(ready);
    dec_verify_canaries(&dev, &dec);

    // Oracle: native planes via the VA path (not the NV12-only derived path).
    let va = va_bridge::VaSession::open().expect("va session for oracle");
    let oracle = va.decode_jpeg_planes(&jpeg).expect("oracle decode");
    assert_eq!((oracle.width, oracle.height), (layout.width, layout.height));
    eprintln!(
        "[gate] oracle: fourcc={:#010x} planes={}",
        oracle.fourcc,
        oracle.planes.len()
    );

    let (diff_count, max_abs) = diff_valid_rows(&layout, &host, &oracle.planes);
    eprintln!("[gate] diff_count={diff_count} max_abs={max_abs}");
    assert_eq!(diff_count, 0, "valid-plane bytes differ from VA oracle");
    assert_eq!(max_abs, 0);

    dec.destroy(&dev);
    queue.destroy(&dev);
    eprintln!("[gate] PASS: no timeout, canaries intact, zero differing bytes");
}

fn dec_verify_canaries(dev: &Device, dec: &VcnJpegDecoder) {
    dec.verify_canaries(dev).expect("canaries unchanged");
}

/// Byte-diff over valid rows only; padding is undefined and never compared.
/// NV12: luma h rows of w bytes, chroma ceil(h/2) rows of ceil(w/2)*2 bytes
/// (matches `DerivedPlanes` geometry). 444P: three planes of h rows of w.
fn diff_valid_rows(
    layout: &redline::vcn_jpeg::JpegSurfaceLayout,
    host: &[u8],
    oracle: &[Vec<u8>],
) -> (u64, u8) {
    let (w, h) = (layout.width as usize, layout.height as usize);
    let pitch = layout.pitch as usize;
    let mut diff = 0u64;
    let mut max_abs = 0u8;
    let mut cmp_row =
        |surf_off: usize, stride: usize, row_bytes: usize, rows: usize, plane: &[u8]| {
            for r in 0..rows {
                let a = &host[surf_off + r * stride..surf_off + r * stride + row_bytes];
                let b = &plane[r * row_bytes..(r + 1) * row_bytes];
                for (&x, &y) in a.iter().zip(b.iter()) {
                    if x != y {
                        diff += 1;
                        max_abs = max_abs.max(x.abs_diff(y));
                    }
                }
            }
        };
    match layout.format {
        JpegNativeFormat::Nv12 => {
            assert_eq!(oracle.len(), 2);
            cmp_row(layout.plane_offsets[0] as usize, pitch, w, h, &oracle[0]);
            let cw = (w + 1) / 2;
            cmp_row(
                layout.plane_offsets[1] as usize,
                pitch,
                cw * 2,
                (h + 1) / 2,
                &oracle[1],
            );
        }
        JpegNativeFormat::Yuv444p => {
            assert_eq!(oracle.len(), 3);
            for i in 0..3 {
                cmp_row(layout.plane_offsets[i] as usize, pitch, w, h, &oracle[i]);
            }
        }
    }
    (diff, max_abs)
}
