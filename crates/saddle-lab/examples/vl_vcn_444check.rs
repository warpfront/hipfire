//! Investigation: does VCN decode 4:4:4 / 4:2:0 correctly when the VA surface
//! carries the stream's chroma format? Reads planes back via vaDeriveImage,
//! converts to RGB on the host (JFIF full-range BT.601), and compares to
//! libjpeg-turbo-rs, which is byte-identical to PIL/libjpeg-turbo.
use std::path::PathBuf;

fn clamp(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/vision/images");
    let mut sess = va_bridge::VaSession::open().expect("va open");
    println!("[444check] {} (node {})", sess.vendor(), sess.node());
    for name in ["barney_cigar.jpg", "scene_1.jpg", "doge.jpeg", "scene_2.jpg", "general_qa.jpg"] {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let turbo = libjpeg_turbo_rs::decompress_to(&bytes, libjpeg_turbo_rs::PixelFormat::Rgb).unwrap();
        let (w, h) = (turbo.width as usize, turbo.height as usize);
        let f = match sess.decode_jpeg_planes(&bytes) {
            Ok(f) => f,
            Err(e) => {
                println!("{name:18} VCN: {e}");
                continue;
            }
        };
        let fourcc = String::from_utf8_lossy(&f.fourcc.to_le_bytes()).to_string();
        // Host YCbCr->RGB per plane layout.
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let yy = f.planes[0][y * w + x] as f32;
                let (cb, cr) = match fourcc.as_str() {
                    "NV12" => {
                        let cw = (w + 1) / 2;
                        let i = (y / 2) * cw * 2 + (x / 2) * 2;
                        (f.planes[1][i] as f32, f.planes[1][i + 1] as f32)
                    }
                    "444P" => (f.planes[1][y * w + x] as f32, f.planes[2][y * w + x] as f32),
                    _ => (128.0, 128.0),
                };
                let (cb, cr) = (cb - 128.0, cr - 128.0);
                let o = (y * w + x) * 3;
                rgb[o] = clamp(yy + 1.402 * cr);
                rgb[o + 1] = clamp(yy - 0.344136 * cb - 0.714136 * cr);
                rgb[o + 2] = clamp(yy + 1.772 * cb);
            }
        }
        // Luma check (decode correctness independent of chroma reconstruction).
        let mut dy = 0f64;
        let mut drgb = 0f64;
        let mut maxrgb = 0u8;
        for i in 0..w * h {
            let (r, g, b) = (turbo.data[i * 3] as f32, turbo.data[i * 3 + 1] as f32, turbo.data[i * 3 + 2] as f32);
            let yt = 0.299 * r + 0.587 * g + 0.114 * b;
            dy += (yt - f.planes[0][i] as f32).abs() as f64;
            for c in 0..3 {
                let d = (rgb[i * 3 + c] as i16 - turbo.data[i * 3 + c] as i16).unsigned_abs() as u8;
                drgb += d as f64;
                maxrgb = maxrgb.max(d);
            }
        }
        // Zero-copy check: import the exported dma-buf into HIP and read the
        // Y plane back linearly (pitch-aware). Surfaces are linear-only, so
        // this matches derived Y.
        match sess.decode_jpeg(&bytes) {
            Ok(va_bridge::DecodeOutcome::Decoded(vf)) => {
                let pitch = vf.y_pitch() as usize;
                match vf.copy_to_host(0, pitch * h) {
                    Ok(buf) => {
                        let mut d = 0f64;
                        for y in 0..h {
                            for x in 0..w {
                                d += (buf[y * pitch + x] as i32 - f.planes[0][y * w + x] as i32).abs() as f64;
                            }
                        }
                        println!(
                            "{name:18}   dma-buf import (pitch {pitch}, {} layers): Y vs derived-Y mean|d|={:.3} LSB",
                            vf.num_layers,
                            d / (w * h) as f64
                        );
                    }
                    Err(e) => println!("{name:18}   dma-buf copy_to_host: {e}"),
                }
            }
            Ok(va_bridge::DecodeOutcome::Unsupported(reason)) => {
                println!("{name:18}   dma-buf path: unsupported ({reason})")
            }
            Err(e) => println!("{name:18}   dma-buf path: {e}"),
        }
        println!(
            "{name:18} {w}x{h} VCN fourcc={fourcc} planes={} | Y vs turbo-luma mean|d|={:.3} LSB | host-RGB vs turbo mean|d|={:.3} max={}",
            f.planes.len(),
            dy / (w * h) as f64,
            drgb / (w * h * 3) as f64,
            maxrgb
        );
    }
}
