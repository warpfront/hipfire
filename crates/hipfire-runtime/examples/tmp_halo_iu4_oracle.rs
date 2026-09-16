//! W5 — IU4 in-register prefetch oracle (throwaway).
//!
//! Bitwise-compares `_occ3_col_gfx1151` vs `_pf_gfx11` on gate_proj / down_proj
//! (qt=44) at N=512, col grid [N/128, M/128, 1].
//!
//! Halo:
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-w5 HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_halo_iu4_oracle -- <model.hfq>
//!   TIME=1 …   # interleaved medians, ≥100 samples
//!   --compile-only   # JIT the four symbols, then return

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor, MQ4V2_GROUP_BYTES};
use std::path::Path;
use std::time::Instant;

const BLOCK: [u32; 3] = [32, 8, 1];
const IU4_LDS: u32 = 30720;
const N: usize = 512;
const QT44: u8 = 44;
const MOD: &str = "halo_w5_iu4_pf";
const SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);

const SHIP_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151";
const SHIP_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151";
const PF_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_pf_gfx11";
const PF_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_pf_gfx11";

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn ensure_entries(gpu: &mut Gpu) {
    for sym in [SHIP_SET, SHIP_ADD, PF_SET, PF_ADD] {
        gpu.ensure_kernel_public(MOD, SRC, sym)
            .unwrap_or_else(|e| panic!("ensure_kernel {sym}: {e:?}"));
        eprintln!("W5 meta: compiled {sym}");
    }
}

fn launch(
    gpu: &mut Gpu,
    name: &str,
    d_a: &GpuTensor,
    xq: *mut std::ffi::c_void,
    d_y: &GpuTensor,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut args = KernargBlob::new();
    args.push_ptr(d_a.buf.as_ptr());
    args.push_ptr(xq);
    args.push_ptr(d_y.buf.as_ptr());
    args.push_i32(m);
    args.push_i32(k);
    args.push_i32(n);
    args.push_i32(add);
    let row_tiles = (m as u32) / 128;
    let col_tiles = (n as u32) / 128;
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.launch_kernel_blob(
        name,
        [col_tiles, row_tiles, 1],
        BLOCK,
        IU4_LDS,
        args.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch {name}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

fn snapshot(gpu: &Gpu, buf: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.size()];
    gpu.hip.memcpy_dtoh(&mut out, buf).unwrap();
    out
}

fn find_qt44<'a>(hfq: &'a HfqFile, needle: &str) -> (&'a str, usize, usize, &'a [u8]) {
    let ti = hfq
        .tensors()
        .iter()
        .find(|t| t.quant_type == QT44 && t.name.contains(needle))
        .unwrap_or_else(|| panic!("no qt=44 tensor matching '{needle}'"));
    assert!(
        ti.shape.len() >= 2,
        "{} shape {:?} (need ≥2)",
        ti.name,
        ti.shape
    );
    let m = ti.shape[0] as usize;
    let k = ti.shape[1] as usize;
    assert!(m % 128 == 0 && k % 256 == 0, "{} m={m} k={k} not full-tile", ti.name);
    let expect = m * (k / 256) * MQ4V2_GROUP_BYTES;
    assert_eq!(ti.data_size, expect, "{} data_size {} != {expect}", ti.name, ti.data_size);
    let (_info, bytes) = hfq
        .tensor_data(&ti.name)
        .unwrap_or_else(|| panic!("tensor_data {}", ti.name));
    assert_eq!(bytes.len(), expect);
    (ti.name.as_str(), m, k, bytes)
}

fn make_x(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut x = vec![0.0f32; n * k];
    for v in &mut x {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (s as f32) * (1.0 / 4294967296.0) - 0.5;
    }
    x
}

fn run_weight(
    gpu: &mut Gpu,
    tag: &str,
    m: usize,
    k: usize,
    a_bytes: &[u8],
    time: bool,
) {
    eprintln!("W5 {tag}: M={m} K={k} N={N}");
    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).unwrap();
    let x_h = make_x(N, k, 0xC0FFEEu32.wrapping_add(m as u32));
    let x_bytes: Vec<u8> = x_h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_x = gpu.upload_raw(&x_bytes, &[N, k]).unwrap();
    let xq = gpu.ensure_int4_mmq_x(&d_x, N, k).unwrap();
    let xq_bytes = (k / 128) * N * 72;
    let xq_buf = unsafe { DeviceBuffer::from_raw(xq, xq_bytes) };

    let a0 = snapshot(gpu, &d_a.buf);
    let xq0 = snapshot(gpu, &xq_buf);

    // --- set ---
    let y_ship = gpu.zeros(&[N * m], DType::F32).unwrap();
    let y_pf = gpu.zeros(&[N * m], DType::F32).unwrap();
    let _ = launch(gpu, SHIP_SET, &d_a, xq, &y_ship, m as i32, k as i32, N as i32, 0);
    let _ = launch(gpu, PF_SET, &d_a, xq, &y_pf, m as i32, k as i32, N as i32, 0);
    let ys = gpu.download_f32(&y_ship).unwrap();
    let yp = gpu.download_f32(&y_pf).unwrap();
    assert_eq!(ys.len(), yp.len());
    let mism = ys
        .iter()
        .zip(yp.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mism, 0, "{tag} SET bitwise mismatch: {mism}/{}", ys.len());
    eprintln!("W5 {tag} SET: bitwise identical ({} f32)", ys.len());

    // --- add, nonzero Y0 ---
    let mut y0 = make_x(N, m, 0xA5A5_0001);
    y0.iter_mut().for_each(|v| *v *= 0.25);
    let y0b: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_ship_a = gpu.zeros(&[N * m], DType::F32).unwrap();
    let y_pf_a = gpu.zeros(&[N * m], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&y_ship_a.buf, &y0b).unwrap();
    gpu.hip.memcpy_htod(&y_pf_a.buf, &y0b).unwrap();
    let _ = launch(gpu, SHIP_ADD, &d_a, xq, &y_ship_a, m as i32, k as i32, N as i32, 1);
    let _ = launch(gpu, PF_ADD, &d_a, xq, &y_pf_a, m as i32, k as i32, N as i32, 1);
    let yas = gpu.download_f32(&y_ship_a).unwrap();
    let yap = gpu.download_f32(&y_pf_a).unwrap();
    let mism_a = yas
        .iter()
        .zip(yap.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mism_a, 0, "{tag} ADD bitwise mismatch: {mism_a}/{}", yas.len());
    assert!(
        yas.iter().any(|&v| v.to_bits() != 0),
        "{tag} ADD produced all-zero Y"
    );
    eprintln!("W5 {tag} ADD: bitwise identical ({} f32)", yas.len());

    let a1 = snapshot(gpu, &d_a.buf);
    let xq1 = snapshot(gpu, &xq_buf);
    assert_eq!(a0, a1, "{tag} A mutated");
    assert_eq!(xq0, xq1, "{tag} Xq mutated");
    eprintln!("W5 {tag}: A/Xq immutable");

    if time {
        let samples = 100usize;
        let mut ship_us = Vec::with_capacity(samples);
        let mut pf_us = Vec::with_capacity(samples);
        // one warmup pair
        let _ = launch(gpu, SHIP_SET, &d_a, xq, &y_ship, m as i32, k as i32, N as i32, 0);
        let _ = launch(gpu, PF_SET, &d_a, xq, &y_pf, m as i32, k as i32, N as i32, 0);
        for _ in 0..samples {
            ship_us.push(launch(
                gpu, SHIP_SET, &d_a, xq, &y_ship, m as i32, k as i32, N as i32, 0,
            ));
            pf_us.push(launch(
                gpu, PF_SET, &d_a, xq, &y_pf, m as i32, k as i32, N as i32, 0,
            ));
        }
        let ms = median_f64(&mut ship_us);
        let mp = median_f64(&mut pf_us);
        eprintln!(
            "TIME SET {tag:>24}  ship_med={ms:.1}us  pf_med={mp:.1}us  n={samples}"
        );

        let mut ship_a = Vec::with_capacity(samples);
        let mut pf_a = Vec::with_capacity(samples);
        let _ = launch(gpu, SHIP_ADD, &d_a, xq, &y_ship_a, m as i32, k as i32, N as i32, 1);
        let _ = launch(gpu, PF_ADD, &d_a, xq, &y_pf_a, m as i32, k as i32, N as i32, 1);
        for _ in 0..samples {
            gpu.hip.memcpy_htod(&y_ship_a.buf, &y0b).unwrap();
            gpu.hip.memcpy_htod(&y_pf_a.buf, &y0b).unwrap();
            ship_a.push(launch(
                gpu, SHIP_ADD, &d_a, xq, &y_ship_a, m as i32, k as i32, N as i32, 1,
            ));
            pf_a.push(launch(
                gpu, PF_ADD, &d_a, xq, &y_pf_a, m as i32, k as i32, N as i32, 1,
            ));
        }
        let msa = median_f64(&mut ship_a);
        let mpa = median_f64(&mut pf_a);
        eprintln!(
            "TIME ADD {tag:>24}  ship_med={msa:.1}us  pf_med={mpa:.1}us  n={samples}"
        );
    }

    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(y_ship);
    let _ = gpu.free_tensor(y_pf);
    let _ = gpu.free_tensor(y_ship_a);
    let _ = gpu.free_tensor(y_pf_a);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let time = std::env::var("TIME").ok().as_deref() == Some("1");
    let hfq_path = args.iter().skip(1).find(|a| a.as_str() != "--compile-only");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("tmp_halo_iu4_oracle arch={}", gpu.arch);
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    }
    ensure_entries(&mut gpu);
    if compile_only {
        eprintln!("W5 --compile-only: ensure_kernel done");
        return;
    }
    let path = hfq_path.expect("usage: tmp_halo_iu4_oracle [--compile-only] <model.hfq>");
    let hfq = HfqFile::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let t0 = Instant::now();
    for needle in ["gate_proj", "down_proj"] {
        let (name, m, k, bytes) = find_qt44(&hfq, needle);
        eprintln!("W5 loaded {name} qt=44");
        run_weight(&mut gpu, name, m, k, bytes, time);
    }
    eprintln!("W5 oracle complete in {:.1}s", t0.elapsed().as_secs_f64());
}
