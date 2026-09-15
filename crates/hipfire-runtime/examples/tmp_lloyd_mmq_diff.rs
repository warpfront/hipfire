//! Throwaway U2 differential: MMQ-LUT twin vs GEMV Lloyd oracle (qt=52) and
//! uniform MMQ vs GEMV (qt=44) floor. Launchers are U3 — this raw-launches the
//! twin via `launch_kernel_blob`.
//!
//! Usage:
//!   tmp_lloyd_mmq_diff <qt52.hfq> <qt44.hfq>
//!   TIME=1 tmp_lloyd_mmq_diff <qt52.hfq> <qt44.hfq>
//!
//! XTX:
//!   ROCR_VISIBLE_DEVICES=0 cargo run --release -p hipfire-runtime \
//!     --example tmp_lloyd_mmq_diff -- \
//!     /home/kaden/qcal/lloyd/qwen3.8-27b.mq4v2l.base.hfq \
//!     /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::{load_lloyd_lut, HfqFile};
use hipfire_runtime::lloyd_lut::apply_lloyd_centering;
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::time::Instant;

const MMQ_X: usize = 128;
const MMQ_Y: usize = 128;
const MMQ_TILE_Y_K: usize = 36;
const MMQ_TILE_X_K: usize = 76;
const MODULE_LLOYD: &str = "gemm_mq4g256v2_residual_mmq_lloyd";
const SRC_LLOYD: &str = concat!(
    "#define HIPFIRE_MMQ_LUT 1\n",
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq.hip")
);

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as f64) - (y as f64);
        num += d * d;
        den += (y as f64) * (y as f64);
    }
    if den < 1e-30 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

fn load_weight(hfq: &HfqFile, name: &str, expect_qt: u8) -> (usize, usize, Vec<u8>) {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    assert_eq!(
        info.quant_type, expect_qt,
        "{name}: qt={} expected {expect_qt}",
        info.quant_type
    );
    assert_eq!(info.shape.len(), 2, "{name} shape {:?}", info.shape);
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    (m, k, data)
}

fn lloyd_symbol(full: bool, add: bool, x128: bool) -> &'static str {
    match (full, add, x128) {
        (true, true, false) => "gemm_mq4g256v2_residual_mmq_full_add_lloyd",
        (true, false, false) => "gemm_mq4g256v2_residual_mmq_full_set_lloyd",
        (false, _, false) => "gemm_mq4g256v2_residual_mmq_lloyd",
        (true, true, true) => "gemm_mq4g256v2_residual_mmq_full_add_x128_lloyd",
        (true, false, true) => "gemm_mq4g256v2_residual_mmq_full_set_x128_lloyd",
        (false, _, true) => "gemm_mq4g256v2_residual_mmq_x128_lloyd",
    }
}

fn launch_lloyd_mmq(
    gpu: &mut Gpu,
    a: &rdna_compute::GpuTensor,
    xq: *mut std::ffi::c_void,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    c16: [u32; 4],
    force_x128: Option<bool>,
) {
    let full = m % 128 == 0 && n % 128 == 0;
    let x128 = force_x128.unwrap_or_else(|| gpu.flags.gfx11_mmq_x128_enabled());
    let kernel = lloyd_symbol(full, add, x128);
    gpu.ensure_kernel_public(MODULE_LLOYD, SRC_LLOYD, kernel)
        .unwrap_or_else(|e| panic!("ensure {kernel}: {e}"));
    let shared =
        ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
    let grid = [m.div_ceil(MMQ_Y) as u32, n.div_ceil(MMQ_X) as u32, 1];
    let block = [32u32, 8, 1];
    let mut blob = KernargBlob::new();
    blob.push_ptr(a.buf.as_ptr());
    blob.push_ptr(xq);
    blob.push_ptr(y.buf.as_ptr());
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);
    blob.push_i32(n as i32);
    blob.push_i32(i32::from(add));
    blob.push_u32(c16[0]);
    blob.push_u32(c16[1]);
    blob.push_u32(c16[2]);
    blob.push_u32(c16[3]);
    gpu.launch_kernel_blob(kernel, grid, block, shared, blob.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {kernel}: {e}"));
}

fn sample_cols(n: usize) -> Vec<usize> {
    let mut cols = vec![0usize, 1, n / 2, n.saturating_sub(2), n.saturating_sub(1)];
    cols.retain(|&c| c < n);
    cols.sort_unstable();
    cols.dedup();
    cols
}

fn col_slice(y: &[f32], m: usize, col: usize) -> &[f32] {
    &y[col * m..(col + 1) * m]
}

fn gemv_lloyd_cols(
    gpu: &mut Gpu,
    a: &rdna_compute::GpuTensor,
    x_host: &[f32],
    m: usize,
    k: usize,
    _n: usize,
    cols: &[usize],
    lut_f16: [u32; 8],
) -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    for &c in cols {
        let x_col = &x_host[c * k..(c + 1) * k];
        let d_x = gpu.upload_f32(x_col, &[k]).expect("upload x col");
        let d_y = gpu.zeros(&[m], DType::F32).expect("y col");
        gpu.gemv_mq4g256v2_lloyd(a, &d_x, &d_y, m, k, lut_f16)
            .expect("gemv lloyd");
        gpu.hip.device_synchronize().expect("sync gemv");
        out.push((c, gpu.download_f32(&d_y).expect("dl gemv")));
    }
    out
}

fn gemv_uniform_cols(
    gpu: &mut Gpu,
    a: &rdna_compute::GpuTensor,
    x_host: &[f32],
    m: usize,
    k: usize,
    _n: usize,
    cols: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    for &c in cols {
        let x_col = &x_host[c * k..(c + 1) * k];
        let d_x = gpu.upload_f32(x_col, &[k]).expect("upload x col");
        let d_y = gpu.zeros(&[m], DType::F32).expect("y col");
        gpu.gemv_mq4g256v2(a, &d_x, &d_y, m, k).expect("gemv uniform");
        gpu.hip.device_synchronize().expect("sync gemv");
        out.push((c, gpu.download_f32(&d_y).expect("dl gemv")));
    }
    out
}

fn report_vs_gemv(tag: &str, y_mmq: &[f32], m: usize, gemv: &[(usize, Vec<f32>)]) {
    for &(c, ref ref_y) in gemv {
        let got = col_slice(y_mmq, m, c);
        let e = rel_l2(got, ref_y);
        eprintln!("  ADVISORY {tag} col={c}: rel-L2={e:.6e}");
    }
}

fn run_lloyd_shape(
    gpu: &mut Gpu,
    tag: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    c16: [u32; 4],
    lut_f16: [u32; 8],
    time: bool,
) {
    eprintln!("=== Lloyd {tag} M={m} K={k} N={n} add={add} ===");
    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A lloyd");
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| (prng(i, 0xA11CE) * 2.0 - 1.0) * 0.25)
        .collect();
    let d_x = gpu.upload_f32(&x_host, &[n * k]).expect("upload X");

    // NONZERO Y0 for add path so += is exercised.
    let y0_host: Vec<f32> = if add {
        (0..n * m)
            .map(|i| (prng(i, 0xADD0) * 2.0 - 1.0) * 0.05)
            .collect()
    } else {
        vec![0.0; n * m]
    };
    let d_y = if add {
        gpu.upload_f32(&y0_host, &[n * m]).expect("upload Y0")
    } else {
        gpu.zeros(&[n * m], DType::F32).expect("zeros Y")
    };

    let xq = if gpu.flags.gfx11_mmq_x128_enabled() {
        gpu.ensure_q8_1_mmq_x128(&d_x, n, k).expect("q8 x128")
    } else {
        gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("q8 x")
    };

    launch_lloyd_mmq(gpu, &d_a, xq, &d_y, m, k, n, add, c16, None);
    gpu.hip.device_synchronize().expect("sync lloyd mmq");
    let y_mmq = gpu.download_f32(&d_y).expect("dl lloyd y");

    // For add: gemv is set-path; compare (y_mmq - y0) vs gemv.
    let cols = sample_cols(n);
    let gemv = gemv_lloyd_cols(gpu, &d_a, &x_host, m, k, n, &cols, lut_f16);
    if add {
        let mut delta = y_mmq.clone();
        for (i, y0) in y0_host.iter().enumerate() {
            delta[i] -= y0;
        }
        report_vs_gemv(tag, &delta, m, &gemv);
    } else {
        report_vs_gemv(tag, &y_mmq, m, &gemv);
    }

    if time && n == 512 {
        // Force x128 set/add symbols for the timing the contract names.
        let iters = 5usize;
        for &(add_t, label) in &[(false, "set_x128"), (true, "add_x128")] {
            let d_yt = gpu.zeros(&[n * m], DType::F32).expect("y time");
            // warmup
            launch_lloyd_mmq(gpu, &d_a, xq, &d_yt, m, k, n, add_t, c16, Some(true));
            gpu.hip.device_synchronize().ok();
            let t0 = Instant::now();
            for _ in 0..iters {
                launch_lloyd_mmq(gpu, &d_a, xq, &d_yt, m, k, n, add_t, c16, Some(true));
            }
            gpu.hip.device_synchronize().ok();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  TIME lloyd {tag} {label}: {us:.1} us/launch");
        }
    }
}

fn run_uniform_shape(
    gpu: &mut Gpu,
    tag: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    time: bool,
) {
    eprintln!("=== Uniform {tag} M={m} K={k} N={n} add={add} ===");
    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A uni");
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| (prng(i, 0xA11CE) * 2.0 - 1.0) * 0.25)
        .collect();
    let d_x = gpu.upload_f32(&x_host, &[n * k]).expect("upload X");
    let y0_host: Vec<f32> = if add {
        (0..n * m)
            .map(|i| (prng(i, 0xADD0) * 2.0 - 1.0) * 0.05)
            .collect()
    } else {
        vec![0.0; n * m]
    };
    let d_y = if add {
        gpu.upload_f32(&y0_host, &[n * m]).expect("Y0")
    } else {
        gpu.zeros(&[n * m], DType::F32).expect("Y")
    };

    let xq = if gpu.flags.gfx11_mmq_x128_enabled() {
        gpu.ensure_q8_1_mmq_x128(&d_x, n, k).expect("q8")
    } else {
        gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("q8")
    };
    if add {
        gpu.gemm_mq4g256v2_mmq_add_prequant(&d_a, xq, &d_y, m, k, n)
            .expect("uniform add");
    } else {
        gpu.gemm_mq4g256v2_mmq_set_prequant(&d_a, xq, &d_y, m, k, n)
            .expect("uniform set");
    }
    gpu.hip.device_synchronize().expect("sync uni");
    let y_mmq = gpu.download_f32(&d_y).expect("dl uni");

    let cols = sample_cols(n);
    let gemv = gemv_uniform_cols(gpu, &d_a, &x_host, m, k, n, &cols);
    if add {
        let mut delta = y_mmq.clone();
        for (i, y0) in y0_host.iter().enumerate() {
            delta[i] -= y0;
        }
        report_vs_gemv(&format!("uniform-{tag}"), &delta, m, &gemv);
    } else {
        report_vs_gemv(&format!("uniform-{tag}"), &y_mmq, m, &gemv);
    }

    if time && n == 512 {
        let iters = 5usize;
        for &(add_t, label) in &[(false, "set_prequant"), (true, "add_prequant")] {
            let d_yt = gpu.zeros(&[n * m], DType::F32).expect("y");
            let launch = |g: &mut Gpu| {
                if add_t {
                    g.gemm_mq4g256v2_mmq_add_prequant(&d_a, xq, &d_yt, m, k, n)
                        .expect("t");
                } else {
                    g.gemm_mq4g256v2_mmq_set_prequant(&d_a, xq, &d_yt, m, k, n)
                        .expect("t");
                }
            };
            launch(gpu);
            gpu.hip.device_synchronize().ok();
            let t0 = Instant::now();
            for _ in 0..iters {
                launch(gpu);
            }
            gpu.hip.device_synchronize().ok();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  TIME uniform {tag} {label}: {us:.1} us/launch");
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path52 = args
        .next()
        .expect("usage: tmp_lloyd_mmq_diff <qt52.hfq> <qt44.hfq>");
    let path44 = args.next().expect("missing qt44.hfq");
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let hfq52 = HfqFile::open(Path::new(&path52)).expect("open qt52");
    let hfq44 = HfqFile::open(Path::new(&path44)).expect("open qt44");

    let gate = "layers.10.mlp.gate_proj.weight";
    let down = "layers.10.mlp.down_proj.weight";

    let (mg, kg, mut gate52) = load_weight(&hfq52, gate, 52);
    let (md, kd, mut down52) = load_weight(&hfq52, down, 52);
    apply_lloyd_centering(&mut gate52, mg, kg).expect("center gate");
    apply_lloyd_centering(&mut down52, md, kd).expect("center down");
    let (_e4g, lut_g, c16_g) = load_lloyd_lut(&hfq52, gate).expect("lut gate");
    let (_e4d, lut_d, c16_d) = load_lloyd_lut(&hfq52, down).expect("lut down");

    let (mg44, kg44, gate44) = load_weight(&hfq44, gate, 44);
    let (md44, kd44, down44) = load_weight(&hfq44, down, 44);
    assert_eq!((mg, kg), (mg44, kg44), "gate shape mismatch 52 vs 44");
    assert_eq!((md, kd), (md44, kd44), "down shape mismatch 52 vs 44");
    eprintln!("gate M={mg} K={kg}; down M={md} K={kd}");
    eprintln!(
        "Outcome A (signed i8 C16); c16_gate={c16_g:08x?} c16_down={c16_d:08x?}"
    );

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!(
        "arch={} x128={}",
        gpu.arch,
        gpu.flags.gfx11_mmq_x128_enabled()
    );

    // N=512 correctness + optional timing; N=511 base (non-full) correctness.
    for &n in &[512usize, 511] {
        run_lloyd_shape(
            &mut gpu, "gate_proj/set", &gate52, mg, kg, n, false, c16_g, lut_g, time && n == 512,
        );
        run_lloyd_shape(
            &mut gpu, "down_proj/add", &down52, md, kd, n, true, c16_d, lut_d, time && n == 512,
        );
        run_uniform_shape(
            &mut gpu, "gate_proj/set", &gate44, mg, kg, n, false, time && n == 512,
        );
        run_uniform_shape(
            &mut gpu, "down_proj/add", &down44, md, kd, n, true, time && n == 512,
        );
    }
    eprintln!("done (ADVISORY diffs only; no hard fail)");
}
