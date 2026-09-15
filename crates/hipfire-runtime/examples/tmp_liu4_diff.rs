//! THROWAWAY K1 differential for the gfx11 iu4-split Lloyd twin.
//!
//! Build only (parent runs the XTX gate):
//!   cargo build --release -p hipfire-runtime --example tmp_liu4_diff
//!
//! Run (parent / XTX):
//!   TIME=1 target/release/examples/tmp_liu4_diff \
//!     /path/to/qt52.hfq /path/to/qt44.hfq
//!
//! argv[1] = qt=52 Lloyd artifact (gate_proj set M=17408 K=5120, down_proj add
//! M=5120 K=17408). argv[2] = qt=44 uniform artifact same shapes (timing only).
//! N=512 full tiles; N=511 base entry (structurally tail-capable).

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::{load_lloyd_lut, HfqFile};
use hipfire_runtime::lloyd_lut::apply_lloyd_centering;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::time::Instant;

const IU4_LUT_SRC: &str = concat!(
    "#define HIPFIRE_MMQ_IU4_LUT 1\n",
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
const MODULE_LUT: &str = "gemm_mq4g256v2_residual_mmq_iu4_lloyd";

const GATE: &str = "model.language_model.layers.10.mlp.gate_proj.weight";
const DOWN: &str = "model.language_model.layers.10.mlp.down_proj.weight";

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn load_centered_weight(hfq: &HfqFile, name: &str) -> (Vec<u8>, usize, usize, [u32; 4], [u32; 8]) {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    assert_eq!(
        info.shape.len(),
        2,
        "{name}: expected rank-2 shape, got {:?}",
        info.shape
    );
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let mut blob = data;
    apply_lloyd_centering(&mut blob, m, k).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (_e4, lut_f16, c16) = load_lloyd_lut(hfq, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    (blob, m, k, c16, lut_f16)
}

fn load_uniform_weight(hfq: &HfqFile, name: &str) -> (Vec<u8>, usize, usize) {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    (data, m, k)
}

fn make_x(n: usize, k: usize, salt: u32) -> Vec<f32> {
    (0..n * k)
        .map(|i| prng(i, salt) * 2.0 - 1.0)
        .collect()
}

fn make_y0(n: usize, m: usize, salt: u32) -> Vec<f32> {
    (0..n * m)
        .map(|i| prng(i, salt) * 0.25 + 0.125) // strictly nonzero
        .collect()
}

fn shared_mem_iu4() -> u32 {
    const MMQ_X: usize = 128;
    const MMQ_Y: usize = 128;
    const MMQ_TILE_Y_K: usize = 18;
    const MMQ_TILE_X_K: usize = 44;
    ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32
}

fn launch_iu4_lloyd(
    gpu: &mut Gpu,
    kernel: &str,
    a: &GpuTensor,
    x_i4: *mut std::ffi::c_void,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    add: i32,
    c16: [u32; 4],
) {
    gpu.ensure_kernel_public(MODULE_LUT, IU4_LUT_SRC, kernel)
        .unwrap_or_else(|e| panic!("JIT {kernel}: {e}"));
    let row_tiles = m.div_ceil(128) as u32;
    let batch_tiles = n.div_ceil(128) as u32;
    let mut blob = KernargBlob::new();
    blob.push_ptr(a.buf.as_ptr());
    blob.push_ptr(x_i4);
    blob.push_ptr(y.buf.as_ptr());
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);
    blob.push_i32(n as i32);
    blob.push_i32(add);
    blob.push_u32(c16[0]);
    blob.push_u32(c16[1]);
    blob.push_u32(c16[2]);
    blob.push_u32(c16[3]);
    let mut bytes = blob.into_vec();
    gpu.launch_kernel_blob(
        kernel,
        [row_tiles, batch_tiles, 1],
        [32, 8, 1],
        shared_mem_iu4(),
        &mut bytes,
    )
    .unwrap_or_else(|e| panic!("launch {kernel}: {e}"));
}

fn rel_l2_cols(got: &[f32], want: &[f32], m: usize, cols: &[usize]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for &j in cols {
        let go = &got[j * m..(j + 1) * m];
        let wo = &want[j * m..(j + 1) * m];
        for (&g, &w) in go.iter().zip(wo.iter()) {
            let d = g as f64 - w as f64;
            num += d * d;
            den += (w as f64) * (w as f64);
        }
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        num.sqrt() / den.sqrt()
    }
}

fn gemv_oracle_cols(
    gpu: &mut Gpu,
    a: &GpuTensor,
    x_host: &[f32],
    m: usize,
    k: usize,
    n: usize,
    cols: &[usize],
    lut_f16: [u32; 8],
    y0: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = match y0 {
        Some(y) => y.to_vec(),
        None => vec![0.0f32; n * m],
    };
    for &j in cols {
        let xj = &x_host[j * k..(j + 1) * k];
        let d_x = gpu.upload_f32(xj, &[k]).unwrap();
        let mut y_col = match y0 {
            Some(y) => y[j * m..(j + 1) * m].to_vec(),
            None => vec![0.0f32; m],
        };
        let d_y = gpu.upload_f32(&y_col, &[m]).unwrap();
        // set-style oracle; for add we fold y0 host-side after
        if y0.is_some() {
            // residual path: y += W x via plain set then add y0
            let d_y_set = gpu.zeros(&[m], DType::F32).unwrap();
            gpu.gemv_mq4g256v2_lloyd(a, &d_x, &d_y_set, m, k, lut_f16)
                .unwrap_or_else(|e| panic!("gemv lloyd: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let set = gpu.download_f32(&d_y_set).unwrap();
            for i in 0..m {
                y_col[i] += set[i];
            }
        } else {
            gpu.gemv_mq4g256v2_lloyd(a, &d_x, &d_y, m, k, lut_f16)
                .unwrap_or_else(|e| panic!("gemv lloyd: {e}"));
            gpu.hip.device_synchronize().unwrap();
            y_col = gpu.download_f32(&d_y).unwrap();
        }
        out[j * m..(j + 1) * m].copy_from_slice(&y_col);
    }
    out
}

fn kernel_for(full: bool, add: bool) -> &'static str {
    match (full, add) {
        (true, true) => "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_lloyd",
        (true, false) => "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_lloyd",
        (false, true) | (false, false) => "gemm_mq4g256v2_residual_mmq_iu4_lloyd",
    }
}

fn run_case(
    gpu: &mut Gpu,
    label: &str,
    blob: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    c16: [u32; 4],
    lut_f16: [u32; 8],
    time: bool,
) {
    let full = m % 128 == 0 && n % 128 == 0;
    let kernel = kernel_for(full, add);
    eprintln!("=== {label}: M={m} K={k} N={n} add={add} full={full} kernel={kernel}");

    let x = make_x(n, k, 0xC16_51A7);
    let y0 = if add {
        Some(make_y0(n, m, 0xADD_A11))
    } else {
        None
    };

    let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y = match &y0 {
        Some(y) => gpu.upload_f32(y, &[n, m]).unwrap(),
        None => gpu.zeros(&[n * m], DType::F32).unwrap(),
    };

    let x_i4 = gpu
        .ensure_int4_mmq_x(&d_x, n, k)
        .unwrap_or_else(|e| panic!("ensure_int4_mmq_x: {e}"));

    // correctness launch
    launch_iu4_lloyd(
        gpu,
        kernel,
        &d_a,
        x_i4,
        &d_y,
        m,
        k,
        n,
        i32::from(add),
        c16,
    );
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_y).unwrap();

    let cols: Vec<usize> = [0usize, 1, n / 2, n.saturating_sub(2), n.saturating_sub(1)]
        .into_iter()
        .filter(|&c| c < n)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let want = gemv_oracle_cols(
        gpu,
        &d_a,
        &x,
        m,
        k,
        n,
        &cols,
        lut_f16,
        y0.as_deref(),
    );
    let rel = rel_l2_cols(&got, &want, m, &cols);
    eprintln!("  rel-L2 cols {cols:?} = {rel:.6e}");
    const TOL: f64 = 3e-4;
    if rel > TOL {
        eprintln!("  FAIL rel-L2 {rel:.6e} > {TOL:.0e}");
        std::process::exit(1);
    } else {
        eprintln!("  PASS rel-L2 ≤ {TOL:.0e}");
    }

    if time {
        // twin timing
        let d_y_t = gpu.zeros(&[n * m], DType::F32).unwrap();
        let warmup = 3;
        let runs = 10;
        for _ in 0..warmup {
            launch_iu4_lloyd(
                gpu,
                kernel,
                &d_a,
                x_i4,
                &d_y_t,
                m,
                k,
                n,
                0, // set for stable timing
                c16,
            );
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..runs {
            launch_iu4_lloyd(
                gpu,
                kernel,
                &d_a,
                x_i4,
                &d_y_t,
                m,
                k,
                n,
                0,
                c16,
            );
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
        eprintln!("  TIME twin {kernel}: {us:.1} us/launch");
    }
}

fn time_uniform_iu4(gpu: &mut Gpu, blob: &[u8], m: usize, k: usize, n: usize) {
    let x = make_x(n, k, 0xC16_51A7);
    let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y = gpu.zeros(&[n * m], DType::F32).unwrap();
    let x_i4 = gpu.ensure_int4_mmq_x(&d_x, n, k).unwrap();
    let warmup = 3;
    let runs = 10;
    for _ in 0..warmup {
        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&d_a, x_i4, &d_y, m, k, n)
            .unwrap_or_else(|e| panic!("uniform iu4: {e}"));
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..runs {
        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&d_a, x_i4, &d_y, m, k, n)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
    let full = m % 128 == 0 && n % 128 == 0;
    let name = if full {
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3"
    } else {
        "gemm_mq4g256v2_residual_mmq_iu4"
    };
    eprintln!("  TIME uniform {name}: {us:.1} us/launch (M={m} K={k} N={n})");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let qt52 = args
        .next()
        .unwrap_or_else(|| "/home/kaden/qcal/lloyd/qwen3.8-27b.mq4v2l-tensor-qt52.base.hfq".into());
    let qt44 = args.next(); // optional for TIME=
    let time = std::env::var("TIME").as_deref() == Ok("1");

    let mut gpu = Gpu::init().unwrap_or_else(|e| panic!("Gpu::init: {e}"));
    eprintln!("arch={} TIME={time}", gpu.arch);

    let hfq52 = HfqFile::open(Path::new(&qt52)).unwrap_or_else(|e| panic!("open {qt52}: {e}"));
    let (gate_blob, gm, gk, gate_c16, gate_f16) = load_centered_weight(&hfq52, GATE);
    let (down_blob, dm, dk, down_c16, down_f16) = load_centered_weight(&hfq52, DOWN);
    eprintln!("gate M={gm} K={gk}; down M={dm} K={dk}");

    // N=512 full tiles (gate set, down add)
    run_case(
        &mut gpu,
        "gate_set_N512",
        &gate_blob,
        gm,
        gk,
        512,
        false,
        gate_c16,
        gate_f16,
        time,
    );
    run_case(
        &mut gpu,
        "down_add_N512",
        &down_blob,
        dm,
        dk,
        512,
        true,
        down_c16,
        down_f16,
        time,
    );

    // N=511 base entry (tail contract)
    run_case(
        &mut gpu,
        "gate_set_N511",
        &gate_blob,
        gm,
        gk,
        511,
        false,
        gate_c16,
        gate_f16,
        false,
    );
    run_case(
        &mut gpu,
        "down_add_N511",
        &down_blob,
        dm,
        dk,
        511,
        true,
        down_c16,
        down_f16,
        false,
    );

    if time {
        let path44 = qt44.unwrap_or_else(|| {
            panic!("TIME=1 requires argv[2] = qt=44 uniform .hfq path for gate number")
        });
        let hfq44 = HfqFile::open(Path::new(&path44))
            .unwrap_or_else(|e| panic!("open {path44}: {e}"));
        let (g44, gm44, gk44) = load_uniform_weight(&hfq44, GATE);
        let (d44, dm44, dk44) = load_uniform_weight(&hfq44, DOWN);
        assert_eq!((gm44, gk44), (gm, gk), "qt44 gate shape mismatch");
        assert_eq!((dm44, dk44), (dm, dk), "qt44 down shape mismatch");
        eprintln!("=== uniform iu4 timing on {path44}");
        time_uniform_iu4(&mut gpu, &g44, gm44, gk44, 512);
        time_uniform_iu4(&mut gpu, &d44, dm44, dk44, 512);
    }

    eprintln!("tmp_liu4_diff: done");
}
