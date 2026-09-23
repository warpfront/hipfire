//! Throwaway iu4 attribution runner: JITs a TU copy from disk and times the
//! N=512 gate_set case exactly like `tmp_iu4_gfx12_oracle -- TIME`.
//!
//! Usage: tmp_iu4_attrib <model.mq4-xt> <tu.hip> <tag>
//! Prints one line: ATTRIB <tag>: med=<us> us/call ...
//! The JIT module is `tmp_iu4_attrib_<tag>`, so each variant keeps its own
//! cache entry (.hip/.hsaco/.radiowave.json) for ISA inspection.

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const SET_SYM: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set";
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";

fn prng_f32(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn cache_dir(arch: &str) -> PathBuf {
    let root = std::env::var_os("HIPFIRE_KERNEL_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into());
        PathBuf::from(home).join(".hipfire_kernels")
    });
    root.join(arch)
}

fn load_mq4v2(hfq: &HfqFile, name: &str) -> (usize, usize, Vec<u8>) {
    let (info, bytes) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    assert_eq!(info.quant_type, QT_MQ4V2, "{name}: quant_type mismatch");
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    (m, k, bytes)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: tmp_iu4_attrib <model.mq4-xt> <tu.hip> <tag>");
        std::process::exit(2);
    }
    let (model, tu_path, tag) = (args[0].clone(), args[1].clone(), args[2].clone());
    let src = std::fs::read_to_string(&tu_path).expect("read TU");
    let module = format!("tmp_iu4_attrib_{tag}");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    assert_eq!(gpu.arch, "gfx1201", "this runner targets gfx1201");
    gpu.ensure_kernel_public(&module, &src, SET_SYM)
        .unwrap_or_else(|e| panic!("JIT {SET_SYM}: {e}"));
    eprintln!("JIT OK: {SET_SYM} [{module}]");
    let _ = cache_dir(&gpu.arch);

    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let (gate_m, gate_k, gate) = load_mq4v2(&hfq, GATE_NAME);
    drop(hfq);

    // gate_set N=512, same numbers as the oracle TIME row.
    let (m, k, n) = (gate_m, gate_k, 512usize);
    let x: Vec<f32> = (0..n * k).map(|i| prng_f32(i, 0x71E0).mul_add(1.0, 0.0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr =
        gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure_int4_mmq_x") as *mut std::ffi::c_void;
    let d_a = gpu.upload_raw(&gate, &[gate.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;
    let y0 = vec![0.0f32; n * m];
    let d_y = gpu.upload_f32(&y0, &[n, m]).expect("upload Y");
    let y_ptr = d_y.buf.as_ptr() as *const std::ffi::c_void;
    let grid = [m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1];
    let mut run = || {
        let mut b = KernargBlob::new();
        b.push_ptr(a_ptr);
        b.push_ptr(xq_ptr as *const std::ffi::c_void);
        b.push_ptr(y_ptr);
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(n as i32);
        b.push_i32(0);
        let mut blob = b.into_vec();
        gpu.launch_kernel_blob(SET_SYM, grid, [256, 1, 1], 19456, &mut blob)
            .expect("launch")
    };
    for _ in 0..10 {
        run();
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut meds = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            run();
        }
        gpu.hip.device_synchronize().expect("sync");
        meds.push(t0.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = meds[2];
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    let tops = gflop / med * 1000.0;
    eprintln!("ATTRIB {tag}: med={med:.1} us/call GFLOP={gflop:.2} -> {tops:.1} TOPS");
    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y);
}
