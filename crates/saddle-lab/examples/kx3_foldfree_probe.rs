//! Untracked IU4 pipeline A/B/A timing probe.
//! Compiles the immutable 99c44a96b baseline and the current candidate under
//! distinct symbols, then times both on the same device-resident buffers.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::Gpu;

const BATCH: usize = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const BASE_LDS: u32 = 19_456;
const PRELUDE: &str = include_str!("../../../kernels/src/block_i4_128_quant.hip");
const BASE_KERNEL: &str = include_str!("../../../scratch-2026-09-17/Iu4Pipe/baseline_99c44a96b.hip");
const CAND_KERNEL: &str = include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip");
const BASE_SET: &str = "kx3_iu4_base_set";
const BASE_ADD: &str = "kx3_iu4_base_add";
const CAND_SET: &str = "kx3_iu4_cand_set";
const CAND_ADD: &str = "kx3_iu4_cand_add";

struct ShapeBuffers {
    a: DeviceBuffer,
    xq: DeviceBuffer,
    y: DeviceBuffer,
    out_rows: usize,
    k: usize,
}

fn zero(hip: &hip_bridge::HipRuntime, bytes: usize) -> DeviceBuffer {
    let b = hip.malloc(bytes).expect("hipMalloc");
    hip.memset(&b, 0, bytes).expect("hipMemset");
    b
}

fn make_buffers(gpu: &Gpu, out_rows: usize, k: usize) -> ShapeBuffers {
    assert_eq!(k % 256, 0);
    ShapeBuffers {
        a: zero(&gpu.hip, out_rows * (k / 256) * 136),
        xq: zero(&gpu.hip, BATCH * (k / 128) * 72),
        y: zero(&gpu.hip, BATCH * out_rows * 4),
        out_rows,
        k,
    }
}

fn renamed_source(kernel: &str, main: &str, set: &str, add: &str) -> String {
    let body = kernel
        .replace("gemm_mq4g256v2_residual_mmq_iu4_full_set", set)
        .replace("gemm_mq4g256v2_residual_mmq_iu4_full_add", add)
        .replace("gemm_mq4g256v2_residual_mmq_iu4", main);
    format!("{PRELUDE}{body}")
}

fn launch(gpu: &Gpu, sym: &str, b: &ShapeBuffers, add: bool, lds: u32) {
    let mut a = KernargBlob::new();
    a.push_ptr(b.a.as_ptr());
    a.push_ptr(b.xq.as_ptr());
    a.push_ptr(b.y.as_ptr());
    a.push_i32(b.out_rows as i32);
    a.push_i32(b.k as i32);
    a.push_i32(BATCH as i32);
    a.push_i32(i32::from(add));
    let mut bytes = a.into_vec();
    gpu.launch_kernel_blob(
        sym,
        [b.out_rows.div_ceil(128) as u32, BATCH.div_ceil(128) as u32, 1],
        [256, 1, 1],
        lds,
        &mut bytes,
    )
    .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn median_event_us<F: FnMut()>(gpu: &Gpu, mut launch: F) -> f64 {
    for _ in 0..WARMUP {
        launch();
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut pairs = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = gpu.hip.event_create().expect("start event");
        let stop = gpu.hip.event_create().expect("stop event");
        gpu.hip.event_record(&start, gpu.active_stream.as_ref()).expect("record start");
        launch();
        gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).expect("record stop");
        pairs.push((start, stop));
    }
    gpu.hip.event_synchronize(&pairs.last().unwrap().1).expect("timing sync");
    let mut us: Vec<f64> = pairs
        .iter()
        .map(|(s, e)| gpu.hip.event_elapsed_ms(s, e).expect("elapsed") as f64 * 1000.0)
        .collect();
    for (s, e) in pairs {
        gpu.hip.event_destroy(s).expect("destroy start");
        gpu.hip.event_destroy(e).expect("destroy stop");
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (us[SAMPLES / 2 - 1] + us[SAMPLES / 2]) * 0.5
}

fn time_triplet(gpu: &Gpu, shape: &str, b: &ShapeBuffers, add: bool, cand_lds: u32) {
    let (base_sym, cand_sym) = if add { (BASE_ADD, CAND_ADD) } else { (BASE_SET, CAND_SET) };
    // One untimed launch per freshly JIT'd arm before each shape, then the
    // mandated 5 warmups/20 measured events inside each A/B/A arm.
    launch(gpu, base_sym, b, add, BASE_LDS);
    launch(gpu, cand_sym, b, add, cand_lds);
    gpu.hip.device_synchronize().expect("build warm sync");
    let a1 = median_event_us(gpu, || launch(gpu, base_sym, b, add, BASE_LDS));
    let bv = median_event_us(gpu, || launch(gpu, cand_sym, b, add, cand_lds));
    let a2 = median_event_us(gpu, || launch(gpu, base_sym, b, add, BASE_LDS));
    let abase = (a1 + a2) * 0.5;
    println!(
        "RESULT,{shape},A1={a1:.3},B={bv:.3},A2={a2:.3},ABASE={abase:.3},SPEEDUP_PCT={:.3}",
        (abase / bv - 1.0) * 100.0
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "probe targets gfx1201");
    assert!(gpu.active_stream.is_none(), "probe expects the default HIP stream");
    let cand_lds = std::env::var("IU4_CAND_LDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_480);
    let base_src = renamed_source(BASE_KERNEL, "kx3_iu4_base", BASE_SET, BASE_ADD);
    let cand_src = renamed_source(CAND_KERNEL, "kx3_iu4_cand", CAND_SET, CAND_ADD);
    for (module, src, sym, lds) in [
        ("kx3_iu4_base_mod", base_src.as_str(), BASE_SET, BASE_LDS),
        ("kx3_iu4_base_mod", base_src.as_str(), BASE_ADD, BASE_LDS),
        ("kx3_iu4_cand_mod", cand_src.as_str(), CAND_SET, cand_lds),
        ("kx3_iu4_cand_mod", cand_src.as_str(), CAND_ADD, cand_lds),
    ] {
        gpu.ensure_kernel_public(module, src, sym)
            .unwrap_or_else(|e| panic!("JIT {module}/{sym}: {e}"));
        let occ = gpu.occupancy_max_active_blocks(sym, [256, 1, 1], lds).unwrap_or(-1);
        eprintln!("RESOURCE_OCC,{sym},lds={lds},blocks_per_cu={occ}");
    }
    let gate = make_buffers(&gpu, 34_816, 5_120);
    time_triplet(&gpu, "gate_preflight", &gate, false, cand_lds);
    time_triplet(&gpu, "gate_up_M4096_K5120_N34816", &gate, false, cand_lds);
    let residual = make_buffers(&gpu, 5_120, 17_408);
    time_triplet(&gpu, "residual_M4096_K17408_N5120", &residual, true, cand_lds);
}
