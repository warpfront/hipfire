//! KX3 fold-free proxy timing probe (untracked research harness).
//! Uses synthetic device-resident buffers. Numerical output is intentionally
//! meaningless; only the body instruction stream and device time are measured.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::Gpu;

const BATCH: usize = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const BM: usize = 128;
const BN: usize = 128;
const FP8_LDS: u32 = 19_968;
const IU4_LDS: u32 = 19_456;

const FP8_GATE_SHIPPED: &str = concat!(
    "#define HIPFIRE_FP8_V2_TILE 1\n#define HIPFIRE_FP8_V2_BM 128\n#define HIPFIRE_FP8_V2_BN 128\n#define HIPFIRE_FP8_V2_BK 64\n#define HIPFIRE_FP8_V2_WAVES 8\n#define HIPFIRE_FP8_GATEUP_KERNEL kx3_fp8_gate_shipped\n",
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip")
);
const FP8_GATE_PROXY: &str = concat!(
    "#define HIPFIRE_FP8_V2_TILE 1\n#define HIPFIRE_FP8_V2_BM 128\n#define HIPFIRE_FP8_V2_BN 128\n#define HIPFIRE_FP8_V2_BK 64\n#define HIPFIRE_FP8_V2_WAVES 8\n#define HIPFIRE_FP8_GATEUP_KERNEL kx3_fp8_gate_foldfree\n",
    include_str!("../../../.codeinsight+research/scratch-2026-09-17/rocm10-levers/kx/KX3/proxy/gemm_gate_up_mq4g256v2_wmma_fp8_foldfree_proxy.gfx12.hip")
);
const FP8_RES_SHIPPED: &str = concat!(
    "#define HIPFIRE_FP8_V2_TILE 1\n#define HIPFIRE_FP8_V2_BM 128\n#define HIPFIRE_FP8_V2_BN 128\n#define HIPFIRE_FP8_V2_BK 64\n#define HIPFIRE_FP8_V2_WAVES 8\n#define HIPFIRE_FP8_RESIDUAL 1\n#define HIPFIRE_FP8_GATEUP_KERNEL kx3_fp8_residual_shipped\n",
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip")
);
const FP8_RES_PROXY: &str = concat!(
    "#define HIPFIRE_FP8_V2_TILE 1\n#define HIPFIRE_FP8_V2_BM 128\n#define HIPFIRE_FP8_V2_BN 128\n#define HIPFIRE_FP8_V2_BK 64\n#define HIPFIRE_FP8_V2_WAVES 8\n#define HIPFIRE_FP8_RESIDUAL 1\n#define HIPFIRE_FP8_GATEUP_KERNEL kx3_fp8_residual_foldfree\n",
    include_str!("../../../.codeinsight+research/scratch-2026-09-17/rocm10-levers/kx/KX3/proxy/gemm_residual_mq4g256v2_wmma_fp8_foldfree_proxy.gfx12.hip")
);
const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip")
);
const IU4_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set";
const IU4_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add";

struct ShapeBuffers {
    a0: DeviceBuffer,
    a1: Option<DeviceBuffer>,
    a_iu4: DeviceBuffer,
    x8: DeviceBuffer,
    sums: DeviceBuffer,
    row_scales: DeviceBuffer,
    xq: DeviceBuffer,
    y0: DeviceBuffer,
    y1: Option<DeviceBuffer>,
    y_iu4: DeviceBuffer,
    out_rows: usize,
    k: usize,
}

fn zero(hip: &hip_bridge::HipRuntime, bytes: usize) -> DeviceBuffer {
    let b = hip.malloc(bytes).expect("hipMalloc");
    hip.memset(&b, 0, bytes).expect("hipMemset");
    b
}

fn make_buffers(gpu: &Gpu, out_rows: usize, k: usize, gate_up: bool) -> ShapeBuffers {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let one_rows = if gate_up { out_rows / 2 } else { out_rows };
    let weight_bytes = one_rows * gpr * 136;
    ShapeBuffers {
        a0: zero(&gpu.hip, weight_bytes),
        a1: gate_up.then(|| zero(&gpu.hip, weight_bytes)),
        a_iu4: zero(&gpu.hip, out_rows * gpr * 136),
        x8: zero(&gpu.hip, BATCH * k),
        sums: zero(&gpu.hip, BATCH * gpr * 2 * 4),
        row_scales: zero(&gpu.hip, BATCH * 4),
        xq: zero(&gpu.hip, BATCH * (k / 128) * 72),
        y0: zero(&gpu.hip, BATCH * one_rows * 4),
        y1: gate_up.then(|| zero(&gpu.hip, BATCH * one_rows * 4)),
        y_iu4: zero(&gpu.hip, BATCH * out_rows * 4),
        out_rows,
        k,
    }
}

fn launch_gate(gpu: &Gpu, sym: &str, b: &ShapeBuffers) {
    let half = b.out_rows / 2;
    let mut a = KernargBlob::new();
    a.push_ptr(b.a0.as_ptr());
    a.push_ptr(b.a1.as_ref().unwrap().as_ptr());
    a.push_ptr(b.x8.as_ptr());
    a.push_ptr(b.sums.as_ptr());
    a.push_ptr(b.row_scales.as_ptr());
    a.push_ptr(b.y0.as_ptr());
    a.push_ptr(b.y1.as_ref().unwrap().as_ptr());
    a.push_i32(half as i32);
    a.push_i32(half as i32);
    a.push_i32(b.k as i32);
    a.push_i32(BATCH as i32);
    let mut bytes = a.into_vec();
    gpu.launch_kernel_blob(
        sym,
        [b.out_rows.div_ceil(BN) as u32, BATCH.div_ceil(BM) as u32, 1],
        [256, 1, 1],
        FP8_LDS,
        &mut bytes,
    )
    .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn launch_residual(gpu: &Gpu, sym: &str, b: &ShapeBuffers) {
    let mut a = KernargBlob::new();
    a.push_ptr(b.a0.as_ptr());
    a.push_ptr(b.x8.as_ptr());
    a.push_ptr(b.sums.as_ptr());
    a.push_ptr(b.row_scales.as_ptr());
    a.push_ptr(b.y0.as_ptr());
    a.push_i32(b.out_rows as i32);
    a.push_i32(b.k as i32);
    a.push_i32(BATCH as i32);
    let mut bytes = a.into_vec();
    gpu.launch_kernel_blob(
        sym,
        [b.out_rows.div_ceil(BN) as u32, BATCH.div_ceil(BM) as u32, 1],
        [256, 1, 1],
        FP8_LDS,
        &mut bytes,
    )
    .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn launch_iu4(gpu: &Gpu, sym: &str, b: &ShapeBuffers, add: bool) {
    let mut a = KernargBlob::new();
    a.push_ptr(b.a_iu4.as_ptr());
    a.push_ptr(b.xq.as_ptr());
    a.push_ptr(b.y_iu4.as_ptr());
    a.push_i32(b.out_rows as i32);
    a.push_i32(b.k as i32);
    a.push_i32(BATCH as i32);
    a.push_i32(i32::from(add));
    let mut bytes = a.into_vec();
    gpu.launch_kernel_blob(
        sym,
        [b.out_rows.div_ceil(128) as u32, BATCH.div_ceil(128) as u32, 1],
        [256, 1, 1],
        IU4_LDS,
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

fn report(label: &str, shape: &str, out_rows: usize, k: usize, us: f64) {
    let flops = 2.0 * BATCH as f64 * out_rows as f64 * k as f64;
    let tfs = flops / us / 1.0e6;
    println!("RESULT,{label},{shape},{us:.3},{tfs:.3}");
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "KX3 targets gfx1201");
    assert!(gpu.active_stream.is_none(), "probe expects the default HIP stream");

    let specs = [
        ("kx3_fp8_gate_shipped_mod", FP8_GATE_SHIPPED, "kx3_fp8_gate_shipped"),
        ("kx3_fp8_gate_foldfree_mod", FP8_GATE_PROXY, "kx3_fp8_gate_foldfree"),
        ("kx3_fp8_residual_shipped_mod", FP8_RES_SHIPPED, "kx3_fp8_residual_shipped"),
        ("kx3_fp8_residual_foldfree_mod", FP8_RES_PROXY, "kx3_fp8_residual_foldfree"),
        ("kx3_iu4_shipped_mod", IU4_SRC, IU4_SET),
        ("kx3_iu4_shipped_mod", IU4_SRC, IU4_ADD),
    ];
    for (module, src, sym) in specs {
        gpu.ensure_kernel_public(module, src, sym)
            .unwrap_or_else(|e| panic!("JIT {module}/{sym}: {e}"));
        let occ = gpu.occupancy_max_active_blocks(sym, [256, 1, 1], if sym.starts_with("kx3_fp8") { FP8_LDS } else { IU4_LDS })
            .unwrap_or(-1);
        eprintln!("RESOURCE_OCC,{sym},blocks_per_cu={occ}");
    }

    // Canonical gate_up is two 17408-row outputs: mathematical [4096,5120]
    // by [5120,34816]. The synthetic iu4 control uses one contiguous 34816-row
    // allocation so the body sees the same total grid as the fused fp8 kernel.
    let gate = make_buffers(&gpu, 34_816, 5_120, true);
    let us = median_event_us(&gpu, || launch_gate(&gpu, "kx3_fp8_gate_shipped", &gate));
    report("fp8v2_as_shipped", "gate_up_M4096_K5120_N34816", gate.out_rows, gate.k, us);
    let us = median_event_us(&gpu, || launch_gate(&gpu, "kx3_fp8_gate_foldfree", &gate));
    report("fp8v2_foldfree_proxy", "gate_up_M4096_K5120_N34816", gate.out_rows, gate.k, us);
    let us = median_event_us(&gpu, || launch_iu4(&gpu, IU4_SET, &gate, false));
    report("iu4_as_shipped", "gate_up_M4096_K5120_N34816", gate.out_rows, gate.k, us);

    let residual = make_buffers(&gpu, 5_120, 17_408, false);
    let us = median_event_us(&gpu, || launch_residual(&gpu, "kx3_fp8_residual_shipped", &residual));
    report("fp8v2_as_shipped", "residual_M4096_K17408_N5120", residual.out_rows, residual.k, us);
    let us = median_event_us(&gpu, || launch_residual(&gpu, "kx3_fp8_residual_foldfree", &residual));
    report("fp8v2_foldfree_proxy", "residual_M4096_K17408_N5120", residual.out_rows, residual.k, us);
    let us = median_event_us(&gpu, || launch_iu4(&gpu, IU4_ADD, &residual, true));
    report("iu4_as_shipped", "residual_M4096_K17408_N5120", residual.out_rows, residual.k, us);

}
