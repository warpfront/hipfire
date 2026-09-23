//! Wave64 iu4 K-loop probe (untracked research rig).
//! Synthetic nonzero MQ4V2/block_i4_128 buffers; shipped and wave64 arms share inputs.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::Gpu;

const BATCH: usize = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const LDS: u32 = 20_480;
const BLOCK: [u32; 3] = [256, 1, 1];
const BLOCK_B: [u32; 3] = [512, 1, 1];

const SHIPPED_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip")
);
const W64_A_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    "#define IU4_W64_RGS 2\n",
    "#define IU4_W64_KERNEL kx3_iu4_w64a\n",
    "#define IU4_W64_ADD_KERNEL kx3_iu4_w64a_add\n",
    "#define IU4_W64_SET_KERNEL kx3_iu4_w64a_set\n",
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4_w64.gfx12.hip")
);
const W64_B_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    "#define IU4_W64B_ADD_KERNEL kx3_iu4_w64b_add\n",
    "#define IU4_W64B_SET_KERNEL kx3_iu4_w64b_set\n",
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4_w64b.gfx12.hip")
);
const CMP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <stdint.h>
extern "C" __global__ void kx3_compare_u32(
    const uint32_t* a, const uint32_t* b, unsigned long long n,
    unsigned long long* stats) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        if (a[i] != b[i]) {
            atomicAdd(stats, 1ull);
            atomicMin(stats + 1, i);
        }
    }
}
"#;

const SHIPPED_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set";
const SHIPPED_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add";
const W64_A_SET: &str = "kx3_iu4_w64a_set";
const W64_A_ADD: &str = "kx3_iu4_w64a_add";
const W64_B_SET: &str = "kx3_iu4_w64b_set";
const W64_B_ADD: &str = "kx3_iu4_w64b_add";

struct Shape {
    label: &'static str,
    rows: usize,
    k: usize,
    add: bool,
    weights: DeviceBuffer,
    xq: DeviceBuffer,
    y_ship: DeviceBuffer,
    y_a: DeviceBuffer,
    y_b: DeviceBuffer,
    output_bytes: usize,
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn alloc_zero(gpu: &Gpu, bytes: usize) -> DeviceBuffer {
    let b = gpu.hip.malloc(bytes).expect("hipMalloc");
    gpu.hip.memset(&b, 0, bytes).expect("hipMemset");
    b
}

fn make_shape(gpu: &Gpu, label: &'static str, rows: usize, k: usize, add: bool, seed: u64) -> Shape {
    assert_eq!(k % 256, 0);
    let groups = k / 256;
    let mut rng = seed;
    let mut weights = vec![0u8; rows * groups * 136];
    for row in 0..rows {
        for g in 0..groups {
            let base = (row * groups + g) * 136;
            // Two (sc,zp) fp16 header pairs.  Finite, exactly representable values.
            weights[base..base + 8].copy_from_slice(&[0x00, 0x3c, 0x00, 0x34, 0x00, 0x38, 0x00, 0xb0]);
            for byte in &mut weights[base + 8..base + 136] {
                *byte = xorshift(&mut rng) as u8;
            }
        }
    }
    let blocks = k / 128;
    let mut xq = vec![0u8; blocks * BATCH * 72];
    for kb in 0..blocks {
        for tok in 0..BATCH {
            let base = (kb * BATCH + tok) * 72;
            let d = if ((kb + tok) & 1) == 0 { 0.125f32 } else { -0.0625f32 };
            let s = ((xorshift(&mut rng) as i32) & 255) - 128;
            xq[base..base + 4].copy_from_slice(&d.to_ne_bytes());
            xq[base + 4..base + 8].copy_from_slice(&s.to_ne_bytes());
            for byte in &mut xq[base + 8..base + 72] {
                *byte = xorshift(&mut rng) as u8;
            }
        }
    }
    let weights_d = gpu.hip.malloc(weights.len()).expect("weights malloc");
    gpu.hip.memcpy_htod(&weights_d, &weights).expect("weights upload");
    let xq_d = gpu.hip.malloc(xq.len()).expect("xq malloc");
    gpu.hip.memcpy_htod(&xq_d, &xq).expect("xq upload");
    drop(weights);
    drop(xq);
    let output_bytes = BATCH * rows * 4;
    Shape {
        label,
        rows,
        k,
        add,
        weights: weights_d,
        xq: xq_d,
        y_ship: alloc_zero(gpu, output_bytes),
        y_a: alloc_zero(gpu, output_bytes),
        y_b: alloc_zero(gpu, output_bytes),
        output_bytes,
    }
}

fn launch(gpu: &Gpu, sym: &str, s: &Shape, y: &DeviceBuffer) {
    let mut args = KernargBlob::new();
    args.push_ptr(s.weights.as_ptr());
    args.push_ptr(s.xq.as_ptr());
    args.push_ptr(y.as_ptr());
    args.push_i32(s.rows as i32);
    args.push_i32(s.k as i32);
    args.push_i32(BATCH as i32);
    args.push_i32(i32::from(s.add));
    let mut bytes = args.into_vec();
    let is_b = sym.starts_with("kx3_iu4_w64b");
    gpu.launch_kernel_blob(
        sym,
        [s.rows.div_ceil(128) as u32, BATCH.div_ceil(if is_b { 256 } else { 128 }) as u32, 1],
        if is_b { BLOCK_B } else { BLOCK },
        LDS,
        &mut bytes,
    ).unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn median_us<F: FnMut()>(gpu: &Gpu, mut f: F) -> f64 {
    for _ in 0..WARMUP { f(); }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut pairs = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let begin = gpu.hip.event_create().expect("event begin");
        let end = gpu.hip.event_create().expect("event end");
        gpu.hip.event_record(&begin, gpu.active_stream.as_ref()).expect("record begin");
        f();
        gpu.hip.event_record(&end, gpu.active_stream.as_ref()).expect("record end");
        pairs.push((begin, end));
    }
    gpu.hip.event_synchronize(&pairs.last().unwrap().1).expect("timing sync");
    let mut times: Vec<f64> = pairs.iter().map(|(a, b)| {
        gpu.hip.event_elapsed_ms(a, b).expect("elapsed") as f64 * 1000.0
    }).collect();
    for (a, b) in pairs {
        gpu.hip.event_destroy(a).expect("destroy begin");
        gpu.hip.event_destroy(b).expect("destroy end");
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[SAMPLES / 2 - 1] + times[SAMPLES / 2]) * 0.5
}

fn compare(gpu: &Gpu, lhs: &DeviceBuffer, rhs: &DeviceBuffer, bytes: usize) -> (u64, u64) {
    let stats = gpu.hip.malloc(16).expect("stats malloc");
    let init = [0u64, u64::MAX];
    let init_bytes = unsafe { std::slice::from_raw_parts(init.as_ptr().cast::<u8>(), 16) };
    gpu.hip.memcpy_htod(&stats, init_bytes).expect("stats init");
    let words = bytes / 4;
    let mut args = KernargBlob::new();
    args.push_ptr(lhs.as_ptr());
    args.push_ptr(rhs.as_ptr());
    args.push_u64(words as u64);
    args.push_ptr(stats.as_ptr());
    let mut arg_bytes = args.into_vec();
    gpu.launch_kernel_blob("kx3_compare_u32", [4096, 1, 1], [256, 1, 1], 0, &mut arg_bytes)
        .expect("compare launch");
    gpu.hip.device_synchronize().expect("compare sync");
    let mut host = [0u8; 16];
    gpu.hip.memcpy_dtoh(&mut host, &stats).expect("stats download");
    let count = u64::from_ne_bytes(host[..8].try_into().unwrap());
    let first = u64::from_ne_bytes(host[8..].try_into().unwrap());
    (count, first)
}

fn mismatch_window(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, first_word: u64, bytes: usize) -> String {
    let off = (first_word as usize * 4).saturating_sub(16);
    let len = 64.min(bytes - off);
    let mut ah = vec![0u8; len];
    let mut bh = vec![0u8; len];
    gpu.hip.memcpy_dtoh_at(&mut ah, a, off).expect("lhs window");
    gpu.hip.memcpy_dtoh_at(&mut bh, b, off).expect("rhs window");
    format!("offset={off},ship={ah:02x?},candidate={bh:02x?}")
}

fn exactness(gpu: &Gpu, s: &Shape, candidate: &str, y: &DeviceBuffer) -> bool {
    gpu.hip.memset(&s.y_ship, 0, s.output_bytes).expect("zero shipped");
    gpu.hip.memset(y, 0, s.output_bytes).expect("zero candidate");
    let shipped = if s.add { SHIPPED_ADD } else { SHIPPED_SET };
    launch(gpu, shipped, s, &s.y_ship);
    launch(gpu, candidate, s, y);
    let (mismatches, first) = compare(gpu, &s.y_ship, y, s.output_bytes);
    println!("EXACT,{candidate},{},{},bytes={},u32_mismatches={},first_u32={}",
             s.label, if mismatches == 0 { "PASS" } else { "FAIL" }, s.output_bytes,
             mismatches, if first == u64::MAX { "none".into() } else { first.to_string() });
    if mismatches != 0 {
        eprintln!("MISMATCH,{candidate},{},{}", s.label,
                  mismatch_window(gpu, &s.y_ship, y, first, s.output_bytes));
    }
    mismatches == 0
}

fn time_aba(gpu: &Gpu, s: &Shape, candidate: &str, y: &DeviceBuffer) -> (f64, f64, f64) {
    let shipped = if s.add { SHIPPED_ADD } else { SHIPPED_SET };
    gpu.hip.memset(&s.y_ship, 0, s.output_bytes).expect("zero A1");
    let a1 = median_us(gpu, || launch(gpu, shipped, s, &s.y_ship));
    gpu.hip.memset(y, 0, s.output_bytes).expect("zero B");
    let b = median_us(gpu, || launch(gpu, candidate, s, y));
    gpu.hip.memset(&s.y_ship, 0, s.output_bytes).expect("zero A2");
    let a2 = median_us(gpu, || launch(gpu, shipped, s, &s.y_ship));
    let abase = 0.5 * (a1 + a2);
    println!("ABA,{candidate},{},A1_us={a1:.3},B_us={b:.3},A2_us={a2:.3},delta_pct={:.3}",
             s.label, (b / abase - 1.0) * 100.0);
    (a1, b, a2)
}

fn ensure(gpu: &mut Gpu, module: &str, src: &str, syms: &[&str], block: [u32; 3]) {
    for &sym in syms {
        gpu.ensure_kernel_public(module, src, sym)
            .unwrap_or_else(|e| panic!("JIT {module}/{sym}: {e}"));
        let occ = gpu.occupancy_max_active_blocks(sym, block, LDS).unwrap_or(-1);
        eprintln!("RESOURCE_OCC,{sym},blocks_per_cu={occ}");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201");
    ensure(&mut gpu, "kx3_iu4_shipped_w64probe", SHIPPED_SRC, &[SHIPPED_SET, SHIPPED_ADD], BLOCK);
    ensure(&mut gpu, "kx3_iu4_w64a_mod", W64_A_SRC, &[W64_A_SET, W64_A_ADD], BLOCK);
    ensure(&mut gpu, "kx3_iu4_w64b512_mod", W64_B_SRC, &[W64_B_SET, W64_B_ADD], BLOCK_B);
    ensure(&mut gpu, "kx3_compare_mod", CMP_SRC, &["kx3_compare_u32"], BLOCK);

    let shapes = [
        make_shape(&gpu, "gate_up_M4096_N34816_K5120", 34_816, 5_120, false, 0x640a),
        make_shape(&gpu, "residual_M4096_N5120_K17408", 5_120, 17_408, true, 0x640b),
    ];

    for s in &shapes {
        let sym = if s.add { W64_A_ADD } else { W64_A_SET };
        if !exactness(&gpu, s, sym, &s.y_a) {
            std::process::exit(3);
        }
    }
    let mut a_deltas = Vec::new();
    for s in &shapes {
        let sym = if s.add { W64_A_ADD } else { W64_A_SET };
        let (a1, b, a2) = time_aba(&gpu, s, sym, &s.y_a);
        a_deltas.push(b / (0.5 * (a1 + a2)) - 1.0);
    }
    if a_deltas.iter().all(|&d| d >= 0.05) {
        println!("OVERRIDE_B,w64a_at_least_5pct_slower_on_both_shapes,parent_requested_4x4_receipt");
    }
    for s in &shapes {
        let sym = if s.add { W64_B_ADD } else { W64_B_SET };
        if !exactness(&gpu, s, sym, &s.y_b) {
            std::process::exit(4);
        }
        time_aba(&gpu, s, sym, &s.y_b);
    }
}
