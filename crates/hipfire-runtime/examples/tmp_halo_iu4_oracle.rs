//! Throwaway A5 oracle: IU4 full-tile baseline vs column-adjacent grid.
//!
//! Loads real qt=44 gate_proj (set) and down_proj (add, nonzero Y0) from an
//! MQ4-XT HFQ, quantizes X via `ensure_int4_mmq_x`, and raw-launches
//! `*_full_{set,add}_occ3` against `*_full_{set,add}_occ3_col_gfx1151` with
//! the swapped grid. Asserts bitwise-equal Y.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-a5 HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     cargo run --release -p hipfire-runtime --example tmp_halo_iu4_oracle --features lab -- \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
//!
//! TIME=1 prints interleaved GPU-event us/launch (>=100 samples each, warm).

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
const MODULE: &str = "tmp_halo_iu4_oracle";

const BASE_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3";
const BASE_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3";
const COL_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151";
const COL_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151";

const SHARED: u32 = (128 * 18 + 128 * 42) * 4;
const BLOCK: [u32; 3] = [32, 8, 1];
const N: usize = 512;
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";

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

fn mk_blob(
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(xq);
    b.push_ptr(y);
    b.push_i32(m);
    b.push_i32(k);
    b.push_i32(n);
    b.push_i32(add);
    b.into_vec()
}

fn load_mq4v2(hfq: &HfqFile, name: &str) -> (usize, usize, Vec<u8>) {
    let (info, bytes) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    assert_eq!(
        info.quant_type, QT_MQ4V2,
        "{name}: quant_type={} want {QT_MQ4V2}",
        info.quant_type
    );
    assert_eq!(info.shape.len(), 2, "{name}: expected 2D shape");
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    assert_eq!(m % 128, 0, "{name}: M={m} not full-tile");
    assert_eq!(k % 256, 0, "{name}: K={k} not multiple of 256");
    let gpr = k / 256;
    let expect = m * gpr * 136;
    assert_eq!(
        bytes.len(),
        expect,
        "{name}: bytes={} expect {expect} (M={m} K={k})",
        bytes.len()
    );
    (m, k, bytes)
}

fn bitwise_eq(a: &[f32], b: &[f32]) -> (bool, usize, Option<(usize, u32, u32)>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if first.is_none() {
                first = Some((i, x.to_bits(), y.to_bits()));
            }
            mism += 1;
        }
    }
    (mism == 0, mism, first)
}

fn launch_once(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    gpu.launch_kernel_blob(func, grid, BLOCK, SHARED, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

fn time_us(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> f64 {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    let stream = gpu.active_stream.as_ref();
    gpu.hip.event_record(&start, stream).expect("record start");
    gpu.launch_kernel_blob(func, grid, BLOCK, SHARED, &mut blob)
        .unwrap_or_else(|e| panic!("timed launch {func}: {e}"));
    gpu.hip.event_record(&stop, stream).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    ms as f64 * 1e3
}

fn run_arm(
    gpu: &mut Gpu,
    label: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    add: bool,
    base_sym: &str,
    col_sym: &str,
    time: bool,
) -> bool {
    assert_eq!(N % 128, 0);
    let row_tiles = m / 128;
    let col_tiles = N / 128;
    let grid_base = [row_tiles as u32, col_tiles as u32, 1];
    let grid_col = [col_tiles as u32, row_tiles as u32, 1];

    let x: Vec<f32> = (0..N * k)
        .map(|i| {
            let col = i / k;
            let scale = 0.75 + (col % 5) as f32 * 0.15;
            prng_f32(i, 0xA5A5_00C0) * scale
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[N, k]).expect("upload X");
    let xq = gpu
        .ensure_int4_mmq_x(&d_x, N, k)
        .expect("ensure_int4_mmq_x") as *const std::ffi::c_void;

    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;

    // Y0 seed: set path zeros; add path NONZERO so both kernels accumulate.
    let y_elems = N * m;
    let y0: Vec<f32> = if add {
        (0..y_elems)
            .map(|i| prng_f32(i, 0xADD0_BEEF) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0f32; y_elems]
    };
    let d_y_base = gpu.upload_f32(&y0, &[N, m]).expect("upload Y base");
    let d_y_col = gpu.upload_f32(&y0, &[N, m]).expect("upload Y col");

    for sym in [base_sym, col_sym] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
    }

    let add_i = i32::from(add);
    launch_once(
        gpu,
        base_sym,
        grid_base,
        a_ptr,
        xq,
        d_y_base.buf.as_ptr() as *const _,
        m as i32,
        k as i32,
        N as i32,
        add_i,
    );
    launch_once(
        gpu,
        col_sym,
        grid_col,
        a_ptr,
        xq,
        d_y_col.buf.as_ptr() as *const _,
        m as i32,
        k as i32,
        N as i32,
        add_i,
    );
    gpu.hip.device_synchronize().expect("sync parity");

    let y_base = gpu.download_f32(&d_y_base).expect("dl base");
    let y_col = gpu.download_f32(&d_y_col).expect("dl col");
    let finite = y_base.iter().all(|v| v.is_finite()) && y_col.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq(&y_base, &y_col);
    // Nonzero check on add: Y must differ from the seed somewhere.
    let moved = if add {
        y_base
            .iter()
            .zip(y0.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    } else {
        y_base.iter().any(|v| v.to_bits() != 0)
    };
    let ok = eq && finite && moved;
    eprintln!(
        "  [{label} M={m} K={k} N={N} add={add}] finite={finite} moved={moved} bitwise_eq={eq} mism={mism} first={first:?} [{}]",
        if ok { "PASS" } else { "FAIL" }
    );

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        // Warm both symbols on their grids.
        for _ in 0..WARM {
            launch_once(
                gpu,
                base_sym,
                grid_base,
                a_ptr,
                xq,
                d_y_base.buf.as_ptr() as *const _,
                m as i32,
                k as i32,
                N as i32,
                add_i,
            );
            launch_once(
                gpu,
                col_sym,
                grid_col,
                a_ptr,
                xq,
                d_y_col.buf.as_ptr() as *const _,
                m as i32,
                k as i32,
                N as i32,
                add_i,
            );
        }
        gpu.hip.device_synchronize().expect("sync warm");

        let mut base_us = Vec::with_capacity(SAMPLES);
        let mut col_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            // Interleave base/col to share thermal state.
            if i % 2 == 0 {
                base_us.push(time_us(
                    gpu,
                    base_sym,
                    grid_base,
                    a_ptr,
                    xq,
                    d_y_base.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    N as i32,
                    add_i,
                ));
                col_us.push(time_us(
                    gpu,
                    col_sym,
                    grid_col,
                    a_ptr,
                    xq,
                    d_y_col.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    N as i32,
                    add_i,
                ));
            } else {
                col_us.push(time_us(
                    gpu,
                    col_sym,
                    grid_col,
                    a_ptr,
                    xq,
                    d_y_col.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    N as i32,
                    add_i,
                ));
                base_us.push(time_us(
                    gpu,
                    base_sym,
                    grid_base,
                    a_ptr,
                    xq,
                    d_y_base.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    N as i32,
                    add_i,
                ));
            }
        }
        let med = |v: &mut [f64]| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            }
        };
        let bmed = med(&mut base_us);
        let cmed = med(&mut col_us);
        let delta = (cmed - bmed) / bmed * 100.0;
        eprintln!(
            "  TIME {label}: base_med={bmed:.3} us  col_med={cmed:.3} us  delta={delta:+.2}%  (n={SAMPLES} interleaved)"
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y_base);
    let _ = gpu.free_tensor(d_y_col);
    ok
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| {
        "/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt".to_string()
    });
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    eprintln!("tmp_halo_iu4_oracle on {}  model={model}  TIME={time}", gpu.arch);
    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: col symbols are #if __gfx1151__; arch={} may fail JIT of _col entries",
            gpu.arch
        );
    }

    let t0 = Instant::now();
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let (gate_m, gate_k, gate) = load_mq4v2(&hfq, GATE_NAME);
    let (down_m, down_k, down) = load_mq4v2(&hfq, DOWN_NAME);
    eprintln!(
        "loaded gate_proj {gate_m}x{gate_k} ({} B)  down_proj {down_m}x{down_k} ({} B) in {:.2}s",
        gate.len(),
        down.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!((gate_m, gate_k), (17408, 5120), "gate shape");
    assert_eq!((down_m, down_k), (5120, 17408), "down shape");
    // Drop mmap pressure before large GPU allocs on UMA.
    drop(hfq);

    let mut ok = true;
    ok &= run_arm(
        &mut gpu,
        "gate_proj/set",
        &gate,
        gate_m,
        gate_k,
        false,
        BASE_SET,
        COL_SET,
        time,
    );
    ok &= run_arm(
        &mut gpu,
        "down_proj/add",
        &down,
        down_m,
        down_k,
        true,
        BASE_ADD,
        COL_ADD,
        time,
    );

    if ok {
        eprintln!("A5 PASS: baseline vs _col bitwise-equal on gate set + down add");
    } else {
        eprintln!("A5 FAIL");
        std::process::exit(1);
    }
}
