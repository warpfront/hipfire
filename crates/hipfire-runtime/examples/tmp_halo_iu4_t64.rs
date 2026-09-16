//! Throwaway T64 oracle: shipping A5 column kernel vs 64x64-tile t64.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, quantizes X via `ensure_int4_mmq_x`, and raw-launches shipping
//! `*_full_{set,add}_occ3_col_gfx1151` (128x128 tile, block [32,8,1],
//! LDS 30720, grid [N/128, M/128, 1]) against candidate
//! `*_full_{set,add}_t64_col_gfx1151` (64x64 tile, block [32,8,1],
//! LDS 15360, grid [N/64, M/64, 1]) on identical inputs. Asserts
//! bitwise-equal Y per output plus A/Xq immutability.
//!
//! Covers the two production arms: gate_proj set and down_proj add (nonzero
//! Y0 so accumulation is exercised).
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-t64 HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1 \
//!     ./target/release/examples/tmp_halo_iu4_t64 \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]
//!
//! `--compile-only` JITs the four symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (100 samples each, warm) with medians, TOPS and % of the
//! 107.8-TOPS peak for gate set and down add.

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
// Fresh module identity (distinct from the retired A5/LF oracles) so no stale
// cache entry can alias these symbols.
const MODULE: &str = "tmp_halo_iu4_t64";

const REF_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151";
const REF_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151";
const T64_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_t64_col_gfx1151";
const T64_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_t64_col_gfx1151";

const REF_SHARED: u32 = (128 * 18 + 128 * 42) * 4; // 30720, shipping planes
const T64_SHARED: u32 = (64 * 18 + 64 * 42) * 4; // 15360, t64 planes
const REF_BLOCK: [u32; 3] = [32, 8, 1];
const T64_BLOCK: [u32; 3] = [32, 8, 1];
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";
const PEAK_TOPS: f64 = 107.8;

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

#[allow(dead_code)]
fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

/// f32 -> f16 bits (round-to-nearest-even mantissa truncation is enough for
/// the benign scales used here; NaN/Inf pass through as Inf/NaN).
#[allow(dead_code)]
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ff_ff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x0200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    if e16 <= 0 {
        return s; // flush denormals; synthetic scales never land here
    }
    s | ((e16 as u16) << 10) | ((m >> 13) as u16)
}

/// Synthetic MQ4G256V2 weights: 136 B/group, dual fp16 headers, full 0..15
/// nibble codes, distinct h0/h1, non-power-of-two scales.
#[allow(dead_code)]
fn pack_mq4g256v2_synth(m: usize, k: usize, salt: u32) -> Vec<u8> {
    const GROUP: usize = 256;
    const HALF: usize = 128;
    const GB: usize = 136;
    assert_eq!(k % GROUP, 0, "K={k} not multiple of 256");
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GB];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GB;
            let sc0 = f16_bits(0.05 + (prng_u32((r * gpr + g) as u64, salt) % 20) as f32 * 0.001);
            let zp0 = f16_bits(-0.35);
            let sc1 =
                f16_bits(0.07 + (prng_u32((r * gpr + g) as u64, salt ^ 1) % 20) as f32 * 0.001);
            let zp1 = f16_bits(-0.45);
            blob[dst..dst + 2].copy_from_slice(&sc0.to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&zp0.to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&sc1.to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&zp1.to_le_bytes());
            for i in 0..HALF {
                let lo = (prng_u32((r * 1024 + g * HALF + i) as u64, salt) % 16) as u8;
                let hi = (prng_u32((r * 1024 + g * HALF + i) as u64, salt ^ 0x55) % 16) as u8;
                blob[dst + 8 + i] = lo | (hi << 4);
            }
        }
    }
    blob
}

fn cache_dir(arch: &str) -> PathBuf {
    let root = std::env::var_os("HIPFIRE_KERNEL_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into());
        PathBuf::from(home).join(".hipfire_kernels")
    });
    root.join(arch)
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
    assert_eq!(bytes.len(), expect, "{name}: {m}x{k} byte len");
    (m, k, bytes)
}

fn bitwise_eq(a: &[f32], b: &[f32]) -> (bool, usize, Option<(usize, u32, u32)>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let (xb, yb) = (x.to_bits(), y.to_bits());
        if xb != yb {
            mism += 1;
            if first.is_none() {
                first = Some((i, xb, yb));
            }
        }
    }
    (mism == 0, mism, first)
}

fn dtoh_raw(gpu: &Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
    let view = unsafe { DeviceBuffer::from_raw(ptr, len) };
    let mut host = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut host, &view)
        .expect("memcpy_dtoh scratch snapshot");
    // Borrowed view: no free (DeviceBuffer has no Drop; nothing to do).
    let _ = view.is_borrowed();
    host
}

fn launch_once(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    gpu.launch_kernel_blob(func, grid, block, shared, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

fn time_us(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
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
    gpu.launch_kernel_blob(func, grid, block, shared, &mut blob)
        .unwrap_or_else(|e| panic!("timed launch {func}: {e}"));
    gpu.hip.event_record(&stop, stream).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms as f64 * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    label: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    ref_sym: &str,
    t64_sym: &str,
    salt: u32,
    time: bool,
) -> bool {
    assert_eq!(m % 64, 0, "{label}: M={m} not 64-full-tile");
    assert_eq!(n % 64, 0, "{label}: N={n} not 64-full-tile");
    assert_eq!(k % 256, 0, "{label}: K={k} not multiple of 256");
    let ref_grid = [(n / 128) as u32, (m / 128) as u32, 1];
    let t64_grid = [(n / 64) as u32, (m / 64) as u32, 1];

    // X through the real quantizer (N rows x K cols, N=batch here).
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            let scale = 0.75 + (col % 5) as f32 * 0.15;
            prng_f32(i, salt) * scale
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr = gpu
        .ensure_int4_mmq_x(&d_x, n, k)
        .expect("ensure_int4_mmq_x") as *mut std::ffi::c_void;
    let xq_len = (k / 128) * n * 72;
    let xq_before = dtoh_raw(gpu, xq_ptr, xq_len);

    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;

    // Y0 seed: set path zeros; add path NONZERO so accumulation is exercised.
    let y_elems = n * m;
    let y0: Vec<f32> = if add {
        (0..y_elems)
            .map(|i| prng_f32(i, 0xADD0_BEEF ^ salt) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0f32; y_elems]
    };
    let d_y_ref = gpu.upload_f32(&y0, &[n, m]).expect("upload Y ref");
    let d_y_t64 = gpu.upload_f32(&y0, &[n, m]).expect("upload Y t64");

    for sym in [ref_sym, t64_sym] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
    }

    let add_i = i32::from(add);
    let (mi, ki, ni) = (m as i32, k as i32, n as i32);
    launch_once(
        gpu,
        ref_sym,
        ref_grid,
        REF_BLOCK,
        REF_SHARED,
        a_ptr,
        xq_ptr as *const _,
        d_y_ref.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    launch_once(
        gpu,
        t64_sym,
        t64_grid,
        T64_BLOCK,
        T64_SHARED,
        a_ptr,
        xq_ptr as *const _,
        d_y_t64.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    gpu.hip.device_synchronize().expect("sync parity");

    let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
    let y_t64 = gpu.download_f32(&d_y_t64).expect("dl t64");
    let finite = y_ref.iter().all(|v| v.is_finite()) && y_t64.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq(&y_ref, &y_t64);
    let first_rc = first.map(|(idx, xb, yb)| {
        let row = idx % m;
        let col = idx / m;
        format!("(row={row},col={col}) ref=0x{xb:08x} t64=0x{yb:08x}")
    });
    let moved = if add {
        y_ref
            .iter()
            .zip(y0.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    } else {
        y_ref.iter().any(|v| v.to_bits() != 0)
    };

    // A/Xq immutability over both launches.
    let mut a_back = vec![0u8; a_bytes.len()];
    gpu.hip
        .memcpy_dtoh(&mut a_back, &d_a.buf)
        .expect("dl A back");
    let a_same = a_back == a_bytes;
    let xq_after = dtoh_raw(gpu, xq_ptr, xq_len);
    let xq_same = xq_after == xq_before;

    let ok = eq && finite && moved && a_same && xq_same;
    eprintln!(
        "  [{label} M={m} K={k} N={n} add={add}] finite={finite} moved={moved} \
         bitwise_eq={eq} mism={mism} first={first_rc:?} a_immutable={a_same} xq_immutable={xq_same} [{}]",
        if ok { "PASS" } else { "FAIL" }
    );

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        let y0_bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let restore = |gpu: &mut Gpu, y_ptr_buf: &DeviceBuffer| {
            gpu.hip
                .memcpy_htod(y_ptr_buf, &y0_bytes)
                .expect("restore Y0");
        };
        for _ in 0..WARM {
            if add {
                restore(gpu, &d_y_ref.buf);
            }
            launch_once(
                gpu, ref_sym, ref_grid, REF_BLOCK, REF_SHARED, a_ptr, xq_ptr as *const _,
                d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
            if add {
                restore(gpu, &d_y_t64.buf);
            }
            launch_once(
                gpu, t64_sym, t64_grid, T64_BLOCK, T64_SHARED, a_ptr, xq_ptr as *const _,
                d_y_t64.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
        }
        gpu.hip.device_synchronize().expect("sync warm");

        let mut ref_us = Vec::with_capacity(SAMPLES);
        let mut t64_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            // Interleave ref/t64 to share thermal state; add Y restored
            // outside each timed interval.
            if i % 2 == 0 {
                if add {
                    restore(gpu, &d_y_ref.buf);
                }
                ref_us.push(time_us(
                    gpu, ref_sym, ref_grid, REF_BLOCK, REF_SHARED, a_ptr, xq_ptr as *const _,
                    d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_t64.buf);
                }
                t64_us.push(time_us(
                    gpu, t64_sym, t64_grid, T64_BLOCK, T64_SHARED, a_ptr, xq_ptr as *const _,
                    d_y_t64.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            } else {
                if add {
                    restore(gpu, &d_y_t64.buf);
                }
                t64_us.push(time_us(
                    gpu, t64_sym, t64_grid, T64_BLOCK, T64_SHARED, a_ptr, xq_ptr as *const _,
                    d_y_t64.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_ref.buf);
                }
                ref_us.push(time_us(
                    gpu, ref_sym, ref_grid, REF_BLOCK, REF_SHARED, a_ptr, xq_ptr as *const _,
                    d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            }
        }
        let rmed = median(&mut ref_us);
        let tmed = median(&mut t64_us);
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let (rtops, ttops) = (ops / rmed / 1e6, ops / tmed / 1e6);
        let (rpct, tpct) = (rtops / PEAK_TOPS * 100.0, ttops / PEAK_TOPS * 100.0);
        let ratio = tmed / rmed;
        eprintln!(
            "  TIME {label}: ref_med={rmed:.3} us ({rtops:.2} TOPS, {rpct:.1}% peak)  \
             t64_med={tmed:.3} us ({ttops:.2} TOPS, {tpct:.1}% peak)  ratio={ratio:.4} (n={SAMPLES} interleaved)"
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y_ref);
    let _ = gpu.free_tensor(d_y_t64);
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: tmp_halo_iu4_t64 <model.mq4-xt> [--compile-only]");
            std::process::exit(2);
        });
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    eprintln!("tmp_halo_iu4_t64 on {}  model={model}  TIME={time}", gpu.arch);
    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: t64 symbols are #if __gfx1151__; arch={} may fail JIT of t64 entries",
            gpu.arch
        );
    }

    // Compile-only: JIT all four entries, print the cache path, return.
    for sym in [REF_SET, REF_ADD, T64_SET, T64_ADD] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("JIT OK: {sym}");
    }
    let dir = cache_dir(&gpu.arch);
    eprintln!("CACHE_PATH {}", dir.display());
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let s = ent.file_name().to_string_lossy().into_owned();
            if s.starts_with(MODULE) {
                eprintln!("  cached: {s}");
            }
        }
    }
    if compile_only {
        return;
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
    // The two production arms, both timed under TIME=1.
    ok &= run_case(&mut gpu, "gate_proj/set", &gate, gate_m, gate_k, 512, false, REF_SET, T64_SET, 0xA5A5_00C0, time);
    ok &= run_case(&mut gpu, "down_proj/add", &down, down_m, down_k, 512, true, REF_ADD, T64_ADD, 0xA5A5_00C3, time);

    if ok {
        eprintln!("T64 PASS: occ3_col vs t64 bitwise-equal on all cases");
    } else {
        eprintln!("T64 FAIL");
        std::process::exit(1);
    }
}
