//! Throwaway group-major (GM) oracle: row-major shipping IU4 vs GM twins.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, builds the group-major copy of A on the host (permute 136-B
//! groups: dst[(kb*M+row)*136] = src[(row*gpr+kb)*136]), quantizes X via
//! `ensure_int4_mmq_x` (N=512), and raw-launches shipping `*_occ3_col_gfx1151`
//! (block [32,8,1]) / `*_lf16_col_gfx1151` (block [32,16,1]) against the `_gm`
//! twins with identical LDS (30720) and grid ([N/128, M/128, 1]).
//! Asserts bitwise-equal Y per output plus A/Xq immutability.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-gm HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1 \
//!     ./target/release/examples/tmp_halo_iu4_gm \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]
//!
//! `--compile-only` JITs the six symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (100 samples each, warm) with medians, TOPS and % of the
//! 107.8-TOPS peak for gate set (occ3 + lf16) and down add (occ3). The host
//! group-major relayout time is printed too (informative only; a production
//! route would relayout once offline).

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
// Fresh module identity so no stale cache entry can alias these symbols.
const MODULE: &str = "tmp_halo_iu4_gm";

const REF_SET_OCC3: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151";
const GM_SET_OCC3: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_gm_col_gfx1151";
const REF_SET_LF16: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151";
const GM_SET_LF16: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gm_col_gfx1151";
const REF_ADD_OCC3: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151";
const GM_ADD_OCC3: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_gm_col_gfx1151";

const SHARED: u32 = (128 * 18 + 128 * 42) * 4; // 30720, same planes as shipping
const OCC3_BLOCK: [u32; 3] = [32, 8, 1];
const LF16_BLOCK: [u32; 3] = [32, 16, 1];
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

/// Row-major (row*gpr+kb) -> group-major (kb*M+row) permutation of 136-B
/// groups. Returns (permuted, seconds).
fn group_major_copy(a: &[u8], m: usize, k: usize) -> (Vec<u8>, f64) {
    const GB: usize = 136;
    let gpr = k / 256;
    assert_eq!(a.len(), m * gpr * GB);
    let t0 = Instant::now();
    let mut dst = vec![0u8; a.len()];
    for row in 0..m {
        for kb in 0..gpr {
            let s = (row * gpr + kb) * GB;
            let d = (kb * m + row) * GB;
            dst[d..d + GB].copy_from_slice(&a[s..s + GB]);
        }
    }
    (dst, t0.elapsed().as_secs_f64())
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

#[allow(clippy::too_many_arguments)]
fn launch_once(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    gpu.launch_kernel_blob(func, grid, block, SHARED, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

#[allow(clippy::too_many_arguments)]
fn time_us(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
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
    gpu.launch_kernel_blob(func, grid, block, SHARED, &mut blob)
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
    a_rm: &[u8],
    a_gm: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    ref_sym: &str,
    gm_sym: &str,
    ref_block: [u32; 3],
    gm_block: [u32; 3],
    salt: u32,
    time: bool,
) -> bool {
    assert_eq!(m % 128, 0, "{label}: M={m} not full-tile");
    assert_eq!(n % 128, 0, "{label}: N={n} not full-tile");
    assert_eq!(k % 256, 0, "{label}: K={k} not multiple of 256");
    assert_eq!(a_rm.len(), a_gm.len());
    let grid = [(n / 128) as u32, (m / 128) as u32, 1];

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

    let d_a_rm = gpu.upload_raw(a_rm, &[a_rm.len()]).expect("upload A rm");
    let a_rm_ptr = d_a_rm.buf.as_ptr() as *const std::ffi::c_void;
    let d_a_gm = gpu.upload_raw(a_gm, &[a_gm.len()]).expect("upload A gm");
    let a_gm_ptr = d_a_gm.buf.as_ptr() as *const std::ffi::c_void;

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
    let d_y_gm = gpu.upload_f32(&y0, &[n, m]).expect("upload Y gm");

    for sym in [ref_sym, gm_sym] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
    }

    let add_i = i32::from(add);
    let (mi, ki, ni) = (m as i32, k as i32, n as i32);
    launch_once(
        gpu,
        ref_sym,
        grid,
        ref_block,
        a_rm_ptr,
        xq_ptr as *const _,
        d_y_ref.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    launch_once(
        gpu,
        gm_sym,
        grid,
        gm_block,
        a_gm_ptr,
        xq_ptr as *const _,
        d_y_gm.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    gpu.hip.device_synchronize().expect("sync parity");

    let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
    let y_gm = gpu.download_f32(&d_y_gm).expect("dl gm");
    let finite = y_ref.iter().all(|v| v.is_finite()) && y_gm.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq(&y_ref, &y_gm);
    let first_rc = first.map(|(idx, xb, yb)| {
        let row = idx % m;
        let col = idx / m;
        format!("(row={row},col={col}) ref=0x{xb:08x} gm=0x{yb:08x}")
    });
    let moved = if add {
        y_ref
            .iter()
            .zip(y0.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    } else {
        y_ref.iter().any(|v| v.to_bits() != 0)
    };

    // A (both copies) / Xq immutability over both launches.
    let mut rm_back = vec![0u8; a_rm.len()];
    gpu.hip
        .memcpy_dtoh(&mut rm_back, &d_a_rm.buf)
        .expect("dl A rm back");
    let rm_same = rm_back == a_rm;
    let mut gm_back = vec![0u8; a_gm.len()];
    gpu.hip
        .memcpy_dtoh(&mut gm_back, &d_a_gm.buf)
        .expect("dl A gm back");
    let gm_same = gm_back == a_gm;
    let xq_after = dtoh_raw(gpu, xq_ptr, xq_len);
    let xq_same = xq_after == xq_before;

    let ok = eq && finite && moved && rm_same && gm_same && xq_same;
    eprintln!(
        "  [{label} M={m} K={k} N={n} add={add}] finite={finite} moved={moved} \
         bitwise_eq={eq} mism={mism} first={first_rc:?} rm_immutable={rm_same} gm_immutable={gm_same} xq_immutable={xq_same} [{}]",
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
                gpu, ref_sym, grid, ref_block, a_rm_ptr, xq_ptr as *const _,
                d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
            if add {
                restore(gpu, &d_y_gm.buf);
            }
            launch_once(
                gpu, gm_sym, grid, gm_block, a_gm_ptr, xq_ptr as *const _,
                d_y_gm.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
        }
        gpu.hip.device_synchronize().expect("sync warm");

        let mut ref_us = Vec::with_capacity(SAMPLES);
        let mut gm_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            // Interleave ref/gm to share thermal state; add Y restored
            // outside each timed interval.
            if i % 2 == 0 {
                if add {
                    restore(gpu, &d_y_ref.buf);
                }
                ref_us.push(time_us(
                    gpu, ref_sym, grid, ref_block, a_rm_ptr, xq_ptr as *const _,
                    d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_gm.buf);
                }
                gm_us.push(time_us(
                    gpu, gm_sym, grid, gm_block, a_gm_ptr, xq_ptr as *const _,
                    d_y_gm.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            } else {
                if add {
                    restore(gpu, &d_y_gm.buf);
                }
                gm_us.push(time_us(
                    gpu, gm_sym, grid, gm_block, a_gm_ptr, xq_ptr as *const _,
                    d_y_gm.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_ref.buf);
                }
                ref_us.push(time_us(
                    gpu, ref_sym, grid, ref_block, a_rm_ptr, xq_ptr as *const _,
                    d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            }
        }
        let rmed = median(&mut ref_us);
        let gmed = median(&mut gm_us);
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let (rtops, gtops) = (ops / rmed / 1e6, ops / gmed / 1e6);
        let (rpct, gpct) = (rtops / PEAK_TOPS * 100.0, gtops / PEAK_TOPS * 100.0);
        let ratio = gmed / rmed;
        eprintln!(
            "  TIME {label}: ref_med={rmed:.3} us ({rtops:.2} TOPS, {rpct:.1}% peak)  \
             gm_med={gmed:.3} us ({gtops:.2} TOPS, {gpct:.1}% peak)  ratio={ratio:.4} (n={SAMPLES} interleaved)"
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a_rm);
    let _ = gpu.free_tensor(d_a_gm);
    let _ = gpu.free_tensor(d_y_ref);
    let _ = gpu.free_tensor(d_y_gm);
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
            eprintln!("usage: tmp_halo_iu4_gm <model.mq4-xt> [--compile-only]");
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
    eprintln!("tmp_halo_iu4_gm on {}  model={model}  TIME={time}", gpu.arch);
    // IU4 twins cover the gfx11 family (occ3 pair also gfx1100; lf16 pair is
    // gfx1151-only, like its shipping counterpart). gfx1201 has no IU4 path:
    // the .gfx11.hip file builds stubs there and gfx12 prefill routes to the
    // FP8/F16 gfx12 WMMA family instead (see yield report) — skip cleanly.
    if gpu.arch.starts_with("gfx12") {
        eprintln!(
            "SKIP: IU4 MMQ is RDNA3-only (arch={}); gfx12 prefill uses \
             gemm_mq4g256v2_residual_wmma_fp8_gfx12_* (opt-in) or \
             gemm_mq4g256v2_residual_wmma_gfx12[_bt*] instead — no gm twin \
             implemented for those files",
            gpu.arch
        );
        return;
    }
    let use_lf16 = gpu.arch == "gfx1151";
    if !use_lf16 {
        eprintln!(
            "NOTE: arch={} runs the occ3 pairs only (lf16 shipping + gm entries are gfx1151-only)",
            gpu.arch
        );
    }

    // Compile-only: JIT the covered entries, print the cache path, return.
    let mut syms: Vec<&str> = vec![REF_SET_OCC3, GM_SET_OCC3, REF_ADD_OCC3, GM_ADD_OCC3];
    if use_lf16 {
        syms.push(REF_SET_LF16);
        syms.push(GM_SET_LF16);
    }
    for sym in syms {
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

    let (gate_gm, gate_re_s) = group_major_copy(&gate, gate_m, gate_k);
    eprintln!("relayout gate_proj row-major -> group-major: {gate_re_s:.3}s host (informative only)");
    let (down_gm, down_re_s) = group_major_copy(&down, down_m, down_k);
    eprintln!("relayout down_proj row-major -> group-major: {down_re_s:.3}s host (informative only)");

    let mut ok = true;
    ok &= run_case(
        &mut gpu,
        "gate_proj/set/occ3",
        &gate,
        &gate_gm,
        gate_m,
        gate_k,
        512,
        false,
        REF_SET_OCC3,
        GM_SET_OCC3,
        OCC3_BLOCK,
        OCC3_BLOCK,
        0x6A6D_0000,
        time,
    );
    if use_lf16 {
        ok &= run_case(
            &mut gpu,
            "gate_proj/set/lf16",
            &gate,
            &gate_gm,
            gate_m,
            gate_k,
            512,
            false,
            REF_SET_LF16,
            GM_SET_LF16,
            LF16_BLOCK,
            LF16_BLOCK,
            0x6A6D_0001,
            time,
        );
    }
    ok &= run_case(
        &mut gpu,
        "down_proj/add/occ3",
        &down,
        &down_gm,
        down_m,
        down_k,
        512,
        true,
        REF_ADD_OCC3,
        GM_ADD_OCC3,
        OCC3_BLOCK,
        OCC3_BLOCK,
        0x6A6D_0002,
        time,
    );

    if ok {
        eprintln!("GM PASS: row-major shipping vs group-major bitwise-equal on all cases");
    } else {
        eprintln!("GM FAIL");
        std::process::exit(1);
    }
}
