//! Throwaway Halo experiment: resident-int8 (pre-unpacked) twin vs the
//! shipping IU4 W4A4 prefill GEMM on gfx1151.
//!
//! Loads real qt=44 gate_proj (17408x5120) from an MQ4-XT HFQ, builds a
//! RESIDENT int8 copy on the host (u8 [M][K] row-major codes 0..15 + a small
//! half2 [M][K/128] (scale, zero) side array), quantizes X via
//! `ensure_int4_mmq_x` and ALSO builds an int8 activation copy on the host
//! (s8 [N][K] + per-128-block int2 (d bits, s) side array) from the same
//! codes, then raw-launches the example-local `gemm_w8a8_resident_gfx1151`
//! (+ `_pf` variant) against shipping
//! `gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151` on identical
//! inputs. Asserts bitwise-equal Y (same integer sums, same IU4_FOLD_RN)
//! plus A/X immutability, and prints resident bytes vs MQ4 bytes.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-w8a8 HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     TIME=1 ./target/release/examples/tmp_halo_w8a8 \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]
//!
//! `--compile-only` JITs the six symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (100 samples each arm, warm) with medians, TOPS and % of the
//! 107.8-TOPS peak for gate set (ref + w8 + pf).

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

const W8A8_SRC: &str = include_str!("tmp_halo_w8a8.hip");
const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
// Fresh module identities so no stale cache entry can alias these symbols.
const MODULE: &str = "tmp_halo_w8a8";
const REF_MODULE: &str = "tmp_halo_w8a8_ref";

const REF_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151";
const REF_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151";
const W8_SET: &str = "gemm_w8a8_resident_gfx1151";
const W8_ADD: &str = "gemm_w8a8_resident_add_gfx1151";
const W8PF_SET: &str = "gemm_w8a8_resident_pf_gfx1151";
const W8PF_ADD: &str = "gemm_w8a8_resident_pf_add_gfx1151";

const W8_SHARED: u32 = (128 * 32 + 128 * 32) * 4; // 32768: A 16K + X 16K
const REF_SHARED: u32 = (128 * 18 + 128 * 42) * 4; // 30720, shipping lf16
const W8_BLOCK: [u32; 3] = [32, 8, 1];
const REF_BLOCK: [u32; 3] = [32, 16, 1];
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const PEAK_TOPS: f64 = 107.8;
const N_TOK: usize = 512;

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

fn mk_blob_ref(
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

#[allow(clippy::too_many_arguments)]
fn mk_blob_w8(
    a: *const std::ffi::c_void,
    ah: *const std::ffi::c_void,
    x: *const std::ffi::c_void,
    xh: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(ah);
    b.push_ptr(x);
    b.push_ptr(xh);
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

/// Unpack MQ4G256V2 groups to resident int8: u8 codes [M][K] row-major plus
/// the raw f16 (sc, zp) header pairs as bytes [M][K/128][4].
///
/// Group layout (136 B): bytes 0..3 = half-0 header (sc f16 lo, zp f16 hi),
/// 4..7 = half-1 header, 8..71 = half-0 nibbles (K 0..127, even K low),
/// 72..135 = half-1 nibbles.
fn unpack_resident(a: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>, f64) {
    let gpr = k / 256;
    let b128 = k / 128;
    let mut codes = vec![0u8; m * k];
    let mut hdr = vec![0u8; m * b128 * 4];
    let t0 = Instant::now();
    for r in 0..m {
        for kb in 0..gpr {
            let gs = (r * gpr + kb) * 136;
            for h in 0..2 {
                let hd = (r * b128 + 2 * kb + h) * 4;
                hdr[hd..hd + 4].copy_from_slice(&a[gs + h * 4..gs + h * 4 + 4]);
                for i in 0..128 {
                    let byte = a[gs + 8 + h * 64 + i / 2];
                    codes[r * k + kb * 256 + h * 128 + i] = (byte >> (4 * (i & 1))) & 15;
                }
            }
        }
    }
    (codes, hdr, t0.elapsed().as_secs_f64())
}

/// Build the int8 activation copy from a downloaded block_i4_128 buffer:
/// s8 values [N][K] row-major (two's-complement bit patterns) plus int2
/// headers [K/128][N] (x = d bits, y = s, verbatim).
fn build_x_i8(xq: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>, f64) {
    let b128 = k / 128;
    assert_eq!(xq.len(), b128 * n * 72);
    let mut xs = vec![0u8; n * k];
    let mut xh = vec![0u8; b128 * n * 8];
    let t0 = Instant::now();
    for b in 0..b128 {
        for nn in 0..n {
            let blk = (b * n + nn) * 72;
            xh[(b * n + nn) * 8..(b * n + nn) * 8 + 8].copy_from_slice(&xq[blk..blk + 8]);
            for i in 0..128 {
                let nib = (xq[blk + 8 + i / 2] >> (4 * (i & 1))) & 15;
                let v: i8 = if nib < 8 { nib as i8 } else { nib as i8 - 16 };
                xs[nn * k + b * 128 + i] = v as u8;
            }
        }
    }
    (xs, xh, t0.elapsed().as_secs_f64())
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

fn launch_blob(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    blob: &mut [u8],
) {
    gpu.launch_kernel_blob(func, grid, block, shared, blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

fn time_us(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    blob: &mut [u8],
) -> f64 {
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    let stream = gpu.active_stream.as_ref();
    gpu.hip.event_record(&start, stream).expect("record start");
    gpu.launch_kernel_blob(func, grid, block, shared, blob)
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: tmp_halo_w8a8 <model.mq4-xt> [--compile-only]");
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
    eprintln!("tmp_halo_w8a8 on {}  model={model}  TIME={time}", gpu.arch);
    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: w8a8 entries are #if __gfx1151__; arch={} may fail JIT",
            gpu.arch
        );
    }

    // JIT everything up front.
    for sym in [W8_SET, W8_ADD, W8PF_SET, W8PF_ADD] {
        gpu.ensure_kernel_public(MODULE, W8A8_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("JIT OK: {sym}");
    }
    for sym in [REF_SET, REF_ADD] {
        gpu.ensure_kernel_public(REF_MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("JIT OK: {sym}");
    }
    let dir = cache_dir(&gpu.arch);
    eprintln!("CACHE_PATH {}", dir.display());
    if compile_only {
        return;
    }

    let t0 = Instant::now();
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let (m, k, gate) = load_mq4v2(&hfq, GATE_NAME);
    eprintln!(
        "loaded gate_proj {m}x{k} ({} B) in {:.2}s",
        gate.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!((m, k), (17408, 5120), "gate shape");
    let n = N_TOK;
    // Drop mmap pressure before large GPU allocs on UMA.
    drop(hfq);

    // Resident int8 weight copy on the host.
    let (codes, hdr, unpack_s) = unpack_resident(&gate, m, k);
    let res_total = codes.len() + hdr.len();
    eprintln!(
        "resident u8 copy {} B + headers {} B = {res_total} B ({:.3} bpw) vs MQ4 {} B ({:.3} bpw) [unpack {unpack_s:.2}s]",
        codes.len(),
        hdr.len(),
        res_total as f64 * 8.0 / (m * k) as f64,
        gate.len(),
        gate.len() as f64 * 8.0 / (m * k) as f64,
    );

    // X through the real quantizer, then the host int8 copy of the same codes.
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            let scale = 0.75 + (col % 5) as f32 * 0.15;
            prng_f32(i, 0xA5A5_00C0) * scale
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr = gpu
        .ensure_int4_mmq_x(&d_x, n, k)
        .expect("ensure_int4_mmq_x");
    let xq_len = (k / 128) * n * 72;
    let xq_host = dtoh_raw(&gpu, xq_ptr, xq_len);
    let (xs, xh, xbuild_s) = build_x_i8(&xq_host, n, k);
    eprintln!(
        "x int8 copy {} B + headers {} B [build {xbuild_s:.2}s]",
        xs.len(),
        xh.len()
    );

    // Upload resident copies.
    let d_a = gpu.upload_raw(&codes, &[codes.len()]).expect("upload A u8");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;
    let d_ah = gpu.upload_raw(&hdr, &[hdr.len()]).expect("upload Ah");
    let ah_ptr = d_ah.buf.as_ptr() as *const std::ffi::c_void;
    let d_x8 = gpu.upload_raw(&xs, &[xs.len()]).expect("upload X s8");
    let x8_ptr = d_x8.buf.as_ptr() as *const std::ffi::c_void;
    let d_xh = gpu.upload_raw(&xh, &[xh.len()]).expect("upload Xh");
    let xh_ptr = d_xh.buf.as_ptr() as *const std::ffi::c_void;
    let d_a_mq4 = gpu.upload_raw(&gate, &[gate.len()]).expect("upload A mq4");
    let a_mq4_ptr = d_a_mq4.buf.as_ptr() as *const std::ffi::c_void;
    let xq_const = xq_ptr as *const std::ffi::c_void;

    let grid = [(n / 128) as u32, (m / 128) as u32, 1];
    let (mi, ki, ni) = (m as i32, k as i32, n as i32);
    let y_elems = n * m;

    // Y0 seeds: set path zeros; add path NONZERO so accumulation is exercised.
    let y0_set = vec![0.0f32; y_elems];
    let y0_add: Vec<f32> = (0..y_elems)
        .map(|i| prng_f32(i, 0xADD0_BEEF ^ 0xA5A5_00C0) * 0.37 + 0.11)
        .collect();
    let mut d_y_ref = gpu.upload_f32(&y0_set, &[n, m]).expect("upload Y ref");
    let mut d_y_w8 = gpu.upload_f32(&y0_set, &[n, m]).expect("upload Y w8");
    let mut d_y_pf = gpu.upload_f32(&y0_set, &[n, m]).expect("upload Y pf");
    let restore = |gpu: &mut Gpu, y_buf: &DeviceBuffer, y0: &[f32]| {
        let bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.hip.memcpy_htod(y_buf, &bytes).expect("restore Y0");
    };

    let mut ok = true;
    for (label, add) in [("set", false), ("add", true)] {
        let y0 = if add { &y0_add } else { &y0_set };
        let add_i = i32::from(add);
        for (yb, name) in [
            (&mut d_y_ref, "ref"),
            (&mut d_y_w8, "w8"),
            (&mut d_y_pf, "pf"),
        ] {
            let _ = name;
            restore(&mut gpu, &yb.buf, y0);
        }
        let mut bref = mk_blob_ref(a_mq4_ptr, xq_const, d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, add_i);
        let ref_sym = if add { REF_ADD } else { REF_SET };
        launch_blob(&mut gpu, ref_sym, grid, REF_BLOCK, REF_SHARED, &mut bref);
        let mut bw8 = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_w8.buf.as_ptr() as *const _, mi, ki, ni, add_i);
        let w8_sym = if add { W8_ADD } else { W8_SET };
        launch_blob(&mut gpu, w8_sym, grid, W8_BLOCK, W8_SHARED, &mut bw8);
        let mut bpf = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_pf.buf.as_ptr() as *const _, mi, ki, ni, add_i);
        let pf_sym = if add { W8PF_ADD } else { W8PF_SET };
        launch_blob(&mut gpu, pf_sym, grid, W8_BLOCK, W8_SHARED, &mut bpf);
        gpu.hip.device_synchronize().expect("sync parity");

        let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
        let y_w8 = gpu.download_f32(&d_y_w8).expect("dl w8");
        let y_pf = gpu.download_f32(&d_y_pf).expect("dl pf");
        let finite = y_ref.iter().all(|v| v.is_finite())
            && y_w8.iter().all(|v| v.is_finite())
            && y_pf.iter().all(|v| v.is_finite());
        let (eq8, mism8, first8) = bitwise_eq(&y_ref, &y_w8);
        let (eqpf, mismpf, firstpf) = bitwise_eq(&y_ref, &y_pf);
        let moved = if add {
            y_ref
                .iter()
                .zip(y0.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
        } else {
            y_ref.iter().any(|v| v.to_bits() != 0)
        };
        let pass = eq8 && eqpf && finite && moved;
        ok &= pass;
        eprintln!(
            "  [gate_proj/{label} M={m} K={k} N={n}] finite={finite} moved={moved} \
             w8_bitwise_eq={eq8} mism={mism8} first={first8:?} \
             pf_bitwise_eq={eqpf} mism={mismpf} first={firstpf:?} [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // A/X immutability over the launches above.
    let mut codes_back = vec![0u8; codes.len()];
    gpu.hip
        .memcpy_dtoh(&mut codes_back, &d_a.buf)
        .expect("dl A back");
    let a_same = codes_back == codes;
    let mut xh_back = vec![0u8; xh.len()];
    gpu.hip
        .memcpy_dtoh(&mut xh_back, &d_xh.buf)
        .expect("dl Xh back");
    let xh_same = xh_back == xh;
    let xq_after = dtoh_raw(&gpu, xq_ptr, xq_len);
    let xq_same = xq_after == xq_host;
    eprintln!("  immutability: a={a_same} xh={xh_same} xq={xq_same}");
    ok &= a_same && xh_same && xq_same;

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        restore(&mut gpu, &d_y_ref.buf, &y0_set);
        restore(&mut gpu, &d_y_w8.buf, &y0_set);
        restore(&mut gpu, &d_y_pf.buf, &y0_set);
        for _ in 0..WARM {
            let mut bref = mk_blob_ref(a_mq4_ptr, xq_const, d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, 0);
            launch_blob(&mut gpu, REF_SET, grid, REF_BLOCK, REF_SHARED, &mut bref);
            let mut bw8 = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_w8.buf.as_ptr() as *const _, mi, ki, ni, 0);
            launch_blob(&mut gpu, W8_SET, grid, W8_BLOCK, W8_SHARED, &mut bw8);
            let mut bpf = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_pf.buf.as_ptr() as *const _, mi, ki, ni, 0);
            launch_blob(&mut gpu, W8PF_SET, grid, W8_BLOCK, W8_SHARED, &mut bpf);
        }
        gpu.hip.device_synchronize().expect("sync warm");

        // Interleave ref/w8/pf to share thermal state (set path: Y stays zero).
        let mut ref_us = Vec::with_capacity(SAMPLES);
        let mut w8_us = Vec::with_capacity(SAMPLES);
        let mut pf_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            let order: [u8; 3] = match i % 3 {
                0 => [0, 1, 2],
                1 => [1, 2, 0],
                _ => [2, 0, 1],
            };
            for arm in order {
                match arm {
                    0 => {
                        let mut b = mk_blob_ref(a_mq4_ptr, xq_const, d_y_ref.buf.as_ptr() as *const _, mi, ki, ni, 0);
                        ref_us.push(time_us(&mut gpu, REF_SET, grid, REF_BLOCK, REF_SHARED, &mut b));
                    }
                    1 => {
                        let mut b = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_w8.buf.as_ptr() as *const _, mi, ki, ni, 0);
                        w8_us.push(time_us(&mut gpu, W8_SET, grid, W8_BLOCK, W8_SHARED, &mut b));
                    }
                    _ => {
                        let mut b = mk_blob_w8(a_ptr, ah_ptr, x8_ptr, xh_ptr, d_y_pf.buf.as_ptr() as *const _, mi, ki, ni, 0);
                        pf_us.push(time_us(&mut gpu, W8PF_SET, grid, W8_BLOCK, W8_SHARED, &mut b));
                    }
                }
            }
        }
        let rmed = median(&mut ref_us);
        let wmed = median(&mut w8_us);
        let pmed = median(&mut pf_us);
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let (rtops, wtops, ptops) = (ops / rmed / 1e6, ops / wmed / 1e6, ops / pmed / 1e6);
        eprintln!(
            "  TIME gate_proj/set: ref_med={rmed:.3} us ({rtops:.2} TOPS, {:.1}% peak)",
            rtops / PEAK_TOPS * 100.0
        );
        eprintln!(
            "  TIME gate_proj/set: w8_med={wmed:.3} us ({wtops:.2} TOPS, {:.1}% peak) ratio={:.4}",
            wtops / PEAK_TOPS * 100.0,
            wmed / rmed
        );
        eprintln!(
            "  TIME gate_proj/set: pf_med={pmed:.3} us ({ptops:.2} TOPS, {:.1}% peak) ratio={:.4} (n={SAMPLES} interleaved)",
            ptops / PEAK_TOPS * 100.0,
            pmed / rmed
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_ah);
    let _ = gpu.free_tensor(d_x8);
    let _ = gpu.free_tensor(d_xh);
    let _ = gpu.free_tensor(d_a_mq4);
    let _ = gpu.free_tensor(d_y_ref);
    let _ = gpu.free_tensor(d_y_w8);
    let _ = gpu.free_tensor(d_y_pf);

    if ok {
        eprintln!("W8A8 PASS: lf16 vs resident-int8 bitwise-equal on gate set+add");
    } else {
        eprintln!("W8A8 FAIL");
        std::process::exit(1);
    }
}
