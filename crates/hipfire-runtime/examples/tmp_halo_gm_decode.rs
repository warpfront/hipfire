//! Throwaway group-major (GM) decode oracle: row-major shipping vs GM twins.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, builds the group-major copy of A on the host (permute 136-B
//! groups: dst[(kb*M+row)*136] = src[(row*gpr+kb)*136]), and raw-launches:
//!   N=1:                gemv_mq4g256v2_multirow_r2 vs _gm (f32 x, grid [M/2];
//!                       daemon default: rows=2 on gfx1151) + scalar
//!                       gemv_mq4g256v2 vs _gm (ROWS=1 route) + residual
//!                       gemv_mq4g256v2_residual vs _gm (f32 x, Y+=)
//! Same ABI/grid/block/LDS per pair; asserts bitwise-equal Y plus A/X
//! immutability.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-gmd HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1 \
//!     ./target/release/examples/tmp_halo_gm_decode \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]
//!
//! `--compile-only` JITs the eight symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (100 samples each, warm) with medians and the gm/ref ratio.
//! The host group-major relayout time is printed too (informative only; a
//! production route would relayout once offline).

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Module sources mirror crates/rdna-compute/src/kernels.rs so the shipping
// symbols are the exact production objects (residual GEMV keeps its
// gfx12_weight_cache_policy.inc preamble; on gfx1151 it lowers to plain
// loads, identical to the raw file).
const GEMV_SRC: &str = include_str!("../../../kernels/src/gemv_mq4g256v2.hip");
const GEMV_RES_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_residual.hip")
);
const GEMV_MR_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_multirow.hip")
);
const GEMM_SRC: &str = include_str!("../../../kernels/src/gemm_mq4g256v2_residual_wmma.hip");
const MW_SRC: &str = include_str!("../../../kernels/src/gemm_mqv2_wmma_gfx11_mw_lds.hip");
const GEMV_MOD: &str = "tmp_halo_gm_decode_gemv";
const GEMV_RES_MOD: &str = "tmp_halo_gm_decode_gemv_res";
const GEMV_MR_MOD: &str = "tmp_halo_gm_decode_gemv_mr";
const GEMM_MOD: &str = "tmp_halo_gm_decode_gemm";
const MW_MOD: &str = "tmp_halo_gm_decode_mw";

const GEMV_REF: &str = "gemv_mq4g256v2";
const GEMV_GM: &str = "gemv_mq4g256v2_gm";
const GEMV_RES_REF: &str = "gemv_mq4g256v2_residual";
const GEMV_RES_GM: &str = "gemv_mq4g256v2_residual_gm";
const GEMV_MR_REF: &str = "gemv_mq4g256v2_multirow_r2";
const GEMV_MR_GM: &str = "gemv_mq4g256v2_multirow_r2_gm";
const GEMM_REF: &str = "gemm_mq4g256v2_residual_wmma";
const GEMM_GM: &str = "gemm_mq4g256v2_residual_wmma_gm";
const MW_REF: &str = "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds";
const MW_GM: &str = "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds_gm";

const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";
const NS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

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
fn blob_gemv(
    a: *const std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(x);
    b.push_ptr(y);
    b.push_i32(m);
    b.push_i32(k);
    b.into_vec()
}
/// Daemon route on gfx1151 at batch N, read off the dispatch selectors:
/// gemm_mq4g256v2_batched_lmhead (batch==1 -> gemv; else wmma gate +
/// memset-zero + gemm_hfq4g256_residual_mq4v2) and gemm_mq4g256v2_residual_wmma
/// (mqv2_mw_waves is gfx1100-only so None on gfx1151; mqv2_prefill_batch_tile
/// admits gfx1151-Residual only at N>=96 -> BT4).
fn daemon_route(n: usize) -> &'static str {
    if n == 1 {
        "gemv_mq4g256v2_multirow_r2 (rows default 2 on gfx1151; residual adds via gemv_mq4g256v2_residual)"
    } else if n < 96 {
        "gemm_mq4g256v2_residual_wmma base 16x16 (N<96: no BT tile; mw policy gfx1100-only)"
    } else {
        "gemm_mq4g256v2_residual_wmma_gfx1151_bt4 (N>=96; not exercised here)"
    }
}

fn blob_gemm(
    a: *const std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(x);
    b.push_ptr(y);
    b.push_i32(m);
    b.push_i32(k);
    b.push_i32(n);
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn launch_blob(
    gpu: &Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    args: Vec<u8>,
) {
    let mut blob = args;
    gpu.launch_kernel_blob(func, grid, block, 0, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

fn time_blob(
    gpu: &Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    args: Vec<u8>,
) -> f64 {
    let mut blob = args;
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    let stream = gpu.active_stream.as_ref();
    gpu.hip.event_record(&start, stream).expect("record start");
    gpu.launch_kernel_blob(func, grid, block, 0, &mut blob)
        .unwrap_or_else(|e| panic!("timed launch {func}: {e}"));
    gpu.hip.event_record(&stop, stream).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms as f64 * 1e3
}

#[allow(clippy::too_many_arguments)]
fn run_pair(
    gpu: &mut Gpu,
    label: &str,
    ref_sym: &str,
    gm_sym: &str,
    grid: [u32; 3],
    block: [u32; 3],
    a_rm: &[u8],
    a_gm: &[u8],
    like: &GpuTensorLike,
    y0: &[f32],
    time: bool,
) -> bool {
    let d_a_rm = gpu.upload_raw(a_rm, &[a_rm.len()]).expect("upload A rm");
    let a_rm_ptr = d_a_rm.buf.as_ptr() as *const std::ffi::c_void;
    let d_a_gm = gpu.upload_raw(a_gm, &[a_gm.len()]).expect("upload A gm");
    let a_gm_ptr = d_a_gm.buf.as_ptr() as *const std::ffi::c_void;

    let d_y_ref = gpu.upload_f32(y0, &[y0.len()]).expect("upload Y ref");
    let d_y_gm = gpu.upload_f32(y0, &[y0.len()]).expect("upload Y gm");
    let y_ref_ptr = d_y_ref.buf.as_ptr() as *const std::ffi::c_void;
    let y_gm_ptr = d_y_gm.buf.as_ptr() as *const std::ffi::c_void;

    let mk_ref = || like.blob(a_rm_ptr, y_ref_ptr);
    let mk_gm = || like.blob(a_gm_ptr, y_gm_ptr);

    launch_blob(gpu, ref_sym, grid, block, mk_ref());
    launch_blob(gpu, gm_sym, grid, block, mk_gm());
    gpu.hip.device_synchronize().expect("sync parity");

    let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
    let y_gm = gpu.download_f32(&d_y_gm).expect("dl gm");
    let finite = y_ref.iter().all(|v| v.is_finite()) && y_gm.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq(&y_ref, &y_gm);
    let moved = y_ref
        .iter()
        .zip(y0.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());

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
    let x_same = like.check_immutable(gpu);

    let ok = eq && finite && moved && rm_same && gm_same && x_same;
    eprintln!(
        "  [{label}] finite={finite} moved={moved} bitwise_eq={eq} mism={mism} \
         first={first:?} rm_immutable={rm_same} gm_immutable={gm_same} x_immutable={x_same} [{}]",
        if ok { "PASS" } else { "FAIL" }
    );

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        let y0_bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
        for _ in 0..WARM {
            gpu.hip
                .memcpy_htod(&d_y_ref.buf, &y0_bytes)
                .expect("restore Y0 ref");
            launch_blob(gpu, ref_sym, grid, block, mk_ref());
            gpu.hip
                .memcpy_htod(&d_y_gm.buf, &y0_bytes)
                .expect("restore Y0 gm");
            launch_blob(gpu, gm_sym, grid, block, mk_gm());
        }
        gpu.hip.device_synchronize().expect("sync warm");
        let mut ref_us = Vec::with_capacity(SAMPLES);
        let mut gm_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            if i % 2 == 0 {
                gpu.hip
                    .memcpy_htod(&d_y_ref.buf, &y0_bytes)
                    .expect("restore Y0 ref");
                ref_us.push(time_blob(gpu, ref_sym, grid, block, mk_ref()));
                gpu.hip
                    .memcpy_htod(&d_y_gm.buf, &y0_bytes)
                    .expect("restore Y0 gm");
                gm_us.push(time_blob(gpu, gm_sym, grid, block, mk_gm()));
            } else {
                gpu.hip
                    .memcpy_htod(&d_y_gm.buf, &y0_bytes)
                    .expect("restore Y0 gm");
                gm_us.push(time_blob(gpu, gm_sym, grid, block, mk_gm()));
                gpu.hip
                    .memcpy_htod(&d_y_ref.buf, &y0_bytes)
                    .expect("restore Y0 ref");
                ref_us.push(time_blob(gpu, ref_sym, grid, block, mk_ref()));
            }
        }
        let rmed = median(&mut ref_us);
        let gmed = median(&mut gm_us);
        eprintln!(
            "  TIME {label}: ref_med={rmed:.3} us  gm_med={gmed:.3} us  ratio={:.4} (n={SAMPLES} interleaved)",
            gmed / rmed
        );
    }

    let _ = gpu.free_tensor(d_a_rm);
    let _ = gpu.free_tensor(d_a_gm);
    let _ = gpu.free_tensor(d_y_ref);
    let _ = gpu.free_tensor(d_y_gm);
    ok
}

struct GpuTensorLike {
    x_ptr: *mut std::ffi::c_void,
    x_len_bytes: usize,
    x_before: Vec<u8>,
    m: i32,
    k: i32,
    n: i32,
    is_gemv: bool,
    held: Vec<rdna_compute::GpuTensor>,
}

impl GpuTensorLike {
    fn blob(
        &self,
        a_ptr: *const std::ffi::c_void,
        y_ptr: *const std::ffi::c_void,
    ) -> Vec<u8> {
        if self.is_gemv {
            blob_gemv(
                a_ptr,
                self.x_ptr as *const _,
                y_ptr,
                self.m,
                self.k,
            )
        } else {
            blob_gemm(
                a_ptr,
                self.x_ptr as *const _,
                y_ptr,
                self.m,
                self.k,
                self.n,
            )
        }
    }
    fn check_immutable(&self, gpu: &Gpu) -> bool {
        let view = unsafe { DeviceBuffer::from_raw(self.x_ptr, self.x_len_bytes) };
        let mut host = vec![0u8; self.x_len_bytes];
        gpu.hip
            .memcpy_dtoh(&mut host, &view)
            .expect("dl X back");
        let _ = view.is_borrowed();
        host == self.x_before
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
            eprintln!("usage: tmp_halo_gm_decode <model.mq4-xt> [--compile-only]");
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
    eprintln!("tmp_halo_gm_decode on {}  model={model}  TIME={time}", gpu.arch);
    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: gm symbols are #if __gfx1151__; arch={} may fail JIT of gm entries",
            gpu.arch
        );
    }

    // Compile-only: JIT all ten entries, print the cache path, return.
    for (md, src, sym) in [
        (GEMV_MOD, GEMV_SRC, GEMV_REF),
        (GEMV_MOD, GEMV_SRC, GEMV_GM),
        (GEMV_RES_MOD, GEMV_RES_SRC, GEMV_RES_REF),
        (GEMV_RES_MOD, GEMV_RES_SRC, GEMV_RES_GM),
        (GEMV_MR_MOD, GEMV_MR_SRC, GEMV_MR_REF),
        (GEMV_MR_MOD, GEMV_MR_SRC, GEMV_MR_GM),
        (GEMM_MOD, GEMM_SRC, GEMM_REF),
        (GEMM_MOD, GEMM_SRC, GEMM_GM),
        (MW_MOD, MW_SRC, MW_REF),
        (MW_MOD, MW_SRC, MW_GM),
    ] {
        gpu.ensure_kernel_public(md, src, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("JIT OK: {sym}");
    }
    let dir = cache_dir(&gpu.arch);
    eprintln!("CACHE_PATH {}", dir.display());
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let s = ent.file_name().to_string_lossy().into_owned();
            if s.starts_with("tmp_halo_gm_decode") {
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
    drop(hfq);

    let (gate_gm, gate_re_s) = group_major_copy(&gate, gate_m, gate_k);
    eprintln!("relayout gate_proj row-major -> group-major: {gate_re_s:.3}s host (informative only)");
    let (down_gm, down_re_s) = group_major_copy(&down, down_m, down_k);
    eprintln!("relayout down_proj row-major -> group-major: {down_re_s:.3}s host (informative only)");

    let mut ok = true;
    for (tname, m, k, a_rm, a_gm) in [
        ("gate_proj", gate_m, gate_k, &gate, &gate_gm),
        ("down_proj", down_m, down_k, &down, &down_gm),
    ] {
        for n in NS {
            eprintln!("N={n}: daemon route on gfx1151: {}", daemon_route(n));
            if n == 1 {
                // N=1 decode GEMV, f32 x [K], overwrite Y.
                let x: Vec<f32> =
                    (0..k).map(|i| prng_f32(i, 0x6DEC_0001) * 0.9).collect();
                let d_x = gpu.upload_f32(&x, &[k]).expect("upload X f32");
                let x_ptr = d_x.buf.as_ptr() as *mut std::ffi::c_void;
                let x_len = k * 4;
                let mut xb = vec![0u8; x_len];
                gpu.hip.memcpy_dtoh(&mut xb, &d_x.buf).expect("snap X");
                let mut like = GpuTensorLike {
                    x_ptr,
                    x_len_bytes: x_len,
                    x_before: xb,
                    m: m as i32,
                    k: k as i32,
                    n: n as i32,
                    is_gemv: true,
                    held: vec![d_x],
                };
                let y0 = vec![0.0f32; m];
                ok &= run_pair(
                    &mut gpu,
                    &format!("{tname}/gemv/N=1"),
                    GEMV_REF,
                    GEMV_GM,
                    [m as u32, 1, 1],
                    [32, 1, 1],
                    a_rm,
                    a_gm,
                    &like,
                    &y0,
                    time,
                );
                // N=1 daemon-default path on gfx1151 (rows default 2):
                // multirow r2, same X buffer, grid ceil(M/2).
                ok &= run_pair(
                    &mut gpu,
                    &format!("{tname}/gemv_multirow_r2/N=1"),
                    GEMV_MR_REF,
                    GEMV_MR_GM,
                    [m.div_ceil(2) as u32, 1, 1],
                    [32, 1, 1],
                    a_rm,
                    a_gm,
                    &like,
                    &y0,
                    time,
                );
                for t in like.held.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
                // N=1 residual GEMV, f32 x, Y += (nonzero seed exercises add).
                let x2: Vec<f32> =
                    (0..k).map(|i| prng_f32(i, 0x6DEC_0002) * 0.9).collect();
                let d_x2 = gpu.upload_f32(&x2, &[k]).expect("upload X f32 res");
                let x2_ptr = d_x2.buf.as_ptr() as *mut std::ffi::c_void;
                let mut xb2 = vec![0u8; x_len];
                gpu.hip.memcpy_dtoh(&mut xb2, &d_x2.buf).expect("snap X2");
                let mut like2 = GpuTensorLike {
                    x_ptr: x2_ptr,
                    x_len_bytes: x_len,
                    x_before: xb2,
                    m: m as i32,
                    k: k as i32,
                    n: n as i32,
                    is_gemv: true,
                    held: vec![d_x2],
                };
                let y0r: Vec<f32> =
                    (0..m).map(|i| prng_f32(i, 0xADD0_0001) * 0.37 + 0.11).collect();
                ok &= run_pair(
                    &mut gpu,
                    &format!("{tname}/gemv_residual/N=1"),
                    GEMV_RES_REF,
                    GEMV_RES_GM,
                    [m as u32, 1, 1],
                    [32, 1, 1],
                    a_rm,
                    a_gm,
                    &like2,
                    &y0r,
                    time,
                );
                for t in like2.held.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
            } else {
                // N>=2 batched-decode GEMM, F16 x [N x K] row-major (same
                // bytes the shipping ensure_fp16_x convert would produce are
                // unnecessary here: both sides share one X buffer, so any
                // finite F16 content proves layout parity).
                let xf: Vec<u16> = (0..n * k)
                    .map(|i| {
                        half::f16::from_f32(prng_f32(i, 0x6DEC_1000 + n as u32) * 0.9)
                            .to_bits()
                    })
                    .collect();
                let d_x = gpu.upload_f16_bits(&xf, &[n, k]).expect("upload X f16");
                let x_ptr = d_x.buf.as_ptr() as *mut std::ffi::c_void;
                let x_len = n * k * 2;
                let mut xb = vec![0u8; x_len];
                gpu.hip.memcpy_dtoh(&mut xb, &d_x.buf).expect("snap Xf");
                let mut like = GpuTensorLike {
                    x_ptr,
                    x_len_bytes: x_len,
                    x_before: xb,
                    m: m as i32,
                    k: k as i32,
                    n: n as i32,
                    is_gemv: false,
                    held: vec![d_x],
                };
                // Y += needs a nonzero seed so accumulation (not just the
                // seed copy) is compared.
                let y0: Vec<f32> = (0..n * m)
                    .map(|i| prng_f32(i, 0xADD0_1000 + n as u32) * 0.37 + 0.11)
                    .collect();
                let row_tiles = m.div_ceil(16) as u32;
                let bt = n.div_ceil(16) as u32;
                ok &= run_pair(
                    &mut gpu,
                    &format!("{tname}/gemm_base/N={n}"),
                    GEMM_REF,
                    GEMM_GM,
                    [row_tiles, bt, 1],
                    [32, 1, 1],
                    a_rm,
                    a_gm,
                    &like,
                    &y0,
                    time,
                );
                let mw_bt = n.div_ceil(64) as u32;
                ok &= run_pair(
                    &mut gpu,
                    &format!("{tname}/gemm_mw4/N={n}"),
                    MW_REF,
                    MW_GM,
                    [row_tiles, mw_bt, 1],
                    [128, 1, 1],
                    a_rm,
                    a_gm,
                    &like,
                    &y0,
                    time,
                );
                for t in like.held.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
            }
        }
    }

    eprintln!(
        "NOTE gfx1151 decode routes: N=1 gemv_mq4g256v2 (+residual twin where the \
         caller holds a residual); N=2..95 base gemm_mq4g256v2_residual_wmma; N>=96 \
         gfx1151 BT4 (prefill, not covered here). DFlash ksplit/ldsstage and the \
         draft-collapse overwrite GEMM are exact-gfx1100-only (Off on gfx1151); \
         gfx11 MW4 is off the big-M policy but serves small-M MMQ tails via \
         small_tail_set (covered above as the tail oracle). Fused gate_up/qkvza \
         GEMMs are prefill-only (N>=96 policy) — same 136-B groups, untouched."
    );
    if ok {
        eprintln!("GM DECODE PASS: row-major shipping vs group-major bitwise-equal on all cases");
    } else {
        eprintln!("GM DECODE FAIL");
        std::process::exit(1);
    }
}
