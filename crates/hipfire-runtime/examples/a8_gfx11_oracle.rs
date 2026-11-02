//! GPU-vs-CPU bit oracle and per-shape timing for the gfx11 A8 route
//! (`kernels/src/gemm_mq4g256v2_residual_mmq_i8.gfx11.hip` + the
//! `block_i8_128` quantizer) on gfx1100 / gfx1151.
//!
//! Default mode: real symmetric MQ4V2 rows of qwen3.8-27b — gate/up layers
//! 0/31/63 (fused SiLU), down SET/ADD (ADD over a non-zero Y), QKVZA and QKV
//! SET, wo ADD, M48/M100 row tails — at N in {48, 64, 80, 100, 200, 333,
//! 511, 512} (partial and odd token tiles) plus one N=1024 case per shape.
//! Every output is compared bit-for-bit against a CPU reference of the
//! pinned integer fold (C = sum (q-8)*x8 per K128, acc = fma(RN(sc*d), C,
//! acc)), plus repeat-launch identity and a bitwise check of the GPU
//! quantizer against the CPU quantizer. Gate/up: g and u come from the CPU
//! reference and only h = g / (1 + expf(-g)) * u is evaluated on the GPU
//! (same expression as the kernel epilogue), because OCML expf has no
//! bit-exact CPU twin.
//!
//! TIME=<shape,...> times every arm in TIME_ARMS on each shape at TIME_N
//! tokens (default 8192): TIME_ROUNDS rounds (default 4), each one
//! TIME_SECS burst (default 1 s) per arm in turn, so the power-limited
//! clock drift of halo hits every arm alike; reports the median us, TOPS and
//! the median sclk / power from hwmon. Arms: `a8:<i>` (TIME_VARIANTS entry
//! i, "defs;defs", default one variant = I8_DEFINES), `iu4` (the production
//! IU4 consumer through the `Gpu` launchers: V2B builder images on gfx1151),
//! `off` (the IU4-off Q8_1 X128 MMQ). Shapes: `qkv` (GDN in_proj_qkv SET,
//! M 10240), `z` (in_proj_z SET, M 6144), `q` (FA q_proj SET, M 12288),
//! `kv` (FA k_proj SET, M 1024), `gate` (gate+up, M 17408 each: a8 fused
//! SiLU, iu4 fused SiLU+A4 down sidecar, off two SETs), `down` (ADD,
//! K 17408), `wo` (out_proj ADD, K 6144). Activation quantization is
//! excluded (fused into the producers on the a8/iu4 routes).
//! I8_DEFINES="NAME=V,..." prepends kernel defines to the oracle module.

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::path::Path;
use std::time::{Duration, Instant};

const I8_MODULE: &str = "a8_gfx11_oracle_i8";
const I8_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_i8.gfx11.hip")
);
const SILU_MODULE: &str = "a8_gfx11_oracle_silu";
const SILU_SRC: &str = r#"
#include <hip/hip_runtime.h>
// Same expression as the i8 gate/up epilogue (I8_SILU_MUL).
extern "C" __global__ void i8_oracle_silu_mul(
    const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        h[i] = (gv) / (1.0f + expf(-(gv))) * (u[i]);
    }
}
"#;
const QUANT: &str = "quantize_int8_mmq_ds128";
/// CTA tile of a define set (`I8_TILE=256` selects the 1024-thread `_t256`
/// entries, otherwise the 512-thread 128 tile).
fn tile_of(defs: &str) -> usize {
    if defs.split(',').any(|d| d.trim() == "I8_TILE=256") {
        256
    } else {
        128
    }
}

fn tile_sym(base: &str, tile: usize) -> String {
    if tile == 256 {
        format!("{base}_t256")
    } else {
        base.to_string()
    }
}
const BLOCK_BYTES: usize = 136;
const MODEL_PREFIX: &str = "model.language_model.layers.";

type Matrix = (usize, usize, Vec<u8>);

fn rand(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn weight(hfq: &HfqFile, layer: usize, suffixes: &[&str]) -> Matrix {
    let mut bytes = Vec::new();
    let mut m = 0;
    let mut k = 0;
    for suffix in suffixes {
        let name = format!("{MODEL_PREFIX}{layer}.{suffix}.weight");
        let (info, data) = hfq
            .tensor_data_vec(&name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.quant_type, 44, "{name}: not qt44");
        assert_eq!(info.shape.len(), 2);
        let (rows, cols) = (info.shape[0] as usize, info.shape[1] as usize);
        assert_eq!(cols % 256, 0);
        assert_eq!(
            data.len(),
            rows * (cols / 256) * 136,
            "{name}: weight layout"
        );
        if m != 0 {
            assert_eq!(k, cols, "{name}: incompatible K");
        }
        m += rows;
        k = cols;
        bytes.extend_from_slice(&data);
    }
    (m, k, bytes)
}

/// Activation with structure the quantizer must survive: smooth per-token
/// scale, sparse x40 outliers, and one all-zero K128 block every 29 tokens.
fn activation(n: usize, k: usize, salt: u32) -> Vec<f32> {
    let kbn = k / 128;
    (0..n * k)
        .map(|i| {
            let (t, c) = (i / k, i % k);
            if t % 29 == 3 && c / 128 == t % kbn {
                return 0.0;
            }
            let mut v = rand(i, salt) * (0.75 + (t % 5) as f32 * 0.15);
            if (t * 131 + c) % 997 == 0 {
                v *= 40.0;
            }
            v
        })
        .collect()
}

/// CPU quantizer (the shared contract): amax, d = RN(amax/127),
/// q = clamp(rint(x/d), -127, 127), zero block d = 0, q = 0; s = sum q.
/// Returns the device-layout blocks ([K/128][N] x 136 B, qs in nibble-pair
/// order), natural-order codes [N][K] and d [K/128][N].
fn quantize_cpu(x: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>) {
    let kbn = k / 128;
    let mut blocks = vec![0u8; kbn * n * BLOCK_BYTES];
    let mut x8 = vec![0i8; n * k];
    let mut ds = vec![0f32; kbn * n];
    for t in 0..n {
        for kb in 0..kbn {
            let src = &x[t * k + kb * 128..t * k + kb * 128 + 128];
            let amax = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let mut s = 0i32;
            let q = &mut x8[t * k + kb * 128..t * k + kb * 128 + 128];
            for (e, &v) in src.iter().enumerate() {
                let qi = if amax == 0.0 {
                    0
                } else {
                    (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
                };
                q[e] = qi as i8;
                s += qi;
            }
            ds[kb * n + t] = d;
            let blk = &mut blocks[(kb * n + t) * BLOCK_BYTES..(kb * n + t + 1) * BLOCK_BYTES];
            blk[0..4].copy_from_slice(&d.to_bits().to_le_bytes());
            blk[4..8].copy_from_slice(&s.to_le_bytes());
            for g in 0..16 {
                for i in 0..4 {
                    blk[8 + 8 * g + i] = q[8 * g + 2 * i] as u8;
                    blk[8 + 8 * g + 4 + i] = q[8 * g + 2 * i + 1] as u8;
                }
            }
        }
    }
    (blocks, x8, ds)
}

/// Symmetric MQ4V2 rows as signed codes (q - 8) [M][K] and per-128 f32 sc.
struct Wprep {
    m: usize,
    k: usize,
    w8: Vec<i8>,
    sc: Vec<f32>,
}

fn prep(a: &Matrix) -> Wprep {
    let (m, k, bytes) = a;
    let (m, k) = (*m, *k);
    let gpr = k / 256;
    let mut w8 = vec![0i8; m * k];
    let mut sc = vec![0f32; m * (k / 128)];
    for r in 0..m {
        for g in 0..gpr {
            let grp = &bytes[(r * gpr + g) * 136..(r * gpr + g + 1) * 136];
            for h in 0..2 {
                sc[r * (k / 128) + 2 * g + h] =
                    half::f16::from_bits(u16::from_le_bytes([grp[4 * h], grp[4 * h + 1]])).to_f32();
            }
            for b in 0..128 {
                let byte = grp[8 + b];
                w8[r * k + g * 256 + 2 * b] = (byte & 15) as i8 - 8;
                w8[r * k + g * 256 + 2 * b + 1] = (byte >> 4) as i8 - 8;
            }
        }
    }
    Wprep { m, k, w8, sc }
}

#[inline(always)]
fn dot128(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    let mut s = 0i32;
    for i in 0..128 {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// CPU reference of the pinned fold, output [N][M] (Y[col*M + row]):
/// per K128 block in order, acc = fma(RN(sc*d), (float)C, acc).
fn gemm_cpu(w: &Wprep, x8: &[i8], ds: &[f32], n: usize) -> Vec<f32> {
    const TILE: usize = 16;
    let (m, k) = (w.m, w.k);
    let kbn = k / 128;
    let mut out = vec![0f32; n * m];
    out.par_chunks_mut(TILE * m)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let n0 = ti * TILE;
            let nt = chunk.len() / m;
            let mut acc = [0f32; TILE];
            for row in 0..m {
                acc[..nt].fill(0.0);
                for kb in 0..kbn {
                    let wr: &[i8; 128] = w.w8[row * k + kb * 128..row * k + kb * 128 + 128]
                        .try_into()
                        .unwrap();
                    let sc = w.sc[row * kbn + kb];
                    for t in 0..nt {
                        let tok = n0 + t;
                        let xr: &[i8; 128] = x8[tok * k + kb * 128..tok * k + kb * 128 + 128]
                            .try_into()
                            .unwrap();
                        let c = dot128(wr, xr);
                        let t1 = sc * ds[kb * n + tok];
                        acc[t] = t1.mul_add(c as f32, acc[t]);
                    }
                }
                for t in 0..nt {
                    chunk[t * m + row] = acc[t];
                }
            }
        });
    out
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn quantize_gpu(gpu: &mut Gpu, d_x: &GpuTensor, n: usize, k: usize) -> GpuTensor {
    let out = gpu
        .zeros(&[(k / 128) * n * BLOCK_BYTES / 4], DType::F32)
        .expect("alloc Xq");
    let mut b = KernargBlob::new();
    b.push_ptr(d_x.buf.as_ptr() as *const c_void);
    b.push_ptr(out.buf.as_ptr() as *const c_void);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    let mut blob = b.into_vec();
    gpu.launch_kernel_blob(
        QUANT,
        [(k.div_ceil(1024) * n) as u32, 1, 1],
        [256, 1, 1],
        0,
        &mut blob,
    )
    .expect("launch quantizer");
    out
}

fn i8_symbol(fused: bool, add: bool) -> &'static str {
    match (fused, add) {
        (true, _) => "gemm_mq4g256v2_gate_up_silu_mmq_i8",
        (false, true) => "gemm_mq4g256v2_residual_mmq_i8_full_add",
        (false, false) => "gemm_mq4g256v2_residual_mmq_i8_full_set",
    }
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    suffix: &str,
    tile: usize,
    a: *const c_void,
    u: Option<*const c_void>,
    xq: *const c_void,
    y: *const c_void,
    m: usize,
    k: usize,
    n: usize,
    add: bool,
) {
    let sym = format!("{}{suffix}", tile_sym(i8_symbol(u.is_some(), add), tile));
    let sym = sym.as_str();
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    if let Some(u) = u {
        b.push_ptr(u);
    }
    b.push_ptr(xq);
    b.push_ptr(y);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    if u.is_none() {
        b.push_i32(i32::from(add));
    }
    let mut blob = b.into_vec();
    let rows = if u.is_some() { 2 * m } else { m };
    let lds = if tile == 256 { 55296 } else { 27648 };
    gpu.launch_kernel_blob(
        sym,
        [rows.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1],
        [4 * tile as u32, 1, 1],
        lds,
        &mut blob,
    )
    .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn compare(label: &str, reference: &[f32], candidate: &[f32], m: usize) -> usize {
    assert_eq!(reference.len(), candidate.len());
    let mut diffs = 0;
    for (i, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        if a.to_bits() != b.to_bits() {
            if diffs < 8 {
                eprintln!(
                    "DIFF {label} n={} row={} ref={:08x} gpu={:08x} ({a:?} {b:?})",
                    i / m,
                    i % m,
                    a.to_bits(),
                    b.to_bits()
                );
            }
            diffs += 1;
        }
    }
    diffs
}

fn silu_gpu(gpu: &mut Gpu, g: &[f32], u: &[f32]) -> Vec<f32> {
    let dg = gpu.upload_f32(g, &[g.len()]).expect("upload g");
    let du = gpu.upload_f32(u, &[u.len()]).expect("upload u");
    let dh = gpu.zeros(&[g.len()], DType::F32).expect("alloc h");
    let mut b = KernargBlob::new();
    b.push_ptr(dg.buf.as_ptr() as *const c_void);
    b.push_ptr(du.buf.as_ptr() as *const c_void);
    b.push_ptr(dh.buf.as_ptr() as *const c_void);
    b.push_i32(g.len() as i32);
    let mut blob = b.into_vec();
    gpu.launch_kernel_blob(
        "i8_oracle_silu_mul",
        [g.len().div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        &mut blob,
    )
    .expect("launch silu ref");
    gpu.hip.device_synchronize().expect("sync silu");
    let h = gpu.download_f32(&dh).expect("download h");
    for t in [dg, du, dh] {
        gpu.free_tensor(t).expect("free silu");
    }
    h
}

#[derive(Default)]
struct Tally {
    cases: usize,
    outputs: usize,
    gemm_diffs: usize,
    repeat_diffs: usize,
    quant_diffs: usize,
}

#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    tally: &mut Tally,
    tile: usize,
    label: &str,
    a: &Matrix,
    u: Option<&Matrix>,
    n: usize,
    add: bool,
    salt: u32,
) {
    let (m, k, bytes) = a;
    let (m, k) = (*m, *k);
    if let Some((um, uk, _)) = u {
        assert_eq!((m, k), (*um, *uk));
    }
    let x = activation(n, k, salt);
    let (blocks_cpu, x8, ds) = quantize_cpu(&x, n, k);
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let d_xq = quantize_gpu(gpu, &d_x, n, k);
    gpu.hip.device_synchronize().expect("sync quant");
    let blocks_gpu = download_bytes(gpu, &d_xq);
    let qd = blocks_cpu
        .iter()
        .zip(&blocks_gpu)
        .filter(|(a, b)| a != b)
        .count();
    let reference: Vec<f32> = if let Some(up) = u {
        let g = gemm_cpu(&prep(a), &x8, &ds, n);
        let uu = gemm_cpu(&prep(up), &x8, &ds, n);
        silu_gpu(gpu, &g, &uu)
    } else {
        gemm_cpu(&prep(a), &x8, &ds, n)
    };
    let y0: Vec<f32> = if add {
        (0..n * m)
            .map(|i| rand(i, salt ^ 0xADD0_BEEF) * 0.37 + 0.11)
            .collect()
    } else {
        vec![f32::from_bits(0x7fc0_dead); n * m]
    };
    let reference: Vec<f32> = if add {
        reference.iter().zip(&y0).map(|(r, o)| o + r).collect()
    } else {
        reference
    };
    let d_a = gpu
        .upload_raw(bytes, &[bytes.len()])
        .expect("upload weights");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload up"));
    let ynew = gpu.upload_f32(&y0, &[n, m]).expect("new Y");
    let yrep = gpu.upload_f32(&y0, &[n, m]).expect("repeat Y");
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let xq = d_xq.buf.as_ptr() as *const c_void;
    launch(
        gpu,
        "",
        tile,
        ap,
        up,
        xq,
        ynew.buf.as_ptr() as *const c_void,
        m,
        k,
        n,
        add,
    );
    launch(
        gpu,
        "",
        tile,
        ap,
        up,
        xq,
        yrep.buf.as_ptr() as *const c_void,
        m,
        k,
        n,
        add,
    );
    gpu.hip.device_synchronize().expect("sync G1");
    let candidate = gpu.download_f32(&ynew).expect("download");
    let repeat = gpu.download_f32(&yrep).expect("download repeat");
    let differing = compare(label, &reference, &candidate, m);
    let repeated = compare(&format!("{label}/repeat"), &candidate, &repeat, m);
    eprintln!("G1 {label} N={n} M={m} K={k} ADD={add} compared={} differing={differing} repeat_differing={repeated} quant_bytes_differing={qd}",
              candidate.len());
    tally.cases += 1;
    tally.outputs += candidate.len();
    tally.gemm_diffs += differing;
    tally.repeat_diffs += repeated;
    tally.quant_diffs += qd;
    for t in [ynew, yrep, d_x, d_xq, d_a] {
        gpu.free_tensor(t).expect("free");
    }
    if let Some(t) = d_u {
        gpu.free_tensor(t).expect("free U");
    }
}

/// halo hwmon (amdgpu): sclk in Hz and power in uW, sampled every 20 ms
/// while a burst runs; returns the medians (MHz, W).
fn hwmon_dir() -> Option<std::path::PathBuf> {
    let drm = std::fs::read_dir("/sys/class/drm").ok()?;
    for card in drm.flatten() {
        let hw = card.path().join("device/hwmon");
        if let Ok(dirs) = std::fs::read_dir(&hw) {
            for d in dirs.flatten() {
                if d.path().join("freq1_input").exists() {
                    return Some(d.path());
                }
            }
        }
    }
    None
}

fn read_num(p: &Path) -> Option<f64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

/// One TIME_SECS burst of `call`; returns (us per call, median MHz, median W).
fn burst(gpu: &mut Gpu, call: &mut dyn FnMut(&mut Gpu)) -> (f64, f64, f64) {
    let secs: f64 = std::env::var("TIME_SECS")
        .ok()
        .map(|v| v.parse().expect("TIME_SECS"))
        .unwrap_or(1.0);
    for _ in 0..2 {
        call(gpu);
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let (mut mhz, mut watts) = (Vec::new(), Vec::new());
            if let Some(dir) = hwmon_dir() {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Some(f) = read_num(&dir.join("freq1_input")) {
                        mhz.push(f / 1e6);
                    }
                    if let Some(p) = read_num(&dir.join("power1_input"))
                        .or_else(|| read_num(&dir.join("power1_average")))
                    {
                        watts.push(p / 1e6);
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            (median(&mut mhz), median(&mut watts))
        })
    };
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < Duration::from_secs_f64(secs) {
        for _ in 0..4 {
            call(gpu);
        }
        gpu.hip.device_synchronize().expect("timed sync");
        calls += 4;
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (mhz, watts) = sampler.join().expect("sampler");
    (us, mhz, watts)
}

/// Interleaved timing of every arm on one shape: TIME_ROUNDS rounds, each
/// running one burst per arm in order, so drift in power/clock state hits
/// every arm alike. Arms: `a8:<variant>` (TIME_VARIANTS index), `iu4`, `off`.
#[allow(clippy::too_many_arguments)]
fn timed(
    gpu: &mut Gpu,
    arms: &[String],
    tiles: &[usize],
    shape: &str,
    a: &Matrix,
    u: Option<&Matrix>,
    awq: Option<&GpuTensor>,
    n: usize,
    add: bool,
) {
    let (m, k, bytes) = a;
    let (m, k) = (*m, *k);
    let rounds: usize = std::env::var("TIME_ROUNDS")
        .ok()
        .map(|v| v.parse().expect("TIME_ROUNDS"))
        .unwrap_or(4);
    let x: Vec<f32> = (0..n * k).map(|i| rand(i, 0x71e0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload A");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload U"));
    let y = gpu.zeros(&[n * m], DType::F32).expect("alloc Y");
    let y2 = gpu.zeros(&[n * m], DType::F32).expect("alloc Y2");
    let xq8 = quantize_gpu(gpu, &d_x, n, k);
    let xq_off = gpu.ensure_q8_1_mmq_x128(&d_x, n, k).expect("quantize X q8");
    gpu.ensure_int4_mmq_x(&d_x, n, k).expect("quantize X i4");
    let prepared = rdna_compute::scratch::Int4MmqPrepared::from_reservation(
        gpu.reserve_int4_mmq(k, n).expect("reserve i4"),
    );
    let xq4 = gpu.int4_mmq_prepared_ptr(&prepared, k, n).expect("i4 ptr");
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let yp = y.buf.as_ptr() as *const c_void;
    let xp8 = xq8.buf.as_ptr() as *const c_void;
    let mut samples: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); arms.len()];
    for _ in 0..rounds {
        for (ai, arm) in arms.iter().enumerate() {
            let mut call: Box<dyn FnMut(&mut Gpu)> = match arm.split_once(':') {
                Some(("a8", v)) => {
                    let suffix = format!("_v{v}");
                    let tile = tiles[v.parse::<usize>().expect("a8 variant index")];
                    Box::new(move |g: &mut Gpu| {
                        launch(g, &suffix, tile, ap, up, xp8, yp, m, k, n, add)
                    })
                }
                _ if arm == "iu4" => match d_u.as_ref() {
                    Some(du) => {
                        let awq = awq.expect("gate arm needs w_down awq");
                        let (da, prep) = (&d_a, &prepared);
                        Box::new(move |g: &mut Gpu| {
                            let r = g
                                .gemm_gate_up_silu_a4_mq4g256v2_iu4_prepared(
                                    da, du, prep, awq, m, m, k, n,
                                )
                                .expect("iu4 gate/up");
                            assert!(r.is_some(), "iu4 SiLU+A4 gate/up not admitted at N={n}");
                        })
                    }
                    None => {
                        let (da, yt) = (&d_a, &y);
                        Box::new(move |g: &mut Gpu| {
                            if add {
                                g.gemm_mq4g256v2_mmq_add_prequant_iu4(da, xq4, yt, m, k, n)
                            } else {
                                g.gemm_mq4g256v2_mmq_set_prequant_iu4(da, xq4, yt, m, k, n)
                            }
                            .expect("iu4 GEMM")
                        })
                    }
                },
                _ if arm == "off" => {
                    let (da, du, yt, y2t) = (&d_a, d_u.as_ref(), &y, &y2);
                    Box::new(move |g: &mut Gpu| {
                        if add {
                            g.gemm_mq4g256v2_mmq_add_prequant(da, xq_off, yt, m, k, n)
                                .expect("off add")
                        } else {
                            g.gemm_mq4g256v2_mmq_set_prequant(da, xq_off, yt, m, k, n)
                                .expect("off set")
                        }
                        if let Some(du) = du {
                            g.gemm_mq4g256v2_mmq_set_prequant(du, xq_off, y2t, m, k, n)
                                .expect("off up");
                        }
                    })
                }
                _ => panic!("arm {arm}"),
            };
            samples[ai].push(burst(gpu, &mut *call));
        }
    }
    let mats = if u.is_some() { 2.0 } else { 1.0 };
    for (arm, s) in arms.iter().zip(samples) {
        let mut us: Vec<f64> = s.iter().map(|v| v.0).collect();
        let mut mhz: Vec<f64> = s.iter().map(|v| v.1).collect();
        let mut w: Vec<f64> = s.iter().map(|v| v.2).collect();
        let rounds_us: Vec<String> = us.iter().map(|v| format!("{v:.0}")).collect();
        let med = median(&mut us);
        let tops = 2.0 * mats * m as f64 * k as f64 * n as f64 / (med * 1e6);
        eprintln!("TIME arm={arm} shape={shape} N={n} M={m} K={k} us_median={med:.1} TOPS={tops:.2} sclk_mhz={:.0} watts={:.1} rounds_us=[{}]",
                  median(&mut mhz), median(&mut w), rounds_us.join(","));
    }
    for t in [d_x, d_a, y, y2, xq8] {
        gpu.free_tensor(t).expect("free");
    }
    if let Some(t) = d_u {
        gpu.free_tensor(t).expect("free U");
    }
}

fn defines(defs: &str) -> String {
    defs.split(',')
        .filter(|d| !d.is_empty())
        .map(|d| {
            let (name, value) = d.split_once('=').expect("define entries are NAME=VALUE");
            format!("#define {name} {value}\n")
        })
        .collect()
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: a8_gfx11_oracle <qwen3.8-27b.mq4-xts> [--compile-only]");
    let base_defs = std::env::var("I8_DEFINES").unwrap_or_default();
    let src = format!("{}{I8_SRC}", defines(&base_defs));
    let mut gpu = Gpu::init().expect("initialize GPU");
    assert!(
        matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151"),
        "arch {}",
        gpu.arch
    );
    let base_tile = tile_of(&base_defs);
    for sym in [
        i8_symbol(true, false),
        i8_symbol(false, true),
        i8_symbol(false, false),
    ] {
        let sym = tile_sym(sym, base_tile);
        gpu.ensure_kernel_public(I8_MODULE, &src, &sym)
            .unwrap_or_else(|e| panic!("compile {sym}: {e}"));
    }
    gpu.ensure_kernel_public(I8_MODULE, &src, QUANT)
        .expect("compile quantizer");
    gpu.ensure_kernel_public(SILU_MODULE, SILU_SRC, "i8_oracle_silu_mul")
        .expect("compile silu ref");
    // TIME_VARIANTS="defs;defs": one module per variant, entry symbols
    // suffixed `_v<i>` (via the kernel's I8_ENTRY macro) so every variant
    // stays loaded for interleaving.
    let variants: Vec<String> = std::env::var("TIME_VARIANTS")
        .unwrap_or(base_defs)
        .split(';')
        .map(str::to_string)
        .collect();
    let tiles: Vec<usize> = variants.iter().map(|v| tile_of(v)).collect();
    for (i, v) in variants.iter().enumerate() {
        let vsrc = format!("{}{I8_SRC}", defines(v))
            .replace(
                "#define I8_ENTRY(name) name##_t256\n",
                &format!("#define I8_ENTRY(name) name##_t256_v{i}\n"),
            )
            .replace(
                "#define I8_ENTRY(name) name\n",
                &format!("#define I8_ENTRY(name) name##_v{i}\n"),
            );
        for sym in [
            i8_symbol(true, false),
            i8_symbol(false, true),
            i8_symbol(false, false),
        ] {
            let sym = format!("{}_v{i}", tile_sym(sym, tiles[i]));
            gpu.ensure_kernel_public(&format!("{I8_MODULE}_v{i}"), &vsrc, &sym)
                .unwrap_or_else(|e| panic!("compile variant {i} [{v}] {sym}: {e}"));
        }
        eprintln!("VARIANT a8:{i} = [{v}] tile {}", tiles[i]);
    }
    if std::env::args().any(|a| a == "--compile-only") {
        return;
    }
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    assert!(hfq.mq4v2_symmetric(), "A8 requires symmetric qt44");
    gpu.mq4v2_symmetric = true;

    // TIME=<shape,...>, TIME_ARMS=<arm,...> (default every a8 variant, iu4, off).
    if let Ok(seq) = std::env::var("TIME") {
        let n: usize = std::env::var("TIME_N")
            .ok()
            .map(|v| v.parse().expect("TIME_N"))
            .unwrap_or(8192);
        let arms: Vec<String> = match std::env::var("TIME_ARMS") {
            Ok(a) => a.split(',').map(str::to_string).collect(),
            Err(_) => (0..variants.len())
                .map(|i| format!("a8:{i}"))
                .chain(["iu4".into(), "off".into()])
                .collect(),
        };
        let gate = weight(&hfq, 0, &["mlp.gate_proj"]);
        let up = weight(&hfq, 0, &["mlp.up_proj"]);
        let down = weight(&hfq, 0, &["mlp.down_proj"]);
        // Timing only: the A4 epilogue's AWQ divide cost does not depend on values.
        let awq = Some(
            gpu.upload_f32(&vec![1.0f32; gate.0], &[gate.0])
                .expect("upload awq"),
        );
        let qkv = weight(&hfq, 0, &["linear_attn.in_proj_qkv"]);
        let z = weight(&hfq, 0, &["linear_attn.in_proj_z"]);
        let wo = weight(&hfq, 0, &["linear_attn.out_proj"]);
        let q = weight(&hfq, 3, &["self_attn.q_proj"]);
        let kv = weight(&hfq, 3, &["self_attn.k_proj"]);
        for shape in seq.split(',') {
            match shape {
                "gate" => timed(
                    &mut gpu,
                    &arms,
                    &tiles,
                    shape,
                    &gate,
                    Some(&up),
                    awq.as_ref(),
                    n,
                    false,
                ),
                "down" => timed(&mut gpu, &arms, &tiles, shape, &down, None, None, n, true),
                "wo" => timed(&mut gpu, &arms, &tiles, shape, &wo, None, None, n, true),
                "qkv" => timed(&mut gpu, &arms, &tiles, shape, &qkv, None, None, n, false),
                "z" => timed(&mut gpu, &arms, &tiles, shape, &z, None, None, n, false),
                "q" => timed(&mut gpu, &arms, &tiles, shape, &q, None, None, n, false),
                "kv" => timed(&mut gpu, &arms, &tiles, shape, &kv, None, None, n, false),
                other => panic!("shape {other}"),
            }
        }
        return;
    }

    let mut tally = Tally::default();
    let quick = std::env::var("QUICK").ok().as_deref() == Some("1");
    let ns: &[usize] = if quick {
        &[48, 200, 512]
    } else {
        &[48, 64, 80, 100, 200, 333, 511, 512]
    };
    let layers: &[usize] = if quick { &[0] } else { &[0, 31, 63] };
    for &layer in layers {
        let gate = weight(&hfq, layer, &["mlp.gate_proj"]);
        let up = weight(&hfq, layer, &["mlp.up_proj"]);
        for &n in ns {
            check(
                &mut gpu,
                &mut tally,
                base_tile,
                &format!("gate-up/layer{layer}"),
                &gate,
                Some(&up),
                n,
                false,
                0x7000 + layer as u32 + n as u32,
            );
        }
    }
    let down = weight(&hfq, 0, &["mlp.down_proj"]);
    let wo = weight(&hfq, 3, &["self_attn.o_proj"]);
    let qkvza = weight(
        &hfq,
        0,
        &[
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
        ],
    );
    let qkv = weight(
        &hfq,
        31,
        &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    );
    let tail48 = weight(&hfq, 0, &["linear_attn.in_proj_a"]);
    let tail100 = (
        100,
        tail48.1,
        [
            tail48.2.as_slice(),
            &weight(&hfq, 0, &["linear_attn.in_proj_b"]).2,
            &tail48.2[..4 * (tail48.1 / 256) * 136],
        ]
        .concat(),
    );
    let big: &[usize] = if quick { &[] } else { &[1024] };
    for &n in ns.iter().chain(big) {
        check(
            &mut gpu,
            &mut tally,
            base_tile,
            "down/set",
            &down,
            None,
            n,
            false,
            0xD000 + n as u32,
        );
        check(
            &mut gpu,
            &mut tally,
            base_tile,
            "down/add",
            &down,
            None,
            n,
            true,
            0xD100 + n as u32,
        );
        if !quick {
            check(
                &mut gpu,
                &mut tally,
                base_tile,
                "wo/add",
                &wo,
                None,
                n,
                true,
                0xC000 + n as u32,
            );
            check(
                &mut gpu,
                &mut tally,
                base_tile,
                "qkvza/set",
                &qkvza,
                None,
                n,
                false,
                0xA000 + n as u32,
            );
            check(
                &mut gpu,
                &mut tally,
                base_tile,
                "qkv/set",
                &qkv,
                None,
                n,
                false,
                0xB000 + n as u32,
            );
        }
    }
    for (label, matrix) in [("m48tail/set", &tail48), ("m100tail/set", &tail100)] {
        for &n in if quick {
            &[48usize][..]
        } else {
            &[48usize, 80, 333, 512][..]
        } {
            check(
                &mut gpu,
                &mut tally,
                base_tile,
                label,
                matrix,
                None,
                n,
                false,
                0xE000 + n as u32,
            );
        }
    }
    eprintln!("SUMMARY cases={} outputs_compared={} gemm_differing={} repeat_differing={} quant_bytes_differing={}",
              tally.cases, tally.outputs, tally.gemm_diffs, tally.repeat_diffs, tally.quant_diffs);
    if tally.gemm_diffs + tally.repeat_diffs + tally.quant_diffs != 0 {
        eprintln!("GFX11-A8 ORACLE FAIL");
        std::process::exit(1);
    }
    eprintln!("GFX11-A8 ORACLE PASS");
}
