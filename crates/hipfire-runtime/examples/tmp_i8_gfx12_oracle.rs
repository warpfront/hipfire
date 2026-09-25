//! GPU-vs-CPU bit oracle and sustained timing for the gfx1201 A8 route
//! (`kernels/src/gemm_mq4g256v2_residual_mmq_i8.gfx12.hip` + the
//! `block_i8_128` quantizer), design /home/kaden/qcal/perf/fp8-4k5/a8-route.md.
//!
//! Default mode (G1): K1's 34-case pattern on real symmetric MQ4V2 rows —
//! gate/up layers 0/31/63 (fused SiLU), down SET/ADD (ADD over a non-zero Y),
//! QKVZA and QKV SET, N in {48, 80, 512, 8192}, M48/M100 tails — every output
//! compared bit-for-bit against a CPU reference of the pinned integer fold
//! (C = sum (q-8)*x8 per K128, acc = fma(RN(sc*d), C, acc)), plus
//! repeat-launch identity and a bitwise check of the GPU quantizer against the
//! CPU quantizer on every activation. Gate/up: g and u come from the CPU
//! reference and only h = g / (1 + expf(-g)) * u is evaluated on the GPU (same
//! expression as the kernel epilogue, one tiny kernel), because OCML expf has
//! no bit-exact CPU twin.
//!
//! TIME=<arm:shape,...> runs sustained >= 10 s arms in the given order (arms
//! `a8` / `k1`, shapes `gate` (fused gate/up), `down` (down ADD), `set`
//! (in_proj_qkv SET)) at TIME_N tokens (default 8192), sampling AMD-SMI at
//! 20 Hz into TIME_OUT (default /home/kaden/qcal/perf/fp8-4k5/a8/g2).
//! I8_DEFINES="NAME=V,..." prepends kernel defines (variant builds).
//! Card E only: HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES must be its UUID.

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CARD_UUID: &str = "GPU-05f92432f2312a0e";
const CARD_BDF: &str = "0000:c3:00.0";
const I8_MODULE: &str = "gemm_mq4g256v2_residual_mmq_i8_gfx12";
const I8_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_i8.gfx12.hip")
);
const V3_MODULE: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_v3";
const V3_SRC: &str = concat!(
    "#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n",
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip")
);
const SILU_MODULE: &str = "tmp_i8_oracle_silu";
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
const I8_LDS: u32 = 28672;
const V3_LDS: u32 = 20480;
const BLOCK_BYTES: usize = 136;
const TELEMETRY: &str = r#"
import csv, os, sys, time
os.environ['AMDSMI_GPU_METRICS_CACHE_MS'] = '50'
sys.path.insert(0, '/opt/rocm/core/share/amd_smi')
import amdsmi
amdsmi.amdsmi_init()
h = next(h for h in amdsmi.amdsmi_get_processor_handles()
         if amdsmi.amdsmi_get_gpu_device_bdf(h).lower() == sys.argv[2])
assert amdsmi.amdsmi_get_power_cap_info(h)['power_cap'] == 300000000
with open(sys.argv[1], 'w', newline='', buffering=1) as file:
    w = csv.writer(file)
    w.writerow(['monotonic_ns', 'socket_power_w', 'gfx_clock_mhz', 'gfx_busy_pct', 'error'])
    print('READY', flush=True)
    nxt = time.monotonic()
    while True:
        try:
            m = amdsmi.amdsmi_get_gpu_metrics_info(h)
            w.writerow([time.monotonic_ns(),m['average_socket_power'],m['current_gfxclk'],m['average_gfx_activity'],''])
        except Exception as exc:
            w.writerow([time.monotonic_ns(),'','','',str(exc).replace(',', ';')])
        nxt += .05
        time.sleep(max(0, nxt-time.monotonic()))
"#;
const MODEL_PREFIX: &str = "model.language_model.layers.";

type Matrix = (usize, usize, Vec<u8>);

fn rand(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9e3779b1).wrapping_add(salt).wrapping_mul(0x85ebca6b);
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
        let (info, data) = hfq.tensor_data_vec(&name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.quant_type, 44, "{name}: not qt44");
        assert_eq!(info.shape.len(), 2);
        let (rows, cols) = (info.shape[0] as usize, info.shape[1] as usize);
        assert_eq!(cols % 256, 0);
        assert_eq!(data.len(), rows * (cols / 256) * 136, "{name}: weight layout");
        if m != 0 { assert_eq!(k, cols, "{name}: incompatible K"); }
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
    (0..n * k).map(|i| {
        let (t, c) = (i / k, i % k);
        if t % 29 == 3 && c / 128 == t % kbn { return 0.0; }
        let mut v = rand(i, salt) * (0.75 + (t % 5) as f32 * 0.15);
        if (t * 131 + c) % 997 == 0 { v *= 40.0; }
        v
    }).collect()
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
                let qi = if amax == 0.0 { 0 } else { (v / d).round_ties_even().clamp(-127.0, 127.0) as i32 };
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
struct Wprep { m: usize, k: usize, w8: Vec<i8>, sc: Vec<f32> }

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
/// `split` (negative control, CPU_FOLD=split) rounds the product before the
/// add instead; the bit oracle must reject it.
fn gemm_cpu(w: &Wprep, x8: &[i8], ds: &[f32], n: usize) -> Vec<f32> {
    const TILE: usize = 16;
    let split = std::env::var("CPU_FOLD").ok().as_deref() == Some("split");
    let (m, k) = (w.m, w.k);
    let kbn = k / 128;
    let mut out = vec![0f32; n * m];
    out.par_chunks_mut(TILE * m).enumerate().for_each(|(ti, chunk)| {
        let n0 = ti * TILE;
        let nt = chunk.len() / m;
        let mut acc = [0f32; TILE];
        for row in 0..m {
            acc[..nt].fill(0.0);
            for kb in 0..kbn {
                let wr: &[i8; 128] = w.w8[row * k + kb * 128..row * k + kb * 128 + 128].try_into().unwrap();
                let sc = w.sc[row * kbn + kb];
                for t in 0..nt {
                    let tok = n0 + t;
                    let xr: &[i8; 128] = x8[tok * k + kb * 128..tok * k + kb * 128 + 128].try_into().unwrap();
                    let c = dot128(wr, xr);
                    let t1 = sc * ds[kb * n + tok];
                    acc[t] = if split { t1 * c as f32 + acc[t] } else { t1.mul_add(c as f32, acc[t]) };
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
    gpu.download_f32(t).expect("download").iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

fn quantize_gpu(gpu: &mut Gpu, d_x: &GpuTensor, n: usize, k: usize) -> GpuTensor {
    let out = gpu.zeros(&[(k / 128) * n * BLOCK_BYTES / 4], DType::F32).expect("alloc Xq");
    let mut b = KernargBlob::new();
    b.push_ptr(d_x.buf.as_ptr() as *const c_void);
    b.push_ptr(out.buf.as_ptr() as *const c_void);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    let mut blob = b.into_vec();
    gpu.launch_kernel_blob(QUANT, [k.div_ceil(1024) as u32, n as u32, 1], [256, 1, 1], 0, &mut blob)
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

fn v3_symbol(fused: bool, add: bool) -> &'static str {
    match (fused, add) {
        (true, _) => "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3",
        (false, true) => "gemm_mq4g256v2_residual_mmq_iu4_full_add_v3",
        (false, false) => "gemm_mq4g256v2_residual_mmq_iu4_full_set_v3",
    }
}

#[allow(clippy::too_many_arguments)]
fn launch(gpu: &mut Gpu, sym: &str, lds: u32, a: *const c_void, u: Option<*const c_void>,
          xq: *const c_void, y: *const c_void, m: usize, k: usize, n: usize, add: bool) {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    if let Some(u) = u { b.push_ptr(u); }
    b.push_ptr(xq);
    b.push_ptr(y);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    if u.is_none() { b.push_i32(i32::from(add)); }
    let mut blob = b.into_vec();
    let rows = if u.is_some() { 2 * m } else { m };
    gpu.launch_kernel_blob(sym, [rows.div_ceil(128) as u32, n.div_ceil(128) as u32, 1],
                           [256, 1, 1], lds, &mut blob)
        .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn compare(label: &str, reference: &[f32], candidate: &[f32], m: usize) -> usize {
    assert_eq!(reference.len(), candidate.len());
    let mut diffs = 0;
    for (i, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        if a.to_bits() != b.to_bits() {
            if diffs < 8 {
                eprintln!("DIFF {label} n={} row={} ref={:08x} gpu={:08x} ({a:?} {b:?})",
                          i / m, i % m, a.to_bits(), b.to_bits());
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
    gpu.launch_kernel_blob("i8_oracle_silu_mul", [g.len().div_ceil(256) as u32, 1, 1], [256, 1, 1], 0, &mut blob)
        .expect("launch silu ref");
    gpu.hip.device_synchronize().expect("sync silu");
    let h = gpu.download_f32(&dh).expect("download h");
    for t in [dg, du, dh] { gpu.free_tensor(t).expect("free silu"); }
    h
}

struct Tally { cases: usize, outputs: usize, gemm_diffs: usize, repeat_diffs: usize, quant_diffs: usize,
               silu_host_max_ulp: u32 }

#[allow(clippy::too_many_arguments)]
fn check(gpu: &mut Gpu, tally: &mut Tally, label: &str, a: &Matrix, u: Option<&Matrix>, n: usize,
         add: bool, salt: u32) {
    let (m, k, bytes) = a;
    let (m, k) = (*m, *k);
    if let Some((um, uk, _)) = u { assert_eq!((m, k), (*um, *uk)); }
    let x = activation(n, k, salt);
    let (blocks_cpu, x8, ds) = quantize_cpu(&x, n, k);
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let d_xq = quantize_gpu(gpu, &d_x, n, k);
    gpu.hip.device_synchronize().expect("sync quant");
    let blocks_gpu = download_bytes(gpu, &d_xq);
    let qd = blocks_cpu.iter().zip(&blocks_gpu).filter(|(a, b)| a != b).count();
    if qd != 0 {
        let first = blocks_cpu.iter().zip(&blocks_gpu).position(|(a, b)| a != b).unwrap();
        eprintln!("QDIFF {label} N={n} bytes={qd} first block={} byte={}", first / BLOCK_BYTES, first % BLOCK_BYTES);
    }
    // CPU reference.
    let reference: Vec<f32> = if let Some(up) = u {
        let g = gemm_cpu(&prep(a), &x8, &ds, n);
        let uu = gemm_cpu(&prep(up), &x8, &ds, n);
        let h = silu_gpu(gpu, &g, &uu);
        // Host sanity for the GPU-evaluated SiLU (not the bit gate).
        let worst = h.iter().zip(g.iter().zip(&uu)).map(|(&hv, (&gv, &uv))| {
            let host = gv / (1.0 + (-gv).exp()) * uv;
            (hv.to_bits() as i64 - host.to_bits() as i64).unsigned_abs() as u32
        }).max().unwrap_or(0);
        tally.silu_host_max_ulp = tally.silu_host_max_ulp.max(worst);
        h
    } else {
        gemm_cpu(&prep(a), &x8, &ds, n)
    };
    let y0: Vec<f32> = if add { (0..n * m).map(|i| rand(i, salt ^ 0xADD0_BEEF) * 0.37 + 0.11).collect() }
                        else { vec![0.0; n * m] };
    let reference: Vec<f32> = if add { reference.iter().zip(&y0).map(|(r, o)| o + r).collect() } else { reference };
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload weights");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload up"));
    let ynew = gpu.upload_f32(&y0, &[n, m]).expect("new Y");
    let yrep = gpu.upload_f32(&y0, &[n, m]).expect("repeat Y");
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let xq = d_xq.buf.as_ptr() as *const c_void;
    let sym = i8_symbol(u.is_some(), add);
    launch(gpu, sym, I8_LDS, ap, up, xq, ynew.buf.as_ptr() as *const c_void, m, k, n, add);
    launch(gpu, sym, I8_LDS, ap, up, xq, yrep.buf.as_ptr() as *const c_void, m, k, n, add);
    gpu.hip.device_synchronize().expect("sync G1");
    let candidate = gpu.download_f32(&ynew).expect("download");
    let repeat = gpu.download_f32(&yrep).expect("download repeat");
    let differing = compare(label, &reference, &candidate, m);
    let repeated = compare(&format!("{label}/repeat"), &candidate, &repeat, m);
    let nonzero = candidate.iter().filter(|v| **v != 0.0).count();
    eprintln!("G1 {label} N={n} M={m} K={k} ADD={add} compared={} nonzero={nonzero} differing={differing} repeat_differing={repeated} quant_bytes_differing={qd}",
              candidate.len());
    tally.cases += 1;
    tally.outputs += candidate.len();
    tally.gemm_diffs += differing;
    tally.repeat_diffs += repeated;
    tally.quant_diffs += qd;
    for t in [ynew, yrep, d_x, d_xq, d_a] { gpu.free_tensor(t).expect("free"); }
    if let Some(t) = d_u { gpu.free_tensor(t).expect("free U"); }
}

fn median(xs: &mut [f64]) -> f64 {
    assert!(!xs.is_empty());
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn timed(gpu: &mut Gpu, out: &str, idx: usize, arm: &str, shape: &str, a: &Matrix, u: Option<&Matrix>,
         n: usize, add: bool) -> f64 {
    let (m, k, bytes) = a;
    let x: Vec<f32> = (0..n * k).map(|i| rand(i, 0x71e0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, *k]).expect("upload X");
    let (xq, d_xq, sym, lds) = if arm == "a8" {
        let t = quantize_gpu(gpu, &d_x, n, *k);
        (t.buf.as_ptr() as *const c_void, Some(t), i8_symbol(u.is_some(), add), I8_LDS)
    } else {
        let p = gpu.ensure_int4_mmq_x(&d_x, n, *k).expect("quantize X") as *const c_void;
        (p, None, v3_symbol(u.is_some(), add), V3_LDS)
    };
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload A");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload U"));
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let y0: Vec<f32> = (0..n * m).map(|i| rand(i, 0x5eed) * 0.25).collect();
    let y = gpu.upload_f32(&y0, &[n, *m]).expect("upload Y");
    let yp = y.buf.as_ptr() as *const c_void;
    for _ in 0..16 { launch(gpu, sym, lds, ap, up, xq, yp, *m, *k, n, add); }
    gpu.hip.device_synchronize().expect("warmup sync");
    let csv = format!("{out}/{idx:02}-{arm}-{shape}-{n}-telemetry.csv");
    let mut child = Command::new("python3").args(["-u", "-c", TELEMETRY, &csv, CARD_BDF])
        .stdout(Stdio::piped()).spawn().expect("start AMD-SMI sampler");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().expect("sampler stdout")).read_line(&mut ready).expect("sampler ready");
    assert_eq!(ready.trim(), "READY", "AMD-SMI sampler failed");
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < Duration::from_secs(10) {
        for _ in 0..16 { launch(gpu, sym, lds, ap, up, xq, yp, *m, *k, n, add); }
        gpu.hip.device_synchronize().expect("timed sync");
        calls += 16;
    }
    let elapsed = start.elapsed().as_secs_f64();
    child.kill().expect("stop sampler");
    child.wait().expect("reap sampler");
    let data = std::fs::read_to_string(&csv).expect("read telemetry");
    let (mut watts, mut clocks, mut busy, mut times) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in data.lines().skip(1) {
        let f: Vec<_> = row.splitn(5, ',').collect();
        assert_eq!(f.len(), 5, "telemetry row");
        assert!(f[4].is_empty(), "AMD-SMI error: {row}");
        times.push(f[0].parse::<u64>().expect("time"));
        watts.push(f[1].parse::<f64>().expect("watts"));
        clocks.push(f[2].parse::<f64>().expect("sclk"));
        busy.push(f[3].parse::<f64>().expect("busy"));
    }
    assert!(times.len() >= 180, "too few 20-Hz samples: {}", times.len());
    let mut gaps: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]) as f64 / 1e6).collect();
    let spacing = median(&mut gaps);
    assert!((40.0..=60.0).contains(&spacing), "telemetry spacing {spacing} ms");
    let us = elapsed * 1e6 / calls as f64;
    let mats = if u.is_some() { 2.0 } else { 1.0 };
    let tops = 2.0 * mats * (*m as f64) * (*k as f64) * n as f64 / (us * 1e6);
    eprintln!("G2 idx={idx} arm={arm} shape={shape} sym={sym} N={n} M={m} K={k} calls={calls} seconds={elapsed:.3} us_per_call={us:.3} TOPS={tops:.3} sclk_mhz={:.1} watts={:.1} busy_pct={:.1} samples={} spacing_ms={spacing:.1} csv={csv}",
              median(&mut clocks), median(&mut watts), median(&mut busy), times.len());
    for t in [d_x, d_a, y] { gpu.free_tensor(t).expect("free"); }
    if let Some(t) = d_xq { gpu.free_tensor(t).expect("free Xq"); }
    if let Some(t) = d_u { gpu.free_tensor(t).expect("free U"); }
    tops
}

fn main() {
    let model = std::env::args().nth(1).expect("usage: tmp_i8_gfx12_oracle <model.hfq> [--compile-only]");
    for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
        assert_eq!(std::env::var(key).as_deref(), Ok(CARD_UUID), "{key} must pin card E by UUID");
    }
    let mut src = String::new();
    if let Ok(defs) = std::env::var("I8_DEFINES") {
        for d in defs.split(',').filter(|d| !d.is_empty()) {
            let (name, value) = d.split_once('=').expect("I8_DEFINES entries are NAME=VALUE");
            src.push_str(&format!("#define {name} {value}\n"));
        }
    }
    src.push_str(I8_SRC);
    let mut gpu = Gpu::init().expect("initialize GPU");
    assert_eq!(gpu.arch, "gfx1201");
    for sym in [QUANT, i8_symbol(true, false), i8_symbol(false, true), i8_symbol(false, false)] {
        gpu.ensure_kernel_public(I8_MODULE, &src, sym).unwrap_or_else(|e| panic!("compile {sym}: {e}"));
        eprintln!("JIT OK {sym}");
    }
    for sym in [v3_symbol(true, false), v3_symbol(false, true), v3_symbol(false, false)] {
        gpu.ensure_kernel_public(V3_MODULE, V3_SRC, sym).unwrap_or_else(|e| panic!("compile {sym}: {e}"));
        eprintln!("JIT OK {sym}");
    }
    gpu.ensure_kernel_public(SILU_MODULE, SILU_SRC, "i8_oracle_silu_mul").expect("compile silu ref");
    if std::env::args().any(|a| a == "--compile-only") { return; }
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    assert!(hfq.mq4v2_symmetric(), "A8 requires symmetric qt44");

    if let Ok(seq) = std::env::var("TIME") {
        let n: usize = std::env::var("TIME_N").ok().map(|v| v.parse().expect("TIME_N")).unwrap_or(8192);
        let out = std::env::var("TIME_OUT").unwrap_or_else(|_| "/home/kaden/qcal/perf/fp8-4k5/a8/g2".into());
        std::fs::create_dir_all(&out).expect("create telemetry directory");
        let gate = weight(&hfq, 0, &["mlp.gate_proj"]);
        let up = weight(&hfq, 0, &["mlp.up_proj"]);
        let down = weight(&hfq, 0, &["mlp.down_proj"]);
        let set = weight(&hfq, 0, &["linear_attn.in_proj_qkv"]);
        for (idx, item) in seq.split(',').enumerate() {
            let (arm, shape) = item.split_once(':').expect("TIME items are arm:shape");
            assert!(matches!(arm, "a8" | "k1"), "arm {arm}");
            match shape {
                "gate" => timed(&mut gpu, &out, idx, arm, shape, &gate, Some(&up), n, false),
                "down" => timed(&mut gpu, &out, idx, arm, shape, &down, None, n, true),
                "set" => timed(&mut gpu, &out, idx, arm, shape, &set, None, n, false),
                other => panic!("shape {other}"),
            };
        }
        return;
    }

    let mut tally = Tally { cases: 0, outputs: 0, gemm_diffs: 0, repeat_diffs: 0, quant_diffs: 0, silu_host_max_ulp: 0 };
    let quick = std::env::var("QUICK").ok().as_deref() == Some("1");
    let ns: &[usize] = if quick { &[48, 512] } else { &[48, 80, 512, 8192] };
    let layers: &[usize] = if quick { &[0] } else { &[0, 31, 63] };
    for &layer in layers {
        let gate = weight(&hfq, layer, &["mlp.gate_proj"]);
        let up = weight(&hfq, layer, &["mlp.up_proj"]);
        for &n in ns {
            check(&mut gpu, &mut tally, &format!("gate-up/layer{layer}"), &gate, Some(&up), n, false, 0x7000 + layer as u32 + n as u32);
        }
    }
    let down = weight(&hfq, 0, &["mlp.down_proj"]);
    let qkvza = weight(&hfq, 0, &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b"]);
    let qkv = weight(&hfq, 31, &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]);
    let tail48 = weight(&hfq, 0, &["linear_attn.in_proj_a"]);
    let tail100 = (100, tail48.1, [tail48.2.as_slice(), &weight(&hfq, 0, &["linear_attn.in_proj_b"]).2, &tail48.2[..4 * (tail48.1 / 256) * 136]].concat());
    for &n in ns {
        check(&mut gpu, &mut tally, "down/set", &down, None, n, false, 0xD000 + n as u32);
        check(&mut gpu, &mut tally, "down/add", &down, None, n, true, 0xD100 + n as u32);
        if !quick {
            check(&mut gpu, &mut tally, "qkvza/set", &qkvza, None, n, false, 0xA000 + n as u32);
            check(&mut gpu, &mut tally, "qkv/set", &qkv, None, n, false, 0xB000 + n as u32);
        }
    }
    for (label, matrix) in [("m48tail/set", &tail48), ("m100tail/set", &tail100)] {
        for &n in if quick { &[48usize][..] } else { &[48usize, 80, 512][..] } {
            check(&mut gpu, &mut tally, label, matrix, None, n, false, 0xE000 + n as u32);
        }
    }
    eprintln!("SUMMARY cases={} outputs_compared={} gemm_differing={} repeat_differing={} quant_bytes_differing={} silu_gpu_vs_host_max_ulp={}",
              tally.cases, tally.outputs, tally.gemm_diffs, tally.repeat_diffs, tally.quant_diffs, tally.silu_host_max_ulp);
    if tally.gemm_diffs + tally.repeat_diffs + tally.quant_diffs != 0 {
        eprintln!("GFX12-I8 ORACLE FAIL");
        std::process::exit(1);
    }
    eprintln!("GFX12-I8 ORACLE PASS");
}
