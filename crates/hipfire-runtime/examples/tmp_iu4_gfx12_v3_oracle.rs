//! GPU-vs-GPU bit oracle for gfx1201 K1 against the shipping symmetric raster kernel.
//! Run with HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES both set to card C's UUID.
//! TIME=1 runs sustained paired measurements and writes 20-Hz AMD-SMI CSVs to
//! /home/kaden/qcal/perf/gemm-v3/k1/.

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const BASE: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_symfold_g12r";
const V3: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_v3";
const SHIP_SRC: &str = concat!(
    "#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n",
    "#define gemm_mq4g256v2_residual_mmq_iu4 gemm_mq4g256v2_residual_mmq_iu4_symfold_g12r\n",
    "#define gemm_mq4g256v2_residual_mmq_iu4_full_add gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold_g12r\n",
    "#define gemm_mq4g256v2_residual_mmq_iu4_full_set gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold_g12r\n",
    "#define gemm_mq4g256v2_gate_up_silu_mmq_iu4 gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold_g12r\n",
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip")
);
const V3_SRC: &str = concat!(
    "#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n",
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip")
);
const TELEMETRY: &str = r#"
import csv, os, sys, time
os.environ['AMDSMI_GPU_METRICS_CACHE_MS'] = '50'
sys.path.insert(0, '/opt/rocm/core/share/amd_smi')
import amdsmi
amdsmi.amdsmi_init()
h = next(h for h in amdsmi.amdsmi_get_processor_handles()
         if amdsmi.amdsmi_get_gpu_device_bdf(h) == '0000:e3:00.0')
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
const OUT: &str = "/home/kaden/qcal/perf/gemm-v3/k1";
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

fn symbol(v3: bool, fused: bool, add: bool) -> &'static str {
    match (v3, fused, add) {
        (false, true, _) => "gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold_g12r",
        (true, true, _) => "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3",
        (false, false, true) => "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold_g12r",
        (false, false, false) => "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold_g12r",
        (true, false, true) => "gemm_mq4g256v2_residual_mmq_iu4_full_add_v3",
        (true, false, false) => "gemm_mq4g256v2_residual_mmq_iu4_full_set_v3",
    }
}

fn launch(gpu: &mut Gpu, sym: &str, a: *const c_void, u: Option<*const c_void>, xq: *const c_void,
          y: *const c_void, m: usize, k: usize, n: usize, add: bool) {
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
                           [256, 1, 1], 20480, &mut blob)
        .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

fn compare(label: &str, left: &[f32], right: &[f32], m: usize) -> usize {
    assert_eq!(left.len(), right.len());
    let mut diffs = 0;
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        if a.to_bits() != b.to_bits() {
            if diffs < 8 {
                eprintln!("DIFF {label} n={} row={} ref={:08x} v3={:08x} ({a:?} {b:?})",
                          i / m, i % m, a.to_bits(), b.to_bits());
            }
            diffs += 1;
        }
    }
    diffs
}

fn check(gpu: &mut Gpu, label: &str, a: &Matrix, u: Option<&Matrix>, n: usize, add: bool, salt: u32) -> usize {
    let (m, k, bytes) = a;
    if let Some((um, uk, _)) = u { assert_eq!((*m, *k), (*um, *uk)); }
    let x: Vec<f32> = (0..n * k).map(|i| rand(i, salt) * (0.75 + (i / k % 5) as f32 * 0.15)).collect();
    let d_x = gpu.upload_f32(&x, &[n, *k]).expect("upload X");
    let xq = gpu.ensure_int4_mmq_x(&d_x, n, *k).expect("quantize X") as *const c_void;
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload weights");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload up"));
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let y0: Vec<f32> = if add { (0..n * m).map(|i| rand(i, salt ^ 0xADD0_BEEF) * 0.37 + 0.11).collect() }
                         else { vec![0.0; n * m] };
    let yref = gpu.upload_f32(&y0, &[n, *m]).expect("ref Y");
    let ynew = gpu.upload_f32(&y0, &[n, *m]).expect("new Y");
    let yrep = gpu.upload_f32(&y0, &[n, *m]).expect("repeat Y");
    let rp = yref.buf.as_ptr() as *const c_void;
    let np = ynew.buf.as_ptr() as *const c_void;
    let pp = yrep.buf.as_ptr() as *const c_void;
    launch(gpu, symbol(false, u.is_some(), add), ap, up, xq, rp, *m, *k, n, add);
    launch(gpu, symbol(true, u.is_some(), add), ap, up, xq, np, *m, *k, n, add);
    launch(gpu, symbol(true, u.is_some(), add), ap, up, xq, pp, *m, *k, n, add);
    gpu.hip.device_synchronize().expect("sync G1");
    let reference = gpu.download_f32(&yref).expect("download shipping");
    let candidate = gpu.download_f32(&ynew).expect("download v3");
    let repeat = gpu.download_f32(&yrep).expect("download repeat");
    let differing = compare(label, &reference, &candidate, *m);
    let repeated = compare(&format!("{label}/repeat"), &candidate, &repeat, *m);
    eprintln!("G1 {label} N={n} M={m} K={k} ADD={add} compared={} differing={differing} repeat_differing={repeated}", candidate.len());
    gpu.free_tensor(yref).expect("free ref");
    gpu.free_tensor(ynew).expect("free candidate");
    gpu.free_tensor(yrep).expect("free repeat");
    gpu.free_tensor(d_x).expect("free X");
    gpu.free_tensor(d_a).expect("free A");
    if let Some(u) = d_u { gpu.free_tensor(u).expect("free U"); }
    differing + repeated
}

fn median(xs: &mut [f64]) -> f64 {
    assert!(!xs.is_empty());
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn timed(gpu: &mut Gpu, label: &str, a: &Matrix, u: Option<&Matrix>, n: usize, v3: bool) -> f64 {
    let (m, k, bytes) = a;
    let x: Vec<f32> = (0..n * k).map(|i| rand(i, 0x71e0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, *k]).expect("upload X");
    let xq = gpu.ensure_int4_mmq_x(&d_x, n, *k).expect("quantize X") as *const c_void;
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload A");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload U"));
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let y0 = vec![0.0f32; n * m];
    let y = gpu.upload_f32(&y0, &[n, *m]).expect("upload Y");
    let yp = y.buf.as_ptr() as *const c_void;
    let sym = symbol(v3, u.is_some(), false);
    for _ in 0..16 { launch(gpu, sym, ap, up, xq, yp, *m, *k, n, false); }
    gpu.hip.device_synchronize().expect("warmup sync");
    let arm = if v3 { "v3" } else { "ship" };
    let csv = format!("{OUT}/{label}-{n}-{arm}-telemetry.csv");
    let mut child = Command::new("python3").args(["-u", "-c", TELEMETRY, &csv])
        .stdout(Stdio::piped()).spawn().expect("start AMD-SMI sampler");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().expect("sampler stdout")).read_line(&mut ready).expect("sampler ready");
    assert_eq!(ready.trim(), "READY", "AMD-SMI sampler failed");
    eprintln!("BEGIN {label} N={n} arm={arm}");
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < Duration::from_secs(10) {
        for _ in 0..32 { launch(gpu, sym, ap, up, xq, yp, *m, *k, n, false); }
        gpu.hip.device_synchronize().expect("timed sync");
        calls += 32;
    }
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("END {label} N={n} arm={arm}");
    child.kill().expect("stop sampler");
    child.wait().expect("reap sampler");
    let data = std::fs::read_to_string(&csv).expect("read telemetry");
    let mut watts = Vec::new();
    let mut clocks = Vec::new();
    let mut busy = Vec::new();
    let mut sample_times = Vec::new();
    for row in data.lines().skip(1) {
        let fields: Vec<_> = row.splitn(5, ',').collect();
        assert_eq!(fields.len(), 5, "telemetry row");
        assert!(fields[4].is_empty(), "AMD-SMI error: {row}");
        sample_times.push(fields[0].parse::<u64>().expect("sample time"));
        watts.push(fields[1].parse::<f64>().expect("watts"));
        clocks.push(fields[2].parse::<f64>().expect("sclk"));
        busy.push(fields[3].parse::<f64>().expect("busy"));
    }
    assert!(sample_times.len() >= 180, "too few 20-Hz samples: {}", sample_times.len());
    let mut intervals: Vec<f64> = sample_times.windows(2).map(|w| (w[1] - w[0]) as f64 / 1e6).collect();
    let spacing = median(&mut intervals);
    assert!((40.0..=60.0).contains(&spacing), "telemetry median spacing {spacing}ms");
    let us = elapsed * 1e6 / calls as f64;
    let tops = 2.0 * (if u.is_some() { 2.0 } else { 1.0 }) * (*m as f64) * (*k as f64) * n as f64 / (us * 1e6);
    eprintln!("G2 {label} N={n} arm={arm} calls={calls} seconds={elapsed:.3} us/call={us:.3} TOPS={tops:.3} sclk_mhz={:.1} watts={:.1} busy_pct={:.1} samples={} spacing_ms={spacing:.1} csv={csv}",
              median(&mut clocks), median(&mut watts), median(&mut busy), sample_times.len());
    gpu.free_tensor(d_x).expect("free X");
    gpu.free_tensor(d_a).expect("free A");
    if let Some(u) = d_u { gpu.free_tensor(u).expect("free U"); }
    gpu.free_tensor(y).expect("free Y");
    tops
}

fn main() {
    let model = std::env::args().nth(1).expect("usage: tmp_iu4_gfx12_v3_oracle <model.hfq> [--compile-only]");
    let uuid = "GPU-085289909a86cc63";
    for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
        assert_eq!(std::env::var(key).as_deref(), Ok(uuid), "{key} must pin card C by UUID");
    }
    let mut gpu = Gpu::init().expect("initialize GPU");
    assert_eq!(gpu.arch, "gfx1201");
    for v3 in [false, true] {
        for fused in [false, true] {
            for add in if fused { &[false][..] } else { &[false, true][..] } {
                let sym = symbol(v3, fused, *add);
                gpu.ensure_kernel_public(if v3 { V3 } else { BASE }, if v3 { V3_SRC } else { SHIP_SRC }, sym)
                    .unwrap_or_else(|e| panic!("compile {sym}: {e}"));
                eprintln!("JIT OK {sym}");
            }
        }
    }
    if std::env::args().any(|a| a == "--compile-only") { return; }
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    assert!(hfq.mq4v2_symmetric(), "K1 requires symmetric qt44");
    if std::env::var("OCC").ok().as_deref() == Some("1") {
        for v3 in [false, true] {
            for fused in [false, true] {
                let sym = symbol(v3, fused, false);
                eprintln!("OCC {sym} active_blocks_per_CU={}", gpu.occupancy_max_active_blocks(sym, [256, 1, 1], 20480).expect("occupancy"));
            }
        }
        return;
    }
    if std::env::var("TIME").ok().as_deref() == Some("1") {
        std::fs::create_dir_all(OUT).expect("create telemetry directory");
        let gate = weight(&hfq, 0, &["mlp.gate_proj"]);
        let up = weight(&hfq, 0, &["mlp.up_proj"]);
        let down = weight(&hfq, 0, &["mlp.down_proj"]);
        let set = weight(&hfq, 0, &["linear_attn.in_proj_qkv"]);
        for n in [512, 8192] {
            for (label, matrix, other) in [("gate", &gate, Some(&up)), ("down", &down, None), ("set", &set, None)] {
                let ship = timed(&mut gpu, label, matrix, other, n, false);
                let v3 = timed(&mut gpu, label, matrix, other, n, true);
                let gain = (v3 / ship - 1.0) * 100.0;
                eprintln!("G2_COMPARE {label} N={n} shipping_TOPS={ship:.3} v3_TOPS={v3:.3} gain_pct={gain:.2} {}",
                          if (n == 8192 && gain < if label == "set" { 10.0 } else { 15.0 }) || (n == 512 && gain < -3.0) { "STOP" } else { "PASS" });
            }
        }
        return;
    }
    let mut diffs = 0usize;
    for layer in [0, 31, 63] {
        let gate = weight(&hfq, layer, &["mlp.gate_proj"]);
        let up = weight(&hfq, layer, &["mlp.up_proj"]);
        for n in [48, 80, 512, 8192] {
            diffs += check(&mut gpu, &format!("gate-up/layer{layer}"), &gate, Some(&up), n, false, 0x7000 + layer as u32 + n as u32);
        }
    }
    let down = weight(&hfq, 0, &["mlp.down_proj"]);
    let qkvza = weight(&hfq, 0, &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b"]);
    let qkv = weight(&hfq, 31, &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]);
    let tail48 = weight(&hfq, 0, &["linear_attn.in_proj_a"]);
    let tail100 = (100, tail48.1, [tail48.2.as_slice(), &weight(&hfq, 0, &["linear_attn.in_proj_b"]).2, &tail48.2[..4 * (tail48.1 / 256) * 136]].concat());
    for n in [48, 80, 512, 8192] {
        diffs += check(&mut gpu, "down/set", &down, None, n, false, 0xD000 + n as u32);
        diffs += check(&mut gpu, "down/add", &down, None, n, true, 0xD100 + n as u32);
        diffs += check(&mut gpu, "qkvza/set", &qkvza, None, n, false, 0xA000 + n as u32);
        diffs += check(&mut gpu, "qkv/set", &qkv, None, n, false, 0xB000 + n as u32);
    }
    for (label, matrix) in [("m48tail/set", &tail48), ("m100tail/set", &tail100)] {
        for n in [48, 80, 512] {
            diffs += check(&mut gpu, label, matrix, None, n, false, 0xE000 + n as u32);
        }
    }
    if diffs != 0 { eprintln!("GFX12-IU4-V3 ORACLE FAIL differing_bits={diffs}"); std::process::exit(1); }
    eprintln!("GFX12-IU4-V3 ORACLE PASS");
}
