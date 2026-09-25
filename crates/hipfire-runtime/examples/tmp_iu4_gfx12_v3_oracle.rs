//! GPU-vs-GPU bit oracle for gfx1201 builder kernels against hipcc K1.
//! Pin HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to card A's UUID.
//! TIME=1 runs sustained ABBA measurements with 20-Hz AMD-SMI telemetry.
//!
//! `HIPFIRE_ISA_MODULE=<path>` loads the candidate module; the reference is
//! hipcc `_v3` JIT-compiled from source. `HIPFIRE_ISA_SYMBOL_SUFFIX` selects
//! candidate symbols, and `_b1t256` uses a 256-row/512-thread/30,720-B tile.
//! `HIPFIRE_ISA_TILE_ROWS=128|256` can override the suffix-derived geometry.
//! Timed arm labels include the module SHA-256.

use hip_bridge::{Function, KernargBlob, Module};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::c_void;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const V3: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_v3";
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
         if amdsmi.amdsmi_get_gpu_device_bdf(h) == '0000:03:00.0')
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
const OUT: &str = "/home/kaden/qcal/perf/iu4-6k/s2";
const MODEL_PREFIX: &str = "model.language_model.layers.";

type Matrix = (usize, usize, Vec<u8>);

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    block: u32,
    lds: u32,
}

const K1_GEOMETRY: Geometry = Geometry { rows: 128, block: 256, lds: 20480 };

/// Candidate kernel set: either JIT `_v3` (default) or a path-loaded module.
struct Candidate {
    suffix: String,
    geometry: Geometry,
    /// TIME=1 arm label: `"v3"` or `"isa-<sha256>"`.
    arm: String,
    /// Path-loaded functions keyed by full symbol name. Empty when JIT path.
    funcs: HashMap<String, Function>,
    /// Keeps `Module` alive so `Function` handles remain valid.
    _module: Option<Module>,
}

impl Candidate {
    fn func(&self, name: &str) -> Option<&Function> {
        self.funcs.get(name)
    }
}

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

fn cand_symbol(fused: bool, add: bool, suffix: &str) -> String {
    match (fused, add) {
        (true, _) => format!("gemm_mq4g256v2_gate_up_silu_mmq_iu4{suffix}"),
        (false, true) => format!("gemm_mq4g256v2_residual_mmq_iu4_full_add{suffix}"),
        (false, false) => format!("gemm_mq4g256v2_residual_mmq_iu4_full_set{suffix}"),
    }
}

fn launch(
    gpu: &mut Gpu,
    func: Option<&Function>,
    sym: &str,
    a: *const c_void,
    u: Option<*const c_void>,
    xq: *const c_void,
    y: *const c_void,
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    geometry: Geometry,
) {
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
    let grid = [rows.div_ceil(geometry.rows) as u32, n.div_ceil(128) as u32, 1];
    let block = [geometry.block, 1, 1];
    let shared = geometry.lds;
    match func {
        Some(f) => unsafe {
            gpu.hip
                .launch_kernel_blob(f, grid, block, shared, None, &mut blob)
                .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
        },
        None => {
            gpu.launch_kernel_blob(sym, grid, block, shared, &mut blob)
                .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
        }
    }
}

fn compare(label: &str, left: &[f32], right: &[f32], m: usize) -> usize {
    assert_eq!(left.len(), right.len());
    let mut diffs = 0;
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        if a.to_bits() != b.to_bits() {
            if diffs < 8 {
                eprintln!(
                    "DIFF {label} n={} row={} ref={:08x} v3={:08x} ({a:?} {b:?})",
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

fn check(
    gpu: &mut Gpu,
    cand: &Candidate,
    label: &str,
    a: &Matrix,
    u: Option<&Matrix>,
    n: usize,
    add: bool,
    salt: u32,
) -> usize {
    let (m, k, bytes) = a;
    if let Some((um, uk, _)) = u {
        assert_eq!((*m, *k), (*um, *uk));
    }
    let x: Vec<f32> = (0..n * k)
        .map(|i| rand(i, salt) * (0.75 + (i / k % 5) as f32 * 0.15))
        .collect();
    let d_x = gpu.upload_f32(&x, &[n, *k]).expect("upload X");
    let xq = gpu.ensure_int4_mmq_x(&d_x, n, *k).expect("quantize X") as *const c_void;
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload weights");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload up"));
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let y0: Vec<f32> = if add {
        (0..n * m)
            .map(|i| rand(i, salt ^ 0xADD0_BEEF) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0; n * m]
    };
    let yref = gpu.upload_f32(&y0, &[n, *m]).expect("ref Y");
    let ynew = gpu.upload_f32(&y0, &[n, *m]).expect("new Y");
    let yrep = gpu.upload_f32(&y0, &[n, *m]).expect("repeat Y");
    let rp = yref.buf.as_ptr() as *const c_void;
    let np = ynew.buf.as_ptr() as *const c_void;
    let pp = yrep.buf.as_ptr() as *const c_void;
    let fused = u.is_some();
    let cname = cand_symbol(fused, add, &cand.suffix);
    let cfunc = cand.func(&cname);
    let reference_symbol = cand_symbol(fused, add, "_v3");
    launch(gpu, None, &reference_symbol, ap, up, xq, rp, *m, *k, n, add, K1_GEOMETRY);
    launch(gpu, cfunc, &cname, ap, up, xq, np, *m, *k, n, add, cand.geometry);
    launch(gpu, cfunc, &cname, ap, up, xq, pp, *m, *k, n, add, cand.geometry);
    gpu.hip.device_synchronize().expect("sync G1");
    let reference = gpu.download_f32(&yref).expect("download K1");
    let candidate = gpu.download_f32(&ynew).expect("download builder");
    let repeat = gpu.download_f32(&yrep).expect("download repeat");
    let differing = compare(label, &reference, &candidate, *m);
    let repeated = compare(&format!("{label}/repeat"), &candidate, &repeat, *m);
    eprintln!(
        "G1 {label} N={n} M={m} K={k} ADD={add} compared={} differing={differing} repeat_differing={repeated}",
        candidate.len()
    );
    gpu.free_tensor(yref).expect("free ref");
    gpu.free_tensor(ynew).expect("free candidate");
    gpu.free_tensor(yrep).expect("free repeat");
    gpu.free_tensor(d_x).expect("free X");
    gpu.free_tensor(d_a).expect("free A");
    if let Some(u) = d_u {
        gpu.free_tensor(u).expect("free U");
    }
    differing + repeated
}

fn median(xs: &mut [f64]) -> f64 {
    assert!(!xs.is_empty());
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn timed(
    gpu: &mut Gpu,
    cand: &Candidate,
    label: &str,
    a: &Matrix,
    u: Option<&Matrix>,
    n: usize,
    candidate: bool,
    add: bool,
    repetition: usize,
) -> (f64, f64, f64) {
    let (m, k, bytes) = a;
    let x: Vec<f32> = (0..n * k).map(|i| rand(i, 0x71e0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, *k]).expect("upload X");
    let xq = gpu.ensure_int4_mmq_x(&d_x, n, *k).expect("quantize X") as *const c_void;
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload A");
    let d_u = u.map(|(_, _, b)| gpu.upload_raw(b, &[b.len()]).expect("upload U"));
    let ap = d_a.buf.as_ptr() as *const c_void;
    let up = d_u.as_ref().map(|t| t.buf.as_ptr() as *const c_void);
    let y0 = if add { vec![0.11f32; n * m] } else { vec![0.0f32; n * m] };
    let y = gpu.upload_f32(&y0, &[n, *m]).expect("upload Y");
    let yp = y.buf.as_ptr() as *const c_void;
    let fused = u.is_some();
    let (sym_owned, func, geometry): (String, Option<&Function>, Geometry) = if candidate {
        let name = cand_symbol(fused, add, &cand.suffix);
        let f = cand.func(&name);
        (name, f, cand.geometry)
    } else {
        (cand_symbol(fused, add, "_v3"), None, K1_GEOMETRY)
    };
    let sym = sym_owned.as_str();
    for _ in 0..16 {
        launch(gpu, func, sym, ap, up, xq, yp, *m, *k, n, add, geometry);
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let arm = if candidate { cand.arm.as_str() } else { "k1" };
    let csv = format!("{OUT}/{label}-{n}-{}-{arm}-{repetition}-telemetry.csv", cand.suffix);
    let mut child = Command::new("python3")
        .args(["-u", "-c", TELEMETRY, &csv])
        .stdout(Stdio::piped())
        .spawn()
        .expect("start AMD-SMI sampler");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().expect("sampler stdout"))
        .read_line(&mut ready)
        .expect("sampler ready");
    assert_eq!(ready.trim(), "READY", "AMD-SMI sampler failed");
    eprintln!("BEGIN {label} N={n} arm={arm} repetition={repetition}");
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < Duration::from_secs(10) {
        for _ in 0..32 {
            launch(gpu, func, sym, ap, up, xq, yp, *m, *k, n, add, geometry);
        }
        gpu.hip.device_synchronize().expect("timed sync");
        calls += 32;
    }
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("END {label} N={n} arm={arm} repetition={repetition}");
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
    assert!(
        sample_times.len() >= 180,
        "too few 20-Hz samples: {}",
        sample_times.len()
    );
    let mut intervals: Vec<f64> = sample_times
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / 1e6)
        .collect();
    let spacing = median(&mut intervals);
    assert!(
        (40.0..=60.0).contains(&spacing),
        "telemetry median spacing {spacing}ms"
    );
    let us = elapsed * 1e6 / calls as f64;
    let tops = 2.0 * (if u.is_some() { 2.0 } else { 1.0 }) * (*m as f64) * (*k as f64) * n as f64
        / (us * 1e6);
    let clock = median(&mut clocks);
    let power = median(&mut watts);
    eprintln!(
        "G2 {label} N={n} arm={arm} repetition={repetition} calls={calls} seconds={elapsed:.3} us/call={us:.3} TOPS={tops:.3} sclk_mhz={clock:.1} watts={power:.1} busy_pct={:.1} samples={} spacing_ms={spacing:.1} csv={csv}",
        median(&mut busy),
        sample_times.len()
    );
    gpu.free_tensor(d_x).expect("free X");
    gpu.free_tensor(d_a).expect("free A");
    if let Some(u) = d_u {
        gpu.free_tensor(u).expect("free U");
    }
    gpu.free_tensor(y).expect("free Y");
    (tops, clock, power)
}

fn load_candidate(gpu: &mut Gpu) -> Candidate {
    let suffix = std::env::var("HIPFIRE_ISA_SYMBOL_SUFFIX").unwrap_or_else(|_| "_v3".to_string());
    let path = std::env::var("HIPFIRE_ISA_MODULE")
        .ok()
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty());

    let rows = std::env::var("HIPFIRE_ISA_TILE_ROWS")
        .ok()
        .map(|value| value.parse::<usize>().expect("HIPFIRE_ISA_TILE_ROWS integer"))
        .unwrap_or(if suffix.ends_with("t256") { 256 } else { 128 });
    let geometry = match rows {
        128 => K1_GEOMETRY,
        256 => Geometry { rows: 256, block: 512, lds: 30720 },
        _ => panic!("HIPFIRE_ISA_TILE_ROWS must be 128 or 256"),
    };
    eprintln!("CANDIDATE GEOMETRY rows={} block={} lds={}", geometry.rows, geometry.block, geometry.lds);
    for fused in [false, true] {
        for add in if fused { &[false][..] } else { &[false, true][..] } {
            let sym = cand_symbol(fused, *add, "_v3");
            gpu.ensure_kernel_public(V3, V3_SRC, &sym)
                .unwrap_or_else(|e| panic!("compile {sym}: {e}"));
            eprintln!("K1 JIT OK {sym}");
        }
    }

    if let Some(path) = path {
        let path_str = path
            .to_str()
            .unwrap_or_else(|| panic!("HIPFIRE_ISA_MODULE path is not UTF-8: {}", path.display()));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "HIPFIRE_ISA_MODULE read {}: {e}",
                path.display()
            )
        });
        let digest = Sha256::digest(&bytes);
        let arm = format!("isa-{digest:x}");
        eprintln!(
            "ISA MODULE path={} bytes={} sha256={:x}",
            path.display(),
            bytes.len(),
            digest
        );
        let module = gpu
            .hip
            .module_load(path_str)
            .unwrap_or_else(|e| panic!("hipModuleLoad {}: {e}", path.display()));
        let mut funcs = HashMap::new();
        for fused in [false, true] {
            for add in if fused {
                &[false][..]
            } else {
                &[false, true][..]
            } {
                let sym = cand_symbol(fused, *add, &suffix);
                let func = gpu
                    .hip
                    .module_get_function(&module, &sym)
                    .unwrap_or_else(|e| panic!("module_get_function {sym}: {e}"));
                eprintln!("ISA OK {sym}");
                funcs.insert(sym, func);
            }
        }
        Candidate {
            suffix,
            geometry,
            arm,
            funcs,
            _module: Some(module),
        }
    } else {
        for fused in [false, true] {
            for add in if fused {
                &[false][..]
            } else {
                &[false, true][..]
            } {
                let sym = cand_symbol(fused, *add, &suffix);
                gpu.ensure_kernel_public(V3, V3_SRC, &sym)
                    .unwrap_or_else(|e| panic!("compile {sym}: {e}"));
                eprintln!("JIT OK {sym}");
            }
        }
        Candidate {
            suffix,
            geometry,
            arm: "v3".to_string(),
            funcs: HashMap::new(),
            _module: None,
        }
    }
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: tmp_iu4_gfx12_v3_oracle <model.hfq> [--compile-only]");
    let uuid = "GPU-9eb7aeda51c88ffd";
    for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
        assert_eq!(
            std::env::var(key).as_deref(),
            Ok(uuid),
            "{key} must pin card A by UUID"
        );
    }
    let mut gpu = Gpu::init().expect("initialize GPU");
    assert_eq!(gpu.arch, "gfx1201");
    let cand = load_candidate(&mut gpu);
    if std::env::args().any(|a| a == "--compile-only") {
        return;
    }
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    assert!(hfq.mq4v2_symmetric(), "K1 requires symmetric qt44");
    if std::env::var("OCC").ok().as_deref() == Some("1") {
        for fused in [false, true] {
            let k1 = cand_symbol(fused, false, "_v3");
            eprintln!(
                "OCC {k1} active_blocks_per_CU={}",
                gpu.occupancy_max_active_blocks(&k1, [256, 1, 1], 20480)
                    .expect("occupancy K1")
            );
            let cname = cand_symbol(fused, false, &cand.suffix);
            let blocks = if let Some(f) = cand.func(&cname) {
                gpu.hip
                    .occupancy_max_active_blocks(f, cand.geometry.block, cand.geometry.lds as usize)
                    .expect("occupancy cand")
            } else {
                gpu.occupancy_max_active_blocks(&cname, [cand.geometry.block, 1, 1], cand.geometry.lds)
                    .expect("occupancy cand")
            };
            eprintln!("OCC {cname} active_blocks_per_CU={blocks}");
        }
        return;
    }
    if std::env::var("TIME").ok().as_deref() == Some("1") {
        std::fs::create_dir_all(OUT).expect("create telemetry directory");
        let gate = weight(&hfq, 0, &["mlp.gate_proj"]);
        let up = weight(&hfq, 0, &["mlp.up_proj"]);
        let down = weight(&hfq, 0, &["mlp.down_proj"]);
        let set = weight(&hfq, 0, &["linear_attn.in_proj_qkv"]);
        for n in [8192, 512] {
            for (label, matrix, other, add) in [
                ("gate", &gate, Some(&up), false),
                ("down", &down, None, true),
                ("set", &set, None, false),
            ] {
                let a = timed(&mut gpu, &cand, label, matrix, other, n, false, add, 0);
                let b = timed(&mut gpu, &cand, label, matrix, other, n, true, add, 1);
                let b2 = timed(&mut gpu, &cand, label, matrix, other, n, true, add, 2);
                let a2 = timed(&mut gpu, &cand, label, matrix, other, n, false, add, 3);
                let k1 = (a.0 + a2.0) / 2.0;
                let builder = (b.0 + b2.0) / 2.0;
                let gain = (builder / k1 - 1.0) * 100.0;
                let threshold = if n == 512 { -3.0 } else if label == "set" { 3.0 } else { 4.0 };
                eprintln!(
                    "G2_COMPARE {label} N={n} K1_TOPS={k1:.3} builder_TOPS={builder:.3} gain_pct={gain:.2} K1_sclk_mhz={:.1} builder_sclk_mhz={:.1} K1_watts={:.1} builder_watts={:.1} threshold_pct={threshold:.1} {}",
                    (a.1 + a2.1) / 2.0, (b.1 + b2.1) / 2.0,
                    (a.2 + a2.2) / 2.0, (b.2 + b2.2) / 2.0,
                    if gain >= threshold { "PASS" } else { "FAIL" }
                );
            }
        }
        return;
    }
    let mut diffs = 0usize;
    for layer in [0, 31, 63] {
        let gate = weight(&hfq, layer, &["mlp.gate_proj"]);
        let up = weight(&hfq, layer, &["mlp.up_proj"]);
        for n in [48, 80, 512, 8192] {
            diffs += check(
                &mut gpu,
                &cand,
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
    for n in [48, 80, 512, 8192] {
        diffs += check(
            &mut gpu,
            &cand,
            "down/set",
            &down,
            None,
            n,
            false,
            0xD000 + n as u32,
        );
        diffs += check(
            &mut gpu,
            &cand,
            "down/add",
            &down,
            None,
            n,
            true,
            0xD100 + n as u32,
        );
        diffs += check(
            &mut gpu,
            &cand,
            "qkvza/set",
            &qkvza,
            None,
            n,
            false,
            0xA000 + n as u32,
        );
        diffs += check(
            &mut gpu,
            &cand,
            "qkv/set",
            &qkv,
            None,
            n,
            false,
            0xB000 + n as u32,
        );
    }
    for (label, matrix) in [("m48tail/set", &tail48), ("m100tail/set", &tail100)] {
        for n in [48, 80, 512] {
            diffs += check(
                &mut gpu,
                &cand,
                label,
                matrix,
                None,
                n,
                false,
                0xE000 + n as u32,
            );
        }
    }
    if diffs != 0 {
        eprintln!("GFX12-IU4-V3 ORACLE FAIL differing_bits={diffs}");
        std::process::exit(1);
    }
    eprintln!("GFX12-IU4-V3 ORACLE PASS");
}
