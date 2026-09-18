//! Throwaway gfx12 iu4 oracle: candidate K32 iu4-direct MMQ vs CPU reference.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, quantizes X via `ensure_int4_mmq_x`, and raw-launches the
//! gfx12 `gemm_mq4g256v2_residual_mmq_iu4_full_{set,add}` symbols (grid
//! [M/16, N/16], block [32,1,1], LDS 0) on identical inputs. Checks:
//!   1. bitwise equality vs a CPU f32 reference with the pinned IU4_FOLD_RN
//!      DAG (full for small cases, edge stripes for the big real ones);
//!   2. bit-identical repeat (same inputs twice);
//!   3. vs the shipped fp8 route (`gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8`)
//!      with max-abs/mean-abs/p99.9 stats (no gate — contract is KLD).
//! Also covers phase edges with synthetic MQ4G256V2 data (M=48 tail rows,
//! N=80 partial cols, K=256).
//!
//! gfx1201 (ordinal 0):
//!   HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 \
//!     HIPFIRE_KERNEL_CACHE=$HOME/.hipfire_kernels/gfx1201 \
//!     cargo run --release -p hipfire-runtime --example tmp_iu4_gfx12_oracle -- \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const IU4_GFX12_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip")
);
const MODULE: &str = "tmp_iu4_gfx12_oracle";
const SET_SYM: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set";
const ADD_SYM: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add";

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

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

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
        return s;
    }
    s | ((e16 as u16) << 10) | ((m >> 13) as u16)
}

fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as u32;
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as u32;
    let bits = if e == 0x1f {
        (s << 31) | (0xff << 23) | (m << 13)
    } else if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
            let mut mm = m;
            let mut ee = -14i32;
            while mm & 0x400 == 0 {
                mm <<= 1;
                ee -= 1;
            }
            mm &= 0x3ff;
            (s << 31) | (((ee + 127) as u32) << 23) | (mm << 13)
        }
    } else {
        (s << 31) | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

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

fn dtoh_raw(gpu: &Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
    let view = unsafe { DeviceBuffer::from_raw(ptr, len) };
    let mut host = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut host, &view)
        .expect("memcpy_dtoh scratch snapshot");
    let _ = view.is_borrowed();
    host
}

fn launch_iu4(
    gpu: &mut Gpu,
    sym: &str,
    a_ptr: *const std::ffi::c_void,
    xq_ptr: *const std::ffi::c_void,
    y_ptr: *const std::ffi::c_void,
    m: usize,
    k: usize,
    n: usize,
    add: bool,
) {
    let mut b = KernargBlob::new();
    b.push_ptr(a_ptr);
    b.push_ptr(xq_ptr);
    b.push_ptr(y_ptr);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    b.push_i32(i32::from(add));
    let mut blob = b.into_vec();
    let grid = [m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1];
    gpu.launch_kernel_blob(sym, grid, [32, 1, 1], 0, &mut blob)
        .unwrap_or_else(|e| panic!("launch {sym}: {e}"));
}

/// CPU reference with the pinned IU4_FOLD_RN DAG. `rows: None` = all rows,
/// `Some(&[...])` = only those rows (edge stripes for big cases).
fn cpu_ref(
    a_bytes: &[u8],
    xq: &[u8],
    y0: &[f32],
    m: usize,
    k: usize,
    n: usize,
    rows: Option<&[usize]>,
) -> Vec<f32> {
    let gpr = k / 256;
    let check_row = |r: usize| rows.map(|rs| rs.contains(&r)).unwrap_or(true);
    let mut y = y0.to_vec();
    for col in 0..n {
        for r in 0..m {
            if !check_row(r) {
                continue;
            }
            let mut acc = 0.0f32;
            for g in 0..gpr {
                let gp = (r * gpr + g) * 136;
                let sc0 = f16_to_f32(u16::from_le_bytes([a_bytes[gp], a_bytes[gp + 1]]));
                let zp0 = f16_to_f32(u16::from_le_bytes([a_bytes[gp + 2], a_bytes[gp + 3]]));
                let sc1 = f16_to_f32(u16::from_le_bytes([a_bytes[gp + 4], a_bytes[gp + 5]]));
                let zp1 = f16_to_f32(u16::from_le_bytes([a_bytes[gp + 6], a_bytes[gp + 7]]));
                for h in 0..2 {
                    let kb = 2 * g + h;
                    let xb = (kb * n + col) * 72;
                    let d = f32::from_le_bytes([xq[xb], xq[xb + 1], xq[xb + 2], xq[xb + 3]]);
                    let s = i32::from_le_bytes([xq[xb + 4], xq[xb + 5], xq[xb + 6], xq[xb + 7]]);
                    let mut c: i32 = 0;
                    for t in 0..128 {
                        let kk = h * 128 + t;
                        let wbyte = a_bytes[gp + 8 + kk / 2];
                        let nib = (wbyte >> (4 * (kk % 2))) & 15;
                        let qw = nib as i32;
                        let xbyte = xq[xb + 8 + t / 2];
                        let xnib = (xbyte >> (4 * (t % 2))) & 15;
                        let qx = if xnib & 8 != 0 {
                            (xnib as i32) - 16
                        } else {
                            xnib as i32
                        };
                        c += qw * qx;
                    }
                    let (sc, zp) = if h == 0 { (sc0, zp0) } else { (sc1, zp1) };
                    let t1 = sc * d;
                    let p = t1 * (c as f32);
                    let t2 = zp * d;
                    let term = t2.mul_add(s as f32, p);
                    acc += term;
                }
            }
            y[col * m + r] += acc;
        }
    }
    y
}

fn bitwise_eq(a: &[f32], b: &[f32], rows: Option<&[usize]>, m: usize) -> (bool, usize) {
    let mut mism = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if let Some(rs) = rows {
            let r = i % m;
            if !rs.contains(&r) {
                continue;
            }
        }
        if x.to_bits() != y.to_bits() {
            mism += 1;
            if mism <= 4 {
                let col = i / m;
                let r = i % m;
                eprintln!("  mism[{}] (col={col} row={r}): gpu={x:e} cpu={y:e}", i);
            }
        }
    }
    (mism == 0, mism)
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
    salt: u32,
    full_cpu: bool,
) -> bool {
    assert_eq!(k % 256, 0, "{label}: K={k} not multiple of 256");
    let sym = if add { ADD_SYM } else { SET_SYM };
    for s in [SET_SYM, ADD_SYM] {
        gpu.ensure_kernel_public(MODULE, IU4_GFX12_SRC, s)
            .unwrap_or_else(|e| panic!("JIT {s}: {e}"));
    }

    // X [n, k] f32 through the real quantizer.
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            let scale = 0.75 + (col % 5) as f32 * 0.15;
            prng_f32(i, salt) * scale
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr =
        gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure_int4_mmq_x") as *mut std::ffi::c_void;
    let xq_len = (k / 128) * n * 72;
    let xq = dtoh_raw(gpu, xq_ptr, xq_len);

    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;

    let y_elems = n * m;
    let y0: Vec<f32> = if add {
        (0..y_elems)
            .map(|i| prng_f32(i, 0xADD0_BEEF ^ salt) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0f32; y_elems]
    };
    let d_y1 = gpu.upload_f32(&y0, &[n, m]).expect("upload Y1");
    let d_y2 = gpu.upload_f32(&y0, &[n, m]).expect("upload Y2");
    let y1_ptr = d_y1.buf.as_ptr() as *const std::ffi::c_void;
    let y2_ptr = d_y2.buf.as_ptr() as *const std::ffi::c_void;

    launch_iu4(gpu, sym, a_ptr, xq_ptr as *const _, y1_ptr, m, k, n, add);
    launch_iu4(gpu, sym, a_ptr, xq_ptr as *const _, y2_ptr, m, k, n, add);
    gpu.hip.device_synchronize().expect("sync");
    let y1 = gpu.download_f32(&d_y1).expect("dl y1");
    let y2 = gpu.download_f32(&d_y2).expect("dl y2");
    assert!(y1.iter().all(|v| v.is_finite()), "{label}: non-finite gpu output");

    // 2. repeat bit-identical (full).
    let (eq_rep, mism_rep) = bitwise_eq(&y1, &y2, None, m);

    // 1. vs CPU reference (full for small, edge stripes for big).
    let rows: Option<Vec<usize>> = if full_cpu {
        None
    } else {
        let mut rs: Vec<usize> = (0..64.min(m)).collect();
        if m > 128 {
            rs.extend((m - 64..m).collect::<Vec<_>>());
            rs.extend((m / 2..m / 2 + 16).collect::<Vec<_>>());
        }
        Some(rs)
    };
    let y_ref = cpu_ref(a_bytes, &xq, &y0, m, k, n, rows.as_deref());
    let (eq_cpu, mism_cpu) = bitwise_eq(&y1, &y_ref, rows.as_deref(), m);

    // 3. vs shipped fp8 route (N%64 only).
    let mut fp8_stat = String::from("n/a");
    if n % 64 == 0 {
        let d_yf = gpu.upload_f32(&y0, &[n, m]).expect("upload Yf");
        gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8(
            &d_a,
            &d_x,
            &d_yf,
            m,
            k,
            n,
            1,
        )
        .expect("fp8 route");
        gpu.hip.device_synchronize().expect("sync fp8");
        let yf = gpu.download_f32(&d_yf).expect("dl yf");
        // For add mode the fp8 route also accumulates onto y0: compare deltas.
        let mut absv: Vec<f32> = y1.iter().zip(yf.iter()).map(|(a, b)| (a - b).abs()).collect();
        absv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max = absv[absv.len() - 1];
        let mean = absv.iter().sum::<f32>() / absv.len() as f32;
        let p999 = absv[(absv.len() * 999 / 1000).min(absv.len() - 1)];
        let denom = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        fp8_stat = format!("max_abs={max:.4e} mean_abs={mean:.4e} p99.9={p999:.4e} max_rel={:.4e}", max / denom);
        let _ = gpu.free_tensor(d_yf);
    }

    let ok = eq_cpu && eq_rep;
    eprintln!(
        "{label}: cpu_bitwise={} (mism={mism_cpu}) repeat_identical={} (mism={mism_rep}) fp8[{fp8_stat}] {}",
        if eq_cpu { "OK" } else { "FAIL" },
        if eq_rep { "OK" } else { "FAIL" },
        if ok { "PASS" } else { "FAIL" },
    );
    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y1);
    let _ = gpu.free_tensor(d_y2);
    ok
}

/// GPU-event us/call: warmup then 100 timed launches, median. Returns
/// (median_us, tops_vs_539.7).
fn time_case(
    gpu: &mut Gpu,
    label: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
) {
    let sym = if add { ADD_SYM } else { SET_SYM };
    for s in [SET_SYM, ADD_SYM] {
        gpu.ensure_kernel_public(MODULE, IU4_GFX12_SRC, s)
            .unwrap_or_else(|e| panic!("JIT {s}: {e}"));
    }
    // Random X once (timing only; values don't matter).
    let x: Vec<f32> = (0..n * k).map(|i| prng_f32(i, 0x71E0).mul_add(1.0, 0.0)).collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr =
        gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure_int4_mmq_x") as *mut std::ffi::c_void;
    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;
    let y0 = vec![0.0f32; n * m];
    let d_y = gpu.upload_f32(&y0, &[n, m]).expect("upload Y");
    let y_ptr = d_y.buf.as_ptr() as *const std::ffi::c_void;
    for _ in 0..10 {
        launch_iu4(gpu, sym, a_ptr, xq_ptr as *const _, y_ptr, m, k, n, add);
    }
    gpu.hip.device_synchronize().expect("sync");
    // Host-batch timing (GPU events proved unreliable here): 5 batches of
    // 20 launches, median batch/20.
    let mut meds = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            launch_iu4(gpu, sym, a_ptr, xq_ptr as *const _, y_ptr, m, k, n, add);
        }
        gpu.hip.device_synchronize().expect("sync");
        meds.push(t0.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = meds[2];
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    // 1 GFLOP/us = 1000 TOPS.
    let tops = gflop / med * 1000.0;
    eprintln!("TIME {label}: med={med:.1} us/call GFLOP={gflop:.2} -> {tops:.1} TOPS ({:.1}% of 539.7)", tops / 539.7 * 100.0);
    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let model = args.iter().find(|a| !a.starts_with('-')).cloned().unwrap_or_else(|| {
        eprintln!("usage: tmp_iu4_gfx12_oracle <model.mq4-xt> [--compile-only]");
        std::process::exit(2);
    });

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    eprintln!("tmp_iu4_gfx12_oracle on {}", gpu.arch);
    assert_eq!(gpu.arch, "gfx1201", "this oracle targets gfx1201");

    for s in [SET_SYM, ADD_SYM] {
        gpu.ensure_kernel_public(MODULE, IU4_GFX12_SRC, s)
            .unwrap_or_else(|e| panic!("JIT {s}: {e}"));
        eprintln!("JIT OK: {s}");
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

    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let (gate_m, gate_k, gate) = load_mq4v2(&hfq, GATE_NAME);
    let (down_m, down_k, down) = load_mq4v2(&hfq, DOWN_NAME);
    eprintln!("loaded gate_proj {gate_m}x{gate_k}  down_proj {down_m}x{down_k}");
    // TIME=1: per-kernel us/call table on the 4 profile shapes (N=512),
    // then return (no correctness checks).
    if std::env::var("TIME").ok().as_deref() == Some("1") {
        // gate_up row: gate + up (real gate weights stand in for both halves).
        time_case(&mut gpu, "gate-M17408-K5120-N512/set", &gate, gate_m, gate_k, 512, false);
        time_case(&mut gpu, "gate-M17408-K5120-N512/add", &gate, gate_m, gate_k, 512, true);
        // residual row: down.
        time_case(&mut gpu, "down-M5120-K17408-N512/add", &down, down_m, down_k, 512, true);
        // qkvza row (M=16480 = 10240+6144+48+48, K=5120) and qkv row
        // (M=14336 = 12288+1024+1024, K=5120): synthetic weights, same shapes.
        let aqkvza = pack_mq4g256v2_synth(16480, 5120, 0x71E0_A000);
        time_case(&mut gpu, "qkvza-M16480-K5120-N512/set", &aqkvza, 16480, 5120, 512, false);
        let aqkv = pack_mq4g256v2_synth(14336, 5120, 0x71E0_B000);
        time_case(&mut gpu, "qkv-M14336-K5120-N512/set", &aqkv, 14336, 5120, 512, false);
        return;
    }
    drop(hfq);

    let mut ok = true;
    ok &= run_case(&mut gpu, "gate/set", &gate, gate_m, gate_k, 512, false, 0xC0DE_0001, false);
    ok &= run_case(&mut gpu, "gate/add", &gate, gate_m, gate_k, 512, true, 0xC0DE_0002, false);
    ok &= run_case(&mut gpu, "down/set", &down, down_m, down_k, 512, false, 0xC0DE_0003, false);
    ok &= run_case(&mut gpu, "down/add", &down, down_m, down_k, 512, true, 0xC0DE_0004, false);
    // Phase edges, synthetic data (full CPU check: small).
    let a48 = pack_mq4g256v2_synth(48, 1024, 0xED6E_4000);
    ok &= run_case(&mut gpu, "m48tail/set", &a48, 48, 1024, 512, false, 0xE040, true);
    let a80 = pack_mq4g256v2_synth(2048, 1024, 0xED6E_5000);
    ok &= run_case(&mut gpu, "n80cols/set", &a80, 2048, 1024, 80, false, 0xE050, true);
    let a256 = pack_mq4g256v2_synth(256, 256, 0xED6E_6000);
    ok &= run_case(&mut gpu, "k256/set", &a256, 256, 256, 512, false, 0xE060, true);
    ok &= run_case(&mut gpu, "k256/add", &a256, 256, 256, 512, true, 0xE061, true);
    let apart = pack_mq4g256v2_synth(100, 512, 0xED6E_7000);
    ok &= run_case(&mut gpu, "m100n100/set", &apart, 100, 512, 100, false, 0xE070, true);


    if ok {
        eprintln!("GFX12-IU4 ORACLE PASS");
    } else {
        eprintln!("GFX12-IU4 ORACLE FAIL");
        std::process::exit(1);
    }
}
