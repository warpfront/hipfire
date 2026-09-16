//! Throwaway group-major (GM) multirow decode oracle: shipping row-major vs GM twins.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, builds the group-major copy of A on the host (permute 136-B
//! groups: dst[(kb*M+row)*136] = src[(row*gpr+kb)*136]), and raw-launches six
//! N=1 arms with f32 x and overwrite-Y:
//!   r2     shipping gemv_mq4g256v2_multirow_r2 on row-major A, grid ceil(M/2)
//!          (the gfx1151 daemon decode route: rows default 2)
//!   r2_gm  GM twin on group-major A, same grid
//!   r4_gm  GM twin on group-major A, grid ceil(M/4)
//!   r8_gm  GM twin on group-major A, grid ceil(M/8)
//!   r4     shipping row-major r4 on row-major A, grid ceil(M/4)
//!          (control: does multirow alone change decode vs the r2 route?)
//!   r8     shipping row-major r8 on row-major A, grid ceil(M/8) (same control)
//! Same ABI/block 32/LDS 0 per arm; asserts all six Y are bitwise-equal plus
//! A/X immutability.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-gmmr TIME=1 \
//!     ./target/release/examples/tmp_halo_gm_decode \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt [--compile-only]
//!
//! `--compile-only` JITs the six symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (100 samples each, warm) with medians and the arm/r2 ratio.
//! The host group-major relayout time is printed too (informative only; a
//! production route would relayout once offline).

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Module source mirrors crates/rdna-compute/src/kernels.rs
// (GEMV_MQ4G256V2_MULTIROW_SRC) so the shipping symbols are the exact
// production objects.
const GEMV_MR_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_multirow.hip")
);
const GEMV_MR_MOD: &str = "tmp_halo_gm_multirow_decode";

const R2: &str = "gemv_mq4g256v2_multirow_r2";
const R2_GM: &str = "gemv_mq4g256v2_multirow_r2_gm";
const R4_GM: &str = "gemv_mq4g256v2_multirow_r4_gm";
const R8_GM: &str = "gemv_mq4g256v2_multirow_r8_gm";
const R4: &str = "gemv_mq4g256v2_multirow_r4";
const R8: &str = "gemv_mq4g256v2_multirow_r8";

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

fn launch_blob(gpu: &Gpu, func: &str, grid: [u32; 3], args: Vec<u8>) {
    let mut blob = args;
    gpu.launch_kernel_blob(func, grid, [32, 1, 1], 0, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

fn time_blob(gpu: &Gpu, func: &str, grid: [u32; 3], args: Vec<u8>) -> f64 {
    let mut blob = args;
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    let stream = gpu.active_stream.as_ref();
    gpu.hip.event_record(&start, stream).expect("record start");
    gpu.launch_kernel_blob(func, grid, [32, 1, 1], 0, &mut blob)
        .unwrap_or_else(|e| panic!("timed launch {func}: {e}"));
    gpu.hip.event_record(&stop, stream).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms as f64 * 1e3
}

struct Arm {
    label: &'static str,
    sym: &'static str,
    grid: [u32; 3],
    /// Group-major weights (true) or row-major (false).
    gm: bool,
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
            "WARN: GM multirow twins are gfx11/gfx12-guarded; arch={} may fail JIT of _gm entries",
            gpu.arch
        );
    }

    // Compile-only: JIT all six entries, print the cache path, return.
    for sym in [R2, R2_GM, R4_GM, R8_GM, R4, R8] {
        gpu.ensure_kernel_public(GEMV_MR_MOD, GEMV_MR_SRC, sym)
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
        let arms = [
            Arm { label: "r2", sym: R2, grid: [m.div_ceil(2) as u32, 1, 1], gm: false },
            Arm { label: "r2_gm", sym: R2_GM, grid: [m.div_ceil(2) as u32, 1, 1], gm: true },
            Arm { label: "r4_gm", sym: R4_GM, grid: [m.div_ceil(4) as u32, 1, 1], gm: true },
            Arm { label: "r8_gm", sym: R8_GM, grid: [m.div_ceil(8) as u32, 1, 1], gm: true },
            Arm { label: "r4", sym: R4, grid: [m.div_ceil(4) as u32, 1, 1], gm: false },
            Arm { label: "r8", sym: R8, grid: [m.div_ceil(8) as u32, 1, 1], gm: false },
        ];

        // One f32 x [K] shared by all six arms; one A upload per layout.
        let x: Vec<f32> = (0..k).map(|i| prng_f32(i, 0x6DEC_0001) * 0.9).collect();
        let d_x = gpu.upload_f32(&x, &[k]).expect("upload X f32");
        let x_ptr = d_x.buf.as_ptr() as *const std::ffi::c_void;
        let x_len = k * 4;
        let mut x_before = vec![0u8; x_len];
        gpu.hip.memcpy_dtoh(&mut x_before, &d_x.buf).expect("snap X");
        let d_a_rm = gpu.upload_raw(a_rm, &[a_rm.len()]).expect("upload A rm");
        let a_rm_ptr = d_a_rm.buf.as_ptr() as *const std::ffi::c_void;
        let d_a_gm = gpu.upload_raw(a_gm, &[a_gm.len()]).expect("upload A gm");
        let a_gm_ptr = d_a_gm.buf.as_ptr() as *const std::ffi::c_void;

        let y0 = vec![0.0f32; m];
        let y0_bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mk = |arm: &Arm, y_ptr: *const std::ffi::c_void| {
            blob_gemv(
                if arm.gm { a_gm_ptr } else { a_rm_ptr },
                x_ptr,
                y_ptr,
                m as i32,
                k as i32,
            )
        };

        // Parity: one Y per arm, r2 (daemon route) is the reference.
        let mut ys: Vec<Vec<f32>> = Vec::with_capacity(arms.len());
        for arm in &arms {
            let d_y = gpu.upload_f32(&y0, &[m]).expect("upload Y");
            let y_ptr = d_y.buf.as_ptr() as *const std::ffi::c_void;
            launch_blob(&gpu, arm.sym, arm.grid, mk(arm, y_ptr));
            gpu.hip.device_synchronize().expect("sync parity");
            let y = gpu.download_f32(&d_y).expect("dl Y");
            let _ = gpu.free_tensor(d_y);
            ys.push(y);
        }
        let y_ref = &ys[0];
        let finite = ys.iter().all(|y| y.iter().all(|v| v.is_finite()));
        let moved = y_ref.iter().zip(y0.iter()).any(|(a, b)| a.to_bits() != b.to_bits());
        let mut arm_ok = finite && moved;
        for (arm, y) in arms.iter().zip(ys.iter()) {
            let (eq, mism, first) = bitwise_eq(y_ref, y);
            eprintln!(
                "  [{tname}/N=1/{}] grid={} bitwise_eq_vs_r2={eq} mism={mism} first={first:?}",
                arm.label, arm.grid[0],
            );
            arm_ok &= eq;
        }
        let mut rm_back = vec![0u8; a_rm.len()];
        gpu.hip.memcpy_dtoh(&mut rm_back, &d_a_rm.buf).expect("dl A rm back");
        let rm_same = rm_back == *a_rm;
        let mut gm_back = vec![0u8; a_gm.len()];
        gpu.hip.memcpy_dtoh(&mut gm_back, &d_a_gm.buf).expect("dl A gm back");
        let gm_same = gm_back == *a_gm;
        let mut x_back = vec![0u8; x_len];
        gpu.hip.memcpy_dtoh(&mut x_back, &d_x.buf).expect("dl X back");
        let x_same = x_back == x_before;
        arm_ok &= rm_same && gm_same && x_same;
        eprintln!(
            "  [{tname}/N=1] finite={finite} moved={moved} rm_immutable={rm_same} \
             gm_immutable={gm_same} x_immutable={x_same} [{}]",
            if arm_ok { "PASS" } else { "FAIL" }
        );
        ok &= arm_ok;

        if time {
            const WARM: usize = 10;
            const SAMPLES: usize = 100;
            // One resident Y per arm; rotating launch order interleaves arms
            // against drift. Y0 is restored before every launch.
            let mut d_ys = Vec::with_capacity(arms.len());
            for _ in &arms {
                d_ys.push(gpu.upload_f32(&y0, &[m]).expect("upload Y timed"));
            }
            for _ in 0..WARM {
                for (arm, d_y) in arms.iter().zip(d_ys.iter()) {
                    gpu.hip.memcpy_htod(&d_y.buf, &y0_bytes).expect("restore Y0");
                    launch_blob(
                        &gpu,
                        arm.sym,
                        arm.grid,
                        mk(arm, d_y.buf.as_ptr() as *const _),
                    );
                }
            }
            gpu.hip.device_synchronize().expect("sync warm");
            let mut acc = vec![Vec::with_capacity(SAMPLES); arms.len()];
            for i in 0..SAMPLES {
                for j in 0..arms.len() {
                    let a = (i + j) % arms.len();
                    let d_y = &d_ys[a];
                    gpu.hip.memcpy_htod(&d_y.buf, &y0_bytes).expect("restore Y0");
                    acc[a].push(time_blob(
                        &gpu,
                        arms[a].sym,
                        arms[a].grid,
                        mk(&arms[a], d_y.buf.as_ptr() as *const _),
                    ));
                }
            }
            let mut meds = Vec::with_capacity(arms.len());
            for (arm, v) in arms.iter().zip(acc.iter_mut()) {
                meds.push(median(v));
                let _ = arm;
            }
            for (arm, med) in arms.iter().zip(meds.iter()) {
                eprintln!(
                    "  TIME {tname}/N=1/{}: med={med:.3} us  ratio_vs_r2={:.4} (n={SAMPLES} interleaved)",
                    arm.label,
                    med / meds[0]
                );
            }
            for d_y in d_ys {
                let _ = gpu.free_tensor(d_y);
            }
        }

        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_a_rm);
        let _ = gpu.free_tensor(d_a_gm);
    }
    eprintln!(
        "NOTE gfx1151 N=1 decode dispatches gemv_mq4g256v2_multirow_r2 (rows default 2); \
         r4/r8 arms are controls for multirow-alone effects, r2_gm/r4_gm/r8_gm read the \
         group-major relayout (136*(kb*M+row)) under test."
    );
    if ok {
        eprintln!("GM MULTIROW DECODE PASS: all six arms bitwise-equal on gate+down");
    } else {
        eprintln!("GM MULTIROW DECODE FAIL");
        std::process::exit(1);
    }
}
