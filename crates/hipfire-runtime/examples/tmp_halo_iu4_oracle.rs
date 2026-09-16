//! W3 bit oracle: shipping IU4 full_{set,add}_occ3 vs MC gfx1151 candidates.
//!
//! Loads real qt=44 gate_proj (set) and down_proj (add, nonzero Y0) from an
//! MQ4-XT HFQ, quantizes X via `ensure_int4_mmq_x`, and raw-launches
//! `*_full_{set,add}_occ3` (block [32,8,1], LDS 30720) against
//! `*_full_{set,add}_{mc}_gfx1151` (same block/LDS) on identical grid.
//! Asserts bitwise-equal Y and immutable A/Xq. Phase-edge synthetic
//! fixtures cover G=1/2/3. TIME reports medians, TOPS, and % of 107.8 peak.
//!
//! Candidate select (argv wins over env, default mc2):
//!   --w2-sym mc2|mc4     or     W2_SYM=mc2|mc4
//!
//! Compile-only (metadata gate; no alloc/launch):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-mc HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     W2_SYM=mc2 \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_halo_iu4_oracle -- --compile-only
//!
//! Correctness / TIME:
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-mc HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_halo_iu4_oracle -- --w2-sym mc4 \
//!       /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
//!   TIME=1 … (same)   # interleaved medians ≥100 samples, warm
//!
//! Production module/source identity matches gemm.rs:
//!   MODULE = "gemm_mq4g256v2_residual_mmq_iu4"
//!   SRC    = kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC (quant + IU4 HIP)


use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Same concat as `kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC` (module private).
const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip")
);
/// Production module name — same cache key as `Gpu::gemm_mq4g256v2_mmq_prequant_iu4`.
const MODULE: &str = "gemm_mq4g256v2_residual_mmq_iu4";
const BASE_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3";
const BASE_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3";

const SHARED_BASE: u32 = (128 * 18 + 128 * 42) * 4; // 30720
const BLOCK_BASE: [u32; 3] = [32, 8, 1];
// MC candidates share block/LDS with shipping (same body, MC WMMA issue).
const SHARED_MC: u32 = SHARED_BASE;
const BLOCK_MC: [u32; 3] = BLOCK_BASE;

/// Candidate suffix: `--w2-sym mc2|mc4` (argv) or `W2_SYM` (env); argv wins.
fn parse_mc(argv: &mut Vec<String>) -> String {
    let mut mc: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--w2-sym" && i + 1 < argv.len() {
            mc = Some(argv[i + 1].clone());
            argv.drain(i..i + 2);
            continue;
        }
        if let Some(s) = argv[i].strip_prefix("--w2-sym=") {
            mc = Some(s.to_string());
            argv.remove(i);
            continue;
        }
        i += 1;
    }
    let mc = mc
        .or_else(|| std::env::var("W2_SYM").ok())
        .unwrap_or_else(|| "mc2".to_string());
    assert!(
        mc == "mc2" || mc == "mc4",
        "candidate suffix must be mc2|mc4 (got {mc})"
    );
    mc
}

fn cand_sym(op: &str, mc: &str) -> String {
    format!("gemm_mq4g256v2_residual_mmq_iu4_full_{op}_{mc}_gfx1151")
}
const N_DEFAULT: usize = 512;
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

fn prng_f32(i: usize, salt: u32) -> f32 {
    (prng_u32(i as u64, salt) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn f16_bits(x: f32) -> u16 {
    // Round-to-nearest-even f32→f16 (same helper shape as A0 calibrate).
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let shift = (1 - new_exp) as u32;
        let m = (mant | 0x800000) >> (shift + 13);
        let round = ((mant | 0x800000) >> (shift + 12)) & 1;
        return sign | ((m + round) as u16);
    }
    let mut m = mant >> 13;
    if (mant & 0x1000) != 0 {
        m += 1;
        if m == 0x400 {
            m = 0;
            new_exp += 1;
            if new_exp >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((new_exp as u16) << 10) | (m as u16)
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
    assert_eq!(
        bytes.len(),
        expect,
        "{name}: bytes={} expect {expect} (M={m} K={k})",
        bytes.len()
    );
    (m, k, bytes)
}

fn bitwise_eq_f32(a: &[f32], b: &[f32]) -> (bool, usize, Option<(usize, u32, u32)>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if first.is_none() {
                first = Some((i, x.to_bits(), y.to_bits()));
            }
            mism += 1;
        }
    }
    (mism == 0, mism, first)
}

fn bytes_eq(a: &[u8], b: &[u8]) -> (bool, usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            if first.is_none() {
                first = Some(i);
            }
            mism += 1;
        }
    }
    (mism == 0, mism, first)
}

fn download_raw(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = t.buf.size();
    let mut out = vec![0u8; n];
    gpu.hip
        .memcpy_dtoh(&mut out, &t.buf)
        .unwrap_or_else(|e| panic!("dtoh raw: {e}"));
    out
}

fn download_ptr(gpu: &Gpu, ptr: *const std::ffi::c_void, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    // Borrowed view — does not free.
    let view = unsafe { DeviceBuffer::from_raw(ptr as *mut _, n) };
    gpu.hip
        .memcpy_dtoh(&mut out, &view)
        .unwrap_or_else(|e| panic!("dtoh ptr: {e}"));
    out
}

fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_KERNEL_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".hipfire_kernels");
    }
    PathBuf::from(".hipfire_kernels")
}

fn print_module_cache(arch: &str, module: &str) {
    let dir = cache_root().join(arch);
    eprintln!("MODULE={module}");
    eprintln!("CACHE_DIR={}", dir.display());
    let mut hits = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let name = ent.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(module) {
                hits.push(ent.path());
            }
        }
    }
    hits.sort();
    if hits.is_empty() {
        eprintln!("CACHE_ENTRIES: (none yet under {})", dir.display());
    } else {
        for p in &hits {
            eprintln!("CACHE_ENTRY={}", p.display());
        }
    }
    if let Some(hs) = hits.iter().find(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "hsaco")
    }) {
        eprintln!("HSACO={}", hs.display());
    }
    if let Some(rw) = hits.iter().find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("radiowave"))
    }) {
        eprintln!("RADIOWAVE={}", rw.display());
    }
}


fn ensure_all(gpu: &mut Gpu, cand_set: &str, cand_add: &str) {
    for sym in [BASE_SET, BASE_ADD, cand_set, cand_add] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("compiled {sym} (module {MODULE})");
    }
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
    ms as f64 * 1e3
}

/// Dense integer ops → TOPS from event µs: 2·M·K·N / (us · 1e6).
fn tops_from_us(m: usize, k: usize, n: usize, us: f64) -> f64 {
    if !(us > 0.0) {
        return f64::NAN;
    }
    (2.0 * m as f64 * k as f64 * n as f64) / (us * 1.0e6)
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Pack MQ4G256V2 weights with boundary coverage: dual headers, codes 0/15,
/// alternating nibbles, and mixed patterns per row/group.
fn pack_mq4g256v2_edge(m: usize, k: usize, salt: u32) -> Vec<u8> {
    const GROUP: usize = 256;
    const HALF: usize = 128;
    const GB: usize = 136;
    assert!(k % GROUP == 0 && m % 128 == 0);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GB];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GB;
            let mode = (r + g + salt as usize) % 6;
            let sc0 = f16_bits(0.04 + ((r * 3 + g) % 17) as f32 * 0.002);
            let zp0 = f16_bits(-0.31 - ((r + g) % 5) as f32 * 0.02);
            let sc1 = f16_bits(0.09 + ((r * 5 + g) % 13) as f32 * 0.002);
            let zp1 = f16_bits(-0.47 + ((r + 2 * g) % 7) as f32 * 0.015);
            blob[dst..dst + 2].copy_from_slice(&sc0.to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&zp0.to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&sc1.to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&zp1.to_le_bytes());
            for i in 0..HALF {
                let (lo, hi) = match mode {
                    0 => (0u8, 15u8),
                    1 => (15u8, 0u8),
                    2 => (0xAu8, 0x5u8),
                    3 => ((i & 0xf) as u8, ((i >> 2) & 0xf) as u8),
                    4 => (0u8, 0u8),
                    _ => {
                        let p = prng_u32((r * 1024 + g * HALF + i) as u64, salt);
                        ((p & 0xf) as u8, ((p >> 4) & 0xf) as u8)
                    }
                };
                blob[dst + 8 + i] = (lo & 0xf) | ((hi & 0xf) << 4);
            }
        }
    }
    blob
}

/// Host-pack block_i4_128 with signed extremes −8/+7, nonzero s, all-zero
/// and mixed-sign columns.
fn pack_i4_xq_edge(k: usize, n: usize, salt: u32) -> Vec<u8> {
    assert!(k % 128 == 0 && n % 128 == 0);
    let blocks = k / 128;
    let mut blob = vec![0u8; blocks * n * 72];
    for b in 0..blocks {
        for t in 0..n {
            let off = (b * n + t) * 72;
            let mode = (t + b + salt as usize) % 5;
            let d = match mode {
                0 => 0.0f32,
                _ => 0.015 + (prng_u32((b * n + t) as u64, salt) % 40) as f32 * 0.0005,
            };
            let mut s: i32 = 0;
            let mut qs = [0u8; 64];
            for i in 0..64 {
                let (e0, e1): (i8, i8) = match mode {
                    0 => (0, 0),
                    1 => (-8, 7),
                    2 => (7, -8),
                    3 => {
                        let v = ((i as i8) % 8) - 4;
                        (v, -v)
                    }
                    _ => {
                        let p = prng_u32((b * 256 + t * 64 + i) as u64, salt ^ 0xC0DE);
                        let a = ((p % 16) as i8) - 8;
                        let bb = (((p >> 4) % 16) as i8) - 8;
                        (a, bb)
                    }
                };
                qs[i] = ((e0 as u8) & 0xf) | (((e1 as u8) & 0xf) << 4);
                s += e0 as i32 + e1 as i32;
            }
            if mode != 0 && s == 0 {
                qs[0] = (1i8 as u8) & 0xf;
                s = 1;
            }
            blob[off..off + 4].copy_from_slice(&d.to_le_bytes());
            blob[off + 4..off + 8].copy_from_slice(&s.to_le_bytes());
            blob[off + 8..off + 72].copy_from_slice(&qs);
        }
    }
    blob
}

enum XqMode<'a> {
    EnsureInt4,
    HostPacked(&'a [u8]),
}

fn y0_bytes(y0: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(y0.as_ptr() as *const u8, y0.len() * 4) }
}

fn run_pair(
    gpu: &mut Gpu,
    label: &str,
    a_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    xq_mode: XqMode<'_>,
    time: bool,
    cand_set: &str,
    cand_add: &str,
) -> bool {
    assert!(m % 128 == 0 && n % 128 == 0 && k % 256 == 0);
    let grid = [(m / 128) as u32, (n / 128) as u32, 1];
    let (base_sym, mc_sym) = if add {
        (BASE_ADD, cand_add)
    } else {
        (BASE_SET, cand_set)
    };

    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;
    let a_before = a_bytes.to_vec();

    let (xq_ptr, xq_bytes, xq_owner, x_owner) = match xq_mode {
        XqMode::EnsureInt4 => {
            let x: Vec<f32> = (0..n * k)
                .map(|i| {
                    let col = i / k;
                    let scale = 0.75 + (col % 5) as f32 * 0.15;
                    prng_f32(i, 0xA5A5_00C0) * scale
                })
                .collect();
            let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
            let xq = gpu
                .ensure_int4_mmq_x(&d_x, n, k)
                .expect("ensure_int4_mmq_x") as *const std::ffi::c_void;
            let xq_len = (k / 128) * n * 72;
            let snap = download_ptr(gpu, xq, xq_len);
            (xq, snap, None, Some(d_x))
        }
        XqMode::HostPacked(bytes) => {
            let d_xq = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload Xq");
            let ptr = d_xq.buf.as_ptr() as *const std::ffi::c_void;
            (ptr, bytes.to_vec(), Some(d_xq), None)
        }
    };

    let y_elems = n * m;
    let y0: Vec<f32> = if add {
        (0..y_elems)
            .map(|i| prng_f32(i, 0xADD0_BEEF) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0f32; y_elems]
    };
    let d_y_base = gpu.upload_f32(&y0, &[n, m]).expect("upload Y base");
    let d_y_mc = gpu.upload_f32(&y0, &[n, m]).expect("upload Y mc");

    let add_i = i32::from(add);
    launch_once(
        gpu,
        base_sym,
        grid,
        BLOCK_BASE,
        SHARED_BASE,
        a_ptr,
        xq_ptr,
        d_y_base.buf.as_ptr() as *const _,
        m as i32,
        k as i32,
        n as i32,
        add_i,
    );
    launch_once(
        gpu,
        mc_sym,
        grid,
        BLOCK_MC,
        SHARED_MC,
        a_ptr,
        xq_ptr,
        d_y_mc.buf.as_ptr() as *const _,
        m as i32,
        k as i32,
        n as i32,
        add_i,
    );
    gpu.hip.device_synchronize().expect("sync parity");

    let y_base = gpu.download_f32(&d_y_base).expect("dl base");
    let y_mc = gpu.download_f32(&d_y_mc).expect("dl mc");
    let finite = y_base.iter().all(|v| v.is_finite()) && y_mc.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq_f32(&y_base, &y_mc);
    let moved = if add {
        y_base
            .iter()
            .zip(y0.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    } else {
        y_base.iter().any(|v| v.to_bits() != 0)
    };

    let a_after = download_raw(gpu, &d_a);
    let (a_ok, a_mism, a_first) = bytes_eq(&a_before, &a_after);
    let xq_after = download_ptr(gpu, xq_ptr, xq_bytes.len());
    let (xq_ok, xq_mism, xq_first) = bytes_eq(&xq_bytes, &xq_after);

    let ok = eq && finite && moved && a_ok && xq_ok;
    eprintln!(
        "  [{label} M={m} K={k} N={n} add={add}] finite={finite} moved={moved} \
         bitwise_eq={eq} mism={mism} first={first:?} A_ok={a_ok}(mism={a_mism} first={a_first:?}) \
         Xq_ok={xq_ok}(mism={xq_mism} first={xq_first:?}) [{}]",
        if ok { "PASS" } else { "FAIL" }
    );
    if !eq {
        if let Some((i, b, w)) = first {
            let row = i % m;
            let col = i / m;
            eprintln!(
                "    first Y mismatch idx={i} row={row} col={col} base={b:#010x} mc={w:#010x}"
            );
        }
    }

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        let yb = y0_bytes(&y0);
        for _ in 0..WARM {
            if add {
                gpu.hip
                    .memcpy_htod(&d_y_base.buf, yb)
                    .expect("reseed Y base warm");
                gpu.hip
                    .memcpy_htod(&d_y_mc.buf, yb)
                    .expect("reseed Y mc warm");
            }
            launch_once(
                gpu,
                base_sym,
                grid,
                BLOCK_BASE,
                SHARED_BASE,
                a_ptr,
                xq_ptr,
                d_y_base.buf.as_ptr() as *const _,
                m as i32,
                k as i32,
                n as i32,
                add_i,
            );
            launch_once(
                gpu,
                mc_sym,
                grid,
                BLOCK_MC,
                SHARED_MC,
                a_ptr,
                xq_ptr,
                d_y_mc.buf.as_ptr() as *const _,
                m as i32,
                k as i32,
                n as i32,
                add_i,
            );
        }
        gpu.hip.device_synchronize().expect("sync warm");

        let mut base_us = Vec::with_capacity(SAMPLES);
        let mut mc_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            if add {
                gpu.hip
                    .memcpy_htod(&d_y_base.buf, yb)
                    .expect("reseed Y base");
                gpu.hip
                    .memcpy_htod(&d_y_mc.buf, yb)
                    .expect("reseed Y mc");
            }
            if i % 2 == 0 {
                base_us.push(time_us(
                    gpu,
                    base_sym,
                    grid,
                    BLOCK_BASE,
                    SHARED_BASE,
                    a_ptr,
                    xq_ptr,
                    d_y_base.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    n as i32,
                    add_i,
                ));
                mc_us.push(time_us(
                    gpu,
                    mc_sym,
                    grid,
                    BLOCK_MC,
                    SHARED_MC,
                    a_ptr,
                    xq_ptr,
                    d_y_mc.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    n as i32,
                    add_i,
                ));
            } else {
                mc_us.push(time_us(
                    gpu,
                    mc_sym,
                    grid,
                    BLOCK_MC,
                    SHARED_MC,
                    a_ptr,
                    xq_ptr,
                    d_y_mc.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    n as i32,
                    add_i,
                ));
                base_us.push(time_us(
                    gpu,
                    base_sym,
                    grid,
                    BLOCK_BASE,
                    SHARED_BASE,
                    a_ptr,
                    xq_ptr,
                    d_y_base.buf.as_ptr() as *const _,
                    m as i32,
                    k as i32,
                    n as i32,
                    add_i,
                ));
            }
        }
        let bmed = median_f64(&mut base_us);
        let mcmed = median_f64(&mut mc_us);
        let delta = (mcmed - bmed) / bmed * 100.0;
        let bmin = base_us.iter().cloned().fold(f64::INFINITY, f64::min);
        let bmax = base_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mcmin = mc_us.iter().cloned().fold(f64::INFINITY, f64::min);
        let mcmax = mc_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        const PEAK_TOPS: f64 = 107.8;
        let btops = tops_from_us(m, k, n, bmed);
        let mctops = tops_from_us(m, k, n, mcmed);
        let bpct = 100.0 * btops / PEAK_TOPS;
        let mcpct = 100.0 * mctops / PEAK_TOPS;
        eprintln!(
            "  TIME {label}: base_med={bmed:.3} us (min={bmin:.3} max={bmax:.3})  \
             mc_med={mcmed:.3} us (min={mcmin:.3} max={mcmax:.3})  delta={delta:+.2}%  \
             (n={SAMPLES} interleaved)"
        );
        eprintln!(
            "  TOPS {label}: base={btops:.2} ({bpct:.1}% of {PEAK_TOPS})  \
             mc={mctops:.2} ({mcpct:.1}% of {PEAK_TOPS})  ops=2*M*K*N={ops}",
            ops = 2usize.saturating_mul(m).saturating_mul(k).saturating_mul(n)
        );
    }

    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_y_base);
    let _ = gpu.free_tensor(d_y_mc);
    if let Some(t) = x_owner {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = xq_owner {
        let _ = gpu.free_tensor(t);
    }
    ok
}

fn main() {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let compile_only = argv.iter().any(|a| a == "--compile-only");
    argv.retain(|a| a != "--compile-only");
    let mc = parse_mc(&mut argv);
    let cand_set = cand_sym("set", &mc);
    let cand_add = cand_sym("add", &mc);


    let model = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt".to_string());
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    eprintln!(
        "tmp_halo_iu4_oracle on {}  model={model}  TIME={time}  mc={mc}  compile_only={compile_only}",
        gpu.arch
    );

    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: MC symbols are #if __gfx1151__; arch={} may fail JIT of _mc2/_mc4 entries",
            gpu.arch
        );
    }

    // Production module ensure — same MODULE + SRC as gemm.rs launcher.
    ensure_all(&mut gpu, &cand_set, &cand_add);
    print_module_cache(&gpu.arch, MODULE);

    if compile_only {
        eprintln!("--compile-only: module ensured; returning before allocation/launch");
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

    let mut ok = true;

    ok &= run_pair(
        &mut gpu,
        "gate_proj/set",
        &gate,
        gate_m,
        gate_k,
        N_DEFAULT,
        false,
        XqMode::EnsureInt4,
        time,
        &cand_set,
        &cand_add,
    );
    ok &= run_pair(
        &mut gpu,
        "down_proj/add",
        &down,
        down_m,
        down_k,
        N_DEFAULT,
        true,
        XqMode::EnsureInt4,
        time,
        &cand_set,
        &cand_add,
    );

    // Phase-edge synthetic: G=1,2,3 (K=256,512,768), set+add, boundary packs.
    // Plus a few extra full-tile inventory shapes.
    let edges: &[(usize, usize, usize, u32, &str, bool, bool)] = &[
        // m, k, n, salt, tag, do_set, do_add
        (128, 256, 128, 0xE1, "G1", true, true),
        (256, 512, 256, 0xE2, "G2", true, true),
        (128, 768, 128, 0xE3, "G3", true, true),
        (1024, 5120, 128, 0x51, "set_1024x5120", true, false),
        (128, 5120, 512, 0x52, "set_128x5120_N512", true, false),
        (128, 5120, 256, 0x53, "add_128x5120", false, true),
        (5120, 5120, 128, 0x54, "add_5120x5120", false, true),
    ];
    for &(m, k, n, salt, tag, do_set, do_add) in edges {
        let a = pack_mq4g256v2_edge(m, k, salt);
        let xq = pack_i4_xq_edge(k, n, salt ^ 0x51);
        if do_set {
            ok &= run_pair(
                &mut gpu,
                &format!("edge_{tag}/set"),
                &a,
                m,
                k,
                n,
                false,
                XqMode::HostPacked(&xq),
                false,
                &cand_set,
                &cand_add,
            );
        }
        if do_add {
            ok &= run_pair(
                &mut gpu,
                &format!("edge_{tag}/add"),
                &a,
                m,
                k,
                n,
                true,
                XqMode::HostPacked(&xq),
                false,
                &cand_set,
                &cand_add,
            );
        }
    }

    if ok {
        eprintln!(
            "MC PASS: shipping occ3 vs {mc} bitwise-equal (gate set + down add + edges)"
        );
    } else {
        eprintln!("MC FAIL");
        std::process::exit(1);
    }
}
