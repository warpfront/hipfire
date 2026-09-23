// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Temporary XTX baseline probe: scalar-ELF vs packed-ELF kernel comparison.
//!
//! Compares two independently-loaded HIP modules that export the SAME kernel
//! symbols — one compiled from scalar HEAD sources, one from the packed
//! candidate sources — so the comparison is an exact kernel-variant diff, not
//! a different-Ksplit association artifact.
//!
//! Usage:
//!   cargo build -p hipfire-runtime --example xtx_packed_kernel_probe
//!   ./target/debug/examples/xtx_packed_kernel_probe \
//!     --scalar /path/to/scalar.hsaco --packed /path/to/packed.hsaco \
//!     --kind residual --out /tmp/probe_residual.json
//!   ./target/debug/examples/xtx_packed_kernel_probe \
//!     --scalar /path/to/scalar.hsaco --packed /path/to/packed.hsaco \
//!     --kind qkvza --out /tmp/probe_qkvza.json
//!
//! Source grounding (exact ABIs read from kernel sources, never guessed):
//! - Residual split-K ABI `(A, X, Y, M, K, N)`, grid `[ceil(M/16),
//!   ceil(N/16)]`, block `[32*KW]`, symbols
//!   `gemm_mq4g256v2_residual_wmma_gfx1100_ks{2,4,8}_lds` from
//!   `kernels/src/gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds.hip`
//!   (`GEN_RESID_KSPLIT_LDS`) and the launcher symbol table in
//!   `crates/rdna-compute/src/gemm.rs`
//!   (`gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds`).
//! - Base residual ABI `(A, X, Y, M, K, batch_size)` from
//!   `kernels/src/gemm_mq4g256v2_residual_wmma.hip` (authority only; the probe
//!   compares same-variant ks kernels scalar-vs-packed, never base-vs-ks).
//! - QKVZA ABI `(A_qkv, A_z, A_beta, A_alpha, X, Y_qkv, Y_z, Y_beta,
//!   Y_alpha, qkv_m, z_m, beta_m, alpha_m, K, N)`, grid
//!   `[ceil(total_m/16), ceil(N/16)]`, block `[32]`, symbol
//!   `gemm_qkvza_mq4g256v2_wmma` from
//!   `kernels/src/gemm_qkvza_mq4g256v2_wmma.hip` and its launcher body in
//!   `crates/rdna-compute/src/gemm.rs` (`gemm_qkvza_mq4g256v2_wmma`).
//! - MQ4V2 layout (136 B groups, dual fp16 headers, nibble packing) from the
//!   two `.hip` sources above; deterministic-input / NaN-sentinel /
//!   bit-equality patterns reuse
//!   `crates/rdna-compute/examples/test_mq4v2_residual_ksplit_gfx1100.rs` and
//!   `crates/rdna-compute/examples/test_mq4v2_qkvza_bt_gfx1100.rs`.
//! - Raw HIP API surface (`HipRuntime::module_load`, `module_get_function`,
//!   `launch_kernel`, `event_*`, `malloc`/`memcpy_*`/`free`) from
//!   `crates/hip-bridge/src/ffi.rs`.
//! - QKVZA routing dims derived from
//!   `crates/hipfire-arch-qwen35/src/qwen35/weights.rs` (`in_proj_z`
//!   `[2048, dim]`, `in_proj_b/a` `[n_heads, dim]`), `forward.rs`
//!   (`qkv_dim = k_dim*2 + v_dim`), and the dense-shape test in
//!   `qwen35/config.rs` (n_heads=24, linear key 16x128, value 48x128).

use hip_bridge::{Event, Function, HipRuntime, Module};
use std::ffi::c_void;

// ── Constants ─────────────────────────────────────────────────────────────

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;
/// Distinct output canaries: scalar arm fills 0xCD, packed arm fills 0xAB, so
/// a leftover canary identifies which arm left an element unwritten.
const CANARY_SCALAR: u8 = 0xCD;
const CANARY_PACKED: u8 = 0xAB;

const WARMUPS: usize = 20;
const BRACKET_LAUNCHES: usize = 100;
const SAMPLES_PER_ARM: usize = 40;

/// Residual production shapes under test (layer-0 out_proj / down_proj).
const RESID_OUT_M: usize = 5120;
const RESID_OUT_K: usize = 6144;
const RESID_DOWN_M: usize = 5120;
const RESID_DOWN_K: usize = 17408;
const RESID_KS: [usize; 3] = [2, 4, 8];
const RESID_NS: [usize; 4] = [1, 8, 16, 17];

/// QKVZA routing dims: qkv = 16*128*2 + 48*128 = 10240, z = 16*128 = 2048,
/// beta = alpha = n_heads = 24, K = hidden dim 5120 (G256-aligned).
const QKV_M: usize = 10240;
const Z_M: usize = 2048;
const BETA_M: usize = 24;
const ALPHA_M: usize = 24;
const QKV_K: usize = 5120;
const QKV_NS: [usize; 4] = [1, 8, 16, 17];

// ── Deterministic host helpers ────────────────────────────────────────────

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_f32(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            let x = xorshift64(&mut s);
            lo + (hi - lo) * ((x >> 11) as f32 / (1u64 << 53) as f32)
        })
        .collect()
}

/// f32 -> IEEE binary16 bits, round-to-nearest-even. Mirrors the hardware cvt
/// used by the `convert_f32_to_f16` X-staging kernel, so host-staged X matches
/// what the production launchers feed the kernels. X is finite in [-1, 1].
fn f32_to_f16_bits_rne(v: f32) -> u16 {
    debug_assert!(v.is_finite());
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00;
    }
    if e == 0 {
        return s;
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    if e16 >= 1 {
        let half = (m >> 13) as u16;
        let rest = m & 0x1fff;
        let round_up = rest > 0x1000 || (rest == 0x1000 && (half & 1) == 1);
        let mut h = half + round_up as u16;
        let mut e16 = e16;
        if h == 0x400 {
            h = 0;
            e16 += 1;
        }
        if e16 >= 31 {
            return s | 0x7c00;
        }
        return s | ((e16 as u16) << 10) | (h & 0x3ff);
    }
    let m32 = (1u64 << 23) | m as u64;
    let sh = (126 - e) as u32;
    let (q, r) = if sh >= 64 {
        (0u64, m32)
    } else if sh == 0 {
        (m32, 0)
    } else {
        (m32 >> sh, m32 & ((1u64 << sh) - 1))
    };
    let half_bit = if sh == 0 || sh > 64 {
        0
    } else {
        1u64 << (sh - 1)
    };
    let round_up = if sh == 0 {
        false
    } else if sh > 64 {
        m32 != 0
    } else {
        r > half_bit || (r == half_bit && (q & 1) == 1)
    };
    let h = q + round_up as u64;
    if h >= 0x400 {
        s | (1u16 << 10)
    } else {
        s | (h as u16)
    }
}

/// Alias-safe 8-half2 (16 lanes = 32 bytes) block copy: memcpy semantics via
/// `copy_nonoverlapping`, never a reference reinterpret-cast, so overlapping
/// or differently-aligned staging buffers cannot alias.
fn copy_half2x8(dst: &mut [u16], src: &[u16; 16]) {
    assert!(dst.len() >= 16);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), 16);
    }
}

/// Stage host F32 X to F16 bits exactly as the staging kernel would, moved in
/// alias-safe 8-half2 blocks.
fn stage_x_f16(x_f32: &[f32]) -> Vec<u16> {
    let conv: Vec<u16> = x_f32.iter().map(|&v| f32_to_f16_bits_rne(v)).collect();
    let mut out = vec![0u16; conv.len()];
    let mut i = 0;
    while i + 16 <= conv.len() {
        let mut blk = [0u16; 16];
        blk.copy_from_slice(&conv[i..i + 16]);
        copy_half2x8(&mut out[i..i + 16], &blk);
        i += 16;
    }
    for (d, s) in out[i..].iter_mut().zip(conv[i..].iter()) {
        *d = *s;
    }
    out
}

/// Deterministic MQ4V2 weight blob: nibble payload bytes cycle through every
/// value 0..255 across the buffer. Headers are valid finite nonzero fp16
/// scales with zero-points from {+0.0, -0.0, 0.3, -0.3}; the non-power-of-two
/// entries (0.1/1/3/0.7 scales, 0.3/-0.3 zero-points) make most q*sc+zp
/// products require fp16 rounding, so a scalar-FMA vs packed-FMA discrepancy
/// cannot hide behind exact arithmetic. Signed zeros are retained in both
/// headers. `seed` keeps the four QKVZA projections distinct so a swapped
/// routing cannot bit-match.
fn pack_weights_deterministic(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let gpr = k / GROUP;
    // Finite normal nonzero fp16 scales: 0.25 exact plus 0.1 (0x2E66),
    // 1/3 (0x3555), 0.7 (0x399A) which force rounding for most q values.
    const SCALES: [u16; 4] = [0x3400, 0x2E66, 0x3555, 0x399A];
    // Zero-points: +0.0 (0x0000) and -0.0 (0x8000) retained, plus 0.3
    // (0x34CD) and -0.3 (0xB4CD) which force rounding in q*sc+zp.
    const ZEROS: [u16; 4] = [0x0000, 0x8000, 0x34CD, 0xB4CD];
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let s0 = SCALES[(r + g) % 4];
            let z0 = ZEROS[(r + g) % 4];
            let s1 = SCALES[(r * 7 + g * 3) % 4];
            let z1 = ZEROS[(r * 7 + g * 3 + 1) % 4];
            blob[dst..dst + 2].copy_from_slice(&s0.to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&z0.to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&s1.to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&z1.to_le_bytes());
            for j in 0..128 {
                blob[dst + 8 + j] =
                    ((((r * gpr + g) as u64 * 128 + j as u64 + seed) % 256) as u8);
            }
        }
    }
    blob
}

fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn variance(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Full bit parity over every element (all bits, not just L2): returns
/// (mismatch_count, first_mismatch_index, first_scalar_bits, first_packed_bits).
fn bit_parity(s: &[f32], p: &[f32]) -> (usize, Option<usize>, Option<u32>, Option<u32>) {
    assert_eq!(s.len(), p.len());
    let mut mism = 0usize;
    let mut first: Option<(usize, u32, u32)> = None;
    for (i, (a, b)) in s.iter().zip(p.iter()).enumerate() {
        let (ba, bb) = (a.to_bits(), b.to_bits());
        if ba != bb {
            mism += 1;
            if first.is_none() {
                first = Some((i, ba, bb));
            }
        }
    }
    match first {
        Some((i, ba, bb)) => (mism, Some(i), Some(ba), Some(bb)),
        None => (mism, None, None, None),
    }
}

fn count_canary_words(v: &[f32], canary_byte: u8) -> usize {
    let pat = u32::from_ne_bytes([canary_byte; 4]);
    v.iter().filter(|x| x.to_bits() == pat).count()
}

// ── Raw HIP launch shims (exact kernel ABIs) ──────────────────────────────

fn enqueue_residual(
    hip: &HipRuntime,
    func: &Function,
    a: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
    kw: usize,
) {
    let mut ap = a;
    let mut xp = x;
    let mut yp = y;
    let mut mv = m as i32;
    let mut kv = k as i32;
    let mut nv = n as i32;
    let mut params: Vec<*mut c_void> = vec![
        &mut ap as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void,
        &mut mv as *mut _ as *mut c_void,
        &mut kv as *mut _ as *mut c_void,
        &mut nv as *mut _ as *mut c_void,
    ];
    let grid = [((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1];
    let block = [(32 * kw) as u32, 1, 1];
    // Enqueue ONLY: no synchronization, so timed brackets queue back-to-back
    // without host round trips. Completion is observed via the bracket's
    // stop-event synchronize.
    unsafe {
        hip.launch_kernel(func, grid, block, 0, None, &mut params)
            .expect("residual hipModuleLaunchKernel failed");
    }
}

/// Synchronized correctness launch: enqueue + device-wide completion. Used by
/// every parity path; NEVER inside a timed bracket.
fn launch_residual(
    hip: &HipRuntime,
    func: &Function,
    a: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
    kw: usize,
) {
    enqueue_residual(hip, func, a, x, y, m, k, n, kw);
    hip.device_synchronize()
        .expect("sync after residual launch");
}

#[allow(clippy::too_many_arguments)]
fn enqueue_qkvza(
    hip: &HipRuntime,
    func: &Function,
    a_qkv: *mut c_void,
    a_z: *mut c_void,
    a_beta: *mut c_void,
    a_alpha: *mut c_void,
    x: *mut c_void,
    y_qkv: *mut c_void,
    y_z: *mut c_void,
    y_beta: *mut c_void,
    y_alpha: *mut c_void,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) {
    let mut aq = a_qkv;
    let mut az = a_z;
    let mut ab = a_beta;
    let mut aa = a_alpha;
    let mut xp = x;
    let mut yq = y_qkv;
    let mut yz = y_z;
    let mut yb = y_beta;
    let mut ya = y_alpha;
    let mut qm = qkv_m as i32;
    let mut zm = z_m as i32;
    let mut bm = beta_m as i32;
    let mut am = alpha_m as i32;
    let mut kv = k as i32;
    let mut nv = n as i32;
    let mut params: Vec<*mut c_void> = vec![
        &mut aq as *mut _ as *mut c_void,
        &mut az as *mut _ as *mut c_void,
        &mut ab as *mut _ as *mut c_void,
        &mut aa as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut yq as *mut _ as *mut c_void,
        &mut yz as *mut _ as *mut c_void,
        &mut yb as *mut _ as *mut c_void,
        &mut ya as *mut _ as *mut c_void,
        &mut qm as *mut _ as *mut c_void,
        &mut zm as *mut _ as *mut c_void,
        &mut bm as *mut _ as *mut c_void,
        &mut am as *mut _ as *mut c_void,
        &mut kv as *mut _ as *mut c_void,
        &mut nv as *mut _ as *mut c_void,
    ];
    let total = qkv_m + z_m + beta_m + alpha_m;
    let grid = [((total + 15) / 16) as u32, ((n + 15) / 16) as u32, 1];
    let block = [32, 1, 1];
    // Enqueue ONLY: no synchronization; the bracket's stop-event synchronize
    // is the sole completion gate.
    unsafe {
        hip.launch_kernel(func, grid, block, 0, None, &mut params)
            .expect("qkvza hipModuleLaunchKernel failed");
    }
}

/// Synchronized correctness launch: enqueue + device-wide completion. Used by
/// every parity path; NEVER inside a timed bracket.
#[allow(clippy::too_many_arguments)]
fn launch_qkvza(
    hip: &HipRuntime,
    func: &Function,
    a_qkv: *mut c_void,
    a_z: *mut c_void,
    a_beta: *mut c_void,
    a_alpha: *mut c_void,
    x: *mut c_void,
    y_qkv: *mut c_void,
    y_z: *mut c_void,
    y_beta: *mut c_void,
    y_alpha: *mut c_void,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) {
    enqueue_qkvza(hip, func, a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n);
    hip.device_synchronize()
        .expect("sync after qkvza launch");
}

// ── Device buffer helpers (bounded: freed per case) ───────────────────────

fn htod(hip: &HipRuntime, bytes: &[u8]) -> hip_bridge::DeviceBuffer {
    let buf = hip.malloc(bytes.len()).expect("hipMalloc failed");
    hip.memcpy_htod(&buf, bytes).expect("memcpy_htod failed");
    buf
}

fn htod_f32(hip: &HipRuntime, v: &[f32]) -> hip_bridge::DeviceBuffer {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    htod(hip, bytes)
}

fn htod_u16(hip: &HipRuntime, v: &[u16]) -> hip_bridge::DeviceBuffer {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    htod(hip, bytes)
}

fn dtoh_f32(hip: &HipRuntime, buf: &hip_bridge::DeviceBuffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    hip.memcpy_dtoh(bytes, buf).expect("memcpy_dtoh failed");
    out
}

fn fill_canary_f32(n: usize, canary: u8) -> Vec<f32> {
    vec![f32::from_bits(u32::from_ne_bytes([canary; 4])); n]
}

// ── Timing: HIP events, 100 launches/bracket, ABBA, 40 samples ────────────

/// One timed bracket: 100 enqueue-only launches between two HIP events. The
/// launch closure MUST NOT synchronize per launch; the stop-event synchronize
/// below is the sole completion gate, so the event delta measures queued
/// kernel time rather than host round trips. Requires positive finite time.
fn time_bracket_ms(
    hip: &HipRuntime,
    start: &Event,
    stop: &Event,
    launch: &mut dyn FnMut(),
) -> f64 {
    hip.event_record(start, None).expect("event_record start");
    for _ in 0..BRACKET_LAUNCHES {
        launch();
    }
    hip.event_record(stop, None).expect("event_record stop");
    hip.event_synchronize(stop)
        .expect("event_synchronize stop (completion gate)");
    let ms = hip
        .event_elapsed_ms(start, stop)
        .expect("hipEventElapsedTime") as f64;
    assert!(
        ms.is_finite() && ms > 0.0,
        "non-positive or non-finite event time: {ms}"
    );
    ms
}

// ── Residual case ─────────────────────────────────────────────────────────

// (Both module handles stay alive for the whole run; every symbol below is
// resolved from its arm's own handle, never a shared module cache.)

fn resid_symbol(kw: usize) -> &'static str {
    match kw {
        2 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds",
        4 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds",
        8 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds",
        _ => panic!("kw must be 2, 4, or 8"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_residual_parity(
    hip: &HipRuntime,
    f_scalar: &Function,
    f_packed: &Function,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    kw: usize,
    w_blob: &[u8],
    x_f16: &[u16],
    y_init: &[f32],
) -> (serde_json::Value, bool) {
    let y_len = n * m;
    let pad = 16 * m;
    let can_s = u32::from_ne_bytes([CANARY_SCALAR; 4]);
    let can_p = u32::from_ne_bytes([CANARY_PACKED; 4]);
    // Per arm: a zero-init buffer plus a controlled nonzero-init buffer over
    // the SAME weights/X. Every controlled output must bitwise equal the
    // correctly-rounded f32 sum of its zero-init output and the initial Y,
    // proving Y += reads prior Y and every logical output (including N=17
    // tail columns) was written. Trailing guards carry the arm canary.
    let mk_host = |y0: &[f32], canary: u8| {
        let mut v = Vec::with_capacity(y_len + pad);
        v.extend_from_slice(y0);
        v.extend(fill_canary_f32(pad, canary));
        v
    };
    let zero = vec![0.0f32; y_len];
    let sz_host = mk_host(&zero, CANARY_SCALAR);
    let sc_host = mk_host(y_init, CANARY_SCALAR);
    let pz_host = mk_host(&zero, CANARY_PACKED);
    let pc_host = mk_host(y_init, CANARY_PACKED);

    let d_as = htod(hip, w_blob);
    let d_ap = htod(hip, w_blob);
    let d_xs = htod_u16(hip, x_f16);
    let d_xp = htod_u16(hip, x_f16);
    let d_sz = htod_f32(hip, &sz_host);
    let d_sc = htod_f32(hip, &sc_host);
    let d_pz = htod_f32(hip, &pz_host);
    let d_pc = htod_f32(hip, &pc_host);

    launch_residual(hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_sz.as_ptr(), m, k, n, kw);
    launch_residual(hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_sc.as_ptr(), m, k, n, kw);
    launch_residual(hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_pz.as_ptr(), m, k, n, kw);
    launch_residual(hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_pc.as_ptr(), m, k, n, kw);
    let sz = dtoh_f32(hip, &d_sz, y_len + pad);
    let sc = dtoh_f32(hip, &d_sc, y_len + pad);
    let pz = dtoh_f32(hip, &d_pz, y_len + pad);
    let pc = dtoh_f32(hip, &d_pc, y_len + pad);

    // Correctly-rounded f32 reference: single-rounding host add of the
    // zero-init GPU result and the initial Y (mirrors the kernel's one Y +=).
    let exp_s: Vec<f32> = sz[..y_len].iter().zip(y_init.iter()).map(|(a, b)| a + b).collect();
    let exp_p: Vec<f32> = pz[..y_len].iter().zip(y_init.iter()).map(|(a, b)| a + b).collect();
    let (plus_mism_s, plus_first_s, _, _) = bit_parity(&sc[..y_len], &exp_s);
    let (plus_mism_p, plus_first_p, _, _) = bit_parity(&pc[..y_len], &exp_p);
    let (zero_mism, zero_first, _, _) = bit_parity(&sz[..y_len], &pz[..y_len]);
    let zero_exact_zeros = sz[..y_len].iter().filter(|x| x.to_bits() == 0).count()
        + pz[..y_len].iter().filter(|x| x.to_bits() == 0).count();

    let (mism, first_idx, first_s, first_p) = bit_parity(&sc[..y_len], &pc[..y_len]);
    let guard_s_intact = sc[y_len..].iter().all(|x| x.to_bits() == can_s)
        && sz[y_len..].iter().all(|x| x.to_bits() == can_s);
    let guard_p_intact = pc[y_len..].iter().all(|x| x.to_bits() == can_p)
        && pz[y_len..].iter().all(|x| x.to_bits() == can_p);
    let guards_intact = guard_s_intact && guard_p_intact;
    let ok = mism == 0
        && plus_mism_s == 0
        && plus_mism_p == 0
        && guards_intact
        && is_finite(&sc[..y_len])
        && is_finite(&pc[..y_len])
        && variance(&sc[..y_len]) > 1e-12
        && variance(&pc[..y_len]) > 1e-12;

    let v = serde_json::json!({
        "case": format!("residual/{label}/ks{kw}/N{n}"),
        "m": m, "k": k, "n": n, "kw": kw,
        "parity": {
            "bit_equal": mism == 0,
            "elements": y_len,
            "mismatches": mism,
            "first_mismatch": first_idx,
            "first_scalar_bits": first_s,
            "first_packed_bits": first_p,
            "rel_l2": rel_l2(&sc[..y_len], &pc[..y_len]),
            "max_abs": max_abs_diff(&sc[..y_len], &pc[..y_len]),
            "scalar_finite": is_finite(&sc[..y_len]),
            "packed_finite": is_finite(&pc[..y_len]),
            "scalar_variance": variance(&sc[..y_len]),
            "packed_variance": variance(&pc[..y_len]),
        },
        "y_plus": {
            "reference": "controlled output must bitwise equal correctly-rounded f32(zero_init output + initial Y)",
            "scalar_plus_mismatches": plus_mism_s,
            "scalar_plus_first_mismatch": plus_first_s,
            "packed_plus_mismatches": plus_mism_p,
            "packed_plus_first_mismatch": plus_first_p,
            "zero_init_parity_mismatches": zero_mism,
            "zero_init_parity_first_mismatch": zero_first,
            "zero_init_exact_zero_elements": zero_exact_zeros,
        },
        "canary": {
            "scalar_byte": format!("0x{:02X}", CANARY_SCALAR),
            "packed_byte": format!("0x{:02X}", CANARY_PACKED),
            "guard_floats_per_buffer": pad,
            "guard_buffers": 4,
            "guard_bytes_scanned": pad * 4 * 4,
            "scalar_guard_intact": guard_s_intact,
            "packed_guard_intact": guard_p_intact,
        },
        "pass": ok,
    });

    hip.free(d_as).ok();
    hip.free(d_ap).ok();
    hip.free(d_xs).ok();
    hip.free(d_xp).ok();
    hip.free(d_sz).ok();
    hip.free(d_sc).ok();
    hip.free(d_pz).ok();
    hip.free(d_pc).ok();
    (v, ok)
}

#[allow(clippy::too_many_arguments)]
fn run_residual_timing(
    hip: &HipRuntime,
    f_scalar: &Function,
    f_packed: &Function,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    kw: usize,
    w_blob: &[u8],
    x_f16: &[u16],
    y_init: &[f32],
) -> (serde_json::Value, bool) {
    // Buffers allocated and loaded once, outside the timed region.
    let d_as = htod(hip, w_blob);
    let d_ap = htod(hip, w_blob);
    let d_xs = htod_u16(hip, x_f16);
    let d_xp = htod_u16(hip, x_f16);
    let d_ys = htod_f32(hip, y_init);
    let d_yp = htod_f32(hip, y_init);

    // Full timed outputs verified BEFORE timing.
    launch_residual(
        hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_ys.as_ptr(), m, k, n, kw,
    );
    launch_residual(
        hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_yp.as_ptr(), m, k, n, kw,
    );
    let ys = dtoh_f32(hip, &d_ys, y_init.len());
    let yp = dtoh_f32(hip, &d_yp, y_init.len());
    let (mism, first_idx, first_s, first_p) = bit_parity(&ys, &yp);
    if mism != 0 {
        let v = serde_json::json!({
            "case": format!("residual/{label}/ks{kw}/N{n}/timing"),
            "m": m, "k": k, "n": n, "kw": kw,
            "timing_precheck": {
                "bit_equal": false, "mismatches": mism,
                "first_mismatch": first_idx,
                "first_scalar_bits": first_s, "first_packed_bits": first_p,
            },
            "pass": false,
        });
        hip.free(d_as).ok();
        hip.free(d_ap).ok();
        hip.free(d_xs).ok();
        hip.free(d_xp).ok();
        hip.free(d_ys).ok();
        hip.free(d_yp).ok();
        return (v, false);
    }

    // Warmups enqueue without per-launch sync; one device sync per arm drains
    // the queue before event timing starts.
    for _ in 0..WARMUPS {
        enqueue_residual(
            hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_ys.as_ptr(), m, k, n, kw,
        );
    }
    hip.device_synchronize().expect("sync after scalar warmups");
    for _ in 0..WARMUPS {
        enqueue_residual(
            hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_yp.as_ptr(), m, k, n, kw,
        );
    }
    hip.device_synchronize().expect("sync after packed warmups");

    let start = hip.event_create().expect("event_create start");
    let stop = hip.event_create().expect("event_create stop");
    let mut s_ms = Vec::with_capacity(SAMPLES_PER_ARM);
    let mut p_ms = Vec::with_capacity(SAMPLES_PER_ARM);
    // ABBA interleaving cancels linear drift between arms. Each bracket
    // enqueues 100 launches with NO per-launch sync; the stop-event
    // synchronize inside time_bracket_ms is the sole completion gate.
    for _ in 0..SAMPLES_PER_ARM / 2 {
        s_ms.push(time_bracket_ms(hip, &start, &stop, &mut || {
            enqueue_residual(
                hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_ys.as_ptr(), m, k, n, kw,
            );
        }));
        p_ms.push(time_bracket_ms(hip, &start, &stop, &mut || {
            enqueue_residual(
                hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_yp.as_ptr(), m, k, n, kw,
            );
        }));
        p_ms.push(time_bracket_ms(hip, &start, &stop, &mut || {
            enqueue_residual(
                hip, f_packed, d_ap.as_ptr(), d_xp.as_ptr(), d_yp.as_ptr(), m, k, n, kw,
            );
        }));
        s_ms.push(time_bracket_ms(hip, &start, &stop, &mut || {
            enqueue_residual(
                hip, f_scalar, d_as.as_ptr(), d_xs.as_ptr(), d_ys.as_ptr(), m, k, n, kw,
            );
        }));
    }
    hip.event_destroy(start).ok();
    hip.event_destroy(stop).ok();

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let v = serde_json::json!({
        "case": format!("residual/{label}/ks{kw}/N{n}/timing"),
        "m": m, "k": k, "n": n, "kw": kw,
        "timing_precheck": {"bit_equal": true, "mismatches": 0},
        "timing": {
            "warmups_per_arm": WARMUPS,
            "launches_per_sample": BRACKET_LAUNCHES,
            "samples_per_arm": SAMPLES_PER_ARM,
            "order": "ABBA",
            "scalar_ms_per_100": s_ms,
            "packed_ms_per_100": p_ms,
            "scalar_mean_ms_per_100": mean(&s_ms),
            "packed_mean_ms_per_100": mean(&p_ms),
            "scalar_us_per_launch": mean(&s_ms) * 10.0,
            "packed_us_per_launch": mean(&p_ms) * 10.0,
        },
        "pass": true,
    });

    hip.free(d_as).ok();
    hip.free(d_ap).ok();
    hip.free(d_xs).ok();
    hip.free(d_xp).ok();
    hip.free(d_ys).ok();
    hip.free(d_yp).ok();
    (v, true)
}

// ── QKVZA case ────────────────────────────────────────────────────────────

fn run_qkvza_parity(
    hip: &HipRuntime,
    f_scalar: &Function,
    f_packed: &Function,
    n: usize,
    blobs: &[Vec<u8>; 4],
    x_f16: &[u16],
) -> (serde_json::Value, bool) {
    let lens = [n * QKV_M, n * Z_M, n * BETA_M, n * ALPHA_M];
    let names = ["qkv", "z", "beta", "alpha"];
    // Full-buffer overwrite canaries, arm-distinct.
    let ys_hosts: Vec<Vec<f32>> = lens
        .iter()
        .map(|&l| fill_canary_f32(l, CANARY_SCALAR))
        .collect();
    let yp_hosts: Vec<Vec<f32>> = lens
        .iter()
        .map(|&l| fill_canary_f32(l, CANARY_PACKED))
        .collect();

    let d_ws: Vec<_> = blobs.iter().map(|b| htod(hip, b)).collect();
    let d_wp: Vec<_> = blobs.iter().map(|b| htod(hip, b)).collect();
    let d_xs = htod_u16(hip, x_f16);
    let d_xp = htod_u16(hip, x_f16);
    let d_ys: Vec<_> = ys_hosts.iter().map(|v| htod_f32(hip, v)).collect();
    let d_yp: Vec<_> = yp_hosts.iter().map(|v| htod_f32(hip, v)).collect();

    let launch_both = |d_w: &[hip_bridge::DeviceBuffer],
                       d_x: &hip_bridge::DeviceBuffer,
                       d_y: &[hip_bridge::DeviceBuffer],
                       f: &Function| {
        launch_qkvza(
            hip,
            f,
            d_w[0].as_ptr(),
            d_w[1].as_ptr(),
            d_w[2].as_ptr(),
            d_w[3].as_ptr(),
            d_x.as_ptr(),
            d_y[0].as_ptr(),
            d_y[1].as_ptr(),
            d_y[2].as_ptr(),
            d_y[3].as_ptr(),
            QKV_M,
            Z_M,
            BETA_M,
            ALPHA_M,
            QKV_K,
            n,
        );
    };
    launch_both(&d_ws, &d_xs, &d_ys, f_scalar);
    launch_both(&d_wp, &d_xp, &d_yp, f_packed);

    let mut projs = Vec::new();
    let mut ok = true;
    let mut scanned: usize = 0;
    for i in 0..4 {
        let ys = dtoh_f32(hip, &d_ys[i], lens[i]);
        let yp = dtoh_f32(hip, &d_yp[i], lens[i]);
        scanned += lens[i] * 4;
        let (mism, first_idx, first_s, first_p) = bit_parity(&ys, &yp);
        let rem_s = count_canary_words(&ys, CANARY_SCALAR);
        let rem_p = count_canary_words(&yp, CANARY_PACKED);
        let pok = mism == 0
            && rem_s == 0
            && rem_p == 0
            && is_finite(&ys)
            && is_finite(&yp)
            && variance(&ys) > 1e-12
            && variance(&yp) > 1e-12;
        if !pok {
            ok = false;
        }
        projs.push(serde_json::json!({
            "proj": names[i],
            "rows": ([QKV_M, Z_M, BETA_M, ALPHA_M][i]),
            "elements": lens[i],
            "bit_equal": mism == 0,
            "mismatches": mism,
            "first_mismatch": first_idx,
            "first_scalar_bits": first_s,
            "first_packed_bits": first_p,
            "rel_l2": rel_l2(&ys, &yp),
            "max_abs": max_abs_diff(&ys, &yp),
            "scalar_finite": is_finite(&ys),
            "packed_finite": is_finite(&yp),
            "scalar_variance": variance(&ys),
            "packed_variance": variance(&yp),
            "scalar_canary_remaining": rem_s,
            "packed_canary_remaining": rem_p,
            "pass": pok,
        }));
    }

    let v = serde_json::json!({
        "case": format!("qkvza/N{n}"),
        "qkv_m": QKV_M, "z_m": Z_M, "beta_m": BETA_M, "alpha_m": ALPHA_M,
        "k": QKV_K, "n": n,
        "projs": projs,
        "canary": {
            "scalar_byte": format!("0x{:02X}", CANARY_SCALAR),
            "packed_byte": format!("0x{:02X}", CANARY_PACKED),
            "bytes_scanned_per_arm": scanned,
            "coverage": "full overwrite: every output element prefilled with canary; zero remaining proves full overwrite on both arms",
        },
        "pass": ok,
    });

    for b in d_ws.into_iter().chain(d_wp) {
        hip.free(b).ok();
    }
    hip.free(d_xs).ok();
    hip.free(d_xp).ok();
    for b in d_ys.into_iter().chain(d_yp) {
        hip.free(b).ok();
    }
    (v, ok)
}

fn run_qkvza_timing(
    hip: &HipRuntime,
    f_scalar: &Function,
    f_packed: &Function,
    n: usize,
    blobs: &[Vec<u8>; 4],
    x_f16: &[u16],
) -> (serde_json::Value, bool) {
    let lens = [n * QKV_M, n * Z_M, n * BETA_M, n * ALPHA_M];
    let d_ws: Vec<_> = blobs.iter().map(|b| htod(hip, b)).collect();
    let d_wp: Vec<_> = blobs.iter().map(|b| htod(hip, b)).collect();
    let d_xs = htod_u16(hip, x_f16);
    let d_xp = htod_u16(hip, x_f16);
    let zeros: Vec<Vec<f32>> = lens.iter().map(|&l| vec![0.0f32; l]).collect();
    let d_ys: Vec<_> = zeros.iter().map(|v| htod_f32(hip, v)).collect();
    let d_yp: Vec<_> = zeros.iter().map(|v| htod_f32(hip, v)).collect();

    // Enqueue-only form: NO per-launch sync inside timed brackets or warmup
    // queues; the bracket stop-event (or the explicit syncs below) completes.
    let go = |d_w: &[hip_bridge::DeviceBuffer],
              d_x: &hip_bridge::DeviceBuffer,
              d_y: &[hip_bridge::DeviceBuffer],
              f: &Function| {
        enqueue_qkvza(
            hip,
            f,
            d_w[0].as_ptr(),
            d_w[1].as_ptr(),
            d_w[2].as_ptr(),
            d_w[3].as_ptr(),
            d_x.as_ptr(),
            d_y[0].as_ptr(),
            d_y[1].as_ptr(),
            d_y[2].as_ptr(),
            d_y[3].as_ptr(),
            QKV_M,
            Z_M,
            BETA_M,
            ALPHA_M,
            QKV_K,
            n,
        );
    };
    // Full timed outputs verified BEFORE timing (all four projections).
    go(&d_ws, &d_xs, &d_ys, f_scalar);
    go(&d_wp, &d_xp, &d_yp, f_packed);
    let mut pre_ok = true;
    let mut pre_mism = 0usize;
    for i in 0..4 {
        let ys = dtoh_f32(hip, &d_ys[i], lens[i]);
        let yp = dtoh_f32(hip, &d_yp[i], lens[i]);
        let (mism, _, _, _) = bit_parity(&ys, &yp);
        pre_mism += mism;
        if mism != 0 {
            pre_ok = false;
        }
    }
    if !pre_ok {
        let v = serde_json::json!({
            "case": format!("qkvza/N{n}/timing"),
            "timing_precheck": {"bit_equal": false, "total_mismatches": pre_mism},
            "pass": false,
        });
        for b in d_ws.into_iter().chain(d_wp) {
            hip.free(b).ok();
        }
        hip.free(d_xs).ok();
        hip.free(d_xp).ok();
        for b in d_ys.into_iter().chain(d_yp) {
            hip.free(b).ok();
        }
        return (v, false);
    }

    for _ in 0..WARMUPS {
        go(&d_ws, &d_xs, &d_ys, f_scalar);
    }
    hip.device_synchronize().expect("sync after scalar warmups");
    for _ in 0..WARMUPS {
        go(&d_wp, &d_xp, &d_yp, f_packed);
    }
    hip.device_synchronize().expect("sync after packed warmups");

    let start = hip.event_create().expect("event_create start");
    let stop = hip.event_create().expect("event_create stop");
    let mut s_ms = Vec::with_capacity(SAMPLES_PER_ARM);
    let mut p_ms = Vec::with_capacity(SAMPLES_PER_ARM);
    for _ in 0..SAMPLES_PER_ARM / 2 {
        s_ms.push(time_bracket_ms(hip, &start, &stop,
            &mut || go(&d_ws, &d_xs, &d_ys, f_scalar)));
        p_ms.push(time_bracket_ms(hip, &start, &stop,
            &mut || go(&d_wp, &d_xp, &d_yp, f_packed)));
        p_ms.push(time_bracket_ms(hip, &start, &stop,
            &mut || go(&d_wp, &d_xp, &d_yp, f_packed)));
        s_ms.push(time_bracket_ms(hip, &start, &stop,
            &mut || go(&d_ws, &d_xs, &d_ys, f_scalar)));
    }
    hip.event_destroy(start).ok();
    hip.event_destroy(stop).ok();

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let v = serde_json::json!({
        "case": format!("qkvza/N{n}/timing"),
        "qkv_m": QKV_M, "z_m": Z_M, "beta_m": BETA_M, "alpha_m": ALPHA_M,
        "k": QKV_K, "n": n,
        "timing_precheck": {"bit_equal": true, "total_mismatches": 0},
        "timing": {
            "warmups_per_arm": WARMUPS,
            "launches_per_sample": BRACKET_LAUNCHES,
            "samples_per_arm": SAMPLES_PER_ARM,
            "order": "ABBA",
            "scalar_ms_per_100": s_ms,
            "packed_ms_per_100": p_ms,
            "scalar_mean_ms_per_100": mean(&s_ms),
            "packed_mean_ms_per_100": mean(&p_ms),
            "scalar_us_per_launch": mean(&s_ms) * 10.0,
            "packed_us_per_launch": mean(&p_ms) * 10.0,
        },
        "pass": true,
    });

    for b in d_ws.into_iter().chain(d_wp) {
        hip.free(b).ok();
    }
    hip.free(d_xs).ok();
    hip.free(d_xp).ok();
    for b in d_ys.into_iter().chain(d_yp) {
        hip.free(b).ok();
    }
    (v, true)
}

// ── CLI / driver ──────────────────────────────────────────────────────────

fn arg_val(args: &[String], name: &str) -> String {
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == name {
            return it
                .next()
                .unwrap_or_else(|| panic!("{name} requires a value"))
                .clone();
        }
        if let Some(v) = a.strip_prefix(&format!("{name}=")) {
            return v.to_string();
        }
    }
    panic!("missing required arg {name} (want --scalar ELF --packed ELF --kind residual|qkvza --out PATH)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scalar_elf = arg_val(&args, "--scalar");
    let packed_elf = arg_val(&args, "--packed");
    let kind = arg_val(&args, "--kind");
    let out = arg_val(&args, "--out");
    assert!(
        kind == "residual" || kind == "qkvza",
        "--kind must be residual|qkvza (got {kind})"
    );
    assert!(
        scalar_elf != packed_elf,
        "scalar and packed ELFs must be distinct paths"
    );

    let hip = HipRuntime::load().expect("HipRuntime::load failed");
    hip.set_device(0).expect("hipSetDevice(0) failed");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "unknown".to_string());

    // Two INDEPENDENT module handles from two ELF paths — never a cached
    // module alias: each arm resolves its symbols from its own handle.
    let scalar_mod: Module = hip
        .module_load(&scalar_elf)
        .expect("hipModuleLoad(scalar) failed");
    let packed_mod: Module = hip
        .module_load(&packed_elf)
        .expect("hipModuleLoad(packed) failed");
    // Both handles outlive every launch below; each arm resolves symbols from
    // its own handle, never a shared module cache.

    let mut cases: Vec<serde_json::Value> = Vec::new();
    // Fail-nonzero on the FIRST mismatch/canary failure: write JSON so far.
    let fail_here = |cases: &Vec<serde_json::Value>,
                     kind: &str,
                     scalar_elf: &str,
                     packed_elf: &str,
                     arch: &str,
                     out: &str| -> ! {
        let report = serde_json::json!({
            "probe": "xtx_packed_kernel_probe",
            "kind": kind,
            "scalar_elf": scalar_elf,
            "packed_elf": packed_elf,
            "arch": arch,
            "result": "FAIL",
            "cases": cases,
        });
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
            .expect("write --out failed");
        eprintln!("FAIL: first mismatch/canary failure recorded in {out}");
        std::process::exit(1);
    };

    if arch != "gfx1100" {
        let report = serde_json::json!({
            "probe": "xtx_packed_kernel_probe",
            "kind": kind,
            "scalar_elf": scalar_elf,
            "packed_elf": packed_elf,
            "arch": arch,
            "result": "SKIP",
            "reason": "exact gfx1100 required",
            "cases": cases,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
            .expect("write --out failed");
        eprintln!("SKIP: arch {arch} is not exact gfx1100");
        return;
    }

    if kind == "residual" {
        let mut f_scalar = Vec::new();
        let mut f_packed = Vec::new();
        for &kw in &RESID_KS {
            f_scalar.push(
                hip.module_get_function(&scalar_mod, resid_symbol(kw))
                    .unwrap_or_else(|e| panic!("scalar missing {}: {e:?}", resid_symbol(kw))),
            );
            f_packed.push(
                hip.module_get_function(&packed_mod, resid_symbol(kw))
                    .unwrap_or_else(|e| panic!("packed missing {}: {e:?}", resid_symbol(kw))),
            );
        }
        let shapes = [
            ("out_proj", RESID_OUT_M, RESID_OUT_K),
            ("down_proj", RESID_DOWN_M, RESID_DOWN_K),
        ];
        for (label, m, k) in shapes {
            let g = k / 256;
            let w_blob = pack_weights_deterministic(m, k, 0x5EED_0000 + k as u64);
            for (ki, &kw) in RESID_KS.iter().enumerate() {
                if g < kw || g % kw != 0 {
                    cases.push(serde_json::json!({
                        "case": format!("residual/{label}/ks{kw}"),
                        "m": m, "k": k, "kw": kw,
                        "skipped": format!("K/256={g} not divisible by kw={kw} (kernel-design contract)"),
                    }));
                    continue;
                }
                for &n in &RESID_NS {
                    let x_f32: Vec<f32> = (0..n * k)
                        .map(|i| prng(i, 0xC0FF_EE00 ^ k as u32) * 2.0 - 1.0)
                        .collect();
                    let x_f16 = stage_x_f16(&x_f32);
                    let mut y_init =
                        random_f32(n * m, 0xBEEF_1234 + n as u64 + k as u64, -0.5, 1.5);
                    // Signed-zero exercise in the += init.
                    if y_init.len() >= 2 {
                        y_init[0] = -0.0;
                        y_init[1] = 0.0;
                    }
                    let (v, ok) = run_residual_parity(
                        &hip, &f_scalar[ki], &f_packed[ki], label, m, k, n, kw, &w_blob,
                        &x_f16, &y_init,
                    );
                    let pass = v["pass"].as_bool().unwrap_or(false);
                    cases.push(v);
                    if !ok || !pass {
                        fail_here(&cases, &kind, &scalar_elf, &packed_elf, &arch, &out);
                    }
                    eprintln!("PASS residual/{label}/ks{kw}/N{n}");
                }
            }
        }
        // Timed ks4 on the real out/down shapes at N=16.
        for (label, m, k) in shapes {
            let w_blob = pack_weights_deterministic(m, k, 0x5EED_0000 + k as u64);
            let n = 16;
            let x_f32: Vec<f32> = (0..n * k)
                .map(|i| prng(i, 0xC0FF_EE00 ^ k as u32) * 2.0 - 1.0)
                .collect();
            let x_f16 = stage_x_f16(&x_f32);
            let y_init = random_f32(n * m, 0xBEEF_1234 + n as u64 + k as u64, -0.5, 1.5);
            let (v, ok) =
                run_residual_timing(&hip, &f_scalar[1], &f_packed[1], label, m, k, n, 4, &w_blob, &x_f16, &y_init);
            let pass = v["pass"].as_bool().unwrap_or(false);
            cases.push(v);
            if !ok || !pass {
                fail_here(&cases, &kind, &scalar_elf, &packed_elf, &arch, &out);
            }
            eprintln!("TIMED residual/{label}/ks4/N{n}");
        }
    } else {
        let f_scalar: Function = hip
            .module_get_function(&scalar_mod, "gemm_qkvza_mq4g256v2_wmma")
            .expect("scalar missing gemm_qkvza_mq4g256v2_wmma");
        let f_packed: Function = hip
            .module_get_function(&packed_mod, "gemm_qkvza_mq4g256v2_wmma")
            .expect("packed missing gemm_qkvza_mq4g256v2_wmma");
        // Distinct deterministic weights per projection: swapped routing
        // cannot bit-match.
        let blobs = [
            pack_weights_deterministic(QKV_M, QKV_K, 0x1111_2222),
            pack_weights_deterministic(Z_M, QKV_K, 0x3333_4444),
            pack_weights_deterministic(BETA_M, QKV_K, 0x5555_6666),
            pack_weights_deterministic(ALPHA_M, QKV_K, 0x7777_8888),
        ];
        assert_ne!(blobs[0], blobs[1], "qkv/z blobs must differ");
        assert_ne!(blobs[2], blobs[3], "beta/alpha blobs must differ");
        for &n in &QKV_NS {
            let x_f32: Vec<f32> = (0..n * QKV_K)
                .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
                .collect();
            let x_f16 = stage_x_f16(&x_f32);
            let (v, ok) = run_qkvza_parity(&hip, &f_scalar, &f_packed, n, &blobs, &x_f16);
            let pass = v["pass"].as_bool().unwrap_or(false);
            cases.push(v);
            if !ok || !pass {
                fail_here(&cases, &kind, &scalar_elf, &packed_elf, &arch, &out);
            }
            eprintln!("PASS qkvza/N{n}");
        }
        // Timing at N=16.
        let n = 16;
        let x_f32: Vec<f32> = (0..n * QKV_K)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        let x_f16 = stage_x_f16(&x_f32);
        let (v, ok) = run_qkvza_timing(&hip, &f_scalar, &f_packed, n, &blobs, &x_f16);
        let pass = v["pass"].as_bool().unwrap_or(false);
        cases.push(v);
        if !ok || !pass {
            fail_here(&cases, &kind, &scalar_elf, &packed_elf, &arch, &out);
        }
        eprintln!("TIMED qkvza/N{n}");
    }

    let report = serde_json::json!({
        "probe": "xtx_packed_kernel_probe",
        "kind": kind,
        "scalar_elf": scalar_elf,
        "packed_elf": packed_elf,
        "arch": arch,
        "canaries": {"scalar": "0xCD", "packed": "0xAB"},
        "timing": {
            "warmups_per_arm": WARMUPS,
            "launches_per_sample": BRACKET_LAUNCHES,
            "samples_per_arm": SAMPLES_PER_ARM,
            "order": "ABBA",
            "mechanism": "hipEvent record/event_elapsed_ms with stop-event completion gate",
        },
        "result": "PASS",
        "cases": cases,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .expect("write --out failed");
    eprintln!("PASS: all cases bit-equal, canaries intact; report in {out}");
}
