// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Support V device-memcmp oracle: frozen legacy vs row-serial candidate.
//!
//! Bit-exactness gate for Slice 1 (`HIPFIRE_GFX1151_RESIDUAL_ROW_SERIAL`).
//! Builds two modules from the same
//! `include_str!("../../../kernels/src/gemv_mq4g256v2_residual.hip")` plus an
//! identical cache-policy prefix:
//! - frozen legacy: `HIPFIRE_RESIDUAL_KERNEL =
//!   gemv_mq4g256v2_residual_legacy_gfx1151`, selector absent (the
//!   simultaneous eight-header schedule — the pre-Slice-1 gfx1151 path);
//! - candidate: selector present, `HIPFIRE_RESIDUAL_KERNEL =
//!   gemv_mq4g256v2_residual_row_serial_gfx1151` (the gfx1100 row0-load/FMA,
//!   barrier, row1-load/FMA schedule; the exact production
//!   `GEMV_MQ4G256V2_RESIDUAL_ROW_SERIAL_GFX1151_SRC` composition).
//!
//! Both launch with the frozen 32-byte ABI (A/x/y/M/K), grid `ceil(M/2)`,
//! block 32, LDS 0, via `Gpu::ensure_kernel_public` +
//! `Gpu::launch_kernel_blob`. All M output f32 bit patterns are compared ON
//! DEVICE by an inline `memcmp_u32_flags` kernel (one 0/1 flag per thread, no
//! atomics); only flags are downloaded and all must be zero. Host asserts
//! finite/nondegenerate outputs, unchanged guard canaries past M, and that
//! the legacy output actually moved off the initial Y (non-vacuity).
//!
//! Mandatory shapes: (5120,5120), (5120,6144), (5120,17408) and the odd-tail
//! (5119,5120), all with deterministic nonzero initial Y. Any mismatch exits
//! nonzero.
//!
//! Run (Halo):
//!   HOME=/tmp/home-lloyd-hipx ROCR_VISIBLE_DEVICES=1 \
//!     cargo run --release -p rdna-compute \
//!       --example test_mq4v2_residual_row_serial_gfx1151
use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
/// Guard floats past M per Y buffer; must be bit-unchanged after launch.
const GUARD: usize = 32;
/// Distinct NaN sentinel for guards (never a legal output).
const SENTINEL: u32 = 0x7FC0_00FF;

const CACHE_POLICY_SRC: &str =
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc");
const RESIDUAL_BODY_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256v2_residual.hip");

const LEGACY_ENTRY: &str = "gemv_mq4g256v2_residual_legacy_gfx1151";
const CAND_ENTRY: &str = "gemv_mq4g256v2_residual_row_serial_gfx1151";
const MEMCMP_ENTRY: &str = "memcmp_u32_flags";

const MEMCMP_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void memcmp_u32_flags(
    const unsigned int* __restrict__ a,
    const unsigned int* __restrict__ b,
    unsigned int* __restrict__ flags,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = (a[i] != b[i]) ? 1u : 0u;
}
"#;

fn legacy_src() -> String {
    format!(
        "#define HIPFIRE_RESIDUAL_KERNEL {}\n#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n{}\n{}",
        LEGACY_ENTRY, CACHE_POLICY_SRC, RESIDUAL_BODY_SRC
    )
}

fn cand_src() -> String {
    format!(
        "#define HIPFIRE_GFX1151_RESIDUAL_ROW_SERIAL 1\n#define HIPFIRE_RESIDUAL_KERNEL {}\n#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n{}\n{}",
        CAND_ENTRY, CACHE_POLICY_SRC, RESIDUAL_BODY_SRC
    )
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// Disjoint-half weights (half0 in [-1,1], half1 in [96,160]) so a wrong
/// half-header or lane test is unmissable.
fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                let lo_q = codes[2 * i] & 0xF;
                let hi_q = codes[2 * i + 1] & 0xF;
                blob[dst + 8 + i] = lo_q | (hi_q << 4);
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
    v.iter()
        .map(|x| (*x as f64 - mean).powi(2))
        .sum::<f64>()
        / v.len() as f64
}

fn htod(gpu: &Gpu, dst: &rdna_compute::GpuTensor, host_bytes: &[u8], what: &str) {
    gpu.hip
        .memcpy_htod(&dst.buf, host_bytes)
        .unwrap_or_else(|e| panic!("htod {what}: {e:?}"));
    gpu.hip
        .device_synchronize()
        .unwrap_or_else(|e| panic!("sync after htod {what}: {e:?}"));
}

fn launch_residual(
    gpu: &mut Gpu,
    entry: &str,
    a_ptr: *const c_void,
    x_ptr: *const c_void,
    y_ptr: *const c_void,
    m: usize,
    k: usize,
) {
    let mut kb = KernargBlob::new();
    kb.push_ptr(a_ptr);
    kb.push_ptr(x_ptr);
    kb.push_ptr(y_ptr);
    kb.push_i32(m as i32);
    kb.push_i32(k as i32);
    assert_eq!(
        kb.len(),
        32,
        "residual ABI must stay exactly 32 bytes (A/x/y/M/K)"
    );
    let grid = m.div_ceil(2) as u32;
    gpu.launch_kernel_blob(entry, [grid, 1, 1], [32, 1, 1], 0, kb.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {entry} M={m} K={k}: {e:?}"));
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize) -> bool {
    assert_eq!(k % GROUP, 0, "K must be a multiple of 256");
    eprintln!("--- shape M={m} K={k} ---");

    let w = build_disjoint_halves(m, k);
    let blob = pack_mq4g256v2(&w, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).expect("upload A");

    let x_host: Vec<f32> = (0..k).map(|i| prng(i, 0xC0DE_0002) * 2.0 - 1.0).collect();
    let d_x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
    htod(
        gpu,
        &d_x,
        unsafe { std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4) },
        "x",
    );

    // Deterministic strictly-nonzero initial Y.
    let y_init: Vec<f32> = (0..m)
        .map(|i| {
            let v = 0.25 + prng(i, 0x5EED_0001) * 3.0;
            if i % 2 == 0 {
                v
            } else {
                -v
            }
        })
        .collect();
    assert!(y_init.iter().all(|x| *x != 0.0 && x.is_finite()));

    let mut y_host = Vec::with_capacity(m + GUARD);
    y_host.extend_from_slice(&y_init);
    y_host.extend(std::iter::repeat(f32::from_bits(SENTINEL)).take(GUARD));
    let y_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(y_host.as_ptr() as *const u8, y_host.len() * 4) };

    let d_y_leg = gpu
        .alloc_tensor(&[m + GUARD], DType::F32)
        .expect("alloc y legacy");
    htod(gpu, &d_y_leg, y_bytes, "y legacy");
    let d_y_cand = gpu
        .alloc_tensor(&[m + GUARD], DType::F32)
        .expect("alloc y cand");
    htod(gpu, &d_y_cand, y_bytes, "y cand");

    let a_ptr = d_a.buf.as_ptr() as *const c_void;
    let x_ptr = d_x.buf.as_ptr() as *const c_void;
    launch_residual(
        gpu,
        LEGACY_ENTRY,
        a_ptr,
        x_ptr,
        d_y_leg.buf.as_ptr() as *const c_void,
        m,
        k,
    );
    launch_residual(
        gpu,
        CAND_ENTRY,
        a_ptr,
        x_ptr,
        d_y_cand.buf.as_ptr() as *const c_void,
        m,
        k,
    );
    gpu.hip.device_synchronize().expect("sync residuals");

    // On-device u32 compare of all M outputs; download flags only.
    let d_flags = gpu.zeros(&[m], DType::F32).expect("alloc flags");
    let mut kb = KernargBlob::new();
    kb.push_ptr(d_y_leg.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_y_cand.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_flags.buf.as_ptr() as *const c_void);
    kb.push_i32(m as i32);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        MEMCMP_ENTRY,
        [m.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch memcmp");
    gpu.hip.device_synchronize().expect("sync memcmp");
    let flags = gpu.download_f32(&d_flags).expect("download flags");
    let mism = flags.iter().filter(|x| x.to_bits() != 0).count();

    // Host-side health: finite/nondegenerate, canaries, non-vacuity.
    let y_leg = gpu.download_f32(&d_y_leg).expect("download y legacy");
    let y_cand = gpu.download_f32(&d_y_cand).expect("download y cand");
    let (out_leg, guard_leg) = y_leg.split_at(m);
    let (out_cand, guard_cand) = y_cand.split_at(m);
    let finite = is_finite(out_leg) && is_finite(out_cand);
    let var_leg = variance(out_leg);
    let var_cand = variance(out_cand);
    let nondeg = var_leg > 1e-12 && var_cand > 1e-12;
    let canary_ok = guard_leg.iter().all(|x| x.to_bits() == SENTINEL)
        && guard_cand.iter().all(|x| x.to_bits() == SENTINEL);
    let moved = out_leg
        .iter()
        .zip(y_init.iter())
        .filter(|(g, y)| g.to_bits() != y.to_bits())
        .count();
    let max_abs = out_leg
        .iter()
        .zip(out_cand.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let ok = mism == 0 && finite && nondeg && canary_ok && moved > m / 2;
    eprintln!(
        "  M={m} K={k}: device-mismatches={mism}/{m} maxAbsDiff={max_abs:e} finite={finite} \
         varLeg={var_leg:.3e} varCand={var_cand:.3e} canaries={canary_ok} movedOffInit={moved}/{m} -> {}",
        if ok { "OK" } else { "FAIL" }
    );
    if mism != 0 {
        let idx: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|(_, x)| x.to_bits() != 0)
            .map(|(i, _)| i)
            .take(8)
            .collect();
        eprintln!("    first mismatch rows: {idx:?}");
    }
    ok
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FAIL: Gpu::init failed ({e:?})");
            std::process::exit(1);
        }
    };
    eprintln!("arch {} — legacy-vs-rowserial oracle", gpu.arch);
    gpu.ensure_kernel_public(LEGACY_ENTRY, &legacy_src(), LEGACY_ENTRY)
        .expect("compile legacy module");
    gpu.ensure_kernel_public(CAND_ENTRY, &cand_src(), CAND_ENTRY)
        .expect("compile candidate module");
    gpu.ensure_kernel_public(MEMCMP_ENTRY, MEMCMP_SRC, MEMCMP_ENTRY)
        .expect("compile memcmp module");

    let shapes = [(5120usize, 5120usize), (5120, 6144), (5120, 17408), (5119, 5120)];
    let mut all_ok = true;
    for (m, k) in shapes {
        if !run_shape(&mut gpu, m, k) {
            all_ok = false;
        }
    }
    if all_ok {
        eprintln!("PASS: legacy == row-serial bit-exact on all 4 shapes (device memcmp, 0 flags)");
    } else {
        eprintln!("FAIL: legacy-vs-rowserial mismatch (see above)");
        std::process::exit(1);
    }
}
