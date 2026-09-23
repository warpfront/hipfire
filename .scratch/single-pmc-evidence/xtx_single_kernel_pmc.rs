// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Temporary one-shot gfx1100 probe: exactly ONE residual ks4 launch, no timing.
//
// Usage:
//   cargo build -p hipfire-runtime --example xtx_single_kernel_pmc
//   ./target/debug/examples/xtx_single_kernel_pmc \
//     --elf /path/to/packed-residual.gfx1100.o --out /tmp/xtx_single_ks4.json
//
// Source grounding (ABIs read from sources, never guessed):
// - Residual split-K ABI `(A, X, Y, M, K, N)`, grid `[ceil(M/16),
//   ceil(N/16)]`, block `[32*KW]`, symbol
//   `gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds` from
//   `kernels/src/gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds.hip`
//   (`GEN_RESID_KSPLIT_LDS`: `const char* A, const _Float16* X,
//   float* Y, int M, int K, int N`).
// - MQ4V2 layout (136 B groups, dual fp16 headers, nibble payload),
//   deterministic X staging, and weight packing copied from the archived
//   `.scratch/xtx_packed_kernel_probe.rs` helpers (`pack_weights_deterministic`,
//   `stage_x_f16`, `f32_to_f16_bits_rne`, `prng`/`random_f32`).
// - Raw HIP surface (`HipRuntime::module_load`, `module_get_function`,
//   `launch_kernel`, `malloc`/`memcpy_*`/`free`, `device_synchronize`,
//   `get_arch`) from `crates/hip-bridge/src/ffi.rs`.
//
// Contract: one explicit module handle, one `launch_kernel` total (ks4,
// block 128, grid [ceil(M/16), ceil(N/16)]), shape out_proj M=5120 K=6144
// N=16, init via HIP memcpy only, sync after launch, finite check, free
// after proved sync, no retry, no capture/replay, no second GPU context,
// no model/GpuController dependency. Parent runs plain HIP first, then a
// minimal SQ_WAVES counter-only pass against this same shape/symbol.
// A single cold launch proves liveness only; it makes NO perf claim.

use hip_bridge::HipRuntime;
use std::ffi::c_void;
use std::io::Write;

// ── Shape / kernel (out_proj production shape) ──────────────────────────────

const M: usize = 5120;
const K: usize = 6144;
const N: usize = 16;
const KW: usize = 4;
const KERNEL: &str = "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds";

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;
const MAX_TOTAL_BYTES: usize = 128 << 20; // 128 MiB hard cap

// ── Deterministic host helpers (copied semantics from archived probe) ───────

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

/// f32 -> IEEE binary16 bits, round-to-nearest-even.
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

fn stage_x_f16(x_f32: &[f32]) -> Vec<u16> {
    x_f32.iter().map(|&v| f32_to_f16_bits_rne(v)).collect()
}

/// Deterministic MQ4V2 blob: valid finite nonzero fp16 scales, zero-points
/// from {+0.0, -0.0, 0.3, -0.3}; nibble payload cycles 0..255. Non-power-of-two
/// headers (0.1/1/3/0.7 scales, 0.3/-0.3 zero-points) force fp16 rounding.
fn pack_weights_deterministic(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let gpr = k / GROUP;
    const SCALES: [u16; 4] = [0x3400, 0x2E66, 0x3555, 0x399A];
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
                    (((r * gpr + g) as u64 * 128 + j as u64 + seed) % 256) as u8;
            }
        }
    }
    blob
}

fn arg_val(args: &[String], name: &str) -> String {
    let flag = format!("{name}=");
    for a in args {
        if let Some(v) = a.strip_prefix(&flag) {
            return v.to_string();
        }
    }
    let pos = args
        .iter()
        .position(|a| a == name)
        .unwrap_or_else(|| panic!("missing required arg {name} PATH"));
    args.get(pos + 1)
        .unwrap_or_else(|| panic!("missing value for {name}"))
        .clone()
}

fn write_report_and_exit(out: &str, report: serde_json::Value, code: i32) -> ! {
    if let Err(e) = std::fs::write(out, serde_json::to_string_pretty(&report).unwrap()) {
        eprintln!("FAIL: could not write --out {out}: {e:?}");
    }
    std::process::exit(code);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let elf = arg_val(&args, "--elf");
    let out = arg_val(&args, "--out");

    // Hang-stage markers: PID first, then BEFORE_LAUNCH / AFTER_SYNC, flushed.
    println!("PID {}", std::process::id());
    std::io::stdout().flush().ok();

    let hip = HipRuntime::load().expect("HipRuntime::load failed");
    hip.set_device(0).expect("hipSetDevice(0) failed");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "unknown".to_string());

    // Metrics/report require the actual gfx1100 device; anything else fails.
    if arch != "gfx1100" {
        let report = serde_json::json!({
            "probe": "xtx_single_kernel_pmc",
            "elf": elf,
            "arch": arch,
            "shape": {"m": M, "k": K, "n": N},
            "kernel": KERNEL,
            "result": "FAIL",
            "reason": "exact gfx1100 required",
        });
        eprintln!("FAIL: arch {arch} is not exact gfx1100");
        write_report_and_exit(&out, report, 1);
    }

    // One explicit module handle from the packed ELF path.
    let module = match hip.module_load(&elf) {
        Ok(m) => m,
        Err(e) => {
            let report = serde_json::json!({
                "probe": "xtx_single_kernel_pmc",
                "elf": elf,
                "arch": arch,
                "shape": {"m": M, "k": K, "n": N},
                "kernel": KERNEL,
                "result": "FAIL",
                "reason": format!("hipModuleLoad failed: {e:?}"),
            });
            eprintln!("FAIL: hipModuleLoad({elf}) failed: {e:?}");
            write_report_and_exit(&out, report, 1);
        }
    };
    let func = match hip.module_get_function(&module, KERNEL) {
        Ok(f) => f,
        Err(e) => {
            let report = serde_json::json!({
                "probe": "xtx_single_kernel_pmc",
                "elf": elf,
                "arch": arch,
                "shape": {"m": M, "k": K, "n": N},
                "kernel": KERNEL,
                "result": "FAIL",
                "reason": format!("missing symbol {KERNEL}: {e:?}"),
            });
            eprintln!("FAIL: module missing {KERNEL}: {e:?}");
            write_report_and_exit(&out, report, 1);
        }
    };

    // Deterministic inputs: valid MQ4V2 scales/zeros, X in [-1, 1] staged to
    // f16 exactly as the production staging path does.
    let w_blob = pack_weights_deterministic(M, K, 0x5EED_0000 + K as u64);
    let x_f32: Vec<f32> = (0..N * K)
        .map(|i| prng(i, 0xC0FF_EE00 ^ K as u32) * 2.0 - 1.0)
        .collect();
    let x_f16 = stage_x_f16(&x_f32);
    let mut y_init = random_f32(N * M, 0xBEEF_1234 + N as u64 + K as u64, -0.5, 1.5);
    if y_init.len() >= 2 {
        y_init[0] = -0.0;
        y_init[1] = 0.0;
    }

    let a_bytes = w_blob.len();
    let x_bytes = x_f16.len() * 2;
    let y_bytes = y_init.len() * 4;
    let total_bytes = a_bytes + x_bytes + y_bytes;
    let max_allocation_bytes = a_bytes.max(x_bytes).max(y_bytes);
    assert!(
        total_bytes <= MAX_TOTAL_BYTES,
        "probe over budget: {total_bytes} > {MAX_TOTAL_BYTES}"
    );

    // Initialize device state via HIP memcpy only (no kernels, no staging
    // launches, no second context).
    let buf_a = hip.malloc(a_bytes).expect("hipMalloc(A) failed");
    hip.memcpy_htod(&buf_a, &w_blob).expect("memcpy_htod(A) failed");
    let x_host_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x_f16.as_ptr() as *const u8, x_bytes)
    };
    let buf_x = hip.malloc(x_bytes).expect("hipMalloc(X) failed");
    hip.memcpy_htod(&buf_x, x_host_bytes)
        .expect("memcpy_htod(X) failed");
    let y_host_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(y_init.as_ptr() as *const u8, y_bytes)
    };
    let buf_y = hip.malloc(y_bytes).expect("hipMalloc(Y) failed");
    hip.memcpy_htod(&buf_y, y_host_bytes)
        .expect("memcpy_htod(Y) failed");
    let mut a_ptr = buf_a.as_ptr() as *mut c_void;
    let mut x_ptr = buf_x.as_ptr() as *mut c_void;
    let mut y_ptr = buf_y.as_ptr() as *mut c_void;
    let mut mv = M as i32;
    let mut kv = K as i32;
    let mut nv = N as i32;
    let mut params: Vec<*mut c_void> = vec![
        &mut a_ptr as *mut _ as *mut c_void,
        &mut x_ptr as *mut _ as *mut c_void,
        &mut y_ptr as *mut _ as *mut c_void,
        &mut mv as *mut _ as *mut c_void,
        &mut kv as *mut _ as *mut c_void,
        &mut nv as *mut _ as *mut c_void,
    ];
    let grid = [((M + 15) / 16) as u32, ((N + 15) / 16) as u32, 1];
    let block = [(32 * KW) as u32, 1, 1];

    println!("BEFORE_LAUNCH kernel={KERNEL} grid=[{}, {}, 1] block=[{}, 1, 1]", grid[0], grid[1], block[0]);
    std::io::stdout().flush().ok();

    // THE one and only kernel submission of this process. No warmups, no
    // loop, no retry: a launch error is a FAIL report, never a second submit.
    if let Err(e) = unsafe { hip.launch_kernel(&func, grid, block, 0, None, &mut params) } {
        let report = serde_json::json!({
            "probe": "xtx_single_kernel_pmc",
            "elf": elf,
            "arch": arch,
            "shape": {"m": M, "k": K, "n": N, "kw": KW},
            "kernel": KERNEL,
            "grid": [grid[0], grid[1], grid[2]],
            "block": [block[0], block[1], block[2]],
            "result": "FAIL",
            "reason": format!("hipModuleLaunchKernel failed: {e:?}"),
        });
        eprintln!("FAIL: launch {KERNEL} failed: {e:?}");
        write_report_and_exit(&out, report, 1);
    }
    if let Err(e) = hip.device_synchronize() {
        let report = serde_json::json!({
            "probe": "xtx_single_kernel_pmc",
            "elf": elf,
            "arch": arch,
            "shape": {"m": M, "k": K, "n": N, "kw": KW},
            "kernel": KERNEL,
            "result": "FAIL",
            "reason": format!("device_synchronize failed: {e:?}"),
        });
        eprintln!("FAIL: sync after {KERNEL} failed: {e:?}");
        write_report_and_exit(&out, report, 1);
    }

    println!("AFTER_SYNC kernel={KERNEL}");
    std::io::stdout().flush().ok();

    // Copy out and prove finite AFTER the sync above; then free.
    let mut y_out = vec![0.0f32; N * M];
    let y_out_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(y_out.as_mut_ptr() as *mut u8, y_bytes)
    };
    hip.memcpy_dtoh(y_out_bytes, &buf_y)
        .expect("memcpy_dtoh(Y) failed");
    hip.free(buf_a).expect("hipFree(A) failed");
    hip.free(buf_x).expect("hipFree(X) failed");
    hip.free(buf_y).expect("hipFree(Y) failed");
    drop(module);

    let finite = y_out.iter().all(|v| v.is_finite());
    if !finite {
        let bad = y_out.iter().filter(|v| !v.is_finite()).count();
        let report = serde_json::json!({
            "probe": "xtx_single_kernel_pmc",
            "elf": elf,
            "arch": arch,
            "shape": {"m": M, "k": K, "n": N, "kw": KW},
            "kernel": KERNEL,
            "grid": [grid[0], grid[1], grid[2]],
            "block": [block[0], block[1], block[2]],
            "finite": false,
            "nonfinite_count": bad,
            "result": "FAIL",
            "reason": "non-finite outputs after synced single launch",
        });
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
            .expect("write --out failed");
        eprintln!("FAIL: {bad} non-finite outputs; report in {out}");
        std::process::exit(1);
    }

    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in &y_out {
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    let mean = sum / y_out.len() as f64;
    let first8: Vec<f32> = y_out.iter().copied().take(8).collect();

    let report = serde_json::json!({
        "probe": "xtx_single_kernel_pmc",
        "elf": elf,
        "arch": arch,
        "shape": {"m": M, "k": K, "n": N, "kw": KW},
        "kernel": KERNEL,
        "grid": [grid[0], grid[1], grid[2]],
        "block": [block[0], block[1], block[2]],
        "launches": 1,
        "steps": ["PID", "BEFORE_LAUNCH", "AFTER_SYNC"],
        "allocations": {
            "a_bytes": a_bytes,
            "x_bytes": x_bytes,
            "y_bytes": y_bytes,
            "total_bytes": total_bytes,
            "max_allocation_bytes": max_allocation_bytes,
            "budget_bytes": MAX_TOTAL_BYTES,
        },
        "finite": true,
        "output": {
            "len": y_out.len(),
            "min": min,
            "max": max,
            "mean": mean,
            "first8": first8,
        },
        "perf_claim": "none: single cold launch proves liveness only",
        "result": "PASS",
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .expect("write --out failed");
    println!("PASS kernel={KERNEL} M={M} K={K} N={N} finite=true min={min} max={max} mean={mean:.6}");
}
