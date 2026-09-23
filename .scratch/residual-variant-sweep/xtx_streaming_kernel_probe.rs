// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Temporary raw-HIP benchmark: resident-vs-streaming weight working sets over
//! the ACTUAL production kernels, under identical resident-vs-streaming sets.
//!
//! Compares one operator at a time (`--kind gateup|out|down|lmhead`) on the host GPU:
//! resident mode launches only weight copy 0; streaming mode cycles all weight
//! copies continuously. X is one shared buffer, Y is one shared buffer pair —
//! only the weight working set differs, so the delta isolates weight residency.
//!
//! Kernel routing (production defaults at N=16, no env overrides; omit `--variant`):
//! - gfx1100 residual (out/down): `gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds`,
//!   ABI (A,X,Y,M,K,N), grid ceil(M/16)xceil(N/16), block 128 (32*kw, kw=4).
//! - gfx1201 residual (out/down): `gemm_mq4g256v2_residual_wmma_gfx12`,
//!   same ABI/grid, block 32.
//! - gfx1100 gateup: `gemm_gate_up_mq4g256v2_wmma`, 9-arg ABI
//!   (Ag,Au,X,Yg,Yu,gm,um,K,N), grid ceil((gm+um)/16)xceil(N/16), block 32.
//! - gfx1201 gateup: `gemm_gate_up_mq4g256v2_wmma_gfx12`, same ABI/grid/block.
//!
//! Optional `--variant base|ks2|ks4|ks8|ldsstage` (gfx1100 residual out/down/lmhead only):
//! - base: `gemm_mq4g256v2_residual_wmma`, block 32
//! - ks2/ks4/ks8: `…_gfx1100_ks{KW}_lds`, block 32*KW; requires (K/256)%KW==0
//! - ldsstage: `…_gfx1100_ldsstage`, block 256; requires K%512==0
//! Illegal before launch: gateup+variant, non-gfx1100+variant, down+ks8
//! (G=68), lmhead+ks8 (G=20), unknown variant name.
//!
//! Shapes (N=16): gateup gm=um=17408 K=5120; out M=5120 K=6144;
//! down M=5120 K=17408; lmhead M=248320 K=5120 (same residual Y+= path;
//! one weight matrix already >512 MiB so copy_count=1, resident≡streaming).
//!
//! Usage (parent owns compile/run):
//!   cargo build -p hipfire-runtime --example xtx_streaming_kernel_probe
//!   ./target/debug/examples/xtx_streaming_kernel_probe \
//!     --kind out --elf /path/to/kernels.hsaco --out /tmp/xtx_stream.json \
//!     [--device 0] [--output-bits /tmp/xtx_stream_copy0.f32] \
//!     [--variant base|ks2|ks4|ks8|ldsstage]
//!
//! Grounding: ABIs/grids/blocks from MatchedKernelContracts;
//! `pack_weights_deterministic` / `stage_x_f16` / `f32_to_f16_bits_rne` /
//! event-bracket / Y+= oracle patterns reused from
//! `.scratch/xtx_packed_kernel_probe.rs`; raw HIP surface from
//! `crates/hip-bridge/src/ffi.rs` (`module_load`, `module_get_function`,
//! `launch_kernel`, `event_*`, `malloc`/`memcpy_*`/`free`).

use hip_bridge::{Event, Function, HipRuntime, Module};
use std::ffi::c_void;

// ── Constants ─────────────────────────────────────────────────────────────

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;
const N: usize = 16;

// Contract shapes.
const GATEUP_M_EACH: usize = 17_408;
const GATEUP_K: usize = 5_120;
const OUT_M: usize = 5_120;
const OUT_K: usize = 6_144;
const DOWN_M: usize = 5_120;
const DOWN_K: usize = 17_408;
// Batched lm_head uses the residual tier (zero Y then Y+=). Trace:
// grid[15520,1,1]*16 = 248320 rows; G=K/256=20 → ks8 illegal, ks2/4 ok.
const LMHEAD_M: usize = 248_320;
const LMHEAD_K: usize = 5_120;

// Streaming aggregate floor: rotating weight set must cover >= 512 MiB.
const STREAM_FLOOR_BYTES: usize = 512 * 1024 * 1024;
// Hard device-memory ceiling for this probe (weights + X + Y + events).
const VRAM_CEIL_BYTES: usize = 1024 * 1024 * 1024 + 64 * 1024 * 1024;

// Timing protocol.
const WARMUP_LAUNCHES: usize = 20;
const PRE_CYCLES: usize = 3; // full weight-set cycles before timing, outside events
const BRACKET_LAUNCHES: usize = 128; // enqueue-ONLY launches per sample
const SAMPLES_PER_MODE: usize = 40; // ABBA resident/stream interleave

// ── Deterministic host helpers (reused from .scratch/xtx_packed_kernel_probe.rs) ──

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
/// used by the `convert_f32_to_f16` X-staging kernel. X is finite in [-1, 1].
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

/// Deterministic MQ4V2 weight blob: nibble payload bytes cycle 0..255 across
/// the buffer; headers are valid finite nonzero fp16 scales with zero-points
/// from {+0.0, -0.0, 0.3, -0.3}; the non-power-of-two entries (0.1/1/3/0.7
/// scales, 0.3/-0.3 zero-points) force fp16 rounding for most q*sc+zp
/// products. Identical bytes on every host for the same (m, k, seed).
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
                    ((((r * gpr + g) as u64 * 128 + j as u64 + seed) % 256) as u8);
            }
        }
    }
    blob
}

/// FNV-1a 64 over raw bytes. Labelled everywhere it is reported.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv1a64_f32(v: &[f32]) -> u64 {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    fnv1a64(bytes)
}

fn fnv1a64_u16(v: &[u16]) -> u64 {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    fnv1a64(bytes)
}

fn bit_parity(a: &[f32], b: &[f32]) -> (usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            mism += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    (mism, first)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

// ── Raw HIP launch shims (exact production ABIs) ──────────────────────────

#[allow(clippy::too_many_arguments)]
fn enqueue_residual(
    hip: &HipRuntime,
    func: &Function,
    a: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
    block_x: usize,
) {
    // Enqueue ONLY: no synchronization. The bracket's stop-event synchronize
    // is the sole completion gate. NEVER call device_synchronize here.
    let mut ap = a;
    let mut xp = x;
    let mut yp = y;
    let mut mv = m as i32;
    let mut kv = k as i32;
    let mut nv = n as i32;
    let mut params = [
        &mut ap as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void,
        &mut mv as *mut _ as *mut c_void,
        &mut kv as *mut _ as *mut c_void,
        &mut nv as *mut _ as *mut c_void,
    ];
    let grid = [((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1];
    let block = [block_x as u32, 1, 1];
    unsafe {
        hip.launch_kernel(func, grid, block, 0, None, &mut params)
            .expect("residual hipModuleLaunchKernel failed");
    }
}

/// Synchronized correctness/warmup launch. NEVER inside a timed bracket.
#[allow(clippy::too_many_arguments)]
fn launch_residual(
    hip: &HipRuntime,
    func: &Function,
    a: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
    block_x: usize,
) {
    enqueue_residual(hip, func, a, x, y, m, k, n, block_x);
    hip.device_synchronize()
        .expect("sync after residual launch");
}

#[allow(clippy::too_many_arguments)]
fn enqueue_gateup(
    hip: &HipRuntime,
    func: &Function,
    a_gate: *mut c_void,
    a_up: *mut c_void,
    x: *mut c_void,
    y_gate: *mut c_void,
    y_up: *mut c_void,
    gm: usize,
    um: usize,
    k: usize,
    n: usize,
) {
    // Enqueue ONLY: no synchronization. The bracket's stop-event synchronize
    // is the sole completion gate. NEVER call device_synchronize here.
    let mut ag = a_gate;
    let mut au = a_up;
    let mut xp = x;
    let mut yg = y_gate;
    let mut yu = y_up;
    let mut gmv = gm as i32;
    let mut umv = um as i32;
    let mut kv = k as i32;
    let mut nv = n as i32;
    let mut params = [
        &mut ag as *mut _ as *mut c_void,
        &mut au as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut yg as *mut _ as *mut c_void,
        &mut yu as *mut _ as *mut c_void,
        &mut gmv as *mut _ as *mut c_void,
        &mut umv as *mut _ as *mut c_void,
        &mut kv as *mut _ as *mut c_void,
        &mut nv as *mut _ as *mut c_void,
    ];
    let total = gm + um;
    let grid = [((total + 15) / 16) as u32, ((n + 15) / 16) as u32, 1];
    let block = [32, 1, 1];
    unsafe {
        hip.launch_kernel(func, grid, block, 0, None, &mut params)
            .expect("gateup hipModuleLaunchKernel failed");
    }
}

/// Synchronized correctness/warmup launch. NEVER inside a timed bracket.
#[allow(clippy::too_many_arguments)]
fn launch_gateup(
    hip: &HipRuntime,
    func: &Function,
    a_gate: *mut c_void,
    a_up: *mut c_void,
    x: *mut c_void,
    y_gate: *mut c_void,
    y_up: *mut c_void,
    gm: usize,
    um: usize,
    k: usize,
    n: usize,
) {
    enqueue_gateup(hip, func, a_gate, a_up, x, y_gate, y_up, gm, um, k, n);
    hip.device_synchronize()
        .expect("sync after gateup launch");
}

// ── Device buffer helpers (all alloc/H2D outside timed events) ─────────────

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

fn upload_f32(hip: &HipRuntime, buf: &hip_bridge::DeviceBuffer, v: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    hip.memcpy_htod(buf, bytes).expect("memcpy_htod re-upload failed");
}

/// One timed bracket: 128 enqueue-ONLY launches between two HIP events.
/// `launch` MUST NOT synchronize; the stop-event synchronize below is the
/// sole completion gate. Requires positive finite time.
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

// ── CLI ────────────────────────────────────────────────────────────────────

fn arg_val(args: &[String], name: &str) -> Option<String> {
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().cloned();
        }
        if let Some(v) = a.strip_prefix(&format!("{name}=")) {
            return Some(v.to_string());
        }
    }
    None
}

fn usage() -> ! {
    eprintln!(
        "usage: xtx_streaming_kernel_probe --kind gateup|out|down|lmhead --elf PATH --out PATH \
         [--device N] [--output-bits PATH] [--variant base|ks2|ks4|ks8|ldsstage]"
    );
    std::process::exit(2);
}

// Allocation formulas (exact, host-computed before any device alloc):
//   weight_bytes_per_matrix(m, k) = m * (k / 256) * 136
//   residual logical_weight_bytes = weight_bytes_per_matrix(M, K)
//   gateup   logical_weight_bytes = 2 * weight_bytes_per_matrix(17408, 5120)
//   lmhead   logical_weight_bytes = weight_bytes_per_matrix(248320, 5120) already >512 MiB
//   copy_count = ceil(512 MiB / logical_weight_bytes)  (lmhead → 1)
//   weight_working_bytes = copy_count * logical_weight_bytes  (>= 512 MiB)
fn weight_bytes_per_matrix(m: usize, k: usize) -> usize {
    assert_eq!(k % GROUP, 0);
    m * (k / GROUP) * GROUP_BYTES
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = arg_val(&args, "--kind").unwrap_or_else(|| usage());
    let elf = arg_val(&args, "--elf").unwrap_or_else(|| usage());
    let out = arg_val(&args, "--out").unwrap_or_else(|| usage());
    let device: i32 = arg_val(&args, "--device")
        .map(|s| s.parse::<i32>().unwrap_or_else(|_| usage()))
        .unwrap_or(0);
    let output_bits = arg_val(&args, "--output-bits");
    let variant_arg = arg_val(&args, "--variant");
    if kind != "gateup" && kind != "out" && kind != "down" && kind != "lmhead" {
        eprintln!("--kind must be gateup|out|down|lmhead (got {kind})");
        std::process::exit(2);
    }
    if let Some(v) = variant_arg.as_deref() {
        if !matches!(v, "base" | "ks2" | "ks4" | "ks8" | "ldsstage") {
            eprintln!("--variant must be base|ks2|ks4|ks8|ldsstage (got {v})");
            std::process::exit(2);
        }
    }

    let is_gateup = kind == "gateup";
    // Per-kind deterministic seeds: identical bytes on every host.
    // X salt differs per kind so shapes never alias; weight seeds differ per
    // projection/operator so a swapped routing cannot bit-match.
    let (gm, um, gk, rm, rk, x_salt) = if is_gateup {
        (GATEUP_M_EACH, GATEUP_M_EACH, GATEUP_K, 0, 0, 0xC0FF_EE01u32)
    } else if kind == "out" {
        (0, 0, 0, OUT_M, OUT_K, 0xC0FF_EE02)
    } else if kind == "down" {
        (0, 0, 0, DOWN_M, DOWN_K, 0xC0FF_EE03)
    } else {
        // lmhead: residual-tier consumer (M=248320 K=5120 N=16).
        (0, 0, 0, LMHEAD_M, LMHEAD_K, 0xC0FF_EE04)
    };

    let hip = HipRuntime::load().expect("HipRuntime::load failed");
    hip.set_device(device)
        .expect("hipSetDevice failed");
    let arch_raw = hip
        .get_arch(device)
        .unwrap_or_else(|_| "unknown".to_string());
    // Normalize gcnArchName ("gfx1100", possibly with ":sramecc+" suffix).
    let arch = arch_raw.split([':', ' ']).next().unwrap_or("").to_string();
    let is_1100 = arch == "gfx1100";
    let is_1201 = arch == "gfx1201";
    // No SKIP, no zero-exit on wrong arch: the contract forbids it.
    if !is_1100 && !is_1201 {
        eprintln!("FAIL: arch {arch_raw} is neither gfx1100 nor gfx1201; refusing to run");
        std::process::exit(2);
    }

    // PCI bus id: hip-bridge exposes no PCI string API, so report unavailable
    // rather than fabricate one.
    let pci = "unavailable (hip-bridge exposes no PCI string API)".to_string();

    // Resolve symbol/block. Omitted --variant keeps production defaults for
    // every (arch, kind). Explicit --variant is gfx1100 residual out/down only.
    // gfx1100 residual default ks4: (K/256)=24 (out) and 68 (down) %4==0.
    let (symbol, block_x, variant): (&str, usize, Option<&str>) = match variant_arg.as_deref() {
        None => {
            if is_gateup {
                if is_1100 {
                    ("gemm_gate_up_mq4g256v2_wmma", 32, None)
                } else {
                    ("gemm_gate_up_mq4g256v2_wmma_gfx12", 32, None)
                }
            } else if is_1100 {
                (
                    "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds",
                    128,
                    None,
                )
            } else {
                ("gemm_mq4g256v2_residual_wmma_gfx12", 32, None)
            }
        }
        Some(v) => {
            if is_gateup {
                eprintln!(
                    "FAIL: --variant {v} is residual-only; gateup rejects residual variants"
                );
                std::process::exit(2);
            }
            if !is_1100 {
                eprintln!(
                    "FAIL: --variant {v} requires gfx1100 residual (arch={arch}, kind={kind})"
                );
                std::process::exit(2);
            }
            // Residual out/down/lmhead on gfx1100. Shape gates before module load.
            let groups = rk / GROUP; // K/256; contract K values divisible by 256.
            match v {
                "base" => ("gemm_mq4g256v2_residual_wmma", 32, Some("base")),
                "ks2" => {
                    if groups % 2 != 0 {
                        eprintln!(
                            "FAIL: --variant ks2 requires (K/256)%2==0 (kind={kind} K={rk} G={groups})"
                        );
                        std::process::exit(2);
                    }
                    (
                        "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds",
                        64,
                        Some("ks2"),
                    )
                }
                "ks4" => {
                    if groups % 4 != 0 {
                        eprintln!(
                            "FAIL: --variant ks4 requires (K/256)%4==0 (kind={kind} K={rk} G={groups})"
                        );
                        std::process::exit(2);
                    }
                    (
                        "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds",
                        128,
                        Some("ks4"),
                    )
                }
                "ks8" => {
                    if groups % 8 != 0 {
                        eprintln!(
                            "FAIL: --variant ks8 requires (K/256)%8==0 (kind={kind} K={rk} G={groups}; down G=68 illegal)"
                        );
                        std::process::exit(2);
                    }
                    (
                        "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds",
                        256,
                        Some("ks8"),
                    )
                }
                "ldsstage" => {
                    if rk % 512 != 0 {
                        eprintln!(
                            "FAIL: --variant ldsstage requires K%512==0 (kind={kind} K={rk})"
                        );
                        std::process::exit(2);
                    }
                    (
                        "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage",
                        256,
                        Some("ldsstage"),
                    )
                }
                _ => unreachable!("variant name validated above"),
            }
        }
    };
    let module: Module = hip.module_load(&elf).unwrap_or_else(|e| {
        eprintln!("FAIL: hipModuleLoad({elf}) failed: {e:?}");
        std::process::exit(1);
    });
    let func: Function = hip.module_get_function(&module, symbol).unwrap_or_else(|e| {
        eprintln!("FAIL: symbol {symbol} missing in {elf}: {e:?}");
        std::process::exit(1);
    });

    // Host inputs: same deterministic valid MQ4V2 headers + X bit pattern on all hosts.
    let (logical_weight_bytes, w_blob0, w_blob1) = if is_gateup {
        let g = pack_weights_deterministic(gm, gk, 0x6A11_E000);
        let u = pack_weights_deterministic(um, gk, 0x9A11_0000);
        let lb = g.len() + u.len();
        (lb, g, u)
    } else {
        let seed = match kind.as_str() {
            "out" => 0x5EED_17E4u64,
            "down" => 0x5EED_4402,
            _ => 0x5EED_1A3D, // lmhead
        };
        let w = pack_weights_deterministic(rm, rk, seed);
        let lb = w.len();
        (lb, w, Vec::new())
    };
    // Cross-check the closed-form allocation formula against the built blob.
    let formula_bytes = if is_gateup {
        2 * weight_bytes_per_matrix(gm, gk)
    } else {
        weight_bytes_per_matrix(rm, rk)
    };
    assert_eq!(
        logical_weight_bytes, formula_bytes,
        "blob length disagrees with allocation formula"
    );
    // ceil(512MiB / logical). lmhead single matrix already exceeds the floor
    // so copy_count==1: resident and streaming launch the same buffer.
    let copy_count = (STREAM_FLOOR_BYTES + logical_weight_bytes - 1) / logical_weight_bytes;
    assert!(copy_count >= 1, "need >= 1 weight copy");
    let weight_working_bytes = copy_count * logical_weight_bytes;
    assert!(
        weight_working_bytes >= STREAM_FLOOR_BYTES,
        "working set must cover >= 512 MiB (single-matrix shapes count as covered)"
    );
    let single_copy_no_residency_delta = copy_count == 1;

    let (x_len, y_len) = if is_gateup {
        (gk * N, gm * N)
    } else {
        (rk * N, rm * N)
    };
    let x_f32: Vec<f32> = (0..x_len)
        .map(|i| prng(i, x_salt) * 2.0 - 1.0)
        .collect();
    let x_f16 = stage_x_f16(&x_f32);
    let input_checksum = serde_json::json!({
        "algo": "FNV-1a-64",
        "weight_blob_copy0_bytes": w_blob0.len(),
        "weight_blob_copy0_fnv1a64": format!("{:016x}", fnv1a64(&w_blob0)),
        "weight_blob_copy1_fnv1a64": if is_gateup { format!("{:016x}", fnv1a64(&w_blob1)) } else { "n/a".to_string() },
        "x_f16_len": x_f16.len(),
        "x_f16_fnv1a64": format!("{:016x}", fnv1a64_u16(&x_f16)),
    });

    // Device buffers, all allocated + H2D-loaded OUTSIDE timed events.
    // Each weight copy is a SEPARATE allocation with identical bytes so
    // compression/reuse cannot change the data; VRAM caches by address.
    let mut d_w0: Vec<hip_bridge::DeviceBuffer> = Vec::with_capacity(copy_count);
    let mut d_w1: Vec<hip_bridge::DeviceBuffer> = Vec::with_capacity(copy_count);
    for _ in 0..copy_count {
        d_w0.push(htod(&hip, &w_blob0));
        if is_gateup {
            d_w1.push(htod(&hip, &w_blob1));
        }
    }
    let d_x = htod_u16(&hip, &x_f16);
    let y_zero = vec![0.0f32; y_len];
    let d_y0 = htod_f32(&hip, &y_zero);
    let d_y1 = if is_gateup {
        htod_f32(&hip, &y_zero)
    } else {
        // Placeholder never launched; keeps free/drop symmetric simple.
        htod(&hip, &[0u8; 8])
    };
    // VRAM ceiling check (weights + X + Y, small overhead).
    let y_bytes = if is_gateup { 2 * y_len * 4 } else { y_len * 4 };
    let total_working_bytes =
        weight_working_bytes + x_f16.len() * 2 + y_bytes;
    if total_working_bytes > VRAM_CEIL_BYTES {
        eprintln!(
            "FAIL: total working bytes {total_working_bytes} exceed 1 GiB + 64 MiB overhead"
        );
        std::process::exit(1);
    }

    let fail_with = |msg: &str, cases: &serde_json::Value| -> ! {
        eprintln!("FAIL: {msg}");
        let report = serde_json::json!({
            "probe": "xtx_streaming_kernel_probe",
            "kind": kind, "device": device, "arch": arch, "arch_raw": arch_raw,
            "pci": pci, "compiler_elf": elf,
            "toolchain": "not detected by probe; see parent-recorded compiler provenance",
            "variant": variant, "symbol": symbol, "block_x": block_x,
            "result": "FAIL", "error": msg, "cases": cases,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
            .expect("write --out failed");
        std::process::exit(1);
    };
    let cases = serde_json::json!({});

    // ── Correctness BEFORE timing ──────────────────────────────────────
    // Resident (copy 0) vs EVERY copy, bit-exact, using zero Y.
    // Gateup: distinct output canaries before each copy launch, then verify
    // no canary word survives (every logical output written) + bit parity.
    // Residual: zero-Y parity over all copies + nonzero-Y Y+= oracle on
    // copy 0 (controlled output must bitwise equal correctly-rounded
    // f32(zero_init output + initial Y), signed zeros exercised).
    let copy0_out: Vec<f32>;
    let mut copy0_out1: Vec<f32> = Vec::new();
    if is_gateup {
        let mut refs: Vec<Vec<f32>> = Vec::with_capacity(copy_count);
        let mut refs1: Vec<Vec<f32>> = Vec::with_capacity(copy_count);
        for c in 0..copy_count {
            // Distinct canary per copy: byte 0xC0+c (wraps fit in u8 for our counts).
            let cb = 0xC0u8.wrapping_add(c as u8);
            let can = f32::from_bits(u32::from_ne_bytes([cb; 4]));
            let canary = vec![can; y_len];
            upload_f32(&hip, &d_y0, &canary);
            upload_f32(&hip, &d_y1, &canary);
            launch_gateup(
                &hip, &func,
                d_w0[c].as_ptr(), d_w1[c].as_ptr(), d_x.as_ptr(),
                d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
            );
            let yg = dtoh_f32(&hip, &d_y0, y_len);
            let yu = dtoh_f32(&hip, &d_y1, y_len);
            let can_bits = can.to_bits();
            let left_g = yg.iter().filter(|x| x.to_bits() == can_bits).count();
            let left_u = yu.iter().filter(|x| x.to_bits() == can_bits).count();
            if left_g != 0 || left_u != 0 {
                fail_with(
                    &format!("gateup copy {c}: {left_g}/{left_u} canary words survive (unwritten outputs)"),
                    &cases,
                );
            }
            if !yg.iter().all(|x| x.is_finite()) || !yu.iter().all(|x| x.is_finite()) {
                fail_with(&format!("gateup copy {c}: non-finite output"), &cases);
            }
            refs.push(yg);
            refs1.push(yu);
        }
        for c in 1..copy_count {
            let (m0, f0) = bit_parity(&refs[c], &refs[0]);
            let (m1, f1) = bit_parity(&refs1[c], &refs1[0]);
            if m0 != 0 || m1 != 0 {
                fail_with(
                    &format!("gateup copy {c} vs copy 0: Yg mism {m0}@{f0:?} Yu mism {m1}@{f1:?}"),
                    &cases,
                );
            }
        }
        let var_g: f64 = {
            let m = refs[0].iter().map(|x| *x as f64).sum::<f64>() / y_len as f64;
            refs[0].iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / y_len as f64
        };
        if !(var_g > 1e-12) {
            fail_with("gateup copy 0 output variance ~ 0", &cases);
        }
        copy0_out = refs.into_iter().next().unwrap();
        copy0_out1 = refs1.into_iter().next().unwrap();
    } else {
        let (m, k) = (rm, rk);
        let mut refs: Vec<Vec<f32>> = Vec::with_capacity(copy_count);
        for c in 0..copy_count {
            upload_f32(&hip, &d_y0, &y_zero);
            launch_residual(&hip, &func, d_w0[c].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            let y = dtoh_f32(&hip, &d_y0, y_len);
            if !y.iter().all(|x| x.is_finite()) {
                fail_with(&format!("residual copy {c}: non-finite output"), &cases);
            }
            refs.push(y);
        }
        for c in 1..copy_count {
            let (mm, f) = bit_parity(&refs[c], &refs[0]);
            if mm != 0 {
                fail_with(
                    &format!("residual copy {c} vs copy 0: {mm} bit mismatches first @{f:?}"),
                    &cases,
                );
            }
        }
        // Nonzero-Y Y+= oracle on copy 0 (archived-probe semantics).
        let mut y_init = random_f32(y_len, 0xBEEF_1234 + N as u64 + k as u64, -0.5, 1.5);
        if y_len >= 2 {
            y_init[0] = -0.0;
            y_init[1] = 0.0;
        }
        upload_f32(&hip, &d_y0, &y_init);
        launch_residual(&hip, &func, d_w0[0].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
        let y_plus = dtoh_f32(&hip, &d_y0, y_len);
        let expected: Vec<f32> = refs[0]
            .iter()
            .zip(y_init.iter())
            .map(|(a, b)| a + b)
            .collect();
        let (pm, pf) = bit_parity(&y_plus, &expected);
        if pm != 0 {
            fail_with(
                &format!("residual Y+= oracle: {pm} mismatches first @{pf:?}"),
                &cases,
            );
        }
        let var: f64 = {
            let mu = refs[0].iter().map(|x| *x as f64).sum::<f64>() / y_len as f64;
            refs[0].iter().map(|x| (*x as f64 - mu).powi(2)).sum::<f64>() / y_len as f64
        };
        if !(var > 1e-12) {
            fail_with("residual copy 0 output variance ~ 0", &cases);
        }
        copy0_out = refs.into_iter().next().unwrap();
    }
    let output_checksum = if is_gateup {
        serde_json::json!({
            "algo": "FNV-1a-64 over f32 LE bits",
            "y_gate_fnv1a64": format!("{:016x}", fnv1a64_f32(&copy0_out)),
            "y_up_fnv1a64": format!("{:016x}", fnv1a64_f32(&copy0_out1)),
        })
    } else {
        serde_json::json!({
            "algo": "FNV-1a-64 over f32 LE bits",
            "y_fnv1a64": format!("{:016x}", fnv1a64_f32(&copy0_out)),
        })
    };
    // Optional canonical copy-0 f32 dump for parent cross-host comparison.
    if let Some(p) = output_bits.as_ref() {
        let mut bytes = Vec::with_capacity((copy0_out.len() + copy0_out1.len()) * 4);
        for v in copy0_out.iter().chain(copy0_out1.iter()) {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        if let Err(e) = std::fs::write(p, &bytes) {
            fail_with(&format!("write --output-bits failed: {e:?}"), &cases);
        }
    }

    // ── Warmup BEFORE timing (outside events) ───────────────────────────
    // 20 warmup launches on copy 0 + PRE_CYCLES full weight-set cycles.
    // Y reset between warmup launches for residual (no unbounded +=).
    if is_gateup {
        for _ in 0..WARMUP_LAUNCHES {
            launch_gateup(
                &hip, &func,
                d_w0[0].as_ptr(), d_w1[0].as_ptr(), d_x.as_ptr(),
                d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
            );
        }
        for _ in 0..PRE_CYCLES {
            for c in 0..copy_count {
                launch_gateup(
                    &hip, &func,
                    d_w0[c].as_ptr(), d_w1[c].as_ptr(), d_x.as_ptr(),
                    d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
                );
            }
        }
    } else {
        let (m, k) = (rm, rk);
        for _ in 0..WARMUP_LAUNCHES {
            upload_f32(&hip, &d_y0, &y_zero);
            launch_residual(&hip, &func, d_w0[0].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
        }
        for _ in 0..PRE_CYCLES {
            for c in 0..copy_count {
                upload_f32(&hip, &d_y0, &y_zero);
                launch_residual(&hip, &func, d_w0[c].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            }
        }
    }

    // ── Timed ABBA: 40 samples/mode, 128 enqueue-ONLY launches/sample ────
    let start = hip.event_create().expect("event_create start");
    let stop = hip.event_create().expect("event_create stop");
    let mut resident_ms: Vec<f64> = Vec::with_capacity(SAMPLES_PER_MODE);
    let mut streaming_ms: Vec<f64> = Vec::with_capacity(SAMPLES_PER_MODE);
    // Streaming cursor is continuous across samples so the aggregate set
    // (> L2/cache) actually rotates instead of restarting per sample.
    let mut cursor: usize = 0;
    // Each timed bracket is: [Y reset H2D if residual, OUTSIDE events] then
    // [start-record, 128x enqueue ONLY, stop-record, stop-sync].
    for _ in 0..SAMPLES_PER_MODE / 2 {
        // A
        if is_gateup {
            resident_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                enqueue_gateup(
                    &hip, &func,
                    d_w0[0].as_ptr(), d_w1[0].as_ptr(), d_x.as_ptr(),
                    d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
                );
            }));
            streaming_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                let c = cursor % copy_count;
                cursor = cursor.wrapping_add(1);
                enqueue_gateup(
                    &hip, &func,
                    d_w0[c].as_ptr(), d_w1[c].as_ptr(), d_x.as_ptr(),
                    d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
                );
            }));
            // B (second half of ABBA)
            streaming_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                let c = cursor % copy_count;
                cursor = cursor.wrapping_add(1);
                enqueue_gateup(
                    &hip, &func,
                    d_w0[c].as_ptr(), d_w1[c].as_ptr(), d_x.as_ptr(),
                    d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
                );
            }));
            resident_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                enqueue_gateup(
                    &hip, &func,
                    d_w0[0].as_ptr(), d_w1[0].as_ptr(), d_x.as_ptr(),
                    d_y0.as_ptr(), d_y1.as_ptr(), gm, um, gk, N,
                );
            }));
        } else {
            let (m, k) = (rm, rk);
            // Residual Y+= reset OUTSIDE the event brackets, per sample: no
            // accumulation over the unbounded benchmark.
            upload_f32(&hip, &d_y0, &y_zero);
            resident_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                enqueue_residual(&hip, &func, d_w0[0].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            }));
            upload_f32(&hip, &d_y0, &y_zero);
            streaming_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                let c = cursor % copy_count;
                cursor = cursor.wrapping_add(1);
                enqueue_residual(&hip, &func, d_w0[c].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            }));
            upload_f32(&hip, &d_y0, &y_zero);
            streaming_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                let c = cursor % copy_count;
                cursor = cursor.wrapping_add(1);
                enqueue_residual(&hip, &func, d_w0[c].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            }));
            upload_f32(&hip, &d_y0, &y_zero);
            resident_ms.push(time_bracket_ms(&hip, &start, &stop, &mut || {
                enqueue_residual(&hip, &func, d_w0[0].as_ptr(), d_x.as_ptr(), d_y0.as_ptr(), m, k, N, block_x);
            }));
        }
    }
    hip.event_destroy(start).ok();
    hip.event_destroy(stop).ok();

    for &t in resident_ms.iter().chain(streaming_ms.iter()) {
        assert!(t.is_finite() && t > 0.0, "non-positive timing sample");
    }

    // Effective (NOT physical DRAM) bandwidth: logical weight bytes consumed
    // per kernel-nanosecond. Caches may serve the resident set, and cache-line
    // amplification is unknown, so this is a residency-comparison index, not
    // a DRAM measurement.
    let gb_per_sample = |ms: &[f64]| -> Vec<f64> {
        ms.iter()
            .map(|t| {
                let ns_per_launch = *t * 1e6 / BRACKET_LAUNCHES as f64;
                logical_weight_bytes as f64 / ns_per_launch
            })
            .collect()
    };
    let r_gb = gb_per_sample(&resident_ms);
    let s_gb = gb_per_sample(&streaming_ms);

    // Post-timing proof sync, then free. Fail nonzero otherwise.
    if let Err(e) = hip.device_synchronize() {
        fail_with(&format!("post-timing device_synchronize failed: {e:?}"), &cases);
    }
    for b in d_w0 {
        if let Err(e) = hip.free(b) {
            fail_with(&format!("free weights0 failed: {e:?}"), &cases);
        }
    }
    for b in d_w1 {
        if let Err(e) = hip.free(b) {
            fail_with(&format!("free weights1 failed: {e:?}"), &cases);
        }
    }
    if let Err(e) = hip.free(d_x) {
        fail_with(&format!("free X failed: {e:?}"), &cases);
    }
    if let Err(e) = hip.free(d_y0) {
        fail_with(&format!("free Y0 failed: {e:?}"), &cases);
    }
    if let Err(e) = hip.free(d_y1) {
        fail_with(&format!("free Y1 failed: {e:?}"), &cases);
    }
    // `module`/`func` drop here; handles outlived every launch above.

    let shape = if is_gateup {
        serde_json::json!({"operator": "gateup", "gate_m": gm, "up_m": um, "k": gk, "n": N,
            "grid": [((gm + um + 15) / 16), ((N + 15) / 16), 1], "block": [32, 1, 1]})
    } else {
        serde_json::json!({"operator": kind, "m": rm, "k": rk, "n": N,
            "grid": [((rm + 15) / 16), ((N + 15) / 16), 1], "block": [block_x, 1, 1]})
    };
    let report = serde_json::json!({
        "probe": "xtx_streaming_kernel_probe",
        "kind": kind,
        "device": device,
        "arch": arch,
        "arch_raw": arch_raw,
        "pci": pci,
        "compiler_elf": elf,
        "toolchain": "not detected by probe; see parent-recorded compiler provenance",
        "variant": variant,
        "symbol": symbol,
        "block_x": block_x,
        "shape": shape,
        "allocation_formula": {
            "weight_bytes_per_matrix": "m * (k / 256) * 136",
            "logical_weight_bytes": if is_gateup {
                "2 * 17408 * (5120 / 256) * 136 = 94699520".to_string()
            } else if kind == "out" {
                "5120 * (6144 / 256) * 136 = 16711680".to_string()
            } else if kind == "down" {
                "5120 * (17408 / 256) * 136 = 47349760".to_string()
            } else {
                // lmhead: 248320 * 20 * 136 = 675430400 (> 512 MiB)
                "248320 * (5120 / 256) * 136 = 675430400".to_string()
            },
            "copy_count": format!("ceil(536870912 / {logical_weight_bytes}) = {copy_count}"),
            "weight_working_bytes": format!("{copy_count} * {logical_weight_bytes} = {weight_working_bytes}"),
        },
        "logical_weight_bytes": logical_weight_bytes,
        "copy_count": copy_count,
        "weight_working_bytes": weight_working_bytes,
        "single_copy_no_residency_delta": single_copy_no_residency_delta,
        "residency_control": if single_copy_no_residency_delta {
            "single_copy: logical weight already >=512MiB; resident and streaming launch the same buffer; no multi-copy cache-miss control for this shape"
        } else {
            "multi_copy: resident uses copy0 only; streaming rotates all copies across >=512MiB working set"
        },
        "total_working_bytes": total_working_bytes,
        "input_checksum": input_checksum,
        "correctness": {
            "copies_compared": copy_count,
            "zero_y_bit_exact_all_copies": true,
            "y_plus_oracle": if is_gateup { "n/a (Y= overwrite; distinct per-copy canaries verified unwritten=0)" } else { "pass (copy0 controlled output == f32(zero output + init), signed zeros exercised)" },
            "gateup_canary": if is_gateup { "pass (distinct 0xC0+c canary per copy, 0 survivors)" } else { "n/a (residual)" },
        },
        "timing": {
            "warmup_launches_copy0": WARMUP_LAUNCHES,
            "pre_cycles_full_set": PRE_CYCLES,
            "launches_per_sample": BRACKET_LAUNCHES,
            "samples_per_mode": SAMPLES_PER_MODE,
            "order": "ABBA",
            "mechanism": "hipEvent record/elapsed with stop-event completion gate; enqueue closures contain ONLY launch_kernel, never device_synchronize",
            "resident_ms_per_128": resident_ms,
            "streaming_ms_per_128": streaming_ms,
            "resident_mean_ms_per_128": mean(&resident_ms),
            "resident_median_ms_per_128": median(&resident_ms),
            "streaming_mean_ms_per_128": mean(&streaming_ms),
            "streaming_median_ms_per_128": median(&streaming_ms),
        },
        "effective_weight_GBps": {
            "definition": "logical_weight_bytes / kernel_ns_per_launch; NOT physical DRAM GB/s (resident set may be cache-served; cache-line amplification unknown)",
            "resident_per_sample": r_gb,
            "streaming_per_sample": s_gb,
            "resident_mean": mean(&r_gb),
            "resident_median": median(&r_gb),
            "streaming_mean": mean(&s_gb),
            "streaming_median": median(&s_gb),
        },
        "output_checksum": output_checksum,
        "output_bits": output_bits,
        "result": "PASS",
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .expect("write --out failed");
    eprintln!(
        "PASS: {kind} arch={arch} variant={variant:?} symbol={symbol} copies={copy_count} single_copy_no_residency_delta={single_copy_no_residency_delta} working={weight_working_bytes}B; report in {out}"
    );
}
