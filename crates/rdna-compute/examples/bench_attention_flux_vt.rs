// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! A/B the new V-transposed WMMA flash kernel against the incumbent v5 (and
//! v6, the only variant within noise of it on gfx1151) at the real FLUX.1-dev
//! MMDiT attention shape: `n_q = n_kv = 4608`, 24 heads, head_dim 128, Q/out
//! f32, K/V f16, non-causal.
//!
//! Protocol: every variant gets its own warm-up launches at the exact shape
//! before any timing (kernels JIT per shape, and a cold first run is 3-7x off),
//! then `REPS` timed launches with a device sync around each; the median is
//! reported. Run it inside `flock /tmp/hipfire-gpu.lock` and from a fresh
//! process.
//!
//! FLOPs = 4 * n^2 * heads * hd (QK^T plus PV, both n x n x hd per head).
//! `s/step` assumes 57 attention calls per FLUX denoise step (19 double + 38
//! single blocks).
//!
//! ```
//! flock /tmp/hipfire-gpu.lock \
//!   cargo run --release --features lab --example bench_attention_flux_vt -p rdna-compute
//! ```
//! Env: `REPS` (default 5), `WARM` (default 2), `N` (default 4608),
//! `HEADS` (default 24), `PEAK_TF` (measured f16 WMMA peak, for a `%peak`
//! column; omit to hide it), `SKIP_VTK` (drop every gfx11-wave32-only kernel,
//! for a gfx12 part), `SKIP_SUPERSEDED` (drop v5, v6, `vt` and `vtk`, leaving
//! the `v2` / routed A/B — the sweep's own length moves the iGPU clock, so a
//! short run keeps the absolute figures comparable).
//!
//! `AB=<a>,<b>` + `WARM_SECS` (default 5) is the **interleaved** mode, and the
//! only sound protocol here for a delta of a few percent: it soaks the clock,
//! then runs `REPS` rounds of one timed launch of each variant back to back
//! with the within-round order alternating between rounds, printing every raw
//! run and both orders' medians. The block-per-variant sweep gives whichever
//! kernel runs first a several-percent head start on a small part, which is
//! larger than most deltas worth measuring. Accepted names are `vt`, `vtk`,
//! `v2`, `v5`, `v6` and `routed`; an unknown name is an error rather than a
//! silent skip, and naming a gfx11-wave32-only kernel (`vt`, `vtk`, `v2`) on a
//! gfx12 part fails at launch — use the plain sweep with `SKIP_VTK` there.
//! Example: `AB=v2,v5 REPS=9`.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const HD: usize = 128;

/// The variant names `AB=a,b` accepts, and the launcher each one names.
///
/// This is a `match` on a string rather than a function-pointer table because
/// the launchers are inherent methods on `&mut Gpu`: taking `&mut self` makes
/// them awkward to store, and the list is short enough that an unknown name
/// failing loudly here is worth more than the indirection.
#[allow(clippy::too_many_arguments)]
fn launch_named(
    gpu: &mut Gpu,
    name: &str,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    n: usize,
    heads: usize,
) -> Result<(), String> {
    let r = match name {
        "vt" => gpu.attention_flux_vt_wmma_f16kv_f32(q, k, v, out, n, n, heads, heads, HD),
        "vtk" => gpu.attention_flux_vtk_wmma_f16kv_f32(q, k, v, out, n, n, heads, heads, HD),
        "v2" => gpu.attention_flux_v2_wmma_f16kv(q, k, v, out, n, n, heads, heads, HD),
        "v5" => {
            gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(q, k, v, out, n, n, heads, heads, HD)
        }
        "v6" => {
            gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(q, k, v, out, n, n, heads, heads, HD)
        }
        "routed" => gpu.attention_flux_best_f16kv_f32(q, k, v, out, n, n, heads, heads, HD),
        other => {
            return Err(format!(
                "AB: '{other}' is not a variant. Accepted: vt, vtk, v2, v5, v6, routed."
            ))
        }
    };
    r.map_err(|e| format!("AB: launching '{name}' failed: {e}"))
}

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn f16_buf(gpu: &mut Gpu, rows: usize, cols: usize, seed: u64) -> GpuTensor {
    let n = rows * cols;
    let mut host = vec![0u8; n * 2];
    // Spread of small non-degenerate f16 magnitudes: no zero/denormal fast
    // path can flatter one variant over another.
    for (i, c) in host.chunks_exact_mut(2).enumerate() {
        let mut x = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 32;
        // exponent 0x2c..0x33 -> ~0.06..2.0, random sign + mantissa
        let exp = 0x2c + (x & 0x7) as u16;
        let bits = ((x >> 20) as u16 & 0x8000) | (exp << 10) | ((x >> 8) as u16 & 0x3ff);
        c.copy_from_slice(&bits.to_le_bytes());
    }
    let buf = gpu.hip.malloc(host.len()).expect("malloc f16");
    gpu.hip.memcpy_htod(&buf, &host).expect("htod f16");
    let t = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buf.as_ptr(), host.len()) },
        shape: vec![rows, cols],
        dtype: DType::F16,
    };
    std::mem::forget(buf);
    t
}

/// The f32 payload `f32_buf` uploads, in [-1, 1).
fn f32_val(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x >> 40) as f32 / 8_388_608.0) - 1.0
}

/// The same payload as `f32_buf(seed)` rounded to f16, so the f16-Q variants
/// are timed on the values the f32-Q variants see rather than a different
/// distribution (softmax cost is data-dependent through `__expf`).
fn f16_of_f32_buf(gpu: &mut Gpu, rows: usize, cols: usize, seed: u64) -> GpuTensor {
    let n = rows * cols;
    let mut host = Vec::with_capacity(n * 2);
    for i in 0..n {
        host.extend_from_slice(&f32_to_f16_bits(f32_val(seed, i)).to_le_bytes());
    }
    let buf = gpu.hip.malloc(host.len()).expect("malloc f16");
    gpu.hip.memcpy_htod(&buf, &host).expect("htod f16");
    let t = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buf.as_ptr(), host.len()) },
        shape: vec![rows, cols],
        dtype: DType::F16,
    };
    std::mem::forget(buf);
    t
}

/// f32 -> f16, round-to-nearest-even, **including the subnormal range** — the
/// same routine as `test_attention_flux_vt_parity`'s, kept byte-identical so
/// the bench feeds the f16-Q entries exactly the values the parity example
/// proves them correct on. (Inlined rather than shared: examples are separate
/// crates, and this is the whole of the dependency.)
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let biased = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if biased == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let unbiased = biased - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let mut exp = unbiased + 15;
        let mut m = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                m = 0;
                exp += 1;
                if exp >= 0x1f {
                    return sign | 0x7c00;
                }
            }
        }
        return sign | ((exp as u16) << 10) | (m as u16);
    }
    let drop = 13 + (-unbiased - 14) as u32;
    if drop > 24 {
        return sign;
    }
    let full = mant | 0x0080_0000;
    let m = full >> drop;
    let rem = full & ((1u32 << drop) - 1);
    let half = 1u32 << (drop - 1);
    let mut r = m;
    if rem > half || (rem == half && (m & 1) == 1) {
        r += 1;
    }
    sign | (r as u16)
}

fn f32_buf(gpu: &mut Gpu, rows: usize, cols: usize, seed: u64) -> GpuTensor {
    let n = rows * cols;
    let mut host = Vec::with_capacity(n * 4);
    for i in 0..n {
        host.extend_from_slice(&f32_val(seed, i).to_le_bytes());
    }
    let buf = gpu.hip.malloc(host.len()).expect("malloc f32");
    gpu.hip.memcpy_htod(&buf, &host).expect("htod f32");
    let t = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buf.as_ptr(), host.len()) },
        shape: vec![rows, cols],
        dtype: DType::F32,
    };
    std::mem::forget(buf);
    t
}

fn main() {
    let reps = env_usize("REPS", 5);
    let warm = env_usize("WARM", 2);
    let n = env_usize("N", 4608);
    let heads = env_usize("HEADS", 24);
    let peak_tf: Option<f64> = std::env::var("PEAK_TF").ok().and_then(|s| s.parse().ok());

    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);
    println!(
        "shape: n_q = n_kv = {n}, heads = {heads}, head_dim = {HD}, non-causal, q/out f32, k/v f16"
    );
    println!(
        "protocol: {warm} warm launches per variant at this exact shape, then median of {reps}"
    );

    let d_q = f32_buf(&mut gpu, n, heads * HD, 0xfeed);
    let d_k = f16_buf(&mut gpu, n, heads * HD, 0xbeef);
    let d_v = f16_buf(&mut gpu, n, heads * HD, 0xcafe);
    let d_out = f32_buf(&mut gpu, n, heads * HD, 0);
    // Same Q payload, f16; and an f16 destination. The kernels instantiate all
    // four {q dtype} x {out dtype} combinations, so all four get timed: the
    // f16 store must not cost anything (it replaces a 4 B store with a 2 B
    // store plus a v_cvt_f16_f32 the epilogue can hide).
    let d_q16 = f16_of_f32_buf(&mut gpu, n, heads * HD, 0xfeed);
    let d_out16 = f16_buf(&mut gpu, n, heads * HD, 0);

    let flops = 4.0 * (n as f64) * (n as f64) * (heads as f64) * (HD as f64);
    let calls_per_step = 57.0;

    macro_rules! variant {
        ($label:expr, $m:ident) => {
            variant!($label, $m, d_q, d_out)
        };
        ($label:expr, $m:ident, $q:ident, $o:ident) => {{
            for _ in 0..warm {
                gpu.$m(&$q, &d_k, &d_v, &$o, n, n, heads, heads, HD)
                    .expect("launch");
                gpu.hip.device_synchronize().expect("sync");
            }
            let mut ts = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0 = Instant::now();
                gpu.$m(&$q, &d_k, &d_v, &$o, n, n, heads, heads, HD)
                    .expect("launch");
                gpu.hip.device_synchronize().expect("sync");
                ts.push(t0.elapsed().as_secs_f64());
            }
            let lo = ts.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = ts.iter().cloned().fold(0.0f64, f64::max);
            let t = med(&mut ts);
            let tf = flops / t / 1e12;
            let pk = match peak_tf {
                Some(p) => format!("{:>7.1}%", 100.0 * tf / p),
                None => "      -".to_string(),
            };
            println!(
                "{:<40} {:>9.2} {:>9.3} {:>8} {:>9.3} {:>7.1}",
                $label,
                t * 1e3,
                tf,
                pk,
                t * calls_per_step,
                100.0 * (hi - lo) / t
            );
            t
        }};
    }

    // ── AB=a,b: interleaved A/B, the only sound protocol here for a small
    //           delta between two variants ──
    //
    // The sweep below times each variant in a contiguous block. On a 16 CU
    // iGPU that is not sound for a small delta: the clock falls monotonically
    // as the run heats up (measured: the *same* v2 f32/f32 cell reads 34.3 ms
    // as the third block of a short sweep, 36.2 ms as the first block of a
    // shorter one and 41.1 ms as the fifth block of a long one — a 20% span
    // that has nothing to do with the kernel). Whichever variant is timed
    // first wins by construction.
    //
    // This mode removes the ordering bias instead of arguing about it:
    //
    //   1. heat the part to a steady clock for `WARM_SECS`, alternating both
    //      variants so neither is warmed preferentially;
    //   2. run `REPS` rounds of one timed launch of each, back to back, so any
    //      monotone drift lands on both equally;
    //   3. **swap the within-round order on odd rounds** (a-then-b, b-then-a,
    //      a-then-b, ...). Back-to-back is not symmetric on its own — the
    //      second launch of a round inherits whatever state the first left in
    //      the caches and the clock — so alternating the order cancels that
    //      residue too, and the two orders' medians are reported separately.
    //      If they disagree by more than the a-vs-b delta, the delta is an
    //      ordering artefact and the run should be discarded, which is exactly
    //      the failure this protocol exists to make visible.
    //
    // The whole thing reports every raw run, so the medians can be re-derived
    // by hand from the output.
    if let Ok(spec) = std::env::var("AB") {
        let names: Vec<&str> = spec.split(',').map(str::trim).collect();
        let [a, b] = names.as_slice() else {
            panic!(
                "AB expects exactly two comma-separated variant names, e.g. AB=v2,v5; got {spec:?}"
            );
        };
        let (a, b) = (*a, *b);
        let warm_secs: f64 = std::env::var("WARM_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        println!();
        println!(
            "AB={a},{b} interleaved: {warm_secs:.0}s clock soak, then {reps} rounds with \
             alternating within-round order"
        );
        let t_soak = Instant::now();
        while t_soak.elapsed().as_secs_f64() < warm_secs {
            for name in [a, b] {
                launch_named(&mut gpu, name, &d_q, &d_k, &d_v, &d_out, n, heads)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            gpu.hip.device_synchronize().expect("sync");
        }
        // `ta`/`tb` collect per-variant times; `first_*` splits them by which
        // variant led the round, so the two orders can be reported separately.
        let mut ta = Vec::with_capacity(reps);
        let mut tb = Vec::with_capacity(reps);
        let mut a_led_a = Vec::new();
        let mut a_led_b = Vec::new();
        let mut b_led_a = Vec::new();
        let mut b_led_b = Vec::new();
        for round in 0..reps {
            let a_first = round % 2 == 0;
            let order = if a_first { [a, b] } else { [b, a] };
            let mut round_times = [0.0f64; 2];
            for (slot, name) in order.iter().enumerate() {
                let t0 = Instant::now();
                launch_named(&mut gpu, name, &d_q, &d_k, &d_v, &d_out, n, heads)
                    .unwrap_or_else(|e| panic!("{e}"));
                gpu.hip.device_synchronize().expect("sync");
                round_times[slot] = t0.elapsed().as_secs_f64();
            }
            let (t_a, t_b) = if a_first {
                (round_times[0], round_times[1])
            } else {
                (round_times[1], round_times[0])
            };
            ta.push(t_a);
            tb.push(t_b);
            if a_first {
                a_led_a.push(t_a);
                a_led_b.push(t_b);
            } else {
                b_led_a.push(t_a);
                b_led_b.push(t_b);
            }
        }
        let raw = |v: &[f64]| -> String {
            v.iter()
                .map(|t| format!("{:.2}", t * 1e3))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("  {a} raw ms: {}", raw(&ta));
        println!("  {b} raw ms: {}", raw(&tb));
        // Per-order medians first: they are the check on the headline number,
        // so print them before it rather than as a footnote after it.
        let mut order_line = Vec::new();
        if !a_led_a.is_empty() {
            order_line.push(format!(
                "{a}-first rounds: {a} {:.2} ms, {b} {:.2} ms",
                med(&mut a_led_a) * 1e3,
                med(&mut a_led_b) * 1e3
            ));
        }
        if !b_led_a.is_empty() {
            order_line.push(format!(
                "{b}-first rounds: {a} {:.2} ms, {b} {:.2} ms",
                med(&mut b_led_a) * 1e3,
                med(&mut b_led_b) * 1e3
            ));
        }
        println!("  medians by order — {}", order_line.join(" | "));
        let ma = med(&mut ta);
        let mb = med(&mut tb);
        println!(
            "  {a} median {:.2} ms ({:.3} TFLOP/s)   {b} median {:.2} ms ({:.3} TFLOP/s)",
            ma * 1e3,
            flops / ma / 1e12,
            mb * 1e3,
            flops / mb / 1e12
        );
        println!(
            "  {b} vs {a}: {:.4}x  ({:+.2}% per call)   s/step {:.3} -> {:.3}",
            ma / mb,
            100.0 * (ma - mb) / ma,
            ma * calls_per_step,
            mb * calls_per_step
        );
        return;
    }

    println!();
    println!(
        "{:<40} {:>9} {:>9} {:>8} {:>9} {:>7}",
        "variant", "ms", "TFLOP/s", "%peak", "s/step", "spread%"
    );
    println!("{}", "-".repeat(88));

    // `SKIP_SUPERSEDED` drops the four kernels no live route can pick on a
    // gfx11 part — v5, v6, vt and vtk — leaving the v2 / routed pair.
    // It exists because the run length is itself a confounder on a 16 CU iGPU:
    // adding four more variants to the sweep pushed `v2` from 34.3 ms to
    // 41.1 ms in the same session purely through clock drift, so a comparison
    // taken at the end of a long sweep is measured in a different clock regime
    // than the number it is compared against. This keeps the absolute figures
    // comparable to a short run; for an actual A/B use `AB=<a>,<b>`, which
    // removes the ordering bias rather than just shortening it.
    let include_superseded = std::env::var("SKIP_SUPERSEDED").is_err();
    let t_v5 = if include_superseded {
        variant!(
            "v5 m64_n32 (incumbent)",
            attention_dflash_wmma_m64_n32_f16kv_v5_f32
        )
    } else {
        f64::NAN
    };
    let t_v6 = if include_superseded {
        variant!("v6 m64_n32", attention_dflash_wmma_m64_n32_f16kv_v6_f32)
    } else {
        f64::NAN
    };
    let (t_new, t_vt_of16, t_vt_qf16, t_vt_both) = if include_superseded {
        (
            variant!(
                "NEW flux_vt (Vt LDS + reg softmax)",
                attention_flux_vt_wmma_f16kv_f32
            ),
            variant!(
                "  flux_vt  q f32 / out f16",
                attention_flux_vt_wmma_f16kv_f32,
                d_q,
                d_out16
            ),
            variant!(
                "  flux_vt  q f16 / out f32",
                attention_flux_vt_wmma_f16kv_f32,
                d_q16,
                d_out
            ),
            variant!(
                "  flux_vt  q f16 / out f16",
                attention_flux_vt_wmma_f16kv_f32,
                d_q16,
                d_out16
            ),
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };
    let t_vtk = if std::env::var("SKIP_VTK").is_ok() || !include_superseded {
        f64::NAN
    } else {
        variant!(
            "NEW flux_vtk (+ K staged in LDS)",
            attention_flux_vtk_wmma_f16kv_f32
        )
    };
    let (t_vtk_of16, t_vtk_qf16, t_vtk_both) =
        if std::env::var("SKIP_VTK").is_ok() || !include_superseded {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (
                variant!(
                    "  flux_vtk q f32 / out f16",
                    attention_flux_vtk_wmma_f16kv_f32,
                    d_q,
                    d_out16
                ),
                variant!(
                    "  flux_vtk q f16 / out f32",
                    attention_flux_vtk_wmma_f16kv_f32,
                    d_q16,
                    d_out
                ),
                variant!(
                    "  flux_vtk q f16 / out f16",
                    attention_flux_vtk_wmma_f16kv_f32,
                    d_q16,
                    d_out16
                ),
            )
        };

    // The barrier-/gather-reworked third generation. Timed in all four dtype
    // cells like the other two, and interleaved with them in source order so an
    // A/B is not confounded by clock drift across the run. `SKIP_VTK` gates it
    // with `vtk`: both are gfx11-wave32-only and would refuse to launch on a
    // gfx12 part, where only `vt` has a sibling.
    let (t_v2, t_v2_of16, t_v2_qf16, t_v2_both) = if std::env::var("SKIP_VTK").is_ok() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        (
            variant!(
                "NEW flux_v2 (M128, 2 barriers/tile)",
                attention_flux_v2_wmma_f16kv
            ),
            variant!(
                "  flux_v2  q f32 / out f16",
                attention_flux_v2_wmma_f16kv,
                d_q,
                d_out16
            ),
            variant!(
                "  flux_v2  q f16 / out f32",
                attention_flux_v2_wmma_f16kv,
                d_q16,
                d_out
            ),
            variant!(
                "  flux_v2  q f16 / out f16",
                attention_flux_v2_wmma_f16kv,
                d_q16,
                d_out16
            ),
        )
    };

    // Smoke-tests the arch router as well as timing it; must land on whichever
    // of vt / vtk is fastest on this arch.
    let t_best = variant!(
        "routed (attention_flux_best)",
        attention_flux_best_f16kv_f32
    );
    let t_best_both = variant!(
        "routed  q f16 / out f16",
        attention_flux_best_f16kv_f32,
        d_q16,
        d_out16
    );

    println!();
    println!("dtype surface (vs the f32/f32 entry of the same kernel; >1.00x = f16 is slower):");
    if t_new.is_finite() {
        println!(
            "  vt   of16 {:.3}x   qf16 {:.3}x   both {:.3}x",
            t_vt_of16 / t_new,
            t_vt_qf16 / t_new,
            t_vt_both / t_new
        );
    }
    if t_vtk.is_finite() {
        println!(
            "  vtk  of16 {:.3}x   qf16 {:.3}x   both {:.3}x",
            t_vtk_of16 / t_vtk,
            t_vtk_qf16 / t_vtk,
            t_vtk_both / t_vtk
        );
    }
    if t_v2.is_finite() {
        println!(
            "  v2   of16 {:.3}x   qf16 {:.3}x   both {:.3}x",
            t_v2_of16 / t_v2,
            t_v2_qf16 / t_v2,
            t_v2_both / t_v2
        );
    }
    println!("  routed both-f16 {:.3}x", t_best_both / t_best);

    println!();
    if t_new.is_finite() {
        println!(
            "new vs v5: {:.2}x    new vs v6: {:.2}x",
            t_v5 / t_new,
            t_v6 / t_new
        );
        println!("routed vs v5: {:.2}x", t_v5 / t_best);
    }
    // `v2 vs routed` needs neither v5 nor vt, so it is printed on its own —
    // under `SKIP_SUPERSEDED` those two are NaN and folding this into the line
    // below would suppress the one ratio a short run is there to produce.
    if t_v2.is_finite() {
        println!("v2 vs routed: {:.2}x", t_best / t_v2);
    }
    if t_v2.is_finite() && t_new.is_finite() {
        println!(
            "v2 vs v5: {:.2}x    v2 vs vt: {:.2}x",
            t_v5 / t_v2,
            t_new / t_v2
        );
    }
    if t_vtk.is_finite() {
        println!(
            "vtk vs v5: {:.2}x    vtk vs vt: {:.2}x",
            t_v5 / t_vtk,
            t_new / t_vtk
        );
    }
    println!(
        "FLOPs = 4*n^2*heads*hd = {flops:.3e}; s/step assumes {calls_per_step} attention calls per denoise step."
    );
    if peak_tf.is_none() {
        println!("(set PEAK_TF=<measured f16 WMMA TFLOP/s> for a %peak column)");
    }
}
