//! gfx1010 FP16-packed multirow GEMV correctness test.
//!
//! Compares the new HFQ4-G256 multirow GEMV with __hfma2 / v_pk_fma_f16
//! inner loop (commit 159db6f) against the existing FP32 multirow path.
//!
//! Tolerance is calibrated for FP16-multiply / FP32-accumulate (cross-group):
//! per-summand precision ~0.1% (FP16 mantissa) + cumulative cross-group sum
//! adds ~sqrt(k/256) levels of FP32 add noise. Decode-realistic K=2048,
//! M=1024 should land well inside the thresholds.
//!
//! Run on any arch (kernel JIT-compiles for all RDNA targets — the kernel
//! is portable HIP, the gfx1010 dispatch divert is what's arch-specific):
//!   cargo run --release -p rdna-compute --example test_gemv_fp16_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let rows: u32 = std::env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    assert!(k % 256 == 0, "K must be a multiple of 256 (HFQ4-G256 group)");
    assert!(matches!(rows, 2 | 4 | 8), "ROWS must be 2, 4, or 8");
    assert!(m % rows as usize == 0, "M must be divisible by ROWS");

    // Force the FP32 baseline path for the first call (env is cached at first
    // read; setting before Gpu::init ensures the FP32 dispatch is used).
    // SAFETY: single-threaded test, no other threads observe env at this point.
    unsafe {
        std::env::set_var("HIPFIRE_GEMV_FP16", "0");
        std::env::set_var("HIPFIRE_GEMV_ROWS", rows.to_string());
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    eprintln!("=== gfx1010 FP16-packed multirow GEMV correctness ===");
    eprintln!("M={m}, K={k}, R={rows}");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    let x_host: Vec<f32> = (0..k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();

    let x_gpu = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
    let y_fp32 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_fp32");
    let y_fp16 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_fp16");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();

    // Path A: FP32 multirow reference (current decode hot path).
    gpu.hip.device_synchronize().unwrap();
    gpu.gemv_hfq4g256(&a_raw, &x_gpu, &y_fp32, m, k).expect("fp32 multirow");
    gpu.hip.device_synchronize().unwrap();
    let y_fp32_host: Vec<f32> = gpu.download_f32(&y_fp32).expect("download y_fp32");

    // Path B: FP16-packed multirow (new opt-in path). Direct method call
    // bypasses the env-gated dispatch, so we exercise the new kernel here
    // regardless of arch.
    gpu.hip.device_synchronize().unwrap();
    gpu.gemv_hfq4g256_multirow_fp16(&a_raw, &x_gpu, &y_fp16, m, k, rows)
        .expect("fp16 multirow");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    assert_eq!(y_fp32_host.len(), m);
    assert_eq!(y_fp16_host.len(), m);

    const REL_FLOOR: f32 = 0.1;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_rel_err: f64 = 0.0;
    let mut max_loc: usize = 0;
    let mut samples_above_10pct: usize = 0;
    let mut rel_eligible: usize = 0;

    for row in 0..m {
        let a = y_fp32_host[row];
        let b = y_fp16_host[row];
        let err = (a - b).abs();
        if err > max_abs_err {
            max_abs_err = err;
            max_loc = row;
        }
        sum_abs_err += err as f64;
        if a.abs() > REL_FLOOR {
            let rel = err / a.abs();
            if rel > max_rel_err { max_rel_err = rel; }
            sum_rel_err += rel as f64;
            rel_eligible += 1;
            if rel > 0.10 { samples_above_10pct += 1; }
        }
    }

    let total = m as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 { sum_rel_err / (rel_eligible as f64) } else { 0.0 };
    let pct_above = 100.0 * samples_above_10pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-row error (M={m} rows) ---");
    eprintln!("  max abs err:           {:.6}  at row {}", max_abs_err, max_loc);
    eprintln!("  mean abs err:          {:.6}", mean_abs_err);
    eprintln!("  rel-err eligible:      {} / {} ({:.1}%)",
              rel_eligible, m, 100.0 * rel_eligible as f32 / m as f32);
    eprintln!("  max rel err:           {:.4}", max_rel_err);
    eprintln!("  mean rel err:          {:.4}", mean_rel_err);
    eprintln!("  samples > 10% rel:     {} / {} ({:.3}%)",
              samples_above_10pct, rel_eligible.max(1), pct_above);

    eprintln!("\n--- sample rows (0..6) ---");
    for row in 0..6.min(m) {
        let a = y_fp32_host[row];
        let b = y_fp16_host[row];
        eprintln!("  row {row}: fp32={a:>10.4}  fp16={b:>10.4}  err={:.4}", (a - b).abs());
    }

    // Thresholds: FP16 mul / FP32 add cross-group accumulation.
    // Per-summand FP16 noise ~0.1%; cumulative across K/256 group sums small.
    let max_abs_thresh = 0.5;
    let mean_abs_thresh = 0.05;
    let max_rel_thresh = 0.5;
    let mean_rel_thresh = 0.05;
    let pct_thresh = 10.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (FP16 mul / FP32 cross-group acc) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:  {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err   < {max_rel_thresh}:   {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err  < {mean_rel_thresh}:  {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel  < {pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: FP16-packed multirow GEMV is numerically equivalent to FP32 \
                   reference within FP16 tolerance.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: FP16-packed multirow GEMV diverges beyond FP16 tolerance.");
        std::process::exit(1);
    }
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            // scale: small positive ~1e-3 to 1e-2 range
            let scale_bits = 0x3a000000u32 | (next() & 0x007F_FFFF);
            // zero point: small magnitude, either sign
            let zp_bits = ((next() & 0x80) << 24) | 0x39000000u32 | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
