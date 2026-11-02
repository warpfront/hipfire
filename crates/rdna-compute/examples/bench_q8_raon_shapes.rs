//! Q8_0 prefill GEMM routes at the Raon-1-mini Q8-class shapes (gfx1100 / gfx1151).
//! Arms: (1) production `gemm_q8_0_batched_chunked` (F16 route), (2) landed i8 64x64
//! `gemm_q8_0_mmq_4w_gfx1151`, (3) each `Q8MmqTile` (GEMM only on pre-quantized activations, quantizer
//! timed separately), (4) `gemm_q8_0_mmq_gfx11_batched` (selector + quantize).
//! Arms 3/4 must be byte-identical to arm 2 (k % 128 == 0); any mismatch exits with code 2.
//! Weights rotate over a ring of copies (>= 256 MiB total, max 16) so they are not cache-resident.
//! usage: bench_q8_raon_shapes [N ...]   (default 512 4096)
use rdna_compute::{q8_mmq_pick_tile, DType, Gpu, GpuTensor, Q8MmqTile};
use std::time::Instant;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*s >> 33) as u32) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn quantize_q8(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / 32 * 34);
    for blk in w.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        out.extend_from_slice(&f32_to_f16_bits(sc).to_le_bytes());
        for &v in blk {
            out.push(((v / sc).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

fn wrap(ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
        shape,
        dtype,
    }
}

const WARMUP: usize = 20;
const TRIALS: usize = 100;

fn time_us<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151") {
        println!("skip: arch {} is not gfx1100/gfx1151", gpu.arch);
        return;
    }
    // (name, M, K, calls per prefill chunk)
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("rda_qkvg/mla_g", 5120, 2560, 79),
        ("rda_o/mla_o", 2560, 5120, 25),
        ("mla_kv_b", 10240, 512, 7),
        ("mla_q_a/kv_a", 576, 2560, 14),
        ("shared_gu", 1088, 2560, 48),
        ("shared_dn", 2560, 1280, 24),
        ("dense_gu", 12032, 2560, 2),
        ("dense_dn", 2560, 12032, 1),
        ("mla_q_b", 7680, 576, 7),
    ];
    let ns: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ns = if ns.is_empty() { vec![512, 4096] } else { ns };
    let mut mismatches = 0usize;
    for &n in &ns {
        println!("\n##### N = {n} rows");
        let (mut tot_prod, mut tot_landed, mut tot_best, mut tot_sel) = (0f64, 0f64, 0f64, 0f64);
        for &(name, m, k, calls) in shapes {
            let mut seed = 0x9E3779B97F4A7C15u64 ^ (m as u64 * 31 + k as u64);
            let w: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 0.05).collect();
            let x: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let wq = quantize_q8(&w);
            drop(w);
            let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ring = (((256usize << 20) / wq.len()).max(1)).min(16);
            let mut w_bufs = Vec::with_capacity(ring);
            for _ in 0..ring {
                let b = gpu.hip.malloc(wq.len()).unwrap();
                gpu.hip.memcpy_htod(&b, &wq).unwrap();
                w_bufs.push(b);
            }
            let ws: Vec<GpuTensor> = w_bufs
                .iter()
                .map(|b| wrap(b.as_ptr(), wq.len(), vec![m, k], DType::Q8_0))
                .collect();
            let x_g = gpu.hip.malloc(xb.len()).unwrap();
            gpu.hip.memcpy_htod(&x_g, &xb).unwrap();
            let ybytes = n * m * 4;
            let y_ref_g = gpu.hip.malloc(ybytes).unwrap();
            let y_new_g = gpu.hip.malloc(ybytes).unwrap();
            let xt = wrap(x_g.as_ptr(), xb.len(), vec![n, k], DType::F32);
            let y_ref = wrap(y_ref_g.as_ptr(), ybytes, vec![n, m], DType::F32);
            let y_new = wrap(y_new_g.as_ptr(), ybytes, vec![n, m], DType::F32);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;

            // (1) production F16 route
            let mut i = 0usize;
            let prod_us = time_us(&mut gpu, |g| {
                g.gemm_q8_0_batched_chunked(&ws[i % ring], &xt, &y_new, m, k, n)
                    .unwrap();
                i += 1;
            });
            let mut line = format!(
                "{name:14} M={m:5} K={k:5} x{calls:3} prod {prod_us:8.1}us {:5.1}TOPS",
                flops / prod_us / 1e6
            );
            let (mut landed_us, mut best_us, mut sel_us) = (prod_us, prod_us, prod_us);
            if k % 128 == 0 {
                // (2) landed i8 64x64 (reference Y)
                gpu.gemm_q8_0_mmq_4w_gfx1151(&ws[0], &xt, &y_ref, m, k, n)
                    .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let mut yref = vec![0u8; ybytes];
                gpu.hip.memcpy_dtoh(&mut yref, &y_ref_g).unwrap();
                let mut i = 0usize;
                landed_us = time_us(&mut gpu, |g| {
                    g.gemm_q8_0_mmq_4w_gfx1151(&ws[i % ring], &xt, &y_ref, m, k, n)
                        .unwrap();
                    i += 1;
                });
                line += &format!(" | landed {landed_us:8.1}us ({:.2}x)", prod_us / landed_us);

                let fill = vec![0xffu8; ybytes];
                let mut yb = vec![0u8; ybytes];
                let mut check = |gpu: &mut Gpu, label: &str, line: &mut String| {
                    gpu.hip.device_synchronize().unwrap();
                    gpu.hip.memcpy_dtoh(&mut yb, &y_new_g).unwrap();
                    if yb != yref {
                        mismatches += 1;
                        *line += &format!(" MISMATCH[{label}]");
                    }
                };

                // (3) each tile, GEMM only on pre-quantized activations
                let acts = gpu.q8_mmq_quantize_x(&xt, n, k).unwrap();
                let mut best_gemm = f64::INFINITY;
                for tile in Q8MmqTile::ALL {
                    gpu.hip.memcpy_htod(&y_new_g, &fill).unwrap();
                    gpu.gemm_q8_0_mmq_gfx11_gemm_tile(&ws[0], &acts, &y_new, m, tile)
                        .unwrap();
                    check(&mut gpu, tile.label(), &mut line);
                    let mut i = 0usize;
                    let us = time_us(&mut gpu, |g| {
                        g.gemm_q8_0_mmq_gfx11_gemm_tile(&ws[i % ring], &acts, &y_new, m, tile)
                            .unwrap();
                        i += 1;
                    });
                    line += &format!(" | {} {us:8.1}us ({:.2}x)", tile.label(), landed_us / us);
                    best_gemm = best_gemm.min(us);
                }
                let quant_us = time_us(&mut gpu, |g| {
                    g.q8_mmq_quantize_x(&xt, n, k).unwrap();
                });
                best_us = best_gemm + quant_us;
                line += &format!(" | quant {quant_us:6.1}us");

                // (4) selector + quantize
                gpu.hip.memcpy_htod(&y_new_g, &fill).unwrap();
                gpu.gemm_q8_0_mmq_gfx11_batched(&ws[0], &xt, &y_new, m, k, n)
                    .unwrap();
                check(&mut gpu, "selector", &mut line);
                let mut i = 0usize;
                sel_us = time_us(&mut gpu, |g| {
                    g.gemm_q8_0_mmq_gfx11_batched(&ws[i % ring], &xt, &y_new, m, k, n)
                        .unwrap();
                    i += 1;
                });
                line += &format!(
                    " | picked {} {sel_us:8.1}us ({:.2}x)",
                    q8_mmq_pick_tile(m, n).label(),
                    landed_us / sel_us
                );
            } else {
                line += " | (K % 128 != 0: prod only)";
            }
            println!("{line}");
            tot_prod += prod_us * calls as f64;
            tot_landed += landed_us * calls as f64;
            tot_best += best_us * calls as f64;
            tot_sel += sel_us * calls as f64;
            for t in ws {
                std::mem::forget(t); // buffers are owned by w_bufs
            }
            std::mem::forget(xt);
            std::mem::forget(y_ref);
            std::mem::forget(y_new);
        }
        println!(
            "TOTAL per {n}-row chunk (calls-weighted ms): prod {:.2} | landed {:.2} | best-tile(+quant) {:.2} | selector {:.2}",
            tot_prod / 1e3,
            tot_landed / 1e3,
            tot_best / 1e3,
            tot_sel / 1e3
        );
    }
    println!("\nMISMATCHES: {mismatches}");
    if mismatches > 0 {
        std::process::exit(2);
    }
}
