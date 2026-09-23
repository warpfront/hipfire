//! X-LOAD toggle microbench for `gemm_mixed_moe_grouped_wmma_gfx12` (tag-4 E8 arm).
//!
//! Decision gate: is the gfx12 MoE-grouped expert GEMM X-load-bound (L2 re-reads)?
//!
//! Three variants compiled from the same HIP source with -DTOGGLE_VARIANT=N:
//!   A (variant 0, BASELINE)  — X loaded from global/L2 every kt iteration
//!   B (variant 1, X_CONST)   — X replaced by register constant; dequant + WMMA unchanged
//!   C (variant 2, X_LDS)     — X staged in LDS once per group; read from LDS in kt-loop
//!
//! Decision criterion (from the plan):
//!   B/A >= 1.5x AND C recovers >= 60% of (B-A) gap → X-load-bound → build LDS kernel
//!   B/A <  1.5x                                      → NOT load-bound → pivot to ILP lever
//!
//! Shapes: A3B gate_up — K=2048, M=768, E=128, k_top=8. Also sweeps n=128,1024.
//!
//! Run on hiptrx (gfx1201):
//!   cargo run --release -p rdna-compute --example bench_toggle_moe_x_load_gfx12

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor, DType};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Kernel source: one .hip file, three variants via -DTOGGLE_VARIANT=N
// ---------------------------------------------------------------------------
const TOGGLE_KERNEL_BASE: &str =
    include_str!("../../../kernels/src/toggle_moe_x_load_gfx12.hip");

fn variant_src(v: u8) -> String {
    format!("#define TOGGLE_VARIANT {}\n{}", v, TOGGLE_KERNEL_BASE)
}

const KNAME_A: &str = "toggle_moe_x_load_baseline";
const KNAME_B: &str = "toggle_moe_x_load_x_const";
const KNAME_C: &str = "toggle_moe_x_load_x_lds";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

/// Random E8 weight buffer (tag=4 layout: 16-byte f16 row_scale header + (K/32)*17 bytes).
fn build_e8_weight(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0);
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut buf = vec![0u8; m * row_bytes];
    let mut s = seed;
    // f16 row_scale = 1.0 (bits 0x3C00)
    for row in 0..m {
        let roff = row * row_bytes;
        buf[roff] = 0x00; buf[roff + 1] = 0x3C;  // f16(1.0)
        for b in 0..n_blocks {
            let boff = roff + 16 + b * 17;
            buf[boff] = (0x30 + (lcg(&mut s) & 0xf)) as u8;  // e4m3 scale
            for c in 0..4 {
                let cw = lcg(&mut s).wrapping_mul(2654435761) ^ lcg(&mut s);
                let cwoff = boff + 1 + c * 4;
                buf[cwoff]     = (cw & 0xff) as u8;
                buf[cwoff + 1] = ((cw >>  8) & 0xff) as u8;
                buf[cwoff + 2] = ((cw >> 16) & 0xff) as u8;
                buf[cwoff + 3] = ((cw >> 24) & 0xff) as u8;
            }
        }
    }
    buf
}

/// Random X as raw f16 bytes (n × k elements).
fn build_x_f16_raw(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = vec![0u8; n * k * 2];
    for i in 0..n * k {
        let v = -1.0f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
        let h = f32_to_f16(v);
        out[i * 2]     = (h & 0xff) as u8;
        out[i * 2 + 1] = (h >> 8) as u8;
    }
    out
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp  = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff { return (sign << 15) | 0x7c00 | (if mant != 0 { 0x200 } else { 0 }); }
    if exp > 0x70 + 0x1f { return (sign << 15) | 0x7c00; }
    if exp < 0x71 { return sign << 15; }
    let he = (exp - 0x70) as u16;
    (sign << 15) | (he << 10) | (mant >> 13) as u16
}

fn upload_raw(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, data).unwrap();
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    let t = gpu.alloc_tensor(&[data.len() * 4], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    let t = gpu.alloc_tensor(&[data.len() * 8], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
fn alloc_zeros_f32(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    gpu.hip.memset(&t.buf, 0, n * 4).unwrap();
    t
}

// ---------------------------------------------------------------------------
// Launch a toggle variant via the blob path
// ---------------------------------------------------------------------------
fn launch_toggle(
    gpu: &mut Gpu,
    kname: &'static str,
    expert_ptrs: &GpuTensor,
    dtype_tags:  &GpuTensor,
    tile_ids:    &GpuTensor,
    sorted_slot: &GpuTensor,
    x_src:       &GpuTensor,   // raw f16 bytes: [n_tokens × K]
    y_out:       &GpuTensor,
    m: usize, k: usize, k_top: usize, n: usize, m_total: usize,
) {
    let row_tiles  = ((m + 15) / 16) as u32;
    let slot_tiles = ((m_total + 15) / 16) as u32;

    let ep  = expert_ptrs.buf.as_ptr();
    let dtp = dtype_tags.buf.as_ptr();
    let tp  = tile_ids.buf.as_ptr();
    let sp  = sorted_slot.buf.as_ptr();
    let xp  = x_src.buf.as_ptr();   // already f16, n × K rows
    let yp  = y_out.buf.as_ptr();
    let m_i = m as i32;
    let k_i = k as i32;
    // x_row_div = k_top: sorted_slot_index carries the flat slot index (0..m_total),
    // and x_row = flat / k_top maps each slot back to its source token row (0..n).
    let xrd: i32 = k_top as i32;
    let mt  = m_total as i32;

    let mut kb = KernargBlob::new();
    kb.push_ptr(ep);
    kb.push_ptr(dtp);
    kb.push_ptr(tp);
    kb.push_ptr(sp);
    kb.push_ptr(xp);
    kb.push_ptr(yp);
    kb.push_i32(m_i);
    kb.push_i32(k_i);
    kb.push_i32(xrd);
    kb.push_i32(mt);

    gpu.launch_kernel_blob(
        kname,
        [row_tiles, slot_tiles, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    ).expect("launch_toggle");
}

// Warm + measure one variant, median-of-3 runs of `iters` launches each.
fn bench_variant(
    gpu:         &mut Gpu,
    kname:       &'static str,
    iters:       usize,
    expert_ptrs: &GpuTensor,
    dtype_tags:  &GpuTensor,
    tile_ids:    &GpuTensor,
    sorted_slot: &GpuTensor,
    x_src:       &GpuTensor,
    y_out:       &GpuTensor,
    m: usize, k: usize, k_top: usize, n: usize, m_total: usize,
) -> f64 {
    // Warmup pass
    for _ in 0..5 {
        launch_toggle(gpu, kname, expert_ptrs, dtype_tags, tile_ids, sorted_slot,
                      x_src, y_out, m, k, k_top, n, m_total);
    }
    gpu.hip.device_synchronize().unwrap();

    // Three runs → take median
    let mut times = [0f64; 3];
    for t in times.iter_mut() {
        let t0 = Instant::now();
        for _ in 0..iters {
            launch_toggle(gpu, kname, expert_ptrs, dtype_tags, tile_ids, sorted_slot,
                          x_src, y_out, m, k, k_top, n, m_total);
        }
        gpu.hip.device_synchronize().unwrap();
        *t = t0.elapsed().as_micros() as f64 / iters as f64;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[1]  // median of 3
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !gpu.arch_caps.is_rdna4() {
        eprintln!("SKIP — arch={} is not gfx12 (RDNA4); run on hiptrx.", arch);
        std::process::exit(0);
    }

    println!("=== X-LOAD TOGGLE MICROBENCH — gfx12 MoE grouped-WMMA E8 arm ===");
    println!("arch={}", arch);
    println!("Kernel: tag-4 E8 inner loop of gemm_mixed_moe_grouped_wmma_gfx12");
    println!("Variants: A=baseline B=X_const(no X load) C=X_LDS(LDS-staged)");
    println!("Decision: LDS worth building IFF B/A >= 1.5x AND C recovers >= 60% of (B-A) gap\n");

    // JIT-compile all three variants up front (reports any compile error immediately)
    println!("Compiling variants (JIT, first time ~10-30 s each)...");
    {
        let src_a = Box::leak(variant_src(0).into_boxed_str());
        let src_b = Box::leak(variant_src(1).into_boxed_str());
        let src_c = Box::leak(variant_src(2).into_boxed_str());
        gpu.ensure_kernel_public(KNAME_A, src_a, KNAME_A).expect("variant A compile");
        println!("  variant A (baseline) compiled OK");
        gpu.ensure_kernel_public(KNAME_B, src_b, KNAME_B).expect("variant B compile");
        println!("  variant B (X_const) compiled OK");
        gpu.ensure_kernel_public(KNAME_C, src_c, KNAME_C).expect("variant C compile");
        println!("  variant C (X_LDS) compiled OK");
    }
    println!();

    // A3B gate_up shape: K=2048, M=768 intermediate, E=128, k_top=8
    let k:     usize = 2048;
    let m:     usize = 768;
    let e:     usize = 128;
    let k_top: usize = 8;

    // Build E (E=128) expert weight tensors, all tag=4 (E8)
    let mut _keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs:  Vec<u64>       = Vec::new();
    for ei in 0..e {
        let bytes = build_e8_weight(m, k, 0x1234_5678u32.wrapping_add(ei as u32 * 9973));
        let t = upload_raw(&mut gpu, &bytes);
        ptrs.push(t.buf.as_ptr() as u64);
        _keep.push(t);
    }
    let expert_ptrs_gpu = upload_u64(&mut gpu, &ptrs);
    let tags: Vec<u8>  = vec![4u8; e];
    let dtype_tags_gpu  = upload_raw(&mut gpu, &tags);

    let iters = 50usize;

    // Sweep token counts
    for &n in &[128usize, 512, 1024] {
        let m_total = n * k_top;
        println!("==== n={} (m_total={}) ====", n, m_total);

        // sorted_slot_index: identity
        let sorted: Vec<i32> = (0..m_total as i32).collect();
        let sorted_gpu = upload_i32(&mut gpu, &sorted);

        // expert_tile_ids: round-robin across E experts
        let tile_count = (m_total + 15) / 16;
        let tile_ids: Vec<i32> = (0..tile_count).map(|t| (t % e) as i32).collect();
        let tile_ids_gpu = upload_i32(&mut gpu, &tile_ids);

        // X as raw f16 bytes: n rows × K cols (gate_up shape)
        let x_bytes = build_x_f16_raw(n, k, 0xCAFE_0000u32 ^ n as u32);
        let x_gpu   = upload_raw(&mut gpu, &x_bytes);
        let y_gpu   = alloc_zeros_f32(&mut gpu, m_total * m);

        let us_a = bench_variant(&mut gpu, KNAME_A, iters,
            &expert_ptrs_gpu, &dtype_tags_gpu, &tile_ids_gpu, &sorted_gpu,
            &x_gpu, &y_gpu, m, k, k_top, n, m_total);
        let us_b = bench_variant(&mut gpu, KNAME_B, iters,
            &expert_ptrs_gpu, &dtype_tags_gpu, &tile_ids_gpu, &sorted_gpu,
            &x_gpu, &y_gpu, m, k, k_top, n, m_total);
        let us_c = bench_variant(&mut gpu, KNAME_C, iters,
            &expert_ptrs_gpu, &dtype_tags_gpu, &tile_ids_gpu, &sorted_gpu,
            &x_gpu, &y_gpu, m, k, k_top, n, m_total);

        let ratio_ba = us_a / us_b;
        let gap = us_a - us_b;
        let c_recovery = if gap < 0.01 { 0.0f64 } else { (us_a - us_c) / gap };

        // TFLOPS estimate (one gate_up projection, 2 flop/MAC)
        let flops   = 2.0 * m_total as f64 * m as f64 * k as f64;
        let tflop_a = flops / us_a / 1e6;
        let tflop_b = flops / us_b / 1e6;

        println!("  A (baseline):  {:8.1} us  ({:.1} TFLOPS)", us_a, tflop_a);
        println!("  B (X_const):   {:8.1} us  ({:.1} TFLOPS)  B/A={:.2}x", us_b, tflop_b, ratio_ba);
        println!("  C (X_LDS):     {:8.1} us   C-recovery={:.0}%  speedup C/A={:.2}x", us_c, c_recovery * 100.0, us_a / us_c);

        let x_load_bound = ratio_ba >= 1.5;
        let lds_viable   = x_load_bound && c_recovery >= 0.60;
        println!("  X-LOAD-BOUND: {}   LDS-WORTH-BUILDING: {}", x_load_bound, lds_viable);
        if lds_viable {
            println!("  -> CONFIRMED: build gemm_mixed_moe_grouped_wmma X-LDS gfx12 variant.");
            println!("     Projected speedup from LDS: ~{:.1}x (C/A ratio, gate_up n={})",
                     us_a / us_c, n);
        } else if x_load_bound {
            println!("  -> X is load-bound (B/A {:.2}x) but LDS recovery {:.0}% < 60%. Check bank conflicts.", ratio_ba, c_recovery * 100.0);
        } else {
            println!("  -> NOT X-load-bound (B/A {:.2}x < 1.5x). LDS is a dead end.", ratio_ba);
            println!("     Pivot gfx12 to ILP/inline-decode lever (same as gfx11 plan).");
        }
        println!();
    }

    println!("=== DONE. Record result in .moe-plan/MoE_progress.md (W1 gfx12 toggle). ===");
}
