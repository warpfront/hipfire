// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Long-dispatch microbench for the `gemv_hfq4g256`-family GEMV body at the
//! `moe_down` proxy shape (M=2048, K=4096; see docs/gfx1201-native-surface.md
//! "Precise per-kernel achieved-BW table" — `moe_down` measured 44.1% of
//! DRAM peak, attributed to small-transaction granularity, not ALU).
//!
//! Wraps the same per-row HFQ4G256 dequant+dot body used by
//! `gemv_hfq4g256` / `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` in
//! an INTERNAL kernel loop (ITERS) so a SINGLE dispatch runs for
//! milliseconds instead of the ~10-16us a real dispatch takes. The prior
//! gfx1100 PMC probe (docs/gfx1201-native-surface.md, "Pre-check resolution
//! (ALU-vs-mem)") found every %-busy-style counter (MemUnitBusy,
//! TA_BUSY_avr, GRBM_GUI_ACTIVE, SQ_WAVE_CYCLES, SQ_WAIT_INST_ANY,
//! SQ_INSTS_VALU) read exactly 0.0 across 640 short (~9.2us) dispatches even
//! though raw accumulators (SQ_BUSY_CYCLES, Wavefronts) latched fine in the
//! SAME runs — a ROCm 7.2.2 dispatch-scoped-counter capture floor, not a
//! methodology bug. This bench makes ONE dispatch long enough
//! (ITERS x ~16.5us) for %-busy counters to have a chance to latch.
//!
//! Row addresses rotate through NBUF distinct M x K weight blocks (mod
//! NBUF) so repeated iterations don't degenerate into pure L2/Infinity-Cache
//! replay of the SAME ~4.46 MB block — NBUF x 4.46 MB should exceed the
//! card's cache capacity to keep the access pattern representative of the
//! real (DRAM-touching) decode workload rather than an artificially
//! cache-resident one.
//!
//! Usage:
//!   HIP_VISIBLE_DEVICES=0 cargo run --release -p rdna-compute \
//!       --example bench_gemv_moe_down_pmc_loop
//!   BENCH_ITERS=4000 BENCH_NBUF=32 <as above>   # override defaults

use std::env;
use std::time::Instant;

const M: usize = 2048; // moe_down output rows
const K: usize = 4096; // moe_down reduction dim
const GROUP: usize = 256;
const GROUP_BYTES: usize = 136; // 4 (scale) + 4 (zero) + 128 (packed nibbles)

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Internal-loop variant of the gemv_hfq4g256 narrow-path body
// (kernels/src/gemv_hfq4g256.hip). Identical per-row dequant+dot math;
// wrapped ITERS times so a single dispatch runs long enough for
// dispatch-scoped PMC counters to latch. `buf_idx` rotates the row base
// through NBUF distinct weight blocks to avoid full cache residency.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_pmc_loop(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int ITERS, int NBUF
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const int boff = tid * 4;
    const long long row_stride = (long long)groups_per_row * 136;
    const long long buf_stride = (long long)M * row_stride;

    float acc_total = 0.0f;
    for (int it = 0; it < ITERS; it++) {
        const int buf_idx = it % NBUF;
        const char* row_ptr = A + (long long)buf_idx * buf_stride + (long long)row * row_stride;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        // K=4096 -> groups_per_row=16 -> quads=4, tail=0 (no tail path needed).
        const int quads = groups_per_row >> 2;

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            const char* gp0 = row_ptr + g * 136;
            const char* gp1 = gp0 + 136;
            const char* gp2 = gp1 + 136;
            const char* gp3 = gp2 + 136;

            float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
            float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
            float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
            float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
            float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
            float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
            float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
            float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

            unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

            int base = g * 256 + tid * 8;

            #define DOG(pk, sc, zp, b, a) \
                (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                     + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                     + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                     + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                     + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                     + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                     + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                     + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

            DOG(pk0, sc0, zp0, base, acc0);
            DOG(pk1, sc1, zp1, base + 256, acc1);
            DOG(pk2, sc2, zp2, base + 512, acc2);
            DOG(pk3, sc3, zp3, base + 768, acc3);
            #undef DOG
        }

        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        acc_total += acc;
    }

    if (tid == 0) y[row] = acc_total;
}
"#;

fn main() {
    let iters: i64 = env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let nbuf: i64 = env::var("BENCH_NBUF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    println!("[bench] GPU arch={} device_id={}", gpu.arch, gpu.device_id);
    println!("[bench] shape M={M} K={K} ITERS={iters} NBUF={nbuf}");

    let groups_per_row = K / GROUP;
    let row_bytes = groups_per_row * GROUP_BYTES;
    let buf_bytes = M * row_bytes;
    let total_bytes = buf_bytes * nbuf as usize;
    println!(
        "[bench] weight buffer: {} MB ({} bufs x {} MB, row_bytes={})",
        total_bytes / (1024 * 1024),
        nbuf,
        buf_bytes / (1024 * 1024),
        row_bytes
    );

    // Values don't need to be numerically meaningful — this is a
    // bandwidth/occupancy probe, not a correctness test. Cheap xorshift-ish
    // fill so the buffer isn't degenerate all-zero (which some memory
    // controllers/compressors could special-case).
    let mut a_data = vec![0u8; total_bytes];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for b in a_data.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (seed >> 33) as u8;
    }
    let x_data: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();

    let t_up = Instant::now();
    let a = gpu
        .upload_raw(&a_data, &[nbuf as usize, M, K])
        .expect("upload A");
    let x = gpu.upload_f32(&x_data, &[K]).expect("upload x");
    let y = gpu
        .zeros(&[M], rdna_compute::DType::F32)
        .expect("alloc y");
    println!("[bench] upload done in {:?}", t_up.elapsed());

    gpu.ensure_kernel_public(
        "gemv_hfq4g256_moe_down_pmc_loop",
        KERNEL_SRC,
        "gemv_hfq4g256_moe_down_pmc_loop",
    )
    .expect("compile kernel");

    let a_ptr = a.buf.as_ptr();
    let x_ptr = x.buf.as_ptr();
    let y_ptr = y.buf.as_ptr();
    let m_val = M as i32;
    let k_val = K as i32;
    let iters_val = iters as i32;
    let nbuf_val = nbuf as i32;

    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(a_ptr);
    blob.push_ptr(x_ptr);
    blob.push_ptr(y_ptr);
    blob.push_i32(m_val);
    blob.push_i32(k_val);
    blob.push_i32(iters_val);
    blob.push_i32(nbuf_val);

    // One warmup dispatch (untimed, small ITERS) to force JIT + clock
    // ramp-up before the timed/profiled dispatch.
    {
        let mut warm_blob = hip_bridge::KernargBlob::new();
        warm_blob.push_ptr(a_ptr);
        warm_blob.push_ptr(x_ptr);
        warm_blob.push_ptr(y_ptr);
        warm_blob.push_i32(m_val);
        warm_blob.push_i32(k_val);
        warm_blob.push_i32(4);
        warm_blob.push_i32(nbuf_val);
        gpu.launch_kernel_blob(
            "gemv_hfq4g256_moe_down_pmc_loop",
            [M as u32, 1, 1],
            [32, 1, 1],
            0,
            warm_blob.as_mut_slice(),
        )
        .expect("warmup launch");
        gpu.hip.device_synchronize().expect("warmup sync");
    }

    println!("[pmc-probe] BEGIN moe_down_loop m={M} k={K} iters={iters} nbuf={nbuf}");
    let t0 = Instant::now();
    gpu.launch_kernel_blob(
        "gemv_hfq4g256_moe_down_pmc_loop",
        [M as u32, 1, 1],
        [32, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .expect("launch");
    gpu.hip.device_synchronize().expect("sync");
    let elapsed = t0.elapsed();
    let us_per_iter = elapsed.as_secs_f64() * 1e6 / iters as f64;
    println!(
        "[pmc-probe] END moe_down_loop wall={:?} ({:.4} us/iter-equiv, single_dispatch)",
        elapsed, us_per_iter
    );
    println!("[pmc-probe] DONE_MARKER");

    // Touch output to prevent any (unlikely, given runtime addressing)
    // dead-code elimination and as a basic no-crash sanity check.
    let y_host = gpu.download_f32(&y).expect("download y");
    let checksum: f32 = y_host.iter().sum();
    println!("[bench] y checksum (sanity, not correctness): {checksum:.3}");
}
