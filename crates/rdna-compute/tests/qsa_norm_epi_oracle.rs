// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: the fused QSA q / k RMSNorm + RoPE epilogue of the int8 PM GEMM
//! (`Gpu::gemm_mq6g256v2_i8_prequant_norm_rope`, entry
//! `gemm_mq6g256v2_mmq_i8_pm_set_nr_gfx1201_t256x128`) must write exactly the
//! bytes of `gemm_mq6g256v2_i8_prequant` (F32) followed by
//! `indexed_attention_norm_rope_batch`.
//!
//! Two layouts, as the QSA prefill runs them: the interleaved q-gate rows
//! (24 heads of `[q(256) gate(256)]`, M = 12288, norm+RoPE on the q half, the
//! gate half must stay the plain GEMM output) and the plain k rows (2 heads of
//! 256, M = 512). Every byte of `Y` is compared for N in {1, 127, 128, 129,
//! 4000, 4096}, several `position_start` values (including past 16384) and
//! plain / scaled yarn. The fused destination is poisoned first, so a byte the
//! fused entry left unwritten cannot pass. The rope table itself is compared
//! against a host twin of `qsa_norm_rope_head`'s angles for plain RoPE.
//!
//! `#[ignore]`d: needs an exact gfx1201 with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test qsa_norm_epi_oracle \
//!       -- --ignored --test-threads=1 --nocapture

use half::f16;
use rdna_compute::rope::RopeYarn;
use rdna_compute::tensor_ops::{
    indexed_attention_norm_rope_batch, indexed_attention_rope_table, IndexedAttentionNormRopeBatch,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 2048;
const GROUP_BYTES: usize = 200;
const HEAD_DIM: usize = 256;
const ROTARY: usize = 64;
const ENTRY: &str = "gemm_mq6g256v2_mmq_i8_pm_set_nr_gfx1201_t256x128";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn gen_weights(m: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut w = vec![0u8; m * (K / 256) * GROUP_BYTES];
    for g in w.as_chunks_mut::<GROUP_BYTES>().0 {
        for h in 0..2 {
            let s = f16::from_f32(1e-4 + r.unit().abs() * 1e-3);
            let z = f16::from_f32(-32.0 * s.to_f32() + r.unit() * s.to_f32());
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_bits().to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

fn require_gfx1201() -> Option<Gpu> {
    // The int8 route is on by default only in the Qwen4 scope; force it on.
    std::env::set_var("HIPFIRE_MQ6_I8", "1");
    let gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: needs gfx1201, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip.memset(&t.buf, 0xA5, t.numel() * 4).expect("poison");
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

struct Layout {
    name: &'static str,
    m: usize,
    heads: usize,
    head_stride: usize,
}

const LAYOUTS: [Layout; 2] = [
    Layout {
        name: "q-gate",
        m: 24 * 512,
        heads: 24,
        head_stride: 512,
    },
    Layout {
        name: "k",
        m: 2 * 256,
        heads: 2,
        head_stride: 256,
    },
];

fn yarns() -> [(&'static str, RopeYarn); 2] {
    [
        ("plain", RopeYarn::OFF),
        (
            "yarn",
            RopeYarn {
                freq_scale: 0.25,
                corr_low: 2.0,
                corr_high: 22.0,
                mscale: 1.0693,
            },
        ),
    ]
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn fused_epilogue_bytes_equal_the_unfused_chain() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let mut r = Rng(0x51ed_cafe);
    let norm_bits: Vec<u8> = (0..HEAD_DIM)
        .flat_map(|_| bf16_bits(r.unit() * 0.75).to_le_bytes())
        .collect();
    let mut norm = gpu.zeros(&[HEAD_DIM], DType::BF16).expect("norm");
    gpu.hip.memcpy_htod(&norm.buf, &norm_bits).expect("norm up");
    norm.dtype = DType::BF16;
    let mut failures = Vec::new();
    for layout in &LAYOUTS {
        let m = layout.m;
        let w = gpu
            .upload_raw(
                &gen_weights(m, 0x77 + m as u64),
                &[m * (K / 256) * GROUP_BYTES],
            )
            .expect("weights");
        for (case, n) in [1usize, 127, 128, 129, 4000, 4096].into_iter().enumerate() {
            let x: Vec<f32> = (0..n * K)
                .map(|_| {
                    let v = r.unit() * 3.0;
                    // A few outliers keep the A8 scales honest.
                    if r.next() % 97 == 0 {
                        v * 40.0
                    } else {
                        v
                    }
                })
                .collect();
            let x = gpu.upload_f32(&x, &[n * K]).expect("x");
            let y_ref = gpu.zeros(&[n * m], DType::F32).expect("y_ref");
            let y_fused = gpu.zeros(&[n * m], DType::F32).expect("y_fused");
            let table = gpu.zeros(&[n * ROTARY], DType::F32).expect("table");
            let xq = gpu.mq6_i8_prepare_x(&x, K, n).expect("producer");
            gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y_ref, m, K, n)
                .expect("plain gemm");
            let plain = gpu.download_raw_bytes(&y_ref).expect("plain bytes");
            for (yarn_name, yarn) in yarns() {
                for position_start in [0usize, 1, 4093, 16384, 16385, 100_000 + 7 * case] {
                    // Reference: the plain GEMM bytes, then the unfused norm+RoPE.
                    gpu.hip.memcpy_htod(&y_ref.buf, &plain).expect("restore");
                    indexed_attention_norm_rope_batch(
                        &mut gpu,
                        &IndexedAttentionNormRopeBatch {
                            values: &y_ref,
                            norm: &norm,
                            rows: n,
                            row_stride: m,
                            heads: layout.heads,
                            head_dim: HEAD_DIM,
                            head_stride: layout.head_stride,
                            position_start,
                            rotary_dim: ROTARY,
                            yarn,
                        },
                    )
                    .expect("unfused norm+rope");
                    let want = gpu.download_raw_bytes(&y_ref).expect("want");
                    // Fused.
                    poison(&gpu, &y_fused);
                    indexed_attention_rope_table(&mut gpu, &table, n, position_start, ROTARY, yarn)
                        .expect("rope table");
                    let ran = gpu
                        .gemm_mq6g256v2_i8_prequant_norm_rope(
                            &w,
                            xq,
                            &y_fused,
                            m,
                            K,
                            n,
                            layout.head_stride,
                            &norm,
                            &table,
                        )
                        .expect("fused launch");
                    gpu.hip.device_synchronize().expect("sync");
                    let label = format!("{} n={n} pos={position_start} {yarn_name}", layout.name);
                    assert!(ran, "{label}: fused entry declined");
                    assert_eq!(gpu.last_launched_kernel(), Some(ENTRY), "{label}");
                    let got = gpu.download_raw_bytes(&y_fused).expect("got");
                    if let Some(i) = first_diff(&want, &got) {
                        let tok = i / 4 / m;
                        let col = (i / 4) % m;
                        failures.push(format!(
                            "{label}: first byte diff at {i} (token {tok}, column {col}): want {:#010x} got {:#010x}",
                            u32::from_le_bytes(want[i / 4 * 4..i / 4 * 4 + 4].try_into().unwrap()),
                            u32::from_le_bytes(got[i / 4 * 4..i / 4 * 4 + 4].try_into().unwrap()),
                        ));
                    }
                    // The untouched half (q-gate: the gate columns) is the plain GEMM.
                    if layout.head_stride == 512 {
                        for t in 0..n {
                            for h in 0..layout.heads {
                                let lo = (t * m + h * 512 + 256) * 4;
                                if plain[lo..lo + 1024] != got[lo..lo + 1024] {
                                    failures.push(format!(
                                        "{label}: gate half differs from the plain GEMM (token {t}, head {h})"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            for t in [x, y_ref, y_fused, table] {
                gpu.free_tensor(t).expect("free");
            }
        }
        gpu.free_tensor(w).expect("free w");
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The table kernel against the host twin of the angle expression: plain RoPE
/// only (`sinf` / `cosf` on the device are not the host libm's), so values
/// are compared to a tolerance, rows must be position-shifted copies of the
/// unfused kernel's own rotation (covered above bytewise).
#[test]
#[ignore = "needs an exact gfx1201"]
fn rope_table_rows_follow_the_positions() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let rows = 300;
    let table = gpu.zeros(&[rows * ROTARY], DType::F32).expect("table");
    for position_start in [0usize, 4097, 9001] {
        indexed_attention_rope_table(
            &mut gpu,
            &table,
            rows,
            position_start,
            ROTARY,
            RopeYarn::OFF,
        )
        .expect("table");
        let got = gpu.download_f32(&table).expect("download");
        for row in 0..rows {
            for base in 0..ROTARY / 2 {
                let angle = (position_start + row) as f32
                    * 1.0e7f32.powf(-((2 * base) as f32) / ROTARY as f32);
                let (sine, cosine) = (angle as f64).sin_cos();
                assert!(
                    (got[row * ROTARY + base] as f64 - cosine).abs() < 5e-3,
                    "cos row {row} base {base} pos {position_start}"
                );
                assert!(
                    (got[row * ROTARY + ROTARY / 2 + base] as f64 - sine).abs() < 5e-3,
                    "sin row {row} base {base} pos {position_start}"
                );
            }
        }
    }
    gpu.free_tensor(table).expect("free");
}

/// Timing of the fused epilogue against the chain it replaces (card B, `--class timing`):
/// per layout and token count, `gemm (default tile) + norm_rope`, the fused entry, and the plain
/// GEMM alone (the epilogue's whole cost is `fused - gemm`). K = 2560 (Flash-Next hidden).
///
///   cargo test -p rdna-compute --release --test qsa_norm_epi_oracle \
///       fused_epilogue_timing -- --ignored --nocapture
#[test]
#[ignore = "needs an exact gfx1201 on an idle timing card"]
fn fused_epilogue_timing() {
    const TK: usize = 2560;
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let mut r = Rng(0xbeef_1234);
    let norm_bits: Vec<u8> = (0..HEAD_DIM)
        .flat_map(|_| bf16_bits(r.unit() * 0.75).to_le_bytes())
        .collect();
    let mut norm = gpu.zeros(&[HEAD_DIM], DType::BF16).expect("norm");
    gpu.hip.memcpy_htod(&norm.buf, &norm_bits).expect("norm up");
    norm.dtype = DType::BF16;
    let iters = 20usize;
    for layout in &LAYOUTS {
        let m = layout.m;
        // Weights of TK columns: regenerate with the right K.
        let mut wb = vec![0u8; m * (TK / 256) * GROUP_BYTES];
        {
            let mut rr = Rng(0x99 + m as u64);
            for g in wb.as_chunks_mut::<GROUP_BYTES>().0 {
                for h in 0..2 {
                    let s = f16::from_f32(1e-4 + rr.unit().abs() * 1e-3);
                    let z = f16::from_f32(-32.0 * s.to_f32());
                    g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
                    g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_bits().to_le_bytes());
                }
                for c in g[8..].chunks_mut(8) {
                    let v = rr.next().to_le_bytes();
                    c.copy_from_slice(&v[..c.len()]);
                }
            }
        }
        let w = gpu
            .upload_raw(&wb, &[m * (TK / 256) * GROUP_BYTES])
            .expect("weights");
        for n in [4096usize, 16384] {
            let x: Vec<f32> = (0..n * TK).map(|_| r.unit() * 3.0).collect();
            let x = gpu.upload_f32(&x, &[n * TK]).expect("x");
            let y = gpu.zeros(&[n * m], DType::F32).expect("y");
            let table = gpu.zeros(&[n * ROTARY], DType::F32).expect("table");
            let xq = gpu.mq6_i8_prepare_x(&x, TK, n).expect("producer");
            indexed_attention_rope_table(&mut gpu, &table, n, 0, ROTARY, RopeYarn::OFF).expect("t");
            let nr = IndexedAttentionNormRopeBatch {
                values: &y,
                norm: &norm,
                rows: n,
                row_stride: m,
                heads: layout.heads,
                head_dim: HEAD_DIM,
                head_stride: layout.head_stride,
                position_start: 0,
                rotary_dim: ROTARY,
                yarn: RopeYarn::OFF,
            };
            let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu)| -> f64 {
                for _ in 0..3 {
                    f(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            let gemm = time(&mut gpu, &mut |g| {
                g.gemm_mq6g256v2_i8_prequant(&w, xq, &y, m, TK, n).unwrap();
            });
            let nrope = time(&mut gpu, &mut |g| {
                indexed_attention_norm_rope_batch(g, &nr).unwrap();
            });
            let chain = time(&mut gpu, &mut |g| {
                g.gemm_mq6g256v2_i8_prequant(&w, xq, &y, m, TK, n).unwrap();
                indexed_attention_norm_rope_batch(g, &nr).unwrap();
            });
            let fused = time(&mut gpu, &mut |g| {
                assert!(g
                    .gemm_mq6g256v2_i8_prequant_norm_rope(
                        &w,
                        xq,
                        &y,
                        m,
                        TK,
                        n,
                        layout.head_stride,
                        &norm,
                        &table
                    )
                    .unwrap());
            });
            println!(
                "TIMING {} M={m} N={n}: gemm {gemm:.0} us, norm_rope {nrope:.0} us, gemm+norm_rope {chain:.0} us, fused {fused:.0} us (saves {:.0} us, {:.1}%)",
                layout.name,
                chain - fused,
                100.0 * (chain - fused) / chain
            );
            for t in [x, y, table] {
                gpu.free_tensor(t).expect("free");
            }
        }
        gpu.free_tensor(w).expect("free w");
    }
}
