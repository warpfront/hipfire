// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact gfx1151: the Qwen4 decode MoE head fusion
//! `gemv_bf16_xf32_x2_q8_0_k2560_staged_pair` (module `qwen4_decode_head_fuse`)
//! must write the same bytes as the two incumbent launches it replaces,
//! `gemv_bf16_xf32_x4` (router and selector rows, with the gate/up parts at
//! M = 0 as the pipeline passes them for a Q8_0 shared expert) and
//! `gemv_q8_0_pair` (shared gate/up at K = 2560).
//!
//! All four outputs (router logits, selector scalar, shared gate, shared up)
//! are compared bit for bit, at the production shape (512 + 1 BF16 rows,
//! 640 + 640 Q8_0 rows) and at ragged shapes that move the block routing
//! boundaries. Every output starts as a NaN sentinel, so a row the fused
//! launch skips or writes twice with different bytes fails.
//!
//! `#[ignore]`d: needs an exact-gfx1151 GPU with a working HIP toolchain. Run
//! explicitly:
//!
//!   cargo test -p rdna-compute --release --test qwen4_decode_head_fuse -- --ignored

use rdna_compute::{Gpu, GpuTensor};

const K: usize = 2560;
const Q8_ROW_BYTES: usize = K / 32 * 34;

/// (router rows, selector rows, shared gate/up rows): production, then shapes
/// that put the BF16/Q8_0 and gate/up boundaries off every alignment.
const SHAPES: [(usize, usize, usize); 5] = [
    (512, 1, 640),
    (37, 3, 5),
    (1, 1, 1),
    (64, 1, 33),
    (3, 2, 64),
];

const SENTINEL: u32 = 0x7fc0_dead;

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

/// Deterministic values in [-scale, scale).
fn fill(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| (next(seed) as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 * scale - scale)
        .collect()
}

/// `rows` x K BF16 weights as raw little-endian bytes.
fn bf16_rows(seed: &mut u64, rows: usize) -> Vec<u8> {
    fill(seed, rows * K, 0.25)
        .into_iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// `rows` Q8_0 rows of K elements: an f16 scale in [2^-5, 2^-3) and 32 int8
/// quants per 34-byte block.
fn q8_rows(seed: &mut u64, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * Q8_ROW_BYTES);
    for _ in 0..rows * (K / 32) {
        let scale = 0x2800u16 + (next(seed) % 0x0800) as u16;
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend((0..32).map(|_| next(seed) as u8));
    }
    out
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let data = vec![f32::from_bits(SENTINEL); n];
    gpu.upload_f32(&data, &[n]).expect("upload sentinel")
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn check_shape(gpu: &mut Gpu, (m_router, m_selector, m_shared): (usize, usize, usize), round: u64) {
    let mut seed =
        0x4ead_0000 + round * 7919 + (m_router * 131 + m_selector * 17 + m_shared) as u64;
    let x_data = fill(&mut seed, K, 2.0);
    let x = gpu.upload_f32(&x_data, &[K]).expect("upload x");
    let w_router = gpu
        .upload_raw(&bf16_rows(&mut seed, m_router), &[m_router * K * 2])
        .unwrap();
    let w_selector = gpu
        .upload_raw(&bf16_rows(&mut seed, m_selector), &[m_selector * K * 2])
        .unwrap();
    let w_gate = gpu
        .upload_raw(&q8_rows(&mut seed, m_shared), &[m_shared * Q8_ROW_BYTES])
        .unwrap();
    let w_up = gpu
        .upload_raw(&q8_rows(&mut seed, m_shared), &[m_shared * Q8_ROW_BYTES])
        .unwrap();

    let outputs = |gpu: &mut Gpu| {
        [
            sentinel(gpu, m_router),
            sentinel(gpu, m_selector),
            sentinel(gpu, m_shared),
            sentinel(gpu, m_shared),
        ]
    };

    // Incumbent: the x4 launch with the Q8_0 gate/up parts at M = 0, then the
    // Q8_0 pair.
    let [router_ref, selector_ref, gate_ref, up_ref] = outputs(gpu);
    gpu.gemv_bf16_xf32_x4(
        [
            (&w_router, &router_ref, m_router),
            (&w_selector, &selector_ref, m_selector),
            (&w_gate, &gate_ref, 0),
            (&w_up, &up_ref, 0),
        ],
        &x,
        K,
    )
    .expect("gemv_bf16_xf32_x4");
    gpu.gemv_q8_0_pair(&w_gate, &w_up, &x, &gate_ref, &up_ref, m_shared, K)
        .expect("gemv_q8_0_pair");

    // Fused: one launch, run twice to catch a launch-to-launch difference.
    for pass in 0..2 {
        let [router, selector, gate, up] = outputs(gpu);
        gpu.gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
            [
                (&w_router, &router, m_router),
                (&w_selector, &selector, m_selector),
            ],
            [(&w_gate, &gate), (&w_up, &up)],
            m_shared,
            &x,
            K,
        )
        .expect("fused head launch");
        for (name, fused, incumbent) in [
            ("router", &router, &router_ref),
            ("selector", &selector, &selector_ref),
            ("shared gate", &gate, &gate_ref),
            ("shared up", &up, &up_ref),
        ] {
            let (got, want) = (bits(gpu, fused), bits(gpu, incumbent));
            assert!(
                want.iter().all(|&b| b != SENTINEL),
                "incumbent left {name} rows unwritten ({m_router}, {m_selector}, {m_shared})"
            );
            let bad: Vec<usize> = (0..got.len())
                .filter(|&i| got[i] != want[i])
                .take(8)
                .collect();
            assert!(
                bad.is_empty(),
                "{name} differs from the incumbent at rows {bad:?} (pass {pass}, shapes \
                 {m_router}/{m_selector}/{m_shared})"
            );
        }
    }
}

#[test]
#[ignore = "needs an exact-gfx1151 GPU"]
fn qwen4_decode_head_fuse_bitwise_incumbent() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1151" {
        eprintln!(
            "skip: the decode head fusion is exact gfx1151 only, got {}",
            gpu.arch
        );
        return;
    }
    // The launcher refuses every other K.
    {
        let t = sentinel(&mut gpu, 4);
        assert!(gpu
            .gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
                [(&t, &t, 1), (&t, &t, 1)],
                [(&t, &t), (&t, &t)],
                1,
                &t,
                K - 32,
            )
            .is_err());
    }
    for round in 0..3 {
        for shape in SHAPES {
            check_shape(&mut gpu, shape, round);
        }
    }
}
