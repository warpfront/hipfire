// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-GPU contract for the all-row packed LM head past the projection
//! output descriptor bound.
//!
//! An all-row head over `rows` chunk positions of the 248320-wide Qwen4
//! vocabulary writes `rows * 248320 * 4` bytes of logits.  The gfx1151 PM
//! projection kernels address `Y` through a raw-buffer descriptor of
//! `i32::MAX` bytes, so one call over a >= 2 GiB output drops its late stores
//! and wraps its byte offsets past 4 GiB.  `execute_lm_head` therefore
//! projects whole row tiles; this test runs the actual head over outputs of
//! 4.6 GB and 8.1 GB and checks sampled logits rows of every tile (first,
//! middle and last, including rows whose byte offset is past 4 GiB) against
//! the same projection run alone on that tile's rows into a small output below
//! the limit.
//!
//! The test runs only on gfx1151 and asserts that the head's tiles take the
//! MQ4G256V2 PM route (`Gpu::gemm_mq4g256v2_pm_applies`); it fails, rather than
//! passing vacuously, when `HIPFIRE_QWEN4_MQ4_X4_PM=0` or a graph capture
//! keeps the launch off that route.
//!
//! `#[ignore]`d: needs a gfx1151 GPU with about 12 GB of free VRAM.
//!
//!   cargo test -p hipfire-arch-qwen4 --release --test lm_head_long_output_hw -- --ignored --nocapture

use hipfire_arch_qwen4::program::{execute_lm_head, validate_lm_head};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::project_weight;
use rdna_compute::{DType, Gpu};

const VOCAB: usize = 248_320;
const K: usize = 512;
/// The tile the head plans for this vocabulary (`i32::MAX / (VOCAB * 4)`
/// rounded down to whole 256-row tiles).
const TILE: usize = 2048;
const GROUP_ELEMS: usize = 256;
const GROUP_BYTES: usize = 136;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Normal f16 bit pattern with biased exponent in `lo..=hi`.
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

/// `VOCAB` rows of MQ4G256V2 groups of 136 B:
/// `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][32 x u32 nibbles]` (one scale/zero
/// pair per 128-element half), every header a finite normal f16.
fn synthetic_mq4_weight() -> Vec<u8> {
    let groups = VOCAB * (K / GROUP_ELEMS);
    let mut out = vec![0u8; groups * GROUP_BYTES];
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for group in 0..groups {
        let off = group * GROUP_BYTES;
        for half in 0..2 {
            let scale = rng.f16_normal(8, 13, false); // 2^-7 .. 2^-2
            let zero = rng.f16_normal(8, 14, true); // +/- 2^-7 .. 2^-1
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&scale.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zero.to_le_bytes());
        }
        for word in 0..32 {
            let packed = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 8 + 4 * word..off + 12 + 4 * word].copy_from_slice(&packed.to_le_bytes());
        }
    }
    out
}

fn run(rows: usize) {
    let mut gpu = Gpu::init().expect("GPU");
    if gpu.arch != "gfx1151" {
        eprintln!("skip: {} has no gfx1151 PM projection route", gpu.arch);
        return;
    }
    // The defect lives in the gfx1151 PM Y descriptor: every tile the head
    // projects (the full 2048-row tile and the ragged tail) must take it.
    for tile_rows in [TILE, rows % TILE] {
        if tile_rows > 0 {
            assert!(
                gpu.gemm_mq4g256v2_pm_applies(K, tile_rows),
                "{tile_rows}-row tile is off the MQ4G256V2 PM route \
                 (HIPFIRE_QWEN4_MQ4_X4_PM=0, graph capture or replay recording?)"
            );
        }
    }

    let bytes = synthetic_mq4_weight();
    let weight_buf = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("weight upload");
    drop(bytes);
    let weight = WeightRef {
        buf: &weight_buf,
        dtype: DType::MQ4G256V2,
        m: VOCAB,
        k: K,
        row_stride: K / GROUP_ELEMS * GROUP_BYTES,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };

    let mut rng = Lcg(0xD1B5_4A32_D192_ED03);
    let hidden_host: Vec<f32> = (0..rows * K)
        .map(|_| (rng.next_u32() & 0xFF_FFFF) as f32 / (1u32 << 24) as f32 - 0.5)
        .collect();
    let hidden = gpu.upload_f32(&hidden_host, &[rows * K]).expect("hidden");
    drop(hidden_host);
    let rotation = gpu.alloc_tensor(&[rows * K], DType::F32).expect("rotation");
    assert!(
        rows * VOCAB * 4 > i32::MAX as usize,
        "the case must exceed the projection output descriptor"
    );
    let logits = gpu
        .alloc_tensor(&[rows * VOCAB], DType::F32)
        .expect("logits (needs >= 2 GiB of free VRAM)");

    validate_lm_head(&weight, &hidden, &logits, rows, rows).expect("validate");
    execute_lm_head(
        &mut gpu,
        &weight,
        &hidden,
        &logits,
        rows,
        rows,
        Some(&rotation),
    )
    .expect("all-row head");
    gpu.hip.device_synchronize().expect("sync");

    // The oracle is the same projection of one tile alone, below the limit.
    let ref_logits = gpu
        .alloc_tensor(&[TILE * VOCAB], DType::F32)
        .expect("reference tile");
    let ref_rotation = gpu
        .alloc_tensor(&[TILE * K], DType::F32)
        .expect("ref rotation");
    let mut first = 0usize;
    while first < rows {
        let len = TILE.min(rows - first);
        let input = hidden.sub_offset(first * K, len * K);
        let output = ref_logits.sub_offset(0, len * VOCAB);
        let scratch = ref_rotation.sub_offset(0, len * K);
        project_weight(&mut gpu, &weight, &input, &output, len, Some(&scratch))
            .expect("reference tile");
        gpu.hip.device_synchronize().expect("sync");
        for local in [0, len / 2, len - 1] {
            let want = gpu
                .download_f32(&ref_logits.sub_offset(local * VOCAB, VOCAB))
                .expect("reference row");
            let got = gpu
                .download_f32(&logits.sub_offset((first + local) * VOCAB, VOCAB))
                .expect("head row");
            assert!(
                want.iter().any(|v| *v != 0.0) && want.iter().all(|v| v.is_finite()),
                "reference row {} is degenerate",
                first + local
            );
            let mismatches = want
                .iter()
                .zip(&got)
                .filter(|(w, g)| w.to_bits() != g.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "row {} (byte offset {}): {mismatches} of {VOCAB} logits differ from its tile projection",
                first + local,
                (first + local) * VOCAB * 4
            );
        }
        first += len;
    }
}

#[test]
#[ignore = "needs a gfx1151 GPU with ~10 GB free VRAM"]
fn all_row_head_is_exact_past_2_gib_of_logits() {
    // 4608 rows: 4.58 GB of logits, tiles 2048 + 2048 + 512 (ragged tail).
    run(4608);
}

#[test]
#[ignore = "needs a gfx1151 GPU with ~12 GB free VRAM"]
fn all_row_head_is_exact_past_4_gib_of_logits() {
    // 8192 rows: 8.14 GB of logits, four 2048-row tiles; tile rows start past
    // the 32-bit byte offset.
    run(8192);
}
