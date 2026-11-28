// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The device qt44 embedding decode (`Gpu::inkling2_embed_qt44_rows`, the
//! blocked prefill's embed) must equal the host decoder (`EmbedPacked::row_f32`,
//! the scalar decode's embed) **bit for bit**.
//!
//! Synthetic rows (random codes, random finite F16 scales and zeros including
//! subnormals and negative values), then, with `INKLING2_MQ4` set, every row of
//! the artifact's packed `embed.weight`. Needs a GPU (`--ignored`).
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq cargo test --release \
//!   -p hipfire-arch-inkling2 --test embed_decode -- --ignored --nocapture
//! ```

use hipfire_arch_inkling2::gpu::EmbedPacked;
use rdna_compute::{DType, Gpu};

const COLS: usize = 2304;
const ROW_BYTES: usize = COLS / 256 * 136;
/// Rows per device call (the 2048-row block plus slack).
const CHUNK: usize = 4096;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 17
}

/// A random finite F16 bit pattern (exponent below all-ones).
fn finite_f16(s: &mut u64) -> u16 {
    loop {
        let h = lcg(s) as u16;
        if (h >> 10) & 0x1f != 0x1f {
            return h;
        }
    }
}

/// Bitwise mismatches of the device decode against `row_f32` over `ids`.
fn mismatches(gpu: &mut Gpu, packed: &EmbedPacked, ids: &[u32]) -> usize {
    let mut bytes = Vec::new();
    packed.gather_rows(ids, &mut bytes).expect("gather");
    let dev = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload");
    let out = gpu
        .alloc_tensor(&[ids.len() * COLS], DType::F32)
        .expect("alloc");
    gpu.inkling2_embed_qt44_rows(&dev, &out, ids.len(), COLS)
        .expect("decode");
    let got = gpu.download_f32(&out).expect("download");
    gpu.free_tensor(dev).expect("free");
    gpu.free_tensor(out).expect("free");
    let mut bad = 0usize;
    for (r, &id) in ids.iter().enumerate() {
        let want = packed.row_f32(id).expect("host row");
        let row = &got[r * COLS..(r + 1) * COLS];
        for (c, (w, g)) in want.iter().zip(row).enumerate() {
            if w.to_bits() != g.to_bits() {
                if bad < 8 {
                    eprintln!(
                        "token {id} col {c}: host {w:e} ({:08x}) device {g:e} ({:08x})",
                        w.to_bits(),
                        g.to_bits()
                    );
                }
                bad += 1;
            }
        }
    }
    bad
}

#[test]
#[ignore = "needs a GPU (and INKLING2_MQ4 for the artifact sweep)"]
fn device_embed_decode_is_bitwise_the_host_decoder() {
    let mut gpu = Gpu::init().expect("gpu init");

    // Synthetic table.
    let rows = 1000usize;
    let mut s = 0x5eed_0f_e4b3du64;
    let mut bytes = vec![0u8; rows * ROW_BYTES];
    for g in bytes.chunks_exact_mut(136) {
        for h in 0..4 {
            g[2 * h..2 * h + 2].copy_from_slice(&finite_f16(&mut s).to_le_bytes());
        }
        for b in g[8..].iter_mut() {
            *b = lcg(&mut s) as u8;
        }
    }
    let synth = EmbedPacked {
        quant_type: 44,
        rows,
        cols: COLS,
        bytes,
    };
    // Every row once, out of order, plus repeats in one call.
    let mut ids: Vec<u32> = (0..rows as u32).rev().collect();
    ids.extend([0, 0, 999, 500]);
    let bad = mismatches(&mut gpu, &synth, &ids);
    eprintln!("synthetic: {} rows, {bad} mismatching elements", ids.len());
    assert_eq!(bad, 0, "synthetic rows differ from the host decoder");

    // The artifact's embed table, every row.
    let Ok(art) = std::env::var("INKLING2_MQ4") else {
        eprintln!("INKLING2_MQ4 unset: artifact sweep skipped");
        return;
    };
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&art)).expect("open");
    let (info, data) = hfq
        .tensor_data_vec("model.llm.embed.weight")
        .expect("embed tensor");
    assert_eq!(info.quant_type, 44);
    let vocab = data.len() / ROW_BYTES;
    let table = EmbedPacked {
        quant_type: 44,
        rows: vocab,
        cols: COLS,
        bytes: data,
    };
    let mut total = 0usize;
    let all: Vec<u32> = (0..vocab as u32).collect();
    for chunk in all.chunks(CHUNK) {
        total += mismatches(&mut gpu, &table, chunk);
    }
    eprintln!("artifact: {vocab} rows, {total} mismatching elements");
    assert_eq!(total, 0, "artifact rows differ from the host decoder");
}
