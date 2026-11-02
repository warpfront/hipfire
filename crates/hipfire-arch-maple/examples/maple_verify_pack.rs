// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! maple_verify_pack — end-to-end exactness check of a converted Maple `.hfq`
//! against the ORIGINAL safetensors shard.
//!
//! source bf16 → convert → container → dequantize  ==  source, exactly.
//!
//! The unit tests in `hipfire-quantize` verify the packer against its own
//! dequantizer, which cannot catch a shared misunderstanding of the layout, and
//! they do not exercise the container at all — a tensor could be packed
//! perfectly and then written to the `.hfq` with the wrong shape, quant_type or
//! byte range. This reads what the RUNTIME will read.
//!
//! The MQ2G256LloydU decoder below is written from the layout spec, deliberately
//! NOT reusing `hipfire-quantize`'s (which lives in that crate's bin and is not
//! importable anyway), so agreement is cross-implementation evidence rather than
//! a round trip. See `python3 -m tools.models.maple.make_parity_fixture` for the same argument
//! on the packing side.
//!
//! Usage:
//!   maple_verify_pack --model <model.hfq> --shard <model-0000N-of-00009.safetensors>
//!                     [--tensor <name>]...

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

const GROUP: usize = 256;
const BLOCK_BYTES: usize = 72;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // Subnormal: normalise.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (e + 127 - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

/// Decode MQ2G256LloydU from the layout spec: 72 B per group of 256 —
/// [0..8) four fp16 codebook entries ascending, [8..72) 2-bit indices
/// 4-per-byte LSB-first. No FWHT: these weights are in the natural basis.
fn dequant_lloyd_u(data: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = n.div_ceil(GROUP);
    assert_eq!(
        data.len(),
        n_blocks * BLOCK_BYTES,
        "byte length {} != {n_blocks} blocks * {BLOCK_BYTES}",
        data.len()
    );
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let cb: [f32; 4] = [
            f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
            f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
            f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
            f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
        ];
        for i in 0..64 {
            let byte = blk[8 + i];
            for j in 0..4 {
                let pos = b * GROUP + 4 * i + j;
                if pos >= n {
                    break;
                }
                out[pos] = cb[((byte >> (2 * j)) & 0x3) as usize];
            }
        }
    }
    out
}

/// Minimal safetensors reader: u64 header length, header JSON, then the slice.
fn read_source_bf16(shard: &Path, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(shard).expect("read shard");
    let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + hdr_len]).expect("parse safetensors header");
    let meta = header.get(name)?;
    assert_eq!(
        meta["dtype"].as_str().unwrap(),
        "BF16",
        "{name}: expected BF16 source"
    );
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let start = meta["data_offsets"][0].as_u64().unwrap() as usize;
    let end = meta["data_offsets"][1].as_u64().unwrap() as usize;
    let raw = &bytes[8 + hdr_len + start..8 + hdr_len + end];
    let vals: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    Some((shape, vals))
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut shard = None;
    let mut tensors: Vec<String> = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv[i + 1].clone());
                i += 2;
            }
            "--shard" => {
                shard = Some(argv[i + 1].clone());
                i += 2;
            }
            "--tensor" => {
                tensors.push(argv[i + 1].clone());
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let model = model.expect("--model <model.hfq>");
    let shard = shard.expect("--shard <...safetensors>");
    if tensors.is_empty() {
        tensors = vec![
            "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
            "model.layers.0.mlp.experts.0.down_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.o_proj.weight".into(),
        ];
    }

    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    assert_eq!(hfq.arch_id, 15, "expected arch_id 15, got {}", hfq.arch_id);

    let mut checked = 0usize;
    let mut failures = 0usize;
    for name in &tensors {
        let Some((shape, src)) = read_source_bf16(Path::new(&shard), name) else {
            eprintln!("SKIP {name}: not in this shard");
            continue;
        };
        let Some((info, data)) = hfq.tensor_data_vec(name) else {
            eprintln!("FAIL {name}: absent from the .hfq");
            failures += 1;
            continue;
        };
        if info.quant_type != 51 {
            eprintln!(
                "FAIL {name}: quant_type {} (expected 51 = MQ2G256LloydU)",
                info.quant_type
            );
            failures += 1;
            continue;
        }
        let got = dequant_lloyd_u(&data, src.len());
        let max_err = src
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Signed zeros legitimately differ: ~19% of source weights are -0.0 and
        // come back +0.0. Numerically identical, so the bar is VALUE-exact.
        let sign_only: usize = src
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .filter(|(a, b)| **a == 0.0 && **b == 0.0)
            .count();
        let bitwise_diff = src
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let status = if max_err == 0.0 { "OK  " } else { "FAIL" };
        if max_err != 0.0 {
            failures += 1;
        }
        checked += 1;
        println!(
            "{status} {name}\n     shape={shape:?} n={} max|err|={max_err} \
             bitwise-diff={bitwise_diff} (of which signed-zero={sign_only})",
            src.len()
        );
    }

    println!("\n{checked} tensor(s) checked, {failures} failure(s)");
    if checked == 0 {
        eprintln!("nothing was checked — that is not a pass");
        std::process::exit(2);
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
