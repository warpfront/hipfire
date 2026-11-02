// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of `Gpu::gemv_mq2g256v2_xbatch` (the MTP draft head's batched ranking
//! GEMV over the MQ2G256V2 copy): for every shape and B = 1..8, the whole `y` of the
//! x-batched kernel (launched as production does, in chunks of four) must equal the
//! bytes of `Gpu::gemv_mq2g256v2` launched once per column. Random weights (finite
//! f16 scale / zero-point headers, random packed 2-bit payloads) and random
//! activations; shapes cover K = 256 .. 2560 and odd / non-multiple-of-four M.
//!
//! `mq2_xbatch_oracle --out FILE.json`; exit 0 only when every case matches.

use rdna_compute::{DType, Gpu};
use serde_json::json;

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// xorshift64* stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// `m` rows of `k / 256` 72-byte groups: two (scale, zero-point) f16 pairs, then 64
/// packed bytes.
fn weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = m * (k / 256);
    let mut bytes = Vec::with_capacity(groups * 72);
    for _ in 0..groups {
        for _ in 0..2 {
            let scale = 0.01 + 0.5 * (rng.f32().abs());
            let zero = -0.5 * scale * 3.0 + 0.05 * rng.f32();
            bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f16_bits(zero).to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    }
    bytes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let run = || -> Result<(serde_json::Value, usize)> {
        let mut gpu = Gpu::init().map_err(err)?;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut cases = Vec::new();
        let mut bad_total = 0usize;
        for (m, k) in [
            (1usize, 256usize),
            (7, 256),
            (31, 768),
            (130, 2560),
            (1001, 2560),
            (4093, 2048),
            (100003, 2560),
        ] {
            let a_bytes = weights(&mut rng, m, k);
            let a = gpu.upload_raw(&a_bytes, &[a_bytes.len()]).map_err(err)?;
            let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32() * 3.0).collect();
            let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
            let y_ref = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            let y = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            for column in 0..8 {
                gpu.gemv_mq2g256v2(
                    &a,
                    &x.sub_offset(column * k, k),
                    &y_ref.sub_offset(column * m, m),
                    m,
                    k,
                )
                .map_err(err)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let reference = gpu.download_raw_bytes(&y_ref).map_err(err)?;
            for b in 1..=8usize {
                gpu.hip.memset(&y.buf, 0x7f, 8 * m * 4).map_err(err)?;
                gpu.gemv_mq2g256v2_xbatch(&a, &x, &y, m, k, b)
                    .map_err(err)?;
                gpu.hip.device_synchronize().map_err(err)?;
                let got = gpu.download_raw_bytes(&y).map_err(err)?;
                let written = b * m * 4;
                let bad_written = got[..written]
                    .iter()
                    .zip(&reference[..written])
                    .filter(|(g, r)| g != r)
                    .count();
                let bad_guard = got[written..].iter().filter(|&&byte| byte != 0x7f).count();
                bad_total += bad_written + bad_guard;
                cases.push(json!({
                    "m": m, "k": k, "b": b,
                    "mismatch_bytes": bad_written, "overwritten_guard_bytes": bad_guard,
                }));
            }
            gpu.free_tensor(a).map_err(err)?;
            gpu.free_tensor(x).map_err(err)?;
            gpu.free_tensor(y_ref).map_err(err)?;
            gpu.free_tensor(y).map_err(err)?;
        }
        Ok((
            json!({"schema": "mq2-xbatch-oracle-v1", "cases": cases, "mismatch_bytes_total": bad_total}),
            bad_total,
        ))
    };
    match run() {
        Ok((value, bad)) => {
            if let Some(out) = out {
                std::fs::write(&out, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
                eprintln!("wrote {out}");
            }
            eprintln!("mq2 xbatch oracle: {bad} mismatching bytes");
            std::process::exit(i32::from(bad != 0));
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
