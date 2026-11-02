// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle: the fused gfx1201 / gfx1151 Qwen4 HC-down launch
//! (`Gpu::gemm_bf16_xf16_hc_down_fused`; kernel `qwen4_hc_down_fused_b64_k32` on gfx1201,
//! `qwen4_hc_down_fused_gfx1151*` on gfx1151) against the production chain it replaces on the
//! same arch, bit for bit.
//!
//! Chain (the F16 WMMA route's HC read down projection + activation):
//!   `gemm_bf16_xf16_f16_wmma` (gfx1201: split-K=4 tile + reducer; gfx1151: the ascending-K
//!   160 x 64 tile from 2048 rows, 64 x 64 below) -> F32 scratch `y`,
//!   `hc_activation_fused_f32` with `bf16_out` -> packed BF16 `R`.
//! Fused: one launch -> packed BF16 `F` (launched on every row count, also below the route's
//! threshold, where the chain's tile differs but keeps each output's K order; the host gate's
//! applies() verdict is checked against the threshold per case).
//! `R` and `F` are allocated `rows * 320 + 4096` elements, pre-filled with 0xAB bytes; every
//! u16 of the active region is compared and the guard region must keep its 0xAB fill (no write
//! past `rows * 320` elements).
//!
//! Weight: random BF16 [320, 10240], uploaded as BF16 (the host builds the F16 shadow).
//! Classes: `random` (N(0,1)-like X, N(0,0.05) weights) and `adversarial` (NaN / +-inf / +-0 /
//! F16 subnormal / 65504 values sprinkled into X by row kind, and NaN / inf / subnormal /
//! F16-range-edge BF16 patterns in the weight rows).
//!
//! Exit 0: every case byte-identical. 1: a mismatch (first (row, col, bits) printed) or a wrong
//! applies() verdict. 2: the fused route does not apply here (kill switch
//! `HIPFIRE_QWEN4_HC_DOWN_FUSED=0`, other arch).
//! `ROWS=a,b,..` overrides the row list. Receipt: `HC_DOWN_ORACLE_RECEIPT` (default
//! `./hc-down-fused-oracle.txt`).
//!
//! Run: cargo run --release --features lab --example hc_down_fused_oracle -p rdna-compute

use half::f16;
use rdna_compute::tensor_ops::{hc_activation_fused_f32, HcActivationFused};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;

const M: usize = 320;
const K: usize = 10240;
const GUARD: usize = 4096;
const SCALE: f32 = 0.25;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// N(0,1)-like: sum of four uniforms on [-1,1) scaled to unit variance.
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn to_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for h in v {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> GpuTensor {
    let mut t = gpu.upload_raw(&to_bytes(v), &[v.len()]).expect("upload");
    t.dtype = dtype;
    t
}

/// F16 bit patterns sprinkled into X rows.
const X_SPECIALS: [u16; 17] = [
    0x7e00, 0xfe00, 0x7c01, 0xffff, // NaN (quiet, negative, signalling, payload)
    0x7c00, 0xfc00, // +-inf
    0x0000, 0x8000, // +-0
    0x0001, 0x8001, 0x03ff, 0x83ff, 0x0200, // subnormals
    0x7bff, 0xfbff, // +-65504
    0x0400, 0x8400, // min normal
];

/// Weight rows: BF16 patterns that survive (or are clamped by) the F16 shadow.
const W_BENIGN: [u16; 14] = [
    0x0000, 0x8000, // +-0
    0x0001, 0x007f, 0x8040, // BF16 subnormals (flush in the shadow)
    0x3380, 0x3800, 0xb7ff, 0x387f, // F16-subnormal range
    0x477f, 0xc77f, 0x4700, // 65280-range, just inside F16
    0x3f80, 0xbf80,
];
const W_POISON: [u16; 8] = [
    0x7f80, 0xff80, // +-inf
    0x7fc0, 0xffc1, // NaN
    0x4780, 0xc780, // 65536: rounds to F16 inf in the shadow
    0x7f7f, 0xff7f, // BF16 max: F16 inf
];

fn gen_weight(adversarial: bool) -> Vec<u16> {
    let mut rng = Rng(if adversarial { 0xA11CE_u64 } else { 0xB0B_u64 });
    let mut w = vec![0u16; M * K];
    for m in 0..M {
        let row = &mut w[m * K..(m + 1) * K];
        for x in row.iter_mut() {
            *x = bf16_bits(rng.gauss() * 0.05);
        }
        if !adversarial {
            continue;
        }
        match Rng(0x5eed ^ m as u64).next() % 8 {
            4 => {
                for _ in 0..8 {
                    row[rng.below(K)] = W_BENIGN[rng.below(W_BENIGN.len())];
                }
            }
            5 => {
                // dense huge, still inside F16: 0x4600..0x477f
                for x in row.iter_mut() {
                    *x = 0x4600 + rng.below(0x180) as u16 | ((rng.next() & 1) as u16) << 15;
                }
            }
            6 => {
                row[rng.below(K)] = W_POISON[rng.below(W_POISON.len())];
            }
            7 => {
                // dense F16 subnormals reached through BF16 (2^-24 .. 2^-14)
                for x in row.iter_mut() {
                    *x = 0x3380 + rng.below(0x500) as u16 | ((rng.next() & 1) as u16) << 15;
                }
            }
            _ => {}
        }
    }
    w
}

/// `rows` F16 rows of K. Adversarial rows by kind (hash of the row index, so a smaller row
/// count is a prefix of a larger one): 0-3 clean with `kind` sprinkled specials, 4 dense
/// subnormal, 5 dense huge, 6 dense +-0, 7 NaN/inf sprinkle.
fn gen_x(rows: usize, adversarial: bool) -> Vec<u16> {
    let mut rng = Rng(if adversarial {
        0xFEED_u64
    } else {
        0xC0FFEE_u64
    });
    let mut x = vec![0u16; rows * K];
    for b in 0..rows {
        let row = &mut x[b * K..(b + 1) * K];
        let kind = if adversarial {
            Rng(0xabcd ^ b as u64).next() % 8
        } else {
            0
        };
        for e in row.iter_mut() {
            *e = match kind {
                4 if rng.next() % 8 != 0 => {
                    let s = (rng.next() & 1) as u16;
                    s << 15 | (1 + rng.below(0x3ff)) as u16
                }
                5 => {
                    let mag = if rng.next() % 4 == 0 {
                        65504.0
                    } else {
                        20000.0 + (rng.next() % 45504) as f32
                    };
                    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
                }
                6 if rng.next() % 64 != 0 => (rng.next() & 1) as u16 * 0x8000,
                _ => f16_bits(rng.gauss()),
            };
        }
        match kind {
            0..=3 => {
                for _ in 0..kind {
                    row[rng.below(K)] = X_SPECIALS[rng.below(X_SPECIALS.len())];
                }
            }
            7 => {
                for e in row.iter_mut() {
                    if rng.next() % 4096 == 0 {
                        *e = X_SPECIALS[rng.below(8)];
                    }
                }
            }
            _ => {}
        }
    }
    x
}

fn git_sha() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let sha = run(&["rev-parse", "HEAD"]);
    let dirty = !run(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    format!(
        "{}{}",
        if sha.is_empty() { "unknown" } else { &sha },
        if dirty { "+dirty" } else { "" }
    )
}

fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.to_string();
    let switch = std::env::var("HIPFIRE_QWEN4_HC_DOWN_FUSED").unwrap_or_default();
    let probe_rows = 4096;
    let applies = gpu.qwen4_hc_down_fused_applies(M, K, probe_rows);
    println!("arch {arch} HIPFIRE_QWEN4_HC_DOWN_FUSED={switch:?} applies(320,10240,{probe_rows})={applies}");
    if !applies {
        eprintln!(
            "the fused route does not apply (kill switch HIPFIRE_QWEN4_HC_DOWN_FUSED={switch:?} \
             or arch {arch} is not gfx1201/gfx1151): the oracle needs the switch on"
        );
        std::process::exit(2);
    }

    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                1, 16, 511, 1023, 1024, 1025, 1535, 1536, 1537, 2047, 2048, 2049, 4096, 4097, 4099,
                8188, 8192, 16384,
            ]
        });
    let max_rows = *rows_list.iter().max().unwrap();

    let receipt_path = std::env::var("HC_DOWN_ORACLE_RECEIPT")
        .unwrap_or_else(|_| "./hc-down-fused-oracle.txt".into());
    let mut receipt = String::new();
    let date = std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    writeln!(receipt, "hc_down_fused_oracle receipt").unwrap();
    writeln!(receipt, "date {date}").unwrap();
    writeln!(receipt, "git {}", git_sha()).unwrap();
    writeln!(receipt, "arch {arch}").unwrap();
    writeln!(
        receipt,
        "kernel {} vs chain gemm_bf16_xf16_f16_wmma + hc_activation_fused_f32(bf16_out)",
        if arch == "gfx1151" {
            "qwen4_hc_down_fused_gfx1151"
        } else {
            "qwen4_hc_down_fused_b64_k32"
        }
    )
    .unwrap();
    writeln!(
        receipt,
        "switch HIPFIRE_QWEN4_HC_DOWN_FUSED={switch:?} scale={SCALE} m={M} k={K}"
    )
    .unwrap();

    let mut failures = 0usize;
    let mut cases = 0usize;
    for class in ["random", "adversarial"] {
        let adversarial = class == "adversarial";
        let w_host = gen_weight(adversarial);
        let weight = upload_u16(&mut gpu, &w_host, DType::BF16);
        drop(w_host);
        let x_host = gen_x(max_rows, adversarial);
        for &rows in &rows_list {
            let n = rows * M;
            let x = upload_u16(&mut gpu, &x_host[..rows * K], DType::F16);
            let y = gpu.zeros(&[n], DType::F32).unwrap();
            let fill = vec![0xABABu16; n + GUARD];
            let r = upload_u16(&mut gpu, &fill, DType::BF16);
            let rb;

            // chain
            gpu.gemm_bf16_xf16_f16_wmma(&weight, &x, &y, M, K, rows)
                .unwrap();
            hc_activation_fused_f32(
                &mut gpu,
                &HcActivationFused {
                    values: &y,
                    scale: SCALE,
                    bf16_out: Some(&r),
                },
            )
            .unwrap();
            // fused
            let case_applies = gpu.qwen4_hc_down_fused_applies(M, K, rows);
            let min_rows = if arch == "gfx1151" {
                rdna_compute::gemm::HC_DOWN_FUSED_GFX1151_MIN_ROWS
            } else {
                rdna_compute::gemm::HC_DOWN_FUSED_MIN_ROWS
            };
            let expect_applies = switch != "0" && rows >= min_rows;
            if case_applies != expect_applies {
                failures += 1;
                let line = format!(
                    "ORACLE FAIL rows={rows} class={class} applies()={case_applies} expected {expect_applies} (min rows {min_rows})"
                );
                println!("{line}");
                writeln!(receipt, "{line}").unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            rb = le_u16(&gpu.download_raw_bytes(&r).unwrap());
            {
                let vname = if arch == "gfx1151" {
                    "qwen4_hc_down_fused_gfx1151"
                } else {
                    "qwen4_hc_down_fused_b64_k32"
                };
                let f = upload_u16(&mut gpu, &fill, DType::BF16);
                gpu.gemm_bf16_xf16_hc_down_fused(&weight, &x, &f, M, K, rows, SCALE)
                    .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let fb = le_u16(&gpu.download_raw_bytes(&f).unwrap());

                let mut mismatches = 0usize;
                let mut first = None;
                for i in 0..n {
                    if rb[i] != fb[i] {
                        mismatches += 1;
                        first.get_or_insert((i / M, i % M, rb[i], fb[i]));
                    }
                }
                let guard_bad = fb[n..].iter().filter(|&&v| v != 0xABAB).count();
                let chain_guard_bad = rb[n..].iter().filter(|&&v| v != 0xABAB).count();
                let (mut nan, mut inf, mut zero) = (0usize, 0usize, 0usize);
                for &v in &rb[..n] {
                    match v & 0x7fff {
                        0 => zero += 1,
                        0x7f80 => inf += 1,
                        e if e > 0x7f80 => nan += 1,
                        _ => {}
                    }
                }
                let pass = mismatches == 0 && guard_bad == 0 && chain_guard_bad == 0;
                cases += 1;
                let line = if pass {
                    format!(
                        "ORACLE PASS rows={rows} class={class} kernel={vname} applies={case_applies} elems={n} bytes={} differing_bytes=0 nan={nan} inf={inf} zero={zero}",
                        n * 2
                    )
                } else {
                    failures += 1;
                    let (row, col, rbits, fbits) = first.unwrap_or((0, 0, 0, 0));
                    format!(
                        "ORACLE FAIL rows={rows} class={class} kernel={vname} applies={case_applies} mismatches={mismatches} \
                         first(row={row}, col={col}, chain={rbits:#06x}, fused={fbits:#06x}) \
                         fused_guard_dirty={guard_bad} chain_guard_dirty={chain_guard_bad}"
                    )
                };
                println!("{line}");
                writeln!(receipt, "{line}").unwrap();
                gpu.free_tensor(f).unwrap();
            }

            for t in [x, y, r] {
                gpu.free_tensor(t).unwrap();
            }
        }
        gpu.free_tensor(weight).unwrap();
    }
    let summary = format!(
        "SUMMARY cases={cases} failures={failures} {}",
        if failures == 0 {
            "ALL BYTE-IDENTICAL"
        } else {
            "MISMATCH"
        }
    );
    println!("{summary}");
    writeln!(receipt, "{summary}").unwrap();
    std::fs::write(&receipt_path, &receipt).expect("write receipt");
    println!("receipt {receipt_path}");
    if failures != 0 {
        std::process::exit(1);
    }
}
