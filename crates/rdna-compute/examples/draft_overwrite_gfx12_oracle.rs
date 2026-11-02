// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the gfx1201 DFlash-draft overwrite GEMM
//! (`Gpu::gemm_mq4g256v2_overwrite_gfx12_dflash`) against today's legacy tail
//! of `gemm_mq4g256v2_batched_lmhead`: `memset(y, 0)` +
//! `Gpu::gemm_hfq4g256_residual_wmma_gfx12_mq4v2` (`Y += acc`).
//!
//! The overwrite kernels store `acc + 0.0f` once per output, which must equal
//! `(+0.0f) + acc` bit for bit (including the signed-zero case). For
//! (m, k) in {(5120,5120), (1024,5120), (17408,5120), (5120,17408),
//! (4096,5120)} and batch in {2, 9, 16, 17, 33, 48, 63} (one-tile kernel for
//! batch <= 16 and, with `HIPFIRE_WMMA_BATCH_TILES=0`, everywhere; the 128-K
//! verify-tile `_vt` overwrite kernels for 17..=63 otherwise):
//!
//! - weights: random valid MQ4G256V2 (finite normal f16 scale/zero per half
//!   group, random nibbles), except row 0 (all-zero payload, +0.0 headers) and
//!   row 1 (all-zero payload, -0.0 headers) so some outputs are exactly +-0;
//! - X: random F32 with every 5th element exactly 0 and one all-zero row, so
//!   more outputs are exactly zero;
//! - arm A: y zero-filled by `memset` then the legacy residual launcher;
//!   arm B: y pre-filled with NaN bytes then the overwrite launcher (must
//!   return `true`: a NaN left behind means a missed store).
//! - N = 64 (per shape): inside `Gpu::draft_rows64_enter(true)` the overwrite
//!   must be admitted (when `HIPFIRE_WMMA_BATCH_TILES` is on) and equal, bit
//!   for bit, four N = 16 one-tile overwrite launches on 16-row slices; with
//!   the scope closed N = 64 must be refused.
//!
//! Outputs are compared as `u32` bit patterns; +0.0 / -0.0 output counts are
//! printed to show the signed-zero path is exercised. Requires exact gfx1201.
//!
//! `cargo run --release -p rdna-compute --features lab --example draft_overwrite_gfx12_oracle`
//! (`HIPFIRE_WMMA_BATCH_TILES=0` forces the one-tile kernel for every batch.)
//! Exits nonzero on any mismatch.

use rdna_compute::{Gpu, GpuTensor};

const NAN_BITS: u32 = 0x7fc0_0000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0), hB (s1, z1),
/// then 128 payload bytes. Row 0: zero payload, +0.0 headers. Row 1: zero
/// payload, -0.0 headers (0x8000).
fn pack_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 136);
    for row in 0..m {
        for _ in 0..groups {
            if row < 2 {
                let h: u16 = if row == 0 { 0x0000 } else { 0x8000 };
                for _half in 0..2 {
                    out.extend_from_slice(&h.to_le_bytes());
                    out.extend_from_slice(&h.to_le_bytes());
                }
                out.extend(std::iter::repeat_n(0u8, 128));
                continue;
            }
            for _half in 0..2 {
                out.extend_from_slice(&random_scale(rng).to_le_bytes());
                out.extend_from_slice(&random_zero(rng).to_le_bytes());
            }
            for _ in 0..128 {
                out.push((rng.next() & 0xff) as u8);
            }
        }
    }
    out
}

fn fill_nan(gpu: &Gpu, t: &GpuTensor, words: usize) {
    gpu.hip.device_synchronize().expect("sync before NaN fill");
    gpu.hip
        .memcpy_htod(&t.buf, &NAN_BITS.to_le_bytes().repeat(words))
        .expect("NaN fill");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!(
            "draft_overwrite_gfx12_oracle: needs exact gfx1201, got {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    println!(
        "arch={} wmma_batch_tiles={} draft_collapse_off={} ldsstage={}",
        gpu.arch,
        gpu.flags.wmma_batch_tiles,
        gpu.flags.draft_collapse_off,
        gpu.flags.hfq4g256_ldsstage_wmma,
    );

    const SHAPES: [(usize, usize); 5] = [
        (5120, 5120),
        (1024, 5120),
        (17408, 5120),
        (5120, 17408),
        (4096, 5120),
    ];
    const BATCHES: [usize; 7] = [2, 9, 16, 17, 33, 48, 63];

    let mut rng = Rng(0xD0F1_A5B7);
    let mut failures = 0usize;
    let mut cases = 0usize;
    let mut total_pz = 0usize;
    let mut total_nz = 0usize;

    for &(m, k) in &SHAPES {
        let bytes = pack_mq4g256v2(m, k, &mut rng);
        let w = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .expect("upload weights");

        for &batch in &BATCHES {
            cases += 1;
            let mut x: Vec<f32> = (0..batch * k)
                .map(|_| (rng.unit() + rng.unit() - 1.0) * 1.5)
                .collect();
            for (i, v) in x.iter_mut().enumerate() {
                if i % 5 == 0 {
                    *v = 0.0;
                }
            }
            // One all-zero activation row (the last) -> that batch row's outputs are all +-0.
            for v in &mut x[(batch - 1) * k..] {
                *v = 0.0;
            }
            let d_x = gpu.upload_f32(&x, &[x.len()]).expect("upload x");
            let d_ya = gpu
                .upload_f32(&vec![1.0; batch * m], &[batch * m])
                .expect("alloc ya");
            let d_yb = gpu
                .upload_f32(&vec![1.0; batch * m], &[batch * m])
                .expect("alloc yb");

            // Arm A: legacy tail (memset + residual).
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            gpu.hip.device_synchronize().expect("sync");
            gpu.hip
                .memset(&d_ya.buf, 0, batch * m * 4)
                .expect("memset ya");
            gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2(&w, &d_x, &d_ya, m, k, batch)
                .expect("legacy residual launch");
            let a = fetch(&gpu, &d_ya);

            // Arm B: overwrite into NaN-prefilled y.
            fill_nan(&gpu, &d_yb, batch * m);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let admitted = gpu
                .gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &d_x, &d_yb, m, k, batch)
                .expect("overwrite launch");
            let b = fetch(&gpu, &d_yb);

            let mismatches = a.iter().zip(&b).filter(|(p, q)| p != q).count();
            let pz = b.iter().filter(|&&v| v == 0x0000_0000).count();
            let nz = b.iter().filter(|&&v| v == 0x8000_0000).count();
            let nans = b.iter().filter(|&&v| f32::from_bits(v).is_nan()).count();
            total_pz += pz;
            total_nz += nz;
            let ok = admitted && mismatches == 0 && nans == 0;
            if !ok {
                failures += 1;
            }
            println!(
                "{} m={m:<5} k={k:<5} batch={batch:<2} admitted={admitted} mismatches={mismatches} nan_left={nans} +0={pz} -0={nz}",
                if ok { "PASS" } else { "FAIL" },
            );
        }

        // N = 64: admitted ONLY inside the draft 64-row scope (the `_vt4`
        // overwrite kernel, needs the batch-tile family); per-row identical to
        // the one-tile kernel, so the reference is four N = 16 overwrite
        // launches on 16-row slices. Outside the scope N = 64 is refused.
        {
            cases += 1;
            let batch = 64usize;
            let mut x: Vec<f32> = (0..batch * k)
                .map(|_| (rng.unit() + rng.unit() - 1.0) * 1.5)
                .collect();
            for (i, v) in x.iter_mut().enumerate() {
                if i % 5 == 0 {
                    *v = 0.0;
                }
            }
            for v in &mut x[(batch - 1) * k..] {
                *v = 0.0;
            }
            let d_x = gpu.upload_f32(&x, &[x.len()]).expect("upload x64");
            let d_ya = gpu
                .upload_f32(&vec![1.0; batch * m], &[batch * m])
                .expect("alloc ya64");
            let d_yb = gpu
                .upload_f32(&vec![1.0; batch * m], &[batch * m])
                .expect("alloc yb64");
            let want_admit = gpu.flags.wmma_batch_tiles;

            // Scope closed: N = 64 must NOT be admitted (device untouched).
            fill_nan(&gpu, &d_yb, batch * m);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let closed_admit = gpu
                .gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &d_x, &d_yb, m, k, batch)
                .expect("overwrite launch (scope closed)");

            // Reference: four N = 16 one-tile overwrite launches.
            fill_nan(&gpu, &d_ya, batch * m);
            let mut ref_admitted = true;
            for q in 0..4 {
                let xs = d_x.sub_offset(q * 16 * k, 16 * k);
                let ys = d_ya.sub_offset(q * 16 * m, 16 * m);
                gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
                ref_admitted &= gpu
                    .gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &xs, &ys, m, k, 16)
                    .expect("N=16 reference launch");
            }
            let a = fetch(&gpu, &d_ya);

            // Scoped N = 64 into NaN-prefilled y.
            fill_nan(&gpu, &d_yb, batch * m);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let prev = gpu.draft_rows64_enter(true);
            let scoped = gpu.gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &d_x, &d_yb, m, k, batch);
            gpu.draft_rows64_exit(prev);
            let scoped_admit = scoped.expect("overwrite launch (scope open)");
            let b = fetch(&gpu, &d_yb);

            let mismatches = if scoped_admit {
                a.iter().zip(&b).filter(|(p, q)| p != q).count()
            } else {
                usize::MAX
            };
            let pz = b.iter().filter(|&&v| v == 0x0000_0000).count();
            let nz = b.iter().filter(|&&v| v == 0x8000_0000).count();
            let nans = if scoped_admit {
                b.iter().filter(|&&v| f32::from_bits(v).is_nan()).count()
            } else {
                0
            };
            let scope_restored = !gpu.draft_rows64_scope;
            total_pz += pz;
            total_nz += nz;
            let ok = ref_admitted
                && !closed_admit
                && scoped_admit == want_admit
                && scope_restored
                && (!want_admit || (mismatches == 0 && nans == 0));
            if !ok {
                failures += 1;
            }
            let mismatches_txt = if scoped_admit {
                mismatches.to_string()
            } else {
                "n/a".into()
            };
            println!(
                "{} m={m:<5} k={k:<5} batch=64 scoped_admitted={scoped_admit} (want {want_admit}) closed_admitted={closed_admit} ref_admitted={ref_admitted} mismatches_vs_4xN16={mismatches_txt} nan_left={nans} +0={pz} -0={nz}",
                if ok { "PASS" } else { "FAIL" },
            );
        }
    }

    println!("cases={cases} failures={failures} total(+0.0 outputs)={total_pz} total(-0.0 outputs)={total_nz}");
    if failures != 0 {
        std::process::exit(1);
    }
}
