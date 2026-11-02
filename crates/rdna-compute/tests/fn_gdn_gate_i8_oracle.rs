// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: `fn_gdn_dense::gate_i8` (`fn_gdn_dense_gate_i8`: the dense GDN gated
//! RMSNorm fused with the out projection's FWHT rotation and A8 quantization)
//! must emit exactly the `block_i8_128` blocks of the chain it replaces,
//! `fn_gdn_dense_gate_bf16in` (F32 gate output) followed by `mq6_i8_prepare_x`
//! (`mq_rotate_x_i8_gfx12`), and the out-projection GEMM on them must therefore
//! produce the same bytes.
//!
//! Every output byte of the whole `[K/128][rows]` block array is compared, for
//! row counts {1, 16, 777, 4093, 4096, 16384} (K = 6144), on random inputs with
//! injected NaN / +-inf / +-0 / subnormal / extreme-magnitude BF16 recurrent and F32
//! gate values and all-zero, all-special and outlier rows. The destination scratch
//! is poisoned between the two producers, so a byte the fused kernel left unwritten
//! cannot pass. The GEMM output is compared bitwise at M = 2560 on two shapes.
//!
//! `#[ignore]`d: needs an exact gfx1201 with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test fn_gdn_gate_i8_oracle -- --ignored --test-threads=1 --nocapture
//!
//! `microbench` prints hipEvent medians of the fused launch against the launches
//! it replaces (`fn_gdn_dense_gate_rotate_bf16in` + `mq_rotate_x_i8_gfx12`, and the
//! rotate-less `fn_gdn_dense_gate_bf16in` + `mq_rotate_x_i8_gfx12` pair).

#![cfg(feature = "deltanet")]

use half::f16;
use rdna_compute::fn_gdn_dense;
use rdna_compute::tensor_ops::GatedDeltaGateBatched;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const HEADS: usize = 48;
const DIM: usize = 128;
const K: usize = HEADS * DIM;
const BLOCK_BYTES: usize = 136;
const GROUP_BYTES: usize = 200;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// BF16 values that stress the gated norm and the quantizer: NaN (two payloads),
/// +-inf, +-0, subnormals, the largest finite, tiny normals.
const SPECIAL_BF16: [u16; 14] = [
    0x7fc0, 0xffc1, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x7f7f, 0xff7f,
    0x0080, 0x3f80,
];

fn bf16_bits(x: f32) -> u16 {
    // Round to nearest even; finite inputs only.
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// `rows * K` BF16 recurrent bits, `rows * K` F32 gate inputs: row classes keep the
/// zero-block, all-special, outlier and plain paths all exercised at every row count.
fn gen_inputs(rows: usize, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut r = Rng(seed | 1);
    let mut rec = vec![0u16; rows * K];
    let mut z = vec![0f32; rows * K];
    for row in 0..rows {
        let class = (row + 4) % 16;
        for i in 0..K {
            let scale = 1.0 + 4.0 * (((i / DIM) % 7) as f32);
            let mut a = bf16_bits(r.unit() * scale);
            let mut b = r.unit() * 6.0;
            match class {
                0 => {
                    // All-zero recurrent head rows: zero norm input, zero block.
                    a = if r.below(2) == 0 { 0x0000 } else { 0x8000 };
                }
                1 => {
                    // Every element special.
                    a = SPECIAL_BF16[r.below(SPECIAL_BF16.len() as u64) as usize];
                    b = f32::from_bits(
                        [
                            0x7fc0_0000u32,
                            0xffc0_0001,
                            0x7f80_0000,
                            0xff80_0000,
                            0,
                            0x8000_0000,
                            1,
                            0x8000_0001,
                            0x7f7f_ffff,
                            0xff7f_ffff,
                        ][r.below(10) as usize],
                    );
                }
                2 => {
                    // One outlier per head pair: exercises amax/scale extremes.
                    if i % 256 == (row * 7) % 256 {
                        a = bf16_bits(r.unit() * 3.0e4);
                    }
                }
                3 => {
                    // Huge magnitudes: the square-sum overflows to +inf.
                    a = if r.below(4) == 0 {
                        0x7f00 | (r.below(0x7f) as u16)
                    } else {
                        a
                    };
                }
                _ => {}
            }
            if class >= 4 && r.below(200) == 0 {
                a = SPECIAL_BF16[r.below(SPECIAL_BF16.len() as u64) as usize];
            }
            if class >= 4 && r.below(300) == 0 {
                b = f32::from_bits(
                    [
                        0x7fc0_0000u32,
                        0x7f80_0000,
                        0xff80_0000,
                        0,
                        0x8000_0000,
                        1,
                        0x7f7f_ffff,
                    ][r.below(7) as usize],
                );
            }
            rec[row * K + i] = a;
            z[row * K + i] = b;
        }
    }
    (rec, z)
}

fn norm_bits(seed: u64) -> Vec<u16> {
    let mut r = Rng(seed | 1);
    (0..DIM)
        .map(|_| bf16_bits(0.25 + 1.5 * (r.unit() * 0.5 + 0.5)))
        .collect()
}

fn dtoh(gpu: &Gpu, ptr: *mut c_void, bytes: usize) -> Vec<u8> {
    // SAFETY: `ptr` is the live A8 scratch with at least `bytes` bytes; the
    // borrowed wrapper neither owns nor frees it.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &buf).expect("dtoh");
    v
}

fn poison(gpu: &Gpu, ptr: *mut c_void, bytes: usize) {
    // SAFETY: as in `dtoh`.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    gpu.hip.memset(&buf, 0xA5, bytes).expect("poison");
}

struct Operands {
    rec: GpuTensor,
    z: GpuTensor,
    norm: GpuTensor,
    out: GpuTensor,
    rotated: GpuTensor,
    rows: usize,
}

fn operands(gpu: &mut Gpu, rows: usize, seed: u64) -> Operands {
    let (rec, z) = gen_inputs(rows, seed);
    let rec_bytes: Vec<u8> = rec.iter().flat_map(|v| v.to_le_bytes()).collect();
    let norm_bytes: Vec<u8> = norm_bits(seed ^ 0x9e37)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rec = gpu.upload_raw(&rec_bytes, &[rows * K * 2]).expect("rec");
    let norm = gpu.upload_raw(&norm_bytes, &[DIM * 2]).expect("norm");
    let norm = GpuTensor {
        dtype: DType::BF16,
        shape: vec![DIM],
        ..norm
    };
    let z = gpu.upload_f32(&z, &[rows * K]).expect("z");
    let out = gpu.zeros(&[rows * K], DType::F32).expect("out");
    let rotated = gpu.zeros(&[rows * K], DType::F32).expect("rotated");
    Operands {
        rec,
        z,
        norm,
        out,
        rotated,
        rows,
    }
}

impl Operands {
    fn gate_args(&self) -> GatedDeltaGateBatched<'_> {
        GatedDeltaGateBatched {
            recurrent_output: &self.rec,
            z: &self.z,
            norm: &self.norm,
            output: &self.out,
            rows: self.rows,
            value_heads: HEADS,
            value_dim: DIM,
        }
    }
}

fn free(gpu: &mut Gpu, o: Operands) {
    for t in [o.rec, o.z, o.norm, o.out, o.rotated] {
        gpu.free_tensor(t).expect("free");
    }
}

fn require_gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: needs gfx1201, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

/// Whole-array byte comparison of the fused blocks against the two-launch chain.
/// Returns the (reference, fused) block bytes.
fn compare_blocks(gpu: &mut Gpu, o: &Operands) -> (Vec<u8>, Vec<u8>) {
    let rows = o.rows;
    let bytes = (K / 128) * rows * BLOCK_BYTES;
    let args = o.gate_args();
    fn_gdn_dense::gate(gpu, &args, None).expect("reference gate");
    let ref_ptr = gpu
        .mq6_i8_prepare_x(&o.out, K, rows)
        .expect("reference producer");
    assert_eq!(gpu.last_launched_kernel(), Some("mq_rotate_x_i8_gfx12"));
    let reference = dtoh(gpu, ref_ptr, bytes);
    poison(gpu, ref_ptr, bytes);
    let fused_ptr = fn_gdn_dense::gate_i8(gpu, &args).expect("fused gate");
    assert_eq!(gpu.last_launched_kernel(), Some("fn_gdn_dense_gate_i8"));
    assert_eq!(fused_ptr, ref_ptr, "both producers use the one A8 scratch");
    let fused = dtoh(gpu, fused_ptr, bytes);
    let bad = reference.iter().zip(&fused).filter(|(a, b)| a != b).count();
    let first = reference.iter().zip(&fused).position(|(a, b)| a != b);
    assert_eq!(
        bad, 0,
        "rows {rows}: {bad} of {bytes} block bytes differ (first at {first:?} = block {:?}, byte {:?})",
        first.map(|p| p / BLOCK_BYTES),
        first.map(|p| p % BLOCK_BYTES),
    );
    (reference, fused)
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn blocks_are_bytewise_the_two_launch_chain() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    for (n, rows) in [1usize, 16, 777, 4093, 4096, 16384].into_iter().enumerate() {
        let o = operands(&mut gpu, rows, 0x1234_5678 + n as u64);
        let (reference, fused) = compare_blocks(&mut gpu, &o);
        // Not vacuous: real blocks (non-zero scales) were produced and compared.
        let nonzero_d = reference
            .as_chunks::<BLOCK_BYTES>()
            .0
            .iter()
            .filter(|b| b[..4] != [0; 4])
            .count();
        assert!(nonzero_d > 0, "rows {rows}: no block has a scale");
        if rows >= 16 {
            assert!(
                nonzero_d < reference.len() / BLOCK_BYTES,
                "rows {rows}: no all-zero block"
            );
        }
        assert_eq!(reference.len(), fused.len());
        eprintln!(
            "rows {rows:>6}: {} block bytes identical ({} blocks, {nonzero_d} with scale)",
            reference.len(),
            reference.len() / BLOCK_BYTES
        );
        free(&mut gpu, o);
    }
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

#[test]
#[ignore = "needs an exact gfx1201"]
fn out_projection_gemm_bytes_are_unchanged() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let m = 2560;
    let w = gpu
        .upload_raw(&gen_weights(m, 0x77), &[m * (K / 256) * GROUP_BYTES])
        .expect("w");
    for (n, rows) in [4096usize, 777].into_iter().enumerate() {
        let o = operands(&mut gpu, rows, 0xabcd_0000 + n as u64);
        let args = o.gate_args();
        let y_ref = gpu.zeros(&[rows * m], DType::F32).expect("y_ref");
        let y_fused = gpu.zeros(&[rows * m], DType::F32).expect("y_fused");
        let y_ref16 = gpu.zeros(&[rows * m], DType::BF16).expect("y_ref16");
        let y_fused16 = gpu.zeros(&[rows * m], DType::BF16).expect("y_fused16");
        fn_gdn_dense::gate(&mut gpu, &args, None).expect("gate");
        let xq = gpu.mq6_i8_prepare_x(&o.out, K, rows).expect("producer");
        gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y_ref, m, K, rows)
            .expect("gemm ref");
        gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y_ref16, m, K, rows)
            .expect("gemm ref bf16");
        let xq = fn_gdn_dense::gate_i8(&mut gpu, &args).expect("fused");
        gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y_fused, m, K, rows)
            .expect("gemm fused");
        gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y_fused16, m, K, rows)
            .expect("gemm fused bf16");
        for (a, b, what) in [(&y_ref, &y_fused, "F32"), (&y_ref16, &y_fused16, "BF16")] {
            let (a, b) = (
                gpu.download_raw_bytes(a).unwrap(),
                gpu.download_raw_bytes(b).unwrap(),
            );
            let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert_eq!(
                bad,
                0,
                "rows {rows} {what}: {bad} of {} output bytes differ",
                a.len()
            );
            eprintln!(
                "rows {rows:>5} M={m} {what}: {} GEMM output bytes identical",
                a.len()
            );
        }
        for t in [y_ref, y_fused, y_ref16, y_fused16] {
            gpu.free_tensor(t).unwrap();
        }
        free(&mut gpu, o);
    }
}

fn median_us(gpu: &mut Gpu, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..5 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        gpu.hip.event_record(&start, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        s.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0);
    }
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

#[test]
#[ignore = "needs an exact gfx1201; timing"]
fn microbench() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    for rows in [4096usize, 16384] {
        let o = operands(&mut gpu, rows, 0x5eed);
        let args = o.gate_args();
        // JIT every kernel and fill the F32 gate output for the producer-only timing.
        fn_gdn_dense::gate(&mut gpu, &args, Some(&o.rotated)).unwrap();
        let _ = fn_gdn_dense::gate_i8(&mut gpu, &args).unwrap();
        let t_gate_rot = median_us(&mut gpu, 30, |g| {
            fn_gdn_dense::gate(g, &args, Some(&o.rotated)).unwrap();
        });
        let t_gate = median_us(&mut gpu, 30, |g| {
            fn_gdn_dense::gate(g, &args, None).unwrap();
        });
        let t_i8 = median_us(&mut gpu, 30, |g| {
            let r = g.reserve_int8_mmq(K, rows).unwrap();
            g.rotate_x_mq_i8_batched(&o.out, None, r, K, rows).unwrap();
        });
        let t_fused = median_us(&mut gpu, 30, |g| {
            fn_gdn_dense::gate_i8(g, &args).unwrap();
        });
        let sum_rot = t_gate_rot + t_i8;
        let sum_plain = t_gate + t_i8;
        let row_before = (K * 2 + K * 4 + K * 4 + K * 4) + (K * 4 + (K / 128) * BLOCK_BYTES);
        let row_after = K * 2 + K * 4 + (K / 128) * BLOCK_BYTES;
        println!(
            "MICROBENCH rows={rows} gate_rotate_bf16in={t_gate_rot:.1}us gate_bf16in={t_gate:.1}us \
             mq_rotate_x_i8_gfx12={t_i8:.1}us | chain(rotate+i8)={sum_rot:.1}us \
             chain(plain+i8)={sum_plain:.1}us | fused={t_fused:.1}us | \
             fused/chain(rotate)={:.3} fused/chain(plain)={:.3} | DRAM bytes/row before={row_before} after={row_after}",
            t_fused / sum_rot,
            t_fused / sum_plain,
        );
        free(&mut gpu, o);
    }
}
